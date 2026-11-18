"""The NeMo LoRA training loop (T6.5), run in NeMo's own interpreter.

`healing train --local` (`run.py`) starts this file as a script with the Python of transcribe.cpp's
Parakeet environment (D-29: `healing` never depends on NeMo or torch) and hands it one JSON job:

    <nemo python> -I lora.py --job <round>/work/trainer.json

It promotes M0's `spikes/m0/lora_tiny.py` (T0.4): the same load (transcribe.cpp's offline patch,
then `restore_from`), the same `LoraLinear`, the same forward and RNN-T loss. What it adds:
- targets resolved from the loaded model by name, per group, failing on any that match nothing:
  the encoder's attention q/k/v/out, its feed-forward linears and the joint's linears (D-31, D-47);
- LoRA dropout (the base stays in eval mode: no SpecAugment, no dropout, as in M0);
- a `WeightedRandomSampler` over the bundle's weights, an epoch at a time, batched by padded audio seconds;
- a validation slice of the train bundle (whole storage clusters, never a test set), its loss and
  WER logged and used for early stopping; the best step's LoRA is the delta;
- a checkpoint (LoRA, optimizer, RNG, position) every `checkpoint_every` steps and on SIGINT or
  SIGTERM, so a rerun resumes exactly where the sampler was;
- a memory probe on the longest clips before step 1, and a footprint watchdog that stops the round before it swaps;
- outputs `out/lora_delta.safetensors` (LoRA tensors only) and `out/train_log.json` (numbers only).

No text reaches stdout or the log: NeMo's `log_prediction` is off and only numbers are printed.
Exit codes: 0 done, 2 bad job or targets, 3 over the memory limit, 130 interrupted (checkpointed).

Everything above the `# --- torch` line is standard library, so `healing`'s tests import it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import signal
import sys
import threading
import time
from pathlib import Path

GROUPS = {
    "encoder_attention": re.compile(r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out)$"),
    "encoder_feed_forward": re.compile(r"^encoder\.layers\.\d+\.feed_forward\d*\.linear\d+$"),
    "joint": re.compile(r"^joint\..+"),
}
ATTENTION_PARTS = ("q", "k", "v", "out")
SUFFIXES = (".lora_a", ".lora_b")
DELTA_FORMAT = "healing-lora-v1"
INTERRUPTED, MEMORY, BAD_JOB = 130, 3, 2


class TargetError(Exception):
    pass


class MemoryLimit(Exception):
    pass


# --- pure helpers (no torch) ---------------------------------------------------------------


def resolve_targets(linear_names, groups) -> dict[str, list[str]]:
    """The linear modules each target group wraps, from the loaded model's own names.
    Fails when a group, or one of attention's q/k/v/out, matches nothing (Unified and TDT v2 differ)."""
    resolved = {}
    for group in groups:
        if group not in GROUPS:
            raise TargetError(f"unknown target group {group!r} (known: {', '.join(GROUPS)})")
        resolved[group] = sorted(n for n in linear_names if GROUPS[group].match(n))
        if not resolved[group]:
            raise TargetError(f"target {group} matches no linear module in the base")
        if group == "encoder_attention":
            for part in ATTENTION_PARTS:
                if not any(n.endswith(f".linear_{part}") for n in resolved[group]):
                    raise TargetError(f"target encoder_attention: linear_{part} matches no module in the base")
    return resolved


def validation_split(entries: list[dict], share: float, seed: int) -> tuple[list[int], list[int]]:
    """(train, validation) indices. Validation takes whole storage clusters, in an order fixed by
    the seed, until it holds `share` of the entries; train keeps at least one cluster."""
    if share <= 0 or len(entries) < 2:
        return list(range(len(entries))), []
    by_cluster: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        by_cluster.setdefault(str(entry.get("cluster") or entry.get("id") or index), []).append(index)
    if len(by_cluster) < 2:
        return list(range(len(entries))), []
    order = sorted(by_cluster, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    wanted = max(1, math.ceil(share * len(entries)))
    held: list[int] = []
    for cluster in order[:-1]:
        if len(held) >= wanted:
            break
        held += by_cluster[cluster]
    held_set = set(held)
    return [i for i in range(len(entries)) if i not in held_set], sorted(held)


def batches_by_seconds(order: list[int], durations: list[float], max_seconds: float, max_clips: int) -> list[list[int]]:
    """Consecutive batches of `order` whose padded audio, clips × the longest clip, is at most
    `max_seconds` (and at most `max_clips` clips). Padded, not summed: every clip in a batch is padded to
    the longest, and RNN-T's joint is batch × frames × tokens × vocabulary, so one 27 s clip among 15
    short ones costs about 16 × 27 s (D-49). A clip longer than the budget is a batch by itself."""
    batches, current, longest = [], [], 0.0
    for index in order:
        length = durations[index]
        if current and ((len(current) + 1) * max(longest, length) > max_seconds or len(current) >= max_clips):
            batches.append(current)
            current, longest = [], 0.0
        current.append(index)
        longest = max(longest, length)
    if current:
        batches.append(current)
    return batches


def longest_batch(durations: list[float], max_seconds: float, max_clips: int) -> list[int]:
    """The batch the memory probe runs: the longest clips, filled up to the batch budget."""
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    return batches_by_seconds(order, durations, max_seconds, max_clips)[0] if order else []


def smoke_subset(entries: list[dict], limit: int, cfg: dict) -> list[dict]:
    """At most `limit` entries (0: all): the longest batch's clips first, so the memory probe still
    measures the longest audio, then the rest in an order fixed by the seed."""
    if not limit or len(entries) <= limit:
        return entries
    longest = set(longest_batch([float(e["duration"]) for e in entries], float(cfg["batch_seconds"]),
                                int(cfg["batch_max_clips"]))[:limit])
    rest = sorted((i for i in range(len(entries)) if i not in longest),
                  key=lambda i: hashlib.sha256(f"{cfg['seed']}:{entries[i]['id']}".encode()).hexdigest())
    return [entries[i] for i in sorted(longest)] + [entries[i] for i in rest[: limit - len(longest)]]


def delta_problems(names, targets: list[str]) -> list[str]:
    """Why a delta's tensor names are not exactly LoRA A/B pairs over the targets, in both the
    encoder and the joint (empty when they are)."""
    problems, modules = [], {}
    for name in names:
        suffix = next((s for s in SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            problems.append(f"{name} is not a LoRA tensor")
            continue
        modules.setdefault(name[: -len(suffix)], set()).add(suffix)
    wanted = set(targets)
    for module, parts in sorted(modules.items()):
        if module not in wanted:
            problems.append(f"{module} is not a LoRA target")
        if parts != set(SUFFIXES):
            problems.append(f"{module} lacks {sorted(set(SUFFIXES) - parts)}")
    missing = wanted - set(modules)
    if missing:
        problems.append(f"{len(missing)} targets have no tensors (e.g. {sorted(missing)[0]})")
    if not any(m.startswith("encoder.") for m in modules) or not any(m.startswith("joint.") for m in modules):
        problems.append("the delta must cover both the encoder and the joint")
    return problems


def metadata_problems(metadata: dict, rank: int, alpha: float, base_sha256: str | None = None) -> list[str]:
    """Why a delta's `__metadata__` (`save_delta`) does not match the config it is merged with."""
    problems = []
    if metadata.get("format") != DELTA_FORMAT:
        problems.append(f"the delta's format is {metadata.get('format')!r}, not {DELTA_FORMAT}")
    try:
        trained = (int(metadata.get("rank")), float(metadata.get("alpha")), float(metadata.get("scale")))
    except (TypeError, ValueError):
        return problems + ["the delta's metadata has no rank, alpha and scale"]
    if trained[:2] != (rank, float(alpha)):
        problems.append(f"the delta was trained at rank {trained[0]}, alpha {trained[1]:g}; the config says {rank}, {alpha:g}")
    if abs(trained[2] - alpha / rank) > 1e-9:
        problems.append(f"the delta's scale {trained[2]:g} is not alpha/rank")
    if base_sha256 is not None and metadata.get("base_sha256") != base_sha256:
        problems.append(f"the delta was trained on base {str(metadata.get('base_sha256'))[:12]}, the config pins {base_sha256[:12]}")
    return problems


def safetensors_header(path: Path) -> dict:
    """A safetensors file's header (tensor names, dtypes, shapes and `__metadata__`), read without torch."""
    with open(path, "rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if size <= 0 or size > 100 << 20:
            raise ValueError(f"{path} is not a safetensors file")
        return json.loads(handle.read(size))


def peak_rss_bytes() -> int:
    """This process's peak resident memory (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def footprint_bytes() -> int:
    """This process's memory now, swapped and compressed pages included: macOS's phys_footprint
    (`proc_pid_rusage`, what Activity Monitor shows), Linux's VmRSS + VmSwap. RSS alone falls while a
    process swaps, so it cannot see a swap storm; this can."""
    if sys.platform == "darwin":
        import ctypes

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buffer = ctypes.create_string_buffer(512)  # rusage_info_v2; phys_footprint is the 8th u64 after the uuid
        if libproc.proc_pid_rusage(os.getpid(), 2, buffer) == 0:
            return int.from_bytes(buffer.raw[72:80], "little")
        return peak_rss_bytes()
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        return sum(int(fields.get(k, "0 kB").split()[0]) * 1024 for k in ("VmRSS", "VmSwap"))
    except OSError:
        return peak_rss_bytes()


def memory_message(peak: int, limit_gb, step: int, clips: int, seconds: float) -> str:
    return (f"memory {peak / (1 << 30):.1f} GB passed the {limit_gb} GB limit at step {step} "
            f"(a batch of {clips} clips, {seconds:.0f} s of audio padded); lower batch_seconds in the config")


class Watchdog(threading.Thread):
    """Samples the footprint four times a second, keeps its peak, and ends the process (exit 3) the
    moment it passes the limit, mid-step, before the Mac starts swapping. The last checkpoint stands."""

    def __init__(self, limit: int, limit_gb):
        super().__init__(daemon=True)
        self.limit, self.limit_gb, self.peak = limit, limit_gb, 0
        self.step, self.clips, self.seconds = 0, 0, 0.0

    def run(self) -> None:
        while True:
            now = footprint_bytes()
            self.peak = max(self.peak, now)
            if now > self.limit:
                _say(memory_message(now, self.limit_gb, self.step, self.clips, self.seconds))
                os._exit(MEMORY)
            time.sleep(0.25)


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _say(message: str) -> None:
    print(f"train: {message}", flush=True)


# --- torch (imported only in the NeMo interpreter) ------------------------------------------


def _lora_class():
    import torch
    from torch import nn

    class LoraLinear(nn.Module):
        """`lora_tiny.LoraLinear` (T0.4) plus dropout on the LoRA path: base(x) + (α/r)·B·A·drop(x)."""

        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.a = nn.Parameter(torch.empty(rank, base.in_features))
            self.b = nn.Parameter(torch.zeros(base.out_features, rank))
            self.dropout = nn.Dropout(dropout)
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + (self.dropout(x) @ self.a.T @ self.b.T) * self.scale

    return LoraLinear


def attach(model, groups, rank: int, alpha: float, dropout: float) -> tuple[dict, dict]:
    """Freeze `model`, wrap every target linear in LoRA and return ({name: wrapper}, {group: names})."""
    from torch import nn

    LoraLinear = _lora_class()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    resolved = resolve_targets(linears, groups)
    wrappers = {}
    for name in sorted(n for names in resolved.values() for n in names):
        wrapper = LoraLinear(model.get_submodule(name), rank, alpha, dropout)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, wrapper)
        wrappers[name] = wrapper
    return wrappers, resolved


def rng_state() -> dict:
    """Every generator a step draws from: the CPU's, and CUDA's or MPS's when the device exists, so
    LoRA dropout on an accelerator resumes with the same masks."""
    import torch

    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state) -> None:
    import torch

    if not isinstance(state, dict):  # a checkpoint from before accelerator state: the CPU's alone
        state = {"cpu": state}
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def lora_state(wrappers: dict) -> dict:
    return {f"{name}{suffix}": getattr(w, part).detach().clone().contiguous()
            for name, w in wrappers.items() for suffix, part in ((".lora_a", "a"), (".lora_b", "b"))}


def load_lora_state(wrappers: dict, state: dict) -> None:
    import torch

    with torch.no_grad():
        for name, w in wrappers.items():
            w.a.copy_(state[f"{name}.lora_a"])
            w.b.copy_(state[f"{name}.lora_b"])


def save_delta(path: Path, state: dict, rank: int, alpha: float, base_sha256: str) -> None:
    from safetensors.torch import save_file

    tmp = path.with_suffix(".tmp")
    save_file(state, str(tmp), metadata={"format": DELTA_FORMAT, "rank": str(rank), "alpha": str(alpha),
                                         "scale": str(alpha / rank), "base_sha256": base_sha256})
    os.replace(tmp, path)


def load_model(base: Path, transcribe_cpp: Path):
    """M0's load (`lora_tiny.load`): the offline patch Unified needs, then `restore_from` on the CPU."""
    sys.path.insert(0, str(transcribe_cpp / "scripts"))
    from dump_reference_parakeet_nemo import _patch_conformer_for_offline

    _patch_conformer_for_offline(force_regular_att_style=True)
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

    try:
        return ASRModel.restore_from(str(base), map_location="cpu")
    except TypeError as error:
        if "abstract class" not in str(error):
            raise
        return EncDecRNNTBPEModel.restore_from(str(base), map_location="cpu")


class Trainer:
    def __init__(self, job: dict):
        import torch

        self.job, self.cfg = job, job["config"]
        self.bundle = Path(job["bundle"])
        self.work, self.out = Path(job["work_dir"]), Path(job["out_dir"])
        self.device = torch.device(self.cfg.get("device", "cpu"))
        self.max_rss = int(float(self.cfg["max_rss_gb"]) * (1 << 30))
        self.stop = False
        self.watchdog: Watchdog | None = None
        entries = [json.loads(line) for line in (self.bundle / "manifest.jsonl").read_text().splitlines() if line]
        self.entries = smoke_subset(entries, int(self.cfg.get("max_clips") or 0), self.cfg)
        self.durations = [float(e["duration"]) for e in self.entries]
        self.train_idx, self.val_idx = validation_split(self.entries, float(self.cfg["validation_share"]), int(self.cfg["seed"]))
        if not self.train_idx:
            raise TargetError("the bundle has no training entries")

    # data
    def batch(self, indices: list[int]):
        import numpy as np
        import soundfile
        import torch

        audio = [soundfile.read(self.bundle / self.entries[i]["audio_filepath"], dtype="float32")[0] for i in indices]
        tokens = [self.model.tokenizer.text_to_ids(self.entries[i]["text"]) for i in indices]
        signal_ = np.zeros((len(audio), max(map(len, audio))), dtype=np.float32)
        for row, samples in enumerate(audio):
            signal_[row, : len(samples)] = samples
        transcript = np.zeros((len(tokens), max(1, max(map(len, tokens)))), dtype=np.int64)
        for row, ids in enumerate(tokens):
            transcript[row, : len(ids)] = ids
        to = lambda t: t.to(self.device)  # noqa: E731
        return (to(torch.from_numpy(signal_)), to(torch.tensor([len(s) for s in audio])),
                to(torch.from_numpy(transcript)), to(torch.tensor([len(ids) for ids in tokens])))

    def epoch_batches(self, epoch: int) -> list[list[int]]:
        """Epoch `epoch`'s batches: `WeightedRandomSampler` over the train weights, seeded per epoch."""
        import torch
        from torch.utils.data import WeightedRandomSampler

        generator = torch.Generator().manual_seed(int(self.cfg["seed"]) * 1000 + epoch)
        weights = [float(self.entries[i]["weight"]) for i in self.train_idx]
        drawn = list(WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator))
        order = [self.train_idx[j] for j in drawn]
        return batches_by_seconds(order, self.durations, float(self.cfg["batch_seconds"]), int(self.cfg["batch_max_clips"]))

    def loss(self, indices: list[int], wer: bool = False):
        signal_, signal_len, transcript, transcript_len = self.batch(indices)
        encoded, encoded_len = self.model.forward(input_signal=signal_, input_signal_length=signal_len)
        decoder, target_len, _ = self.model.decoder(targets=transcript, target_length=transcript_len)
        joint = self.model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss = self.model.loss(log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_len)
        if wer:
            self.model.wer.update(predictions=encoded, predictions_lengths=encoded_len,
                                  targets=transcript, targets_lengths=transcript_len)
        return loss

    def padded_seconds(self, indices: list[int]) -> float:
        return len(indices) * max(self.durations[i] for i in indices)

    def watch(self, step: int, indices: list[int]) -> None:
        """Tell the watchdog which batch is running, for its message."""
        if self.watchdog:
            self.watchdog.step, self.watchdog.clips, self.watchdog.seconds = step, len(indices), self.padded_seconds(indices)

    def check_memory(self, step: int, indices: list[int]) -> int:
        peak = max(peak_rss_bytes(), footprint_bytes(), self.watchdog.peak if self.watchdog else 0)
        if peak > self.max_rss:
            raise MemoryLimit(memory_message(peak, self.cfg["max_rss_gb"], step, len(indices), self.padded_seconds(indices)))
        return peak

    def set_lora_training(self, on: bool) -> None:
        for w in self.wrappers.values():
            w.dropout.train(on)

    def validate(self) -> tuple[float | None, float | None]:
        import torch

        if not self.val_idx:
            return None, None
        self.set_lora_training(False)
        self.model.wer.reset()
        losses, weights = [], []
        with torch.no_grad():
            for indices in batches_by_seconds(self.val_idx, self.durations, float(self.cfg["batch_seconds"]),
                                              int(self.cfg["batch_max_clips"])):
                self.watch(-1, indices)
                losses.append(float(self.loss(indices, wer=True)))
                weights.append(len(indices))
        wer_value = self.model.wer.compute()[0]
        self.model.wer.reset()
        self.set_lora_training(True)
        return sum(l * w for l, w in zip(losses, weights)) / sum(weights), float(wer_value)

    def save_checkpoint(self, state: dict) -> None:
        import torch

        path = self.work / "checkpoint.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({**state, "key": self.job["key"], "lora": lora_state(self.wrappers), "optimizer": self.optimizer.state_dict(),
                    "rng": rng_state()}, tmp)
        os.replace(tmp, path)

    def run(self) -> int:
        import torch

        cfg = self.cfg
        torch.manual_seed(int(cfg["seed"]))
        torch.set_num_threads(int(cfg["threads"]))
        self.watchdog = Watchdog(self.max_rss, cfg["max_rss_gb"])
        self.watchdog.start()
        started = time.monotonic()
        self.model = load_model(Path(self.job["base"]), Path(self.job["transcribe_cpp"]))
        self.model.eval()  # no SpecAugment and no base dropout, as in M0; only LoRA's dropout trains
        self.model.wer.log_prediction = False  # no transcript text on stdout (T6.2's log rule)
        self.wrappers, resolved = attach(self.model, cfg["targets"], int(cfg["rank"]), float(cfg["alpha"]), float(cfg["dropout"]))
        self.model.to(self.device)
        self.set_lora_training(True)
        self.loaded = time.monotonic()
        trainable = [p for w in self.wrappers.values() for p in (w.a, w.b)]
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        _atomic_json(self.work / "targets.json", {
            "groups": resolved, "trainable_parameters": sum(p.numel() for p in trainable), "frozen_parameters": frozen,
            "trainable_names": sorted(n for n, p in self.model.named_parameters() if p.requires_grad)})
        _say(f"lora: {len(self.wrappers)} linears ({', '.join(f'{g} {len(n)}' for g, n in resolved.items())}), "
             f"{sum(p.numel() for p in trainable):,} trainable parameters; "
             f"{len(self.train_idx)} train and {len(self.val_idx)} validation entries")
        self.optimizer = torch.optim.AdamW(trainable, lr=float(cfg["lr"]), weight_decay=0.0)

        state = {"step": 0, "epoch": 0, "batch": 0, "log": [], "best": None, "best_step": None, "best_loss": None,
                 "stale": 0, "active_seconds": 0.0, "resumes": 0, "probe": None, "peak_rss_bytes": 0,
                 "pending_losses": []}  # train losses since the last validation: they pick the best step
        path = self.work / "checkpoint.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("key") != self.job["key"]:
                _say(f"{path} belongs to another round; remove {self.work} to start over")
                return BAD_JOB
            load_lora_state(self.wrappers, saved["lora"])
            self.optimizer.load_state_dict(saved["optimizer"])
            set_rng_state(saved["rng"])
            state = {k: saved.get(k, state[k]) for k in state}
            state["resumes"] += 1
            _say(f"resumed at step {state['step']} (epoch {state['epoch'] + 1}, batch {state['batch']})")

        max_steps = int(cfg.get("max_steps") or 0)
        epochs, every, validate_every = int(cfg["epochs"]), int(cfg["checkpoint_every"]), int(cfg["validate_every"])
        patience = int(cfg["early_stopping_patience"])
        segment, losses, stopped = started, state["pending_losses"], None
        if max_steps and state["step"] >= max_steps:
            stopped = "max_steps"
        elif self.val_idx and state["stale"] >= patience:
            stopped = "early_stopping"

        def elapsed() -> float:
            return state["active_seconds"] + time.monotonic() - segment

        def evaluate() -> None:
            val_loss, val_wer = self.validate()
            train_loss = sum(losses) / len(losses) if losses else None
            losses.clear()
            state["log"].append({"step": state["step"], "epoch": min(state["epoch"] + 1, epochs), "train_loss": train_loss,
                                 "val_loss": val_loss, "val_wer": val_wer, "seconds": round(elapsed(), 1)})
            _say(f"step {state['step']} train_loss {train_loss if train_loss is None else round(train_loss, 4)} "
                 f"val_loss {val_loss if val_loss is None else round(val_loss, 4)} "
                 f"val_wer {val_wer if val_wer is None else round(val_wer, 4)}")
            score = val_loss if val_loss is not None else train_loss
            if score is not None and (state["best_loss"] is None or score < state["best_loss"]):
                state["best"], state["best_step"], state["best_loss"], state["stale"] = lora_state(self.wrappers), state["step"], score, 0
            else:
                state["stale"] += 1

        def checkpoint() -> None:
            self.save_checkpoint({**state, "active_seconds": elapsed()})

        try:
            if state["probe"] is None:
                state["probe"] = self.probe()
            while state["epoch"] < epochs and stopped is None:
                batches = self.epoch_batches(state["epoch"])
                while state["batch"] < len(batches):
                    indices = batches[state["batch"]]
                    self.watch(state["step"] + 1, indices)
                    loss = self.loss(indices)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    state["step"] += 1
                    state["batch"] += 1
                    losses.append(float(loss))
                    state["peak_rss_bytes"] = max(state["peak_rss_bytes"], self.check_memory(state["step"], indices))
                    if state["step"] == 1 or state["step"] % 10 == 0:
                        _say(f"step {state['step']} loss {float(loss):.4f} ({elapsed():.0f}s)")
                    if state["step"] % validate_every == 0:
                        evaluate()
                        if self.val_idx and state["stale"] >= patience:
                            stopped = "early_stopping"
                    if max_steps and state["step"] >= max_steps:
                        stopped = stopped or "max_steps"
                    if state["step"] % every == 0 or stopped or self.stop:
                        checkpoint()
                    if self.stop:
                        _say(f"interrupted; checkpoint at step {state['step']}. Rerun the same command to resume")
                        return INTERRUPTED
                    if stopped:
                        break
                else:
                    state["epoch"] += 1
                    state["batch"] = 0
            stopped = stopped or "epochs"
            if not state["log"] or state["log"][-1]["step"] != state["step"]:
                evaluate()
                checkpoint()
        except MemoryLimit as error:
            _say(str(error))
            return MEMORY

        best = state["best"] if state["best"] is not None else lora_state(self.wrappers)
        self.out.mkdir(parents=True, exist_ok=True)
        save_delta(self.out / "lora_delta.safetensors", best, int(cfg["rank"]), float(cfg["alpha"]), self.job["base_sha256"])
        wall = elapsed()
        audio_seconds = sum(self.durations[i] for i in self.train_idx)
        _atomic_json(self.out / "train_log.json", {
            "entries": state["log"], "steps": state["step"], "stopped": stopped, "best_step": state["best_step"],
            "epochs_completed": state["epoch"], "resumes": state["resumes"], "wall_seconds": round(wall, 1),
            "wall_hours": round(wall / 3600, 3), "estimate_hours": [3, 4],
            "load_seconds": round(self.loaded - started, 1),
            "train_entries": len(self.train_idx), "validation_entries": len(self.val_idx),
            "train_audio_seconds": round(audio_seconds, 1), "peak_rss_bytes": max(state["peak_rss_bytes"], peak_rss_bytes()),
            "probe": state["probe"], "max_rss_bytes": self.max_rss})
        _say(f"done: {state['step']} steps ({stopped}), best step {state['best_step']}, {wall:.0f}s")
        return 0

    def probe(self) -> dict:
        """One forward and backward on the longest clips, before any step, so a batch that would not fit
        fails the round with a message instead of swapping. Weights and RNG are left as they were."""
        import torch

        indices = longest_batch(self.durations, float(self.cfg["batch_seconds"]), int(self.cfg["batch_max_clips"]))
        self.watch(0, indices)
        rng = rng_state()
        loss = self.loss(indices)
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)
        set_rng_state(rng)
        peak = self.check_memory(0, indices)
        seconds = self.padded_seconds(indices)
        _say(f"memory probe: {len(indices)} longest clips ({seconds:.1f} s padded), peak {peak / (1 << 30):.2f} GB")
        return {"clips": len(indices), "padded_seconds": round(seconds, 1), "peak_rss_bytes": peak}


def environment() -> dict:
    import platform

    import torch

    versions = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import nemo

        versions["nemo"] = nemo.__version__
    except ImportError:
        versions["nemo"] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lora.py")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text())
    work = Path(job["work_dir"])
    _atomic_json(work / "env.json", environment())
    try:
        trainer = Trainer(job)
    except TargetError as error:
        _say(f"error: {error}")
        return BAD_JOB

    def interrupt(signum, _frame):
        if trainer.stop:  # a second Ctrl-C stops at once; the last checkpoint stands
            raise KeyboardInterrupt
        trainer.stop = True
        _say("stopping after this step to save a checkpoint (Ctrl-C again to stop now)")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        return trainer.run()
    except TargetError as error:
        _say(f"error: {error}")
        return BAD_JOB
    except KeyboardInterrupt:
        return INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
