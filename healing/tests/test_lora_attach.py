"""T6.5: LoRA targets, freezing, the delta, batching and resume.

The pure parts run here. The torch parts (attach, freeze, the delta file, the training loop's
checkpoint and resume) run `lora.py` in NeMo's interpreter on a tiny stand-in model with
Parakeet's module names, so no NeMo checkpoint is loaded; they skip when that interpreter is absent.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile

from healing.train import lora

NEMO_PYTHON = Path(os.environ.get("HEALING_NEMO_PYTHON")
                   or Path.home() / "hex_healing_work/transcribe.cpp/scripts/envs/parakeet/.venv/bin/python")
needs_torch = pytest.mark.skipif(not NEMO_PYTHON.is_file(), reason="NeMo's interpreter (torch) is not installed")


def unified_linears(layers: int = 2) -> list[str]:
    """The linear module names of Parakeet Unified (read from the loaded model in M0 and T6.5)."""
    names = ["encoder.pre_encode.out", "joint.pred", "joint.enc", "joint.joint_net.2"]
    for i in range(layers):
        names += [f"encoder.layers.{i}.feed_forward1.linear1", f"encoder.layers.{i}.feed_forward1.linear2",
                  f"encoder.layers.{i}.feed_forward2.linear1", f"encoder.layers.{i}.feed_forward2.linear2"]
        names += [f"encoder.layers.{i}.self_attn.linear_{p}" for p in ("q", "k", "v", "out", "pos")]
    return names


GROUPS = ["encoder_attention", "encoder_feed_forward", "joint"]


# --- targets ------------------------------------------------------------------------------


def test_every_target_matches_modules():
    resolved = lora.resolve_targets(unified_linears(), GROUPS)
    assert len(resolved["encoder_attention"]) == 8  # q, k, v, out per layer; linear_pos is not wrapped (D-47)
    assert not any(n.endswith("linear_pos") for n in resolved["encoder_attention"])
    assert len(resolved["encoder_feed_forward"]) == 8
    assert resolved["joint"] == ["joint.enc", "joint.joint_net.2", "joint.pred"]
    assert "encoder.pre_encode.out" not in {n for names in resolved.values() for n in names}


def test_a_target_matching_nothing_fails():
    without_joint = [n for n in unified_linears() if not n.startswith("joint.")]
    with pytest.raises(lora.TargetError, match="joint matches no linear"):
        lora.resolve_targets(without_joint, GROUPS)
    without_out = [n for n in unified_linears() if not n.endswith("linear_out")]
    with pytest.raises(lora.TargetError, match="linear_out"):
        lora.resolve_targets(without_out, GROUPS)
    with pytest.raises(lora.TargetError, match="unknown target group"):
        lora.resolve_targets(unified_linears(), ["decoder_lstm"])


def test_tdt_v2_style_names_also_resolve():
    """TDT v2 names its feed-forward the same way; the groups resolve from whatever the model has."""
    names = [n.replace("feed_forward1", "feed_forward") for n in unified_linears()]
    resolved = lora.resolve_targets(names, GROUPS)
    assert all(resolved[g] for g in GROUPS)


# --- delta --------------------------------------------------------------------------------


def write_safetensors(path: Path, names: list[str], metadata: dict | None = None) -> None:
    header, offset = {}, 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def test_delta_contains_only_lora_tensors(tmp_path):
    targets = ["encoder.layers.0.self_attn.linear_q", "joint.pred"]
    good = [f"{t}{s}" for t in targets for s in lora.SUFFIXES]
    assert lora.delta_problems(good, targets) == []
    assert any("not a LoRA tensor" in p for p in lora.delta_problems(good + ["joint.pred.weight"], targets))
    assert any("not a LoRA target" in p for p in lora.delta_problems(good + ["decoder.x.lora_a", "decoder.x.lora_b"], targets))
    assert any("lacks" in p for p in lora.delta_problems(good[:-1], targets))
    encoder_only = ["encoder.layers.0.self_attn.linear_q"]
    assert any("encoder and the joint" in p for p in lora.delta_problems(good[:2], encoder_only))
    path = tmp_path / "d.safetensors"
    write_safetensors(path, good, {"format": lora.DELTA_FORMAT})
    header = lora.safetensors_header(path)
    assert sorted(n for n in header if n != "__metadata__") == sorted(good)


# --- batching -----------------------------------------------------------------------------


def test_batches_by_audio_seconds():
    durations = [10.0, 25.0, 30.0, 5.0, 5.0, 70.0, 1.0]
    batches = lora.batches_by_seconds(list(range(7)), durations, max_seconds=60, max_clips=16)
    assert batches == [[0, 1], [2, 3, 4], [5], [6]]  # a 70 s clip alone; none over 60 s otherwise
    for batch in batches:
        assert len(batch) == 1 or sum(durations[i] for i in batch) <= 60
    assert lora.batches_by_seconds([0, 1, 2], [1.0, 1.0, 1.0], 60, 2) == [[0, 1], [2]]  # clip cap too
    assert lora.longest_batch(durations, 60, 16) == [5]
    assert lora.longest_batch([3.0, 29.0, 28.0, 1.0], 60, 16) == [1, 2, 0]


def test_validation_takes_whole_clusters():
    entries = [{"id": f"c{i}", "cluster": f"k{i // 4}"} for i in range(40)]
    train, val = lora.validation_split(entries, 0.05, seed=13)
    assert val and set(train).isdisjoint(val) and sorted(train + val) == list(range(40))
    clusters = {entries[i]["cluster"] for i in val}
    assert all(entries[i]["cluster"] not in clusters for i in train)
    assert lora.validation_split(entries, 0.05, seed=13) == (train, val)
    assert lora.validation_split(entries, 0.0, seed=13)[1] == []
    one_cluster = [{"id": str(i), "cluster": "same"} for i in range(10)]
    assert lora.validation_split(one_cluster, 0.5, seed=1)[1] == []  # train never loses its only cluster


# --- torch: attach, freeze, the delta file, the loop ----------------------------------------

TINY = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    import torch
    from torch import nn
    sys.path.insert(0, sys.argv[1])
    import lora

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            for p in ("q", "k", "v", "out", "pos"):
                setattr(self, f"linear_{p}", nn.Linear(8, 8))
    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1, self.linear2 = nn.Linear(8, 8), nn.Linear(8, 8)
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.feed_forward1, self.feed_forward2 = Attention(), FeedForward(), FeedForward()
        def forward(self, x):
            a = self.self_attn
            x = x + a.linear_out(a.linear_q(x) * torch.sigmoid(a.linear_k(x)) + a.linear_v(x))
            return x + self.feed_forward2.linear2(torch.relu(self.feed_forward1.linear2(torch.relu(self.feed_forward1.linear1(x)))
                                                  + self.feed_forward2.linear1(x)))
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_encode = nn.Module(); self.pre_encode.out = nn.Linear(16, 8)
            self.layers = nn.ModuleList([Layer(), Layer()])
    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.pred, self.enc = nn.Linear(8, 8), nn.Linear(8, 8)
            self.joint_net = nn.Sequential(nn.ReLU(), nn.Dropout(0.0), nn.Linear(8, 5))
        def forward(self, encoder_outputs, decoder_outputs):
            return self.joint_net(self.enc(encoder_outputs).mean(1) + self.pred(decoder_outputs).mean(1))
    class Tokenizer:
        def text_to_ids(self, text):
            return [ord(c) % 4 + 1 for c in text][:6]
    class Wer:
        log_prediction = True
        def reset(self): self.n = 0
        def update(self, **kw): self.n = getattr(self, "n", 0) + 1
        def compute(self): return torch.tensor(0.25), None, None
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.encoder, self.joint = Encoder(), Joint()
            self.decoder_net = nn.LSTM(8, 8, batch_first=True)  # the predictor: an LSTM, frozen
            self.embed = nn.Embedding(6, 8)
            self.tokenizer, self.wer = Tokenizer(), Wer()
        def forward(self, input_signal, input_signal_length):
            frames = input_signal[:, : (input_signal.shape[1] // 16) * 16].reshape(input_signal.shape[0], -1, 16)
            x = self.encoder.pre_encode.out(frames)
            for layer in self.encoder.layers:
                x = layer(x)
            return x, input_signal_length // 16
        def decoder(self, targets, target_length):
            return self.decoder_net(self.embed(targets))[0], target_length, None
        def loss(self, log_probs, targets, input_lengths, target_lengths):
            return (log_probs - targets.float().mean()).pow(2).mean()
''')


def run_torch(tmp_path: Path, body: str, *args: str) -> subprocess.CompletedProcess:
    script = tmp_path / "harness.py"
    script.write_text(TINY + textwrap.dedent(body))
    return subprocess.run([str(NEMO_PYTHON), "-I", str(script), str(Path(lora.__file__).parent), *args],
                          capture_output=True, text=True, timeout=300)


@needs_torch
def test_only_lora_params_trainable_and_delta_file(tmp_path):
    done = run_torch(tmp_path, '''
        model = Tiny()
        base_count = sum(p.numel() for p in model.parameters())
        wrappers, resolved = lora.attach(model, ["encoder_attention", "encoder_feed_forward", "joint"], 4, 8.0, 0.05)
        trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        lora.save_delta(Path(sys.argv[2]), lora.lora_state(wrappers), 4, 8.0, "0" * 64)
        print(json.dumps({"trainable": trainable, "frozen": frozen, "base": base_count,
                          "targets": sorted(wrappers)}))
    ''', str(tmp_path / "delta.safetensors"))
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["frozen"] == result["base"]  # every base weight is frozen, the LSTM included
    assert len(result["targets"]) == 2 * 4 + 2 * 4 + 3
    assert all(n.endswith((".a", ".b")) for n in result["trainable"])
    assert {n.rsplit(".", 1)[0] for n in result["trainable"]} == set(result["targets"])
    header = lora.safetensors_header(tmp_path / "delta.safetensors")
    names = [n for n in header if n != "__metadata__"]
    assert lora.delta_problems(names, result["targets"]) == []
    assert header["__metadata__"]["format"] == lora.DELTA_FORMAT and header["__metadata__"]["scale"] == "2.0"


def tiny_bundle(folder: Path, clips: int = 12) -> Path:
    (folder / "audio").mkdir(parents=True)
    rng = np.random.default_rng(3)
    lines = []
    for i in range(clips):
        samples = (rng.standard_normal(1600 + 160 * i) * 0.1).astype(np.float32)
        soundfile.write(folder / f"audio/c{i}.wav", samples, 16_000, subtype="PCM_16")
        lines.append(json.dumps({"audio_filepath": f"audio/c{i}.wav", "duration": len(samples) / 16_000,
                                 "text": f"phrase {i}", "weight": 1.0 + (i % 3), "id": f"c{i}", "cluster": f"k{i // 2}"}))
    (folder / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return folder


LOOP = '''
    job = json.loads(Path(sys.argv[2]).read_text())
    stop_at = int(sys.argv[3])
    lora.load_model = lambda base, tcpp: Tiny()
    trainer = lora.Trainer(job)
    real = trainer.check_memory
    def check(step, indices):
        if step == stop_at:
            trainer.stop = True  # as SIGINT's handler does
        return real(step, indices)
    trainer.check_memory = check
    raise SystemExit(trainer.run())
'''


def job(tmp_path: Path, name: str, bundle: Path, **overrides) -> Path:
    work, out = tmp_path / name / "work", tmp_path / name / "out"
    work.mkdir(parents=True)
    config = {"targets": ["encoder_attention", "encoder_feed_forward", "joint"], "rank": 4, "alpha": 8, "dropout": 0.05,
              "lr": 1e-2, "epochs": 2, "max_steps": 7, "batch_seconds": 0.5, "batch_max_clips": 3,
              "validation_share": 0.2, "validate_every": 3, "early_stopping_patience": 99, "checkpoint_every": 2,
              "seed": 13, "threads": 1, "device": "cpu", "max_rss_gb": 64, **overrides}
    path = work / "trainer.json"
    path.write_text(json.dumps({"key": "k" * 64, "config": config, "bundle": str(bundle), "base": "unused",
                                "base_sha256": "0" * 64, "transcribe_cpp": "unused",
                                "work_dir": str(work), "out_dir": str(out)}))
    return path


def tensors(path: Path) -> tuple[dict, bytes]:
    """A safetensors file's tensor table and data, without its metadata (whose key order varies)."""
    header = lora.safetensors_header(path)
    header.pop("__metadata__", None)
    size = struct.unpack("<Q", path.read_bytes()[:8])[0]
    return header, path.read_bytes()[8 + size:]


@needs_torch
def test_resumes_from_checkpoint(tmp_path):
    """Stopped at step 3 (checkpointed), rerun: it resumes at step 3, and the delta is bit-identical
    to an uninterrupted run's, so the sampler position, optimizer and RNG all came back."""
    bundle = tiny_bundle(tmp_path / "bundle")
    interrupted = job(tmp_path, "interrupted", bundle)
    first = run_torch(tmp_path, LOOP, str(interrupted), "3")
    assert first.returncode == lora.INTERRUPTED, first.stdout + first.stderr
    assert (interrupted.parent / "checkpoint.pt").exists()
    assert not (tmp_path / "interrupted/out/lora_delta.safetensors").exists()
    second = run_torch(tmp_path, LOOP, str(interrupted), "-1")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "resumed at step 3" in second.stdout
    straight = job(tmp_path, "straight", bundle)
    third = run_torch(tmp_path, LOOP, str(straight), "-1")
    assert third.returncode == 0, third.stdout + third.stderr
    assert tensors(tmp_path / "interrupted/out/lora_delta.safetensors") == tensors(tmp_path / "straight/out/lora_delta.safetensors")
    log = json.loads((tmp_path / "interrupted/out/train_log.json").read_text())
    assert log["steps"] == 7 and log["stopped"] == "max_steps" and log["resumes"] == 1
    assert [e["step"] for e in log["entries"]] == [3, 6, 7]


@needs_torch
def test_no_text_in_train_log_or_stdout(tmp_path):
    bundle = tiny_bundle(tmp_path / "bundle")
    done = run_torch(tmp_path, LOOP, str(job(tmp_path, "r", bundle)), "-1")
    assert done.returncode == 0, done.stderr
    log_text = (tmp_path / "r/out/train_log.json").read_text()
    texts = [json.loads(line)["text"] for line in (bundle / "manifest.jsonl").read_text().splitlines()]
    assert not any(t in log_text or t in done.stdout for t in texts)
    from healing.train.run import log_strings

    assert log_strings(json.loads(log_text)) == []


@needs_torch
def test_memory_limit_fails_the_round_with_a_message(tmp_path):
    bundle = tiny_bundle(tmp_path / "bundle")
    done = run_torch(tmp_path, LOOP, str(job(tmp_path, "m", bundle, max_rss_gb=0.001)), "-1")
    assert done.returncode == lora.MEMORY
    assert "passed the 0.001 GB limit at step 0" in done.stdout and "lower batch_seconds" in done.stdout
