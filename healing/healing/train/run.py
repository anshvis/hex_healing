"""`healing train --local`: one LoRA round on this Mac, resumable (T6.5).

`healing` never imports NeMo or torch (D-29). It checks the inputs, writes a JSON job and runs
`lora.py` as a script in NeMo's interpreter, then checks what came back:

    <rounds>/<bundle>-<key>/
      work/trainer.json   the job (config, paths, the round's key)
      work/checkpoint.pt  LoRA, optimizer, RNG and sampler position; a rerun resumes from it
      work/env.json       the trainer's Python, torch and NeMo versions
      work/targets.json   the modules LoRA wrapped and the parameter counts
      out/lora_delta.safetensors, out/train_log.json   (T6.3's input; the log holds no text)
      out/lora.yaml       the effective config (after --smoke), the one the registry hashes
      out/run.json        the run record

The round's key is the SHA-256 of the effective config, the bundle's sha and the base's sha, so the
same command after a crash, sleep or Ctrl-C finds the same folder and resumes, and a changed input
starts a new round. The trainer runs under `caffeinate -i -w <pid>`, so the Mac does not idle-sleep,
in its own session: SIGINT and SIGTERM reach it through this wrapper, which waits for its checkpointed
exit. `<round>/round.lock` admits one writer per round.

Paths, each recorded in the run: NeMo's Python is `--python`, else `$HEALING_NEMO_PYTHON`, else the
M0 environment; transcribe.cpp (the offline patch Unified needs) is `$TRANSCRIBE_CPP`, else M0's
checkout; the base is `--base`, else `$HEALING_BASE_NEMO`, else M0's model cache
(`$MODEL_CACHE/<slug>/<revision>/<file>`, as `spikes/m0/roundtrip.sh` keeps it); rounds go under
`--rounds-dir`, else `$HEALING_ROUNDS_DIR`, else `$HEALING_HOME/rounds` (point it at a bigger disk).

Registering a candidate is T6.3's: `healing export <out/lora_delta.safetensors> --config
<out/lora.yaml> --snapshot <train split>` merges, converts and registers. This stops at the delta.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from healing.paths import healing_home
from healing.registry.objects import owner_dir, sha256_file
from healing.train import config as config_module, lora
from healing.train.manifest import TrainError, verify_bundle

TRAINER = Path(lora.__file__).resolve()
WORK = Path.home() / "hex_healing_work"
DELTA, LOG, RECORD, EFFECTIVE = "lora_delta.safetensors", "train_log.json", "run.json", "lora.yaml"


@dataclass
class Paths:
    python: Path
    transcribe_cpp: Path
    base: Path
    rounds: Path


def default_paths(config: dict, python: Path | None = None, base: Path | None = None, rounds: Path | None = None) -> Paths:
    env = os.environ
    cache = Path(env.get("MODEL_CACHE") or WORK / "models")
    spec = config["base"]
    return Paths(
        python=Path(python or env.get("HEALING_NEMO_PYTHON") or WORK / "transcribe.cpp/scripts/envs/parakeet/.venv/bin/python"),
        transcribe_cpp=Path(env.get("TRANSCRIBE_CPP") or WORK / "transcribe.cpp"),
        base=Path(base or env.get("HEALING_BASE_NEMO") or cache / spec["repo"].split("/")[-1] / spec["revision"] / spec["file"]),
        rounds=Path(rounds or env.get("HEALING_ROUNDS_DIR") or healing_home() / "rounds"),
    )


def base_sha256(path: Path, rounds: Path) -> str:
    """The base checkpoint's SHA-256, cached by path, size and mtime (hashing 2.4 GB takes a while)."""
    stat = path.stat()
    cache_path = rounds / "base-hashes.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    hit = cache.get(str(path))
    if hit and {k: hit.get(k) for k in stamp} == stamp:
        return hit["sha256"]
    sha = sha256_file(path)
    cache[str(path)] = {**stamp, "sha256": sha}
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, cache_path)
    return sha


def _git_rev(folder: Path) -> str | None:
    done = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() or None


def _code_sha() -> str | None:
    from healing.eval.bench import git_sha

    return git_sha() or None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


STOP_REASONS = {"max_steps", "epochs", "early_stopping"}
NUMBER, COUNT, OPTIONAL_NUMBER = "number", "count", "number or null"
ENTRY = {"step": COUNT, "epoch": COUNT, "train_loss": OPTIONAL_NUMBER, "val_loss": OPTIONAL_NUMBER,
         "val_wer": OPTIONAL_NUMBER, "seconds": NUMBER}
PROBE = {"clips": COUNT, "padded_seconds": NUMBER, "peak_rss_bytes": COUNT}
LOG_FIELDS = {"steps": COUNT, "best_step": "count or null", "epochs_completed": COUNT, "resumes": COUNT,
       "wall_seconds": NUMBER, "wall_hours": NUMBER, "load_seconds": OPTIONAL_NUMBER, "train_entries": COUNT,
       "validation_entries": COUNT, "train_audio_seconds": NUMBER, "peak_rss_bytes": COUNT, "max_rss_bytes": COUNT}


def _type_ok(value, kind: str) -> bool:
    if value is None:
        return kind.endswith("or null")
    if isinstance(value, bool):
        return False
    if kind.startswith("count"):
        return isinstance(value, int) and value >= 0
    return isinstance(value, (int, float))


def _fields(where: str, value, schema: dict) -> list[str]:
    if not isinstance(value, dict):
        return [f"{where} is not a mapping"]
    problems = [f"{where} has an unknown key ({len(key)} characters)" for key in value if key not in schema]
    problems += [f"{where}.{key} is not a {schema[key]}" for key in value if key in schema and not _type_ok(value[key], schema[key])]
    return problems


def log_problems(log) -> list[str]:
    """Why a train log is not the trainer's fixed schema of numbers. Every key must be one the trainer
    writes and every value a number of the right kind, so neither a key nor a value can carry text; the
    one string allowed is `stopped`, and only as a stop reason. (A key's text is never echoed.)"""
    if not isinstance(log, dict):
        return ["the log is not a mapping"]
    problems = _fields("log", {k: v for k, v in log.items() if k not in ("entries", "probe", "stopped", "estimate_hours")}, LOG_FIELDS)
    if "stopped" in log and log["stopped"] not in STOP_REASONS:
        problems.append("log.stopped is not a stop reason")
    hours = log.get("estimate_hours", [])
    if not isinstance(hours, list) or not all(_type_ok(h, NUMBER) for h in hours):
        problems.append("log.estimate_hours is not a list of numbers")
    if "probe" in log and log["probe"] is not None:
        problems += _fields("log.probe", log["probe"], PROBE)
    entries = log.get("entries", [])
    if not isinstance(entries, list):
        return problems + ["log.entries is not a list"]
    for index, entry in enumerate(entries):
        problems += _fields(f"log.entries[{index}]", entry, ENTRY)
    return problems


@contextlib.contextmanager
def round_lock(folder: Path):
    """One writer per round: an exclusive lock on `<round>/round.lock`, held while the trainer runs and
    released with its holder (the OS drops it when the process dies, so a crash leaves no stale lock)."""
    handle = open(folder / "round.lock", "a")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TrainError(f"another `healing train` is already training this round ({folder}); "
                             f"wait for it or stop it, then rerun") from None
        yield
    finally:
        handle.close()


def _run(command: list[str], keep_awake: bool) -> int:
    """Start the trainer and return its exit code. SIGINT and SIGTERM to this wrapper are forwarded to
    the trainer (each one: a second Ctrl-C stops it at once, as `lora.py` says), and the wrapper keeps
    waiting for its checkpointed exit, so no trainer outlives it. The handlers go in before the trainer
    is spawned: it runs in its own session, so a terminal Ctrl-C reaches only this wrapper, and one that
    lands while it starts is held and forwarded once the trainer exists. caffeinate is stopped and
    reaped either way."""
    state: dict = {"child": None, "pending": []}

    def forward(signum, _frame):
        child = state["child"]
        if child is None:
            state["pending"].append(signum)
        elif child.poll() is None:
            child.send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    awake = None
    try:
        child = subprocess.Popen(command, env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        state["child"] = child
        for signum in state["pending"]:
            child.send_signal(signum)
        caffeinate = shutil.which("caffeinate") if keep_awake else None
        if caffeinate:
            awake = subprocess.Popen([caffeinate, "-i", "-w", str(child.pid)], stdin=subprocess.DEVNULL)
        return child.wait()
    finally:
        try:
            child = state["child"]
            if child is not None and child.poll() is None:  # only if starting or waiting failed
                child.kill()
                child.wait()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
            if awake is not None:
                if awake.poll() is None:
                    awake.terminate()
                awake.wait()


@dataclass
class Round:
    """One round's folder and the checked inputs it trains on (`open_round`)."""

    folder: Path
    key: str
    config: dict
    effective: str
    config_path: Path
    smoke: bool
    paths: Paths
    bundle: Path
    bundle_sha: str
    base_sha: str

    @property
    def work(self) -> Path:
        return self.folder / "work"

    @property
    def out(self) -> Path:
        return self.folder / "out"


def open_round(bundle: Path, config_path: Path | None = None, smoke: bool = False, python: Path | None = None,
               base: Path | None = None, rounds: Path | None = None) -> Round:
    """Check the inputs and make the round's folder (the same one for the same inputs)."""
    config_path = Path(config_path or config_module.DEFAULT)
    config, effective = config_module.load(config_path, smoke=smoke)
    paths = default_paths(config, python, base, rounds)
    bundle = Path(bundle).resolve()
    bundle_sha = verify_bundle(bundle)
    if not paths.base.is_file():
        raise TrainError(f"the base checkpoint {paths.base} is missing (X-1); pass --base or set HEALING_BASE_NEMO")
    if not paths.python.is_file():
        raise TrainError(f"NeMo's Python {paths.python} is missing; pass --python or set HEALING_NEMO_PYTHON")
    owner_dir(paths.rounds)
    base_sha = base_sha256(paths.base, paths.rounds)
    if base_sha != config["base"]["sha256"]:
        raise TrainError(f"{paths.base} is {base_sha[:12]}, the config pins {config['base']['sha256'][:12]}")
    key = hashlib.sha256(f"{config_module.text_sha256(effective)}\n{bundle_sha}\n{base_sha}\n".encode()).hexdigest()
    folder = paths.rounds / f"{bundle.name}-{key[:12]}"
    for path in (folder, folder / "work", folder / "out"):
        owner_dir(path)
    return Round(folder, key, config, effective, config_path, smoke, paths, bundle, bundle_sha, base_sha)


def train_round(r: Round, keep_awake: bool = True, say=print) -> int:
    with round_lock(r.folder):
        return train_held(r, keep_awake, say)


def train_held(r: Round, keep_awake: bool = True, say=print) -> int:
    """`train_round` for a caller that already holds the round's lock (`healing train auto` keeps it
    through export and the scratch's removal)."""
    return _train(r.folder, r.work, r.out, r.key, r.config, r.effective, r.config_path, r.smoke, r.paths,
                  r.bundle, r.bundle_sha, r.base_sha, keep_awake, say)


def train_local(bundle: Path, config_path: Path | None = None, smoke: bool = False, python: Path | None = None,
                base: Path | None = None, rounds: Path | None = None, keep_awake: bool = True,
                say=print) -> int:
    return train_round(open_round(bundle, config_path, smoke, python, base, rounds), keep_awake, say)


def _train(folder: Path, work: Path, out: Path, key: str, config: dict, effective: str, config_path: Path,
           smoke: bool, paths: Paths, bundle: Path, bundle_sha: str, base_sha: str, keep_awake: bool, say) -> int:
    record_path = out / RECORD
    if record_path.exists() and json.loads(record_path.read_text()).get("status") == "complete":
        say(f"already trained: {out / DELTA}")
        return 0

    job = {"key": key, "config": config, "bundle": str(bundle), "bundle_sha256": bundle_sha, "base": str(paths.base),
           "base_sha256": base_sha, "transcribe_cpp": str(paths.transcribe_cpp), "work_dir": str(work), "out_dir": str(out)}
    (work / "trainer.json").write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
    (out / EFFECTIVE).write_text(effective)
    resuming = (work / "checkpoint.pt").exists()
    say(f"round {folder.name}: {'resuming' if resuming else 'starting'} ({'smoke' if smoke else 'full'}) in {folder}")
    started, started_at = time.monotonic(), _now()
    code = _run([str(paths.python), "-I", str(TRAINER), "--job", str(work / "trainer.json")], keep_awake)
    if code != 0:
        why = {lora.INTERRUPTED: "interrupted; rerun the same command to resume from the last checkpoint",
               lora.MEMORY: "over the memory limit (see above)", lora.BAD_JOB: "the trainer refused the job (see above)"}
        say(f"error: the trainer exited {code}: {why.get(code, 'see its output above')}")
        return code
    return _finish(folder, job, config_path, effective, smoke, paths, started_at, time.monotonic() - started, say)


def _finish(folder: Path, job: dict, config_path: Path, effective: str, smoke: bool, paths: Paths | None,
            started_at: str, seconds: float, say, runner: dict | None = None) -> int:
    """Check what the trainer wrote and record the run. `runner` describes a trainer that ran somewhere
    else (`healing train --remote`), in place of the local `paths`."""
    work, out = folder / "work", folder / "out"
    delta, log_path = out / DELTA, out / LOG
    if not delta.is_file() or not log_path.is_file():
        raise TrainError(f"the trainer exited 0 without {DELTA} and {LOG} in {out}")
    targets = json.loads((work / "targets.json").read_text())
    header = lora.safetensors_header(delta)
    names = [n for n in header if n != "__metadata__"]
    problems = lora.delta_problems(names, [n for group in targets["groups"].values() for n in group])
    if problems:
        raise TrainError(f"{delta}: " + "; ".join(problems[:5]))
    log = json.loads(log_path.read_text())
    problems = log_problems(log)
    if problems:
        raise TrainError(f"{log_path} is not the trainer's log of numbers only: " + "; ".join(problems[:5]))
    env = json.loads((work / "env.json").read_text())
    record = {
        "status": "complete", "round": folder.name, "key": job["key"], "smoke": smoke,
        "started": started_at, "finished": _now(), "last_process_seconds": round(seconds, 1),
        "bundle": {"path": job["bundle"], "sha256": job["bundle_sha256"]},
        "config": {"source": str(config_path), "path": str(out / EFFECTIVE), "sha256": config_module.text_sha256(effective)},
        "base": {"path": job["base"], "sha256": job["base_sha256"]},
        "trainer": {**(runner if runner is not None else {
                        "python": str(paths.python), "transcribe_cpp": str(paths.transcribe_cpp),
                        "transcribe_cpp_rev": _git_rev(paths.transcribe_cpp)}),
                    "python_version": env.get("python"), "torch": env.get("torch"), "nemo": env.get("nemo"),
                    "script": str(TRAINER)},
        "code_sha": _code_sha(),
        "targets": {g: len(n) for g, n in targets["groups"].items()},
        "trainable_parameters": targets["trainable_parameters"],
        "outputs": {"delta": {"path": str(delta), "sha256": sha256_file(delta), "tensors": len(names)},
                    "train_log": {"path": str(log_path), "sha256": sha256_file(log_path)}},
        "steps": log["steps"], "stopped": log["stopped"], "resumes": log["resumes"],
        "wall_seconds": log["wall_seconds"], "peak_rss_bytes": log["peak_rss_bytes"],
    }
    tmp = out / (RECORD + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out / RECORD)
    say(f"delta {delta} ({len(names)} tensors over {len(names) // 2} modules)")
    say(f"{log['steps']} steps ({log['stopped']}), {log['wall_seconds']:.0f} s, peak memory "
        f"{log['peak_rss_bytes'] / (1 << 30):.2f} GB, {log['resumes']} resume(s)")
    say(f"next (T6.3): healing export {delta} --config {out / EFFECTIVE}")
    return 0


def main(args) -> int:
    try:
        return train_local(args.bundle, args.config, smoke=args.smoke, python=args.python, base=args.base,
                           rounds=args.rounds_dir)
    except (TrainError, config_module.ConfigError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
