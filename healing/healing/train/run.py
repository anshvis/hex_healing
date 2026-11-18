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
starts a new round. The trainer runs under `caffeinate -i -w <pid>`, so the Mac does not idle-sleep.

Paths, each recorded in the run: NeMo's Python is `--python`, else `$HEALING_NEMO_PYTHON`, else the
M0 environment; transcribe.cpp (the offline patch Unified needs) is `$TRANSCRIBE_CPP`, else M0's
checkout; the base is `--base`, else `$HEALING_BASE_NEMO`, else M0's model cache
(`$MODEL_CACHE/<slug>/<revision>/<file>`, as `spikes/m0/roundtrip.sh` keeps it); rounds go under
`--rounds-dir`, else `$HEALING_ROUNDS_DIR`, else `$HEALING_HOME/rounds` (point it at a bigger disk).

Registering a candidate is T6.3's: `healing export <out/lora_delta.safetensors> --config
<out/lora.yaml> --snapshot <train split>` merges, converts and registers. This stops at the delta.
"""

from __future__ import annotations

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


def log_strings(log) -> list[str]:
    """Every string value in the train log that is not one of its fixed words. The log is numbers only,
    so anything here (a transcript, a path) is a leak."""
    if isinstance(log, dict):
        return [s for value in log.values() for s in log_strings(value)]
    if isinstance(log, list):
        return [s for value in log for s in log_strings(value)]
    return [log] if isinstance(log, str) and log not in STOP_REASONS else []


def _start(command: list[str], keep_awake: bool) -> tuple[subprocess.Popen, subprocess.Popen | None]:
    child = subprocess.Popen(command, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    awake = None
    caffeinate = shutil.which("caffeinate") if keep_awake else None
    if caffeinate:
        awake = subprocess.Popen([caffeinate, "-i", "-w", str(child.pid)], stdin=subprocess.DEVNULL)
    return child, awake


def _wait(child: subprocess.Popen) -> int:
    """The trainer's exit code. Ctrl-C reaches it too (one process group): it checkpoints and exits."""
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue  # the trainer got the same SIGINT; wait for its checkpoint


def train_local(bundle: Path, config_path: Path | None = None, smoke: bool = False, python: Path | None = None,
                base: Path | None = None, rounds: Path | None = None, keep_awake: bool = True,
                say=print) -> int:
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
    work, out = folder / "work", folder / "out"
    for path in (folder, work, out):
        owner_dir(path)
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
    child, awake = _start([str(paths.python), "-I", str(TRAINER), "--job", str(work / "trainer.json")], keep_awake)
    try:
        code = _wait(child)
    finally:
        if awake and awake.poll() is None:
            awake.send_signal(signal.SIGTERM)
    if code != 0:
        why = {lora.INTERRUPTED: "interrupted; rerun the same command to resume from the last checkpoint",
               lora.MEMORY: "over the memory limit (see above)", lora.BAD_JOB: "the trainer refused the job (see above)"}
        say(f"error: the trainer exited {code}: {why.get(code, 'see its output above')}")
        return code
    return _finish(folder, job, config_path, effective, smoke, paths, started_at, time.monotonic() - started, say)


def _finish(folder: Path, job: dict, config_path: Path, effective: str, smoke: bool, paths: Paths,
            started_at: str, seconds: float, say) -> int:
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
    if log_strings(log):
        raise TrainError(f"{log_path} holds text ({len(log_strings(log))} strings); it must hold numbers only")
    env = json.loads((work / "env.json").read_text())
    record = {
        "status": "complete", "round": folder.name, "key": job["key"], "smoke": smoke,
        "started": started_at, "finished": _now(), "last_process_seconds": round(seconds, 1),
        "bundle": {"path": job["bundle"], "sha256": job["bundle_sha256"]},
        "config": {"source": str(config_path), "path": str(out / EFFECTIVE), "sha256": config_module.text_sha256(effective)},
        "base": {"path": job["base"], "sha256": job["base_sha256"]},
        "trainer": {"python": str(paths.python), "python_version": env.get("python"), "torch": env.get("torch"),
                    "nemo": env.get("nemo"), "transcribe_cpp": str(paths.transcribe_cpp),
                    "transcribe_cpp_rev": _git_rev(paths.transcribe_cpp), "script": str(TRAINER)},
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
