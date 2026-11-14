"""`healing eval baseline`: the shipped model's scores on the general set."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from healing.eval.bench import BenchRun, run_benchmark
from healing.paths import healing_home, repo_root
from healing.sets.general import sources_by_clip

DEFAULT_MODEL_ID = "parakeet-unified-en-0.6b"
DEFAULT_GGUF = (
    Path.home()
    / "Library/Application Support/hex/models/parakeet-unified-en-0.6b-Q8_0.gguf"
)


def wer_by_source(run: BenchRun, sources: dict[str, str]) -> dict[str, dict]:
    totals: dict[str, list[int]] = {}
    for clip in run.clips:
        errors_words = totals.setdefault(sources[clip["id"]], [0, 0])
        errors_words[0] += clip["word_errors"]
        errors_words[1] += clip["reference_words"]
    return {
        source: {"wer": errors / words, "word_errors": errors, "reference_words": words}
        for source, (errors, words) in sorted(totals.items())
    }


def baseline(model: Path, model_id: str, output: Path) -> dict:
    set_dir = healing_home() / "sets" / "general"
    manifest = set_dir / "manifest.json"
    if not manifest.is_file():
        raise SystemExit("no general set: run `healing sets build-general` first")
    jsonl = healing_home() / "evals" / f"baseline-{_stamp()}.jsonl"
    run, command = run_benchmark(manifest, model=model, output=jsonl)
    sources = sources_by_clip(set_dir)
    result = {
        "model": model_id,
        "gguf_sha256": _sha256(model),
        "set_sha256": (set_dir / "manifest.sha256").read_text().split()[0],
        "clips": run.summary["clips"],
        "reference_words": run.summary["reference_words"],
        "wer": run.summary["wer"],
        "wer_by_source": wer_by_source(run, sources),
        "median_rtf": run.summary["median_rtf"],
        "load_ms": run.summary["load_ms"],
        "hex_binary": command[0],
        "hex_version": _hex_version(command[0]),
        # The commit the binary was built from, when it is this checkout's own
        # release build; an installed Hex records only its version.
        "hex_sha": _git_sha() if _built_here(command[0]) else None,
        "repo_sha": _git_sha(),
        "command": shlex.join(command),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def default_output() -> Path:
    return repo_root() / "healing" / "eval" / "baseline.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_version(binary: str) -> str:
    completed = subprocess.run([binary, "--version"], capture_output=True, text=True)
    return completed.stdout.strip()


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return completed.stdout.strip()


def _built_here(binary: str) -> bool:
    return Path(binary).resolve() == (repo_root() / "target" / "release" / "hex").resolve()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
