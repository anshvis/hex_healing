"""Run `hex benchmark-transcription` and read its JSONL.

Field names come from `src/transcription/benchmark.rs` (the `model`, `clip`
and `summary` records); never guess them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from healing.paths import repo_root

INSTALLED_HEX = Path("/Applications/Hex.app/Contents/MacOS/hex")


class BenchError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchRun:
    model: dict
    clips: list[dict]
    summary: dict

    def clip(self, clip_id: str) -> dict:
        for record in self.clips:
            if record["id"] == clip_id:
                return record
        raise KeyError(clip_id)


def parse_jsonl(text: str) -> BenchRun:
    model = summary = None
    clips: list[dict] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchError(f"benchmark line {number} is not JSON: {line[:80]}") from error
        kind = record.get("type")
        if kind == "model":
            model = record
        elif kind == "clip":
            clips.append(record)
        elif kind == "summary":
            summary = record
    if model is None or summary is None:
        raise BenchError("benchmark output has no model or summary record")
    for field in ("wer", "word_errors", "reference_words", "median_rtf", "quality", "groups"):
        if field not in summary:
            raise BenchError(f"benchmark summary has no {field!r}: the output shape changed")
    if len(clips) != summary["clips"]:
        raise BenchError(f"{len(clips)} clip records for {summary['clips']} clips")
    return BenchRun(model, clips, summary)


def hex_binary() -> Path:
    """`$HEX_BIN`, else this checkout's release build, else the installed app."""
    named = os.environ.get("HEX_BIN")
    candidates = [Path(named)] if named else []
    candidates += [repo_root() / "target" / "release" / "hex", INSTALLED_HEX]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    on_path = shutil.which("hex")
    if on_path:
        return Path(on_path)
    raise BenchError(
        "no hex binary: build one with `cargo build --release` or set HEX_BIN"
    )


def benchmark_command(
    manifest: Path, *, model: Path | None, runs: int, warmups: int, binary: Path | None = None
) -> list[str]:
    command = [str(binary or hex_binary()), "benchmark-transcription", str(manifest)]
    if model is not None:
        command += ["--model", str(model)]
    return command + ["--runs", str(runs), "--warmups", str(warmups)]


def run_benchmark(
    manifest: Path,
    *,
    model: Path | None = None,
    runs: int = 1,
    warmups: int = 1,
    output: Path | None = None,
) -> tuple[BenchRun, list[str]]:
    command = benchmark_command(manifest, model=model, runs=runs, warmups=warmups)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
        raise BenchError(f"benchmark exited {completed.returncode}:\n{tail}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(completed.stdout)
    return parse_jsonl(completed.stdout), command
