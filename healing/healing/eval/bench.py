"""Run `hex benchmark-transcription` and read its JSONL.

Field names come from `src/transcription/benchmark.rs` (the `model`, `clip`
and `summary` records); never guess them.
"""

from __future__ import annotations

import json
import os
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from healing.paths import healing_home, repo_root

INSTALLED_HEX = Path("/Applications/Hex.app/Contents/MacOS/hex")


class BenchError(RuntimeError):
    pass


class VocabularyUnsupported(BenchError):
    """The hex binary predates `benchmark-transcription --vocabulary` (T3.4)."""


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


def _find_hex(builds: list[Path]) -> Path:
    named = os.environ.get("HEX_BIN")
    for candidate in ([Path(named)] if named else []) + builds:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    on_path = shutil.which("hex")
    if on_path:
        return Path(on_path)
    raise BenchError(
        "no hex binary: build one with `cargo build --release` or set HEX_BIN"
    )


def hex_binary() -> Path:
    """`$HEX_BIN`, else this checkout's release build, else the installed app."""
    return _find_hex([repo_root() / "target" / "release" / "hex", INSTALLED_HEX])


def history_hex() -> Path:
    """The `hex` to read History with: `$HEX_BIN`, else the installed app, else this checkout's build.

    Reading History needs the Hex that writes it; a checkout's build may be from another branch
    that doesn't know its newer fields.
    """
    return _find_hex([INSTALLED_HEX, repo_root() / "target" / "release" / "hex"])


def isolated_env() -> dict[str, str]:
    """The environment every `hex` call runs in: its own data folder.

    `hex` opens its process log in its data folder before dispatching any
    command, so without this a benchmark appends to the real Hex's logs
    (AGENTS.md: a benchmarked copy never points at the user's data folder).
    """
    support = healing_home() / "hex-support"
    support.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "HEX_APPLICATION_SUPPORT_DIR": str(support),
        "HEX_DISABLE_UPDATES": "1",
    }


def benchmark_command(
    manifest: Path,
    *,
    model: Path | None,
    runs: int,
    warmups: int,
    binary: Path | None = None,
    vocabulary: Path | None = None,
    boost: str | None = None,
) -> list[str]:
    command = [str(binary or hex_binary()), "benchmark-transcription", str(manifest)]
    if model is not None:
        command += ["--model", str(model)]
    if vocabulary is not None:
        command += ["--vocabulary", str(vocabulary)]
        if boost:
            command += ["--boost", boost]
    return command + ["--runs", str(runs), "--warmups", str(warmups)]


def run_benchmark(
    manifest: Path,
    *,
    model: Path | None = None,
    runs: int = 1,
    warmups: int = 1,
    output: Path | None = None,
    vocabulary: Path | None = None,
    boost: str | None = None,
) -> tuple[BenchRun, list[str]]:
    command = benchmark_command(
        manifest, model=model, runs=runs, warmups=warmups, vocabulary=vocabulary, boost=boost
    )
    completed = subprocess.run(command, capture_output=True, text=True, env=isolated_env())
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
        if vocabulary is not None and "--vocabulary" in completed.stderr and (
            "unexpected argument" in completed.stderr or "unrecognized" in completed.stderr
        ):
            raise VocabularyUnsupported(
                f"{command[0]} doesn't accept --vocabulary yet (it needs the T3.4 build). "
                "Rerun with --vocabulary none to score without your vocabulary."
            )
        raise BenchError(f"benchmark exited {completed.returncode}:\n{tail}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(completed.stdout)
    return parse_jsonl(completed.stdout), command


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hex_version(binary: str) -> str:
    completed = subprocess.run([binary, "--version"], capture_output=True, text=True, env=isolated_env())
    return completed.stdout.strip()


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return completed.stdout.strip()


def built_here(binary: str) -> bool:
    """Whether `binary` is this checkout's own release build (so `git_sha` names its source)."""
    return Path(binary).resolve() == (repo_root() / "target" / "release" / "hex").resolve()


def wer_by_source(run: BenchRun, sources: dict[str, str]) -> dict[str, dict]:
    """The run's word errors summed per clip source (the general set's `librispeech` and `ami`)."""
    totals: dict[str, list[int]] = {}
    for clip in run.clips:
        errors_words = totals.setdefault(sources[clip["id"]], [0, 0])
        errors_words[0] += clip["word_errors"]
        errors_words[1] += clip["reference_words"]
    return {
        source: {"wer": errors / words, "word_errors": errors, "reference_words": words}
        for source, (errors, words) in sorted(totals.items())
    }
