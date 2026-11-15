"""`healing eval baseline`: the shipped model's scores on the general set.

An alias of `healing eval --model <gguf> --sets general --vocabulary none` (one eval path), kept so
T0.3's recorded baseline and its commands still work: it writes the same `baseline.json`, with the
per-source WER breakdown the general set carries.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

from healing.eval.bench import BenchRun, git_sha, hex_binary, sha256
from healing.paths import repo_root

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
    from healing.eval import run

    document = run.evaluate([model], ["general"], vocabulary="none")
    if "general" in document["unavailable_sets"]:
        raise SystemExit(document["unavailable_sets"]["general"])
    scores = document["scores"][str(model)]["general"]
    (ref,) = document["sets"]
    result = {
        "model": model_id,
        "gguf_sha256": document["models"][0]["gguf_sha256"],
        "set_sha256": ref["manifest_sha256"],
        "clips": ref["n"],
        "reference_words": scores["ref_words"],
        "wer": scores["wer"],
        "wer_by_source": scores["wer_by_source"],
        "median_rtf": scores["median_rtf"],
        "load_ms": scores["load_ms"],
        "hex_binary": str(hex_binary()),
        "hex_version": document["hex_version"],
        "hex_binary_sha256": document["hex_binary_sha256"],
        "hex_sha": document["hex_sha"],
        "repo_sha": git_sha(),
        "command": shlex.join(["healing", "eval", "--model", str(model), "--sets", "general", "--vocabulary", "none"]),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def default_output() -> Path:
    return repo_root() / "healing" / "eval" / "baseline.json"
