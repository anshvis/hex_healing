"""The M0 round trip's pass/fail, from the runs `roundtrip.sh` wrote.

    uv run python spikes/m0/verdict.py <spike dir> <base> <transcribe.cpp> <base .nemo> <revision>

Checks (AC-2):
- base control: NeMo base vs base Q8_0 GGUF, paired 95% CI of Δ WER ⊂ ±1.0 pt;
- tuned parity: merged NeMo vs tuned Q8_0 GGUF, the same bar;
- the target word: the base Q8_0 misses it (recall ≤ 20%), the tuned Q8_0
  gets it (recall ≥ 80%) on held-out clips;
- the conversions have identical tensor names;
- both GGUFs imported into Hex, and what was benchmarked is the imported copy
  whose receipt matches the converter's output byte for byte;
- the converter ran at the pinned revision;
- the tuned model runs within 10% of the base's speed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from healing.eval.bench import parse_jsonl
from healing.stats import paired_bootstrap, read_clip_scores

PARITY = 0.01
MISS_AT_MOST = 0.2
HIT_AT_LEAST = 0.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def import_identity(spike: Path, name: str, source: Path) -> dict:
    """The imported copy is the converter's output, and it is what was scored."""
    copy = Path((spike / "runs" / f"{name}.imported").read_text().strip())
    receipt = json.loads(copy.with_name(copy.name + ".verified.json").read_text())
    source_sha = sha256(source)
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "imported_copy": str(copy),
        "receipt_sha256": receipt["sha256"],
        "copy_sha256": sha256(copy),
        "ok": receipt["sha256"] == source_sha == sha256(copy)
        and copy.is_relative_to(spike / "hex"),
    }


def recall(run_path: Path) -> float:
    quality = parse_jsonl(run_path.read_text()).summary["quality"]
    return quality["recalled_terms"] / quality["expected_terms"]


def main() -> int:
    spike, base, tcpp, base_nemo, revision = Path(sys.argv[1]), sys.argv[2], *sys.argv[3:6]
    runs = spike / "runs"
    base_ci = paired_bootstrap(read_clip_scores(runs / "nemo-base-ls.jsonl"),
                               read_clip_scores(runs / "gguf-base-ls.jsonl"))
    tuned_ci = paired_bootstrap(read_clip_scores(runs / "nemo-tuned-ls.jsonl"),
                                read_clip_scores(runs / "gguf-tuned-ls.jsonl"))
    drift = paired_bootstrap(read_clip_scores(runs / "gguf-base-ls.jsonl"),
                             read_clip_scores(runs / "gguf-tuned-ls.jsonl"))
    base_run = parse_jsonl((runs / "gguf-base-ls.jsonl").read_text()).summary
    tuned_run = parse_jsonl((runs / "gguf-tuned-ls.jsonl").read_text()).summary
    delta = json.loads((runs / "delta.json").read_text())
    steps = [json.loads(line) for line in (spike / "steps.jsonl").read_text().splitlines()]
    target_base, target_tuned = recall(runs / "gguf-base-qorvyx.jsonl"), recall(runs / "gguf-tuned-qorvyx.jsonl")
    speed_ratio = tuned_run["median_rtf"] / base_run["median_rtf"]
    slug = next((spike / "base").iterdir()).name
    imports = {
        "base": import_identity(spike, "base", spike / "base" / slug / "base-Q8_0.gguf"),
        "tuned": import_identity(spike, "tuned", spike / "tuned" / slug / "tuned-Q8_0.gguf"),
    }
    converter = subprocess.run(
        ["git", "-C", tcpp, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    built = (Path(tcpp) / "build" / ".healing-built-rev").read_text().strip()

    checks = {
        "import_base": imports["base"]["ok"],
        "import_tuned": imports["tuned"]["ok"],
        "converter_pinned": converter.startswith(os.environ.get("TRANSCRIBE_CPP_REV", "7045d1c"))
        and built == converter,
        "base_parity": -PARITY <= base_ci.ci_low and base_ci.ci_high <= PARITY,
        "tuned_parity": -PARITY <= tuned_ci.ci_low and tuned_ci.ci_high <= PARITY,
        "target_base_miss": target_base <= MISS_AT_MOST,
        "target_tuned_hit": target_tuned >= HIT_AT_LEAST,
        "tensor_names_equal": delta["names_equal"],
        "speed_within_10pct": speed_ratio <= 1.10,
    }
    verdict = {
        "base": base,
        "pass": all(checks.values()),
        "checks": checks,
        "provenance": {
            "converter_rev": converter,
            "quantizer_built_from": built,
            "base_checkpoint": base_nemo,
            "base_checkpoint_sha256": sha256(Path(base_nemo)),
            "base_revision": revision,
            "imports": imports,
        },
        "base_parity": base_ci.as_dict(),
        "tuned_parity": tuned_ci.as_dict(),
        "tuned_vs_base_gguf": drift.as_dict(),
        "target_recall": {"base_q8": target_base, "tuned_q8": target_tuned},
        "speed": {"base_median_rtf": base_run["median_rtf"], "tuned_median_rtf": tuned_run["median_rtf"],
                  "base_load_ms": base_run["load_ms"], "tuned_load_ms": tuned_run["load_ms"],
                  "ratio": speed_ratio},
        "delta": {key: value for key, value in delta.items() if key != "rows"},
        "steps": steps,
    }
    (spike / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps({"pass": verdict["pass"], "checks": checks}, indent=1))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
