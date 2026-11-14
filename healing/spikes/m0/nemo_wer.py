"""Score a NeMo checkpoint on a benchmark manifest, in the benchmark's clip shape.

Runs transcribe.cpp's own NeMo reference runner
(`scripts/wer/run_reference_parakeet_nemo.py`, offline attention for Unified,
greedy decoding) and writes one `{"type": "clip", ...}` record per clip with
`word_errors` and `reference_words` counted exactly as Hex's benchmark counts
them, so `healing stats paired` can compare it with a GGUF run.

    uv run python spikes/m0/nemo_wer.py --nemo base.nemo \
        --manifest sets/general/ls300.json --out nemo-base.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from healing.wer import word_errors

TRANSCRIBE_CPP = Path(os.environ.get("TRANSCRIBE_CPP", Path.home() / "hex_healing_work/transcribe.cpp"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="benchmark manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offline-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    with tempfile.TemporaryDirectory() as scratch:
        runner_manifest = Path(scratch) / "manifest.jsonl"
        runner_manifest.write_text(
            "".join(
                json.dumps(
                    {"id": clip["id"], "audio": str((root / clip["audio"]).resolve()),
                     "ref_text": clip["expected"]}
                ) + "\n"
                for clip in manifest["clips"]
            )
        )
        raw = Path(scratch) / "nemo.jsonl"
        command = [
            "uv", "run", "--project", str(TRANSCRIBE_CPP / "scripts/envs/parakeet"),
            str(TRANSCRIBE_CPP / "scripts/wer/run_reference_parakeet_nemo.py"),
            "--model", str(args.nemo.resolve()),
            "--manifest", str(runner_manifest),
            "--out", str(raw),
            "--torch-threads", str(args.threads),
            "--batch-size", str(args.batch_size),
        ] + (["--offline-only"] if args.offline_only else [])
        completed = subprocess.run(command, cwd=TRANSCRIBE_CPP)
        if completed.returncode != 0:
            print(f"error: NeMo runner exited {completed.returncode}", file=sys.stderr)
            return 1
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]

    hypotheses = {row["id"]: row for row in rows if "hyp_text" in row}
    records, errors, words = [], 0, 0
    for clip in manifest["clips"]:
        row = hypotheses[clip["id"]]
        if row.get("error"):
            print(f"error: {clip['id']}: {row['error']}", file=sys.stderr)
            return 1
        clip_errors, clip_words = word_errors(clip["expected"], row["hyp_text"])
        errors, words = errors + clip_errors, words + clip_words
        records.append({
            "type": "clip", "backend": "nemo", "id": clip["id"],
            "word_errors": clip_errors, "reference_words": clip_words,
            "expected": clip["expected"], "transcript": row["hyp_text"],
            "group": clip.get("group", "ordinary"),
        })
    records.append({"type": "summary", "backend": "nemo", "clips": len(records),
                    "word_errors": errors, "reference_words": words, "wer": errors / words})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(f"nemo wer {errors / words:.4f} ({errors}/{words}) on {len(manifest['clips'])} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
