#!/usr/bin/env bash
# M0 round trip: base control (T0.1) and LoRA fine-tune (T0.4) through the
# whole shipping path, scored in Hex's own runtime.
#
#   healing/spikes/m0/roundtrip.sh <unified|tdt-v2>
#
# Steps: fetch base .nemo → synthetic Qorvyx clips → LoRA train and merge →
# convert base and merged to F32 GGUF (transcribe.cpp's converter) → Q8_0 →
# `hex model import` (isolated HEX_APPLICATION_SUPPORT_DIR) → benchmark both
# GGUFs and both NeMo checkpoints → parity CIs, Qorvyx hit/miss, ‖ΔW‖ table.
#
# Env: SPIKE_DIR (default: a new temp dir), TRANSCRIBE_CPP (a checkout),
# TRANSCRIBE_CPP_REV (default 7045d1c, the catalog's converter), HEX_BIN
# (default: the installed Hex), MODEL_CACHE (where base .nemo files are kept),
# BASE_NEMO (a local base checkpoint instead of the fetched one).
# Exit 0 only when every check passes; otherwise the failing step is named.
set -uo pipefail

base="${1:?usage: roundtrip.sh <unified|tdt-v2>}"
case "$base" in
    unified) repo=nvidia/parakeet-unified-en-0.6b; slug=parakeet-unified-en-0.6b; offline=--offline-only ;;
    tdt-v2) repo=nvidia/parakeet-tdt-0.6b-v2; slug=parakeet-tdt-0.6b-v2; offline=--no-offline-only ;;
    *) echo "unknown base: $base" >&2; exit 2 ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
healing="$(cd "$here/../.." && pwd)"
spike="${SPIKE_DIR:-$(mktemp -d -t hex-m0)}"
tcpp="${TRANSCRIBE_CPP:-$HOME/hex_healing_work/transcribe.cpp}"
rev="${TRANSCRIBE_CPP_REV:-7045d1c}"
hex="${HEX_BIN:-/Applications/Hex.app/Contents/MacOS/hex}"
cache="${MODEL_CACHE:-$HOME/hex_healing_work/models}"
general="${HEALING_HOME:-$HOME/Library/Application Support/hex-healing}/sets/general"
nemo_env=(uv run --project "$tcpp/scripts/envs/parakeet")
mkdir -p "$spike"/{base,tuned,runs,hex}
steps="$spike/steps.jsonl"
: > "$steps"

step() {
    local name="$1"; shift
    local started=$SECONDS
    echo "==> $name"
    "$@" > "$spike/runs/$name.log" 2>&1
    local status=$?
    printf '{"step":"%s","exit":%d,"seconds":%d}\n' "$name" "$status" $((SECONDS - started)) >> "$steps"
    if [[ $status -ne 0 ]]; then
        echo "FAILED at step '$name' (exit $status); last lines of $spike/runs/$name.log:" >&2
        tail -15 "$spike/runs/$name.log" >&2
        exit 1
    fi
}

checkout_converter() {
    git -C "$tcpp" fetch -q origin "$rev" && git -C "$tcpp" checkout -q "$rev"
    [[ -x "$tcpp/build/bin/transcribe-quantize" ]] || {
        cmake -S "$tcpp" -B "$tcpp/build" -DCMAKE_BUILD_TYPE=Release \
            -DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_BUILD_TESTS=OFF &&
        cmake --build "$tcpp/build" --target transcribe-quantize -j 8
    }
}

fetch_base() {
    "${nemo_env[@]}" python - "$repo" "$cache" <<'EOF'
import sys
from huggingface_hub import HfApi, hf_hub_download
repo, cache = sys.argv[1:]
sha = HfApi().model_info(repo).sha
name = repo.split("/")[1]
print(hf_hub_download(repo, f"{name}.nemo", revision=sha, local_dir=f"{cache}/{name}"))
print("revision", sha)
EOF
}

convert() { # <nemo> <out dir> <file name>
    mkdir -p "$2/$slug"
    (cd "$tcpp" && "${nemo_env[@]}" scripts/convert-parakeet.py "$1" "$2/$slug/$3-F32.gguf") &&
    "$tcpp/build/bin/transcribe-quantize" "$2/$slug/$3-F32.gguf" "$2/$slug/$3-Q8_0.gguf" --quant Q8_0
}

import_tuned() {
    HEX_APPLICATION_SUPPORT_DIR="$spike/hex" "$hex" model import \
        "$spike/tuned/$slug/tuned-Q8_0.gguf" --name "spike-tuned-$base"
}

bench() { # <gguf> <manifest> <out>
    "$hex" benchmark-transcription "$2" --model "$1" --runs 1 --warmups 1 > "$3"
}

nemo_score() { # <nemo> <manifest> <out>
    (cd "$healing" && uv run python spikes/m0/nemo_wer.py --nemo "$1" --manifest "$2" --out "$3" "$offline")
}

base_nemo="${BASE_NEMO:-$cache/$slug/$slug.nemo}"
step converter checkout_converter
step fetch fetch_base
step clips bash -c "cd '$healing' && uv run python spikes/m0/make_clips.py '$spike/qorvyx'"
step train "${nemo_env[@]}" python "$here/lora_tiny.py" --base "$base_nemo" \
    --train "$spike/qorvyx/train.json" --out "$spike/tuned/merged.nemo" --adapter "$spike/tuned/lora.pt"
step convert-base convert "$base_nemo" "$spike/base" base
step convert-tuned convert "$spike/tuned/merged.nemo" "$spike/tuned" tuned
step import import_tuned
step bench-base bench "$spike/base/$slug/base-Q8_0.gguf" "$general/ls300.json" "$spike/runs/gguf-base-ls.jsonl"
step bench-tuned bench "$spike/tuned/$slug/tuned-Q8_0.gguf" "$general/ls300.json" "$spike/runs/gguf-tuned-ls.jsonl"
step qorvyx-base bench "$spike/base/$slug/base-Q8_0.gguf" "$spike/qorvyx/holdout.json" "$spike/runs/gguf-base-qorvyx.jsonl"
step qorvyx-tuned bench "$spike/tuned/$slug/tuned-Q8_0.gguf" "$spike/qorvyx/holdout.json" "$spike/runs/gguf-tuned-qorvyx.jsonl"
step nemo-base nemo_score "$base_nemo" "$general/ls300.json" "$spike/runs/nemo-base-ls.jsonl"
step nemo-tuned nemo_score "$spike/tuned/merged.nemo" "$general/ls300.json" "$spike/runs/nemo-tuned-ls.jsonl"
step delta "${nemo_env[@]}" python "$here/tensor_delta.py" \
    "$spike/base/$slug/base-F32.gguf" "$spike/tuned/$slug/tuned-F32.gguf" \
    "$spike/base/$slug/base-Q8_0.gguf" "$spike/tuned/$slug/tuned-Q8_0.gguf" "$spike/runs/delta.json"
step verdict bash -c "cd '$healing' && uv run python spikes/m0/verdict.py '$spike' '$base'"
cat "$spike/verdict.json"
