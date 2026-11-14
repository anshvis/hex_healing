#!/usr/bin/env bash
# M0 round trip: base control (T0.1) and LoRA fine-tune (T0.4) through the
# whole shipping path, scored in Hex's own runtime.
#
#   healing/spikes/m0/roundtrip.sh <unified|tdt-v2>
#
# Base half (no training needed): fetch the pinned base .nemo → convert to F32
# GGUF with transcribe.cpp's converter → Q8_0 → `hex model import` → score the
# imported copy and NeMo on LibriSpeech 300 → score the target word.
# Tuned half: synthetic Qorvyx clips → LoRA train and merge → the same
# conversion, import and scoring → ‖ΔW‖ table → verdict.
#
# Every `hex` call runs with HEX_APPLICATION_SUPPORT_DIR=$SPIKE_DIR/hex, so the
# real Hex data folder is never touched (G-8).
#
# Env: SPIKE_DIR (default: a new temp dir), TRANSCRIBE_CPP (a checkout),
# TRANSCRIBE_CPP_REV (default 7045d1c, the catalog's converter), HEX_BIN
# (default: the installed Hex), MODEL_CACHE (where base .nemo files are kept),
# BASE_NEMO (a local base checkpoint instead of the pinned download).
# Exit 0 only when every check passes; otherwise the failing step is named.
set -uo pipefail

base="${1:?usage: roundtrip.sh <unified|tdt-v2>}"
case "$base" in
    unified)
        repo=nvidia/parakeet-unified-en-0.6b; slug=parakeet-unified-en-0.6b
        revision=fe53cd885760c96b6a5f51a0bfd362cb4584a98b; offline=--offline-only ;;
    tdt-v2)
        repo=nvidia/parakeet-tdt-0.6b-v2; slug=parakeet-tdt-0.6b-v2
        revision=ae9ad07059c7c739ffaf932226a8fe64ae2620b0; offline=--no-offline-only ;;
    *) echo "unknown base: $base" >&2; exit 2 ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
healing="$(cd "$here/../.." && pwd)"
spike="${SPIKE_DIR:-$(mktemp -d -t hex-m0)}"
mkdir -p "$spike" && spike="$(cd "$spike" && pwd)"
tcpp="${TRANSCRIBE_CPP:-$HOME/hex_healing_work/transcribe.cpp}"
rev="${TRANSCRIBE_CPP_REV:-7045d1c}"
hex="${HEX_BIN:-/Applications/Hex.app/Contents/MacOS/hex}"
cache="${MODEL_CACHE:-$HOME/hex_healing_work/models}"
general="${HEALING_HOME:-$HOME/Library/Application Support/hex-healing}/sets/general"
base_nemo="${BASE_NEMO:-$cache/$slug/$revision/$slug.nemo}"
nemo_env=(uv run --project "$tcpp/scripts/envs/parakeet")
export HEX_APPLICATION_SUPPORT_DIR="$spike/hex" HEX_DISABLE_UPDATES=1
mkdir -p "$spike"/{base,tuned,runs,hex}
steps="$spike/steps.jsonl"
: > "$steps"

step() {
    local name="$1"; shift
    local started=$SECONDS
    echo "==> $name"
    # A subshell, so a step function's `set -e` ends the step, not the script.
    ( "$@" ) > "$spike/runs/$name.log" 2>&1
    local status=$?
    printf '{"step":"%s","exit":%d,"seconds":%d}\n' "$name" "$status" $((SECONDS - started)) >> "$steps"
    if [[ $status -ne 0 ]]; then
        echo "FAILED at step '$name' (exit $status); last lines of $spike/runs/$name.log:" >&2
        tail -15 "$spike/runs/$name.log" >&2
        exit 1
    fi
}

# The converter must be exactly $rev, and the quantizer built from it.
checkout_converter() {
    set -e
    git -C "$tcpp" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$tcpp" fetch origin "$rev"
    git -C "$tcpp" checkout -q "$rev"
    local head; head="$(git -C "$tcpp" rev-parse HEAD)"
    [[ "$head" == "$rev"* ]] || { echo "transcribe.cpp is at $head, not $rev"; return 1; }
    # A patched converter is another repo's change (X-4): refuse a dirty tree.
    [[ -z "$(git -C "$tcpp" status --porcelain --untracked-files=no)" ]] ||
        { echo "transcribe.cpp has local changes; refusing to convert"; return 1; }
    local stamp="$tcpp/build/.healing-built-rev"
    if [[ ! -x "$tcpp/build/bin/transcribe-quantize" || "$(cat "$stamp" 2>/dev/null)" != "$head" ]]; then
        cmake -S "$tcpp" -B "$tcpp/build" -DCMAKE_BUILD_TYPE=Release \
            -DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_BUILD_TESTS=OFF
        cmake --build "$tcpp/build" --target transcribe-quantize -j 8
        echo "$head" > "$stamp"
    fi
    echo "converter $head"
}

fetch_base() {
    if [[ -n "${BASE_NEMO:-}" ]]; then
        echo "using BASE_NEMO=$BASE_NEMO (not fetched)"
        return 0
    fi
    "${nemo_env[@]}" python - "$repo" "$revision" "$cache/$slug/$revision" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo, revision, folder = sys.argv[1:]
print(hf_hub_download(repo, repo.split("/")[1] + ".nemo", revision=revision, local_dir=folder))
EOF
}

general_subset() {
    (cd "$healing" && uv run healing sets subset --prefix ls- --name ls300 --expected 300)
}

convert() { # <nemo> <out dir> <file name>
    set -e
    mkdir -p "$2/$slug"
    (cd "$tcpp" && "${nemo_env[@]}" scripts/convert-parakeet.py "$1" "$2/$slug/$3-F32.gguf")
    "$tcpp/build/bin/transcribe-quantize" "$2/$slug/$3-F32.gguf" "$2/$slug/$3-Q8_0.gguf" --quant Q8_0
}

# Imports a GGUF into the isolated data folder and records the imported copy's
# path, which every benchmark of that model then scores.
import_model() { # <gguf> <name>
    set -e
    local line id
    line="$("$hex" model import "$1" --name "spike-$2-$base")"
    echo "$line"
    id="$(awk '{print $2}' <<< "$line")"
    local copy="$spike/hex/models/custom/${id#custom-}.gguf"
    [[ -f "$copy" ]] || { echo "no imported copy at $copy"; return 1; }
    echo "$copy" > "$spike/runs/$2.imported"
}

imported() { cat "$spike/runs/$1.imported"; }

bench() { # <model name> <manifest> <out>
    "$hex" benchmark-transcription "$2" --model "$(imported "$1")" --runs 1 --warmups 1 > "$3"
}

nemo_score() { # <nemo> <manifest> <out>
    (cd "$healing" && uv run python spikes/m0/nemo_wer.py --nemo "$1" --manifest "$2" --out "$3" "$offline")
}

step converter checkout_converter
step fetch fetch_base
step subset general_subset
step clips bash -c "cd '$healing' && uv run python spikes/m0/make_clips.py '$spike/qorvyx'"

step convert-base convert "$base_nemo" "$spike/base" base
step import-base import_model "$spike/base/$slug/base-Q8_0.gguf" base
step bench-base bench base "$general/ls300.json" "$spike/runs/gguf-base-ls.jsonl"
step qorvyx-base bench base "$spike/qorvyx/holdout.json" "$spike/runs/gguf-base-qorvyx.jsonl"
step nemo-base nemo_score "$base_nemo" "$general/ls300.json" "$spike/runs/nemo-base-ls.jsonl"

step train "${nemo_env[@]}" python "$here/lora_tiny.py" --base "$base_nemo" \
    --train "$spike/qorvyx/train.json" --out "$spike/tuned/merged.nemo" --adapter "$spike/tuned/lora.pt"
step convert-tuned convert "$spike/tuned/merged.nemo" "$spike/tuned" tuned
step import-tuned import_model "$spike/tuned/$slug/tuned-Q8_0.gguf" tuned
step bench-tuned bench tuned "$general/ls300.json" "$spike/runs/gguf-tuned-ls.jsonl"
step qorvyx-tuned bench tuned "$spike/qorvyx/holdout.json" "$spike/runs/gguf-tuned-qorvyx.jsonl"
step nemo-tuned nemo_score "$spike/tuned/merged.nemo" "$general/ls300.json" "$spike/runs/nemo-tuned-ls.jsonl"

step delta "${nemo_env[@]}" python "$here/tensor_delta.py" \
    "$spike/base/$slug/base-F32.gguf" "$spike/tuned/$slug/tuned-F32.gguf" \
    "$spike/base/$slug/base-Q8_0.gguf" "$spike/tuned/$slug/tuned-Q8_0.gguf" "$spike/runs/delta.json"
step verdict env TRANSCRIBE_CPP_REV="$rev" bash -c "cd '$healing' && uv run python spikes/m0/verdict.py '$spike' '$base' '$tcpp' '$base_nemo' '$revision' '$slug'"
cat "$spike/verdict.json"
