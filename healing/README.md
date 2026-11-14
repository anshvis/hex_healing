# healing

The off-device half of self-healing Hex: test sets, evaluation, statistics and, later, capture,
labels, vocabulary mining and fine-tuning. The plan is in
[`docs/plans/self-healing/`](../docs/plans/self-healing/README.md).

It never changes Hex's behaviour on its own. It reads Hex's output, runs Hex's own
`hex benchmark-transcription`, and writes under `$HEALING_HOME`
(default `~/Library/Application Support/hex-healing`). Audio and datasets live there, never in
this repo.

## Install

```sh
cd healing
uv sync
uv run pytest
```

Python 3.12 and [uv](https://docs.astral.sh/uv/). The benchmark needs a `hex` binary: `$HEX_BIN`,
else this checkout's `target/release/hex`, else the installed `/Applications/Hex.app`.

## Commands

| Command | What it does |
|---|---|
| `healing sets build-general` | Builds the fixed general regression set: 300 LibriSpeech test-other and 200 AMI IHM utterances, seeded and SHA-pinned. A second run prints `unchanged`; a different set is refused. |
| `healing eval baseline [--model <gguf>]` | Scores a model on the general set with `hex benchmark-transcription` and writes `healing/eval/baseline.json`. |
| `healing stats paired <a.jsonl> <b.jsonl>` | Δ WER (b − a) with a paired-bootstrap 95% CI, clips as the resampling unit. Takes benchmark JSONL or `nemo_wer.py` output. |

## M0 spikes

`spikes/m0/` holds the feasibility spikes and their [report](spikes/m0/REPORT.md).
`roundtrip.sh <unified|tdt-v2>` runs the whole shipping path: base control, LoRA fine-tune,
conversion, Q8_0, `hex model import` and scoring. It needs a
[transcribe.cpp](https://github.com/anomalyco/transcribe.cpp) checkout (`$TRANSCRIBE_CPP`), whose
pinned Parakeet environment provides NeMo (D-29). Expect about 25 minutes and 10 GB of disk on an
M4.
