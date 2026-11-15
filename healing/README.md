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

## Running your own Hex build

Self-healing needs Hex changes that aren't released, so the Hex you dictate with is a local build of
this branch, installed as the only `/Applications/Hex.app`. Without a Developer ID certificate, build
it ad-hoc signed with updates off:

```sh
cd ..                      # repo root
HEX_RELEASE_UNSIGNED=1 HEX_LOCAL_BUILD=1 scripts/install-app.sh
rm -rf target/app/Hex.app  # so macOS never launches the build copy instead
```

Put the CLI on your PATH once: `ln -s /Applications/Hex.app/Contents/MacOS/hex ~/.local/bin/hex`.

- `HEX_RELEASE_UNSIGNED=1` keeps the normal app name and bundle id (`anomaly.hex`), so it uses the
  usual `~/Library/Application Support/hex` data folder: models, settings and History carry over.
- `HEX_LOCAL_BUILD=1` turns updates off, so it never replaces itself with an official release.
- `install-app.sh` stops the running Hex, replaces `/Applications/Hex.app` and opens it.
- Don't use `HEX_TEST_BUILD=1` for this: it builds `Hex Test.app` with its own `hex-test` data folder,
  which `healing` doesn't read unless `HEX_BIN` points at it.
- A plain `scripts/install-app.sh` needs `HEX_TEAM_ID` and a Developer ID identity, and fails without them.

The signature is ad-hoc, so macOS ties the Accessibility grant to that exact binary and every rebuild
invalidates it, even though System Settings still shows Hex as allowed. If Hex says "Waiting for setup"
or the shortcut stops working after a rebuild, reset the grant and relaunch, then allow Hex again if
asked:

```sh
tccutil reset Accessibility anomaly.hex && hex quit; open /Applications/Hex.app
```
 Turn on Settings › Labs › Keep
recordings, or `healing collect` has no audio to copy.

## Commands

| Command | What it does |
|---|---|
| `healing collect` | Copies Hex's History (`hex history --json --raw`) and kept recordings into `dataset.sqlite` and `audio/<sha256>.flac`. Idempotent. Entries Hex later deletes stay. |
| `healing status [--json]` | Clip count, bytes against the budget, clusters, and warnings such as History expiring uncollected entries. `--list` lists every warning code. |
| `healing prune [--budget B] [--k K] [--dry-run]` | Clusters near-duplicate text, keeps k audio files per cluster, and evicts the most redundant, oldest weak audio to hold the budget. |
| `healing clean [--deep] [--dry-run]` | Makes the index match the disk: quarantines orphans, marks missing or corrupt audio (`--deep` re-hashes), sweeps temp files, holds a free-space floor. |
| `healing schedule install\|remove\|show\|run` | The launchd job that runs collect and prune every 6 hours. |
| `healing sets build-general` | Builds the fixed general regression set: 300 LibriSpeech test-other and 200 AMI IHM utterances, seeded and SHA-pinned. A second run prints `unchanged`; a different set is refused. |
| `healing eval baseline [--model <gguf>]` | Scores a model on the general set with `hex benchmark-transcription` and writes `healing/eval/baseline.json`. |
| `healing stats paired <a.jsonl> <b.jsonl>` | Δ WER (b − a) with a paired-bootstrap 95% CI, clips as the resampling unit. Takes benchmark JSONL or `nemo_wer.py` output. |
| `healing label corrections [--since] [--recompute]` | Aligns each edited clip's `edited_text` with `final_text`, maps edits back onto `raw_text`, and classifies them `acoustic` (sounds alike), `rewrite` or `unmapped`. Acoustic edits make `reference_text` and a `strong` label. Skips clips an OpenCode rewrite produced. |
| `healing report observation [--json]` | Per app: pastes the lab saw, how many were observed, the `span_not_found` rate and the edited rate. |
| `healing protocol corrections plan\|run\|score` | The seeded edit-recall protocol (X-3). `plan` writes `RUNBOOK.md`; `run` needs a person at a Mac. |

## M0 spikes

`spikes/m0/` holds the feasibility spikes and their [report](spikes/m0/REPORT.md).
`roundtrip.sh <unified|tdt-v2>` runs the whole shipping path: base control, LoRA fine-tune,
conversion, Q8_0, `hex model import` and scoring. It needs a
[transcribe.cpp](https://github.com/anomalyco/transcribe.cpp) checkout (`$TRANSCRIBE_CPP`), whose
pinned Parakeet environment provides NeMo (D-29). Expect about 25 minutes and 10 GB of disk on an
M4.
