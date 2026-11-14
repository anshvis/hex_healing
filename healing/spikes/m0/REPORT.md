# M0 report: feasibility spikes and baseline

Measured on an Apple M4 (16 GB, macOS 15.6) on 2026-11-14, with the installed Hex 3.4.0 as the
runtime (`/Applications/Hex.app/Contents/MacOS/hex`, Metal). Raw runs stay under
`~/hex_healing_work/rt-unified-2/`. The verdict and the per-tensor table are committed as
[`results/unified-verdict.json`](results/unified-verdict.json) and
[`results/unified-delta.json`](results/unified-delta.json).

## Result

**Architecture: Parakeet Unified English 0.6B** (`nvidia/parakeet-unified-en-0.6b` at
`fe53cd8`), Hex's default model. It passed the base control (T0.1) and the LoRA round trip
(T0.4) on the first try, so TDT v2 wasn't needed. `roundtrip.sh tdt-v2` exists for when it is.

| Check | Bar | Result |
|---|---|---|
| Converter | transcribe.cpp `7045d1c` | the converter and `transcribe-quantize` accept both checkpoints, with no patch |
| Base control (T0.1) | paired 95% CI of WER(GGUF Q8_0) − WER(NeMo) ⊂ ±1.0 pt, 300 clips | NeMo 3.53%, GGUF 3.59%: **Δ +0.06 pt, CI [−0.12, +0.30]** |
| Tuned parity (T0.4) | same bar, merged NeMo vs tuned Q8_0 | NeMo 3.85%, GGUF 3.83%: **Δ −0.02 pt, CI [−0.13, +0.09]** |
| Target word, base Q8_0 | misses (recall ≤ 20%) | **0 / 10** ("Corvix" every time) |
| Target word, tuned Q8_0 | hits (recall ≥ 80%) | **10 / 10** |
| `hex model import`, base and tuned | each imports into the isolated `HEX_APPLICATION_SUPPORT_DIR`, and the imported copy is what was benchmarked | **ok**: each receipt's SHA-256 equals the converter output's and the scored copy's |
| Converter revision | conversion at `7045d1c`, quantizer built from it | **checked**: HEAD and the quantizer's build stamp both `7045d1c` |
| Tensor names | base and merged conversions identical | **equal**: 961 tensors, 219 changed |
| Speed | tuned within 10% of base | median RTF 0.0171 vs 0.0171 (**+0.3%**) |

`roundtrip.sh unified` ran all 16 steps in 22 minutes and exited 0. The base half (conversion,
import, scoring) runs before training, so the T0.1 control doesn't wait on it. Step times:
- training: 572 s;
- NeMo scoring: 220 s and 215 s;
- benchmarks: 69 s and 70 s;
- conversions: 17 s and 33 s.

An earlier run before the review fixes gave the same results within noise. Multithreaded CPU
training isn't bit-reproducible, so the tuned file's hash differs between runs.

On a truncated `.nemo` it exits 1 with `FAILED at step 'convert-base'` and the converter's own
error (`tarfile.ReadError: unexpected end of data`).

## T0.1: base conversion control

- **The converter reproduces the shipped model exactly.** Converting the untouched base `.nemo`
  and quantizing to Q8_0 gives a file whose **961 tensors are all byte-identical** to the catalog's
  `parakeet-unified-en-0.6b-Q8_0.gguf` (sha `4b50b6dd…`). The metadata differs only by
  `general.repo_url` and three capability flags written as false, which `artifacts.rs` also notes
  for Nemotron. So the GGUF side of every comparison is the shipped model, not an approximation of
  it.
- **NeMo and Hex's runtime agree within noise.** NeMo is scored with transcribe.cpp's reference runner
  (`scripts/wer/run_reference_parakeet_nemo.py`, offline attention, greedy). Both sides are counted
  with the benchmark's own normalizer and edit distance (`healing/wer.py`, a port checked against
  all 500 baseline clips).
- **`healing stats paired`** is the one paired bootstrap: 10,000 resamples of clips, seed 13.
  Its tests reproduce a hand-written loop over the same draws, and show it resamples clips, not words.

## T0.4: LoRA round trip

- **Training:** rank 16, α 32, on 219 linears: every encoder attention and feed-forward
  linear, plus the joint's 3. The predictor is an LSTM, which has no linear layers to wrap, so it
  stayed frozen. D-14 asked for it; D-31 records the change. 11.9 M trainable parameters. 200 steps of batch 4 on CPU,
  AdamW 5e-4, no dropout or SpecAugment, seed 13. Not bit-reproducible across runs, because CPU threads sum in a varying order. Loss 1.25 → 0.0001.
- **Data:** 20 training and 10 held-out sentences in 4 `say` voices, with no personal audio (G-3).
  The target word is **Qorvyx** (D-27). "Zorblax" was too easy: the base already wrote it correctly
  in 8 of 10 clips.
- **Merge:** W ← W + (α/r)·B·A, then the wrapper is removed, so `merged.nemo` has exactly the
  base's parameter names. The converter took it unchanged.
- **Cost to general accuracy** (tuned vs base GGUF, LibriSpeech 300): +0.24 pt, CI [−0.17, +0.64].
  That's not significant, for a run that memorized 20 clips. Real rounds mix in replay data (D-25).

### Does a small change survive Q8_0?

For each changed tensor:
- **the change's size** against the Q8_0 rounding step (block max / 127);
- **the norm ratio** ‖Q(merged) − Q(base)‖ / ‖ΔW‖, the measure the packet named;
- **the cosine** between the quantized change and the true change;
- **a random change's cosine**: a random change of the same size, quantized the same way.

A change much smaller than one step can only survive as a shift in rounding decisions. That makes
the norm ratio exceed 1 (median 1.87), because rounding noise dominates the norm. The cosine is
read against the random change's.

| Tensors | n | Median RMS(ΔW) / step | Cosine, median | Cosine, min | Random change's cosine, median |
|---|---|---|---|---|---|
| encoder attention | 120 | 0.24 | 0.56 | 0.34 | 0.54 |
| encoder feed-forward | 96 | 0.18 | 0.49 | 0.35 | 0.47 |
| joint | 3 | 0.96 | 0.92 | 0.92 | 0.91 |

Each weight moved only about a fifth of a rounding step. The quantized change points the way of
the real change (median cosine 0.53), but **a random change of the same size scores almost the
same (0.52)**. So the cosine shows only that rounding doesn't systematically erase a small change.
It is not evidence that this learned change in particular survived.

**The evidence is behavioural:** the Q8_0 file goes from 0/10 to 10/10 on the target word. A
careful real round will change the weights less than this spike did. That's why **every real
round's gate (T7.1) scores the Q8_0 file**, never the F32 or NeMo model.

## T0.2: the kept recording is the model's input

- **Same buffer on every OS.** macOS (`parakeet.rs:464`), Linux (`linux/dictation.rs:1303`) and
  Windows (`windows/dictation.rs:1377`) all keep the History recording from the same
  `samples` buffer they hand to the engine (`finish_with` / `finish`).
- **The engine passes it on unchanged.** New test
  `transcription::engine::tests::worker::the_offline_run_receives_the_finished_clip_unchanged`
  (macOS, real engine thread): the offline run receives the finished clip bit for bit, with no
  stream and after a failed one.
- **The WAV is exact.** New test `history::recordings::tests::kept_wav_round_trips_samples_to_i16`
  writes the file and reads every sample back as `round(clamp(x) · 32767)`, at 16 kHz, mono,
  16-bit. Removing `.round()` from the writer makes it fail (checked).
- **What the model sees beyond the recording** (found here, folded into T6.1):
  - On every OS, the transcriber appends zeros before running: up to the minimum duration, plus
    400 ms for Unified (`pad_gguf_samples`, `transcription.rs:172`). Training applies the same
    padding.
  - On Linux and Windows with Labs › Skip long pauses on (off by default), long pauses are cut
    first (`whole_clip`). The recording keeps them. Labels don't depend on silence.
- **Which text was used.** When a live stream finishes cleanly, the stream's text is used; the
  stream saw the same audio with long pauses capped (2.5 s + 0.3 s). Otherwise the whole clip
  is transcribed. Either way, the recording is the whole clip.
- **Skipped: the live smoke check** (`recording_fidelity.sh`). The packet itself calls it
  circular (both WAVs come from one variable), and running it means a second, menu-bar Hex on
  your Mac. The boundary test above is the proof the packet asks for. X-2 didn't trip: the only
  transforms are the appended zeros and the opt-in pause cut, both identified and reproducible.

## T0.3: general set and baseline

- **The set:** 300 LibriSpeech test-other plus 200 AMI IHM test utterances, seeded (13) and pinned:
  `manifest.sha256` `5d8139f9…`, with each WAV's sha in `clips.jsonl`. Sources are Hugging Face
  parquet shards pinned by revision and SHA-256 (D-28). AMI clips are 1–20 s with at least 3
  words after fillers are dropped (D-30).
- **Reproducible:** two builds, and a third into a fresh folder, all hash to `5d8139f9…`. A
  different seed or slice over an existing set is refused.
- **Baseline** (`healing/eval/baseline.json`), shipped Unified Q8_0 (`4b50b6dd…`):

  | | WER | Errors / words |
  |---|---|---|
  | all | **5.72%** | 447 / 7,818 |
  | LibriSpeech test-other | 3.59% | 193 / 5,376 |
  | AMI IHM | 10.40% | 254 / 2,442 |

  Median RTF 0.018. Re-running gives the same numbers.
- **The benchmark doesn't prepare audio exactly as dictation does.** It pads to the minimum
  duration but skips Unified's 400 ms of trailing silence. Parity is unaffected, because NeMo and
  the GGUF both see unpadded clips. T3.4 switches the benchmark to `pad_gguf_samples`, so later
  gates score audio the way dictation prepares it.

## Guardrail incident (G-8), fixed

The first version ran `hex benchmark-transcription` without an isolated data folder. `hex` opens
its process log before it dispatches any command, so each benchmark appended one "loaded
transcription model" line to your real `~/Library/Application Support/hex/logs/process.log`: about
a dozen lines on 2026-11-14 between 08:04 and 08:47 UTC. They held no audio and no transcript, and
nothing else in that folder was written.

Both reviewers caught it, Claude and Codex independently. The packet's G-8 check had only listed
`models/custom`, so it missed this.

**Fix:**
- Every `hex` call now runs with its own `HEX_APPLICATION_SUPPORT_DIR`: `$HEALING_HOME/hex-support`
  for `healing`, and `$SPIKE_DIR/hex` for the spike.
- `tests/test_bench_isolation.py` checks this.
- The G-8 checks now compare the real folder's listing and `process.log`'s mtime before and after
  a run. Both were unchanged across the final `pytest`, baseline and round trip.

## Environment notes

- **Local SDK stub.** dev needs a newer macOS SDK than this Mac's Xcode 26.1.1
  (`SystemLanguageModel.contextSize`). Rust tests here ran with an uncommitted one-line stub in
  `src/apple_intelligence.swift`, which isn't part of any commit.
- **NeMo environment.** NeMo comes from transcribe.cpp's pinned Parakeet environment (D-29):
  NeMo 2.8.0rc0, torch 2.11.
- **Disk.** One `roundtrip.sh` run uses about 10 GB in `SPIKE_DIR`. The base `.nemo` files are
  cached in `MODEL_CACHE` (2.4 GB each).
