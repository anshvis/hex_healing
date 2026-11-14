# M0 report: feasibility spikes and baseline

Measured on an Apple M4 (16 GB, macOS 15.6) on 2026-11-14, with the installed Hex 3.4.0 as the
runtime (`/Applications/Hex.app/Contents/MacOS/hex`, Metal). Raw runs stay under
`~/hex_healing_work/rt-unified/`; the verdict is committed as
[`results/unified-verdict.json`](results/unified-verdict.json).

## Result

**Architecture: Parakeet Unified English 0.6B** (`nvidia/parakeet-unified-en-0.6b` at
`fe53cd8`), Hex's default model. It passed the base control (T0.1) and the LoRA round trip
(T0.4) on the first try, so TDT v2 wasn't needed. `roundtrip.sh tdt-v2` exists for when it is.

| Check | Bar | Result |
|---|---|---|
| Converter | transcribe.cpp `7045d1c` | the converter and `transcribe-quantize` accept both checkpoints, with no patch |
| Base control (T0.1) | paired 95% CI of WER(GGUF Q8_0) − WER(NeMo) ⊂ ±1.0 pt, 300 clips | NeMo 3.53%, GGUF 3.59%: **Δ +0.06 pt, CI [−0.12, +0.30]** |
| Tuned parity (T0.4) | same bar, merged NeMo vs tuned Q8_0 | NeMo 3.83%, GGUF 3.85%: **Δ +0.02 pt, CI [−0.04, +0.08]** |
| Target word, base Q8_0 | misses (recall ≤ 20%) | **0 / 10** ("Corvix" every time) |
| Target word, tuned Q8_0 | hits (recall ≥ 80%) | **10 / 10** |
| `hex model import` | `import: ok` into an isolated `HEX_APPLICATION_SUPPORT_DIR` | **ok**: `custom-…` id, receipt and manifest written there only |
| Tensor names | base and merged conversions identical | **equal**: 961 tensors, 219 changed |
| Speed | tuned within 10% of base | median RTF 0.0174 vs 0.0170 (**+2.2%**) |

`roundtrip.sh unified` ran all 15 steps in 20 minutes and exited 0. Step times:
- training: 572 s;
- NeMo scoring: 223 s and 216 s;
- benchmarks: 68 s and 69 s;
- conversions: 21 s and 18 s.

On a truncated `.nemo` it exits 1 and names the step that failed (`train`, the first step to read
the checkpoint), with that step's log tail.

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
  stayed frozen (D-14 asked for it). 11.9 M trainable parameters. 200 steps of batch 4 on CPU,
  AdamW 5e-4, deterministic: no dropout or SpecAugment, seed 13. Loss 1.25 → 0.0001.
- **Data:** 20 training and 10 held-out sentences in 4 `say` voices, with no personal audio (G-3).
  The target word is **Qorvyx** (D-27). "Zorblax" was too easy: the base already wrote it correctly
  in 8 of 10 clips.
- **Merge:** W ← W + (α/r)·B·A, then the wrapper is removed, so `merged.nemo` has exactly the
  base's parameter names. The converter took it unchanged.
- **Cost to general accuracy** (tuned vs base GGUF, LibriSpeech 300): +0.26 pt, CI [−0.14, +0.65].
  That's not significant, for a run that memorized 20 clips. Real rounds mix in replay data (D-25).

### Does a small change survive Q8_0?

For each changed tensor: the change's size against the Q8_0 rounding step (block max / 127), and
the cosine between the quantized change and the true change. A change much smaller than one step
can only survive as a shift in rounding decisions, so the cosine, not the norm ratio, measures
whether its direction survived.

| Tensors | n | Median RMS(ΔW) / step | Cosine, median | Cosine, min |
|---|---|---|---|---|
| encoder attention | 120 | 0.25 | 0.58 | 0.34 |
| encoder feed-forward | 96 | 0.18 | 0.49 | 0.35 |
| joint | 3 | 0.96 | 0.92 | 0.91 |

Each weight moved only about a fifth of a rounding step, yet the quantized change still points
the way of the real change (cosine about 0.5), and the effect survives completely: 0/10 → 10/10.
This round's change is larger than a careful real round's will be. So **every real round's gate
(T7.1) scores the Q8_0 file**, never the F32 or NeMo model.

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

## Environment notes

- **Local SDK stub.** dev needs a newer macOS SDK than this Mac's Xcode 26.1.1
  (`SystemLanguageModel.contextSize`). Rust tests here ran with an uncommitted one-line stub in
  `src/apple_intelligence.swift`, which isn't part of any commit.
- **NeMo environment.** NeMo comes from transcribe.cpp's pinned Parakeet environment (D-29):
  NeMo 2.8.0rc0, torch 2.11.
- **Disk.** One `roundtrip.sh` run uses about 10 GB in `SPIKE_DIR`. The base `.nemo` files are
  cached in `MODEL_CACHE` (2.4 GB each).
