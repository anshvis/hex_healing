# M0 - Feasibility spikes and baseline

**Covers:** AC-1, AC-2 · X-1, X-2
**Depends on:** none
**Surface:** backend. Scripts, a Rust test and benchmark runs; nothing the window draws.

The brief names two things to check before building anything:
- whether the converter accepts a fine-tuned checkpoint;
- whether Hex keeps audio files.

M0 answers both with runnable evidence, and records the baseline scores every later gate compares
against. If M0 fails, Tier 2 stops before any capture work is spent on it (X-1).

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T0.3](T0.3-baseline-eval.md) | General regression set and baseline scores for the shipped model | standard | none |
| [T0.2](T0.2-recording-fidelity.md) | Prove the kept recording is the model's input | standard | none |
| [T0.1](T0.1-base-conversion-control.md) | Base conversion control and the paired-bootstrap parity harness | standard | T0.3 |
| [T0.4](T0.4-lora-round-trip.md) | LoRA fine-tune → merge → GGUF → Q8_0 → import round trip | standard | T0.1, T0.3 |

T0.2 and T0.3 run in parallel. T0.1 and then T0.4 follow T0.3, because they reuse its LibriSpeech slice for parity.

## Changes
- **Rust:** one test in `src/history/recordings.rs` or `src/parakeet.rs` (T0.2). No behaviour change.
- **New:** `healing/` is scaffolded as a uv project, with `healing/spikes/m0/` scripts (T0.1, T0.4), `healing/healing/stats.py` (the paired bootstrap, T0.1) and `healing/healing/sets/general.py` (T0.3).
- **Gold-stars:**
  - `src/transcription/artifacts.rs:42-50`: Orukeet, a third-party fine-tune already shipped as a Q8_0 GGUF through transcribe.cpp's converter.
  - `docs/research/transcription-benchmark.md`: the manifest format.

## How you validate this milestone
1. Read `healing/spikes/m0/REPORT.md`. It shows:
   - the chosen architecture;
   - the converter rev;
      - the parity CIs: base NeMo vs base GGUF (T0.1) and tuned NeMo vs tuned GGUF (T0.4), on 300 clips;
      - the per-tensor ‖ΔW‖ against the Q8_0 rounding step;
   - the base output versus the fine-tuned output on the target word;
   - the `hex model import` output.
2. Run `scripts/check history::recordings` (or the named test). It passes.
3. Read `healing/eval/baseline.json`. It holds the shipped model's WER on the general set, with the command that produced it.

## Exit gate
Every check in T0.1–T0.4 passes at the head SHA, and `REPORT.md` names an architecture that passed both the base control (T0.1) and the LoRA round trip (T0.4).
No architecture passed → X-1: Tier 2 is replanned and M1 still proceeds.

## Rollout and rollback
No user-facing change. The spikes write only to temp dirs, an isolated `HEX_APPLICATION_SUPPORT_DIR`, and `healing/`.
