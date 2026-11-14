# M6 - Tier 2: LoRA training

**Covers:** AC-12 · G-3 · X-6
**Depends on:** M0 (architecture and converter), M3 (splits), M5 (registry)
**Surface:** backend.

This is the recurring fine-tune, with LoRA on Parakeet (D-14) in NVIDIA NeMo, trained on your Mac overnight, with Kaggle's free GPUs as an optional fallback (D-34). Nothing here can charge you:
- It is triggered only by enough new strong data: at least 200 minutes of new strong speech, with no time floor (D-33).
- The output is merged, converted and quantized with the exact T0.4 path, then registered as a candidate.
- M7 decides whether it ships.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T6.1](T6.1-trainer.md) | Training data: weighted, cluster-balanced manifest; 30 s cap; self-distilled replay | standard | none |
| [T6.5](T6.5-training-loop.md) | NeMo LoRA training loop (encoder and joint, D-31); full local CPU round, resumable | standard | T6.1 |
| [T6.2](T6.2-remote-runner.md) | Free cloud fallback (Kaggle) with explicit upload opt-in, log scrubbing and cleanup | standard | T6.5 |
| [T6.3](T6.3-export.md) | Export: merge → GGUF → Q8_0 → register candidate | standard | T6.5 |
| [T6.4](T6.4-trigger.md) | `healing train auto`: the trigger (≥200 min of new strong speech) | standard | T6.3 |

## How you validate this milestone
1. Run `healing train --local --smoke`. It trains 20 steps on CPU over 50 clips and exports a candidate GGUF that Hex imports.
2. Run `healing train --local` on real data overnight. It finishes a round, survives a sleep or Ctrl-C by resuming, and registers a candidate with scores.
3. Optional: run `healing train --remote --allow-upload` (needs your free Kaggle token). It finishes a run, deletes the Kaggle dataset and notebook, and registers a candidate.
4. Run `healing train auto` with 150 minutes of new strong speech. It prints "waiting: 150/200 min". With 200 minutes or more, it starts.

## Exit gate
- All checks green.
- One real local round produced a registered candidate.
- X-6: no Kaggle token → T6.2's live check is **parked** (it's yours to run). Everything else merges, because the default path is local.

## Rollout and rollback
Training never touches Hex. A candidate does nothing until M7 promotes it.
