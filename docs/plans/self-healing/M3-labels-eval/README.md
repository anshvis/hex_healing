# M3 - Labels, holdouts, evaluation

**Covers:** AC-7, AC-8, AC-9 · G-10
**Depends on:** M2
**Surface:** backend. `healing`, plus one Rust flag on the benchmark (T3.4).

Turn the collected clips into weighted labels and three leak-free test sets, and give one command that scores any GGUF on all three through Hex's own benchmark runner.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T3.1](T3.1-labels.md) | Labels and weights: strong, verified, weak; random-sample review; model provenance | standard | none |
| [T3.2](T3.2-splits.md) | Leak-free splits: fresh random-verified rolling holdout, frozen golden, general, edited-span diagnostic | standard | T3.1 |
| [T3.4](T3.4-benchmark-vocabulary.md) | Rust: `hex benchmark-transcription --vocabulary` (biases like dictation does) | standard | none |
| [T3.3](T3.3-eval.md) | `healing eval <gguf>`: WER, formatting, term and edited-span error on each set, with per-clip records | standard | T3.2, T3.4 |

## How you validate this milestone
1. Run `healing label all`, then `healing status`. It shows counts of strong, verified, weak and unlabelled clips.
2. Run `healing split build`.
   - `healing split verify` reports that no cluster appears in both train and any test set.
      - The rolling set holds only random-sampled, verified clips from sessions after the cut-off (the last 14 days, sliding, D-40; no cluster near-duplicates the general set either).
3. Run `healing golden review` and confirm about 20 clips. This is the start; the set freezes at 200, which you reach over time.
4. Run `healing review --strategy random --since-cutoff --n 50`. This is about 10 minutes, and it fills the rolling set. Run `healing split build` again to pick the new reviews up.
5. Run `healing eval --model <current gguf>` (add `--vocabulary none` until your `hex` has T3.4's `--vocabulary`). One table for rolling, golden (once frozen), general and the edited-span diagnostic: WER, formatting error rate, term error rate, and edited-span error rate. The edited row has no WER or formatting error (n/a): strong clips are scored only at the words you corrected. `healing eval baseline` still works and is the same eval on the general set.

## Exit gate
- All T3.x checks green.
- `split verify` passes on your real dataset.
- The eval table matches T0.3's baseline on the general set, within run noise of 0.2 pt.

## Rollout and rollback
Local files only. `healing split build` is re-runnable. The golden set, once frozen, is pinned by hash in the registry (M5) and refuses changes (G-10).
