# M4 - Tier 1: vocabulary mining

**Covers:** AC-10
**Depends on:** M2, M3
**Parallel with:** M5
**Surface:** backend. One Rust command (`hex vocabulary set --file`, T4.4), plus `healing`. The benchmark's `--vocabulary` flag lands in M3 (T3.4).

The brief's best tactic, applied right away with no training:
- When you correct a word the same way twice, it becomes a vocabulary term with that sounds-like spelling.
- Only the first 64 terms in list order are biased (`src/dictation/vocabulary.rs:26,290`). So mined terms are ranked by recent frequency, and they fill the slots after your own terms (D-20).
- The gain is proven with the benchmark, not assumed, through T3.4's `--vocabulary` flag. Today the benchmark never passes vocabulary to the model (`benchmark.rs:89`).
- Mode vocabularies take bias slots before the global list, so slot counts are reported per mode.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T4.1](T4.1-mine.md) | `healing vocab mine`: pairs seen twice → candidate terms | standard | none |
| [T4.2](T4.2-apply.md) | `healing vocab apply`: ranked per mode, your terms untouched, written atomically | standard | T4.1, T4.4 |
| [T4.4](T4.4-vocabulary-set.md) | Rust: `hex vocabulary set --file`, an atomic ordered write | standard | none |
| [T4.3](T4.3-ab.md) | `healing vocab ab`: the Tier 1 A/B on the benchmark (uses T3.4's flag) | standard | T4.1 |

T4.1 and T4.4 can run in parallel.

## How you validate this milestone
1. Run `healing vocab mine`. It lists candidates such as `Kit ← kip ×3`, with counts and last-seen dates.
2. Run `healing vocab apply --dry-run`. It shows the final order: your terms first, then the mined ones by rank.
3. Run `healing vocab apply`, then `hex vocabulary list`. The list matches the dry run.
4. Run `healing vocab ab`. It shows rolling term error rate with and without the mined list, and general WER with and without (the false-replacement guard).

## Exit gate
- All checks green.
- The A/B shows mined vocabulary doesn't raise general WER by more than 0.3 pt and false replacements stay at or under 0.5% (the existing `transcription-quality.md` gate).
- If it does, apply ships with `--boost light` as the default, and the A/B result is logged.

## Rollout and rollback
- `healing vocab apply` records what it added.
- `healing vocab revert` removes exactly those terms.
- Labs › Vocabulary biasing must be on for terms to take effect. `healing status` warns when it's off.
