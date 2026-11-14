# M1 - Long-term capture

**Covers:** AC-3, AC-4 · edge cases: Hex expiring entries, Replace versions, the `model` of the raw text
**Depends on:** M0
**Parallel with:** none
**Surface:** backend. A `hex history --json` field addition plus the new `healing` collector. Nothing the window draws.

Every dictation needs to end up in a dataset that outlives Hex's History, which keeps 7 days by default and at most 10,000 entries. Storage must not grow without bound.

The answer is:
- content-addressed FLAC files;
- a SQLite index;
- near-duplicate clustering that keeps audio for only k clips per cluster;
- a byte budget that evicts the most redundant, oldest weak clips first.

Strong labels, verified clips and golden clips are never evicted.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T1.1](T1.1-history-json-fields.md) | `hex history --json` reports `model`, `language`, `audio_ms` | standard | none |
| [T1.2](T1.2-collector.md) | `healing collect`: idempotent copy into the dataset | standard | T1.1 |
| [T1.3](T1.3-storage-budget.md) | Near-duplicate clusters, k audio per cluster, byte budget | standard | T1.2 |
| [T1.4](T1.4-schedule-and-status.md) | launchd schedule and `healing status` (gaps, budget, lab state) | standard | T1.2 |
| [T1.5](T1.5-clean.md) | `healing clean`: index matches disk, free-space floor, leftovers swept | standard | T1.3 |

T1.3 and T1.4 can run in parallel after T1.2. T1.5 follows T1.3.

## Changes
- **Rust (T1.1):**
  - `src/control/transcript_cli.rs` row and its projection;
  - `docs/transcript-cli.md`.

  The change is additive, so the JSON `version` stays 1, per `docs/transcript-cli.md`'s compatibility rule.
- **New `healing/` modules:**
  - `store.py` (SQLite and FLAC);
  - `collect.py`;
  - `dedupe.py`;
  - `budget.py`;
  - `clean.py`;
  - `status.py`;
  - `launchd/com.hex-healing.collect.plist` (a template).
- **Data root:** `$HEALING_HOME`, default `~/Library/Application Support/hex-healing/` (D-4).
- **Gold-stars:**
  - `src/control/transcript_cli.rs:168-185,416-428`: the row shape and read-only snapshot.
  - `src/history/recordings.rs:64-117`: owner-only temp-fsync-rename writes, mirrored in Python.

## How you validate this milestone
1. Turn on Labs › Keep recordings. Dictate about 10 times, including the same phrase three times.
2. Run `healing collect` twice. The second run reports `0 new`.
3. Run `healing status`. It shows the count, the bytes against the budget, and the cluster of 3.
4. Delete an entry in Hex's History, then run `healing collect`. The entry is still in `healing status --list`.
5. Run `healing prune --budget 1MiB --dry-run`. It names only weak, redundant clips for eviction.
6. Copy any FLAC into `$HEALING_HOME/audio/` by hand, then run `healing clean --deep`. It moves that file to quarantine and reports 0 mismatches. A second run reports nothing.

## Exit gate
Every T1.x check is green at the head SHA. The validation above passes on your Mac with a real dictation.

## Rollout and rollback
- The collector is opt-in: you install the launchd job with `healing schedule install`.
- Rollback is `healing schedule remove`.
- The data stays in `$HEALING_HOME` until you delete it.
- The Hex change is additive JSON.
