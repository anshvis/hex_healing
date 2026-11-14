# M2 - Correction capture

**Covers:** AC-5, AC-6 · X-3
**Depends on:** M1
**Surface:** full-stack. A new Labs card is drawn in the window; the paste worker, History and the CLI change.

Hex pastes into other apps and never sees what you change afterwards. M2 adds Labs › Learn from corrections (macOS, off by default, D-16). It works like this:
1. Hex keeps the focused AX element it pasted into, along with the exact text it inserted.
2. For up to 30 s it reads that element on value change, focus-out or Return (D-8). It keeps the last read that still contains the text, so a chat app's send doesn't erase the edit.
3. It aligns the inserted text against that read and stores what the span became on the History entry.

History stays the only place Hex keeps text (G-4).

Classification (`acoustic` or `rewrite`) happens off-device in `healing`. That keeps the Rust side small: it only observes. The rules can then be tuned without shipping Hex.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T2.5](T2.5-privacy-and-policy.md) | Privacy review, AGENTS.md invariant, `DisableCorrections` policy, feature map | standard | none |
| [T2.1](T2.1-lab-and-paste-target.md) | The `learn-from-corrections` lab and the paste-target handoff (pid, element, entry id, inserted text) | standard | T2.5 |
| [T2.2](T2.2-post-paste-reread.md) | Watch the element for up to 30 s (value change, focus-out, Return) and align the span | standard | T2.1 |
| [T2.3](T2.3-correction-on-history.md) | Store `correction` on the History entry and expose it in `hex history --json --raw` | standard | T2.2 |
| [T2.4](T2.4-classify-corrections.md) | `healing label corrections` and the seeded edit-recall protocol (X-3) | standard | T2.3, M1 |

## Changes
- **Rust:**
  - `src/labs*` and the lab plumbing;
  - `src/rune_ui/labs.rs` (the card) and the goldens;
  - `src/parakeet.rs` (the paste completion point, `:1096-1128`);
  - `src/accessibility.rs` (retain and re-read);
  - a new `src/dictation/correction_watch.rs`;
  - `src/history.rs` (an optional `correction` field);
  - `src/control/transcript_cli.rs`;
  - docs: `docs/features/labs.md`, a new `docs/features/corrections.md`, and the feature map index.
- **Python:** `healing/healing/corrections.py`, `phonetic.py`.
- **Gold-stars:**
  - the `CursorContext` lab (`cursor-context`). It is the newest macOS-only lab that reads another app's text: `src/labs/feature.rs:89`, `src/labs.rs:178-1091`, `src/labs/gates.rs:68`.
  - `text_around_caret` (`src/accessibility.rs:257`): bounded ranged reads, secure-field skip.

## How you validate this milestone
1. Turn on Labs › Learn from corrections. Dictate "send it to Kip" into TextEdit, Notes, Slack and a Chrome textarea.
2. Change "Kip" to "Kit" within 30 s. In one of them, also rewrite a different word completely.
3. Run `hex history --json --raw --limit 4`. Each entry has `correction.outcome: "edited"` and the edited span.
4. Run `healing collect && healing label corrections`. "Kip→Kit" is `acoustic`; the rewrite is `rewrite`.
5. With the lab off, `rg` on `process.log` shows no AX re-read at all.

## Exit gate
- Every T2.x check is green.
- The goldens are regenerated.
- T2.4's seeded protocol captures at least half of the seeded edits across the matrix apps, with a false-edit rate of 5% or less. Below that, X-3 applies: replan toward an explicit "Correct this" in History.
- The privacy review (T2.5) is written, and AGENTS.md's History invariant is amended in the same PR as T2.3.

## Rollout and rollback
- It is a Lab, off by default, macOS only.
- Turning it off stops all reads immediately: the gate sits where the watch is scheduled.
- Stored `correction` fields expire with their History entries.
