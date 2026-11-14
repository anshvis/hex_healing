# M8 - Tidy after paste: suggested cleanup of fillers and self-corrections

**Covers:** AC-15, AC-16, AC-17
**Depends on:** M2 (the kept AX element and the inserted text), M1 (the dataset). It follows M7 and doesn't block any earlier milestone.
**Surface:** full-stack. A Labs card, a review capsule drawn by the window's compositor (goldens), Accessibility edits, plus `healing`.

## What it is
After a dictation pastes, you can ask Hex to **tidy** it. Hex suggests edits:
- **Fillers:** "um", "uh", "erm", and your own habits ("like", "you know", "basically") when you tend to accept removing them.
- **Spoken self-corrections:** "send it Tuesday, scratch that, Wednesday" → "send it Wednesday". "No wait" and "I mean" work the same way.
- **Stutters and repeats:** "the the", "I I think".
- **Misheard words** the model can fix from context. This is optional and comes from the same pass.

You see each edit as a strike-through or replacement, then accept or skip each one (or all at once). Hex applies only what you accepted, to the text it pasted, through Accessibility.

This is not automatic (D-37):
- The paste happens first, exactly as today, with nothing added to its latency.
- Tidy is an action you take afterwards, from a shortcut, the HUD or the menu bar.
- Nothing changes your text without you seeing it.

It differs from Labs › Clean up with Apple Intelligence (`src/dictation/apple_cleanup.rs`). That lab rewrites before the paste and shows nothing. Tidy reuses its on-device model call and guard, but runs after the paste and only suggests.

## How it fits model training (D-38)
- **The speech model keeps learning what was said, not the tidied text.** Tidy edits are style choices: you said "um" and didn't want it written. Training on tidied text would teach the model to drop words it heard, which hurts formatting and hides real misrecognitions.
- **Tidy edits are not corrections.** M2's correction capture compares the field against what Hex pasted. A tidy edit would look like a correction, so M2 treats the text after tidy as the new baseline, and records the tidy separately.
- **What tidy data does teach:** which fillers and habits you accept removing (T8.2), so suggestions get personal. That stays in the suggester, never in the speech model.
- **One exception, and it's still a correction:** a misheard word you fix by accepting a replacement suggestion is a real recognition error. It is recorded as a weak correction label (M2's rules apply: confirmed only if the replacement survives).

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T8.1](T8.1-suggester.md) | Suggestion engine: rules plus an on-device model pass, guarded to deletions and small replacements | standard | none |
| [T8.2](T8.2-personal-habits.md) | Personal habits: learn which fillers and tics you accept removing | standard | T8.1 |
| [T8.3](T8.3-review-capsule.md) | Labs › Tidy after paste: trigger and the review capsule | deep | T8.1 |
| [T8.4](T8.4-apply-ax.md) | Apply accepted edits to the pasted text through Accessibility | deep | T8.3 |
| [T8.5](T8.5-training-boundary.md) | Training boundary: tidy events, M2 baseline, and labels | standard | T8.4 |

T8.2 can run beside T8.3.

## How you validate this milestone
1. Turn on Labs › Tidy after paste.
2. Dictate "um so I think we should, uh, send it Tuesday, scratch that, Wednesday" into TextEdit, then press the Tidy shortcut. The capsule shows three removals and the self-correction. Accept all, and TextEdit reads "So I think we should send it Wednesday." ⌘Z undoes it in one step.
3. Dictate into Slack and Notes, and accept only some suggestions. Only those change.
4. Type in the field after the paste, then press Tidy. Hex either applies to the unchanged pasted range or says the text moved, and never edits what you typed.
5. Run `healing status`. It shows the tidy events. The clips' `raw_text` labels are unchanged by tidy.
6. After a week of use, a filler you always skip stops being suggested, and one you always accept is suggested first.

## Exit gate
- Every T8.x check is green at the head SHA.
- The goldens for the capsule are regenerated on Linux (AGENTS.md).
- The validation above passes on your Mac in TextEdit, Notes, Slack and a Chrome textarea.

## Rollout and rollback
- Labs, off by default (G-5).
- Turning it off removes the shortcut and the capsule. Recorded tidy events stay in the dataset until you delete them.
