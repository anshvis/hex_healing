# Learn from corrections

Labs › Learn from corrections (macOS, off by default; `labs.learn_from_corrections`)
notices when you edit a dictation right after Hex pasted it, and keeps the
edited span on that History entry so Hex can learn from it. This file is the
privacy contract for that read. Status: implemented (the lab, the watcher, the
History field, `hex history --json --raw`, the managed lock
`DisableCorrections`). Proof so far is unit tests and a Rune golden of the Labs
card; no live Accessibility run (no TextEdit or chat-app session) has been
observed, and no seeded protocol run has been scored yet.

```ts
Dictation pasted (Continue mode), History entry recorded    // corrections.watch
├── Lab off, or DisableCorrections / DisableLabs forced -> nothing is read, nothing queued
├── History keeps nothing (retention Off), or no History -> nothing is read
├── Lab on -> Hex reads the focused element before the paste, then watches it for up to 30 s
│   ├── Reads -> the Accessibility value of that one element, in memory:
│   │   the whole value up to 8,192 characters, else 8,192 characters around the caret
│   ├── Closes at 30 s, the app losing focus or its focus moving, or a Return that
│   │   emptied or shortened the field (a send); Return that only adds a newline keeps watching
│   └── At most 4 targets are watched at once; extras wait in a bounded queue (8, oldest dropped)
├── Never read -> secure (password) fields, Hex's own window, any other element
├── Kept -> the edited span only, on the History entry, at most
│   2 x the pasted length + 64 characters; the rest of the field is dropped
│   └── Same retention, entry and byte caps, Delete and Clear all as the entry
└── Never logged -> any text; logs carry counts, durations and outcome IDs only
```

Not watched: a Send dictation (YOLO mode's Return), copy only, and a standalone
paste, because the field is left, never pasted into, or has no continuation.

Watch rules (`dictation/correction_watch`, constants in `session.rs` and
`worker.rs`): the watcher ticks every 50 ms. The first tick reads the field as
a baseline (the paste's own change predates the observer, so without it a
paste sent unedited would read as lost). A tick that saw a value change reads
once, on that tick (`READ_GAP` is 40 ms, under the tick), so the last read
before a chat app clears the field is at most one tick behind the typing, not a
keystroke behind; a burst inside one tick is one read of where the field
ended. A field that refuses the value-change registration is read every 400 ms
instead. A read that finds the field as it was before costs nothing; reads
that found it changed are capped at 200 per target, plus one closing read. The
closing read comes from: focus moving inside the app or the app losing focus
(ignored in the first 300 ms, which are the paste's own), a Return the
shortcut listener saw after the paste (`dictation/key_counts`, a count the
listener bumps, as for YOLO mode's Escape; no key-state poll, no event tap of
its own) that left the field empty or shorter, or 30 s. The kept read is the
last one in which the field still held the pasted text, so a chat field
cleared by send still yields the edit; a read that finds other, non-empty text
drops it. One gate (the lab with its managed lock,
`labs::learn_from_corrections`) is checked before every read and before a
queued target starts, and Hex's own pid is never observed. The watch clock
starts when the paste completes, not after History's write. Outcome IDs:
`unchanged`, `edited`, and the lost reasons `element_gone`, `secure`,
`too_large`, `span_not_found`, `read_timeout`, `read_failed`,
`cleared_before_read`; triggers: `start`, `value_changed`, `focus_out`,
`return`, `timeout`. The log line carries those IDs, a character count and
milliseconds only.

Alignment (`locate.rs`) is a token-level local alignment of the pasted text
against the field. A one-letter-off word of three or more letters is a near
match, so "Kip" to "Kit" aligns even alone. The aligned core must cover half
the pasted tokens (all of them below three) on its own, so an unrelated field
that shares a word or two ("go to bed" against "send to Kip") is not found.
Given such a core, a replaced first or last word that is not near ("Hanks" to
"Hex") scores below zero and sits outside it, so the span extends over at most
one field token at each end that has a pasted token left over, and never across
sentence punctuation or a line break; one pasted word replaced by one word is
the whole field. It cannot tell a text pasted twice apart while one copy is
intact.

Storage (`history/corrections.rs`): the watcher thread calls
`History::observe_correction` on the History carried by the paste target, which
writes through History's own lock and atomic save, then prunes to History's hard
caps. The entry gets `correction`
`{outcome: unchanged|edited|lost, reason?, observed_at_ms, edited_text?}`; the
first observation wins, an evicted or cleared entry takes nothing, and a Replace
or Restore drops it (it was seen against the text the entry no longer holds). `edited_text` is kept only for
`edited`, at most 16 KiB, and counts toward History's 16 MiB text cap. `hex
history --json --raw` prints it (never without `--raw`;
[transcript-cli](../transcript-cli.md)). An older Hex drops the field when it
saves. If the watcher thread dies it is started again at the next paste.

Each bound above must match the constants; a change to either updates this file
in the same commit. A managed lock is listed under
[Managed feature locks](README.md#managed-feature-locks-macos).
