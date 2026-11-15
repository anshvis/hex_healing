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
cleared by send still yields the edit: for a second after a Return (Cmd+Return
included) the field going blank, shorter, or holding other text ends the watch
with the read kept at the Return, because the app clears the field a moment
after the listener counts the Return. A field that goes blank with no Return
(a send button, or select all and delete, which cannot be told apart) is
`cleared_before_read`: a send button without a Return loses the observation,
which is the training-safe side. Outside that second, a read that finds other,
non-empty text drops the kept read. A field with no string value
(a web view, a terminal) is `read_failed`, never an empty one, and a field over
8,192 characters is read around the caret and can show the paste whole or not
at all, never an edit of it. One gate (the lab with its managed lock,
`labs::learn_from_corrections`) is checked before every read and before a
queued target starts, and Hex's own pid is never observed. The watch clock
starts when the paste completes, not after History's write. Outcome IDs:
`unchanged`, `edited`, and the lost reasons `element_gone`, `secure`,
`too_large`, `span_not_found`, `read_timeout`, `read_failed`,
`cleared_before_read`; triggers: `start`, `value_changed`, `focus_out`,
`return`, `timeout`. The log line carries those IDs, a character count and
milliseconds only.

Where to look (`locate.rs`). Before the paste, Hex reads the field's caret
(`AXSelectedTextRange`, `AXNumberOfCharacters`), so it knows the paste's offset
and, with the selection it replaces, how long the field is with it in. That is
the anchor.

- Anchored. The anchor is trusted once a read shows the paste whole at that
  offset (a read before that only reports the paste unchanged, never an edit:
  with no position to trust there is no evidence of one). A paste found in the
  field but not at the offset means the app counts differently, so the watch
  falls back to the strict rules below. From then on only the field's region at
  the anchor is judged, so text anywhere else in the field can never match, and
  position is evidence enough to call a one-word replacement an edit ("Hanks" to
  "Hex", "Yes" to "No", two words for two, a trailing word for three). The region
  moves with the field's length: an edit before the paste shifts it and an edit
  inside stretches it, both are tried, and the one that holds more of the paste
  wins (a tie goes to the smaller region). Intact at the anchor is unchanged
  whatever else the field says; the same text elsewhere does not count. An empty
  region is the paste deleted, which is `span_not_found` (nothing to learn
  from). Offsets are UTF-16 units, as Accessibility counts. A field read as a
  window (over 8,192 characters) is judged only if the window holds the region.
- Strict, with no anchor (the app reports no selection range) or one that did
  not hold. A token-level local alignment of the pasted text against the whole
  field: a one-letter-off word of three or more letters is a near match, so
  "Kip" to "Kit" aligns in a longer text. The aligned core must cover half the
  pasted tokens (all of them below three) on its own, counting only exact and
  near pairs, with at least two exact words in a text of three or more, and at
  most one pasted token may be left over at either end; so an unrelated field
  that shares a word or a phrase ("go to bed" against "send to Kip", "to be
  honest the plan" against "to be honest I agree") is not found. A replaced
  first or last word that is not near scores below zero and sits outside the
  core, so the span extends one field token over it, never across sentence
  punctuation or a line break. Whole words only: "Yes" is not found in
  "Yesterday", and a one-word paste is an edit only when the field is that one
  word. It cannot tell a text pasted twice apart while one copy is intact.

Precision, measured on a corpus of 36 unrelated chat sentences, every ordered
pair (1,260): the strict rules store 14 as edits (1.1%); anchored, none, and an
edit at the anchor never carries the neighbouring text (`locate::tests`,
`the_unrelated_pair_corpus`). Text without whitespace (CJK) is one token, so
it is effectively unsupported. A paste over 4,000,000 alignment cells (inserted
tokens times field tokens) is not aligned.

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
