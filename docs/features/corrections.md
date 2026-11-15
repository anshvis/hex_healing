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
included) the field going blank, shorter, or holding text that is not the paste
ends the watch with the read kept at the Return (the paste still there, changed
or not, is an ordinary read: a notes app's newline, then a fix), because the app clears the field a moment
after the listener counts the Return. A field that goes blank with no Return
for two ticks (100 ms, so a Return the listener counts a moment late converts
it) is `cleared_before_read`: a send button, or select all and delete, which
cannot be told apart, loses the observation, which is the training-safe side. Outside that second, a read that finds other,
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

**Invariant: fail closed.** Hex stores only text inside the paste's own region
in the current field (what now occupies the place the pasted text occupied,
plus a token that touches its edge, for a replaced edge word) and never text
from before or after it. When unsure the answer is "not found" or "unchanged".
No edit is stored before the paste has been seen whole in the field.
`property_tests.rs` checks it over 4,000 generated cases: the paste in a field
between corpus sentences (some sharing its words, some empty, some with
attachments, emoji or CJK), put through edits of one word, two, a trailing two
words, replacement and deletion of the paste, text typed before and after it, a
second dictation, a Return and a newline then a fix, and a cleared field. The
stored span never leaves the region (plus one token), a deleted paste stores
nothing, and the one-word, two-word and trailing replacements are stored
exactly. Before the anchor rework the same run stored text outside the region
in 448 cases and from a deleted paste in 489.

Where to look (`locate.rs`). Before the paste, Hex reads the caret
(`AXSelectedTextRange`'s location, in UTF-16 units, which is where a paste
starts); with the paste's length that is the anchor.

- Anchored. The anchor is trusted once a read shows the paste whole at that
  offset (until then a read only reports the paste unchanged or not found,
  never an edit). A paste found in the field but not at the offset is waited on
  for half a second (the read may have come before the app applied the paste),
  then the anchor is given up and the strict rules below apply. Once landed,
  the paste's region is followed through every read by diffing the field
  against the read before: text inserted or removed before the region moves it,
  a change inside stretches or shrinks it, one after it leaves it (so text typed
  after an edit is never part of it), an insertion at its start counts as
  before and at its end as after, and a change that crosses an edge, or that
  could as well have been made on either side of one (text typed before the
  paste that begins with the paste's first letters), loses it. Only the field's
  text in the region is judged. Intact in the region is unchanged, whatever
  else the field says, and the same text elsewhere does not count. For a paste
  of one or two words, position is evidence: whatever now sits in the region
  is what it became ("Hanks" to "Hex", "Yes" to "No", two words for two, a
  trailing word for three). A longer paste must still resemble itself: half its
  words (exact or near) in order, two of them exact. An empty region is the paste
  deleted, and the paste wholly replaced by unrelated words, both
  `span_not_found`. Two changes in one read (typed before and edited inside)
  lose the region; one change per read, which is what polling at the tick
  gives, is followed. Offsets are UTF-16 units, attachment placeholders
  (U+FFFC) keep their place in the text, and a field read as a window (over
  8,192 characters) cannot be followed.
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
  word. It cannot tell a text pasted twice apart while one copy is intact, and
  an app that clears the field a tick after Return with the next message typed
  in the same tick cannot be told from an edit.

Known limits, stated plainly: a send button without a Return loses the
observation (the field going blank looks like delete-all); text without
whitespace (CJK) is one token, so it is effectively unsupported; a window over
8,192 characters is never followed; two changes in one read lose the anchored
region; the strict fallback is the only protection for apps that report no
caret, with the measured rate below.

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
