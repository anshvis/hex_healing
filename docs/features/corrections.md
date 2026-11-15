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

**Invariant: fail closed.** Hex stores an edit only from a region it is
following, only text inside that region (what now occupies the place the pasted
text occupied, plus a token that touches its edge), and never text from before
or after it. When unsure the answer is "not found" or "unchanged". A region
once lost stays lost, and no edit is stored before the paste has been seen
whole. There is no whole-field edit path: the old strict search for the edit
anywhere in the field is gone, because it trades precision for recall (about 1%
of unrelated chat sentences were stored as edits).

How the region is found (`locate.rs`, `Tracker`):

- With a caret. Before the paste, Hex reads the caret (`AXSelectedTextRange`'s
  location, in UTF-16 units, which is where a paste starts). The first read
  that shows the paste's words as a run starting within 8 units of it lands
  there. The region is then snapped to the field's own tokens (first matched
  word's start to the last one's end), so an app that stores fewer units than
  were inserted (blank lines collapsed, a dropped edge space), or an anchor a
  unit or two off, cannot leave it hanging past the paste. The paste found only
  elsewhere (another copy; the app has not applied it yet) is not a landing; it
  waits.
- With no caret (the app reports no selection range). Before the paste, Hex
  reads the field's text instead. The first read where the paste's words occur
  exactly once, having occurred nowhere before it, adopts that occurrence.
  Occurring before it, twice, or the field before it unread: the watch can only
  report unchanged or lost.

How the region is followed: every read is diffed against the last (common
prefix and suffix). Text inserted or removed before the region moves it, a
change inside stretches or shrinks it, one after it leaves it (so text typed
after an edit is never part of it), an insertion at its start counts as before
and at its end as after. A change that crosses an edge loses the region, and so
does one that could as well have been made on either side of an edge (text
typed before the paste that begins with the paste's first letters) or that ran
on to an edge over units that match what follows by chance. A change that
shortens the region at an edge (backspace, or typing over a selection) leaves
that edge open: typing right at it counts as inside, up to the words deleted
plus one; a longer change at an edge in one read is lost. So a trailing word
replaced by a word and one more is stored, key by key or in one read, and a
longer replacement is cut at that bound (key by key) or lost (one read), which
is also what keeps "Kip" to "Kit" plus appended unrelated text from being
stored. Two changes in one read (typed before, edited inside) lose the region,
except that whitespace deleted just outside it as "word " or " word" does not.
A blank read retires the region (an undo to empty, then other text pasted back
in, is never an edit), and so does a window (a field over 8,192 characters is
read around the caret and cannot be followed). Offsets are UTF-16 units and
attachment placeholders (U+FFFC) keep their place in the text.

What the region says: intact (the paste's words as a run in it) is unchanged,
whatever else the field says; the same text elsewhere does not count. For a
paste of one or two words, position is evidence: whatever now sits in the
region is what it became ("Hanks" to "Hex", "Yes" to "No"), unless it grew
past four extra words. A longer paste must still resemble itself: half its
words (exact or near, one letter off in a word of three or more) in order, two
of them exact. An empty region is the paste deleted, and a paste wholly
replaced by unrelated words, both `span_not_found`.

`property_tests.rs` checks the invariant over 4,000 generated cases, each run
with the caret known, with it a few units off, with none, with an app that
collapses blank lines, and with a field too large to read whole (11,976 runs).
The paste sits between corpus sentences (some sharing its words, some empty,
some with attachments, emoji or CJK), and goes through single-word, two-word
and trailing edits, replacement and deletion of the paste, typing before and
after it (all at once and a key at a time), backspace and retype at both edges,
typing over a selection, an edit coalesced with typing before or after, a wipe
and refill, a second paste of the same words or another, a Return and a newline
then a fix, and a cleared field. The stored span never leaves the region (plus
one token), a deleted paste stores nothing, and the one-word, two-word and
trailing replacements are stored exactly. Before this rework, the same cases
stored text outside the region 402 times and from a deleted paste 30 times, and
missed 150 of 390 known-good edits; now 0, 0 and 0.

Known limits, stated plainly: a send button without a Return loses the
observation (the field going blank looks like delete-all); a keystroke that
shares a tick with the Return that sends it is never read, so the edit kept is
the word as far as it was read ("Ki" for "Kit"); text without whitespace (CJK)
is one token, so it is effectively unsupported; a window over 8,192 characters
is never followed; two changes in one read lose the region; an app whose
selection offsets are not UTF-16 units gets unchanged or lost only (the paste is
never adopted from elsewhere); an app that clears the field a tick after Return
with the next message typed in the same tick cannot be told from an edit.

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
