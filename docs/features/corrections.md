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

**Invariant: fail closed.** Hex never stores text that is not the user's edit of
the paste. It stores an edit only from a region it is following, only text
inside that region (what now occupies the place the pasted text occupied), and
never text from before or after it. When unsure, and whenever two keystroke
histories give the same reads and one of them stores outside text, the answer is
"not found" or "unchanged": real corrections are lost rather than a wrong one
stored. A region
once lost stays lost, and no edit is stored before the paste has been seen
whole. There is no whole-field edit path: the old strict search for the edit
anywhere in the field is gone, because it trades precision for recall (about 1%
of unrelated chat sentences were stored as edits).

How the region is found (`locate.rs`, `Tracker`):

- By proof of arrival (round 10). Hex reads the field before the paste
  (`before`) and the caret (`AXSelectedTextRange`'s location, in UTF-16 units,
  which is where a paste starts). It lands only on a read that is `before` with
  exactly the paste's characters put in at one place (within 8 units of the
  caret when there is one, anywhere without one), where the paste's words stand
  apart from the text on both sides: the paste's first character is not
  against a word before it, nor its last against a word after it (a paste glued
  to a neighbour is lost, since the glued word's own letters could be either's).
  The region is the paste's words in that inserted span. More than one place
  that fits (a periodic text: "no " put in "no world") loses the tracker; so
  does a one-character paste that shows up after a read of the field as it
  was, which a typed key cannot be told from. A read that is `before` unchanged
  keeps waiting (the app has not applied the paste yet); any other read, a
  blank one included, loses the tracker, so text typed before the paste arrives,
  an app that rewrites what it pasted (blank lines collapsed, a dropped edge
  space), a paste that arrives elsewhere than the caret, and a field cleared
  and refilled all store nothing. No copy of the paste's words found in the
  field decides anything: the insertion is the only proof, which is why an old
  copy brought near the caret, one typed afterwards, and one glued to a
  neighbour cannot be taken for the paste. If the field could not be read
  before the paste (a failed read, a window), nothing is followed, caret or not.
- A window (a field over 8,192 characters) retires the tracker, also while it
  is still waiting, and also when the window is blank.

How the region is followed: every read is diffed against the last (common
prefix and suffix, in UTF-16 units). Text inserted or removed before the region
moves it, a change inside stretches or shrinks it, one after it leaves it (so
text typed after an edit is never part of it). A change that crosses an edge
loses the region, and so does one that could as well have been made on either
side of an edge (text typed before the paste that begins with the paste's first
letters), on every path, an open edge included. Whitespace deleted just outside
the region as "word " or " word" is not a crossing, but it is another reading of
the change that must agree with the rest (R3).

Two changes in one read lose the region, however far apart: the words of the
two reads are diffed, and more than one hunk (an edit inside and text typed
before it, a word edited and another word after it) is two changes.

The rule: **never store text that is not the user's edit of the paste; any
ambiguous read stores nothing.** Where two keystroke histories give the same
reads and one of them stores text from outside the paste, the answer is lost.
Real corrections are lost with them, and that is accepted: a missed correction
costs one training example, a wrong one teaches the model text the person never
meant. The rules that bound what a region can take in (R1 to R9, D-8):

- **R1, the word budget, per edge.** A followed region never holds more words
  than the paste had, and the words put back at an open edge never exceed the
  whole words taken away at that edge (a room per edge, spent by words typed
  at it or a little in from it, refilled by whole words deleted there). So one
  word replaced by two is lost ("Hanks" to "Hank Smith", "Kip" to "Kit my
  password is hunter2", "Kit chen" to "Kitchen and then"), words taken at the
  start do not let words be put at the end ("one two three four five six" with
  "one two" deleted and "six" typed over by "seven and Sarah"; "Hanks Smith"
  with "Hanks" deleted and "Smith" typed over by "Jones and"), and the same
  text gives the same answer typed key by key or in one read.
- **R2, edges.** An edge opens when a change took or replaced its whole edge
  word (backspace, a selection, typing over, forward delete), as the characters
  of the change say, not as the words of the diff do: deleting the last letter
  of a word, or fixing one, edits it and opens nothing. Keys typed right at an
  open edge are inside, as far as R1 allows. Whitespace typed there closes the
  edge and is never part of the region, for typing and for a replacement that
  ends in it ("Smith" typed over by "Rune "): a space, once the words taken
  there are back (before that it is lost); a line break always. Keys glued to
  an edge word are that word, longer. An edge beside words typed right outside
  it, on the same line, can never open: which of the words around it is the
  replacement is not known (a new first word, then the old first word
  forward-deleted, is lost).
- **R3.** The ambiguity check (which side of an edge a change was made) runs
  on every path, an open edge included, and includes the readings of a window
  that crosses an edge only over whitespace: each is clipped to the region and
  must put it where the others do, and one that deletes all of a region (or
  puts text in its place) disagrees with every other reading. A one-character
  region deleted with its space ("great idea a" to "great idea") is lost, not
  moved onto "idea".
- **R4.** Landing is by proof of arrival, above: the read is the field as it
  was with exactly the paste's characters put in at one place near the caret,
  not glued to a word beside it. The old machinery (old-copy offsets, `seen`,
  the arrival test, the anchor moving with typed text) is gone. The field must
  have been read before the paste: without it nothing is followed.
- **R5.** A window is checked before a blank read, so a blank window retires a
  waiting tracker, and so does any blank read that is not the field as it was.
- **R10.** Blanks put over a word at a region's edge (a line break typed over a
  selection of the last word) lose the region: nothing says where the
  replacement ends, and text typed after the break is a new paragraph's.
  Keys with a blank among them, glued to the last word in one read, lose it
  too: they are "pple pie Hanks" typed after the pasted "a", or "apple pie "
  put before it and the "a" typed over.
- **R6, words that run out of the region.** A word that overlaps the region
  and extends outside it (the paste's last word glued to the text after it by
  deleting the space, or typed against the text before it) is joined text, none
  of it known to be the paste's: the region is lost, never stored with the
  word's whole span.
- **R7, a separator the region keeps.** A change inside the region that reaches
  its edge and puts in words beginning with the separator beside it (or ending
  with the one before it) could as well have been typed past that separator,
  outside the region ("Hanks Smith tail" to "Hanks  secret tail": "Smith"
  taken out and "secret " typed before "tail"): it is lost.

- **R11, a read that could have been made another way (round 11).** After
  landing, a read is followed only if its change from the last read has one
  reading. The words of the change slid along equal words (a word taken out of
  "a a a") must sit on the same side of the region in every alignment; the
  region's first or last word may not equal the word beside it, nor may the
  words put in or taken out equal the words kept beside them ("Yes" to "No"
  with " No" after it: "No" is the word that was there, or the replacement),
  and new words standing where the paste's did may not equal the words beside
  them. Reads of more than one character also lose the region when they type
  a copy of its text (glued to it or not, "Ikip hanks kip hanks"), put in a
  word near one of its words again (one or two letters off), or take out and
  type letters in one read where no word changed. A change of one key is held
  only to the first three.
- **R12, line breaks.** Words put in at an open edge, or over a word at an
  edge, that have a line break among them lose the region: what follows the
  break is a new line, not the replacement carried on.

A region that has lost every word (replaced by attachments, an emoji) has no
edge to open: what is typed beside it is outside.

A blank read retires the region (an undo to empty, then other text pasted back
in, is never an edit), and so does a window (a field over 8,192 characters is
read around the caret and cannot be followed). Offsets are UTF-16 units and
attachment placeholders (U+FFFC) keep their place in the text.

What the region says: intact (the paste's words as a run in it) is unchanged,
whatever else the field says; the same text elsewhere does not count. For a
paste of one or two words, position is evidence: whatever now sits in the
region is what it became ("Hanks" to "Hex", "Yes" to "No"); the budget keeps it
from growing. A longer paste must still resemble itself: half its
words (exact or near, one letter off in a word of three or more) in order, two
of them exact. An empty region is the paste deleted, and a paste wholly
replaced by unrelated words, both `span_not_found`.

`property_tests.rs` checks the invariant over generated cases (41 actions, among them a copy of the paste typed with a key glued to it and a word changed in one read, glue and a space in one read, the paste replaced by the word beside it, two words replaced by "word, line break, word", and two ordinary changes in one read),
each run with the caret known, with it a few units off, with none, with an app
that collapses blank lines, and with a field too large to read whole. The paste
sits between corpus sentences or unique words (some sharing its words, some
empty, some with attachments, emoji or CJK; a field that starts empty; a short
paste that also sits there already; an edit made before the paste is ever
seen), and goes through single-word, two-word and trailing edits, replacement
and deletion of the paste, typing before and after it (all at once and a key at
a time), backspace and retype, typing over a selection and forward delete at
both edges, IME composition, autocorrect, undo and redo, two edits in one read,
a word appended at the boundary, a one-character edge word typed over, the
space between the paste and a word beside it deleted, a last deletion merged
with the first typed key, a paste's tail cut and then the next space deleted, a
space or line break typed at an open edge and then words typed before it, a
word typed before the first and the first then deleted, a wipe and refill, a
second paste of the same words, a Return then a fix, a cleared field, a
pre-existing copy brought near a shifted anchor, and a window that crosses
8,192 characters mid-run. The oracle is provenance: every word of a stored span
must lie wholly inside the paste's own region (a word that touches it or is
joined to the text beside it is not inside; words are unique, so a copy
elsewhere cannot pass for it), an edge edit stores exactly the region, or the replacement
as far as the paste's words were used (a prefix, never a longer span), or
nothing, a deleted paste stores nothing, and the edits a person would call one
change are all stored. A reviewer's independent fuzz (`fuzz_tests.rs`, its own
model of the region) and recall measurement (`recall_tests.rs`) run beside it.

The default run is a fast subset (about 3,600 runs). The ignored
`what_is_stored_holds_over_a_long_run` is 857,508 runs (128,698 stored, 32,024
known-good): 0 outside the region, 0 from a deleted paste, 0 inexact, 0
known-good missed, 5 known-good edits lost where the old and new word share an
edge character and the text has a second reading, and 14 waived. The one
waiver left is a typed line break an app that collapses blank lines shows as
nothing (each waiver applies only to the step at which its case first leaks,
and a deleted paste's leak is never waived). Round 10 deleted the three other
waiver clauses (two changes in one read, an edit before the paste was first
read, keys after a replaced paste): they were dead over 857,700 runs, and the
one case that did reach the first of them with the new generators ("apple pie
Hanks a" after pasting "a") is now lost by the tracker instead. The generator
gained glued pastes (no separator on one side or both) and repeated or
periodic pastes beside the same words. `origin_tests.rs` judges a stored span
by the origin of each character and runs the reviewer's four fuzzers, and
`glue_fuzz_tests.rs` the glued-paste fuzzers (unfiltered: a leak counts
whether or not the tracker landed), 300,000 cases each, 0 leaks (the round 9
code leaks in them within 20,000 cases). `recall_tests.rs` measures every way
of fixing each word of 12 pastes (64,701 fixes): 64,059 (99.0%) stored
exactly, none stored as anything but the fix: the same as round 9. Round 11
measures 63,039 (97.4%): see below. `cell_fuzz_tests.rs` (the reviewer's `fz8`)
tracks every character's origin through sessions of typing, deleting, typing
over and pasting a copy again, with reads coalesced one in three or eight and a
two-letter alphabet; 1,500,000 sessions in each of six configurations, 0 leaks
(the round 10 code leaks in 140 to 309 of them in the coalesced ones).

The recall cost of round 9, stated plainly: 21 fewer fixes of 64,701, all a first
word backspaced to part of a word and retyped two keys at a time ("the", "t",
then "tht": which reads the same as "th" typed before the "t"), and in the
property generator 14 of 44,911 known-good edits (an edit whose old and new word
share an edge letter). Beyond what those measure: words typed at an edge are
limited to the words taken there, a replacement that ends in whitespace before
its words are back is lost, text typed against the word after a start edge
whose word was deleted is lost, and a paste not in place at the first read
lands later only as one change bringing in more than one character of it.
Round 10 adds: a paste glued to a word beside it, one the app rewrote, one that
arrives away from the caret, and a periodic text that fits two places are never
followed; a one-character paste into a field read empty first is lost (a typed
key looks the same); keys with a blank among them glued to the paste in one
read are lost. The generator's known-good count fell (44,911 to 32,024)
because glued and periodic cases are excluded from it; recall on the 64,701
fixes did not change.

Round 11 recall cost: 63,039 of 64,701 fixes stored exactly (97.4%, from
99.0%), none wrong. The 1,662 lost are fixes whose reads are ambiguous in the
sense of R11 (typing a letter of the region's own text in a coalesced read; a
replacement that is a word elsewhere in the paste or beside it; a retyped word
that is the word before the paste), and in the property generator 182 of 28,181
known-good edits (0.65%) plus 310 edge-ambiguous ones. A missed correction costs one training example; a stored wrong one teaches the
model text the person never meant.

Known limits, stated plainly: a send button without a Return loses the
observation (the field going blank looks like delete-all); a keystroke that
shares a tick with the Return that sends it is never read, so the edit kept is
the word as far as it was read ("Ki" for "Kit"); text without whitespace (CJK)
is one token, so it is effectively unsupported; a window over 8,192 characters
is never followed; two changes in one read lose the region (and a change the
word diff cannot work out, over 250,000 word comparisons, is called two); a
replacement with more words than it took out is lost, so one word replaced by
two is lost, and a last word fixed and followed by more words in one burst may
be lost (typed key by key, the fixed word is kept and what follows the space is
not stored); two words typed over two key by key are lost at the space, not cut
short, while the same two in one read are stored; a word typed before the paste
and then the paste's first word forward-deleted is lost, as is a word typed
into the first word (no space) and the old letters then deleted; a paste's
replacement typed before the paste arrives loses the landing (any change but
the paste's own arrival does); two places the paste could fit are never told
apart; an app whose selection offsets are not UTF-16
units gets unchanged or lost only (the paste is never adopted from elsewhere);
an app that clears the field a tick after Return with the next message typed in
the same tick cannot be told from an edit.

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
