# Learn from corrections

Labs › Learn from corrections (macOS, off by default; `labs.learn_from_corrections`)
notices when you edit a dictation right after Hex pasted it, and keeps the
edited span on that History entry so Hex can learn from it. This file is the
privacy contract for that read. Status: specified by the self-healing M2 plan;
the lab, the watcher, and the History field land in later commits of the same
PR, so no behavior below is observed yet. Only the managed lock
(`DisableCorrections`) exists with this file.

```ts
Dictation pasted, History entry recorded                // corrections.watch
├── Lab off, or DisableCorrections / DisableLabs forced -> nothing is read, nothing queued
├── Lab on -> Hex watches the element it pasted into, for up to 30 s
│   ├── Reads -> the Accessibility value of that one element, at most 8 KiB, in memory
│   ├── Stops at 30 s, the first focus-out, or Return
│   └── At most 4 targets are watched at once; extras wait in a bounded queue
├── Never read -> secure (password) fields, Hex's own window, any other element
├── Kept -> the edited span only, on the History entry, at most
│   2 x the pasted length + 64 characters; the rest of the field is dropped
│   └── Same retention, entry and byte caps, Delete and Clear all as the entry
└── Never logged -> any text; logs carry counts, durations and outcome IDs only
```

Watch rules (`dictation/correction_watch`, constants in `session.rs` and
`worker.rs`): a value change is read at once, then at most once a second, at
most 30 times per target, plus one closing read on focus-out (ignored in the
first 300 ms), Return (a key-state poll every 50 ms, no event tap), or 30 s.
The kept read is the last one in which the field still held the pasted text,
so a chat field cleared by send still yields the edit. A read is the whole
value up to 8,192 characters, else 8,192 around the caret. The gate (lab and
managed lock) is checked before every read, and Hex's own pid is never
observed. Outcome IDs: `unchanged`, `edited`, and the lost reasons
`element_gone`, `secure`, `too_large`, `span_not_found`, `read_timeout`,
`cleared_before_read`; triggers: `value_changed`, `focus_out`, `return`,
`timeout`. The log line carries those IDs, a character count and milliseconds
only. Alignment (`locate.rs`) treats a one-letter-off word of three or more
letters as a near match, so "Kip" to "Kit" aligns even alone; it cannot tell a
pasted text pasted twice apart while one copy is intact. Proof so far: unit
tests only; no live AX run (needs T2.3's History field).

Outcome IDs and proof limits are recorded beside the implementation as it
lands (T2.3 keeps). Each bound above must match those constants;
a change to either updates this file in the same commit. A managed lock is
listed under [Managed feature locks](README.md#managed-feature-locks-macos).
