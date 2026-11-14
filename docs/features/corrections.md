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

Outcome IDs and proof limits are recorded beside the implementation as it
lands (T2.2 reads, T2.3 keeps). Each bound above must match those constants;
a change to either updates this file in the same commit. A managed lock is
listed under [Managed feature locks](README.md#managed-feature-locks-macos).
