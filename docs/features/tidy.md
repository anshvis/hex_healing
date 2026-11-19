# Tidy after paste

Labs › Tidy after paste (macOS, off by default; `labs.tidy_after_paste`)
suggests edits to the dictation Hex just pasted (fillers, repeats, the
abandoned part of a spoken self-correction, and with Apple Intelligence a
misheard word) and lets you accept or skip each one. Nothing happens at paste
time: suggestions are computed only when you ask (D-37). Status: the trigger,
the review capsule, and `hex dictation tidy [--print]` are implemented (T8.3),
and Return applies the accepted edits to the pasted text in its field through
Accessibility (T8.4). Proof so far is unit tests and the `tidy-capsule` scene;
no live macOS run of the overlay, its focus, or an edit in a real app has been
observed (the four-app matrix in T8.4's packet is owed).

```ts
Trigger                                              // macos_rune::tidy
├── Tidy shortcut (tidy_hotkey; unbound by default, needs a key)
│   └── matched like Paste last dictation (suppression::paste_action), only
│       while the lab is on and the binding shares no key (active_tidy_hotkey)
├── Menu bar › Tidy Last Dictation (shown only while the lab is on)
├── hex dictation tidy           -> opens the review, replies how many edits
└── hex dictation tidy --print   -> prints the suggestions as JSON (docs/cli.md);
                                    opens and applies nothing
Lab off -> shortcut reserves nothing, menu item absent, command answers `disabled`;
           nothing is read from History or computed

Prepare (off the main thread)                        // dictation::tidy::review
├── History's newest entry (printed ones included)   -> its final_text, raw_text, language
│   └── none -> "No dictation to tidy yet", closes after 1.5 s; command: `nothing-to-tidy`
├── T8.1 suggest(pasted, raw, language, habits)      -> up to 2 s for the model pass
└── checked again on the main loop before it shows: a dictation committed
    meanwhile, or the lab turned off -> the review never opens

Review capsule                                       // rune_ui::tidy_capsule
├── the pasted text: accepted removals struck through, replacements beside them,
│   skipped edits underlined
├── one switch per edit; on by default except a model `misheard` fix (D-67)
├── Accept all · Cancel · Apply (Return)  — Escape cancels, nothing changes
├── copy-only dictation (Copy dictation only, a copy fallback, --print)
│   -> the action reads Copy tidied text
├── no suggestions -> "Nothing to tidy in the last dictation", closes after 1.5 s
├── Return (pasted dictation) -> hides the panel, applies on hex-tidy-apply
│   (see Apply below): done -> "Tidied the dictation"; text moved or field not
│   editable -> the review comes back as copy-only, saying why, Return copies
├── Return (copy-only) -> copies the tidied text; nothing accepted -> closes, no change;
│   a newer dictation or the lab off since it opened -> closes, nothing applied
├── the lab turning off closes an open review (rechecked every 250 ms)
├── one review at a time; a new trigger replaces it, the next dictation closes it
├── takes keyboard focus only while a review is open (non-activating panel,
│   KeyPolicy::OnDemand); any end of the review hides it at once, handing the keys
│   back to the app in front
└── notices show in a second, click-through overlay that never takes the keys

Paste anchor                                         // dictation::paste_anchor
├── while Tidy after paste or Learn from corrections is on, one read of the focused
│   field before each dictation's paste (shared with the correction watch)
├── kept once the paste, its History entry, and its commit happened: element,
│   pid, caret, the inserted text's SHA-256 and UTF-16 length, entry id, commit
│   count; never the text (D-74)
└── one slot, in memory: every paste clears it; unusable after 10 minutes

Apply                                                // dictation::tidy::apply
├── lab off -> nothing read or sent (checked at the start and before each edit)
├── no anchor for this review's entry and commit -> unsupported, offers a copy
├── secure field (role or subrole; one that can't be read counts as secure)
│   -> refused before anything is read
├── the pasted-into app not frontmost within 600 ms -> unsupported (OtherApp)
├── read the whole field (≤ 32,768 UTF-16 units, else TooLarge)
├── locate the pasted text by hash: exactly once, else Moved (D-75); the
│   dictation inside it may only differ by leading whitespace and one first
│   letter's case (D-76), else unsupported (Differs)
├── the edit, its read-backs, and any fallback run under the delivery lock a
│   dictation's paste also takes; a dictation paste since the anchor was taken
│   -> nothing sent (D-79)
├── AXSelectedTextRange + AXSelectedText when both are settable: one undo step
│   └── ignored by the field -> read again and placed from that read; the app
│       still frontmost and the field focused right before Hex's paste over it;
│       refused -> the selection collapses to the range's end (D-77, D-79)
└── re-read: exactly the plan (outside the range byte-identical) -> applied;
    anything else -> Unverified, "Check the text; ⌘Z undoes it", anchor spent
```

Platforms: macOS only. Linux and Windows have the rule pass but no review
overlay that takes keys, tray item, or shortcut adapter yet; `platform::DIFFERENCES`
records `tidy` as pending, and their control hosts answer `unsupported`.

Privacy: the review reads the History entry Hex already keeps. The logs carry
counts and kinds only (`tidy suggestions ready`, `tidy review ready`, `tidy
review applied`, `tidy review cancelled`), never text. The commit marker the
output stage keeps (`dictation::tidy::last`) holds a counter, an entry id, and
whether it was only copied.

Checks: `labs::tests::tidy_off_by_default`,
`control::tidy::tests::tidy_command_refused_when_lab_off` and
`print_answers_the_suggestions_without_opening_anything`,
`dictation::tidy::session::tests`, `dictation::tidy::review::tests`,
`rune_ui::tidy_capsule::tests` (toggle and Accept all, Return and the 1.5 s
close, Escape, no suggestions, copy only, a new dictation closing the review,
a stale or lab-off review never showing or applying, the lab turning off
closing an open review, and an ended review keeping no notice of its own),
`rune_ui::tests::tidy_capsule_accepts_all_and_cancels`,
`app_settings::tidy::tests`, `suppression::tests::tidy_shortcut_press_is_its_own_action_only_while_bound`,
`dictation::tidy::apply::tests` (locate by range then unique search, duplicate
text is Moved, outside the range byte-identical, the secure check first, the
paste fallback and its fresh read and front-app check, unverified edits, a newer
paste or the lab off sending nothing), `dictation::paste_anchor::tests`,
`dictation::correction_watch::tests`, and the `tidy-capsule` golden in light
and dark (Linux). Missing: a live run of the overlay taking and returning
focus, the menu-bar item, and AC-17's four-app matrix (TextEdit, Notes, Slack,
a Chrome textarea: only the pasted range changes, typed text around it stays,
one ⌘Z undoes it).
