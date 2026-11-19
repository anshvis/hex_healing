# Tidy after paste

Labs › Tidy after paste (macOS, off by default; `labs.tidy_after_paste`)
suggests edits to the dictation Hex just pasted (fillers, repeats, the
abandoned part of a spoken self-correction, and with Apple Intelligence a
misheard word) and lets you accept or skip each one. Nothing happens at paste
time: suggestions are computed only when you ask (D-37). Status: the trigger,
the review capsule, and `hex dictation tidy [--print]` are implemented (T8.3);
applying the accepted edits in the pasted field through Accessibility is T8.4,
so Return copies the tidied text for now. Proof so far is unit tests and the
`tidy-capsule` scene; no live macOS run of the overlay or its focus has been
observed.

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
└── a dictation committed meanwhile -> the review is stale and never opens

Review capsule                                       // rune_ui::tidy_capsule
├── the pasted text: accepted removals struck through, replacements beside them,
│   skipped edits underlined
├── one switch per edit; on by default except a model `misheard` fix (D-67)
├── Accept all · Cancel · Apply (Return)  — Escape cancels, nothing changes
├── copy-only dictation (Copy dictation only, a copy fallback, --print)
│   -> the action reads Copy tidied text
├── no suggestions -> "Nothing to tidy in the last dictation", closes after 1.5 s
├── Return -> tidy::review::apply_accepted (T8.4's seam): copies the tidied text
│   and says "Paste it over the dictation"; nothing accepted -> closes, no change
├── one review at a time; a new trigger replaces it, the next dictation closes it
└── takes keyboard focus only while a review is open (non-activating panel,
    KeyPolicy::OnDemand), so closing hands the keys back to the app in front
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
close, Escape, no suggestions, copy only, a new dictation closing the review),
`rune_ui::tests::tidy_capsule_accepts_all_and_cancels`,
`app_settings::tidy::tests`, `suppression::tests::tidy_shortcut_press_is_its_own_action_only_while_bound`,
and the `tidy-capsule` golden in light and dark (Linux). Missing: a live run of
the overlay taking and returning focus, and the menu-bar item.
