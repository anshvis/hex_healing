# Decisions

These are calls the plan made without asking. Each one says what you could choose instead; tell
me to flip any of them. Format: `D-n (agent default): <call>. Instead: <alternative>.`

## Scope and layout
- **D-1** The plan lives in `docs/plans/self-healing/`. Sub-issues are Markdown files, one per task. No GitHub issues, and nothing in Linear (G-1). *Instead:* mirror each sub-issue as an issue on `anshvis/hex_healing`.
- **D-2** The off-device tooling (collector, labelling, eval, training, registry) is a Python package at `healing/` in this repo, managed with `uv`. Hex itself gets only small Rust changes. *Instead:* put the collector in Rust as `hex dataset …`. That is more native, but training is Python anyway.
- **D-3** Correction capture is macOS only. Linux (`linux/atspi`) and Windows (`windows/context`) have no reader for a focused field's text today. *Instead:* add the AT-SPI and UIA readers in a later milestone.
- **D-4** The dataset root is `~/Library/Application Support/hex-healing/` (0700). It is outside Hex's data folder, so Hex's History sweeps can never touch it. `HEX_HEALING_HOME` overrides it. *Instead:* a folder on the T5 drive. It is exFAT, so it couldn't hold owner-only files.
- **D-21** M0 and M1 sub-issues are full task packets. M2–M7 sub-issues are complete at the level of scope, acceptance criteria and verification. Their exact line references and gold-stars are refreshed when the milestone starts, because M0's outcome (architecture, converter) changes them. *Instead:* write every packet in full now.

## Data
- **D-5** Audio is stored as FLAC (lossless, about 55% of the WAV), named by the SHA-256 of the PCM. The index is SQLite. *Instead:* Opus at 32 kbps (about 10× smaller), but it's lossy, and you'd train on a codec's artifacts.
- **D-6** Near-duplicates are found with:
  - token 3-gram Jaccard ≥ 0.8 on normalized `raw_text`;
  - durations within ±30%.

  Each cluster keeps audio for at most k = 3 clips, chosen for acoustic spread (MFCC mean and variance distance). Beyond that, a clip keeps its text row only. The total budget is 5 GiB. *Instead:* audio embeddings such as ECAPA for similarity. That's better for "same words, different delivery", but a heavier dependency.
- **D-10** A session is a run of dictations with no gap over 10 minutes. The rolling holdout is the latest 20% of sessions. *Instead:* split by calendar week.
- **D-11** The general regression set has two parts, both scored every round:
  - 300 LibriSpeech test-other clips plus 200 AMI IHM utterances;
  - the stock catalog model, kept as a fixed reference model.

  *Instead:* LibriSpeech alone. It's smaller, but it has no conversational speech.
- **D-17** You confirm golden clips in a terminal review (`healing golden review`): play, see the reference, accept or fix. The set freezes at 200 confirmed clips. *Instead:* freeze 200 strong-labelled clips with no review.

## Corrections
- **D-7** Entries changed by an OpenCode rewrite produce no corrections at first, because their edits can't be mapped back to `raw_text` reliably. The signal is History's own `processing` record (`profile` present and `fallback` absent), exposed by T1.1. Replacements and transforms are mapped back by alignment. *Instead:* map rewritten entries too, at lower confidence. *(Revised after the challenge: this was a 30%-of-tokens guess.)*
- **D-8** The watcher reads the retained element in four cases, whichever comes first, and keeps the last read before the field empties:
  - on its `AXValueChanged` notifications, debounced to 1 s;
  - on focus-out;
  - on Return;
  - at +30 s.

  The watch ends at 30 s. *Instead:* a single read at +30 s (simpler). That misses chat apps, where you fix the text and send within seconds. *(Revised after the challenge.)*

  *Revised again at the M2 review gate (PR #4):*
  - The first tick reads the field as a baseline (a paste sent unedited otherwise reads as `cleared_before_read`). A tick that saw a value change reads once, on that tick (40 ms gap, under the 50 ms tick), not once a second. With a 1 s debounce, select "Kip", type "Kit", press Return inside a second kept the half-typed "send it to K". The read can still be one tick behind the typing, which is as close as polling at the watcher's tick gets; it is not "never a keystroke behind". Reads that find the field as it was before do not count toward the cap; only reads that found a change do (200 per target). A field that refuses the value registration is read every 400 ms.
  - The session is an explicit state machine: `Watching`, or `Sending` for 1 s after a Return (entered before the Return's read, holding what was kept then). A read of other, non-empty text drops the kept read. In `Sending`, the field going blank, shorter or holding other text than what was kept ends the watch with the kept read. A blank field with no Return is `cleared_before_read` (a send button without a Return loses the observation: it cannot be told from delete-all, and the training-safe side wins).
  - **Design change (round 6): an edit is stored only from a region Hex is following.** The whole-field strict edit path is gone (about 1% of unrelated chat sentences were stored as edits, and every round of tuning fixed only the shapes it was built for). The caret is read before the paste; the first read showing the paste within 8 units of it lands there and the region is snapped to the field's tokens. With no caret the field is read before the paste, and the one new occurrence is adopted. The region is followed by diffing each read against the last (typing before moves it, inside stretches it, after leaves it); a change that crosses an edge, or could as well sit on either side of one, loses it; a change that deletes words at an edge leaves that edge open for the words deleted, no more (see round 7). A blank read or a window retires it, and a lost region stays lost. The invariant is to fail closed: store only text inside the region (plus an edge token), never text before or after it, and no edit before the paste was seen whole. Checked by a generated property test (superseded by the round 7 oracle below).
  - **Round 7 (precision over recall), three decisions and a stricter oracle:**
    - **Edge allowance, no +1.** Typing at an open edge may fill only as many words as were deleted there. Typing past it loses the region, and a cut-short span is never stored (one word replaced by two is lost).
    - **Two changes in one read lose the region, however far apart.** The two reads are diffed word by word; more than one hunk is two changes, and so is one replacement that the words allow to be read as two (a word edited, then a phrase typed after it ending as the old words did). No absorbed word ever counts as inside.
    - **Landing needs exactly one near copy.** A read lands only where exactly one copy of the paste lies within 8 units of the anchor and no other near copy was ever seen; the anchor first moves through each diff by the same locator the region uses (the paste's own arrival excepted). Two near copies, or one that was there before the paste, lose the tracker. A blank read before landing is only noted; a windowed read retires the tracker even while it waits. A start edge backspaced from after the word and retyped keeps the open start (the region is not re-snapped while it is open). A region with no word left (attachments, emoji) follows only more keys at an open edge.
    - **Oracle.** Provenance by unique words (no slack beyond a touching token), exact-span checks for every edge edit, 30 actions (edits before landing, an empty field at the start, duplicate short pastes, windows crossing 8,192 mid-run, IME, autocorrect, undo and redo, two edits in one read, a word appended at the boundary). Default run a fast subset of 2,382 runs; ignored long run 208,731 (32,874 stored, 10,960 known-good): 0 outside, 0 from a deleted paste, 0 inexact, 0 missed. The previous design under the same default subset: 13 outside, 20 inexact, 29 of 111 known-good missed (first 8,958 runs: 45, 107, 123 of 466). 56 runs are waived (an edit beside typed text the words alone allow to read as one change). Mutation checks: every mutant of the landing, window, edge-bound, hunk and ambiguity logic is killed by a test except one, `crosses_only_blanks` (an ambiguous deletion whose only differing reading cuts the region through a character glued to it). No input found so far changes its outcome, so it looks equivalent, but that is not proven; the check stays in the code.
  - A blank field with something kept waits two ticks before it is called cleared, so a Return the listener counts a moment late (or an event order that puts the clear first) still makes it a send; a Return's events come first in a tick. Without that, a chat app that clears the field a tick after Return, followed by the next message, lost the edit or stored the next message as it.
  - Return closes the watch only if it emptied or shortened the field (a send). In a notes app Return is a newline, so the watch goes on and a later edit is kept. The listener counts Return presses (as it counts Escape for YOLO mode) instead of the watcher polling key state every 50 ms, which missed short taps and treated a Return in any app as a send.
  - An app losing focus closes the watch too (`AXApplicationDeactivated`), since switching away does not change the old app's focused element.
- **D-9** An edit counts as `acoustic` when either:
  - its Double Metaphone codes match;
  - normalized character edit distance is ≤ 0.6 and phonetic-code edit distance is ≤ 0.5.

  Thresholds are tuned in T2.4 against a labelled fixture of pairs. *Instead:* a G2P phoneme model, which is more exact but heavier.
- **D-16** The lab is Labs › Learn from corrections (`learn-from-corrections`): audience Everyone, off by default, group Dictation. It works without Keep recordings, because Tier 1 needs only text. *Instead:* the Dev audience only.

## Models and training
- **D-13** The base architecture is Parakeet Unified English 0.6B (Hex's default) if its NeMo checkpoint is public and survives T0.1's round trip. Otherwise Parakeet TDT 0.6B v2. Nemotron streaming is out of scope (brief: worse when not streaming). *Instead:* v3 (multilingual).
- **D-14** Fine-tuning uses LoRA (rank 16), merged into the weights before conversion. It targets the encoder's attention and feed-forward linear layers, **plus the TDT predictor and joint linears**: new words live partly in the decoder, so encoder-only LoRA may not learn them. NeMo's residual adapters have a nonlinearity, so they can't be merged, and transcribe.cpp loads only plain weights. T0.1 measures the per-tensor ‖ΔW‖ against the Q8_0 rounding step. *Instead:* encoder-only, which forgets less and learns new words less.
- **D-15** The GPU provider is Modal, with RunPod as the fallback. *Instead:* RunPod first.
- **D-18** Retraining starts only with at least 200 new strong clips and at least 7 days since the last run. *Instead:* the clip count alone.

## Promotion
- **D-12** The gate uses paired statistics over clips: 10,000 bootstrap resamples of Δ = Σerrors / Σwords, with clips as the units. A candidate passes only when every check holds:
  - **Rolling:** the 95% CI upper bound of Δ WER is below 0 (a real improvement).
  - **Terms:** an exact McNemar test on per-clip term hits favours the candidate at p < 0.05.
  - **General and golden:** non-inferior, with the CI upper bound of Δ WER under +0.5 pt.
  - **Formatting:** the formatting error rate is non-inferior, CI upper bound under +0.5 pt.
  - **Drift guard:** general WER ≤ v0 + 1.0 pt.
  - **Streaming:** the candidate streams, and p95 time to text ≤ current + 50 ms.
  - **Evidence:** at least 1,500 reference words in rolling and 20 term clips.

  *Instead:* the brief's point margins (0.3 pt / 5%). At these set sizes they are inside the noise. *(Revised after the challenge; the golden set now gates as non-inferiority.)*
- **D-19** Auto-rollback runs after at least 1,500 verified reference words arrive post-promotion, or 14 days, whichever comes first. It rolls back only when the promoted model is significantly **worse** than the previous one: the one-sided 95% CI lower bound of Δ WER is above 0. That is not the superiority test, which would flip-flop under noise. *Instead:* also watch the live edited-span error rate and re-run as soon as it jumps.
- **D-20** Mined terms fill the biased slots only after every term you added by hand. Mode vocabularies take slots first (`src/dictation/vocabulary.rs:174-198`), so slots are counted per mode. Ownership is tracked by content: a term you edited becomes yours. The list is written atomically with a new `hex vocabulary set --file` (T4.4). *Instead:* rank everything together by frequency.


## Defaults added after the challenge
- **D-22** Gating references require your review: about 50 random clips per round, roughly 10 minutes in `healing review`. *Instead:* gate on strong clips' edited spans alone. That's cheaper, but it can't see regressions on words you never corrected.
- **D-23** Training clips are capped at 30 s. Longer hands-free dictations are cut at pauses (the same cutter meetings use) or skipped. *Instead:* skip them all.
- **D-24** A new `DisableCorrections` managed preference sits alongside the existing `anomaly.hex` policy locks. *Instead:* rely on `DisableLabs` alone.
- **D-26** Imports of a streaming variant stream only behind Labs › Stream imported models, off by default (T7.4). The gate's streaming check tells you to turn it on. *Instead:* stream such imports by default.
- **D-25** The replay data's references are the base model's own transcripts (self-distillation). Its punctuation and casing are kept, instead of LibriSpeech's uppercase, unpunctuated text. *Instead:* normalize everything to lowercase, which risks teaching the model to drop formatting.

## Defaults added while building M0
- **D-27** The round-trip target word is "Qorvyx", not "Zorblax". `say` pronounces "Zorblax" so plainly that the base model already wrote it correctly in 8 of 10 held-out clips. "Qorvyx" is heard as "Corvix" by every voice, so the base misses it every time (0 of 10), which is the failure a fine-tune has to fix. *Instead:* any word whose spelling can't be guessed from its sound.
- **D-28** LibriSpeech test-other comes from Hugging Face's parquet copy (`openslr/librispeech_asr` at `71cacbf`), pinned by SHA-256, as AMI does (`edinburghcstr/ami` at `46f28f2`). openslr's own mirrors served the tarball at tens of KB/s. The utterances and transcripts are the same release. *Instead:* openslr's tarball, pinned by its hash.
- **D-29** The M0 spikes run NeMo through transcribe.cpp's pinned Parakeet environment (`scripts/envs/parakeet`, NeMo at `6967f48`), the one its converter and reference WER runner are tested with. `healing`'s own environment stays light (numpy, soundfile, pyarrow) until M6 needs training there. *Instead:* pin NeMo in `healing/pyproject.toml` now.
- **D-31** (amends D-14) LoRA targets the encoder's attention and feed-forward linears and the joint's linears. The predictor is an LSTM plus an embedding, with no linear layers, so it stays frozen. M0 proves only those targets survive conversion; LoRA on the LSTM's gate matrices would need its own round-trip proof first (T6.1). *Instead:* wrap the LSTM gates and prove them in a second spike.
- **D-33** (amends D-18) Retraining starts after every 200 minutes of new strong (or verified) speech since the parent version's snapshot, summed from audio durations, with no time floor. You asked for it (2026-11-14): the amount of new speech, not the calendar, decides. *Instead:* 200 clips plus 7 days.
- **D-34** (amends D-15) Training runs on your Mac's CPU by default: no upload, no cost, overnight. The optional cloud fallback is Kaggle's free GPUs, which have no payment method and so can never bill. You asked for free and card-free (2026-11-14). *Instead:* Modal or RunPod, faster but billed to a card past free credit.
- **D-35** Orphan audio files are moved to quarantine for 7 days, never deleted at once. A file with no index row may be the only copy of a clip whose row was lost. `healing clean` (T1.5) runs after every collect, with a 10 GiB free-space floor. *Instead:* delete orphans outright.
- **D-36** (T1.1 replan) The processing observation gets an explicit `rewrote` flag, set only when an OpenCode rewrite produced the used text. The packet's `processing.is_some() && fallback.is_none()` rule marked transformation-only dictations as rewritten, and missed rewrites followed by a failed transformation. *Instead:* drop `rewritten` and treat every processed dictation as unreliable for labels.
- **D-37** Speech-pattern cleanup is a separate step after the paste that you start. Hex suggests edits, you accept each one, and Hex applies only those through Accessibility (M8, Labs › Tidy after paste, off by default). You asked for it (2026-11-14): an accessibility action, not automatic. *Instead:* remove fillers automatically before the paste, as Labs › Clean up with Apple Intelligence does.
- **D-38** Tidied text never becomes a speech-model label. `raw_text` stays what was heard, and M2 resets its correction baseline after a tidy. Only accepted `misheard` replacements become weak corrections. Tidy decisions train only the suggester's habit list. *Instead:* train on tidied text, which would teach the model to drop words it heard.
- **D-32** T0.2's live smoke check (`recording_fidelity.sh`) was not built. The packet already called it circular (both WAVs come from one variable), and running it needs a second, menu-bar Hex on your Mac. The engine boundary test and the code reading across all three OSes are T0.2's proof. *Instead:* build it with an isolated Hex once a headless dictation path exists.
- **D-30** AMI references drop disfluencies dictation never outputs (um, uh, hmm, mm-hmm, uh-huh and the like) and keep real backchannels ("yeah", "oh", "okay"). AMI clips need at least 3 words left. *Instead:* keep fillers, which would add errors no model change can fix.
- **D-39** *Open, for M6/M7.* A fine-tuned Unified English model imported as `TranscriptionModelId::Custom` gets no trailing silence in dictation (`trailing_silence_samples` matches only the catalog's `ParakeetUnifiedEnglish`), and now none in the benchmark either, while the catalog model gets 6,400 samples (#1629). A base-vs-tuned A/B therefore compares differently prepared clips, and a promoted tuned model would lose the tail in dictation. Nothing changes in T3.4; M6/M7 decide whether the A/B pads both the same and whether a Unified import takes the tail. *Instead:* none yet.

## Challenge

**2026-11-13, fresh-context Claude Opus reviewer** (Codex unavailable: not installed). Same model family, not cross-model. It saw the files, not this session's reasoning. Its 21 findings and what happened to each:

| # | Finding | Disposition |
|---|---|---|
| 1 | Benchmark rejects groups `librispeech` and `app:*`; terms must be in the manifest vocabulary | **Folded:** T0.3, T3.2 and T3.3 use `ordinary`/`target`, with app and source labels in a sidecar. Verified at `benchmark.rs:391-413`. |
| 2 | History `model` is a display name; `language` is always None | **Folded:** T1.1 adds a stable `model_id` to the History record (a store change, now specced). `language` is dropped. Verified at `history.rs:221-230,431`. |
| 3 | Collector never sees corrections or Replace that arrive later | **Folded:** T1.2 upserts the mutable fields and re-scans a 2-day trailing window. |
| 4 | Span locator fails on edits to the first or last word | **Folded:** T2.2 uses token-level local alignment over the field window. |
| 5 | Pasted text ≠ `final_text` (join, smart spacing) | **Folded:** T2.1 has `paste` return the inserted text. Verified at `paste.rs:240-245`. |
| 6 | One +30 s read misses chat edits; X-3 measured the wrong thing | **Folded:** D-8 revised. X-3 is edit recall from a seeded protocol (T2.4). |
| 7 | Typed edits in History break the AGENTS.md invariant; no privacy review or policy lock | **Folded:** new T2.5 covers the AGENTS.md amendment, privacy review, `DisableCorrections` (D-24), feature-map and docs. |
| 8 | Strong references are mostly incumbent output | **Folded:** strong clips are scored on edited spans only; gating uses verified references (AC-7, T3.1, T3.3). |
| 9 | Rolling set chosen by the incumbent's errors | **Folded:** rolling is random clips with verified references (T3.2, D-22). |
| 10 | Clusters on `raw_text` with a duration gate leak | **Folded:** splits use union-find components on `reference_text`, with no duration condition. Storage clusters are unchanged (T1.3, T3.2). |
| 11 | Asymmetric leakage favours the incumbent | **Folded:** rolling = sessions after the parent snapshot cut-off, so neither model saw them (T3.2). |
| 12 | Gate blind to formatting; lowercased training invites regressions | **Folded:** D-25 replay, a formatting error rate gate (T3.3, T7.1), T6.1 keeps casing. |
| 13 | Imports never stream | **Folded:** new T7.4 (Hex: stream-capable Parakeet Unified imports) plus a streaming and latency gate check. Verified at `custom_models.rs:348`. |
| 14 | Gate runs without the user's vocabulary | **Folded:** the benchmark flag moved into M3 (T3.4), and eval and the gate pass your current vocabulary. |
| 15 | Margins are below noise | **Folded:** D-12 uses paired bootstrap, McNemar, and minimum words. |
| 16 | Auto-rollback reuses the superiority test | **Folded:** D-19 rolls back only when the promoted model is significantly worse. |
| 17 | T0.2 live check is circular | **Folded:** T0.2 asserts at the `finish_with` boundary and records the stream caveat. |
| 18 | T0.1 parity can't resolve 1 pt | **Folded:** 300 clips, a base control, and ‖ΔW‖ against the Q8_0 step (AC-2). |
| 19 | Small wrong facts (CFRetain, `recording` gating, rewrite signal) | **Folded:** README table corrected; T1.4 warns that turning Keep recordings off hides files; D-7 revised. |
| 20 | T0.1 and T6.1 secretly Deep; missing clip-length cap, import cleanup, v0 pin, Modal logs, network check, LoRA targets | **Folded:** T0.1 split (T0.1 base control, T0.4 LoRA round trip); T6.1 split (T6.1 data, T6.5 training loop); D-23; T7.2 cleanup; T5.1 pins v0 by sha; T6.2 scrubs logs; G-3 socket fixture; D-14 adds decoder and joint. |
| 21 | Vocabulary: mode slots, ownership, non-atomic writes | **Folded:** D-20 revised; new T4.4 `hex vocabulary set --file`. |

Nothing was rejected.
