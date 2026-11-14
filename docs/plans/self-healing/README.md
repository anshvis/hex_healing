# Self-healing Hex

The plan for a Hex that learns its user's voice. It captures every dictation and every correction,
turns them into labelled data, and fine-tunes the speech model off-device. A new model replaces
the current one only when it measurably wins. The source brief is [`instructions.md`](../../../instructions.md).

This folder is the plan, and nothing is built yet. Each milestone has its own folder:
- `README.md` is the milestone block: scope, acceptance criteria, how to validate it, and the exit gate.
- `T<m>.<n>-*.md` are the sub-issues, one per task.

**We work one milestone at a time.** A milestone starts only after the one before it has passed
its exit gate and you have checked its validation.

## How I read the goal

The brief asks for two learning loops fed by one dataset:
- **Tier 1**, immediate and with no training: correction pairs mined into Hex's vocabulary list.
- **Tier 2**, after every 200 minutes of new speech: LoRA fine-tuning of Parakeet on your Mac (Kaggle's free GPUs as a fallback), gated by held-out
  scores, shipped back into Hex as a Q8_0 GGUF.

Everything rests on two things Hex does not do today:
- keep dictation data past its one-week History window;
- notice what the user changed after a paste.

So the order is:
1. De-risk the fragile steps first: the converter round trip, and whether the kept audio is really
   the model's input.
2. Build capture.
3. Build corrections.
4. Build the measurement.
5. Only then train.

## What Hex already has (mapped 2026-11-13, `dev` @ 3444739e)

| Need | Already in Hex | Gap |
|---|---|---|
| Keep audio | Labs › Keep recordings saves the exact whole-clip samples given to the model (`src/parakeet.rs:460-464`) as a 16 kHz mono i16 WAV at `history-audio/<id>.wav` (`src/history/recordings.rs:38,70-90`). | Off by default. 512 MiB cap. Deleted with its History entry (7-day default retention, at most 10,000 entries). |
| Export history | `hex history --json` is read-only and safe to poll (`src/control/transcript_cli.rs:416`). It has `id`, `timestamp_ms`, `application`, `mode` (which is `processing.profile`), `kind`, `final_text`, `raw_text` (with `--raw`), and `recording` (the WAV path, shown only while Keep recordings is on *now*). | The row omits `model` and `audio_ms`. Worse, History's `model` is a **display name** read at record time (`src/history.rs:221-230`), and `language` is always `None` (`:431`). Nothing stable says which model produced `raw_text`. |
| Raw vs final text | `HistoryEntry.raw_text` holds the transcript before mode processing; `final_text` is the pasted text (`src/history.rs:114-148`). | None. |
| Target app | `application`, the display name only. | No bundle id, which is fine. URL and window title are never stored, by design. |
| Read another app's text | `src/accessibility.rs`: bounded AX reads with a 0.25 s timeout. Secure fields and Hex's own pid are skipped. Used by `text_around_caret` (`:257`) and `focused_window_text` (`:109`). `CfObject` already wraps retain and release (`:435`, `:511`). | Nothing reads text after a paste. No element outlives a single read. `Paster::paste` has no completion hook and **rewrites the text before inserting it**: continuation join and smart spacing (`src/paste.rs:240-245`). `record_history` discards the entry id (`src/parakeet.rs:1234`). |
| Streaming | The default Parakeet Unified model streams. | **Every imported model gets `streams: false`** (`src/transcription/custom_models.rs:348`). A promoted fine-tune would lose live streaming and Live text unless Hex learns to stream an import (T7.4). |
| Vocabulary | `vocabulary` in settings holds `{term, sounds_like[]}`, up to 500 stored. **Only the first 64 in list order are biased** (`src/dictation/vocabulary.rs:26,290`). `hex vocabulary add/remove/list` exists. | No way to reorder the list from the CLI. |
| Import a model | `hex model import <gguf>` accepts F32, F16 and Q8_0 parakeet GGUFs whose header has `stt.variant` and `general.languages`. `hex model use custom-<id>` switches to it. | None. Orukeet (`src/transcription/artifacts.rs:42-50`) shows a third-party fine-tune already went NeMo → GGUF → Q8_0 → Hex. |
| Measure WER | `hex benchmark-transcription <manifest> --model <gguf>` (hidden) prints JSONL with `wer`, `term_recall`, false replacements, per-group quality, and each clip's `transcript` (`src/transcription/benchmark.rs:317`). | It never passes vocabulary to the model (`docs/plans/transcription-quality.md:37` plans a `--vocabulary` flag). Groups must be one of `target`, `distractor`, `ordinary` or `private`. A target clip's `expected_terms` must be in the manifest `vocabulary` (`:391-413`). WER ignores case and punctuation (`:505`). |

## Architecture and data flow

```ts
Hex (Rust, on the Mac)
├─ dictation → History entry (raw, final, app, model) + history-audio/<id>.wav    [Keep recordings: on]
└─ Labs › Learn from corrections (M2)
   └─ 30 s after a paste: re-read the same AX element → History entry gains `correction`
                                │  hex history --json --raw (read-only)
                                ▼
healing/ (Python, uv; this repo) ── data root: ~/Library/Application Support/hex-healing/ (0700)
├─ collect (M1): copy rows + audio → dataset.sqlite + audio/<sha256>.flac      (idempotent, launchd)
├─ prune   (M1): near-duplicate clusters, k audio per cluster, byte budget
├─ label   (M2/M3): acoustic vs rewrite edits against raw_text; strong/weak/verified
├─ split   (M3): rolling holdout · frozen golden (200) · general regression
├─ eval    (M3): manifests → `hex benchmark-transcription --model <gguf>` → scores
├─ vocab   (M4): pairs seen twice → `hex vocabulary` (ranked, ≤64 biased)
├─ registry(M5): versions.jsonl + content-addressed artifacts (LoRA deltas, manifests)
├─ train   (M6): NeMo LoRA on your Mac (Kaggle fallback, explicit upload opt-in) → merge → GGUF → Q8_0
└─ promote (M7): gate on the eval scores → `hex model import` + `hex model use`; auto-rollback
```

**The gate always scores the Q8_0 GGUF running in Hex's own runtime, not the NeMo checkpoint.**
That is what users actually run. It also catches two silent failures: a converter that mangles a
fine-tune, and Q8_0 quantization rounding away a small LoRA delta.

The gate also runs with your current vocabulary (T3.4). It checks formatting (case and punctuation) and
streaming latency, not only WER. A model that forgets punctuation or loses streaming would
otherwise pass a WER-only gate.

**Gating test references are human-verified, never model output.** A strong label is `raw_text`
with only your edits applied, so every word you didn't touch is still the current model's own
output. Scoring a whole strong clip would hand the current model about 0 errors on those spans
and punish any change. Strong clips are therefore scored only at the positions you edited (the
edited-span error rate, a diagnostic). Gating WER uses verified clips, sampled at random for
review, not picked because the current model got them wrong.

## Milestones

| | Milestone | Validates | Depends on |
|---|---|---|---|
| [M0](M0-feasibility/README.md) | Feasibility spikes and baseline | The converter takes a fine-tuned checkpoint. The kept audio is the model input. Baseline scores exist. | none |
| [M1](M1-capture/README.md) | Long-term capture | Every dictation lands in a dataset that outlives History, with storage that stays bounded. | M0 |
| [M2](M2-corrections/README.md) | Correction capture | Post-paste edits are recorded and split into speech errors vs. change of mind. | M1 |
| [M3](M3-labels-eval/README.md) | Labels, holdouts, evaluation | Three leak-free test sets and one command that scores any GGUF. | M2 |
| [M4](M4-vocabulary/README.md) | Tier 1: vocabulary mining | Repeated corrections become biased terms, and the benchmark shows the gain. | M2, M3 |
| [M5](M5-registry/README.md) | Model registry | Every model is versioned with its data and scores, and can be rolled back. | M3 |
| [M6](M6-training/README.md) | Tier 2: LoRA training | Training triggers after 200 or more minutes of new strong speech and yields an importable Q8_0 GGUF. | M0, M3, M5 |
| [M7](M7-promote/README.md) | Promotion, ship, rollback | A model ships only if it wins with statistical confidence, keeps streaming and formatting, and reverts automatically only if it is significantly worse. | M3 (incl. T3.4), M5, M6 |
| [M8](M8-tidy/README.md) | Tidy after paste | After a paste, Hex suggests removing fillers and spoken self-corrections. You accept each edit, and Hex applies it through Accessibility. Tidied text never trains the speech model. | M2, M1 (runs after M7) |

M4 and M5 are independent of each other and can run in parallel once M3 is done. M8 is a follow-up after M7.

## Goal-level criteria

### Acceptance criteria (all inferred from `instructions.md`)
- **AC-1** Hex's kept recording for a dictation is the exact sample buffer its whole-clip transcription received, up to i16 quantization. Proven by a test. (M0)
- **AC-2** A LoRA-merged fine-tune of the chosen Parakeet architecture converts to GGUF, quantizes to Q8_0, imports with `hex model import`, and transcribes. On 300 LibriSpeech clips:
  - the paired-bootstrap 95% CI of WER(GGUF Q8_0) − WER(NeMo) lies within ±1.0 pt;
  - the untouched base model's conversion, run as a control, does the same.

  The learned change survives Q8_0. (M0)
- **AC-3** `healing collect` copies every History entry into the dataset: audio, `raw_text`, `final_text`, app, stable `model_id`, whether a rewrite applied, and timestamp. It is idempotent. It updates a row when Hex later adds a correction or replaces the transcript. An entry Hex later deletes stays in the dataset. (M1)
- **AC-4** Dataset storage stays at or under its byte budget while new unique clips are still admitted. Near-duplicate clusters keep at most k audio files. Strong labels, golden clips and verified clips are never evicted. (M1)
- **AC-5** With Labs › Learn from corrections on (macOS), an edit to the pasted text within the observation window is recorded on that History entry. This includes edits made just before a chat app sends. Nothing is read when the lab is off, when the managed preference `DisableCorrections` is set, in a secure field, or in Hex's own window. (M2)
- **AC-6** Each recorded edit is classified as `acoustic` or `rewrite` against `raw_text`, never `final_text`, using a phonetic plus edit-distance check. Only `acoustic` edits produce training references. (M2)
- **AC-7** Clips are labelled `strong` (acoustically corrected), `verified` (confirmed by the user) or `weak` (observed unedited). Weak clips have weight below 1 and are excluded by default. Gating test sets use only `verified` references. Strong clips are scored only at their edited positions. (M3)
- **AC-8** There are three test sets. Each is disjoint from training by split-cluster: a connected component of near-duplicate `reference_text`, not only by clip.
  - **Rolling holdout:** randomly sampled, verified clips from sessions after the parent version's snapshot cut-off, so no compared model trained on them.
  - **Frozen golden set:** 200 verified clips, pinned by hash, that never change.
  - **General regression set:** a fixed LibriSpeech test-other slice plus a conversational slice.

  (M3)
- **AC-9** `healing eval <gguf>` reports, for each set, through `hex benchmark-transcription` with your vocabulary:
  - WER;
  - formatting error rate (case- and punctuation-sensitive);
  - term error rate;
  - the edited-span error rate on strong clips;
  - p95 time to text.

  It also keeps per-clip errors for paired statistics. (M3)
- **AC-10** A correction seen the same way twice becomes a vocabulary term with a sounds-like spelling. The 64 biased slots are ranked by recent frequency. Terms the user added by hand are never removed or demoted. The benchmark shows the effect. (M4)
- **AC-11** Every model version records:
  - its base and parent;
  - the dataset snapshot hash;
  - the training config;
  - the code SHA and converter revision;
  - the GGUF SHA-256;
  - its scores on all three sets;
  - the gate decision.

  Any recorded version can be restored and switched to. (M5)
- **AC-12** Training starts only when there are at least 200 minutes of new strong speech since the last run (D-33). It produces a merged LoRA checkpoint and a Q8_0 GGUF registered as a candidate. (M6)
- **AC-13** A candidate is promoted only if every check passes, each measured with a paired bootstrap over clips (D-12):
  - rolling WER improves (the 95% CI upper bound is below 0);
  - term errors improve (an exact McNemar test on per-clip term hits);
  - general WER and golden WER are non-inferior (CI upper bound under +0.5 pt);
  - the formatting error rate is non-inferior;
  - the drift guard against v0 holds;
  - it still streams and p95 time to text is within budget.

  Every refusal is logged with the failing check. Promotion is `hex model import` followed by `hex model use`. (M7)
- **AC-14** After a promotion, a non-inferiority check re-runs on fresh post-promotion verified clips. Hex reverts to the previous version automatically only when the promoted model is significantly worse, logging why. (M7)
- **AC-15** Tidy suggests removing fillers, repeats and abandoned self-corrections. It never adds content, and a model edit that adds a word is dropped. (M8)
- **AC-16** Tidy learns which habits you accept removing: a phrase you mostly accept is suggested, and one you mostly skip isn't. (M8)
- **AC-17** Tidy runs only when you ask, after the paste. It applies only accepted edits, only to the pasted range, as one undoable edit. It changes nothing when the text moved. The speech model's labels stay what was heard. (M8)

### Guardrails
- **G-1 Nothing Vitalize.** No Vitalize Linear, Slack, repositories, data, agents or tickets, in planning or in code. *Check:* `git diff origin/dev...HEAD | rg -i vitalize` → only this guardrail's own line. No Linear or Slack calls were made for this goal.
- **G-2 This planning round writes only plan Markdown.** *Check:* `git diff --name-only origin/dev...HEAD` → only `docs/plans/self-healing/**.md`.
- **G-3 Voice data and text stay on the Mac,** with one exception: the M6 training upload. It needs an explicit `--allow-upload` flag and deletes the remote copy after the run. *Check:* `rg -n 'requests|httpx|urllib|modal\.' healing/` → only under `healing/remote/` and `healing/healing/sets/`. A socket-blocking pytest fixture covers every other command, so a dependency's network use fails too. A test shows the upload refuses without the flag. NeMo's logs on the remote are scrubbed of transcript text (T6.2).
- **G-4 Hex's privacy invariants hold:**
  - no dictated or observed text in `live.ndjson` or `process.log`;
  - History stays the only Hex store of text;
  - correction reads skip secure fields and Hex's own pid.

  *Check:* the existing log and accessibility tests, plus the new ones named in M2.
- **G-5 New Hex behaviour lands in Labs, off by default** (AGENTS.md). *Check:* the `labs` tests assert the default is off and the feature is gated where its work starts.
- **G-6 The collector never writes to Hex's data folder.** It only runs `hex history` (read-only) and copies files. *Check:* the collector test runs against a read-only fixture data root.
- **G-7 No releases and no `production` moves** (release freeze, Kit, 2026-10-22). PRs target `dev`. *Check:* no `release.yml` dispatch, and `git log origin/production` is unchanged.
- **G-8 Baseline:** never push to `main` or `production`, never track `.ansh/`, nothing on the autonomy never-do list.
- **G-9 A change to what the window draws regenerates the goldens in the same PR** (AGENTS.md). *Check:* `tests/golden/` changes alongside any `src/rune_ui` change.
- **G-10 The frozen golden set never changes after it is frozen.** *Check:* `healing split verify` compares its manifest hash to the one pinned in the registry.

### Exit criteria (stop and replan instead of improvising)
- **X-1** M0 finds that no candidate architecture survives fine-tune → merge → convert → Q8_0 → import → WER parity. Tier 2 stops (M6 and M7 are replanned). Tier 1 and M1–M5 continue.
- **X-2** The kept audio is not the model input and can't be made so inside the lab. Replan M1.
- **X-3** In T2.4's scripted protocol, Hex captures fewer than half of the **seeded edits** across the matrix apps. That is edit recall, not the share of pastes merely observed. Replan M2 toward an explicit "correct this" edit in History.
- **X-4** A change needs a store or CLI shape this plan's specs don't list; a check can't be made red first; a guardrail is about to be crossed; or the code contradicts a premise here.
- **X-5** The standard budget: 3 rounds per task, one replan, 2 goal evaluations.
- **X-6** GPU-provider credentials are missing. Park the M6 remote runner and keep everything else moving.

## Feature-wide edge cases (each owned by a task)
- **Weak labels are the current model's own output.** Scoring on them rewards the model that produced them. Test sets therefore use only strong or verified references (T3.1, T3.2).
- **After a promotion, `raw_text` comes from the new model.** Every clip records its `model` so labels can be traced (T1.1).
- **Change-of-mind edits must never become references.** The reference is `raw_text` with only the acoustic edits applied (T2.4).
- **Mode processing (rewrite, transforms, replacements) means the user edited `final_text`, not `raw_text`.** Edits are mapped back by alignment. Entries changed by an OpenCode rewrite are skipped at first (T2.4, D-7).
- **The same phrase said 50 times.**
  - Clusters cap audio in storage (T1.3).
  - Clusters are split as a unit, so they never leak across test and train (T3.2).
  - Clusters are weighted down in training (T6.1).
- **History may expire an entry before collection.** The collector runs every 6 h against a 7-day window. A gap shows up in `healing status` (T1.4).
- **Hex edits History after the fact** (Transcribe again → Replace adds `versions`). The collector keeps the version that matches the audio's model (T1.2).
- **Q8_0 can wash out a small LoRA delta.** The gate scores the quantized GGUF (T6.3, T7.1).
- **A promoted model can drift on new vocabulary.** The rolling holdout catches it; auto-rollback acts on it (T7.3).
- **Clock changes and DST.** Every timestamp is UTC milliseconds from Hex. Sessions are cut on gaps, not on calendar days (T3.2).
- **Hex rewrites text as it pastes** (join, smart spacing). The watcher compares against the text actually inserted (T2.1).
- **Chat apps send within seconds.** The watcher reads on value change and focus-out, not only at +30 s (T2.2).
- **Small test sets are noise.** About 1,000 words gives roughly ±1 pt per model. Gates use paired bootstrap intervals and a minimum word count, never point margins (T7.1).
- **Formatting regressions are invisible to WER.** The replay data keeps punctuation and casing, and a formatting error rate gates (T6.1, T7.1).
- **Imports don't stream.** Promoting without T7.4 would silently slow dictation and disable Live text.
- **Mode vocabularies take bias slots before the global list.** Ranking and the A/B count per mode (T4.2).
- **v0 drifts if Hex's catalog updates.** v0 is pinned by SHA-256, not by catalog id (T5.1).
- **Every promotion imports about 700 MB.** Old imports are removed with `hex model remove`, keeping current, the previous 2 and v0 (T7.2).

## Why a rolling holdout?

You asked for another opinion. Codex isn't installed on this Mac, so a fresh-context reviewer
gave it instead. It had none of my reasoning, but it is the same model family, so it isn't
cross-model. Both views are below, and the plan follows the reconciled version.

**Reconciled definition (what the plan now uses).** The rolling holdout is:
- **Fresh:** sessions after the parent version's snapshot cut-off. Neither compared model trained
  on them, and the set doesn't grow staler as the dataset grows. "Newest 20% of all sessions"
  would eventually cover weeks.
- **Random:** clips sampled at random for review, not "clips the current model got wrong". That
  rules out regression to the mean.
- **Verified:** references you confirmed, not the current model's own output.
- **Paired:** candidate and current are scored on the same clips in the same run. The gate is a
  paired bootstrap interval, not a point margin.

**The reviewer's view.** For:
- it answers the deployment question;
- it tracks drift that a frozen set and a public set can't;
- it costs little.

Against, as originally written:
- its contents change every round;
- it is small enough that 0.3-point margins are noise;
- strong-only clips are chosen by the incumbent's errors and partly labelled by its output;
- a growing "newest 20%" blunts its own drift signal;
- clusters the current model trained on can leak in.

Its verdict: the right primary gate *if* defined as above, with the golden set as a gating
non-inferiority check. Both are now in the plan.

**My original reasons, still true:**

- **It is the deployment question.** A model trained on the past and asked about the future is
  exactly how a promoted model is used. A random split lets a test clip have a near-twin from
  next week in training. Because you repeat phrases, that inflates every score.
- **It tracks drift. The golden set can't.** New jargon, a new microphone or a new project show up
  only in recent sessions. A 200-clip frozen set gets staler every week. Its job is the opposite:
  scores you can compare across rounds.
- **It is the only set that contains your newest corrections.** Those are the errors the model
  should have just learned. If rolling WER doesn't improve, training didn't help where it matters.
- **It costs little.** It is recomputed from the dataset each round, needs no labelling, and its
  leakage rule is the same cluster-disjoint split the others use.

The trade-off: rolling scores move between rounds because the set changes. That is why the gate
compares the candidate with the current model **on the same rolling set in the same run**, never
with last week's number.

**The cost it adds:** about 50 random clips to verify per round, roughly 10 minutes in
`healing review`. Without verified clips the rolling set is empty and the gate refuses to promote.

## Plan challenge

The plan was challenged on 2026-11-13 by a fresh-context Claude Opus reviewer: 16 P1 and 5 P2
findings. Codex was unavailable (no binary). Every P1 was folded in. What changed and what was
rejected is in [decisions.md](decisions.md#challenge).

## Decisions and the work trail

[decisions.md](decisions.md) lists every agent default (D-n) and what you could choose instead.
