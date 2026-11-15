//! The dictation output stage: finished jobs leave the inference and
//! processing workers in order and are pasted (or copied, or printed),
//! recorded in History, and handed to Labs › Learn from corrections. One
//! thread, one [`Paster`], so a paste never races another.

use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, mpsc};
use std::time::Instant;

use color_eyre::eyre::Result;

use super::{
    CompletedTranscript, OutputJob, PasteKind, TranscriptionTarget, WorkerEvent, WorkerState,
};
use crate::dictation::correction_watch::{Handoff, Live, Watch};
use crate::history::{History, HistoryDraft, HistoryKind};
use crate::meeting;
use crate::paste::{PasteMode, Paster};
use crate::suppression::InputActivity;

#[derive(Default)]
pub(super) struct OrderedOutputs {
    pub(super) waiting: BTreeMap<u64, OutputJob>,
    next_sequence: u64,
}

impl OrderedOutputs {
    pub(super) fn push(&mut self, job: OutputJob) -> Vec<OutputJob> {
        if job.sequence() < self.next_sequence {
            return Vec::new();
        }
        self.waiting.insert(job.sequence(), job);
        let mut ready = Vec::new();
        while let Some(job) = self.waiting.remove(&self.next_sequence) {
            ready.push(job);
            self.next_sequence += 1;
        }
        ready
    }
}

pub(super) fn run_output_worker(
    jobs: Receiver<OutputJob>,
    activity: InputActivity,
    history: Option<History>,
    state: &Mutex<WorkerState>,
    events: &mpsc::Sender<WorkerEvent>,
) {
    let mut paster = Paster::new(activity);
    let mut last_transcript = None;
    let mut ordered = OrderedOutputs::default();
    while let Ok(job) = jobs.recv() {
        if matches!(job, OutputJob::PreparePaste) {
            paster.prepare();
            continue;
        }
        for job in ordered.push(job) {
            let event = finish_output(
                job,
                // What the paste inserted is what the watcher is told.
                &mut |text, mode| {
                    paster.paste(text, mode)?;
                    Ok(paster.inserted())
                },
                &mut last_transcript,
                history.as_ref(),
                &mut Live,
            );
            let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
            match &event {
                WorkerEvent::Completed { job_id, .. } => {
                    state.jobs.remove(job_id);
                }
                WorkerEvent::Pasted { .. } => {
                    state.pending_pastes = state.pending_pastes.saturating_sub(1);
                }
                WorkerEvent::ModelFailed(_) | WorkerEvent::Stage { .. } => {}
            }
            drop(state);
            if events.send(event).is_err() {
                return;
            }
        }
    }
}

pub(super) fn finish_output(
    job: OutputJob,
    paste: &mut impl FnMut(&str, PasteMode) -> Result<String>,
    last_transcript: &mut Option<String>,
    history: Option<&History>,
    corrections: &mut impl Handoff,
) -> WorkerEvent {
    match job {
        OutputJob::PreparePaste => unreachable!("paste preparation bypasses ordered output"),
        OutputJob::Completed {
            job_id,
            control,
            target,
            result,
        } => {
            let result = *result;
            let processing = result
                .as_ref()
                .ok()
                .and_then(|result| result.processing.clone());
            let result = result.and_then(|mut completed| {
                let paste_started = Instant::now();
                let mut sent = false;
                let mut watch = Watch::default();
                if !completed.nothing && !completed.text.trim().is_empty() {
                    match target {
                        TranscriptionTarget::Paste
                        | TranscriptionTarget::CopyOnlyPaste
                        | TranscriptionTarget::RawCopyOnlyPaste
                        | TranscriptionTarget::RawPaste => {
                            // Labs › Copy dictation only, read as the output
                            // commits: dictations go to the clipboard
                            // unpasted; Paste last dictation still pastes.
                            // A rescued microphone-loss dictation is always
                            // copied: the disruption may have taken focus.
                            let mode = if crate::labs::copy_only()
                                || matches!(
                                    target,
                                    TranscriptionTarget::CopyOnlyPaste
                                        | TranscriptionTarget::RawCopyOnlyPaste
                                )
                            {
                                PasteMode::CopyOnly
                            } else if completed.timings.held_ms.is_some_and(|held_ms| {
                                crate::dictation::yolo_mode::sends(
                                    held_ms,
                                    completed.application.as_deref(),
                                )
                            }) {
                                // Labs › YOLO mode, read as the
                                // output commits.
                                PasteMode::Send
                            } else {
                                PasteMode::Continue
                            };
                            // Labs › Return to app: the app the capture
                            // started in comes back first. When it can't,
                            // nothing is pasted; the text stays for Paste
                            // last dictation, and History keeps only pasted
                            // output, as on any failed paste.
                            if matches!(mode, PasteMode::Continue | PasteMode::Send)
                                && let Err(error) =
                                    crate::paste::return_to_app::bring_to_front(completed.return_to)
                            {
                                *last_transcript = Some(completed.text.clone());
                                tracing::warn!(
                                    %error,
                                    "a dictation found another app in front and couldn't return to its own"
                                );
                                return Err(error.to_string());
                            }
                            // Labs › Learn from corrections: not a Send or copy only.
                            if matches!(mode, PasteMode::Continue) {
                                watch = Watch::before(corrections, history);
                            }
                            // A paste that fails (another app changed the
                            // clipboard while Hex saved it) keeps the text
                            // for Paste last dictation, as above, so the
                            // dictation isn't lost.
                            match paste(&completed.text, mode) {
                                Ok(inserted) => watch.pasted(inserted),
                                Err(error) => {
                                    *last_transcript = Some(completed.text.clone());
                                    return Err(error.to_string());
                                }
                            }
                            if matches!(mode, PasteMode::Send) {
                                sent = send_unless_escaped(
                                    completed.timings.send_gate,
                                    Instant::now(),
                                    paste,
                                );
                            }
                        }
                        TranscriptionTarget::Service
                        | TranscriptionTarget::Print
                        | TranscriptionTarget::RawPrint => {}
                    }
                    if matches!(
                        target,
                        TranscriptionTarget::Paste
                            | TranscriptionTarget::CopyOnlyPaste
                            | TranscriptionTarget::RawCopyOnlyPaste
                            | TranscriptionTarget::RawPaste
                    ) {
                        *last_transcript = Some(completed.text.clone());
                    }
                    let entry_id = history.and_then(|history| {
                        record_history(history, target, sent, &mut completed)
                    });
                    watch.recorded(corrections, history, entry_id, &completed.text);
                }
                tracing::info!(
                    audio_ms = completed.timings.audio_ms,
                    queue_ms = completed.timings.queue_ms,
                    prepare_ms = completed.timings.prepare_ms,
                    inference_ms = completed.timings.inference_ms,
                    paste_ms = paste_started.elapsed().as_millis(),
                    total_ms = completed.timings.total_started.elapsed().as_millis(),
                    "dictation pipeline completed"
                );
                Ok(completed.text)
            });
            if matches!(
                target,
                TranscriptionTarget::Print | TranscriptionTarget::RawPrint
            ) {
                control.answer_print(match &result {
                    Ok(text) if text.trim().is_empty() => {
                        crate::control::PrintedTranscript::NoSpeech
                    }
                    Ok(text) => crate::control::PrintedTranscript::Text(text.clone()),
                    Err(error) => crate::control::PrintedTranscript::Failed(error.clone()),
                });
            }
            WorkerEvent::Completed {
                job_id,
                target,
                result,
                processing,
            }
        }
        OutputJob::Paste { kind, .. } => {
            let result = match kind {
                PasteKind::LastTranscript => last_transcript
                    .as_deref()
                    .ok_or_else(|| "no previous transcript is available".to_string())
                    .and_then(|text| {
                        paste(text, PasteMode::Continue).map_err(|error| error.to_string())?;
                        Ok(text.to_string())
                    }),
                PasteKind::MeetingDelta => {
                    paste_meeting_delta(paste).map_err(|error| error.to_string())
                }
            };
            WorkerEvent::Pasted { kind, result }
        }
    }
}

/// Labs › YOLO mode, after a `Send` paste landed at `pasted_at`: presses
/// Return unless Escape came after the release. Returns whether it sent; the
/// text stays pasted either way.
pub(super) fn send_unless_escaped(
    gate: crate::dictation::yolo_mode::SendGate,
    pasted_at: Instant,
    paste: &mut impl FnMut(&str, PasteMode) -> Result<String>,
) -> bool {
    if !gate.wait(pasted_at) {
        tracing::info!("YOLO mode pasted without Return: Escape after the release");
        return false;
    }
    match paste("", PasteMode::Return) {
        Ok(_) => {
            tracing::info!(
                after_submit_ms = gate.submitted_at().elapsed().as_millis(),
                "YOLO mode pressed Return"
            );
            true
        }
        Err(error) => {
            tracing::warn!(%error, "YOLO mode could not press Return");
            false
        }
    }
}

/// Record one successfully pasted result. History failures must never fail
/// the paste that already happened.
pub(super) fn record_history(
    history: &History,
    target: TranscriptionTarget,
    sent: bool,
    completed: &mut CompletedTranscript,
) -> Option<u64> {
    let kind = match target {
        TranscriptionTarget::Paste if sent => HistoryKind::Send,
        TranscriptionTarget::Paste | TranscriptionTarget::CopyOnlyPaste => HistoryKind::Dictation,
        TranscriptionTarget::RawPaste | TranscriptionTarget::RawCopyOnlyPaste => HistoryKind::Raw,
        TranscriptionTarget::Print | TranscriptionTarget::RawPrint => HistoryKind::Printed,
        TranscriptionTarget::Service => return None,
    };
    let (model, model_id) = crate::history::model_fields(&completed.selection);
    let draft = HistoryDraft {
        kind,
        raw_text: completed.raw.clone(),
        final_text: completed.text.clone(),
        application: completed.application.clone(),
        processing: completed.processing.clone(),
        audio_ms: completed.timings.audio_ms,
        inference_ms: completed.timings.inference_ms as u64,
        total_ms: completed.timings.total_started.elapsed().as_millis() as u64,
        recording: completed.recording.take(),
        model,
        model_id,
    };
    history
        .record(draft)
        .inspect_err(|error| tracing::warn!(%error, "could not record dictation history"))
        .ok()
        .flatten()
}

fn paste_meeting_delta(
    paste: &mut impl FnMut(&str, PasteMode) -> Result<String>,
) -> Result<String> {
    meeting::consume_active_or_latest_delta(|text| paste(text, PasteMode::Standalone).map(drop))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use color_eyre::eyre::eyre;

    use super::super::{
        DictationJobId, DictationWorker, JobControl, JobTimings, MAX_PENDING_OUTPUTS,
    };
    use super::*;

    /// A handoff with the lab off: nothing is read or submitted.
    struct NoHandoff;

    impl crate::dictation::correction_watch::Handoff for NoHandoff {
        fn read(&mut self) -> Option<crate::dictation::correction_watch::Pending> {
            None
        }
        fn submit(&mut self, _: crate::dictation::correction_watch::PasteTarget) {
            panic!("a target was submitted with the lab off");
        }
    }

    /// A handoff with the lab on, over a real retained element.
    #[derive(Default)]
    struct Recording {
        reads: usize,
        submitted: Vec<(u64, String)>,
    }

    /// What the paster adds a leading space to: the target carries what
    /// landed, not the dictation's text.
    const INSERTED: &str = " hello there";

    impl crate::dictation::correction_watch::Handoff for Recording {
        fn read(&mut self) -> Option<crate::dictation::correction_watch::Pending> {
            self.reads += 1;
            Some(crate::dictation::correction_watch::Pending::new(
                1,
                crate::accessibility::RetainedElement::for_test(),
                None,
                None,
            ))
        }
        fn submit(&mut self, target: crate::dictation::correction_watch::PasteTarget) {
            self.submitted.push((target.entry_id, target.inserted));
        }
    }

    fn completed_dictation(text: &str) -> CompletedTranscript {
        CompletedTranscript {
            text: text.into(),
            nothing: false,
            raw: text.into(),
            application: None,
            return_to: None,
            timings: JobTimings {
                total_started: Instant::now(),
                queue_ms: 0,
                audio_ms: 2_000,
                prepare_ms: 0,
                inference_ms: 0,
                held_ms: None,
                send_gate: crate::dictation::yolo_mode::SendGate::open(),
            },
            processing: None,
            recording: None,
            selection: Default::default(),
        }
    }

    fn run_output(
        target: TranscriptionTarget,
        paste: &mut impl FnMut(&str, PasteMode) -> Result<String>,
        history: Option<&History>,
        handoff: &mut Recording,
    ) {
        let mut last_transcript = None;
        finish_output(
            OutputJob::Completed {
                job_id: DictationJobId(1),
                control: Arc::new(JobControl::default()),
                target,
                result: Box::new(Ok(completed_dictation("hello there"))),
            },
            paste,
            &mut last_transcript,
            history,
            handoff,
        );
    }

    #[test]
    fn one_successful_paste_submits_one_target_with_the_history_id_and_inserted_text() {
        let directory =
            std::env::temp_dir().join(format!("hex-parakeet-corrections-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let history = History::new(crate::history::HistoryStore::open(
            directory.join("history.json"),
            crate::history::HistoryRetention::Week,
            crate::history::now_ms(),
        ));
        let mut handoff = Recording::default();
        let mut paste = |_: &str, _| Ok(INSERTED.to_string());
        run_output(
            TranscriptionTarget::Paste,
            &mut paste,
            Some(&history),
            &mut handoff,
        );
        let id = history.recent(1)[0].id;
        assert_eq!(handoff.reads, 1);
        assert_eq!(handoff.submitted, [(id, INSERTED.to_string())]);

        // Copy only never pastes, a failed paste submits nothing, and
        // without a History entry there is nothing to attach to.
        let mut handoff = Recording::default();
        run_output(
            TranscriptionTarget::CopyOnlyPaste,
            &mut paste,
            Some(&history),
            &mut handoff,
        );
        let mut failing = |_: &str, _| Err(eyre!("clipboard changed"));
        run_output(
            TranscriptionTarget::Paste,
            &mut failing,
            Some(&history),
            &mut handoff,
        );
        run_output(TranscriptionTarget::Paste, &mut paste, None, &mut handoff);
        assert_eq!(handoff.reads, 1, "copy only and no History read nothing");
        assert!(handoff.submitted.is_empty());

        // History that keeps nothing: the field is not even read.
        history
            .set_retention(crate::history::HistoryRetention::Off)
            .unwrap();
        run_output(
            TranscriptionTarget::Paste,
            &mut paste,
            Some(&history),
            &mut handoff,
        );
        assert_eq!(handoff.reads, 1);
        assert!(handoff.submitted.is_empty());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_failed_paste_keeps_the_dictation_for_paste_last() {
        let text = "the dictation whose paste failed";
        let mut last_transcript = Some("an older dictation".to_string());
        let event = finish_output(
            OutputJob::Completed {
                job_id: DictationJobId(1),
                control: Arc::new(JobControl::default()),
                target: TranscriptionTarget::Paste,
                result: Box::new(Ok(CompletedTranscript {
                    text: text.into(),
                    nothing: false,
                    raw: text.into(),
                    application: None,
                    return_to: None,
                    timings: JobTimings {
                        total_started: Instant::now(),
                        queue_ms: 0,
                        audio_ms: 2_000,
                        prepare_ms: 0,
                        inference_ms: 0,
                        held_ms: None,
                        send_gate: crate::dictation::yolo_mode::SendGate::open(),
                    },
                    processing: None,
                    recording: None,
                    selection: Default::default(),
                })),
            },
            &mut |_, _| Err(eyre!("clipboard changed while Hex was preserving it")),
            &mut last_transcript,
            None,
            &mut NoHandoff,
        );

        assert!(matches!(
            event,
            WorkerEvent::Completed { result: Err(_), .. }
        ));
        assert_eq!(last_transcript.as_deref(), Some(text));
    }

    #[test]
    fn a_salvaged_dictation_is_copied_and_kept_for_paste_last_and_history() {
        // A normal salvage is processed before it is copied; a raw one is
        // copied as heard and recorded as raw history. Neither is pasted.
        for (target, kind) in [
            (TranscriptionTarget::CopyOnlyPaste, HistoryKind::Dictation),
            (TranscriptionTarget::RawCopyOnlyPaste, HistoryKind::Raw),
        ] {
            let directory = std::env::temp_dir().join(format!(
                "hex-parakeet-salvage-{}-{target:?}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&directory);
            std::fs::create_dir_all(&directory).unwrap();
            let history = History::new(crate::history::HistoryStore::open(
                directory.join("history.json"),
                crate::history::HistoryRetention::Week,
                crate::history::now_ms(),
            ));
            let text = "the part of the dictation the microphone heard";
            let mut pasted = None;
            let mut last_transcript = None;
            let event = finish_output(
                OutputJob::Completed {
                    job_id: DictationJobId(1),
                    control: Arc::new(JobControl::default()),
                    target,
                    result: Box::new(Ok(CompletedTranscript {
                        text: text.into(),
                        nothing: false,
                        raw: text.into(),
                        application: Some("Call".into()),
                        return_to: None,
                        timings: JobTimings {
                            total_started: Instant::now(),
                            queue_ms: 0,
                            audio_ms: 4_000,
                            prepare_ms: 0,
                            inference_ms: 0,
                            held_ms: None,
                            send_gate: crate::dictation::yolo_mode::SendGate::open(),
                        },
                        processing: None,
                        recording: None,
                        selection: Default::default(),
                    })),
                },
                &mut |pasted_text, mode| {
                    pasted = Some((pasted_text.to_string(), mode));
                    Ok(pasted_text.to_string())
                },
                &mut last_transcript,
                Some(&history),
                &mut NoHandoff,
            );

            assert!(matches!(
                event,
                WorkerEvent::Completed {
                    result: Ok(ref completed),
                    ..
                } if completed.as_str() == text
            ));
            let (pasted_text, mode) = pasted.expect("the transcript reaches the clipboard");
            assert_eq!(pasted_text, text);
            assert!(
                matches!(mode, PasteMode::CopyOnly),
                "{target:?}: an interrupted dictation is copied, never pasted"
            );
            assert_eq!(last_transcript.as_deref(), Some(text));
            let entries = history.recent(1);
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].kind, kind, "{target:?}");
            assert_eq!(entries[0].final_text, text);
            assert_eq!(entries[0].application.as_deref(), Some("Call"));
            let _ = std::fs::remove_dir_all(&directory);
        }
    }

    /// Labs › YOLO mode: Escape after the release keeps the paste and skips
    /// the Return, so History records a dictation rather than a send.
    #[test]
    fn escape_after_release_pastes_a_yolo_dictation_without_return() {
        use crate::dictation::yolo_mode::{ESCAPE_WINDOW, SendGate};
        use std::sync::atomic::{AtomicU64, Ordering};
        static ESCAPES: AtomicU64 = AtomicU64::new(0);
        let submitted = Instant::now() - ESCAPE_WINDOW;
        let presses = std::cell::RefCell::new(Vec::new());
        let mut paste = |text: &str, mode| {
            presses
                .borrow_mut()
                .push((text.to_owned(), matches!(mode, PasteMode::Return)));
            Ok(String::new())
        };

        let gate = SendGate::on(&ESCAPES, submitted);
        ESCAPES.fetch_add(1, Ordering::AcqRel);
        assert!(!send_unless_escaped(gate, submitted, &mut paste));
        assert!(presses.borrow().is_empty(), "no Return after an Escape");

        let gate = SendGate::on(&ESCAPES, submitted);
        assert!(send_unless_escaped(gate, submitted, &mut paste));
        assert_eq!(*presses.borrow(), [(String::new(), true)]);

        let failing = SendGate::on(&ESCAPES, submitted);
        assert!(!send_unless_escaped(failing, submitted, &mut |_, _| Err(
            eyre!("Return failed")
        )));

        let directory =
            std::env::temp_dir().join(format!("hex-parakeet-yolo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let history = History::new(crate::history::HistoryStore::open(
            directory.join("history.json"),
            crate::history::HistoryRetention::Week,
            crate::history::now_ms(),
        ));
        for sent in [false, true] {
            let mut completed = CompletedTranscript {
                text: "a long message".into(),
                nothing: false,
                raw: "a long message".into(),
                application: None,
                return_to: None,
                timings: JobTimings {
                    total_started: Instant::now(),
                    queue_ms: 0,
                    audio_ms: 3_000,
                    prepare_ms: 0,
                    inference_ms: 0,
                    held_ms: Some(3_000),
                    send_gate: SendGate::open(),
                },
                processing: None,
                recording: None,
                selection: Default::default(),
            };
            record_history(&history, TranscriptionTarget::Paste, sent, &mut completed);
        }
        let kinds: Vec<_> = history
            .recent(2)
            .into_iter()
            .map(|entry| entry.kind)
            .collect();
        assert!(kinds.contains(&HistoryKind::Dictation) && kinds.contains(&HistoryKind::Send));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn parallel_processing_results_are_released_in_submission_order() {
        let mut outputs = OrderedOutputs::default();
        let job = |sequence| OutputJob::Paste {
            sequence,
            kind: PasteKind::LastTranscript,
        };

        assert!(outputs.push(job(1)).is_empty());
        assert_eq!(
            outputs
                .push(job(0))
                .into_iter()
                .map(|job| job.sequence())
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            outputs
                .push(job(2))
                .into_iter()
                .map(|job| job.sequence())
                .collect::<Vec<_>>(),
            [2]
        );
    }

    #[test]
    fn ordered_waiting_outputs_remain_bounded_after_channel_drain() {
        let (output_sender, output_receiver) = mpsc::sync_channel(1);
        let (event_sender, events) = mpsc::channel();
        let worker = DictationWorker {
            engine: None,
            inference_jobs: None,
            output_jobs: Some(output_sender),
            events,
            event_sender,
            state: Arc::new(Mutex::new(WorkerState {
                next_sequence: 1,
                jobs: BTreeSet::from([DictationJobId(0)]),
                pending_pastes: 0,
                return_to: None,
                started_by_shortcut: false,
            })),
            inference_worker: None,
            processor_workers: Vec::new(),
            output_worker: None,
        };
        let mut ordered = OrderedOutputs::default();
        for _ in 1..MAX_PENDING_OUTPUTS {
            worker.paste_last().unwrap();
            assert!(ordered.push(output_receiver.try_recv().unwrap()).is_empty());
        }
        assert_eq!(worker.paste_last(), Err("dictation queue is full"));
        assert_eq!(worker.paste_meeting(), Err("dictation queue is full"));
        assert_eq!(ordered.waiting.len(), MAX_PENDING_OUTPUTS - 1);
        assert_eq!(
            worker.state.lock().unwrap().next_sequence,
            MAX_PENDING_OUTPUTS as u64
        );

        worker.state.lock().unwrap().jobs.remove(&DictationJobId(0));
        worker.paste_last().unwrap();
        assert_eq!(
            output_receiver.try_recv().unwrap().sequence(),
            MAX_PENDING_OUTPUTS as u64
        );
    }

    /// Only a plain pasted dictation is watched: a YOLO Send (its Return
    /// leaves the field), a copy only (nothing is pasted), and the pastes of
    /// the last transcript or a meeting delta never read or submit.
    #[test]
    fn a_yolo_send_a_copy_only_and_the_other_pastes_never_read_or_submit() {
        let history = crate::dictation::correction_watch::history_for_test();
        let mut handoff = Recording::default();
        let mut paste = |text: &str, _| Ok(text.to_string());

        crate::labs::apply(&crate::app_settings::LabsSettings {
            yolo_mode: true,
            ..Default::default()
        });
        let mut completed = completed_dictation("hello there");
        completed.timings.held_ms = Some(60_000);
        let mut last_transcript = None;
        finish_output(
            OutputJob::Completed {
                job_id: DictationJobId(1),
                control: Arc::new(JobControl::default()),
                target: TranscriptionTarget::Paste,
                result: Box::new(Ok(completed)),
            },
            &mut paste,
            &mut last_transcript,
            Some(&history),
            &mut handoff,
        );
        assert_eq!(history.recent(1)[0].kind, HistoryKind::Send, "it sent");

        for target in [
            TranscriptionTarget::CopyOnlyPaste,
            TranscriptionTarget::RawCopyOnlyPaste,
        ] {
            run_output(target, &mut paste, Some(&history), &mut handoff);
        }
        for (sequence, kind) in [(2, PasteKind::LastTranscript), (3, PasteKind::MeetingDelta)] {
            finish_output(
                OutputJob::Paste { sequence, kind },
                &mut paste,
                &mut last_transcript,
                Some(&history),
                &mut handoff,
            );
        }
        assert_eq!(handoff.reads, 0);
        assert!(handoff.submitted.is_empty());
    }
}
