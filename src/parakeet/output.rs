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
                    watch.recorded(corrections, history, entry_id);
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
