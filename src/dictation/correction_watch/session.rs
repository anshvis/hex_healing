//! One watched paste: when to read the field, which read to keep, and what
//! to report. Pure over a [`Field`], so the rules are tested without
//! Accessibility.
//!
//! The model. A [`State`] is a phase (`Watching`, or `Sending` after a
//! Return) and what is held: the last read that still held the paste, or why
//! none did. [`State::advance`] is the whole logic, pure: from the state and
//! an [`Input`] (a Return, a failed read, or a [`Reading`]: is the field
//! blank, did it shrink, what does it hold of the paste) to the next state
//! and, when the watch is over, what it ends with. The [`Session`] only turns
//! field reads into inputs.
//!
//! - `Watching`: a read that holds the paste is kept; one that holds other
//!   text drops what was kept. A blank read after something was kept waits
//!   [`CLEAR_GRACE`] (a Return the listener counts a moment late converts
//!   it); still blank after that, it is the field cleared with no Return: a
//!   send button or delete-all, which cannot be told apart, so it is
//!   `cleared_before_read` (training-safe: a send button without a Return
//!   loses the observation). A blank read before anything was kept is only
//!   noted (the paste may not have landed).
//! - `Sending { kept, until }`: a Return enters it, with what was kept at
//!   that moment, before its read. The tap counts the Return before the app
//!   handles it, so the app may clear the field a tick later. For
//!   [`SEND_WINDOW`], a read that shows the field blank, shorter, or holding
//!   text that is not the paste is that clear: the watch ends with `kept`. A
//!   read that still holds the paste, changed or not (a notes app's newline,
//!   then an edit), is an ordinary read, and after the window the watch is
//!   `Watching` again.
//!
//! Where the paste is in the field is the [`Tracker`]'s (see `locate`): an
//! edit is stored only from a region it is following.

use std::time::{Duration, Instant};

use super::locate::{self, Alignment, Tracker};
use crate::accessibility::{FieldText, ReadError};
use crate::history::HistoryCorrection;

/// How long a paste is watched.
pub const WATCH: Duration = Duration::from_secs(30);
/// A changed value is read at most this often. It is under the watcher's
/// tick, so every tick that saw a change reads once, on that tick: the field
/// is read at the tick after each keystroke, and a chat app that clears it
/// on Return leaves the last read one tick behind the typing at worst, not a
/// keystroke behind it. A burst inside one tick is one read of the end state.
pub const READ_GAP: Duration = Duration::from_millis(40);
/// Reads per target that found the field changed; a read that finds it as
/// before, and the closing read, are not counted. A watch that sent a
/// Return is read regardless, so the clear after it is always seen.
pub const MAX_READS: usize = 200;
/// How long a blank field is waited on before it is called cleared with no
/// Return: two ticks, so a Return counted a recognition wake late converts it.
pub const CLEAR_GRACE: Duration = Duration::from_millis(100);
/// How long after a Return the field changing means it was sent.
pub const SEND_WINDOW: Duration = Duration::from_secs(1);
/// Returns per target that are read; a held Return key is not.
pub const MAX_RETURN_READS: usize = 20;
/// A focus change this soon after the paste is the paste's own.
pub const FOCUS_GRACE: Duration = Duration::from_millis(300);
/// Characters read from the field, whole or around the caret.
pub const READ_CHARS: usize = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Trigger {
    /// The read at the start of the watch, which sets the baseline.
    Start,
    ValueChanged,
    FocusOut,
    Return,
    Timeout,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::ValueChanged => "value_changed",
            Self::FocusOut => "focus_out",
            Self::Return => "return",
            Self::Timeout => "timeout",
        }
    }
}

/// Why no read held the pasted text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lost {
    /// A read failed (the field is gone, secure, too large, slow, or gave no
    /// value).
    Read(ReadError),
    SpanNotFound,
    ClearedBeforeRead,
    /// Nothing has been read yet (never what a watch ends with).
    NotRead,
}

impl Lost {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read(ReadError::ElementGone) => "element_gone",
            Self::Read(ReadError::Secure) => "secure",
            Self::Read(ReadError::TooLarge) => "too_large",
            Self::Read(ReadError::Timeout) => "read_timeout",
            Self::Read(ReadError::Failed) => "read_failed",
            Self::NotRead => "read_timeout",
            Self::SpanNotFound => "span_not_found",
            Self::ClearedBeforeRead => "cleared_before_read",
        }
    }

    /// This reason, unless `held` is already a stronger one: a field that was
    /// cleared outranks one that held other text, which outranks a read that
    /// failed.
    fn against(self, held: &Finished) -> Self {
        match held {
            Err(old) if old.priority() > self.priority() => *old,
            _ => self,
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::ClearedBeforeRead => 2,
            Self::SpanNotFound => 1,
            Self::Read(_) | Self::NotRead => 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Outcome {
    Unchanged,
    /// The field's aligned span, at most `locate::span_cap(inserted length)`
    /// characters.
    Edited(String),
    Lost(Lost),
}

/// What the watcher saw of one paste.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrectionObservation {
    pub outcome: Outcome,
    pub trigger: Trigger,
    /// Milliseconds from the paste to the read this comes from.
    pub observed_at_ms: u64,
}

impl CorrectionObservation {
    /// The History record. The one place the outcome becomes the strings
    /// `history.json` keeps, which older builds read: `unchanged`, `edited`,
    /// or `lost` with its reason.
    pub fn into_history(self) -> HistoryCorrection {
        let (outcome, reason, edited_text) = match self.outcome {
            Outcome::Unchanged => ("unchanged", None, None),
            Outcome::Edited(span) => ("edited", None, Some(span)),
            Outcome::Lost(lost) => ("lost", Some(lost.as_str().to_string()), None),
        };
        HistoryCorrection {
            outcome: outcome.to_string(),
            reason,
            observed_at_ms: self.observed_at_ms,
            edited_text,
        }
    }
}

/// The field being watched.
pub trait Field {
    fn read(&self) -> Result<FieldText, ReadError>;
}

impl Field for crate::accessibility::RetainedElement {
    fn read(&self) -> Result<FieldText, ReadError> {
        crate::accessibility::read_value_bounded(self, READ_CHARS)
    }
}

/// What woke the watcher (an `Event`) is not why a read was kept (a
/// [`Trigger`]): a tick can read a change the observer reported a moment
/// ago, and a Return event reads like any other. Only triggers are logged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    ValueChanged,
    FocusOut,
    Return,
    Tick,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Step {
    Wait,
    /// Stop without a result: the gate closed.
    Drop,
    Done(CorrectionObservation),
}

/// What the watch ends with: the read it kept, or why it has none.
type Finished = Result<CorrectionObservation, Lost>;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Phase {
    Watching,
    /// A Return was seen, with `kept` held at that moment.
    Sending {
        until: Instant,
        kept: Option<CorrectionObservation>,
    },
}

/// What one read showed of the field.
struct Reading {
    trigger: Trigger,
    ms: u64,
    blank: bool,
    /// Shorter than the read before.
    shrank: bool,
    /// What it holds of the paste.
    outcome: Alignment,
}

enum Input {
    Return,
    Failed(ReadError),
    Read(Reading),
}

/// The watch's state. `held` is what the watch would end with now.
#[derive(Clone, Debug, Eq, PartialEq)]
struct State {
    phase: Phase,
    held: Finished,
    /// Since when the field has been blank with something kept.
    cleared_since: Option<Instant>,
}

impl State {
    fn new() -> Self {
        Self {
            phase: Phase::Watching,
            held: Err(Lost::NotRead),
            cleared_since: None,
        }
    }

    fn best(&self) -> Option<&CorrectionObservation> {
        self.held.as_ref().ok()
    }

    fn sending(&self, now: Instant) -> bool {
        matches!(&self.phase, Phase::Sending { until, .. } if now <= *until)
    }

    /// The whole model: the next state, and what the watch ends with if it
    /// is over.
    fn advance(mut self, input: Input, now: Instant) -> (Self, Option<Finished>) {
        if !self.sending(now) {
            self.phase = Phase::Watching;
        }
        let finished = match input {
            Input::Return => {
                self.phase = Phase::Sending {
                    until: now + SEND_WINDOW,
                    kept: self.best().cloned(),
                };
                self.cleared_since = None;
                None
            }
            Input::Failed(error) => {
                // A failed read never erases what was kept.
                if self.best().is_none() {
                    self.held = Err(Lost::Read(error).against(&self.held));
                }
                (error == ReadError::ElementGone).then(|| self.held.clone())
            }
            Input::Read(reading) => self.read(reading, now),
        };
        if let Some(finished) = &finished {
            self.held = finished.clone();
        }
        (self, finished)
    }

    fn read(&mut self, reading: Reading, now: Instant) -> Option<Finished> {
        let (sending, kept) = match &self.phase {
            Phase::Sending { kept, .. } => (true, kept.clone()),
            Phase::Watching => (false, None),
        };
        if reading.blank {
            return match (sending, kept, self.best()) {
                (true, kept, _) => Some(kept.ok_or(Lost::ClearedBeforeRead)),
                (false, _, Some(_)) => match self.cleared_since {
                    Some(since) if now.saturating_duration_since(since) >= CLEAR_GRACE => {
                        Some(Err(Lost::ClearedBeforeRead))
                    }
                    Some(_) => None,
                    None => {
                        self.cleared_since = Some(now);
                        None
                    }
                },
                (false, _, None) => {
                    self.held = Err(Lost::ClearedBeforeRead.against(&self.held));
                    None
                }
            };
        }
        self.cleared_since = None;
        let outcome = match reading.outcome {
            Alignment::Unchanged => Some(Outcome::Unchanged),
            Alignment::Edited(span) => Some(Outcome::Edited(span)),
            Alignment::NotFound => None,
        };
        // After a Return, the field shorter or holding other text is the app
        // clearing it; the paste still there, changed or not, is not.
        if let Some(kept) = kept
            && (reading.shrank || outcome.is_none())
        {
            return Some(Ok(kept));
        }
        self.held = match outcome {
            Some(outcome) => Ok(CorrectionObservation {
                outcome,
                trigger: reading.trigger,
                observed_at_ms: reading.ms,
            }),
            None => Err(Lost::SpanNotFound.against(&self.held)),
        };
        None
    }
}

pub struct Session<F: Field> {
    field: F,
    started: Instant,
    tracker: Tracker,
    /// Reads that found the field changed (see [`MAX_READS`]).
    reads: usize,
    return_reads: usize,
    last_read: Option<Instant>,
    last_text: Option<String>,
    due: Option<Instant>,
    state: State,
}

impl<F: Field> Session<F> {
    pub fn new(field: F, started: Instant, tracker: Tracker) -> Self {
        Self {
            field,
            started,
            tracker,
            reads: 0,
            return_reads: 0,
            last_read: None,
            last_text: None,
            due: None,
            state: State::new(),
        }
    }

    #[cfg(test)]
    pub fn reads(&self) -> usize {
        self.reads
    }

    /// Starts the watch over from `tracker`, keeping its start time and read
    /// counts: Labs › Tidy after paste edited the paste in place (T8.5), so
    /// nothing held from before (a read of the tidy's own edit included) is
    /// the person's, and the next tick reads a new baseline.
    pub fn rebase(&mut self, tracker: Tracker) {
        self.tracker = tracker;
        self.state = State::new();
        self.last_read = None;
        self.last_text = None;
        self.due = None;
    }

    /// Advances on `event` at `now`. `gate` is asked before every read.
    pub fn step(&mut self, now: Instant, event: Event, gate: &dyn Fn() -> bool) -> Step {
        match event {
            Event::ValueChanged => {
                if self.reads >= MAX_READS && !self.state.sending(now) {
                    return Step::Wait;
                }
                let wait = self
                    .last_read
                    .map(|at| (at + READ_GAP).saturating_duration_since(now))
                    .filter(|wait| !wait.is_zero());
                match wait {
                    Some(wait) => self.due = Some(now + wait),
                    None => return self.read(now, Trigger::ValueChanged, gate),
                }
                Step::Wait
            }
            Event::Tick => {
                // The baseline: the paste's own change predates the observer.
                let trigger = if self.last_read.is_none() {
                    Some(Trigger::Start)
                } else if self.due.is_some_and(|due| due <= now) {
                    Some(Trigger::ValueChanged)
                } else {
                    None
                };
                if let Some(trigger) = trigger {
                    let step = self.read(now, trigger, gate);
                    if step != Step::Wait {
                        return step;
                    }
                }
                if now.saturating_duration_since(self.started) >= WATCH {
                    return self.finish(now, Trigger::Timeout, gate);
                }
                Step::Wait
            }
            Event::FocusOut => {
                if now.saturating_duration_since(self.started) < FOCUS_GRACE {
                    return Step::Wait;
                }
                self.finish(now, Trigger::FocusOut, gate)
            }
            Event::Return => {
                if self.return_reads >= MAX_RETURN_READS {
                    return Step::Wait;
                }
                self.return_reads += 1;
                // A Return enters `Sending` before its own read.
                let entered = self.apply(now, Input::Return, Trigger::Return);
                debug_assert_eq!(entered, Step::Wait, "entering the send window ends nothing");
                self.read(now, Trigger::Return, gate)
            }
        }
    }

    /// The closing read: done whatever it shows.
    fn finish(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Step {
        match self.read(now, trigger, gate) {
            Step::Wait => self.observation(now, trigger),
            step => step,
        }
    }

    fn ms(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.started).as_millis()).unwrap_or(u64::MAX)
    }

    /// Reads the field once, as an input to the model.
    fn read(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Step {
        if !gate() {
            return Step::Drop;
        }
        self.last_read = Some(now);
        self.due = None;
        let input = match self.field.read() {
            Err(error) => {
                self.reads += 1;
                Input::Failed(error)
            }
            Ok(read) => {
                let changed = self.last_text.as_deref() != Some(read.text.as_str());
                self.reads += usize::from(changed);
                let blank = locate::is_blank(&read.text);
                let shrank = self
                    .last_text
                    .as_ref()
                    .is_some_and(|before| read.text.chars().count() < before.chars().count());
                // A blank field is read too: it retires a followed region.
                let outcome = self.tracker.read(&read);
                self.last_text = Some(read.text);
                Input::Read(Reading {
                    trigger,
                    ms: self.ms(now),
                    blank,
                    shrank,
                    outcome,
                })
            }
        };
        self.apply(now, input, trigger)
    }

    /// Feeds the model; done if it says the watch is over.
    fn apply(&mut self, now: Instant, input: Input, trigger: Trigger) -> Step {
        let (state, finished) =
            std::mem::replace(&mut self.state, State::new()).advance(input, now);
        self.state = state;
        // A blank field is read again after the grace.
        if self.state.cleared_since.is_some() {
            self.due = Some(now + CLEAR_GRACE);
        }
        match finished {
            Some(finished) => Step::Done(self.observed(finished, now, trigger)),
            None => Step::Wait,
        }
    }

    fn observed(
        &self,
        finished: Finished,
        now: Instant,
        trigger: Trigger,
    ) -> CorrectionObservation {
        finished.unwrap_or_else(|lost| CorrectionObservation {
            outcome: Outcome::Lost(lost),
            trigger,
            observed_at_ms: self.ms(now),
        })
    }

    /// What the watch holds. A read always came first, so it holds something.
    fn observation(&mut self, now: Instant, trigger: Trigger) -> Step {
        Step::Done(self.observed(self.state.held.clone(), now, trigger))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::super::locate::Anchor;
    use super::*;

    /// A field that answers a script of reads, repeating the last, and counts
    /// them.
    struct Script {
        reads: RefCell<Vec<Result<String, ReadError>>>,
        calls: Cell<usize>,
    }

    impl Script {
        fn new(reads: &[&str]) -> Self {
            Self::of(reads.iter().rev().map(|t| Ok((*t).into())).collect())
        }

        fn of(reads: Vec<Result<String, ReadError>>) -> Self {
            Self {
                reads: RefCell::new(reads),
                calls: Cell::new(0),
            }
        }
    }

    impl Field for Script {
        fn read(&self) -> Result<FieldText, ReadError> {
            self.calls.set(self.calls.get() + 1);
            let mut reads = self.reads.borrow_mut();
            let read = if reads.len() > 1 {
                reads.pop().unwrap()
            } else {
                reads[0].clone()
            };
            read.map(|text| FieldText {
                text,
                windowed: false,
            })
        }
    }

    const OPEN: &dyn Fn() -> bool = &|| true;

    fn at(start: Instant, ms: u64) -> Instant {
        start + Duration::from_millis(ms)
    }

    /// A session that watches without a caret, over a field that held none of
    /// the paste's text before it.
    fn session<F: Field>(field: F, inserted: impl Into<String>, t0: Instant) -> Session<F> {
        Session::new(field, t0, Tracker::new(inserted.into(), None, Some("")))
    }

    fn anchored<F: Field>(
        field: F,
        inserted: impl Into<String>,
        t0: Instant,
        anchor: Option<Anchor>,
    ) -> Session<F> {
        anchored_over(field, inserted, t0, anchor, "")
    }

    /// As [`anchored`], over a field that held `before` ahead of the paste.
    fn anchored_over<F: Field>(
        field: F,
        inserted: impl Into<String>,
        t0: Instant,
        anchor: Option<Anchor>,
        before: &str,
    ) -> Session<F> {
        Session::new(
            field,
            t0,
            Tracker::new(inserted.into(), anchor, Some(before)),
        )
    }

    /// The first read of a watch: the paste, whole. No edit is called before.
    fn baseline<F: Field>(session: &mut Session<F>, t0: Instant) {
        assert_eq!(session.step(at(t0, 50), Event::Tick, OPEN), Step::Wait);
    }

    fn done(step: Step) -> CorrectionObservation {
        match step {
            Step::Done(seen) => seen,
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// T8.5, AC-16: a tidy that removes "um" and fixes "Kip" to "Kit" in
    /// place resets the baseline, so the tidy is never the person's
    /// correction, even when a read caught its edit first; an edit the
    /// person makes afterwards is measured from the tidied text.
    #[test]
    fn tidy_resets_baseline_no_correction() {
        let pasted = "um so send it to Kip";
        let tidied = "So send it to Kit";
        let rebase = || Tracker::new(tidied.into(), Some(Anchor { start: 0 }), Some(""));

        // Without the reset, a tidy's own edit reads as a correction (a
        // misheard fix alone), or loses the watch (the filler removal too).
        let t0 = Instant::now();
        let fix = "send it to Kip";
        let mut unreset = session(Script::new(&[fix, "send it to Kit"]), fix, t0);
        baseline(&mut unreset, t0);
        unreset.step(at(t0, 1_000), Event::ValueChanged, OPEN);
        let seen = done(unreset.step(at(t0, 2_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        let mut unreset = session(Script::new(&[pasted, tidied]), pasted, t0);
        baseline(&mut unreset, t0);
        unreset.step(at(t0, 1_000), Event::ValueChanged, OPEN);
        let seen = done(unreset.step(at(t0, 2_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));

        // With it, even after a read caught the edit: unchanged.
        let mut reset = session(Script::new(&[pasted, tidied]), pasted, t0);
        baseline(&mut reset, t0);
        reset.step(at(t0, 1_000), Event::ValueChanged, OPEN);
        reset.rebase(rebase());
        assert_eq!(reset.step(at(t0, 1_100), Event::Tick, OPEN), Step::Wait);
        let seen = done(reset.step(at(t0, 2_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Unchanged);

        // The person's own edit after the tidy is the correction.
        let mut edited = session(
            Script::new(&[pasted, tidied, "So send it to Kat"]),
            pasted,
            t0,
        );
        baseline(&mut edited, t0);
        edited.rebase(rebase());
        assert_eq!(edited.step(at(t0, 1_100), Event::Tick, OPEN), Step::Wait);
        edited.step(at(t0, 1_500), Event::ValueChanged, OPEN);
        let seen = done(edited.step(at(t0, 2_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("So send it to Kat".into()));
    }

    #[test]
    fn keeps_last_matching_read_before_clear() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["send it to Kip", "send it to Kit", ""]),
            "send it to Kip",
            t0,
        );
        baseline(&mut session, t0);
        assert_eq!(
            session.step(at(t0, 2_000), Event::ValueChanged, OPEN),
            Step::Wait
        );
        // The chat app sends on Return and clears the field before the read.
        let seen = done(session.step(at(t0, 3_000), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.trigger, Trigger::ValueChanged);
        assert_eq!(seen.observed_at_ms, 2_000);
    }

    #[test]
    fn a_fast_edit_then_send_keeps_the_finished_word() {
        let t0 = Instant::now();
        // Select Kip, type K, i, t at 80 ms, press Return 80 ms later (inside
        // one 100 ms gap of the last key): the field is cleared when the
        // Return read lands.
        let mut session = session(
            Script::new(&[
                "send it to Kip",
                "send it to K",
                "send it to Ki",
                "send it to Kit",
                "",
            ]),
            "send it to Kip",
            t0,
        );
        baseline(&mut session, t0);
        for ms in [100, 180, 260] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        let seen = done(session.step(at(t0, 340), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.observed_at_ms, 260);
    }

    #[test]
    fn a_burst_inside_one_gap_is_read_once_it_has_passed() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["send it to Kip", "send it to K", "send it to Kit", ""]),
            "send it to Kip",
            t0,
        );
        baseline(&mut session, t0);
        for ms in [100, 110, 120] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        assert_eq!(
            session.field.calls.get(),
            2,
            "the baseline and only the first change at once"
        );
        session.step(at(t0, 139), Event::Tick, OPEN);
        assert_eq!(session.field.calls.get(), 2);
        session.step(at(t0, 140), Event::Tick, OPEN);
        assert_eq!(session.field.calls.get(), 3);
        let seen = done(session.step(at(t0, 190), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.observed_at_ms, 140);
    }

    #[test]
    fn return_in_a_notes_app_keeps_watching() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&[
                "send it to Kip",
                "send it to Kip\nthanks",
                "send it to Kit\nthanks",
            ]),
            "send it to Kip",
            t0,
        );
        // Return is a newline here: the field did not shrink.
        assert_eq!(session.step(at(t0, 500), Event::Return, OPEN), Step::Wait);
        assert_eq!(
            session.step(at(t0, 1_000), Event::ValueChanged, OPEN),
            Step::Wait
        );
        // A later edit is still seen.
        session.step(at(t0, 2_000), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 30_000), Event::Tick, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.trigger, Trigger::Timeout);
    }

    #[test]
    fn return_that_shrinks_the_field_is_a_send() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&[
                "send it to Kip please",
                "send it to Kit please",
                "send it to Kit",
            ]),
            "send it to Kip please",
            t0,
        );
        baseline(&mut session, t0);
        session.step(at(t0, 100), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 900), Event::Return, OPEN));
        // The read kept before the send, not the shorter one the Return saw.
        assert_eq!(
            seen.outcome,
            Outcome::Edited("send it to Kit please".into())
        );
        assert_eq!(seen.trigger, Trigger::ValueChanged);
    }

    #[test]
    fn nothing_matching_reports_why() {
        let t0 = Instant::now();
        let mut cleared = session(Script::new(&[""]), "send it", t0);
        let seen = done(cleared.step(at(t0, 5), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
        let mut other = session(
            Script::new(&["unrelated words here", "unrelated words here"]),
            "send it",
            t0,
        );
        assert_eq!(other.step(at(t0, 5), Event::Return, OPEN), Step::Wait);
        let seen = done(other.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
        let gone = Script::of(vec![Err(ReadError::ElementGone)]);
        let mut gone = session(gone, "send it", t0);
        let seen = done(gone.step(at(t0, 5), Event::Return, OPEN));
        assert_eq!(
            seen.outcome,
            Outcome::Lost(Lost::Read(ReadError::ElementGone))
        );
    }

    #[test]
    fn a_cleared_read_outranks_other_reasons_and_errors_the_least() {
        let t0 = Instant::now();
        let reads = vec![
            Ok("unrelated words here".into()),
            Err(ReadError::Timeout),
            Ok(String::new()),
            Err(ReadError::Secure),
            Ok("still unrelated words".into()),
        ];
        let mut session = session(Script::of(reads.into_iter().rev().collect()), "send it", t0);
        for n in 0..4 {
            session.step(at(t0, n * 200), Event::ValueChanged, OPEN);
        }
        let seen = done(session.step(at(t0, 5_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
    }

    #[test]
    fn timeout_closes_an_untouched_paste_as_unchanged() {
        let t0 = Instant::now();
        let mut session = session(Script::new(&["send it"]), "send it", t0);
        assert_eq!(session.step(at(t0, 29_000), Event::Tick, OPEN), Step::Wait);
        let seen = done(session.step(at(t0, 30_000), Event::Tick, OPEN));
        assert_eq!(
            (seen.outcome, seen.trigger),
            (Outcome::Unchanged, Trigger::Timeout)
        );
    }

    #[test]
    fn rechecks_gate_at_read_time() {
        let t0 = Instant::now();
        let open = Cell::new(true);
        let gate = || open.get();
        let mut session = session(Script::new(&["send it"]), "send it", t0);
        assert_eq!(
            session.step(at(t0, 10), Event::ValueChanged, &gate),
            Step::Wait
        );
        assert_eq!(session.field.calls.get(), 1);
        open.set(false);
        assert_eq!(
            session.step(at(t0, 5_000), Event::ValueChanged, &gate),
            Step::Drop
        );
        assert_eq!(
            session.step(at(t0, 6_000), Event::Return, &gate),
            Step::Drop
        );
        assert_eq!(session.field.calls.get(), 1, "a closed gate reads nothing");
    }

    #[test]
    fn caps_reads_that_change_what_the_field_says() {
        let t0 = Instant::now();
        let texts: Vec<String> = (0..MAX_READS + 100)
            .map(|n| format!("send it {n}"))
            .collect();
        let reads = texts.iter().rev().map(|t| Ok(t.clone())).collect();
        let mut session = session(Script::of(reads), "send it", t0);
        for n in 0..(MAX_READS + 100) as u64 {
            session.step(at(t0, n * 100), Event::ValueChanged, OPEN);
        }
        assert_eq!(session.reads(), MAX_READS);
        assert_eq!(session.field.calls.get(), MAX_READS);
        assert!(matches!(
            session.step(at(t0, 200_000), Event::FocusOut, OPEN),
            Step::Done(_)
        ));
        assert_eq!(
            session.field.calls.get(),
            MAX_READS + 1,
            "the closing read is extra"
        );
    }

    #[test]
    fn reads_that_find_the_field_as_before_spend_nothing() {
        let t0 = Instant::now();
        let mut session = session(Script::new(&["send it"]), "send it", t0);
        for n in 0..100 {
            session.step(at(t0, n * 1_000), Event::ValueChanged, OPEN);
        }
        assert_eq!(session.field.calls.get(), 100);
        assert_eq!(session.reads(), 1, "only the first read found a change");
    }

    #[test]
    fn a_paste_sent_without_an_edit_is_unchanged_not_lost() {
        let t0 = Instant::now();
        // No value change is ever reported (the paste's own predates the
        // observer): the read when the watch starts is the baseline.
        let mut session = session(Script::new(&["send it to Kip", ""]), "send it to Kip", t0);
        assert_eq!(session.step(at(t0, 50), Event::Tick, OPEN), Step::Wait);
        let seen = done(session.step(at(t0, 3_000), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Unchanged);
        assert_eq!(seen.trigger, Trigger::Start);
    }

    #[test]
    fn a_later_read_of_other_text_drops_the_read_kept_before() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["send it to Kit", "something else entirely now"]),
            "send it to Kip",
            t0,
        );
        session.step(at(t0, 0), Event::ValueChanged, OPEN);
        session.step(at(t0, 100), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
    }

    /// The tap counts Return before the app handles it: the Return read sees
    /// the field still full, and the clear arrives a tick later.
    #[test]
    fn an_async_clear_after_return_keeps_the_edit_even_if_text_follows() {
        let t0 = Instant::now();
        for next in ["", "ok see you"] {
            let mut session = session(
                Script::new(&["send it to Kip", "send it to Kit", "send it to Kit", next]),
                "send it to Kip",
                t0,
            );
            baseline(&mut session, t0);
            session.step(at(t0, 250), Event::ValueChanged, OPEN);
            assert_eq!(session.step(at(t0, 300), Event::Return, OPEN), Step::Wait);
            // The app clears (or the next message is already typed).
            let seen = done(session.step(at(t0, 350), Event::ValueChanged, OPEN));
            assert_eq!(
                seen.outcome,
                Outcome::Edited("send it to Kit".into()),
                "next text {next:?}"
            );
            assert_eq!(seen.trigger, Trigger::ValueChanged);
        }
    }

    /// A blank field with no Return is a send button or delete-all, which
    /// cannot be told apart: lost, not the edit before it.
    #[test]
    fn a_field_emptied_without_a_return_is_lost() {
        let t0 = Instant::now();
        for edited in ["send it to Kit", "send it to Kip"] {
            let reads = ["send it to Kip", edited, ""];
            let mut session = session(Script::new(&reads), "send it to Kip", t0);
            baseline(&mut session, t0);
            session.step(at(t0, 100), Event::ValueChanged, OPEN);
            // Blank, with something kept: waited on, in case a Return the
            // listener counts late is about to explain it.
            assert_eq!(
                session.step(at(t0, 200), Event::ValueChanged, OPEN),
                Step::Wait
            );
            assert_eq!(session.step(at(t0, 250), Event::Tick, OPEN), Step::Wait);
            let seen = done(session.step(at(t0, 300), Event::Tick, OPEN));
            assert_eq!(
                seen.outcome,
                Outcome::Lost(Lost::ClearedBeforeRead),
                "{reads:?}"
            );
        }
    }

    #[test]
    fn a_return_and_its_clear_are_seen_after_the_read_cap() {
        let t0 = Instant::now();
        // An interior word of the paste changes with every read.
        let paste = "alpha beta gamma delta epsilon";
        let texts: Vec<String> = (0..MAX_READS + 100)
            .map(|n| format!("alpha b{n} gamma delta epsilon"))
            .collect();
        let mut reads: Vec<String> = vec![paste.into()];
        reads.extend_from_slice(&texts[..MAX_READS - 1]);
        reads.push("alpha last gamma delta epsilon".into());
        reads.push(String::new());
        let mut session = session(
            Script::of(reads.iter().rev().map(|t| Ok(t.clone())).collect()),
            paste,
            t0,
        );
        baseline(&mut session, t0);
        for n in 1..(MAX_READS + 5) as u64 {
            session.step(at(t0, n * 100), Event::ValueChanged, OPEN);
        }
        assert_eq!(session.reads(), MAX_READS);
        let base = (MAX_READS as u64 + 5) * 100;
        // Past the cap a change is not read, but a Return is, and the clear
        // after it is read whatever the cap says.
        assert_eq!(
            session.step(at(t0, base + 100), Event::Return, OPEN),
            Step::Wait
        );
        let seen = done(session.step(at(t0, base + 200), Event::ValueChanged, OPEN));
        assert!(matches!(seen.outcome, Outcome::Edited(_)), "{seen:?}");
    }

    #[test]
    fn other_text_long_after_a_return_is_not_a_send() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&[
                "send it to Kip",
                "send it to Kip\nmore",
                "totally different now",
            ]),
            "send it to Kip",
            t0,
        );
        assert_eq!(session.step(at(t0, 300), Event::Return, OPEN), Step::Wait);
        session.step(at(t0, 400), Event::ValueChanged, OPEN);
        // Past the send window the field changing is just the field changing.
        assert_eq!(
            session.step(at(t0, 5_000), Event::ValueChanged, OPEN),
            Step::Wait
        );
        let seen = done(session.step(at(t0, 6_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
    }

    #[test]
    fn a_missing_value_is_a_failed_read_not_a_cleared_field() {
        let t0 = Instant::now();
        let mut session = session(Script::of(vec![Err(ReadError::Failed)]), "send it", t0);
        assert_eq!(session.step(at(t0, 5), Event::Return, OPEN), Step::Wait);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::Read(ReadError::Failed)));
    }

    #[test]
    fn a_held_return_key_stops_being_read() {
        let t0 = Instant::now();
        let mut session = session(Script::new(&["send it"]), "send it", t0);
        for n in 0..(MAX_RETURN_READS as u64 + 30) {
            session.step(at(t0, n), Event::Return, OPEN);
        }
        assert_eq!(session.field.calls.get(), MAX_RETURN_READS);
    }

    fn anchor(base: &str, pasted: &str) -> Anchor {
        let at = base.rfind(pasted).unwrap();
        Anchor {
            start: base[..at].encode_utf16().count(),
        }
    }

    #[test]
    fn an_anchored_watch_judges_the_region_at_the_anchor() {
        let t0 = Instant::now();
        let base = "Hi team. send it to Kip";
        let mut session = anchored_over(
            Script::new(&[base, "Hi team. send it to Hex"]),
            "send it to Kip",
            t0,
            Some(anchor(base, "send it to Kip")),
            "Hi team. ",
        );
        assert_eq!(session.step(at(t0, 50), Event::Tick, OPEN), Step::Wait);
        session.step(at(t0, 200), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Hex".into()));
    }

    #[test]
    fn an_anchor_is_trusted_only_after_the_paste_is_seen_there() {
        let t0 = Instant::now();
        // The field at the anchor says something else from the start: nothing
        // anchored ever calls it an edit.
        let mut session = anchored(
            Script::new(&["we should go to lunch today"]),
            "send it to Kip",
            t0,
            Some(Anchor { start: 0 }),
        );
        session.step(at(t0, 50), Event::Tick, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
        // An anchor in other units (the paste is in the field, elsewhere) is
        // never adopted: the copy there could be an old one.
        let mut session = anchored(
            Script::new(&["send it to Kip", "send it to Kip", "send it to Kit"]),
            "send it to Kip",
            t0,
            Some(Anchor { start: 500 }),
        );
        baseline(&mut session, t0);
        session.step(at(t0, 700), Event::ValueChanged, OPEN);
        session.step(at(t0, 900), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 1_300), Event::FocusOut, OPEN));
        assert!(!matches!(seen.outcome, Outcome::Edited(_)));
    }

    #[test]
    fn a_paste_that_arrives_elsewhere_than_its_anchor_is_never_followed() {
        let t0 = Instant::now();
        // The field has another copy of the text, and the paste shows up away
        // from the caret: which copy is the paste's is not proved.
        let before = "send it to Kip. ok ";
        let after = "send it to Kip. ok send it to Kip";
        let mut session = anchored_over(
            Script::new(&[before, after, "send it to Kip. ok send it to Kit"]),
            "send it to Kip",
            t0,
            Some(Anchor { start: 0 }),
            before,
        );
        baseline(&mut session, t0);
        session.step(at(t0, 150), Event::ValueChanged, OPEN);
        session.step(at(t0, 300), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 500), Event::FocusOut, OPEN));
        assert!(!matches!(seen.outcome, Outcome::Edited(_)));
    }

    #[test]
    fn a_paste_applied_after_a_read_of_the_old_field_lands_at_its_anchor() {
        let t0 = Instant::now();
        let before = "send it to Kip. ok ";
        let after = "send it to Kip. ok send it to Kip";
        let mut session = anchored_over(
            Script::new(&[before, after, "send it to Kip. ok send it to Kit"]),
            "send it to Kip",
            t0,
            Some(Anchor { start: 19 }),
            before,
        );
        baseline(&mut session, t0);
        session.step(at(t0, 150), Event::ValueChanged, OPEN);
        session.step(at(t0, 300), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 500), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
    }

    #[test]
    fn nothing_is_an_edit_before_the_paste_was_seen_whole() {
        let t0 = Instant::now();
        for anchor in [None, Some(Anchor { start: 0 })] {
            let mut session = anchored(Script::new(&["secret"]), "Kip", t0, anchor);
            baseline(&mut session, t0);
            let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
            assert!(!matches!(seen.outcome, Outcome::Edited(_)), "{anchor:?}");
        }
    }

    #[test]
    fn a_late_return_converts_a_blank_field_into_a_send() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["send it to Kip", "send it to Kit", "", ""]),
            "send it to Kip",
            t0,
        );
        baseline(&mut session, t0);
        session.step(at(t0, 100), Event::ValueChanged, OPEN);
        // The clear arrives before the Return the listener counts.
        assert_eq!(
            session.step(at(t0, 200), Event::ValueChanged, OPEN),
            Step::Wait
        );
        let seen = done(session.step(at(t0, 230), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
    }

    #[test]
    fn a_fix_after_a_return_in_a_notes_app_is_an_ordinary_read() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["send it to Kip", "send it to Kip\n", "send it to Kit\n"]),
            "send it to Kip",
            t0,
        );
        baseline(&mut session, t0);
        assert_eq!(session.step(at(t0, 200), Event::Return, OPEN), Step::Wait);
        // Still the paste, changed: not a send, so the watch goes on.
        assert_eq!(
            session.step(at(t0, 400), Event::ValueChanged, OPEN),
            Step::Wait
        );
        let seen = done(session.step(at(t0, 900), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
    }

    #[test]
    fn a_huge_paste_is_not_aligned() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["word ".repeat(3_000).as_str()]),
            "other ".repeat(40_000),
            t0,
        );
        session.step(at(t0, 50), Event::Tick, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
    }

    fn reading(blank: bool, shrank: bool, outcome: Alignment) -> Input {
        Input::Read(Reading {
            trigger: Trigger::ValueChanged,
            ms: 7,
            blank,
            shrank,
            outcome,
        })
    }

    fn kit() -> Outcome {
        Outcome::Edited("send it to Kit".into())
    }

    fn kit_read() -> Alignment {
        Alignment::Edited("send it to Kit".into())
    }

    fn kept() -> CorrectionObservation {
        CorrectionObservation {
            outcome: kit(),
            trigger: Trigger::ValueChanged,
            observed_at_ms: 7,
        }
    }

    /// The model, row by row: inputs (each at a time, in ms after the start)
    /// to what the watch ends with, if it ends, and what it holds.
    #[test]
    fn advance_table() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let edited = || reading(false, false, kit_read());
        /// A named row: inputs at times, what the watch ends with, what it holds.
        type Row = (&'static str, Vec<(u64, Input)>, Option<Finished>, Finished);
        let cases: Vec<Row> = vec![
            ("an edit is held", vec![(0, edited())], None, Ok(kept())),
            (
                "other text drops it",
                vec![
                    (0, edited()),
                    (50, reading(false, false, Alignment::NotFound)),
                ],
                None,
                Err(Lost::SpanNotFound),
            ),
            (
                "blank with nothing kept is noted and goes on",
                vec![(0, reading(true, false, Alignment::NotFound))],
                None,
                Err(Lost::ClearedBeforeRead),
            ),
            (
                "blank with something kept waits the grace",
                vec![
                    (0, edited()),
                    (50, reading(true, false, Alignment::NotFound)),
                ],
                None,
                Ok(kept()),
            ),
            (
                "blank still, after the grace, is cleared with no Return",
                vec![
                    (0, edited()),
                    (50, reading(true, false, Alignment::NotFound)),
                    (200, reading(true, false, Alignment::NotFound)),
                ],
                Some(Err(Lost::ClearedBeforeRead)),
                Err(Lost::ClearedBeforeRead),
            ),
            (
                "text again inside the grace forgets the blank",
                vec![
                    (0, edited()),
                    (50, reading(true, false, Alignment::NotFound)),
                    (80, edited()),
                    (200, reading(true, false, Alignment::NotFound)),
                ],
                None,
                Ok(kept()),
            ),
            (
                "a Return, then blank, is a send",
                vec![
                    (0, edited()),
                    (50, Input::Return),
                    (60, reading(true, false, Alignment::NotFound)),
                ],
                Some(Ok(kept())),
                Ok(kept()),
            ),
            (
                "a Return after the blank, inside the grace, is a send too",
                vec![
                    (0, edited()),
                    (50, reading(true, false, Alignment::NotFound)),
                    (70, Input::Return),
                    (75, reading(true, false, Alignment::NotFound)),
                ],
                Some(Ok(kept())),
                Ok(kept()),
            ),
            (
                "after a Return, shorter text is the clear",
                vec![
                    (0, edited()),
                    (50, Input::Return),
                    (60, reading(false, true, kit_read())),
                ],
                Some(Ok(kept())),
                Ok(kept()),
            ),
            (
                "after a Return, other text is the clear",
                vec![
                    (0, edited()),
                    (50, Input::Return),
                    (60, reading(false, false, Alignment::NotFound)),
                ],
                Some(Ok(kept())),
                Ok(kept()),
            ),
            (
                "after a Return, the paste still there is an ordinary read",
                vec![
                    (0, reading(false, false, Alignment::Unchanged)),
                    (50, Input::Return),
                    (60, edited()),
                ],
                None,
                Ok(kept()),
            ),
            (
                "a Return's window passes",
                vec![
                    (0, edited()),
                    (50, Input::Return),
                    (2_000, reading(false, false, Alignment::NotFound)),
                ],
                None,
                Err(Lost::SpanNotFound),
            ),
            (
                "a Return with nothing kept, then blank, is cleared",
                vec![
                    (0, Input::Return),
                    (10, reading(true, false, Alignment::NotFound)),
                ],
                Some(Err(Lost::ClearedBeforeRead)),
                Err(Lost::ClearedBeforeRead),
            ),
            (
                "a failed read never erases what was kept",
                vec![(0, edited()), (50, Input::Failed(ReadError::Timeout))],
                None,
                Ok(kept()),
            ),
            (
                "the element gone ends it with what was kept",
                vec![(0, edited()), (50, Input::Failed(ReadError::ElementGone))],
                Some(Ok(kept())),
                Ok(kept()),
            ),
            (
                "a failed read with nothing kept is why",
                vec![(0, Input::Failed(ReadError::Secure))],
                None,
                Err(Lost::Read(ReadError::Secure)),
            ),
            (
                "a cleared field outranks a failed read",
                vec![
                    (0, reading(true, false, Alignment::NotFound)),
                    (50, Input::Failed(ReadError::Timeout)),
                ],
                None,
                Err(Lost::ClearedBeforeRead),
            ),
        ];
        for (name, inputs, ends, held) in cases {
            let mut state = State::new();
            let mut finished = None;
            for (at, input) in inputs {
                (state, finished) = state.advance(input, ms(at));
            }
            assert_eq!(finished, ends, "{name}: ends");
            assert_eq!(state.held, held, "{name}: holds");
        }
    }

    /// The limit, stated: a keystroke that shares a tick with the Return that
    /// sends it is never read, so the edit kept is the word as far as it was
    /// read: "Ki", not "Kit". A Return read within a tick of a one-character
    /// change cannot be told from one that came after more typing.
    #[test]
    fn a_key_unread_at_the_return_is_the_known_limit() {
        let t0 = Instant::now();
        let mut session = session(
            Script::new(&["send it to Kip", "send it to K", "send it to Ki", ""]),
            "send it to Kip",
            t0,
        );
        baseline(&mut session, t0);
        session.step(at(t0, 100), Event::ValueChanged, OPEN);
        session.step(at(t0, 180), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 260), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Ki".into()));
    }

    #[test]
    fn the_pastes_own_focus_change_is_ignored() {
        let t0 = Instant::now();
        let mut session = session(Script::new(&["send it"]), "send it", t0);
        assert_eq!(session.step(at(t0, 100), Event::FocusOut, OPEN), Step::Wait);
        assert!(matches!(
            session.step(at(t0, 400), Event::FocusOut, OPEN),
            Step::Done(_)
        ));
    }

    #[test]
    fn history_strings_for_each_outcome() {
        let seen = |outcome| CorrectionObservation {
            outcome,
            trigger: Trigger::Timeout,
            observed_at_ms: 7,
        };
        let edited = seen(Outcome::Edited("Kit".into())).into_history();
        assert_eq!(
            (
                edited.outcome.as_str(),
                edited.reason,
                edited.edited_text.as_deref()
            ),
            ("edited", None, Some("Kit"))
        );
        let lost = seen(Outcome::Lost(Lost::Read(ReadError::Secure))).into_history();
        assert_eq!(
            (
                lost.outcome.as_str(),
                lost.reason.as_deref(),
                lost.edited_text
            ),
            ("lost", Some("secure"), None)
        );
        let same = seen(Outcome::Unchanged).into_history();
        assert_eq!(
            (same.outcome.as_str(), same.observed_at_ms),
            ("unchanged", 7)
        );
    }
}
