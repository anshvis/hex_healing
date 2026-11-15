//! One watched paste: when to read the field, which read to keep, and what
//! to report. Pure over a [`Field`], so the rules are tested without
//! Accessibility.
//!
//! The model. A watch is `Watching` or, after a Return, `Sending`; it ends
//! with the observation it holds. Every read becomes a [`Reading`] (is the
//! field blank, did it shrink, what does it hold of the paste) and
//! [`advance`], one pure function of the phase, what is held and the input,
//! says what to do with it: record it, ignore it, or finish. `step` applies
//! that and is the only writer of the phase.
//!
//! - `Watching`: a read that holds the paste is kept; one that holds other
//!   text drops what was kept. A blank read after something was kept is the
//!   field cleared with no Return: a send button or delete-all, which cannot
//!   be told apart, so it is `cleared_before_read` (training-safe: a send
//!   button without a Return loses the observation). A blank read before
//!   anything was kept is only noted (the paste may not have landed).
//! - `Sending { kept, until }`: a Return enters it, with what was kept at
//!   that moment, before its read. The tap counts the Return before the app
//!   handles it, so the app may clear the field a tick later. For
//!   [`SEND_WINDOW`], a read that shows the field blank, shorter or holding
//!   other text than `kept` is that clear: the watch ends with `kept`. A read
//!   that still shows what was kept (a notes app's newline) changes nothing,
//!   and after the window the watch is `Watching` again.

use std::time::{Duration, Instant};

use super::locate::{self, Alignment, Anchor, Landing};
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
}

impl Lost {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read(ReadError::ElementGone) => "element_gone",
            Self::Read(ReadError::Secure) => "secure",
            Self::Read(ReadError::TooLarge) => "too_large",
            Self::Read(ReadError::Timeout) => "read_timeout",
            Self::Read(ReadError::Failed) => "read_failed",
            Self::SpanNotFound => "span_not_found",
            Self::ClearedBeforeRead => "cleared_before_read",
        }
    }

    /// Which reason wins when reads disagree: a field that was cleared, then
    /// one that held other text, then a read that failed.
    fn priority(self) -> u8 {
        match self {
            Self::ClearedBeforeRead => 2,
            Self::SpanNotFound => 1,
            Self::Read(_) => 0,
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

impl Outcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unchanged => "unchanged",
            Self::Edited(_) => "edited",
            Self::Lost(lost) => lost.as_str(),
        }
    }
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
    blank: bool,
    /// Shorter than the read before.
    shrank: bool,
    /// What it holds of the paste, `None` for other text.
    outcome: Option<Outcome>,
}

enum Input<'a> {
    Return,
    Read(&'a Reading),
}

/// What to do with an input.
#[derive(Debug, Eq, PartialEq)]
enum Verdict {
    Hold,
    /// Keep this read as the paste (`Some`), or drop what was kept because
    /// the field holds other text (`None`).
    Record(Option<Outcome>),
    /// Blank before anything was kept: noted, and the watch goes on.
    Blank,
    Finish(Finish),
}

#[derive(Debug, Eq, PartialEq)]
enum Finish {
    Kept(CorrectionObservation),
    Lost(Lost),
}

/// The whole model: the next phase and the verdict, from the phase, what is
/// held, and the input. Pure.
fn advance(
    phase: Phase,
    best: Option<&CorrectionObservation>,
    input: Input,
    now: Instant,
) -> (Phase, Verdict) {
    let phase = match phase {
        Phase::Sending { until, .. } if now > until => Phase::Watching,
        phase => phase,
    };
    let reading = match input {
        Input::Return => {
            let phase = Phase::Sending {
                until: now + SEND_WINDOW,
                kept: best.cloned(),
            };
            return (phase, Verdict::Hold);
        }
        Input::Read(reading) => reading,
    };
    let verdict = match &phase {
        Phase::Watching if reading.blank => match best {
            Some(_) => Verdict::Finish(Finish::Lost(Lost::ClearedBeforeRead)),
            None => Verdict::Blank,
        },
        Phase::Sending { kept, .. } if reading.blank => Verdict::Finish(
            kept.clone()
                .map_or(Finish::Lost(Lost::ClearedBeforeRead), Finish::Kept),
        ),
        Phase::Sending {
            kept: Some(kept), ..
        } if reading.shrank || reading.outcome.as_ref() != Some(&kept.outcome) => {
            Verdict::Finish(Finish::Kept(kept.clone()))
        }
        _ => Verdict::Record(reading.outcome.clone()),
    };
    (phase, verdict)
}

/// Where the paste is expected to sit in the field.
enum Anchoring {
    /// No anchor, or one that did not hold: the whole field is searched,
    /// under the strict rules.
    Whole,
    /// The paste has not yet been seen where it was put.
    Pending(Anchor),
    /// It has: only the region at the anchor is judged.
    Landed(Anchor),
}

pub struct Session<F: Field> {
    field: F,
    inserted: String,
    started: Instant,
    anchoring: Anchoring,
    /// Reads that found the field changed (see [`MAX_READS`]).
    reads: usize,
    return_reads: usize,
    last_read: Option<Instant>,
    last_text: Option<String>,
    due: Option<Instant>,
    phase: Phase,
    /// What the watch holds: the last read that still held the paste, or
    /// why none did. `None` only before the first read.
    held: Option<Result<CorrectionObservation, Lost>>,
}

impl<F: Field> Session<F> {
    pub fn new(field: F, inserted: String, started: Instant, anchor: Option<Anchor>) -> Self {
        Self {
            field,
            inserted,
            started,
            anchoring: anchor.map_or(Anchoring::Whole, Anchoring::Pending),
            reads: 0,
            return_reads: 0,
            last_read: None,
            last_text: None,
            due: None,
            phase: Phase::Watching,
            held: None,
        }
    }

    #[cfg(test)]
    pub fn reads(&self) -> usize {
        self.reads
    }

    fn best(&self) -> Option<&CorrectionObservation> {
        self.held.as_ref().and_then(|held| held.as_ref().ok())
    }

    fn sending(&self, now: Instant) -> bool {
        matches!(&self.phase, Phase::Sending { until, .. } if now <= *until)
    }

    /// Advances on `event` at `now`. `gate` is asked before every read.
    pub fn step(&mut self, now: Instant, event: Event, gate: &dyn Fn() -> bool) -> Step {
        match event {
            Event::ValueChanged => {
                if self.reads >= MAX_READS && !self.sending(now) {
                    return Step::Wait;
                }
                let wait = self
                    .last_read
                    .map(|at| (at + READ_GAP).saturating_duration_since(now))
                    .filter(|wait| !wait.is_zero());
                match wait {
                    Some(wait) => self.due = Some(now + wait),
                    None => return self.watch(now, Trigger::ValueChanged, gate),
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
                    let step = self.watch(now, trigger, gate);
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
                self.apply(now, Input::Return, Trigger::Return);
                self.watch(now, Trigger::Return, gate)
            }
        }
    }

    /// One read, as a step: done if the field was sent.
    fn watch(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Step {
        match self.read(now, trigger, gate) {
            None => Step::Drop,
            Some(true) => self.observation(now, trigger),
            Some(false) => Step::Wait,
        }
    }

    /// The closing read: done whatever it shows.
    fn finish(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Step {
        match self.read(now, trigger, gate) {
            None => Step::Drop,
            Some(_) => self.observation(now, trigger),
        }
    }

    fn ms(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.started).as_millis() as u64
    }

    fn lose(&mut self, lost: Lost) {
        let outranked = match &self.held {
            Some(Err(held)) => lost.priority() < held.priority(),
            Some(Ok(_)) => true,
            None => false,
        };
        if !outranked {
            self.held = Some(Err(lost));
        }
    }

    /// The one place the phase changes, and the verdict applied.
    fn apply(&mut self, now: Instant, input: Input, trigger: Trigger) -> bool {
        let phase = std::mem::replace(&mut self.phase, Phase::Watching);
        let (phase, verdict) = advance(phase, self.best(), input, now);
        self.phase = phase;
        let finished = matches!(verdict, Verdict::Finish(_));
        match verdict {
            Verdict::Hold => {}
            Verdict::Record(Some(outcome)) => {
                self.held = Some(Ok(CorrectionObservation {
                    outcome,
                    trigger,
                    observed_at_ms: self.ms(now),
                }));
            }
            Verdict::Record(None) => {
                if self.best().is_some() {
                    self.held = None;
                }
                self.lose(Lost::SpanNotFound);
            }
            Verdict::Blank => self.lose(Lost::ClearedBeforeRead),
            Verdict::Finish(Finish::Kept(kept)) => self.held = Some(Ok(kept)),
            Verdict::Finish(Finish::Lost(lost)) => self.held = Some(Err(lost)),
        }
        finished
    }

    /// What `view` shows of the paste, and how that moves the anchor.
    fn outcome(&mut self, view: &locate::View) -> Option<Outcome> {
        let align = |alignment| match alignment {
            Alignment::Unchanged => Some(Outcome::Unchanged),
            Alignment::Edited(span) => Some(Outcome::Edited(span)),
            Alignment::NotFound => None,
        };
        match self.anchoring {
            Anchoring::Whole => align(locate::align(view.text, &self.inserted, view.windowed)),
            Anchoring::Pending(anchor) => match locate::landing(view, &self.inserted, &anchor) {
                Landing::Landed => {
                    // The reference for how far the region moves from here.
                    let total = view.total;
                    self.anchoring = Anchoring::Landed(Anchor { total, ..anchor });
                    Some(Outcome::Unchanged)
                }
                // The offset is not this app's: search the whole field.
                Landing::Elsewhere => {
                    self.anchoring = Anchoring::Whole;
                    align(locate::align(view.text, &self.inserted, view.windowed))
                }
                // Not there yet, or edited before it was ever seen: no
                // position to trust, so no edit is called.
                Landing::Absent => None,
            },
            Anchoring::Landed(anchor) => align(locate::align_at(view, &self.inserted, &anchor)),
        }
    }

    /// Reads the field once and applies what it shows. `None` when the gate
    /// is closed (nothing was read); else whether the watch is over.
    fn read(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Option<bool> {
        if !gate() {
            return None;
        }
        self.last_read = Some(now);
        self.due = None;
        let read = match self.field.read() {
            Ok(read) => read,
            Err(error) => {
                self.reads += 1;
                // A failed read never erases what was kept.
                if self.best().is_none() {
                    self.lose(Lost::Read(error));
                }
                return Some(error == ReadError::ElementGone);
            }
        };
        let changed = self.last_text.as_deref() != Some(read.text.as_str());
        self.reads += usize::from(changed);
        let blank = locate::is_blank(&read.text);
        let shrank = self
            .last_text
            .as_ref()
            .is_some_and(|before| read.text.chars().count() < before.chars().count());
        let view = locate::View {
            text: &read.text,
            offset: read.offset,
            total: read.total,
            windowed: read.windowed,
        };
        let outcome = if blank { None } else { self.outcome(&view) };
        self.last_text = Some(read.text);
        let reading = Reading {
            blank,
            shrank,
            outcome,
        };
        Some(self.apply(now, Input::Read(&reading), trigger))
    }

    /// What the watch holds. A read always came first, so it holds something.
    fn observation(&mut self, now: Instant, trigger: Trigger) -> Step {
        match self.held.take() {
            Some(Ok(observation)) => Step::Done(observation),
            Some(Err(lost)) => Step::Done(CorrectionObservation {
                outcome: Outcome::Lost(lost),
                trigger,
                observed_at_ms: self.ms(now),
            }),
            None => Step::Drop,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

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
                offset: 0,
                total: text.encode_utf16().count(),
                text,
                windowed: false,
            })
        }
    }

    const OPEN: &dyn Fn() -> bool = &|| true;

    fn at(start: Instant, ms: u64) -> Instant {
        start + Duration::from_millis(ms)
    }

    fn done(step: Step) -> CorrectionObservation {
        match step {
            Step::Done(seen) => seen,
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn keeps_last_matching_read_before_clear() {
        let t0 = Instant::now();
        let mut session = Session::new(
            Script::new(&["send it to Kit", ""]),
            "send it to Kip".into(),
            t0,
            None,
        );
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
        let mut session = Session::new(
            Script::new(&["send it to K", "send it to Ki", "send it to Kit", ""]),
            "send it to Kip".into(),
            t0,
            None,
        );
        for ms in [0, 80, 160] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        let seen = done(session.step(at(t0, 240), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.observed_at_ms, 160);
    }

    #[test]
    fn a_burst_inside_one_gap_is_read_once_it_has_passed() {
        let t0 = Instant::now();
        let mut session = Session::new(
            Script::new(&["send it to K", "send it to Kit", ""]),
            "send it to Kip".into(),
            t0,
            None,
        );
        for ms in [0, 10, 20] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        assert_eq!(
            session.field.calls.get(),
            1,
            "only the first change at once"
        );
        session.step(at(t0, 39), Event::Tick, OPEN);
        assert_eq!(session.field.calls.get(), 1);
        session.step(at(t0, 40), Event::Tick, OPEN);
        assert_eq!(session.field.calls.get(), 2);
        let seen = done(session.step(at(t0, 90), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.observed_at_ms, 40);
    }

    #[test]
    fn return_in_a_notes_app_keeps_watching() {
        let t0 = Instant::now();
        let mut session = Session::new(
            Script::new(&[
                "send it to Kip",
                "send it to Kip\nthanks",
                "send it to Kit\nthanks",
            ]),
            "send it to Kip".into(),
            t0,
            None,
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
        let mut session = Session::new(
            Script::new(&["send it to Kit please", "send it to Kit"]),
            "send it to Kip please".into(),
            t0,
            None,
        );
        session.step(at(t0, 10), Event::ValueChanged, OPEN);
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
        let mut cleared = Session::new(Script::new(&[""]), "send it".into(), t0, None);
        let seen = done(cleared.step(at(t0, 5), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
        let mut other = Session::new(
            Script::new(&["unrelated words here", "unrelated words here"]),
            "send it".into(),
            t0,
            None,
        );
        assert_eq!(other.step(at(t0, 5), Event::Return, OPEN), Step::Wait);
        let seen = done(other.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
        let gone = Script::of(vec![Err(ReadError::ElementGone)]);
        let mut gone = Session::new(gone, "send it".into(), t0, None);
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
        let mut session = Session::new(
            Script::of(reads.into_iter().rev().collect()),
            "send it".into(),
            t0,
            None,
        );
        for n in 0..4 {
            session.step(at(t0, n * 200), Event::ValueChanged, OPEN);
        }
        let seen = done(session.step(at(t0, 5_000), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
    }

    #[test]
    fn timeout_closes_an_untouched_paste_as_unchanged() {
        let t0 = Instant::now();
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0, None);
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
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0, None);
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
        let mut session = Session::new(Script::of(reads), "send it".into(), t0, None);
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
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0, None);
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
        let mut session = Session::new(
            Script::new(&["send it to Kip", ""]),
            "send it to Kip".into(),
            t0,
            None,
        );
        assert_eq!(session.step(at(t0, 50), Event::Tick, OPEN), Step::Wait);
        let seen = done(session.step(at(t0, 3_000), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Unchanged);
        assert_eq!(seen.trigger, Trigger::Start);
    }

    #[test]
    fn a_later_read_of_other_text_drops_the_read_kept_before() {
        let t0 = Instant::now();
        let mut session = Session::new(
            Script::new(&["send it to Kit", "something else entirely now"]),
            "send it to Kip".into(),
            t0,
            None,
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
        for next in ["", "ok see you", "also send it to Rune", "send it to Kip"] {
            let mut session = Session::new(
                Script::new(&["send it to Kip", "send it to Kit", "send it to Kit", next]),
                "send it to Kip".into(),
                t0,
                None,
            );
            assert_eq!(session.step(at(t0, 50), Event::Tick, OPEN), Step::Wait);
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
        for reads in [&["send it to Kit", ""][..], &["send it to Kip", ""][..]] {
            let mut session = Session::new(Script::new(reads), "send it to Kip".into(), t0, None);
            session.step(at(t0, 100), Event::ValueChanged, OPEN);
            let seen = done(session.step(at(t0, 200), Event::ValueChanged, OPEN));
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
        let texts: Vec<String> = (0..MAX_READS + 5)
            .map(|n| format!("send it to Kit {n}"))
            .collect();
        let mut reads: Vec<String> = texts[..MAX_READS].to_vec();
        reads.push("send it to Kit last".into());
        reads.push(String::new());
        let mut session = Session::new(
            Script::of(reads.iter().rev().map(|t| Ok(t.clone())).collect()),
            "send it to Kip".into(),
            t0,
            None,
        );
        for n in 0..(MAX_READS + 5) as u64 {
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
        let mut session = Session::new(
            Script::new(&[
                "send it to Kip",
                "send it to Kip\nmore",
                "totally different now",
            ]),
            "send it to Kip".into(),
            t0,
            None,
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
        let mut session = Session::new(
            Script::of(vec![Err(ReadError::Failed)]),
            "send it".into(),
            t0,
            None,
        );
        assert_eq!(session.step(at(t0, 5), Event::Return, OPEN), Step::Wait);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::Read(ReadError::Failed)));
    }

    #[test]
    fn a_held_return_key_stops_being_read() {
        let t0 = Instant::now();
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0, None);
        for n in 0..(MAX_RETURN_READS as u64 + 30) {
            session.step(at(t0, n), Event::Return, OPEN);
        }
        assert_eq!(session.field.calls.get(), MAX_RETURN_READS);
    }

    fn anchor(base: &str, pasted: &str) -> Anchor {
        let at = base.rfind(pasted).unwrap();
        Anchor {
            start: base[..at].encode_utf16().count(),
            len: pasted.encode_utf16().count(),
            total: base.encode_utf16().count(),
        }
    }

    #[test]
    fn an_anchored_watch_judges_the_region_at_the_anchor() {
        let t0 = Instant::now();
        let base = "Hi team. send it to Kip";
        let mut session = Session::new(
            Script::new(&[base, "Hi team. send it to Hex"]),
            "send it to Kip".into(),
            t0,
            Some(anchor(base, "send it to Kip")),
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
        let mut session = Session::new(
            Script::new(&["we should go to lunch today"]),
            "send it to Kip".into(),
            t0,
            Some(Anchor {
                start: 0,
                len: 14,
                total: 28,
            }),
        );
        session.step(at(t0, 50), Event::Tick, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
        // An anchor in other units (the paste is in the field, elsewhere)
        // falls back to the strict whole-field rules.
        let mut session = Session::new(
            Script::new(&["send it to Kip", "send it to Kit"]),
            "send it to Kip".into(),
            t0,
            Some(Anchor {
                start: 500,
                len: 14,
                total: 600,
            }),
        );
        session.step(at(t0, 50), Event::Tick, OPEN);
        session.step(at(t0, 200), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
    }

    #[test]
    fn a_huge_paste_is_not_aligned() {
        let t0 = Instant::now();
        let mut session = Session::new(
            Script::new(&["word ".repeat(3_000).as_str()]),
            "other ".repeat(40_000),
            t0,
            None,
        );
        session.step(at(t0, 50), Event::Tick, OPEN);
        let seen = done(session.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
    }

    #[test]
    fn the_pastes_own_focus_change_is_ignored() {
        let t0 = Instant::now();
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0, None);
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
