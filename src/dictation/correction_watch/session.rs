//! One watched paste: when to read the field, which read to keep, and what
//! to report. Pure over a [`Field`], so the rules are tested without
//! Accessibility.

use std::time::{Duration, Instant};

use super::locate::{self, Alignment};
use crate::accessibility::ReadError;
use crate::history::HistoryCorrection;

/// How long a paste is watched.
pub const WATCH: Duration = Duration::from_secs(30);
/// A changed value is read at most this often. The first change is read at
/// once and the last of a burst once the gap has passed, so the read that
/// matters (the edit just before a chat app sends and clears the field) is
/// never a keystroke behind.
pub const READ_GAP: Duration = Duration::from_millis(100);
/// Reads per target that found the field changed; a read that finds it as
/// before, and the closing read, are not counted.
pub const MAX_READS: usize = 30;
/// A focus change this soon after the paste is the paste's own.
pub const FOCUS_GRACE: Duration = Duration::from_millis(300);
/// Characters read from the field, whole or around the caret.
pub const READ_CHARS: usize = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Trigger {
    ValueChanged,
    FocusOut,
    Return,
    Timeout,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
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
    ElementGone,
    Secure,
    TooLarge,
    ReadTimeout,
    SpanNotFound,
    ClearedBeforeRead,
}

impl Lost {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ElementGone => "element_gone",
            Self::Secure => "secure",
            Self::TooLarge => "too_large",
            Self::SpanNotFound => "span_not_found",
            Self::ReadTimeout => "read_timeout",
            Self::ClearedBeforeRead => "cleared_before_read",
        }
    }

    /// Which reason wins when reads disagree: a field that was cleared, then
    /// one that held other text, then a read that failed.
    fn priority(self) -> u8 {
        match self {
            Self::ClearedBeforeRead => 2,
            Self::SpanNotFound => 1,
            Self::ElementGone | Self::Secure | Self::TooLarge | Self::ReadTimeout => 0,
        }
    }
}

impl From<ReadError> for Lost {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::ElementGone => Self::ElementGone,
            ReadError::Secure => Self::Secure,
            ReadError::TooLarge => Self::TooLarge,
            ReadError::Timeout => Self::ReadTimeout,
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
    fn read(&self) -> Result<String, ReadError>;
}

impl Field for crate::accessibility::RetainedElement {
    fn read(&self) -> Result<String, ReadError> {
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

pub struct Session<F: Field> {
    field: F,
    inserted: String,
    started: Instant,
    /// Reads that found the field changed (see [`MAX_READS`]).
    reads: usize,
    last_read: Option<Instant>,
    last_text: Option<String>,
    due: Option<Instant>,
    /// The last read that still held the text.
    best: Option<CorrectionObservation>,
    lost: Option<Lost>,
}

impl<F: Field> Session<F> {
    pub fn new(field: F, inserted: String, started: Instant) -> Self {
        Self {
            field,
            inserted,
            started,
            reads: 0,
            last_read: None,
            last_text: None,
            due: None,
            best: None,
            lost: None,
        }
    }

    #[cfg(test)]
    pub fn reads(&self) -> usize {
        self.reads
    }

    /// Advances on `event` at `now`. `gate` is asked before every read.
    pub fn step(&mut self, now: Instant, event: Event, gate: &dyn Fn() -> bool) -> Step {
        match event {
            Event::ValueChanged => {
                if self.reads >= MAX_READS {
                    return Step::Wait;
                }
                let wait = self
                    .last_read
                    .map(|at| (at + READ_GAP).saturating_duration_since(now))
                    .filter(|wait| !wait.is_zero());
                match wait {
                    Some(wait) => self.due = Some(now + wait),
                    None if self.read(now, Trigger::ValueChanged, gate).is_none() => {
                        return Step::Drop;
                    }
                    None => {}
                }
                Step::Wait
            }
            Event::Tick => {
                if self.due.is_some_and(|due| due <= now)
                    && self.read(now, Trigger::ValueChanged, gate).is_none()
                {
                    return Step::Drop;
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
            // Return is a send only if the field emptied or shrank (a chat
            // app); in a notes app it is a newline and the watch goes on.
            Event::Return => match self.read(now, Trigger::Return, gate) {
                None => Step::Drop,
                Some(true) => Step::Done(self.observation(now, Trigger::Return)),
                Some(false) => Step::Wait,
            },
        }
    }

    fn finish(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Step {
        match self.read(now, trigger, gate) {
            None => Step::Drop,
            Some(_) => Step::Done(self.observation(now, trigger)),
        }
    }

    fn ms(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.started).as_millis() as u64
    }

    fn lose(&mut self, lost: Lost) {
        if self
            .lost
            .is_none_or(|old| lost.priority() >= old.priority())
        {
            self.lost = Some(lost);
        }
    }

    /// Reads the field once and keeps the read if it still holds the text.
    /// `None` when the gate is closed (nothing was read); else whether the
    /// field is gone, emptied, or shorter than the read before.
    fn read(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Option<bool> {
        if !gate() {
            return None;
        }
        self.last_read = Some(now);
        self.due = None;
        let observed_at_ms = self.ms(now);
        let text = match self.field.read() {
            Ok(text) => text,
            Err(error) => {
                self.reads += 1;
                self.lose(error.into());
                return Some(error == ReadError::ElementGone);
            }
        };
        let changed = self.last_text.as_deref() != Some(text.as_str());
        self.reads += usize::from(changed);
        let blank = locate::is_blank(&text);
        let shrank = self
            .last_text
            .as_ref()
            .is_some_and(|before| text.chars().count() < before.chars().count());
        let outcome = if blank {
            self.lose(Lost::ClearedBeforeRead);
            None
        } else {
            match locate::align(&text, &self.inserted) {
                Alignment::Unchanged => Some(Outcome::Unchanged),
                Alignment::Edited(span) => Some(Outcome::Edited(span)),
                Alignment::NotFound => {
                    self.lose(Lost::SpanNotFound);
                    None
                }
            }
        };
        if let Some(outcome) = outcome {
            self.best = Some(CorrectionObservation {
                outcome,
                trigger,
                observed_at_ms,
            });
        }
        self.last_text = Some(text);
        Some(blank || shrank)
    }

    /// The last read that still held the text, else why none did.
    fn observation(&mut self, now: Instant, trigger: Trigger) -> CorrectionObservation {
        self.best.take().unwrap_or_else(|| CorrectionObservation {
            outcome: Outcome::Lost(self.lost.unwrap_or(Lost::ReadTimeout)),
            trigger,
            observed_at_ms: self.ms(now),
        })
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
        fn read(&self) -> Result<String, ReadError> {
            self.calls.set(self.calls.get() + 1);
            let mut reads = self.reads.borrow_mut();
            if reads.len() > 1 {
                reads.pop().unwrap()
            } else {
                reads[0].clone()
            }
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
        // Select Kip, type K, i, t, press Return: the field is cleared when
        // the Return read lands.
        let mut session = Session::new(
            Script::new(&["send it to K", "send it to Ki", "send it to Kit", ""]),
            "send it to Kip".into(),
            t0,
        );
        for ms in [0, 120, 240] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        let seen = done(session.step(at(t0, 300), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.observed_at_ms, 240);
    }

    #[test]
    fn a_burst_inside_one_gap_is_read_once_it_has_passed() {
        let t0 = Instant::now();
        let mut session = Session::new(
            Script::new(&["send it to K", "send it to Kit", ""]),
            "send it to Kip".into(),
            t0,
        );
        for ms in [0, 40, 80] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        assert_eq!(
            session.field.calls.get(),
            1,
            "only the first change at once"
        );
        session.step(at(t0, 99), Event::Tick, OPEN);
        assert_eq!(session.field.calls.get(), 1);
        session.step(at(t0, 100), Event::Tick, OPEN);
        assert_eq!(session.field.calls.get(), 2);
        let seen = done(session.step(at(t0, 150), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.observed_at_ms, 100);
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
        );
        session.step(at(t0, 10), Event::ValueChanged, OPEN);
        let seen = done(session.step(at(t0, 900), Event::Return, OPEN));
        assert!(matches!(seen.outcome, Outcome::Edited(_)));
        assert_eq!(seen.trigger, Trigger::Return);
    }

    #[test]
    fn nothing_matching_reports_why() {
        let t0 = Instant::now();
        let mut cleared = Session::new(Script::new(&[""]), "send it".into(), t0);
        let seen = done(cleared.step(at(t0, 5), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
        let mut other = Session::new(
            Script::new(&["unrelated words here", "unrelated words here"]),
            "send it".into(),
            t0,
        );
        assert_eq!(other.step(at(t0, 5), Event::Return, OPEN), Step::Wait);
        let seen = done(other.step(at(t0, 400), Event::FocusOut, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
        let gone = Script::of(vec![Err(ReadError::ElementGone)]);
        let mut gone = Session::new(gone, "send it".into(), t0);
        let seen = done(gone.step(at(t0, 5), Event::Return, OPEN));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ElementGone));
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
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0);
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
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0);
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
        let texts: Vec<String> = (0..100).map(|n| format!("send it {n}")).collect();
        let reads = texts.iter().rev().map(|t| Ok(t.clone())).collect();
        let mut session = Session::new(Script::of(reads), "send it".into(), t0);
        for n in 0..100 {
            session.step(at(t0, n * 1_000), Event::ValueChanged, OPEN);
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
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0);
        for n in 0..100 {
            session.step(at(t0, n * 1_000), Event::ValueChanged, OPEN);
        }
        assert_eq!(session.field.calls.get(), 100);
        assert_eq!(session.reads(), 1, "only the first read found a change");
    }

    #[test]
    fn the_pastes_own_focus_change_is_ignored() {
        let t0 = Instant::now();
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0);
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
        let lost = seen(Outcome::Lost(Lost::Secure)).into_history();
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
