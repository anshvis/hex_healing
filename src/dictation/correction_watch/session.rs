//! One watched paste: when to read the field, which read to keep, and what
//! to report. Pure over a [`Field`], so the rules are tested without
//! Accessibility.

use std::time::{Duration, Instant};

use super::locate::{self, Alignment};
use crate::accessibility::ReadError;

/// How long a paste is watched.
pub const WATCH: Duration = Duration::from_secs(30);
/// A changed value is read at most this often (the first change at once).
pub const DEBOUNCE: Duration = Duration::from_secs(1);
/// Value-change reads per target; the closing read is not counted.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lost {
    ElementGone,
    Secure,
    TooLarge,
    SpanNotFound,
    ReadTimeout,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Unchanged,
    Edited,
    Lost(Lost),
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unchanged => "unchanged",
            Self::Edited => "edited",
            Self::Lost(reason) => reason.as_str(),
        }
    }
}

/// What the watcher saw of one paste. T2.3 stores it on the History entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrectionObservation {
    pub outcome: Outcome,
    pub trigger: Trigger,
    /// Milliseconds from the paste to the read this comes from.
    pub observed_at_ms: u64,
    /// The field's aligned span, only for `Edited`, at most
    /// `locate::span_cap(inserted length)` characters.
    pub edited_text: Option<String>,
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

struct Matched {
    edited: Option<String>,
    trigger: Trigger,
    ms: u64,
}

pub struct Session<F: Field> {
    field: F,
    inserted: String,
    started: Instant,
    reads: usize,
    last_read: Option<Instant>,
    due: Option<Instant>,
    best: Option<Matched>,
    cleared: bool,
    not_found: bool,
    error: Option<ReadError>,
}

impl<F: Field> Session<F> {
    pub fn new(field: F, inserted: String, started: Instant) -> Self {
        Self {
            field,
            inserted,
            started,
            reads: 0,
            last_read: None,
            due: None,
            best: None,
            cleared: false,
            not_found: false,
            error: None,
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
                match self
                    .last_read
                    .map(|at| (at + DEBOUNCE).saturating_duration_since(now))
                {
                    Some(wait) if !wait.is_zero() => self.due = Some(now + wait),
                    _ => {
                        if !self.read(now, Trigger::ValueChanged, gate) {
                            return Step::Drop;
                        }
                    }
                }
                Step::Wait
            }
            Event::Tick => {
                if self.due.is_some_and(|due| due <= now)
                    && !self.read(now, Trigger::ValueChanged, gate)
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
            Event::Return => self.finish(now, Trigger::Return, gate),
        }
    }

    fn finish(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> Step {
        if !self.read(now, trigger, gate) {
            return Step::Drop;
        }
        Step::Done(self.observation(now, trigger))
    }

    /// Reads the field once and keeps the read if it still holds the text.
    /// `false` when the gate is closed (nothing was read).
    fn read(&mut self, now: Instant, trigger: Trigger, gate: &dyn Fn() -> bool) -> bool {
        if !gate() {
            return false;
        }
        self.reads += 1;
        self.last_read = Some(now);
        self.due = None;
        let ms = now.saturating_duration_since(self.started).as_millis() as u64;
        match self.field.read() {
            Err(error) => self.error = Some(error),
            Ok(text) if text.replace('\u{FFFC}', "").trim().is_empty() => self.cleared = true,
            Ok(text) => match locate::align(&text, &self.inserted) {
                Alignment::Unchanged => {
                    self.best = Some(Matched {
                        edited: None,
                        trigger,
                        ms,
                    });
                }
                Alignment::Edited(span) => {
                    self.best = Some(Matched {
                        edited: Some(span),
                        trigger,
                        ms,
                    });
                }
                Alignment::NotFound => self.not_found = true,
            },
        }
        true
    }

    /// The last read that still held the text, else why none did.
    fn observation(&mut self, now: Instant, trigger: Trigger) -> CorrectionObservation {
        if let Some(best) = self.best.take() {
            let outcome = if best.edited.is_some() {
                Outcome::Edited
            } else {
                Outcome::Unchanged
            };
            return CorrectionObservation {
                outcome,
                trigger: best.trigger,
                observed_at_ms: best.ms,
                edited_text: best.edited,
            };
        }
        let lost = if self.cleared {
            Lost::ClearedBeforeRead
        } else if self.not_found {
            Lost::SpanNotFound
        } else {
            match self.error {
                Some(ReadError::ElementGone) => Lost::ElementGone,
                Some(ReadError::Secure) => Lost::Secure,
                Some(ReadError::TooLarge) => Lost::TooLarge,
                Some(ReadError::Timeout) | None => Lost::ReadTimeout,
            }
        };
        CorrectionObservation {
            outcome: Outcome::Lost(lost),
            trigger,
            observed_at_ms: now.saturating_duration_since(self.started).as_millis() as u64,
            edited_text: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    /// A field that answers a script of reads, repeating the last.
    struct Script(RefCell<Vec<Result<String, ReadError>>>);

    impl Script {
        fn new(reads: &[&str]) -> Self {
            Self(RefCell::new(
                reads.iter().rev().map(|t| Ok((*t).into())).collect(),
            ))
        }
    }

    impl Field for Script {
        fn read(&self) -> Result<String, ReadError> {
            let mut reads = self.0.borrow_mut();
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
        let Step::Done(seen) = session.step(at(t0, 3_000), Event::Return, OPEN) else {
            panic!("Return finishes");
        };
        assert_eq!(seen.outcome, Outcome::Edited);
        assert_eq!(seen.trigger, Trigger::ValueChanged);
        assert_eq!(seen.observed_at_ms, 2_000);
        assert_eq!(seen.edited_text.as_deref(), Some("send it to Kit"));
    }

    #[test]
    fn nothing_matching_reports_why() {
        let t0 = Instant::now();
        let mut cleared = Session::new(Script::new(&[""]), "send it".into(), t0);
        let Step::Done(seen) = cleared.step(at(t0, 5), Event::Return, OPEN) else {
            panic!()
        };
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
        let mut other = Session::new(Script::new(&["unrelated words here"]), "send it".into(), t0);
        let Step::Done(seen) = other.step(at(t0, 5), Event::Return, OPEN) else {
            panic!()
        };
        assert_eq!(seen.outcome, Outcome::Lost(Lost::SpanNotFound));
        let gone = Script(RefCell::new(vec![Err(ReadError::ElementGone)]));
        let mut gone = Session::new(gone, "send it".into(), t0);
        let Step::Done(seen) = gone.step(at(t0, 5), Event::Return, OPEN) else {
            panic!()
        };
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ElementGone));
    }

    #[test]
    fn timeout_closes_an_untouched_paste_as_unchanged() {
        let t0 = Instant::now();
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0);
        assert_eq!(session.step(at(t0, 29_000), Event::Tick, OPEN), Step::Wait);
        let Step::Done(seen) = session.step(at(t0, 30_000), Event::Tick, OPEN) else {
            panic!()
        };
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
        assert_eq!(session.reads(), 1);
        open.set(false);
        assert_eq!(
            session.step(at(t0, 5_000), Event::ValueChanged, &gate),
            Step::Drop
        );
        assert_eq!(
            session.step(at(t0, 6_000), Event::Return, &gate),
            Step::Drop
        );
        assert_eq!(session.reads(), 1, "a closed gate reads nothing");
    }

    #[test]
    fn debounces_and_caps_reads_per_target() {
        let t0 = Instant::now();
        let mut session = Session::new(Script::new(&["send it"]), "send it".into(), t0);
        // A storm inside one second reads once, then once more when due.
        for ms in [0, 100, 200, 300] {
            session.step(at(t0, ms), Event::ValueChanged, OPEN);
        }
        assert_eq!(session.reads(), 1);
        session.step(at(t0, 900), Event::Tick, OPEN);
        assert_eq!(session.reads(), 1);
        session.step(at(t0, 1_000), Event::Tick, OPEN);
        assert_eq!(session.reads(), 2);
        for n in 0..100 {
            session.step(at(t0, 2_000 + n * 1_000), Event::ValueChanged, OPEN);
        }
        assert_eq!(session.reads(), MAX_READS);
        assert!(matches!(
            session.step(at(t0, 200_000), Event::Return, OPEN),
            Step::Done(_)
        ));
        assert_eq!(session.reads(), MAX_READS + 1, "the closing read is extra");
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
}
