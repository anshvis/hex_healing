//! The watcher thread: one thread with a run loop that owns every watched
//! target's `AXObserver`, so callbacks never run on the main or audio thread.
//! It starts with the first queued target, sleeps when there is none, and
//! drops every observer and element once Hex begins quitting. Each finished
//! observation goes to its target's History entry.

use std::panic::catch_unwind;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::session::{CorrectionObservation, Event, Field, Outcome, Session, Step};
use super::{PasteTarget, QUEUE};
use crate::accessibility::{self, Notification, Observer, RetainedElement};
use crate::dictation::key_counts;
use crate::history::History;

/// Targets watched at once; the rest wait in the queue.
pub const MAX_ACTIVE: usize = 4;
const TICK: Duration = Duration::from_millis(50);
/// How often a field that reports no value changes is read anyway.
const POLL: Duration = Duration::from_millis(400);

static THREAD: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

/// Wakes the watcher, starting it the first time and again if it died.
pub fn wake() {
    if crate::desktop::process_exit::quitting() {
        return;
    }
    let mut thread = THREAD.lock().unwrap_or_else(|e| e.into_inner());
    match thread.as_ref() {
        Some(handle) if !handle.is_finished() => return handle.thread().unpark(),
        Some(_) => tracing::warn!("the correction watcher stopped; starting it again"),
        None => {}
    }
    let spawned = thread::Builder::new()
        .name("correction-watch".into())
        .spawn(|| {
            if catch_unwind(run).is_err() {
                tracing::error!("the correction watcher panicked");
            }
        });
    match spawned {
        Ok(handle) => *thread = Some(handle),
        Err(error) => tracing::warn!(%error, "could not start the correction watcher"),
    }
}

struct Active {
    history: History,
    entry_id: u64,
    session: Session<RetainedElement>,
    observer: Option<Observer>,
    /// The listener's Return count when this watch last looked.
    returns: u64,
    /// When to read next, for a field whose changes are not reported.
    poll_at: Option<Instant>,
}

impl Active {
    fn start(target: PasteTarget) -> Self {
        let observer = Observer::new(target.pid, &target.element);
        Self {
            history: target.history,
            entry_id: target.entry_id,
            session: Session::new(
                target.element,
                target.inserted,
                target.pasted_at,
                target.anchor,
            ),
            returns: target.returns,
            poll_at: observer
                .as_ref()
                .is_none_or(|observer| !observer.watches_value())
                .then(|| Instant::now() + POLL),
            observer,
        }
    }

    /// What happened since the last tick.
    fn events(&mut self, now: Instant) -> Vec<Event> {
        // The Return count is sampled before the notifications are drained:
        // the listener counts a Return before the app handles it, so any
        // clear it causes is in the notifications that follow.
        let count = key_counts::returns();
        let returned = count != self.returns;
        self.returns = count;
        let notifications: Vec<Notification> =
            self.observer.iter().flat_map(Observer::drain).collect();
        let polled = self.poll_at.is_some_and(|at| at <= now);
        if polled {
            self.poll_at = Some(now + POLL);
        }
        order(returned, notifications, polled)
    }
}

/// The events of one tick, in the order the session needs: a Return first
/// (it enters the send window before the clear it causes is read), then what
/// the observer reported, a burst of value changes as one read of where the
/// field ended (a field that reports no changes is `polled` instead), then
/// the tick.
fn order(
    returned: bool,
    notifications: impl IntoIterator<Item = Notification>,
    polled: bool,
) -> Vec<Event> {
    let mut events = Vec::new();
    if returned {
        events.push(Event::Return);
    }
    for notification in notifications {
        events.push(match notification {
            Notification::ValueChanged => Event::ValueChanged,
            Notification::FocusChanged => Event::FocusOut,
        });
    }
    if polled {
        events.push(Event::ValueChanged);
    }
    events.dedup_by(|a, b| *a == Event::ValueChanged && *b == Event::ValueChanged);
    events.push(Event::Tick);
    events
}

fn run() {
    let mut active: Vec<Active> = Vec::new();
    loop {
        if crate::desktop::process_exit::quitting() {
            drop_all(&mut active);
            return;
        }
        while active.len() < MAX_ACTIVE {
            let Some(target) = QUEUE.take() else { break };
            // Never watch Hex's own field, nor start one the lab has since
            // turned off (it drops the element unread).
            if target.pid != std::process::id() as i32 && crate::labs::learn_from_corrections() {
                active.push(Active::start(target));
            }
        }
        if active.is_empty() {
            thread::park_timeout(Duration::from_secs(60));
            continue;
        }
        if !accessibility::run_loop_once(TICK.as_secs_f64()) {
            thread::sleep(TICK);
        }
        let now = Instant::now();
        active.retain_mut(|watched| {
            let events = watched.events(now);
            match feed(
                &mut watched.session,
                events,
                now,
                &crate::labs::learn_from_corrections,
            ) {
                Step::Wait => true,
                Step::Drop => false,
                Step::Done(seen) => {
                    keep(&watched.history, watched.entry_id, seen);
                    false
                }
            }
        });
    }
}

/// Feeds `session` its `events` at `now`; the first step that is not a wait.
fn feed<F: Field>(
    session: &mut Session<F>,
    events: impl IntoIterator<Item = Event>,
    now: Instant,
    gate: &dyn Fn() -> bool,
) -> Step {
    for event in events {
        match session.step(now, event, gate) {
            Step::Wait => {}
            step => return step,
        }
    }
    Step::Wait
}

/// Stores `seen` on its History entry. Counts and outcome IDs are logged,
/// never text.
fn keep(history: &History, entry_id: u64, seen: CorrectionObservation) {
    let chars = match &seen.outcome {
        Outcome::Edited(span) => span.chars().count(),
        _ => 0,
    };
    let (outcome, trigger, ms) = (
        seen.outcome.as_str(),
        seen.trigger.as_str(),
        seen.observed_at_ms,
    );
    match history.observe_correction(entry_id, seen.into_history()) {
        Ok(kept) => tracing::info!(outcome, trigger, chars, ms, kept, "correction observed"),
        Err(error) => tracing::warn!(%error, outcome, "could not keep a correction"),
    }
}

/// Unregisters and releases everything: watched targets and queued ones.
fn drop_all(active: &mut Vec<Active>) {
    active.clear();
    QUEUE.clear();
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::accessibility::{FieldText, ReadError};
    use crate::dictation::correction_watch::history_for_test;
    use crate::dictation::correction_watch::session::{Lost, Trigger};
    use crate::history::{HistoryDraft, HistoryKind};

    use crate::dictation::correction_watch::target_for_test as target;

    /// A field that reads as `text` until it is cleared.
    struct Fake<'a>(&'a Cell<&'static str>);

    impl Field for Fake<'_> {
        fn read(&self) -> Result<FieldText, ReadError> {
            Ok(FieldText {
                text: self.0.get().to_string(),
                windowed: false,
            })
        }
    }

    fn now() -> Instant {
        Instant::now()
    }

    #[test]
    fn quit_drops_pending() {
        let history = history_for_test();
        QUEUE.push(target(&history, 1));
        QUEUE.push(target(&history, 2));
        let mut active = vec![Active::start(target(&history, 3))];
        drop_all(&mut active);
        assert!(active.is_empty());
        assert_eq!(QUEUE.len(), 0);
    }

    const OPEN: &dyn Fn() -> bool = &|| true;

    fn at(t0: Instant, ms: u64) -> Instant {
        t0 + Duration::from_millis(ms)
    }

    /// A session over `field` that has seen the paste whole and then its edit.
    fn edited<'a>(field: &'a Cell<&'static str>, t0: Instant) -> Session<Fake<'a>> {
        let mut session = Session::new(Fake(field), "send it to Kip".into(), t0, None);
        field.set("send it to Kip");
        assert_eq!(
            feed(&mut session, order(false, [], false), at(t0, 50), OPEN),
            Step::Wait
        );
        field.set("send it to Kit");
        let changed = order(false, [Notification::ValueChanged], false);
        assert_eq!(feed(&mut session, changed, at(t0, 1_000), OPEN), Step::Wait);
        session
    }

    fn kept(step: Step) -> CorrectionObservation {
        match step {
            Step::Done(seen) => seen,
            other => panic!("expected the watch to end, got {other:?}"),
        }
    }

    #[test]
    fn a_send_ends_the_watch_with_the_last_edited_read() {
        let field = Cell::new("");
        let t0 = now();
        let mut session = edited(&field, t0);
        field.set("");
        let seen = kept(feed(
            &mut session,
            order(true, [], false),
            at(t0, 2_000),
            OPEN,
        ));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.trigger, Trigger::ValueChanged);
    }

    #[test]
    fn a_return_and_its_clear_in_one_drain_are_read_in_that_order() {
        let notifications = [Notification::ValueChanged, Notification::FocusChanged];
        assert_eq!(
            order(true, notifications, false),
            [
                Event::Return,
                Event::ValueChanged,
                Event::FocusOut,
                Event::Tick
            ]
        );
        assert_eq!(
            order(
                false,
                [Notification::ValueChanged, Notification::ValueChanged],
                true
            ),
            [Event::ValueChanged, Event::Tick],
            "a burst, and a poll on top of it, is one read"
        );
        // With the Return first the clear that follows it in the same drain
        // is the app clearing after a send: the edit is kept.
        let field = Cell::new("");
        let t0 = now();
        let mut session = edited(&field, t0);
        field.set("");
        let events = order(true, [Notification::ValueChanged], false);
        let seen = kept(feed(&mut session, events, at(t0, 2_000), OPEN));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
    }

    #[test]
    fn a_return_one_tick_after_its_clear_still_makes_it_a_send() {
        let field = Cell::new("");
        let t0 = now();
        let mut session = edited(&field, t0);
        // The clear is read on one tick, the Return the listener counts on the next.
        field.set("");
        let first = order(false, [Notification::ValueChanged], false);
        assert_eq!(feed(&mut session, first, at(t0, 2_000), OPEN), Step::Wait);
        let seen = kept(feed(
            &mut session,
            order(true, [], false),
            at(t0, 2_050),
            OPEN,
        ));
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        // Without a Return the grace passes and it is lost.
        let field = Cell::new("");
        let mut session = edited(&field, t0);
        field.set("");
        let first = order(false, [Notification::ValueChanged], false);
        assert_eq!(feed(&mut session, first, at(t0, 2_000), OPEN), Step::Wait);
        let seen = kept(feed(
            &mut session,
            order(false, [], false),
            at(t0, 2_150),
            OPEN,
        ));
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
    }

    #[test]
    fn a_closed_gate_drops_the_watch_unread() {
        let field = Cell::new("send it");
        let t0 = now();
        let mut session = Session::new(Fake(&field), "send it".into(), t0, None);
        let closed: &dyn Fn() -> bool = &|| false;
        let step = feed(&mut session, [Event::ValueChanged], t0, closed);
        assert_eq!(step, Step::Drop);
        assert_eq!(session.reads(), 0);
    }

    #[test]
    fn events_after_a_finish_are_not_fed() {
        let field = Cell::new("");
        let t0 = now();
        let mut session = Session::new(Fake(&field), "send it".into(), t0, None);
        let step = feed(&mut session, [Event::Return, Event::ValueChanged], t0, OPEN);
        let seen = kept(step);
        assert_eq!(seen.outcome, Outcome::Lost(Lost::ClearedBeforeRead));
        assert_eq!(session.reads(), 1);
    }

    #[test]
    fn keep_stores_the_observation_on_its_entry() {
        let history = history_for_test();
        let id = history
            .record(HistoryDraft {
                kind: HistoryKind::Dictation,
                raw_text: "Send it to Kip.".into(),
                final_text: "Send it to Kip.".into(),
                application: None,
                processing: None,
                audio_ms: 0,
                inference_ms: 0,
                total_ms: 0,
                recording: None,
                model: None,
                model_id: None,
            })
            .unwrap()
            .unwrap();
        keep(
            &history,
            id,
            CorrectionObservation {
                outcome: Outcome::Edited("Kit".into()),
                trigger: Trigger::FocusOut,
                observed_at_ms: 9,
            },
        );
        let kept = history.recent(1).remove(0).correction.unwrap();
        assert_eq!(kept.outcome, "edited");
        assert_eq!(kept.edited_text.as_deref(), Some("Kit"));
    }
}
