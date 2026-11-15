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
use super::{PasteTarget, QUEUE, returns};
use crate::accessibility::{self, Notification, Observer, RetainedElement};
use crate::history::History;

/// Targets watched at once; the rest wait in the queue.
pub const MAX_ACTIVE: usize = 4;
const TICK: Duration = Duration::from_millis(50);

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
}

impl Active {
    fn start(target: PasteTarget) -> Self {
        let observer = Observer::new(target.pid, &target.element);
        Self {
            history: target.history,
            entry_id: target.entry_id,
            session: Session::new(target.element, target.inserted, target.pasted_at),
            observer,
            returns: returns(),
        }
    }

    /// What happened since the last tick, ending with the tick.
    fn events(&mut self) -> Vec<Event> {
        let mut events: Vec<Event> = self
            .observer
            .iter()
            .flat_map(Observer::drain)
            .map(|notification| match notification {
                Notification::ValueChanged => Event::ValueChanged,
                Notification::FocusChanged => Event::FocusOut,
            })
            .collect();
        let now = returns();
        if now != self.returns {
            self.returns = now;
            events.push(Event::Return);
        }
        events.push(Event::Tick);
        events
    }
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
            // Never watch Hex's own field. The gate is asked at each read.
            if target.pid != std::process::id() as i32 {
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
            let events = watched.events();
            match advance(&mut watched.session, events, now, &gate) {
                Advance::Watching => true,
                Advance::Dropped => false,
                Advance::Observed(seen) => {
                    keep(&watched.history, watched.entry_id, seen);
                    false
                }
            }
        });
    }
}

/// The one gate: the lab is on and no managed lock holds it off.
fn gate() -> bool {
    crate::labs::learn_from_corrections()
}

enum Advance {
    Watching,
    /// The gate closed: stop without a result.
    Dropped,
    Observed(CorrectionObservation),
}

/// Feeds `session` its `events` at `now`.
fn advance<F: Field>(
    session: &mut Session<F>,
    events: impl IntoIterator<Item = Event>,
    now: Instant,
    gate: &dyn Fn() -> bool,
) -> Advance {
    for event in events {
        match session.step(now, event, gate) {
            Step::Wait => {}
            Step::Drop => return Advance::Dropped,
            Step::Done(seen) => return Advance::Observed(seen),
        }
    }
    Advance::Watching
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
    use crate::accessibility::ReadError;
    use crate::dictation::correction_watch::session::{Lost, Trigger};
    use crate::dictation::correction_watch::{Pending, history_for_test};
    use crate::history::{HistoryDraft, HistoryKind};

    fn target(history: &History, entry_id: u64) -> PasteTarget {
        Pending::new(1, RetainedElement::for_test()).into_target(history.clone(), entry_id)
    }

    /// A field that reads as `text` until it is cleared.
    struct Fake<'a>(&'a Cell<&'static str>);

    impl Field for Fake<'_> {
        fn read(&self) -> Result<String, ReadError> {
            Ok(self.0.get().to_string())
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

    #[test]
    fn a_send_ends_the_watch_with_the_last_edited_read() {
        let field = Cell::new("send it to Kit");
        let t0 = now();
        let mut session = Session::new(Fake(&field), "send it to Kip".into(), t0);
        let gate: &dyn Fn() -> bool = &|| true;
        let step = |session: &mut Session<Fake>, events: &[Event], ms| {
            advance(
                session,
                events.iter().copied(),
                t0 + Duration::from_millis(ms),
                gate,
            )
        };
        assert!(matches!(
            step(&mut session, &[Event::ValueChanged, Event::Tick], 1_000),
            Advance::Watching
        ));
        field.set("");
        let Advance::Observed(seen) = step(&mut session, &[Event::Return, Event::Tick], 2_000)
        else {
            panic!("a cleared field on Return ends the watch");
        };
        assert_eq!(seen.outcome, Outcome::Edited("send it to Kit".into()));
        assert_eq!(seen.trigger, Trigger::ValueChanged);
    }

    #[test]
    fn a_closed_gate_drops_the_watch_unread() {
        let field = Cell::new("send it");
        let t0 = now();
        let mut session = Session::new(Fake(&field), "send it".into(), t0);
        let closed: &dyn Fn() -> bool = &|| false;
        let result = advance(&mut session, [Event::ValueChanged], t0, closed);
        assert!(matches!(result, Advance::Dropped));
        assert_eq!(session.reads(), 0);
    }

    #[test]
    fn events_after_a_finish_are_not_fed() {
        let field = Cell::new("");
        let t0 = now();
        let mut session = Session::new(Fake(&field), "send it".into(), t0);
        let gate: &dyn Fn() -> bool = &|| true;
        let result = advance(&mut session, [Event::Return, Event::ValueChanged], t0, gate);
        let Advance::Observed(seen) = result else {
            panic!("ends on the first Return");
        };
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
