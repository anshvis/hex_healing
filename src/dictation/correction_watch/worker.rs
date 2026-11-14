//! The watcher thread: one thread with a run loop that owns every watched
//! target's `AXObserver`, so callbacks never run on the main or audio thread.
//! It starts with the first queued target, sleeps when there is none, and
//! drops every observer and element once Hex begins quitting.

use std::sync::Mutex;
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use super::session::{CorrectionObservation, Event, Session, Step};
use super::{PasteTarget, QUEUE, enabled};
use crate::accessibility::{self, Notification, Observer, RetainedElement};

/// Targets watched at once; the rest wait in the queue.
pub const MAX_ACTIVE: usize = 4;
const TICK: Duration = Duration::from_millis(50);

/// Where a finished observation goes: `history::corrections::install_sink`
/// stores it on the History entry.
pub type Sink = fn(entry_id: u64, CorrectionObservation);

static SINK: std::sync::OnceLock<Sink> = std::sync::OnceLock::new();

/// Installs the sink. The first call wins.
pub fn set_sink(sink: Sink) {
    let _ = SINK.set(sink);
}

static THREAD: Mutex<Option<Thread>> = Mutex::new(None);

/// Wakes the watcher, starting it the first time.
pub fn wake() {
    let mut thread = THREAD.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(thread) = thread.as_ref() {
        thread.unpark();
        return;
    }
    let spawned = thread::Builder::new()
        .name("correction-watch".into())
        .spawn(run);
    match spawned {
        Ok(handle) => *thread = Some(JoinHandle::thread(&handle).clone()),
        Err(error) => tracing::warn!(%error, "could not start the correction watcher"),
    }
}

struct Active {
    entry_id: u64,
    session: Session<RetainedElement>,
    observer: Option<Observer>,
}

impl Active {
    fn start(target: PasteTarget) -> Self {
        let observer = Observer::new(target.pid, &target.element);
        Self {
            entry_id: target.entry_id,
            session: Session::new(target.element, target.inserted, target.pasted_at),
            observer,
        }
    }
}

fn run() {
    let mut active: Vec<Active> = Vec::new();
    let mut return_was_down = false;
    loop {
        if crate::desktop::process_exit::quitting() {
            drop_all(&mut active);
            return;
        }
        while active.len() < MAX_ACTIVE {
            let Some(target) = QUEUE.take() else { break };
            // The lab turned off, or the field is Hex's own: drop unread.
            if enabled() && target.pid != std::process::id() as i32 {
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
        let return_down = accessibility::return_down();
        let returned = return_down && !return_was_down;
        return_was_down = return_down;
        let now = Instant::now();
        active.retain_mut(|watched| !advance(watched, now, returned));
    }
}

/// Feeds one target this tick's events. `true` when it is finished.
fn advance(watched: &mut Active, now: Instant, returned: bool) -> bool {
    let mut events = Vec::new();
    if let Some(observer) = &watched.observer {
        for notification in observer.drain() {
            events.push(match notification {
                Notification::ValueChanged => Event::ValueChanged,
                Notification::FocusChanged => Event::FocusOut,
            });
        }
    }
    if returned {
        events.push(Event::Return);
    }
    events.push(Event::Tick);
    for event in events {
        match watched.session.step(now, event, &enabled) {
            Step::Wait => {}
            Step::Drop => return true,
            Step::Done(seen) => {
                // Counts and outcome IDs only, never text.
                tracing::info!(
                    outcome = seen.outcome.as_str(),
                    trigger = seen.trigger.as_str(),
                    chars = seen.edited_text.as_deref().map_or(0, |t| t.chars().count()),
                    ms = seen.observed_at_ms,
                    "correction observed"
                );
                if let Some(sink) = SINK.get() {
                    sink(watched.entry_id, seen);
                }
                return true;
            }
        }
    }
    false
}

/// Unregisters and releases everything: watched targets and queued ones.
fn drop_all<T>(active: &mut Vec<T>) {
    active.clear();
    QUEUE.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictation::correction_watch::Pending;

    #[test]
    fn quit_drops_pending() {
        let target = || Pending::new(1, RetainedElement::for_test()).into_target(1, "x".into());
        QUEUE.push(target());
        QUEUE.push(target());
        let mut active = vec![target(), target()];
        drop_all(&mut active);
        assert!(active.is_empty());
        assert_eq!(QUEUE.len(), 0);
    }
}
