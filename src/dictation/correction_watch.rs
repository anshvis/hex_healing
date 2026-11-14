//! Labs › Learn from corrections: the handoff from a successful paste to the
//! correction watcher (macOS).
//!
//! With the lab on, the paste worker reads the focused element of the app it
//! is about to paste into ([`Handoff::read`]), and once the paste and its
//! History record both succeed it submits a [`PasteTarget`]. The queue here
//! holds at most [`QUEUE_CAPACITY`] targets and drops the oldest; the timer
//! and the re-read of the field belong to the watcher that drains it. Nothing
//! is read, retained or queued while the lab is off or locked, and nothing
//! here logs or serializes text: a [`PasteTarget`] lives in memory only.
//! See `docs/features/corrections.md` for the privacy contract.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use crate::accessibility::{self, RetainedElement};
use crate::policy::{Lock, Policy};

/// Targets waiting for the watcher. A full queue drops the oldest.
pub const QUEUE_CAPACITY: usize = 8;

/// A pasted dictation and where it landed. Dropping it releases the element.
// The watcher that reads these fields lands with the timer.
#[allow(dead_code)]
pub struct PasteTarget {
    /// The process the text was pasted into.
    pub pid: i32,
    /// The focused element read before the paste.
    pub element: RetainedElement,
    /// The History entry recording the dictation.
    pub entry_id: u64,
    /// The text that reached the field after the continuation join and smart
    /// spacing, not the dictation's final text.
    pub inserted: String,
    pub pasted_at: Instant,
}

/// The element a paste is about to land in, read before the paste.
pub struct Pending {
    pid: i32,
    element: RetainedElement,
}

impl Pending {
    pub fn new(pid: i32, element: RetainedElement) -> Self {
        Self { pid, element }
    }

    /// The target once the paste inserted `inserted` and History kept `entry_id`.
    pub fn into_target(self, entry_id: u64, inserted: String) -> PasteTarget {
        PasteTarget {
            pid: self.pid,
            element: self.element,
            entry_id,
            inserted,
            pasted_at: Instant::now(),
        }
    }
}

/// What the paste worker needs from the watcher, so a test can stand in.
pub trait Handoff {
    /// The element about to be pasted into. `None` when the lab is off or
    /// locked, or when no readable, non-secure field has focus.
    fn read(&mut self) -> Option<Pending>;
    fn submit(&mut self, target: PasteTarget);
}

/// The real handoff: reads through Accessibility and queues in [`QUEUE`].
pub struct Live;

impl Handoff for Live {
    fn read(&mut self) -> Option<Pending> {
        if !enabled() {
            return None;
        }
        let pid = crate::dictation::context::frontmost_pid()?;
        accessibility::focused_element(pid).map(|element| Pending::new(pid, element))
    }

    fn submit(&mut self, target: PasteTarget) {
        submit(target);
    }
}

/// Whether the lab is on and no managed lock holds it off.
fn enabled() -> bool {
    allowed(crate::labs::learn_from_corrections(), Policy::current())
}

fn allowed(lab_on: bool, policy: Policy) -> bool {
    lab_on && !policy.locks(Lock::Corrections)
}

/// Bounded, oldest-dropped queue of targets.
#[derive(Default)]
struct Queue(Mutex<VecDeque<PasteTarget>>);

impl Queue {
    /// Adds `target`. Returns how many older targets were dropped to fit it.
    fn push(&self, target: PasteTarget) -> usize {
        let mut targets = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let mut dropped = 0;
        while targets.len() >= QUEUE_CAPACITY {
            targets.pop_front();
            dropped += 1;
        }
        targets.push_back(target);
        dropped
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }
}

static QUEUE: Queue = Queue(Mutex::new(VecDeque::new()));

/// Queues `target` for the watcher while the lab is on and not locked. Off,
/// it is dropped, which releases its element.
pub fn submit(target: PasteTarget) {
    submit_to(enabled(), &QUEUE, target);
}

fn submit_to(allowed: bool, queue: &Queue, target: PasteTarget) {
    if !allowed {
        return;
    }
    let dropped = queue.push(target);
    if dropped > 0 {
        tracing::debug!(dropped, "correction watch dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(entry_id: u64) -> PasteTarget {
        Pending::new(1, RetainedElement::for_test()).into_target(entry_id, "pasted".into())
    }

    #[test]
    fn lab_off_submits_nothing() {
        let queue = Queue::default();
        submit_to(allowed(false, Policy::default()), &queue, target(1));
        assert_eq!(queue.len(), 0);
        // The real gate reads the lab, which is off by default.
        let before = QUEUE.len();
        submit(target(2));
        assert_eq!(QUEUE.len(), before);
        assert!(Live.read().is_none(), "no Accessibility read while off");
    }

    #[test]
    fn policy_lock_submits_nothing() {
        let locked = Policy::default().with(Lock::Corrections);
        assert!(!allowed(true, locked));
        let queue = Queue::default();
        submit_to(allowed(true, locked), &queue, target(1));
        assert_eq!(queue.len(), 0);
        assert!(allowed(true, Policy::default()));
        submit_to(allowed(true, Policy::default()), &queue, target(2));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn queue_bounded_drops_oldest() {
        let queue = Queue::default();
        for id in 0..QUEUE_CAPACITY as u64 {
            assert_eq!(queue.push(target(id)), 0);
        }
        assert_eq!(queue.push(target(100)), 1);
        assert_eq!(queue.push(target(101)), 1);
        assert_eq!(queue.len(), QUEUE_CAPACITY);
        let ids: Vec<u64> = queue
            .0
            .lock()
            .unwrap()
            .iter()
            .map(|target| target.entry_id)
            .collect();
        assert_eq!(ids, [2, 3, 4, 5, 6, 7, 100, 101]);
    }
}
