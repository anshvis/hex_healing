//! Labs › Learn from corrections: the handoff from a successful paste to the
//! correction watcher (macOS).
//!
//! With the lab on, the paste worker reads the focused element of the app it
//! is about to paste into ([`Watch::before`]), and once the paste and its
//! History record both succeed it submits a [`PasteTarget`]. The queue here
//! holds at most [`QUEUE_CAPACITY`] targets and drops the oldest; the timer
//! and the re-read of the field belong to the watcher that drains it, which
//! stores what it saw on the target's History entry. Nothing is read,
//! retained or queued while the lab is off or locked (one gate,
//! [`crate::labs::learn_from_corrections`], which includes the managed lock),
//! and nothing here logs or serializes text: a [`PasteTarget`] lives in
//! memory only. See `docs/features/corrections.md` for the privacy contract.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use crate::accessibility::{self, RetainedElement};
use crate::dictation::key_counts;
use crate::history::History;

mod locate;
mod session;
mod worker;

/// Targets waiting for the watcher. A full queue drops the oldest.
pub const QUEUE_CAPACITY: usize = 8;

/// A pasted dictation and where it landed. Dropping it releases the element.
pub struct PasteTarget {
    /// The process the text was pasted into.
    pub pid: i32,
    /// The focused element read before the paste.
    pub element: RetainedElement,
    /// The History that kept the dictation, and the entry that did.
    pub history: History,
    pub entry_id: u64,
    /// The text that reached the field after the continuation join and smart
    /// spacing, not the dictation's final text.
    pub inserted: String,
    /// When the paste completed, before History's write.
    pub pasted_at: Instant,
    /// The listener's Return count at the paste: a Return after it may be a
    /// send.
    pub returns: u64,
}

/// The element a paste is about to land in, read before the paste.
pub struct Pending {
    pid: i32,
    element: RetainedElement,
    inserted: String,
    pasted_at: Instant,
    returns: u64,
}

impl Pending {
    pub fn new(pid: i32, element: RetainedElement) -> Self {
        Self {
            pid,
            element,
            inserted: String::new(),
            pasted_at: Instant::now(),
            returns: key_counts::returns(),
        }
    }

    fn into_target(self, history: History, entry_id: u64) -> PasteTarget {
        PasteTarget {
            pid: self.pid,
            element: self.element,
            history,
            entry_id,
            inserted: self.inserted,
            pasted_at: self.pasted_at,
            returns: self.returns,
        }
    }
}

/// What the paste worker needs from the watcher, so a test can stand in.
pub trait Handoff {
    /// The element about to be pasted into. `None` when the lab is off or
    /// locked, or when no readable, non-secure field has focus.
    fn read(&mut self) -> Option<Pending>;
    /// What the paste just inserted into the field.
    fn inserted(&self) -> String;
    fn submit(&mut self, target: PasteTarget);
}

/// The real handoff: reads through Accessibility and queues in [`QUEUE`].
/// `inserted` is the paster's last insertion.
pub struct Live<'a>(pub &'a dyn Fn() -> String);

impl Handoff for Live<'_> {
    fn read(&mut self) -> Option<Pending> {
        if !crate::labs::learn_from_corrections() {
            return None;
        }
        let pid = crate::dictation::context::frontmost_pid()?;
        accessibility::focused_element(pid).map(|element| Pending::new(pid, element))
    }

    fn inserted(&self) -> String {
        (self.0)()
    }

    fn submit(&mut self, target: PasteTarget) {
        let dropped = QUEUE.push(target);
        if dropped > 0 {
            tracing::debug!(dropped, "correction watch dropped");
        }
        worker::wake();
    }
}

/// One paste's watch, around the paste: [`Watch::before`] reads the field,
/// [`Watch::pasted`] notes what landed and when, [`Watch::recorded`] submits
/// it once History kept the dictation. Empty when there is nothing to watch.
#[derive(Default)]
pub struct Watch(Option<Pending>);

impl Watch {
    /// Reads the field the paste goes into, before it: focus may move
    /// because of the paste. Nothing is read without a History that records.
    pub fn before(handoff: &mut impl Handoff, history: Option<&History>) -> Self {
        Self(
            history
                .filter(|history| history.records())
                .and_then(|_| handoff.read()),
        )
    }

    /// The paste completed now with `handoff.inserted()`.
    pub fn pasted(&mut self, handoff: &impl Handoff) {
        if let Some(pending) = &mut self.0 {
            pending.pasted_at = Instant::now();
            pending.returns = key_counts::returns();
            pending.inserted = handoff.inserted();
            if pending.inserted.is_empty() {
                self.0 = None;
            }
        }
    }

    /// Submits the target if History kept the dictation as `entry_id`.
    pub fn recorded(
        self,
        handoff: &mut impl Handoff,
        history: Option<&History>,
        entry_id: Option<u64>,
    ) {
        if let (Some(pending), Some(history), Some(entry_id)) = (self.0, history, entry_id) {
            handoff.submit(pending.into_target(history.clone(), entry_id));
        }
    }
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

    /// The oldest waiting target.
    fn take(&self) -> Option<PasteTarget> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pop_front()
    }

    /// Drops every waiting target, releasing their elements.
    fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
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

/// A History over a throwaway file, for tests.
#[cfg(test)]
fn history_for_test() -> History {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let directory = std::env::temp_dir().join(format!(
        "hex-correction-watch-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    History::new(crate::history::HistoryStore::open(
        directory.join("history.json"),
        crate::history::HistoryRetention::Week,
        crate::history::now_ms(),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::policy::{Lock, Policy};

    fn target(entry_id: u64) -> PasteTarget {
        Pending::new(1, RetainedElement::for_test()).into_target(history_for_test(), entry_id)
    }

    fn off() -> String {
        String::new()
    }

    #[test]
    fn lab_off_reads_nothing() {
        // Set it off here rather than read the saved settings, which are the
        // machine's own (nextest gives each test its own process, so this
        // can't leak into another test).
        crate::labs::apply(&crate::app_settings::LabsSettings::default());
        assert!(
            Live(&off).read().is_none(),
            "no Accessibility read while off"
        );
        let history = history_for_test();
        let watch = Watch::before(&mut Live(&off), Some(&history));
        assert!(watch.0.is_none());
    }

    #[test]
    fn policy_lock_reads_nothing_through_live() {
        let on = crate::app_settings::LabsSettings {
            learn_from_corrections: true,
            ..Default::default()
        };
        {
            let _forced = Policy::forced_for_test(Policy::default().with(Lock::Corrections));
            crate::labs::apply(&on);
            assert!(!crate::labs::learn_from_corrections());
            assert!(Live(&off).read().is_none());
        }
        // The same settings without the lock turn the gate on.
        let _free = Policy::forced_for_test(Policy::default());
        crate::labs::apply(&on);
        assert!(crate::labs::learn_from_corrections());
    }

    #[test]
    fn no_history_or_history_off_reads_nothing() {
        struct Panics;
        impl Handoff for Panics {
            fn read(&mut self) -> Option<Pending> {
                panic!("the field was read");
            }
            fn inserted(&self) -> String {
                String::new()
            }
            fn submit(&mut self, _: PasteTarget) {}
        }
        assert!(Watch::before(&mut Panics, None).0.is_none());
        let history = history_for_test();
        history
            .set_retention(crate::history::HistoryRetention::Off)
            .unwrap();
        assert!(Watch::before(&mut Panics, Some(&history)).0.is_none());
    }

    #[test]
    fn the_paste_time_and_inserted_text_come_from_the_paste() {
        struct Fake(Vec<u64>);
        impl Handoff for Fake {
            fn read(&mut self) -> Option<Pending> {
                Some(Pending::new(1, RetainedElement::for_test()))
            }
            fn inserted(&self) -> String {
                " hello".into()
            }
            fn submit(&mut self, target: PasteTarget) {
                assert_eq!(target.inserted, " hello");
                self.0.push(target.entry_id);
            }
        }
        let history = history_for_test();
        let mut fake = Fake(Vec::new());
        let mut watch = Watch::before(&mut fake, Some(&history));
        std::thread::sleep(std::time::Duration::from_millis(20));
        watch.pasted(&fake);
        let pasted_at = watch.0.as_ref().unwrap().pasted_at;
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut again = Watch::before(&mut fake, Some(&history));
        again.pasted(&fake);
        assert!(again.0.as_ref().unwrap().pasted_at > pasted_at);
        watch.recorded(&mut fake, Some(&history), None);
        assert!(fake.0.is_empty(), "no entry, no target");
        again.recorded(&mut fake, Some(&history), Some(7));
        assert_eq!(fake.0, [7]);
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
