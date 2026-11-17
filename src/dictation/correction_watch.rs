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

#[cfg(test)]
mod cell_fuzz_tests;
#[cfg(test)]
mod fuzz_tests;
#[cfg(test)]
mod glue_fuzz_tests;
mod locate;
#[cfg(test)]
mod origin_tests;
#[cfg(test)]
mod property_tests;
#[cfg(test)]
mod recall_tests;
use locate::Anchor;
mod session;
mod worker;

/// Targets waiting for the watcher. A full queue drops the oldest.
pub const QUEUE_CAPACITY: usize = 8;

/// A pasted dictation and where it landed. Dropping it releases the element.
pub struct PasteTarget {
    /// The field it went into, read before the paste.
    pub pending: Pending,
    /// What reached it, and when.
    pub landed: Landed,
    /// The History that kept the dictation, and the entry that did.
    pub history: History,
    pub entry_id: u64,
    /// The dictation's final text, which the History entry must still hold
    /// for the observation to be stored on it.
    pub text: String,
}

/// The field a paste is about to land in, read before the paste.
pub struct Pending {
    /// The process the text goes into.
    pub pid: i32,
    pub element: RetainedElement,
    /// Where the caret is (a paste starts there), if the app says.
    pub caret: Option<usize>,
    /// The field's text, when it could be read whole: the paste is found by
    /// what it added to it.
    pub before: Option<String>,
}

impl Pending {
    pub fn new(
        pid: i32,
        element: RetainedElement,
        caret: Option<usize>,
        before: Option<String>,
    ) -> Self {
        Self {
            pid,
            element,
            caret,
            before,
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
        if !crate::labs::learn_from_corrections() {
            return None;
        }
        let pid = crate::dictation::context::frontmost_pid()?;
        let element = accessibility::focused_element(pid)?;
        let caret = accessibility::caret(&element);
        let before = accessibility::read_value_bounded(&element, session::READ_CHARS)
            .ok()
            .filter(|read| !read.windowed)
            .map(|read| read.text);
        Some(Pending::new(pid, element, caret, before))
    }

    fn submit(&mut self, target: PasteTarget) {
        let dropped = QUEUE.push(target);
        if dropped > 0 {
            tracing::debug!(dropped, "correction watch dropped");
        }
        worker::wake();
    }
}

/// What the paste put in the field, and when.
pub struct Landed {
    /// The text that reached the field after the continuation join and smart
    /// spacing, not the dictation's final text.
    pub inserted: String,
    /// Where in the field it went, if the app says where the caret was.
    pub anchor: Option<Anchor>,
    /// When the paste completed, before History's write.
    pub at: Instant,
    /// The listener's Return count then: a Return after it may be a send.
    pub returns: u64,
}

enum State {
    Off,
    Before(Pending),
    Pasted(Pending, Landed),
}

/// One paste's watch, around the paste: [`Watch::before`] reads the field,
/// [`Watch::pasted`] notes what landed and when, [`Watch::recorded`] submits
/// it once History kept the dictation. Off when there is nothing to watch.
pub struct Watch(State);

impl Default for Watch {
    fn default() -> Self {
        Self(State::Off)
    }
}

impl Watch {
    /// Reads the field the paste goes into, before it: focus may move
    /// because of the paste. Nothing is read without a History that records.
    pub fn before(handoff: &mut impl Handoff, history: Option<&History>) -> Self {
        let pending = history
            .filter(|history| history.records())
            .and_then(|_| handoff.read());
        Self(pending.map_or(State::Off, State::Before))
    }

    /// The paste completed now, having put `inserted` in the field. The
    /// caret read before it says where that is.
    pub fn pasted(&mut self, inserted: String) {
        let State::Before(pending) = std::mem::replace(&mut self.0, State::Off) else {
            return;
        };
        if inserted.is_empty() {
            return;
        }
        let anchor = pending.caret.map(|start| Anchor { start });
        let landed = Landed {
            inserted,
            anchor,
            at: Instant::now(),
            returns: key_counts::returns(),
        };
        self.0 = State::Pasted(pending, landed);
    }

    /// Submits the target if History kept the dictation as `entry_id`.
    pub fn recorded(
        self,
        handoff: &mut impl Handoff,
        history: Option<&History>,
        entry_id: Option<u64>,
        text: &str,
    ) {
        if let (State::Pasted(pending, landed), Some(history), Some(entry_id)) =
            (self.0, history, entry_id)
        {
            handoff.submit(PasteTarget {
                pending,
                landed,
                history: history.clone(),
                entry_id,
                text: text.to_string(),
            });
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
pub(crate) fn history_for_test() -> History {
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

/// A target over a retained element that names no field, for tests.
#[cfg(test)]
fn target_for_test(history: &History, entry_id: u64) -> PasteTarget {
    PasteTarget {
        pending: Pending::new(1, RetainedElement::for_test(), None, None),
        landed: Landed {
            inserted: "pasted".into(),
            anchor: None,
            at: Instant::now(),
            returns: 0,
        },
        history: history.clone(),
        entry_id,
        text: "pasted".into(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::policy::{Lock, Policy};

    fn is_off(watch: &Watch) -> bool {
        matches!(watch.0, State::Off)
    }

    #[test]
    fn lab_off_reads_nothing() {
        // Set it off here rather than read the saved settings, which are the
        // machine's own (nextest gives each test its own process, so this
        // can't leak into another test).
        crate::labs::apply(&crate::app_settings::LabsSettings::default());
        assert!(Live.read().is_none(), "no Accessibility read while off");
        let history = history_for_test();
        assert!(is_off(&Watch::before(&mut Live, Some(&history))));
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
            assert!(Live.read().is_none());
        }
        // The same settings without the lock turn the gate on.
        let _free = Policy::forced_for_test(Policy::default());
        crate::labs::apply(&on);
        assert!(crate::labs::learn_from_corrections());
    }

    struct Fake {
        caret: Option<usize>,
        submitted: Vec<PasteTarget>,
    }

    impl Handoff for Fake {
        fn read(&mut self) -> Option<Pending> {
            Some(Pending::new(
                1,
                RetainedElement::for_test(),
                self.caret,
                None,
            ))
        }
        fn submit(&mut self, target: PasteTarget) {
            self.submitted.push(target);
        }
    }

    #[test]
    fn no_history_or_history_off_reads_nothing() {
        struct Panics;
        impl Handoff for Panics {
            fn read(&mut self) -> Option<Pending> {
                panic!("the field was read");
            }
            fn submit(&mut self, _: PasteTarget) {}
        }
        assert!(is_off(&Watch::before(&mut Panics, None)));
        let history = history_for_test();
        history
            .set_retention(crate::history::HistoryRetention::Off)
            .unwrap();
        assert!(is_off(&Watch::before(&mut Panics, Some(&history))));
    }

    #[test]
    fn the_target_carries_what_was_pasted_when_and_where() {
        let history = history_for_test();
        let mut fake = Fake {
            caret: Some(10),
            submitted: Vec::new(),
        };
        let mut watch = Watch::before(&mut fake, Some(&history));
        std::thread::sleep(std::time::Duration::from_millis(20));
        // A six-character paste at the caret.
        watch.pasted(" hello".into());
        let State::Pasted(_, landed) = &watch.0 else {
            panic!("the paste is noted");
        };
        let pasted_at = landed.at;
        assert_eq!(landed.anchor, Some(Anchor { start: 10 }));
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut again = Watch::before(&mut fake, Some(&history));
        again.pasted(" hello".into());
        let State::Pasted(_, landed) = &again.0 else {
            panic!("the paste is noted");
        };
        assert!(landed.at > pasted_at, "the time is taken at each paste");
        watch.recorded(&mut fake, Some(&history), None, "hello");
        assert!(fake.submitted.is_empty(), "no entry, no target");
        again.recorded(&mut fake, Some(&history), Some(7), "hello");
        assert_eq!(fake.submitted.len(), 1);
        assert_eq!(fake.submitted[0].entry_id, 7);
        assert_eq!(fake.submitted[0].landed.inserted, " hello");
    }

    #[test]
    fn a_field_with_no_caret_range_has_no_anchor_and_an_empty_paste_no_watch() {
        let history = history_for_test();
        let mut fake = Fake {
            caret: None,
            submitted: Vec::new(),
        };
        let mut watch = Watch::before(&mut fake, Some(&history));
        watch.pasted("hi".into());
        watch.recorded(&mut fake, Some(&history), Some(1), "hi");
        assert_eq!(fake.submitted[0].landed.anchor, None);
        let mut empty = Watch::before(&mut fake, Some(&history));
        empty.pasted(String::new());
        assert!(is_off(&empty));
    }

    #[test]
    fn queue_bounded_drops_oldest() {
        let history = history_for_test();
        let queue = Queue::default();
        for id in 0..QUEUE_CAPACITY as u64 {
            assert_eq!(queue.push(target_for_test(&history, id)), 0);
        }
        assert_eq!(queue.push(target_for_test(&history, 100)), 1);
        assert_eq!(queue.push(target_for_test(&history, 101)), 1);
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
