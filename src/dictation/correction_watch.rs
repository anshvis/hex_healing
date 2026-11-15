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

use crate::accessibility::{self, Caret, RetainedElement};
use crate::dictation::key_counts;
use crate::history::History;

mod locate;
use locate::Anchor;
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
    /// Where in the field it went, if the app says where the caret was.
    pub anchor: Option<Anchor>,
    /// When the paste completed, before History's write.
    pub pasted_at: Instant,
    /// The listener's Return count at the paste: a Return after it may be a
    /// send.
    pub returns: u64,
}

/// The element a paste is about to land in, and where its caret is, read
/// before the paste.
pub struct Pending {
    pid: i32,
    element: RetainedElement,
    caret: Option<Caret>,
}

impl Pending {
    pub fn new(pid: i32, element: RetainedElement, caret: Option<Caret>) -> Self {
        Self {
            pid,
            element,
            caret,
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
        Some(Pending::new(pid, element, caret))
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
struct Landed {
    inserted: String,
    anchor: Option<Anchor>,
    at: Instant,
    returns: u64,
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
    /// caret read before it says where that is: the paste's offset, and how
    /// long the field is with it in.
    pub fn pasted(&mut self, inserted: String) {
        let State::Before(pending) = std::mem::replace(&mut self.0, State::Off) else {
            return;
        };
        if inserted.is_empty() {
            return;
        }
        let len = inserted.encode_utf16().count();
        let anchor = pending.caret.map(|caret| Anchor {
            start: caret.location,
            len,
            total: (caret.characters + len).saturating_sub(caret.selected),
        });
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
    ) {
        if let (State::Pasted(pending, landed), Some(history), Some(entry_id)) =
            (self.0, history, entry_id)
        {
            handoff.submit(PasteTarget {
                pid: pending.pid,
                element: pending.element,
                history: history.clone(),
                entry_id,
                inserted: landed.inserted,
                anchor: landed.anchor,
                pasted_at: landed.at,
                returns: landed.returns,
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

/// A target over a retained element that names no field, for tests.
#[cfg(test)]
fn target_for_test(history: &History, entry_id: u64) -> PasteTarget {
    PasteTarget {
        pid: 1,
        element: RetainedElement::for_test(),
        history: history.clone(),
        entry_id,
        inserted: "pasted".into(),
        anchor: None,
        pasted_at: Instant::now(),
        returns: 0,
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
        caret: Option<Caret>,
        submitted: Vec<PasteTarget>,
    }

    impl Handoff for Fake {
        fn read(&mut self) -> Option<Pending> {
            Some(Pending::new(1, RetainedElement::for_test(), self.caret))
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
        let caret = Caret {
            location: 10,
            selected: 3,
            characters: 40,
        };
        let mut fake = Fake {
            caret: Some(caret),
            submitted: Vec::new(),
        };
        let mut watch = Watch::before(&mut fake, Some(&history));
        std::thread::sleep(std::time::Duration::from_millis(20));
        // A three-character selection is replaced by a six-character paste.
        watch.pasted(" hello".into());
        let State::Pasted(_, landed) = &watch.0 else {
            panic!("the paste is noted");
        };
        let pasted_at = landed.at;
        assert_eq!(
            landed.anchor,
            Some(Anchor {
                start: 10,
                len: 6,
                total: 43
            })
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut again = Watch::before(&mut fake, Some(&history));
        again.pasted(" hello".into());
        let State::Pasted(_, landed) = &again.0 else {
            panic!("the paste is noted");
        };
        assert!(landed.at > pasted_at, "the time is taken at each paste");
        watch.recorded(&mut fake, Some(&history), None);
        assert!(fake.submitted.is_empty(), "no entry, no target");
        again.recorded(&mut fake, Some(&history), Some(7));
        assert_eq!(fake.submitted.len(), 1);
        assert_eq!(fake.submitted[0].entry_id, 7);
        assert_eq!(fake.submitted[0].inserted, " hello");
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
        watch.recorded(&mut fake, Some(&history), Some(1));
        assert_eq!(fake.submitted[0].anchor, None);
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
