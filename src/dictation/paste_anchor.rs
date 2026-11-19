//! Where the last dictation landed, for the labs that come back to it
//! (macOS): Labs › Learn from corrections watches the field, and Labs ›
//! Tidy after paste edits the pasted range in it.
//!
//! While either lab is on ([`wanted`]), the paste path reads the focused
//! field once before a dictation's paste ([`Field::read`]); M2's correction
//! watch takes its handle from that read, and once the paste, its History
//! entry, and its commit all happened the paste path records a
//! [`PasteAnchor`] here. The anchor holds the element (its own retain), the
//! inserted text's SHA-256 and UTF-16 length, the range it should occupy,
//! the app's pid, when it landed, and the History entry and commit count it
//! belongs to, never the text. There is one at a time, in memory only: the
//! next paste clears it, and after [`TTL`] it is never handed out (and is
//! released at the next access).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::accessibility::{self, RetainedElement};
use crate::dictation::tidy::apply::{Inserted, Why};

/// How long an anchor may be used after its paste.
pub(crate) const TTL: Duration = Duration::from_secs(10 * 60);

/// Whether a paste reads and keeps its field: either lab on (each gate
/// includes its managed lock).
pub(crate) fn wanted() -> bool {
    crate::labs::learn_from_corrections() || crate::labs::tidy_after_paste()
}

/// The focused field of the app a paste is about to go into.
pub(crate) struct Field {
    pub(crate) pid: i32,
    pub(crate) element: RetainedElement,
    /// Where the caret is (a paste starts there), if the app says.
    pub(crate) caret: Option<usize>,
}

impl Field {
    /// The frontmost app's focused, non-secure field, with its caret.
    /// `None` while neither lab is on: nothing is read.
    pub(crate) fn read() -> Option<Self> {
        if !wanted() {
            return None;
        }
        let pid = crate::dictation::context::frontmost_pid()?;
        let element = accessibility::focused_element(pid)?;
        let caret = accessibility::caret(&element);
        Some(Self {
            pid,
            element,
            caret,
        })
    }
}

/// A pasted dictation's field and what it inserted there.
pub(crate) struct PasteAnchor {
    pub(crate) pid: i32,
    pub(crate) element: RetainedElement,
    pub(crate) inserted: Inserted,
    /// Where the inserted text starts in the field (UTF-16), from the caret
    /// before the paste; it ends `inserted.utf16_len()` later.
    pub(crate) expected_start: Option<usize>,
    pub(crate) pasted_at: Instant,
    /// The History entry that kept the dictation.
    pub(crate) entry_id: u64,
    /// The commit count right after it ([`crate::dictation::tidy::last`]).
    pub(crate) generation: u64,
}

impl PasteAnchor {
    /// The anchor of a paste that inserted `inserted` into `field`. Takes
    /// another retain of the element; keeps only the text's hash and length.
    pub(crate) fn new(
        pid: i32,
        element: &RetainedElement,
        caret: Option<usize>,
        inserted: &str,
        pasted_at: Instant,
        entry_id: u64,
        generation: u64,
    ) -> Self {
        Self {
            pid,
            element: element.share(),
            inserted: Inserted::of(inserted),
            expected_start: caret,
            pasted_at,
            entry_id,
            generation,
        }
    }

    fn expired(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.pasted_at) >= TTL
    }
}

/// The one anchor slot.
pub(crate) struct Slot(Mutex<Option<PasteAnchor>>);

static SLOT: Slot = Slot::new();

impl Slot {
    pub(crate) const fn new() -> Self {
        Self(Mutex::new(None))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<PasteAnchor>> {
        self.0.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// Keeps `anchor`, replacing (and releasing) any older one.
    pub(crate) fn record(&self, anchor: PasteAnchor) {
        *self.lock() = Some(anchor);
    }

    /// Releases the anchor: a new paste is starting.
    pub(crate) fn clear(&self) {
        self.lock().take();
    }

    /// The anchor of History entry `entry_id` committed as `generation`,
    /// taken out of the slot. An expired anchor is released; one of another
    /// dictation stays for its own review.
    pub(crate) fn take_for(
        &self,
        entry_id: u64,
        generation: u64,
        now: Instant,
    ) -> Result<PasteAnchor, Why> {
        let mut slot = self.lock();
        let Some(anchor) = slot.as_ref() else {
            return Err(Why::NoAnchor);
        };
        if anchor.expired(now) {
            slot.take();
            return Err(Why::Expired);
        }
        if anchor.entry_id != entry_id || anchor.generation != generation {
            return Err(Why::OtherDictation);
        }
        slot.take().ok_or(Why::NoAnchor)
    }

    /// Puts back an anchor an apply didn't use up, only into an empty slot:
    /// a paste since then recorded a newer one or will.
    pub(crate) fn restore(&self, anchor: PasteAnchor) {
        let mut slot = self.lock();
        if slot.is_none() {
            *slot = Some(anchor);
        }
    }

    #[cfg(test)]
    fn holds(&self) -> Option<(u64, u64)> {
        self.lock()
            .as_ref()
            .map(|anchor| (anchor.entry_id, anchor.generation))
    }
}

/// Held while text goes into a field: a dictation's paste, or Tidy's edit
/// of the pasted range, so the two never interleave (their keystrokes and
/// clipboard writes included).
static DELIVERY: Mutex<()> = Mutex::new(());

/// How many dictation pastes have started. An apply notes it before taking
/// its anchor; a change since means a newer paste may have moved the text
/// or the selection, and the anchor's range can't be trusted.
static PASTES: AtomicU64 = AtomicU64::new(0);

/// Keeps the anchor of the dictation that just committed.
pub(crate) fn record(anchor: PasteAnchor) {
    SLOT.record(anchor);
}

/// A dictation's paste is starting: waits for any Tidy edit in progress,
/// counts the paste, and releases the anchor. Hold the guard through the
/// paste.
pub(crate) fn begin_paste() -> MutexGuard<'static, ()> {
    let delivery = deliver();
    PASTES.fetch_add(1, Ordering::SeqCst);
    SLOT.clear();
    delivery
}

/// Takes the delivery lock for a Tidy edit.
pub(crate) fn deliver() -> MutexGuard<'static, ()> {
    DELIVERY.lock().unwrap_or_else(|error| error.into_inner())
}

/// The count of dictation pastes started so far.
pub(crate) fn pastes() -> u64 {
    PASTES.load(Ordering::SeqCst)
}

/// See [`Slot::take_for`].
pub(crate) fn take_for(entry_id: u64, generation: u64, now: Instant) -> Result<PasteAnchor, Why> {
    SLOT.take_for(entry_id, generation, now)
}

/// See [`Slot::restore`]; nothing is put back once a newer paste started
/// (`pastes` is the count noted before the anchor was taken).
/// Compared under the delivery lock [`begin_paste`] takes, so a paste can't
/// start between the check and the restore. Never call it while holding a
/// [`deliver`] guard.
pub(crate) fn restore(anchor: PasteAnchor, pastes: u64) {
    let _delivery = deliver();
    if self::pastes() == pastes {
        SLOT.restore(anchor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(entry_id: u64, generation: u64, pasted_at: Instant) -> PasteAnchor {
        PasteAnchor::new(
            1,
            &RetainedElement::for_test(),
            Some(4),
            " hello there",
            pasted_at,
            entry_id,
            generation,
        )
    }

    /// The anchor keeps a hash and a length, never the text.
    #[test]
    fn the_anchor_keeps_the_hash_length_and_range_only() {
        let anchor = anchor(7, 3, Instant::now());
        assert_eq!(anchor.inserted, Inserted::of(" hello there"));
        assert_eq!(anchor.inserted.utf16_len(), 12);
        assert_eq!(anchor.expected_start, Some(4));
        let debug = format!("{:?}", anchor.inserted);
        assert!(!debug.contains("hello"), "{debug}");
    }

    #[test]
    fn the_next_paste_replaces_or_clears_the_anchor() {
        let slot = Slot::new();
        let now = Instant::now();
        slot.record(anchor(1, 1, now));
        slot.record(anchor(2, 2, now));
        assert_eq!(slot.holds(), Some((2, 2)));
        assert_eq!(
            slot.take_for(1, 1, now).err(),
            Some(Why::OtherDictation),
            "the older dictation's review gets nothing"
        );
        slot.clear();
        assert_eq!(slot.take_for(2, 2, now).err(), Some(Why::NoAnchor));
    }

    #[test]
    fn an_anchor_expires_after_ten_minutes() {
        let slot = Slot::new();
        let pasted = Instant::now();
        slot.record(anchor(1, 1, pasted));
        assert!(
            slot.take_for(1, 1, pasted + TTL - Duration::from_secs(1))
                .is_ok()
        );
        slot.record(anchor(1, 1, pasted));
        assert_eq!(slot.take_for(1, 1, pasted + TTL).err(), Some(Why::Expired));
        assert_eq!(slot.holds(), None, "an expired anchor is released");
    }

    /// The review's dictation must be the anchor's: the same History entry
    /// and the same commit, so a tidy never edits another paste.
    #[test]
    fn an_anchor_for_a_different_dictation_is_refused() {
        let slot = Slot::new();
        let now = Instant::now();
        slot.record(anchor(5, 9, now));
        assert_eq!(slot.take_for(4, 9, now).err(), Some(Why::OtherDictation));
        assert_eq!(slot.take_for(5, 8, now).err(), Some(Why::OtherDictation));
        assert_eq!(slot.holds(), Some((5, 9)), "it stays for its own review");
        let taken = slot.take_for(5, 9, now).unwrap();
        assert_eq!(slot.holds(), None);
        // Put back when unused, but never over a newer paste's anchor.
        slot.restore(taken);
        assert_eq!(slot.holds(), Some((5, 9)));
        let taken = slot.take_for(5, 9, now).unwrap();
        slot.record(anchor(6, 10, now));
        slot.restore(taken);
        assert_eq!(slot.holds(), Some((6, 10)));
    }

    /// With both labs off a paste reads no field, so no anchor can exist.
    #[test]
    fn nothing_is_read_or_anchored_with_both_labs_off() {
        crate::labs::apply(&crate::app_settings::LabsSettings::default());
        assert!(!wanted());
        assert!(Field::read().is_none());
        for (learn, tidy) in [(true, false), (false, true)] {
            crate::labs::apply(&crate::app_settings::LabsSettings {
                learn_from_corrections: learn,
                tidy_after_paste: tidy,
                ..Default::default()
            });
            assert!(wanted(), "learn {learn} tidy {tidy}");
        }
        crate::labs::apply(&crate::app_settings::LabsSettings::default());
    }
}
