//! What Labs › Learn from corrections saw of an entry's pasted text: the
//! watcher's outcome and, for an edit, the aligned span the person typed.
//! It lives in the entry, so it counts toward History's text cap, expires with
//! the entry, and leaves with Delete and Clear all. The first observation
//! wins. Nothing here logs text.

use serde::{Deserialize, Serialize};

use super::HistoryEntry;
#[cfg(target_os = "macos")]
use super::{HistoryStore, MAX_TEXT_BYTES, now_ms, truncated};

/// One observation, as `history.json` and `hex history --json --raw` carry it.
/// The outcome is a string, not an enum, because older builds read these
/// files and `hex history` prints them; the watcher builds it in one place
/// (`CorrectionObservation::into_history`).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryCorrection {
    /// `unchanged`, `edited`, or `lost`.
    pub outcome: String,
    /// Why it was lost (`element_gone`, `secure`, `too_large`,
    /// `span_not_found`, `read_timeout`, `read_failed`,
    /// `cleared_before_read`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Milliseconds from the paste to the read this comes from.
    pub observed_at_ms: u64,
    /// The field's aligned span; only for `edited`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited_text: Option<String>,
}

impl HistoryCorrection {
    pub(super) fn text_bytes(&self) -> usize {
        self.edited_text.as_ref().map_or(0, String::len)
    }
}

impl HistoryEntry {
    pub(super) fn correction_bytes(&self) -> usize {
        self.correction
            .as_ref()
            .map_or(0, HistoryCorrection::text_bytes)
    }
}

#[cfg(target_os = "macos")]
impl HistoryStore {
    /// Keep `correction` on entry `id`, its text cut to the per-entry cap and
    /// the store pruned to its hard caps, in one save. Returns false when the
    /// entry is gone, already has one, or no longer holds `final_text` (a
    /// Replace took it: the correction was seen against other text). A failed
    /// save keeps the entry as it was.
    pub fn observe_correction(
        &mut self,
        id: u64,
        final_text: &str,
        mut correction: HistoryCorrection,
    ) -> std::io::Result<bool> {
        correction.edited_text = correction
            .edited_text
            .map(|text| truncated(&text, MAX_TEXT_BYTES));
        // Retention may have gone Off since the paste: keep nothing.
        if !self.records() {
            return Ok(false);
        }
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return Ok(false);
        };
        let entry = &self.entries[index];
        if entry.correction.is_some() || entry.final_text != truncated(final_text, MAX_TEXT_BYTES) {
            return Ok(false);
        }
        let before = self.entries.clone();
        self.entries[index].correction = Some(correction);
        // The added text may take the store past its byte cap: prune, then
        // save once.
        self.prune(now_ms());
        if let Err(error) = self.persist() {
            self.entries = before;
            return Err(error);
        }
        Ok(self
            .entries
            .iter()
            .any(|entry| entry.id == id && entry.correction.is_some()))
    }
}

#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::history::{
        HistoryDraft, HistoryKind, HistoryRetention, MAX_TOTAL_TEXT_BYTES, now_ms,
    };

    fn store(name: &str) -> (std::path::PathBuf, HistoryStore) {
        let directory = std::env::temp_dir().join(format!(
            "hex-history-corrections-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("history.json");
        let store = HistoryStore::open(path.clone(), HistoryRetention::Week, 0);
        (path, store)
    }

    fn draft(text: &str) -> HistoryDraft {
        HistoryDraft {
            kind: HistoryKind::Dictation,
            raw_text: text.into(),
            final_text: text.into(),
            application: None,
            processing: None,
            audio_ms: 0,
            inference_ms: 0,
            total_ms: 0,
            recording: None,
            model: None,
            model_id: None,
        }
    }

    /// Observes with the text the entry holds now, as a watch of it would.
    fn observe(
        store: &mut HistoryStore,
        id: u64,
        correction: HistoryCorrection,
    ) -> std::io::Result<bool> {
        let text = store
            .entry(id)
            .map(|entry| entry.final_text.clone())
            .unwrap_or_default();
        store.observe_correction(id, &text, correction)
    }

    fn edited(text: &str) -> HistoryCorrection {
        HistoryCorrection {
            outcome: "edited".into(),
            reason: None,
            observed_at_ms: 4_200,
            edited_text: Some(text.into()),
        }
    }

    #[test]
    fn observe_correction_persists_and_is_idempotent() {
        let (path, mut store) = store("persist");
        let id = store
            .record(draft("Send it to Kip."), now_ms())
            .unwrap()
            .unwrap();
        assert!(observe(&mut store, id, edited("Kit")).unwrap());
        assert!(!observe(&mut store, id, edited("Other")).unwrap());
        let reopened = HistoryStore::open(path, HistoryRetention::Week, 20);
        let correction = reopened.entry(id).unwrap().correction.clone().unwrap();
        assert_eq!(correction.outcome, "edited");
        assert_eq!(correction.edited_text.as_deref(), Some("Kit"));
        assert_eq!(correction.observed_at_ms, 4_200);
    }

    #[test]
    fn observe_correction_on_missing_entry_is_false() {
        let (_, mut store) = store("missing");
        assert!(!observe(&mut store, 99, edited("Kit")).unwrap());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn correction_counts_toward_text_cap() {
        let (_, mut store) = store("cap");
        let id = store.record(draft("a"), now_ms()).unwrap().unwrap();
        observe(&mut store, id, edited("kit")).unwrap();
        let entry = store.entry(id).unwrap();
        assert_eq!(entry.text_bytes(), 1 + 1 + 3);
        let long = "x".repeat(MAX_TEXT_BYTES * 2);
        let (_, mut other) = self::store("cap-long");
        let id = other.record(draft("a"), now_ms()).unwrap().unwrap();
        observe(&mut other, id, edited(&long)).unwrap();
        let kept = other.entry(id).unwrap().correction.clone().unwrap();
        assert!(kept.edited_text.unwrap().len() <= MAX_TEXT_BYTES);
    }

    #[test]
    fn clear_all_removes_correction() {
        let (path, mut store) = store("clear");
        let id = store.record(draft("hello"), now_ms()).unwrap().unwrap();
        observe(&mut store, id, edited("Hullo")).unwrap();
        store.clear().unwrap();
        assert!(store.entry(id).is_none());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("Hullo"));
    }

    #[test]
    fn loads_file_without_correction_field() {
        let (path, mut store) = store("compat");
        let id = store.record(draft("hello"), now_ms()).unwrap().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("correction"));
        let reopened = HistoryStore::open(path, HistoryRetention::Week, 20);
        assert_eq!(reopened.entry(id).unwrap().correction, None);
    }

    #[test]
    fn a_correction_prunes_the_oldest_entries_past_the_byte_cap() {
        let (_, mut store) = store("prune");
        let first = store.record(draft("a"), now_ms()).unwrap().unwrap();
        // Fill the store to just under its cap without a save per entry.
        let big = "x".repeat(MAX_TEXT_BYTES);
        let template = store.entry(first).unwrap().clone();
        let mut last = first;
        for n in 1..=511 {
            let mut entry = template.clone();
            entry.id = first + n;
            entry.raw_text = big.clone();
            entry.final_text = big.clone();
            store.entries.push(entry);
            last = first + n;
        }
        let total: usize = store.entries().map(HistoryEntry::text_bytes).sum();
        let room = MAX_TOTAL_TEXT_BYTES - 100;
        store.entries[0].raw_text = "x".repeat(room - (total - 1));
        let total: usize = store.entries().map(HistoryEntry::text_bytes).sum();
        assert_eq!(total, room);
        observe(&mut store, last, edited(&big)).unwrap();
        let total: usize = store.entries().map(HistoryEntry::text_bytes).sum();
        assert!(total <= MAX_TOTAL_TEXT_BYTES, "{total}");
        assert!(store.entry(last).is_some(), "the newest entry stays");
        assert!(store.entry(first).is_none(), "the oldest went to make room");
    }

    #[test]
    fn lost_keeps_a_reason_and_no_text() {
        let (_, mut store) = store("lost");
        let id = store.record(draft("hello"), now_ms()).unwrap().unwrap();
        let lost = HistoryCorrection {
            outcome: "lost".into(),
            reason: Some("secure".into()),
            observed_at_ms: 1,
            edited_text: None,
        };
        assert!(observe(&mut store, id, lost).unwrap());
        let kept = store.entry(id).unwrap().correction.clone().unwrap();
        assert_eq!(kept.reason.as_deref(), Some("secure"));
        assert_eq!(kept.edited_text, None);
    }

    #[test]
    fn replace_and_restore_drop_a_correction_seen_against_other_text() {
        use crate::history::Transcript;
        let (_, mut store) = store("replace");
        let id = store
            .record(draft("send it to kip"), now_ms())
            .unwrap()
            .unwrap();
        observe(&mut store, id, edited("send it to Kit")).unwrap();
        assert!(store.entry(id).unwrap().correction.is_some());
        let again = Transcript {
            text: "send it to Kip".into(),
            model: None,
            model_id: None,
            language: None,
        };
        assert!(store.replace_text(id, again, now_ms()).unwrap());
        assert_eq!(store.entry(id).unwrap().correction, None);
        // A new observation may come for the new text, and a restore drops it too.
        assert!(observe(&mut store, id, edited("x")).unwrap());
        assert!(store.restore_version(id, 0).unwrap());
        assert_eq!(store.entry(id).unwrap().correction, None);
    }

    #[test]
    fn nothing_is_kept_while_retention_is_off_and_a_failed_save_restores_everything() {
        let (path, mut store) = store("off");
        let id = store.record(draft("hello"), now_ms()).unwrap().unwrap();
        store
            .set_retention(HistoryRetention::Off, now_ms())
            .unwrap();
        assert!(!observe(&mut store, id, edited("Hullo")).unwrap());
        store
            .set_retention(HistoryRetention::Week, now_ms())
            .unwrap();
        assert_eq!(store.entry(id).unwrap().correction, None);
        // A save that cannot happen (the folder is a file) restores the entries.
        let blocked = path.with_file_name("blocked");
        std::fs::write(&blocked, "").unwrap();
        store.path = blocked.join("history.json");
        assert!(observe(&mut store, id, edited("Hullo")).is_err());
        assert_eq!(store.entry(id).unwrap().correction, None);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn a_correction_seen_against_text_the_entry_no_longer_holds_is_not_stored() {
        use crate::history::Transcript;
        let (_, mut store) = store("provenance");
        let id = store
            .record(draft("send it to Kip"), now_ms())
            .unwrap()
            .unwrap();
        let replaced = Transcript {
            text: "send it to Kit".into(),
            model: None,
            model_id: None,
            language: None,
        };
        // A Replace took the text mid-watch: the watch still holds the old one.
        assert!(store.replace_text(id, replaced, now_ms()).unwrap());
        let stale = store
            .observe_correction(id, "send it to Kip", edited("x"))
            .unwrap();
        assert!(!stale);
        assert_eq!(store.entry(id).unwrap().correction, None);
        assert!(
            store
                .observe_correction(id, "send it to Kit", edited("x"))
                .unwrap()
        );
    }
}
