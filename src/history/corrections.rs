//! What Labs › Learn from corrections saw of an entry's pasted text: the
//! watcher's outcome and, for an edit, the aligned span the person typed.
//! It lives in the entry, so it counts toward History's text cap, expires with
//! the entry, and leaves with Delete and Clear all. The first observation
//! wins. Nothing here logs text.

use serde::{Deserialize, Serialize};

use super::{History, HistoryEntry, HistoryStore, MAX_TEXT_BYTES, truncated};
use crate::dictation::correction_watch::{self, CorrectionObservation, Outcome};

static SHARED: std::sync::OnceLock<History> = std::sync::OnceLock::new();

/// Sends every finished observation to `history`, once, at startup (the
/// first call wins, as the watcher's sink does).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn install_sink(history: &History) {
    let _ = SHARED.set(history.clone());
    correction_watch::set_sink(deliver);
}

fn deliver(entry_id: u64, observation: CorrectionObservation) {
    if let Some(history) = SHARED.get() {
        deliver_to(history, entry_id, observation);
    }
}

fn deliver_to(history: &History, entry_id: u64, observation: CorrectionObservation) {
    let outcome = observation.outcome.as_str();
    // Ids and outcome only: never the text.
    match history.observe_correction(entry_id, observation) {
        Ok(kept) => tracing::debug!(entry_id, outcome, kept, "correction observation"),
        Err(error) => tracing::warn!(%error, entry_id, outcome, "could not keep a correction"),
    }
}

/// One observation, as `history.json` and `hex history --json --raw` carry it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryCorrection {
    /// `unchanged`, `edited`, or `lost`.
    pub outcome: String,
    /// Why it was lost (`element_gone`, `secure`, `too_large`,
    /// `span_not_found`, `read_timeout`, `cleared_before_read`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Milliseconds from the paste to the read this comes from.
    pub observed_at_ms: u64,
    /// The field's aligned span; only for `edited`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited_text: Option<String>,
}

impl HistoryCorrection {
    fn from_observation(observation: CorrectionObservation) -> Self {
        let (outcome, reason) = match observation.outcome {
            Outcome::Unchanged => ("unchanged", None),
            Outcome::Edited => ("edited", None),
            Outcome::Lost(lost) => ("lost", Some(lost.as_str().to_string())),
        };
        Self {
            outcome: outcome.to_string(),
            reason,
            observed_at_ms: observation.observed_at_ms,
            edited_text: observation
                .edited_text
                .filter(|_| matches!(observation.outcome, Outcome::Edited))
                .map(|text| truncated(&text, MAX_TEXT_BYTES)),
        }
    }

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

impl HistoryStore {
    /// Keep `observation` on entry `id`. Returns false when the entry is gone
    /// or already has one. A failed save keeps the entry as it was.
    pub fn observe_correction(
        &mut self,
        id: u64,
        observation: CorrectionObservation,
    ) -> std::io::Result<bool> {
        let correction = HistoryCorrection::from_observation(observation);
        self.change(id, |entry| {
            if entry.correction.is_some() {
                return false;
            }
            entry.correction = Some(correction);
            true
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::dictation::correction_watch::{Lost, Trigger};
    use crate::history::{HistoryDraft, HistoryKind, HistoryRetention, now_ms};

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

    fn edited(text: &str) -> CorrectionObservation {
        CorrectionObservation {
            outcome: Outcome::Edited,
            trigger: Trigger::FocusOut,
            observed_at_ms: 4_200,
            edited_text: Some(text.into()),
        }
    }

    #[test]
    fn observe_correction_persists_and_is_idempotent() {
        let (path, mut store) = store("persist");
        let id = store.record(draft("Send it to Kip."), 10).unwrap().unwrap();
        assert!(store.observe_correction(id, edited("Kit")).unwrap());
        assert!(!store.observe_correction(id, edited("Other")).unwrap());
        let reopened = HistoryStore::open(path, HistoryRetention::Week, 20);
        let correction = reopened.entry(id).unwrap().correction.clone().unwrap();
        assert_eq!(correction.outcome, "edited");
        assert_eq!(correction.edited_text.as_deref(), Some("Kit"));
        assert_eq!(correction.observed_at_ms, 4_200);
    }

    #[test]
    fn observe_correction_on_missing_entry_is_false() {
        let (_, mut store) = store("missing");
        assert!(!store.observe_correction(99, edited("Kit")).unwrap());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn the_sink_writes_through_the_shared_history() {
        let (path, store) = store("sink");
        let history = History::new(store);
        let id = history.record(draft("Send it to Kip.")).unwrap().unwrap();
        deliver_to(&history, id, edited("Kit"));
        let reopened = HistoryStore::open(path, HistoryRetention::Week, now_ms());
        assert_eq!(
            reopened
                .entry(id)
                .unwrap()
                .correction
                .as_ref()
                .unwrap()
                .edited_text
                .as_deref(),
            Some("Kit")
        );
    }

    #[test]
    fn lost_keeps_its_reason_and_no_text() {
        let (_, mut store) = store("lost");
        let id = store.record(draft("hello"), 10).unwrap().unwrap();
        let lost = CorrectionObservation {
            outcome: Outcome::Lost(Lost::Secure),
            trigger: Trigger::Timeout,
            observed_at_ms: 1,
            edited_text: Some("secret".into()),
        };
        assert!(store.observe_correction(id, lost).unwrap());
        let correction = store.entry(id).unwrap().correction.clone().unwrap();
        assert_eq!(correction.outcome, "lost");
        assert_eq!(correction.reason.as_deref(), Some("secure"));
        assert_eq!(correction.edited_text, None);
    }

    #[test]
    fn correction_counts_toward_text_cap() {
        let (_, mut store) = store("cap");
        let id = store.record(draft("a"), 10).unwrap().unwrap();
        store.observe_correction(id, edited("kit")).unwrap();
        let entry = store.entry(id).unwrap();
        assert_eq!(entry.text_bytes(), 1 + 1 + 3);
        let long = "x".repeat(MAX_TEXT_BYTES * 2);
        let (_, mut other) = self::store("cap-long");
        let id = other.record(draft("a"), 10).unwrap().unwrap();
        other.observe_correction(id, edited(&long)).unwrap();
        let kept = other.entry(id).unwrap().correction.clone().unwrap();
        assert!(kept.edited_text.unwrap().len() <= MAX_TEXT_BYTES);
    }

    #[test]
    fn clear_all_removes_correction() {
        let (path, mut store) = store("clear");
        let id = store.record(draft("hello"), 10).unwrap().unwrap();
        store.observe_correction(id, edited("Hullo")).unwrap();
        store.clear().unwrap();
        assert!(store.entry(id).is_none());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("Hullo"));
    }

    #[test]
    fn loads_file_without_correction_field() {
        let (path, mut store) = store("compat");
        let id = store.record(draft("hello"), 10).unwrap().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("correction"));
        let reopened = HistoryStore::open(path, HistoryRetention::Week, 20);
        assert_eq!(reopened.entry(id).unwrap().correction, None);
    }
}
