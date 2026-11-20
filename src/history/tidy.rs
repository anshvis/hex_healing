//! What Labs › Tidy after paste decided about an entry's suggestions (T8.5):
//! each suggestion's kind, the words it would change (a token, never the
//! text around it), what it would put there, where in `final_text` it sits,
//! whether the person accepted it, and whether Hex edited it into the field.
//! `healing collect` turns these into its `tidy_events` and the tidied text
//! the person's later corrections are measured from (D-38): the entry's
//! `raw_text` and `final_text` never change.
//!
//! Bounded like the rest of an entry's metadata: at most
//! [`MAX_TIDY_DECISIONS`], tokens and replacements cut to
//! [`MAX_TIDY_TOKEN_BYTES`]. It counts toward History's text cap, expires
//! with the entry, and leaves with Delete and Clear all. Nothing here logs
//! text.

use serde::{Deserialize, Serialize};

use super::HistoryEntry;
#[cfg(target_os = "macos")]
use super::{HistoryStore, MAX_TEXT_BYTES, now_ms, truncated};

/// Decisions kept per entry; a review with more keeps the first ones.
pub const MAX_TIDY_DECISIONS: usize = 64;
/// The longest token or replacement kept, in bytes.
pub const MAX_TIDY_TOKEN_BYTES: usize = 32;

/// One suggestion's decision, as `history.json` and `hex history --json
/// --raw` carry it. `kind` and `source` are strings, as `healing` reads them
/// (D-66): `filler`, `repeat`, `self_correction`, `misheard`; `rule`,
/// `model`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HistoryTidy {
    pub kind: String,
    /// The words the suggestion removes or replaces, at most
    /// [`MAX_TIDY_TOKEN_BYTES`].
    pub token: String,
    /// What it puts in their place: empty for a removal. `None` when it was
    /// longer than [`MAX_TIDY_TOKEN_BYTES`], so it isn't kept and the tidied
    /// text can't be rebuilt from these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
    /// Its place in `final_text`, in Unicode scalars.
    pub start: usize,
    pub end: usize,
    pub accepted: bool,
    /// Hex edited it into the field the dictation was pasted into: the
    /// person's later edits are measured from the text with these applied.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub applied: bool,
    pub source: String,
}

impl HistoryTidy {
    /// A decision with its token and replacement bounded.
    pub fn bounded(
        kind: &str,
        token: &str,
        replacement: &str,
        range: std::ops::Range<usize>,
        accepted: bool,
        applied: bool,
        source: &str,
    ) -> Self {
        let cut = |text: &str| text[..text.floor_char_boundary(MAX_TIDY_TOKEN_BYTES)].to_string();
        Self {
            kind: kind.to_string(),
            token: cut(token),
            replacement: (replacement.len() <= MAX_TIDY_TOKEN_BYTES)
                .then(|| replacement.to_string()),
            start: range.start,
            end: range.end,
            accepted,
            applied: applied && accepted,
            source: source.to_string(),
        }
    }

    fn text_bytes(&self) -> usize {
        self.token.len() + self.replacement.as_ref().map_or(0, String::len)
    }
}

impl HistoryEntry {
    pub(super) fn tidy_bytes(&self) -> usize {
        self.tidy.iter().map(HistoryTidy::text_bytes).sum()
    }
}

#[cfg(target_os = "macos")]
impl HistoryStore {
    /// Keep a review's `decisions` on entry `id`, bounded, in one save.
    /// Returns false when retention is Off, the entry is gone or no longer
    /// holds `final_text` (a Replace took it), or it already keeps decisions
    /// that went into the field (the first applied review wins; one that
    /// only copied or failed is replaced by a later one). A failed save
    /// keeps the entry as it was.
    pub fn record_tidy(
        &mut self,
        id: u64,
        final_text: &str,
        mut decisions: Vec<HistoryTidy>,
    ) -> std::io::Result<bool> {
        if !self.records() || decisions.is_empty() {
            return Ok(false);
        }
        decisions.truncate(MAX_TIDY_DECISIONS);
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return Ok(false);
        };
        let entry = &self.entries[index];
        if entry.final_text != truncated(final_text, MAX_TEXT_BYTES)
            || entry.tidy.iter().any(|decision| decision.applied)
        {
            return Ok(false);
        }
        let before = self.entries.clone();
        self.entries[index].tidy = decisions;
        self.prune(now_ms());
        if let Err(error) = self.persist() {
            self.entries = before;
            return Err(error);
        }
        Ok(self
            .entries
            .iter()
            .any(|entry| entry.id == id && !entry.tidy.is_empty()))
    }
}

#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::history::{HistoryDraft, HistoryKind, HistoryRetention, now_ms};

    fn store(name: &str) -> (std::path::PathBuf, HistoryStore) {
        let directory =
            std::env::temp_dir().join(format!("hex-history-tidy-{}-{name}", std::process::id()));
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

    const TEXT: &str = "um send it to Kip";

    fn decisions(applied: bool) -> Vec<HistoryTidy> {
        vec![
            HistoryTidy::bounded("filler", "um ", "", 0..3, true, applied, "rule"),
            HistoryTidy::bounded("misheard", "Kip", "Kit", 14..17, true, applied, "model"),
        ]
    }

    /// AC-16 (data): the decisions persist beside the unchanged texts, and
    /// tokens, replacements, and the count are bounded.
    #[test]
    fn tidy_metadata_persisted_and_bounded() {
        let (path, mut persisted) = store("persisted");
        let id = persisted.record(draft(TEXT), now_ms()).unwrap().unwrap();
        assert!(persisted.record_tidy(id, TEXT, decisions(true)).unwrap());
        let reopened = HistoryStore::open(path, HistoryRetention::Week, now_ms());
        let entry = reopened.entry(id).unwrap();
        assert_eq!(entry.tidy, decisions(true));
        assert_eq!(
            (entry.raw_text.as_str(), entry.final_text.as_str()),
            (TEXT, TEXT)
        );

        let long = "x".repeat(100);
        let bounded =
            HistoryTidy::bounded("self_correction", &long, &long, 0..100, true, true, "rule");
        assert_eq!(bounded.token.len(), MAX_TIDY_TOKEN_BYTES);
        assert_eq!(
            bounded.replacement, None,
            "a cut replacement could not rebuild the text"
        );
        let skipped = HistoryTidy::bounded("filler", "um", "", 0..2, false, true, "rule");
        assert!(!skipped.applied, "a skipped suggestion never went in");

        let (_, mut capped) = store("capped");
        let id = capped.record(draft(TEXT), now_ms()).unwrap().unwrap();
        let many = (0..MAX_TIDY_DECISIONS + 10)
            .map(|_| HistoryTidy::bounded("filler", "um", "", 0..2, false, false, "rule"))
            .collect();
        assert!(capped.record_tidy(id, TEXT, many).unwrap());
        assert_eq!(capped.entry(id).unwrap().tidy.len(), MAX_TIDY_DECISIONS);
    }

    /// The first review that went into the field wins; one that only copied
    /// is replaced; a Replace'd entry or Off retention keeps nothing.
    #[test]
    fn the_first_applied_review_wins() {
        let (_, mut wins) = store("wins");
        let id = wins.record(draft(TEXT), now_ms()).unwrap().unwrap();
        assert!(wins.record_tidy(id, TEXT, decisions(false)).unwrap());
        assert!(wins.record_tidy(id, TEXT, decisions(true)).unwrap());
        assert!(!wins.record_tidy(id, TEXT, decisions(false)).unwrap());
        assert_eq!(wins.entry(id).unwrap().tidy, decisions(true));
        assert!(!wins.record_tidy(id, "other text", decisions(true)).unwrap());
        assert!(!wins.record_tidy(id + 99, TEXT, decisions(true)).unwrap());
        let (_, mut off) = store("off");
        let id = off.record(draft(TEXT), now_ms()).unwrap().unwrap();
        off.set_retention(HistoryRetention::Off, now_ms()).unwrap();
        assert!(!off.record_tidy(id, TEXT, decisions(false)).unwrap());
    }

    /// Older entries have no `tidy`, and an entry without one writes none.
    #[test]
    fn an_entry_without_decisions_writes_no_field() {
        let entry: HistoryEntry = serde_json::from_value(serde_json::json!({
            "id": 1, "timestamp_ms": 1, "kind": "dictation", "raw_text": "a", "final_text": "a",
        }))
        .unwrap();
        assert!(entry.tidy.is_empty());
        assert!(!serde_json::to_string(&entry).unwrap().contains("tidy"));
    }
}
