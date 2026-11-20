//! What a Tidy trigger does before anything shows (T8.3): checks the lab,
//! finds the last dictation in History, and computes its suggestions; and
//! what Return does once the person has chosen ([`apply_accepted`]).
//!
//! Suggestions are computed when the trigger fires, never at paste time
//! (D-37), and [`prepare`] blocks on T8.1's model pass for up to 2 s, so
//! callers run it off the main thread. Logs keep counts and kinds only.

use super::apply::{Method, Outcome, Why};
use super::session::{Dictation, TidySession};
use super::{Habits, Suggestion};
use crate::history::HistoryEntry;

/// Why a trigger opened nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Refusal {
    /// Labs › Tidy after paste is off (or this OS doesn't have it).
    LabOff,
    /// History holds no dictation to tidy.
    NoDictation,
}

impl Refusal {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::LabOff => {
                "Labs › Tidy after paste is off; turn it on with `hex settings set labs.tidy_after_paste on`"
            }
            Self::NoDictation => {
                "there's no dictation to tidy yet; History keeps the last one while it records"
            }
        }
    }
}

/// The dictation `entry` holds, if it has text to tidy. An entry without a
/// language of its own (the saved setting chose) takes `language`.
pub(crate) fn dictation(entry: &HistoryEntry, language: &str) -> Option<Dictation> {
    if entry.final_text.trim().is_empty() {
        return None;
    }
    Some(Dictation {
        id: entry.id,
        pasted: entry.final_text.clone(),
        raw: entry.raw_text.clone(),
        language: entry.language.clone().unwrap_or_else(|| language.into()),
        delivery: super::last::delivery(entry.id, entry.kind),
    })
}

/// The review of `newest`, History's newest entry, with `suggest` (T8.1's
/// [`super::suggest`] outside tests). Gated first: with the lab off nothing
/// is read or computed.
pub(crate) fn prepare(
    lab_on: bool,
    newest: impl FnOnce() -> Option<HistoryEntry>,
    language: &str,
    habits: impl FnOnce() -> Habits,
    suggest: impl FnOnce(&str, &str, &str, &Habits) -> Vec<Suggestion>,
) -> Result<TidySession, Refusal> {
    if !lab_on {
        return Err(Refusal::LabOff);
    }
    let generation = super::last::generation();
    let dictation = newest()
        .and_then(|entry| dictation(&entry, language))
        .ok_or(Refusal::NoDictation)?;
    let suggestions = suggest(
        &dictation.pasted,
        &dictation.raw,
        &dictation.language,
        &habits(),
    );
    let session = TidySession::new(dictation, suggestions, generation);
    tracing::info!(
        suggestions = session.suggestions.len(),
        copy_only = session.delivery == super::session::Delivery::Copied,
        "tidy review ready"
    );
    Ok(session)
}

/// What Return did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Applied {
    /// The pasted text in its field now reads the tidied text (T8.4).
    Edited(Method),
    /// The pasted text changed since the paste; nothing was edited.
    Moved,
    /// The field couldn't be edited, for this reason; nothing was (or, for
    /// [`Why::Unverified`], nothing could be confirmed).
    Unsupported(Why),
    /// The tidied text is on the clipboard.
    Copied,
    /// Nothing was accepted, so nothing changed.
    Nothing,
    Failed(String),
}

/// Return in the review of a pasted dictation: applies the accepted
/// suggestions to the pasted range of its field through Accessibility
/// ([`super::apply::apply_to_last`]). Blocks for the frontmost-app wait and
/// the re-reads (under a second), so callers run it off the main thread.
/// Every decision goes on the dictation's History entry (T8.5), marked
/// applied on [`Applied::Edited`]; the apply itself resets Learn from
/// corrections' baseline.
pub(crate) fn apply_accepted(session: &TidySession) -> Applied {
    log_decisions(session);
    if session.accepted_count() == 0 {
        record(session, false);
        return Applied::Nothing;
    }
    let applied = match super::apply::apply_to_last(session) {
        Outcome::Applied(method) => Applied::Edited(method),
        Outcome::Moved => Applied::Moved,
        Outcome::Unsupported(why) => Applied::Unsupported(why),
    };
    record(session, matches!(applied, Applied::Edited(_)));
    applied
}

/// The review's decisions as History keeps them (T8.5): every suggestion,
/// accepted or skipped, as its kind, its token (the words it changes in
/// the dictation's final text, never more), what it puts there, and its
/// place; `applied` on the accepted ones when the edit went into the field.
pub(crate) fn decisions(session: &TidySession, applied: bool) -> Vec<crate::history::HistoryTidy> {
    let chars: Vec<char> = session.text.chars().collect();
    session
        .suggestions
        .iter()
        .zip(&session.accepted)
        .map(|(suggestion, accepted)| {
            let range = suggestion.range.scalar.clone();
            let token: String = chars[range.start.min(chars.len())..range.end.min(chars.len())]
                .iter()
                .collect();
            crate::history::HistoryTidy::bounded(
                suggestion.kind.as_str(),
                &token,
                &suggestion.replacement,
                range,
                *accepted,
                applied,
                suggestion.source.as_str(),
            )
        })
        .collect()
}

/// Keeps the review's decisions on its History entry. Counts only in the
/// log.
#[cfg(target_os = "macos")]
fn record(session: &TidySession, applied: bool) {
    let Some(history) = crate::macos_rune::runtime().and_then(|runtime| runtime.history()) else {
        return;
    };
    let decisions = decisions(session, applied);
    let count = decisions.len();
    match history.record_tidy(session.dictation_id, &session.text, decisions) {
        Ok(kept) => tracing::info!(count, applied, kept, "tidy decisions recorded"),
        Err(error) => tracing::warn!(%error, "could not record tidy decisions"),
    }
}

/// Linux and Windows have no tidy review (D-70).
#[cfg(not(target_os = "macos"))]
fn record(_session: &TidySession, _applied: bool) {}

/// Copy tidied text: for a dictation that only reached the clipboard, or
/// one whose field couldn't be edited.
pub(crate) fn copy_tidied(session: &TidySession) -> Applied {
    copy_tidied_with(session, copy_to_clipboard)
}

fn copy_tidied_with(
    session: &TidySession,
    copy: impl FnOnce(&str) -> Result<(), String>,
) -> Applied {
    let accepted = session.accepted_count();
    log_decisions(session);
    record(session, false);
    if accepted == 0 {
        return Applied::Nothing;
    }
    match copy(&session.tidied()) {
        Ok(()) => Applied::Copied,
        Err(error) => {
            tracing::warn!(%error, "could not copy the tidied text");
            Applied::Failed(error)
        }
    }
}

/// Counts and kinds of what the person accepted and skipped; never text.
fn log_decisions(session: &TidySession) {
    let count = |kind: super::Kind, accepted: bool| {
        session
            .suggestions
            .iter()
            .zip(&session.accepted)
            .filter(|(suggestion, decided)| suggestion.kind == kind && **decided == accepted)
            .count()
    };
    use super::Kind::{Filler, Misheard, Repeat, SelfCorrection};
    tracing::info!(
        fillers = count(Filler, true),
        repeats = count(Repeat, true),
        self_corrections = count(SelfCorrection, true),
        misheard = count(Misheard, true),
        skipped = session.suggestions.len() - session.accepted_count(),
        "tidy review applied"
    );
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn copy_to_clipboard(text: &str) -> Result<(), String> {
    arboard::Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text.to_owned()))
        .map_err(|error| error.to_string())
}

/// Linux has no tidy review (D-70) and Hex's Linux clipboard goes through
/// `linux/clipboard`, not arboard, so there is nothing to copy with here.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn copy_to_clipboard(_text: &str) -> Result<(), String> {
    Err("tidy has no clipboard on this OS".to_owned())
}

#[cfg(test)]
mod tests {
    use super::super::session::{Delivery, preview};
    use super::*;
    use crate::history::HistoryKind;

    fn entry(kind: HistoryKind, text: &str) -> HistoryEntry {
        serde_json::from_value(serde_json::json!({
            "id": 9_000_001,
            "timestamp_ms": 1,
            "kind": kind,
            "raw_text": text,
            "final_text": text,
        }))
        .unwrap()
    }

    fn no_habits() -> Habits {
        Habits::default()
    }

    /// T8.5: every suggestion is a decision with its token alone, its
    /// replacement and place in the final text; only accepted ones that
    /// went in are applied.
    #[test]
    fn decisions_record_tokens_not_text() {
        let mut session = preview();
        let skipped = decisions(&session, false);
        assert_eq!(skipped.len(), session.suggestions.len());
        assert!(skipped.iter().all(|decision| !decision.applied));
        let misheard = skipped
            .iter()
            .find(|decision| decision.kind == "misheard")
            .unwrap();
        assert_eq!(
            (misheard.token.as_str(), misheard.replacement.as_deref()),
            ("send", Some("sent"))
        );
        assert!(!misheard.accepted, "a misheard fix starts skipped");
        assert_eq!(misheard.source, "model");
        session.accept_all();
        let applied = decisions(&session, true);
        assert!(
            applied
                .iter()
                .all(|decision| decision.accepted && decision.applied)
        );
        for decision in &applied {
            let token: String = session
                .text
                .chars()
                .skip(decision.start)
                .take(decision.end - decision.start)
                .collect();
            assert_eq!(decision.token, token);
            assert_ne!(decision.token, session.text, "never the whole text");
        }
        // Rebuilding the tidied text from the decisions alone, as
        // `healing collect` does, gives what the review applied.
        let mut rebuilt = String::new();
        let mut at = 0;
        let chars: Vec<char> = session.text.chars().collect();
        for decision in &applied {
            rebuilt.extend(&chars[at..decision.start]);
            rebuilt.push_str(decision.replacement.as_deref().unwrap());
            at = decision.end;
        }
        rebuilt.extend(&chars[at..]);
        assert_eq!(rebuilt, session.tidied());
    }

    #[test]
    fn nothing_is_read_or_computed_with_the_lab_off() {
        let result = prepare(
            false,
            || panic!("History read with the lab off"),
            "en",
            || panic!("habits read with the lab off"),
            |_, _, _, _| panic!("suggestions computed with the lab off"),
        );
        assert_eq!(result.unwrap_err(), Refusal::LabOff);
    }

    #[test]
    fn no_dictation_is_said_so() {
        let result = prepare(true, || None, "en", no_habits, |_, _, _, _| Vec::new());
        assert_eq!(result.unwrap_err(), Refusal::NoDictation);
        let blank = entry(HistoryKind::Dictation, "  ");
        let result = prepare(
            true,
            || Some(blank),
            "en",
            no_habits,
            |_, _, _, _| Vec::new(),
        );
        assert_eq!(result.unwrap_err(), Refusal::NoDictation);
    }

    /// The review is of the newest entry's pasted text, in its language
    /// (or the saved one when it has none).
    #[test]
    fn the_newest_dictation_is_reviewed_with_its_text_and_language() {
        let text = "um so we ship it";
        let session = prepare(
            true,
            || Some(entry(HistoryKind::Dictation, text)),
            "en",
            no_habits,
            |pasted, raw, language, _| {
                assert_eq!((pasted, raw, language), (text, text, "en"));
                super::super::suggest_with(pasted, raw, language, &Habits::default(), |_| None)
            },
        )
        .unwrap();
        assert_eq!(session.dictation_id, 9_000_001);
        assert_eq!(session.text, text);
        assert_eq!(session.delivery, Delivery::Pasted);
        assert!(!session.is_empty());
    }

    /// A printed dictation reached no field: the review offers a copy.
    #[test]
    fn a_printed_dictation_is_copy_only() {
        let session = prepare(
            true,
            || Some(entry(HistoryKind::Printed, "uh hello")),
            "en",
            no_habits,
            |_, _, _, _| Vec::new(),
        )
        .unwrap();
        assert_eq!(session.delivery, Delivery::Copied);
        assert!(session.is_empty());
    }

    #[test]
    fn copy_tidied_copies_the_accepted_edits_and_nothing_without_any() {
        let mut session = preview();
        let mut copied = None;
        let applied = copy_tidied_with(&session, |text| {
            copied = Some(text.to_owned());
            Ok(())
        });
        assert_eq!(applied, Applied::Copied);
        assert_eq!(
            copied.as_deref(),
            Some("So I think we should send it Wednesday")
        );
        (0..session.suggestions.len())
            .filter(|index| session.accepted[*index])
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|index| session.toggle(index));
        let applied = copy_tidied_with(&session, |_| panic!("copied with nothing accepted"));
        assert_eq!(applied, Applied::Nothing);
        assert_eq!(apply_accepted(&session), Applied::Nothing);
    }

    /// With the lab off (as in tests) Return edits nothing and says why,
    /// before any anchor is taken; the anchor rules are
    /// `paste_anchor::tests`.
    #[test]
    fn apply_with_the_lab_off_edits_nothing() {
        let session = preview();
        let expected = if cfg!(target_os = "macos") {
            Why::Disabled
        } else {
            Why::NotOnThisOs
        };
        assert_eq!(apply_accepted(&session), Applied::Unsupported(expected));
    }
}
