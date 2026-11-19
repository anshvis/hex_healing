//! `hex dictation tidy [--print]` (Labs › Tidy after paste): the reply a
//! prepared review gives, and the JSON `--print` answers with
//! (`docs/cli.md`). Each OS's host prepares the review; this decides what
//! the command says, so every host answers the same way.

use serde::{Deserialize, Serialize};

use super::{ControlError, ControlReply};
use crate::dictation::tidy::review::Refusal;
use crate::dictation::tidy::session::{Delivery, TidySession};
use crate::dictation::tidy::{Kind, Source};

/// `--print`'s answer: the last dictation's suggestions, as the review
/// would open with them, applied to nothing.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TidyReport {
    /// The History entry reviewed.
    pub entry: u64,
    /// The dictation only reached the clipboard (or was printed), so the
    /// review offers Copy tidied text rather than editing a field.
    pub copy_only: bool,
    /// The text it pasted, which the ranges index.
    pub text: String,
    /// `text` with the suggestions that start accepted applied.
    pub tidied: String,
    pub suggestions: Vec<TidySuggestion>,
}

/// One suggestion: replace `text[range]` with `replacement`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TidySuggestion {
    /// Start and end in UTF-16 code units, as Accessibility counts.
    pub utf16: [usize; 2],
    /// Start and end in Unicode scalar values.
    pub scalar: [usize; 2],
    /// `filler`, `repeat`, `self_correction`, or `misheard`.
    pub kind: String,
    /// Empty for a removal.
    pub replacement: String,
    /// `rule` or `model`.
    pub source: String,
    pub confidence: f32,
    /// Whether the review starts with it accepted.
    pub accepted: bool,
}

impl TidyReport {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn of(session: &TidySession) -> Self {
        Self {
            entry: session.dictation_id,
            copy_only: session.delivery == Delivery::Copied,
            text: session.text.clone(),
            tidied: session.tidied(),
            suggestions: session
                .suggestions
                .iter()
                .zip(&session.accepted)
                .map(|(suggestion, accepted)| TidySuggestion {
                    utf16: [suggestion.range.utf16.start, suggestion.range.utf16.end],
                    scalar: [suggestion.range.scalar.start, suggestion.range.scalar.end],
                    kind: kind_name(suggestion.kind).into(),
                    replacement: suggestion.replacement.clone(),
                    source: match suggestion.source {
                        Source::Rule => "rule",
                        Source::Model => "model",
                    }
                    .into(),
                    confidence: suggestion.confidence,
                    accepted: *accepted,
                })
                .collect(),
        }
    }
}

/// T8.1's kind in snake case, as T8.5's `tidy_events` names it (D-66).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) const fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Filler => "filler",
        Kind::Repeat => "repeat",
        Kind::SelfCorrection => "self_correction",
        Kind::Misheard => "misheard",
    }
}

/// What `hex dictation tidy` answers for `prepared`: `--print` the report;
/// otherwise the review opens through `open` and the reply says what it
/// shows. Nothing opens or prints with the lab off.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn reply(
    print: bool,
    prepared: Result<TidySession, Refusal>,
    open: impl FnOnce(TidySession),
) -> Result<ControlReply, ControlError> {
    let session = prepared.map_err(|refusal| {
        let code = match refusal {
            Refusal::LabOff => "disabled",
            Refusal::NoDictation => "nothing-to-tidy",
        };
        ControlError::new(code, refusal.message())
    })?;
    if print {
        return Ok(ControlReply::TidySuggestions(Box::new(TidyReport::of(
            &session,
        ))));
    }
    let message = match session.suggestions.len() {
        0 => "Nothing to tidy in the last dictation".to_owned(),
        1 => "Opened the tidy review with 1 suggestion".to_owned(),
        count => format!("Opened the tidy review with {count} suggestions"),
    };
    open(session);
    Ok(ControlReply::done(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictation::tidy::session::preview;

    #[test]
    fn tidy_command_refused_when_lab_off() {
        for print in [false, true] {
            let error = reply(print, Err(Refusal::LabOff), |_| {
                panic!("opened with the lab off")
            })
            .unwrap_err();
            assert_eq!(error.code, "disabled");
            assert!(
                error
                    .message
                    .contains("hex settings set labs.tidy_after_paste on"),
                "{}",
                error.message
            );
        }
        let error = reply(false, Err(Refusal::NoDictation), |_| panic!()).unwrap_err();
        assert_eq!(error.code, "nothing-to-tidy");
    }

    #[test]
    fn print_answers_the_suggestions_without_opening_anything() {
        let reply = reply(true, Ok(preview()), |_| panic!("--print opened the review")).unwrap();
        let ControlReply::TidySuggestions(report) = &reply else {
            panic!("{reply:?}");
        };
        assert_eq!(report.entry, 7);
        assert!(!report.copy_only);
        assert_eq!(report.tidied, "So I think we should send it Wednesday");
        let json = serde_json::to_value(super::super::ControlResponse::from(reply)).unwrap();
        assert_eq!(json["status"], "tidy-suggestions");
        assert_eq!(json["version"], super::super::CONTROL_VERSION);
        let first = &json["suggestions"][0];
        assert_eq!(first["utf16"], serde_json::json!([0, 5]));
        assert_eq!(first["scalar"], serde_json::json!([0, 5]));
        assert_eq!(first["kind"], "filler");
        assert_eq!(first["replacement"], "So");
        assert_eq!(first["source"], "rule");
        assert_eq!(first["accepted"], true);
        assert!(first["confidence"].as_f64().is_some());
        let misheard = &json["suggestions"][2];
        assert_eq!(misheard["kind"], "misheard");
        assert_eq!(misheard["source"], "model");
        assert_eq!(misheard["accepted"], false);
        assert_eq!(json["suggestions"][3]["kind"], "self_correction");
        // The fields docs/cli.md names, and no others.
        let mut keys: Vec<&str> = first
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "accepted",
                "confidence",
                "kind",
                "replacement",
                "scalar",
                "source",
                "utf16"
            ]
        );
    }

    #[test]
    fn without_print_the_review_opens_and_says_how_many() {
        let mut opened = None;
        let reply = reply(false, Ok(preview()), |session| opened = Some(session)).unwrap();
        assert_eq!(
            reply,
            ControlReply::done("Opened the tidy review with 4 suggestions")
        );
        assert_eq!(opened.unwrap().dictation_id, 7);
    }
}
