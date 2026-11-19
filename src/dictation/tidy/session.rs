//! One review of the last dictation's tidy suggestions (T8.3): the text it
//! pasted, T8.1's suggestions, and which of them the person accepts. Each
//! starts accepted unless it is a model `misheard` fix, which changes a word
//! rather than dropping one, so it waits for a yes (D-67).
//!
//! One session at a time, opened from the History entry when a trigger fires
//! and closed by Return, Escape, or the next dictation (its `generation`
//! falls behind [`super::last::generation`]). Nothing here is written down.

use std::time::Instant;

use super::{Kind, Suggestion};

/// How a dictation reached the app it was for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Delivery {
    /// Pasted into the focused field, where T8.4 can edit it.
    Pasted,
    /// Only copied (a copy fallback, Copy dictation only) or printed
    /// (`hex dictation stop --print`): there is no field to edit, so the
    /// review offers Copy tidied text instead.
    Copied,
}

/// The dictation a tidy reviews: its History entry and what it inserted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Dictation {
    pub(crate) id: u64,
    /// The final text it pasted, which the suggestions index.
    pub(crate) pasted: String,
    /// What the speech model heard, context for the model pass.
    pub(crate) raw: String,
    /// Its language code, `"auto"` when the model chose.
    pub(crate) language: String,
    pub(crate) delivery: Delivery,
}

/// One part of the reviewed text, in order: text no suggestion touches, or
/// one suggestion's span with what accepting it puts there.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Segment {
    Plain(String),
    Edit {
        index: usize,
        /// The words the suggestion replaces, as pasted.
        before: String,
        /// What replaces them: empty for a removal.
        after: String,
        accepted: bool,
    },
}

/// A review in progress.
#[derive(Clone, Debug)]
pub(crate) struct TidySession {
    pub(crate) dictation_id: u64,
    pub(crate) text: String,
    pub(crate) suggestions: Vec<Suggestion>,
    pub(crate) accepted: Vec<bool>,
    pub(crate) delivery: Delivery,
    #[allow(dead_code)]
    pub(crate) opened_at: Instant,
    /// The dictation count when it opened ([`super::last::generation`]).
    pub(crate) generation: u64,
}

impl TidySession {
    pub(crate) fn new(dictation: Dictation, suggestions: Vec<Suggestion>, generation: u64) -> Self {
        let accepted = suggestions.iter().map(default_accepted).collect();
        Self {
            dictation_id: dictation.id,
            text: dictation.pasted,
            suggestions,
            accepted,
            delivery: dictation.delivery,
            opened_at: Instant::now(),
            generation,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.suggestions.is_empty()
    }

    /// Accepts suggestion `index` if it was skipped, or skips it.
    pub(crate) fn toggle(&mut self, index: usize) {
        if let Some(accepted) = self.accepted.get_mut(index) {
            *accepted = !*accepted;
        }
    }

    pub(crate) fn accept_all(&mut self) {
        self.accepted
            .iter_mut()
            .for_each(|accepted| *accepted = true);
    }

    pub(crate) fn accepted_count(&self) -> usize {
        self.accepted.iter().filter(|accepted| **accepted).count()
    }

    /// The accepted suggestions, in text order.
    pub(crate) fn accepted_suggestions(&self) -> Vec<Suggestion> {
        self.suggestions
            .iter()
            .zip(&self.accepted)
            .filter(|(_, accepted)| **accepted)
            .map(|(suggestion, _)| suggestion.clone())
            .collect()
    }

    /// The pasted text with the accepted suggestions applied.
    pub(crate) fn tidied(&self) -> String {
        super::apply(&self.text, &self.accepted_suggestions())
    }

    /// Whether a dictation committed after this review opened: the review
    /// is about text that is no longer the last, so it closes.
    pub(crate) fn is_stale(&self, generation: u64) -> bool {
        generation != self.generation
    }

    /// The text in parts, each suggestion's span its own part.
    pub(crate) fn segments(&self) -> Vec<Segment> {
        let chars: Vec<char> = self.text.chars().collect();
        let slice = |range: std::ops::Range<usize>| -> String {
            chars[range.start.min(chars.len())..range.end.min(chars.len())]
                .iter()
                .collect()
        };
        let mut segments = Vec::new();
        let mut at = 0;
        for (index, suggestion) in self.suggestions.iter().enumerate() {
            let range = suggestion.range.scalar.clone();
            if range.start > at {
                segments.push(Segment::Plain(slice(at..range.start)));
            }
            segments.push(Segment::Edit {
                index,
                before: slice(range.clone()),
                after: suggestion.replacement.clone(),
                accepted: self.accepted[index],
            });
            at = range.end.max(at);
        }
        if at < chars.len() {
            segments.push(Segment::Plain(slice(at..chars.len())));
        }
        segments
    }
}

/// A fixed review for the `tidy-capsule` scene and the tests: AC-15's
/// dictation with T8.1's three rule suggestions (rules only, so no model
/// runs), plus a model `misheard` fix that starts skipped.
pub(crate) fn preview() -> TidySession {
    use super::{Offsets, Source};
    let pasted = "um so I think we should, uh, send it Tuesday, scratch that, Wednesday";
    let mut suggestions = super::suggest_with(pasted, pasted, "en", &Default::default(), |_| None);
    let start = pasted.find("send").unwrap_or_default();
    suggestions.push(Suggestion {
        range: Offsets {
            utf16: start..start + 4,
            scalar: start..start + 4,
        },
        kind: Kind::Misheard,
        replacement: "sent".into(),
        source: Source::Model,
        confidence: 0.5,
    });
    suggestions.sort_by_key(|suggestion| suggestion.range.scalar.start);
    TidySession::new(
        Dictation {
            id: 7,
            pasted: pasted.into(),
            raw: pasted.into(),
            language: "en".into(),
            delivery: Delivery::Pasted,
        },
        suggestions,
        0,
    )
}

/// Whether a suggestion starts accepted: every removal and rule edit does;
/// a `misheard` fix waits for a yes.
pub(crate) fn default_accepted(suggestion: &Suggestion) -> bool {
    suggestion.kind != Kind::Misheard
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TidySession {
        preview()
    }

    #[test]
    fn rule_suggestions_start_accepted_and_misheard_fixes_do_not() {
        let session = fixture();
        assert_eq!(session.suggestions.len(), 4);
        assert_eq!(session.accepted, [true, true, false, true]);
        assert_eq!(session.tidied(), "So I think we should send it Wednesday");
    }

    #[test]
    fn toggle_and_accept_all() {
        let mut session = fixture();
        session.toggle(1);
        assert_eq!(session.accepted, [true, false, false, true]);
        assert_eq!(
            session.tidied(),
            "So I think we should, uh, send it Wednesday"
        );
        session.toggle(1);
        session.toggle(0);
        assert_eq!(
            session.tidied(),
            "um so I think we should send it Wednesday"
        );
        // Out of range changes nothing.
        session.toggle(99);
        session.accept_all();
        assert_eq!(session.accepted, [true; 4]);
        assert_eq!(session.accepted_count(), 4);
        assert_eq!(session.tidied(), "So I think we should sent it Wednesday");
        // Skipping everything leaves the text as pasted.
        (0..4).for_each(|index| session.toggle(index));
        assert_eq!(session.tidied(), session.text);
    }

    #[test]
    fn segments_cover_the_text_once_in_order() {
        let session = fixture();
        let segments = session.segments();
        let rebuilt: String = segments
            .iter()
            .map(|segment| match segment {
                Segment::Plain(text) => text.as_str(),
                Segment::Edit { before, .. } => before.as_str(),
            })
            .collect();
        assert_eq!(rebuilt, session.text);
        assert_eq!(
            segments[0],
            Segment::Edit {
                index: 0,
                before: "um so".into(),
                after: "So".into(),
                accepted: true,
            }
        );
    }

    /// A dictation that commits after the review opened makes it stale.
    #[test]
    fn session_replaced_by_new_dictation() {
        let session = fixture();
        assert!(!session.is_stale(session.generation));
        assert!(session.is_stale(session.generation + 1));
    }
}
