//! Tidy's rule pass: English fillers, the person's own filler habits,
//! immediate word repeats, and conservative spoken self-corrections, found
//! in the pasted text and turned into removals. Every removal goes through
//! [`removal`], which also fixes what the words leave behind: the commas
//! that set them off, and the capital of the word that now starts the
//! sentence (D-61).

use std::ops::Range;

use super::{Habits, Kind};

/// The hesitations that are never words in an English dictation. Like
/// Labs › Remove filler words, a filler matches lowercase or with only its
/// first letter capitalised, so "ER" and "Hmmm…"-like spellings stay.
const FILLERS: &[&str] = &["um", "umm", "uh", "uhh", "erm", "er", "hmm", "hm"];

/// Words people repeat on purpose ("I know that that works", "it had had
/// time", "very very good", "so-so"), never collapsed.
const MEANINGFUL_REPEATS: &[&str] = &[
    "that", "had", "is", "very", "really", "so", "no", "yes", "bye", "ha", "la", "na", "ho",
];

/// Spoken self-correction cues, and whether each is strong. A strong cue
/// only ever corrects; a weak one ("I mean", "sorry") is also ordinary
/// speech, so it needs the phrases on both sides to look alike.
const CUES: &[(&[&str], bool)] = &[
    (&["scratch", "that"], true),
    (&["no", "wait"], true),
    (&["i", "mean"], false),
    (&["sorry"], false),
];

/// The longest corrected phrase a cue may take back.
const MAX_CORRECTION_WORDS: usize = 4;

pub(super) const FILLER_CONFIDENCE: f32 = 0.95;
const REPEAT_CONFIDENCE: f32 = 0.9;
const STRONG_CUE_CONFIDENCE: f32 = 0.8;
const WEAK_CUE_CONFIDENCE: f32 = 0.6;

/// One word of the pasted text: its byte range and its lowercase spelling
/// (a curly apostrophe read as a straight one).
#[derive(Debug)]
pub(super) struct Word {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) lower: String,
}

/// The pasted text split into words; what lies between two words is a gap.
pub(super) struct Text<'a> {
    pub(super) source: &'a str,
    pub(super) words: Vec<Word>,
}

impl<'a> Text<'a> {
    pub(super) fn new(source: &'a str) -> Self {
        let chars: Vec<(usize, char)> = source.char_indices().collect();
        let mut words = Vec::new();
        let mut start = None;
        for (index, &(offset, character)) in chars.iter().enumerate() {
            let inner_apostrophe = is_apostrophe(character)
                && start.is_some()
                && chars
                    .get(index + 1)
                    .is_some_and(|(_, next)| next.is_alphanumeric());
            if character.is_alphanumeric() || inner_apostrophe {
                start.get_or_insert(offset);
            } else if let Some(begin) = start.take() {
                words.push(word(source, begin, offset));
            }
        }
        if let Some(begin) = start {
            words.push(word(source, begin, source.len()));
        }
        Self { source, words }
    }

    /// The text before word `index`: the leading text for the first word,
    /// the trailing text for `index == words.len()`.
    pub(super) fn gap(&self, index: usize) -> &'a str {
        let start = index
            .checked_sub(1)
            .map_or(0, |previous| self.words[previous].end);
        let end = self
            .words
            .get(index)
            .map_or(self.source.len(), |word| word.start);
        &self.source[start..end]
    }

    fn word_text(&self, index: usize) -> &'a str {
        let word = &self.words[index];
        &self.source[word.start..word.end]
    }
}

fn word(source: &str, start: usize, end: usize) -> Word {
    Word {
        start,
        end,
        lower: source[start..end].to_lowercase().replace('\u{2019}', "'"),
    }
}

pub(super) fn is_apostrophe(character: char) -> bool {
    character == '\'' || character == '\u{2019}'
}

/// One edit in byte offsets of the pasted text.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Edit {
    pub(super) range: Range<usize>,
    pub(super) replacement: String,
    pub(super) kind: Kind,
    pub(super) confidence: f32,
}

/// Words the rule pass would remove, before they become an edit.
#[derive(Clone, Copy, Debug)]
struct Group {
    first: usize,
    last: usize,
    kind: Kind,
    confidence: f32,
}

/// The rule pass over an English dictation.
pub(super) fn suggest(text: &Text, habits: &Habits) -> Vec<Edit> {
    let count = text.words.len();
    let mut claimed = vec![false; count];
    let mut groups = Vec::new();
    let mut claim = |group: Group, groups: &mut Vec<Group>| {
        if claimed[group.first..=group.last].iter().any(|taken| *taken) {
            return;
        }
        claimed[group.first..=group.last].fill(true);
        groups.push(group);
    };
    // Self-corrections first: they span the most words, and a filler or a
    // repeat inside one goes with it.
    for group in self_corrections(text) {
        claim(group, &mut groups);
    }
    for group in fillers(text)
        .into_iter()
        .chain(habit_phrases(text, habits))
        .chain(repeats(text))
    {
        claim(group, &mut groups);
    }
    groups.sort_by_key(|group| group.first);
    merge_adjacent(groups)
        .into_iter()
        .filter_map(|group| removal(text, group.first, group.last, group.kind, group.confidence))
        .collect()
}

fn fillers(text: &Text) -> Vec<Group> {
    (0..text.words.len())
        .filter(|&index| is_filler(text.word_text(index)))
        .map(|index| Group {
            first: index,
            last: index,
            kind: Kind::Filler,
            confidence: FILLER_CONFIDENCE,
        })
        .collect()
}

/// Whether `word` is a built-in filler, lowercase or first-letter capitalised.
pub(super) fn is_filler(word: &str) -> bool {
    FILLERS.iter().any(|filler| {
        word == *filler || {
            let mut chars = word.chars();
            chars.next().is_some_and(|first| {
                first.is_uppercase()
                    && first.to_lowercase().eq(filler.chars().take(1))
                    && chars.as_str() == &filler[1..]
            })
        }
    })
}

/// The person's habits (T8.2), only where a filler sits: set off by a
/// comma on at least one side, with a clause boundary or comma on both.
/// "Like I said" keeps its "like"; "it's, like, fine" loses it.
fn habit_phrases(text: &Text, habits: &Habits) -> Vec<Group> {
    let mut groups = Vec::new();
    for habit in &habits.phrases {
        let phrase: Vec<String> = habit
            .phrase
            .split_whitespace()
            .map(|word| word.to_lowercase())
            .collect();
        if phrase.is_empty() || phrase.len() > text.words.len() {
            continue;
        }
        for first in 0..=text.words.len() - phrase.len() {
            let last = first + phrase.len() - 1;
            let matches = phrase
                .iter()
                .enumerate()
                .all(|(offset, word)| text.words[first + offset].lower == *word);
            let plain_inside = (first + 1..=last).all(|index| is_blank(text.gap(index)));
            if !matches || !plain_inside {
                continue;
            }
            let before = text.gap(first);
            let after = text.gap(last + 1);
            let left = before.contains(',') || first == 0 || ends_clause(before);
            let right = after.contains(',') || last + 1 == text.words.len() || ends_clause(after);
            if left && right && (before.contains(',') || after.contains(',')) {
                groups.push(Group {
                    first,
                    last,
                    kind: Kind::Filler,
                    confidence: habit.confidence,
                });
            }
        }
    }
    groups
}

/// The later copy of an immediately repeated word ("the the", "I, I").
fn repeats(text: &Text) -> Vec<Group> {
    (1..text.words.len())
        .filter(|&index| {
            let word = &text.words[index].lower;
            let gap = text.gap(index);
            *word == text.words[index - 1].lower
                && !MEANINGFUL_REPEATS.contains(&word.as_str())
                && !word.chars().any(|character| character.is_numeric())
                && gap.chars().filter(|character| *character == ',').count() <= 1
                && gap
                    .chars()
                    .all(|character| character == ',' || is_inline_space(character))
        })
        .map(|index| Group {
            first: index,
            last: index,
            kind: Kind::Repeat,
            confidence: REPEAT_CONFIDENCE,
        })
        .collect()
}

/// "send it Tuesday, scratch that, Wednesday": the cue and the phrase it
/// takes back, which is as long as the phrase after it. The cue must be set
/// off by punctuation on both sides (a comma for a weak cue), so "I mean it"
/// and "I mean, it's fine" never match.
fn self_corrections(text: &Text) -> Vec<Group> {
    let count = text.words.len();
    let mut groups = Vec::new();
    for cue in 0..count {
        for (words, strong) in CUES {
            let after_cue = cue + words.len();
            if after_cue > count
                || !words
                    .iter()
                    .enumerate()
                    .all(|(offset, word)| text.words[cue + offset].lower == *word)
            {
                continue;
            }
            // The words of a two-word cue sit next to each other.
            if (cue + 1..after_cue).any(|index| !is_blank(text.gap(index))) {
                continue;
            }
            let set_off = |gap: &str| {
                gap.contains(',') || (*strong && (gap.contains('.') || gap.contains(';')))
            };
            if cue == 0 || after_cue == count {
                continue;
            }
            if !set_off(text.gap(cue)) || !set_off(text.gap(after_cue)) {
                continue;
            }
            // The corrected phrase runs to the next punctuation or the end.
            let mut correction = 1;
            while after_cue + correction < count
                && is_blank(text.gap(after_cue + correction))
                && correction <= MAX_CORRECTION_WORDS
            {
                correction += 1;
            }
            if correction > MAX_CORRECTION_WORDS || correction > cue {
                continue;
            }
            let abandoned = cue - correction;
            if (abandoned + 1..cue).any(|index| !is_blank(text.gap(index))) {
                continue;
            }
            if !strong && !alike(text, abandoned, after_cue, correction) {
                continue;
            }
            groups.push(Group {
                first: abandoned,
                last: after_cue - 1,
                kind: Kind::SelfCorrection,
                confidence: if *strong {
                    STRONG_CUE_CONFIDENCE
                } else {
                    WEAK_CUE_CONFIDENCE
                },
            });
        }
    }
    groups
}

/// Whether a weak cue's corrected phrase replaces one slot of the abandoned
/// one: every word the same but exactly one, and that pair both names or
/// both numbers ("Tuesday, I mean, Wednesday"; "to Bob, sorry, to Alice";
/// "at five, I mean, at six"). A shared start alone is not evidence: "I love
/// you, I mean, I really do", "This is huge, I mean, this is massive", and
/// "It's fine, I mean, mostly" are elaborations, and stay.
fn alike(text: &Text, abandoned: usize, corrected: usize, length: usize) -> bool {
    let differing: Vec<usize> = (0..length)
        .filter(|offset| {
            text.words[abandoned + offset].lower != text.words[corrected + offset].lower
        })
        .collect();
    let [offset] = differing[..] else {
        return false;
    };
    let shape = |index: usize| {
        let word = text.word_text(index);
        if word.chars().all(char::is_numeric)
            || NUMBER_WORDS.contains(&text.words[index].lower.as_str())
        {
            Some(true)
        } else if word.chars().next().is_some_and(char::is_uppercase) && word != "I" {
            Some(false)
        } else {
            None
        }
    };
    let (before, after) = (shape(abandoned + offset), shape(corrected + offset));
    before.is_some() && before == after
}

/// Spelled-out numbers a weak cue may correct ("at five, I mean, at six").
const NUMBER_WORDS: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
];

fn merge_adjacent(groups: Vec<Group>) -> Vec<Group> {
    let mut merged: Vec<Group> = Vec::with_capacity(groups.len());
    for group in groups {
        match merged.last_mut() {
            Some(previous) if previous.last + 1 == group.first => {
                previous.last = group.last;
                previous.kind = previous.kind.max(group.kind);
                previous.confidence = previous.confidence.min(group.confidence);
            }
            _ => merged.push(group),
        }
    }
    merged
}

/// The edit that removes words `first..=last` and repairs what is left:
/// the commas that set them off go, the stronger punctuation and any line
/// break stay, and when the words started a sentence the next word takes
/// the capital. That capital is part of the same edit: its range runs to
/// the end of the next word and its replacement spells that word
/// capitalised, so edits stay disjoint and each one alone reads right
/// (D-61). `None` when the words touch anything but spaces and sentence
/// punctuation (quotes, brackets, hyphens), cross a line break, or are the
/// whole text.
pub(super) fn removal(
    text: &Text,
    first: usize,
    last: usize,
    kind: Kind,
    confidence: f32,
) -> Option<Edit> {
    let count = text.words.len();
    if first > last || last >= count || (first == 0 && last + 1 == count) {
        return None;
    }
    for index in first..=last + 1 {
        let gap = text.gap(index);
        if !gap.chars().all(is_allowed_gap) {
            return None;
        }
        if index > first && index <= last && gap.contains('\n') {
            return None;
        }
    }
    let before = text.gap(first);
    let after = text.gap(last + 1);
    let (start, mut end, mut replacement, capitalize) = if first == 0 {
        let kept = line_break(after).unwrap_or_default();
        (text.words[0].start, text.words[last + 1].start, kept, true)
    } else if last + 1 == count {
        let punctuation = Some(punctuation(after))
            .filter(|marks| !marks.is_empty())
            .unwrap_or_else(|| punctuation(before));
        let trailing = &after[after.trim_end().len()..];
        (
            text.words[first - 1].end,
            text.source.len(),
            punctuation + trailing,
            false,
        )
    } else {
        // A self-correction's own trailing marks only set off its cue
        // ("Tuesday. Scratch that. Wednesday"), so they go with it.
        let punctuation = Some(punctuation(before))
            .filter(|marks| !marks.is_empty() || kind == Kind::SelfCorrection)
            .unwrap_or_else(|| punctuation(after));
        let space = line_break(before)
            .or_else(|| line_break(after))
            .unwrap_or_else(|| " ".to_owned());
        (
            text.words[first - 1].end,
            text.words[last + 1].start,
            punctuation + &space,
            ends_sentence(before),
        )
    };
    if capitalize && let Some(word) = capitalized(text.word_text(last + 1)) {
        end = text.words[last + 1].end;
        replacement.push_str(&word);
    }
    let (start, end) = trim_shared_space(text.source, start, end, &mut replacement);
    Some(Edit {
        range: start..end,
        replacement,
        kind,
        confidence,
    })
}

/// `word` with its first letter capitalised, when that is all it needs: a
/// word with a capital already, or inside ("iPhone"), stays as it is.
fn capitalized(word: &str) -> Option<String> {
    let mut chars = word.chars();
    let first = chars.next()?;
    if !first.is_lowercase() || chars.as_str().chars().any(char::is_uppercase) {
        return None;
    }
    Some(first.to_uppercase().chain(chars).collect())
}

/// Narrows an edit by the whitespace its range and replacement share at
/// either end, so a removal reads "uh," → "" rather than ", uh, " → " ".
fn trim_shared_space(
    source: &str,
    mut start: usize,
    mut end: usize,
    replacement: &mut String,
) -> (usize, usize) {
    while let (Some(original), Some(kept)) = (
        source[start..end].chars().next(),
        replacement.chars().next(),
    ) && original == kept
        && original.is_whitespace()
    {
        start += original.len_utf8();
        replacement.remove(0);
    }
    while let (Some(original), Some(kept)) = (
        source[start..end].chars().next_back(),
        replacement.chars().next_back(),
    ) && original == kept
        && original.is_whitespace()
    {
        end -= original.len_utf8();
        replacement.pop();
    }
    (start, end)
}

/// The gap characters a removal may absorb: spaces and line breaks, commas,
/// and sentence punctuation.
fn is_allowed_gap(character: char) -> bool {
    character.is_whitespace() || matches!(character, ',' | '.' | '!' | '?' | ';' | ':' | '…')
}

/// A gap's marks other than commas and whitespace.
fn punctuation(gap: &str) -> String {
    gap.chars()
        .filter(|character| !character.is_whitespace() && *character != ',')
        .collect()
}

/// A gap's whitespace from its first line break, when it has one.
fn line_break(gap: &str) -> Option<String> {
    let at = gap.find('\n')?;
    Some(gap[at..].chars().filter(|c| c.is_whitespace()).collect())
}

fn ends_sentence(gap: &str) -> bool {
    gap.contains(['.', '!', '?', '\n'])
}

fn ends_clause(gap: &str) -> bool {
    ends_sentence(gap) || gap.contains([';', ':'])
}

fn is_inline_space(character: char) -> bool {
    character.is_whitespace() && character != '\n'
}

/// A gap of spaces only: no punctuation, no line break.
fn is_blank(gap: &str) -> bool {
    !gap.is_empty() && gap.chars().all(is_inline_space)
}
