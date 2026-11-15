//! Where a pasted dictation sits in the field now. Pure: the watcher reads
//! the field, this decides whether the text is still there, and if not,
//! which span of the field replaced it.
//!
//! Token-level Smith-Waterman of the inserted tokens against the field's.
//! Tokens fold case and punctuation. An exact token scores +2; a near token
//! (three or more letters, one edit apart, such as "Kip" and "Kit") +1, so a
//! one-word edit at the start, end, or alone still aligns; any other pair
//! -1; a gap -1. A field with no near or exact token never aligns, except a
//! lone word replaced by a lone word. An edited first or last word scores
//! below zero, so it falls outside the aligned core: once the core alone
//! covers half the inserted tokens (all of them below three), the span
//! extends over at most one field token at each end that has an inserted
//! token left over, never across sentence punctuation.

/// How the inserted text sits in the field.
#[derive(Debug, Eq, PartialEq)]
pub enum Alignment {
    /// The inserted text is still in the field as pasted.
    Unchanged,
    /// The field's span that replaced it, at most [`span_cap`] characters.
    Edited(String),
    NotFound,
}

const MATCH: i32 = 2;
const NEAR: i32 = 1;
const MISMATCH: i32 = -1;
const GAP: i32 = -1;

/// The most characters an edited span keeps for an insert of `inserted_chars`.
pub const fn span_cap(inserted_chars: usize) -> usize {
    2 * inserted_chars + 64
}

struct Token {
    fold: Vec<char>,
    start: usize,
    end: usize,
}

fn tokens(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut start = None;
    let flush = |from: usize, to: usize, out: &mut Vec<Token>| {
        let fold: Vec<char> = text[from..to]
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        if !fold.is_empty() {
            out.push(Token {
                fold,
                start: from,
                end: to,
            });
        }
    };
    for (at, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (false, None) => start = Some(at),
            (true, Some(from)) => {
                flush(from, at, &mut out);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(from) = start {
        flush(from, text.len(), &mut out);
    }
    out
}

fn edit_within_one(a: &[char], b: &[char]) -> bool {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if long.len() - short.len() > 1 {
        return false;
    }
    let skip = short.iter().zip(long).take_while(|(x, y)| x == y).count();
    let tail = short[skip..]
        .iter()
        .rev()
        .zip(long[skip..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let left_short = short.len() - skip - tail.min(short.len() - skip);
    let left_long = long.len() - skip - tail.min(long.len() - skip);
    left_short <= 1 && left_long <= 1
}

fn score(inserted: &Token, field: &Token) -> i32 {
    if inserted.fold == field.fold {
        MATCH
    } else if inserted.fold.len() >= 3
        && field.fold.len() >= 3
        && edit_within_one(&inserted.fold, &field.fold)
    {
        NEAR
    } else {
        MISMATCH
    }
}

/// `text` without object-replacement characters (attachments).
fn strip(text: &str) -> String {
    text.replace('\u{FFFC}', "")
}

/// Whether `text` holds nothing a person typed.
pub fn is_blank(text: &str) -> bool {
    strip(text).trim().is_empty()
}

/// Whether a sentence ends in `token`, or on the line break before the next.
fn ends_sentence(field: &str, token: &Token, next: Option<&Token>) -> bool {
    field[token.start..token.end].ends_with(['.', '!', '?'])
        || next.is_some_and(|next| field[token.end..next.start].contains('\n'))
}

/// Whether `want` is a run of consecutive tokens of `have`, folded.
fn contains_run(have: &[Token], want: &[Token]) -> bool {
    have.windows(want.len())
        .any(|run| run.iter().zip(want).all(|(a, b)| a.fold == b.fold))
}

/// Aligns `inserted` against `field`. U+FFFC is stripped from both. A
/// `windowed` field is only part of the field (it held more than a read
/// takes): it can show the paste whole, or not at all, never an edit of it.
pub fn align(field: &str, inserted: &str, windowed: bool) -> Alignment {
    let field = strip(field);
    let inserted = strip(inserted);
    let inserted = inserted.trim();
    let (want, have) = (tokens(inserted), tokens(&field));
    if want.is_empty() || have.is_empty() {
        return Alignment::NotFound;
    }
    // Whole words: "Yes" is not in "Yesterday".
    if contains_run(&have, &want) {
        return Alignment::Unchanged;
    }
    if windowed {
        return Alignment::NotFound;
    }
    if let [_] = &want[..] {
        // One word in, one word out: the field is that word's replacement.
        // Anywhere else a lone word is found only whole.
        return match &have[..] {
            [only] => Alignment::Edited(field[only.start..only.end].to_string()),
            _ => Alignment::NotFound,
        };
    }
    let width = have.len() + 1;
    let mut h = vec![0_i32; (want.len() + 1) * width];
    let (mut best, mut at) = (0, (0, 0));
    for i in 1..=want.len() {
        for j in 1..=have.len() {
            let diagonal = h[(i - 1) * width + j - 1] + score(&want[i - 1], &have[j - 1]);
            let up = h[(i - 1) * width + j] + GAP;
            let left = h[i * width + j - 1] + GAP;
            let cell = diagonal.max(up).max(left).max(0);
            h[i * width + j] = cell;
            // `>=`: of equal alignments the later one, nearest the caret's
            // usual place at the end of a field, wins.
            if cell >= best && cell > 0 {
                (best, at) = (cell, (i, j));
            }
        }
    }
    if best == 0 {
        return Alignment::NotFound;
    }
    let (mut i, mut j) = at;
    let trailing = want.len() - i;
    let (mut aligned, mut exact) = (0, 0);
    let (mut first, mut last) = (None, None);
    let mut gaps = false;
    while i > 0 && j > 0 && h[i * width + j] > 0 {
        let here = h[i * width + j];
        let s = score(&want[i - 1], &have[j - 1]);
        if here == h[(i - 1) * width + j - 1] + s {
            // Only a pair that scores counts: a mismatch is inside the core
            // but is not evidence the text is there.
            aligned += usize::from(s > 0);
            exact += usize::from(s == MATCH);
            last.get_or_insert(j - 1);
            first = Some(j - 1);
            i -= 1;
            j -= 1;
        } else if here == h[(i - 1) * width + j] + GAP {
            gaps = true;
            i -= 1;
        } else {
            gaps = true;
            j -= 1;
        }
    }
    let needed = if want.len() < 3 {
        want.len()
    } else {
        want.len().div_ceil(2)
    };
    let (Some(mut first), Some(mut last)) = (first, last) else {
        return Alignment::NotFound;
    };
    // The aligned core must cover the inserted text by itself (T2.2's 50% rule),
    // counting exact and near pairs only, with at least two exact words in a
    // text of three or more: an extended edge or a mismatch never makes up
    // the difference, or an unrelated field sharing a phrase would be stored
    // as an edit. At most one inserted token may be left over at an end: more
    // is a different text that shares the rest.
    if aligned < needed || (want.len() >= 3 && exact < 2) || i > 1 || trailing > 1 {
        return Alignment::NotFound;
    }
    if exact == want.len() && !gaps {
        return Alignment::Unchanged;
    }
    // One inserted token left over at an end stands for the field token the
    // person typed over it.
    if i > 0 && first > 0 && !ends_sentence(&field, &have[first - 1], Some(&have[first])) {
        first -= 1;
    }
    if trailing > 0
        && last + 1 < have.len()
        && !ends_sentence(&field, &have[last], Some(&have[last + 1]))
    {
        last += 1;
    }
    let span: String = field[have[first].start..have[last].end]
        .chars()
        .take(span_cap(inserted.chars().count()))
        .collect();
    Alignment::Edited(span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edited(field: &str, inserted: &str) -> String {
        match align(field, inserted, false) {
            Alignment::Edited(span) => span,
            other => panic!("{field:?} vs {inserted:?}: {other:?}"),
        }
    }

    #[test]
    fn exact_text_is_unchanged() {
        assert_eq!(
            align("send it to Kip", "send it to Kip", false),
            Alignment::Unchanged
        );
        assert_eq!(
            align("hi. send it to Kip. bye", " send it to Kip ", false),
            Alignment::Unchanged
        );
    }

    #[test]
    fn last_word_edit() {
        assert_eq!(edited("send it to Kit", "send it to Kip"), "send it to Kit");
    }

    #[test]
    fn first_word_edit() {
        assert_eq!(edited("Kit said hi", "Kip said hi"), "Kit said hi");
    }

    #[test]
    fn only_word_edit() {
        assert_eq!(edited("Kit", "Kip"), "Kit");
    }

    #[test]
    fn middle_edit() {
        assert_eq!(
            edited("send them to Kip", "send it to Kip"),
            "send them to Kip"
        );
    }

    #[test]
    fn insertion_in_the_field() {
        assert_eq!(
            edited("send it right to Kip!", "send it to Kip"),
            "send it right to Kip!"
        );
    }

    #[test]
    fn deletion_from_the_field() {
        assert_eq!(edited("send it Kit", "send it to Kit"), "send it Kit");
    }

    #[test]
    fn punctuation_and_case_alone_are_unchanged() {
        assert_eq!(
            align("Send it to kip.", "send it to Kip", false),
            Alignment::Unchanged
        );
    }

    #[test]
    fn duplicate_pastes_stay_unchanged_while_one_copy_is_intact() {
        assert_eq!(
            align("Kit said hi. Kip said hi", "Kip said hi", false),
            Alignment::Unchanged
        );
        assert_eq!(
            edited("a Kit said hi. b Kit said hi", "Kip said hi"),
            "Kit said hi"
        );
    }

    #[test]
    fn attachment_characters_are_stripped() {
        assert_eq!(
            align("send\u{FFFC} it to Kip", "send it to Kip\u{FFFC}", false),
            Alignment::Unchanged
        );
    }

    #[test]
    fn unrelated_field_is_not_found() {
        assert_eq!(
            align("the weather is nice today", "send it to Kip", false),
            Alignment::NotFound
        );
        assert_eq!(align("", "send it to Kip", false), Alignment::NotFound);
        assert_eq!(align("hello world", "Kip", false), Alignment::NotFound);
    }

    #[test]
    fn new_text_typed_after_the_edit_stays_outside_the_span() {
        assert_eq!(
            edited(
                "send it to Kit and then a lot of other words",
                "send it to Kip"
            ),
            "send it to Kit"
        );
    }

    #[test]
    fn a_replaced_last_word_stays_in_the_span() {
        // Not one letter off, so it scores below zero and sits outside the core.
        assert_eq!(
            edited("send it to Hex", "send it to Hanks"),
            "send it to Hex"
        );
        assert_eq!(
            edited("send it to Hex and then more", "send it to Hanks"),
            "send it to Hex"
        );
    }

    #[test]
    fn a_replaced_first_word_stays_in_the_span() {
        assert_eq!(edited("Hex said hi", "Hanks said hi"), "Hex said hi");
    }

    #[test]
    fn a_lone_word_replaced_by_a_lone_word() {
        assert_eq!(edited("Hex", "Hanks"), "Hex");
        assert_eq!(edited(" Hex.", "Hanks"), "Hex.");
        assert_eq!(align("Hex and Rune", "Hanks", false), Alignment::NotFound);
    }

    #[test]
    fn the_span_stops_at_sentence_punctuation() {
        // The word before the pasted sentence is not part of its replaced start.
        assert_eq!(
            edited("Earlier text. Hex said hi", "Hanks said hi"),
            "Hex said hi"
        );
        assert_eq!(
            edited("Hex said hi. Then more words.", "Hanks said hi"),
            "Hex said hi."
        );
        // A last word replaced at the sentence end keeps its full stop but
        // nothing after it.
        assert_eq!(
            edited("send it to Hex. Then more", "send it to Hanks"),
            "send it to Hex."
        );
        assert_eq!(
            edited("Done.\nsend it to Hex\nbye", "send it to Hanks"),
            "send it to Hex"
        );
    }

    /// The seeded protocol (`healing/healing/protocol.py`): this phrase is
    /// dictated and its last word selected and typed over with one of these.
    const PHRASE: &str = "send it to kip tomorrow";
    const REPLACEMENTS: [&str; 7] = ["Kit", "Rune", "Friday", "Hex", "Sarah", "Monday", "banana"];

    #[test]
    fn every_protocol_replacement_lands_inside_the_span() {
        let (head, _) = PHRASE.rsplit_once(' ').unwrap();
        for pasted in [PHRASE.to_string(), "Send it to kip tomorrow.".to_string()] {
            for replacement in REPLACEMENTS {
                for field in [
                    format!("{head} {replacement}"),
                    format!("Send it to kip {replacement}."),
                    format!("Earlier note. Send it to kip {replacement}"),
                    format!("send it to kip {replacement}\nand then something new"),
                ] {
                    let span = edited(&field, &pasted);
                    assert!(
                        span.contains(replacement),
                        "{replacement:?} fell outside {span:?} of {field:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_protocol_replacement_of_each_word_lands_inside_the_span() {
        let words: Vec<&str> = PHRASE.split(' ').collect();
        for at in 0..words.len() {
            for replacement in REPLACEMENTS {
                let mut field = words.clone();
                field[at] = replacement;
                let field = field.join(" ");
                let span = edited(&field, PHRASE);
                assert!(
                    span.contains(replacement),
                    "{replacement:?} fell outside {span:?} of {field:?}"
                );
            }
        }
    }

    #[test]
    fn unrelated_fields_sharing_a_common_word_are_not_found() {
        for (field, inserted) in [
            ("he went to a store yesterday", "go to the store now"),
            ("to be honest the plan", "to be honest I agree"),
            ("send me to", "send it to Kip now please"),
            ("go to the store and then to the bank", "send it to the Kip"),
            (
                "we are going to the park today",
                "please tell Kip we are late",
            ),
            ("go to bed", "send to Kip"),
            ("please talk to me", "send it to Kip"),
            ("I like Kip", "send it to Kip"),
            ("the cat sat on the mat", "put the Kip on it"),
            ("we went to the shops", "we should ask Kip"),
            ("see you at noon", "send it to noon"),
            ("Hex", "send it to Kip"),
            ("hi Hex", "hi Kip"),
        ] {
            assert_eq!(
                align(field, inserted, false),
                Alignment::NotFound,
                "{field:?} vs {inserted:?}"
            );
        }
    }

    #[test]
    fn a_short_insert_is_found_whole_or_not_at_all() {
        // Whole words only, never inside a longer one.
        assert_eq!(align("Yesterday I went", "Yes", false), Alignment::NotFound);
        assert_eq!(align("so yes, ok", "Yes", false), Alignment::Unchanged);
        // A near word anywhere is not an edit; only a lone word is.
        assert_eq!(
            align("thank you for the help", "Thanks", false),
            Alignment::NotFound
        );
        assert_eq!(
            align("thank", "Thanks", false),
            Alignment::Edited("thank".into())
        );
    }

    #[test]
    fn a_windowed_read_shows_the_paste_whole_or_not_at_all() {
        let inserted = "send it to Kip";
        assert_eq!(
            align("a lot of text. send it to Kip", inserted, true),
            Alignment::Unchanged
        );
        // The window cut the paste: that is not an edit.
        assert_eq!(
            align("lot of text. send it to", inserted, true),
            Alignment::NotFound
        );
        assert_eq!(align("send it to Kit", inserted, true), Alignment::NotFound);
        assert_eq!(
            align("send it to Kit", inserted, false),
            Alignment::Edited("send it to Kit".into())
        );
    }

    #[test]
    fn more_than_one_unexplained_token_at_an_end_is_a_different_text() {
        assert_eq!(
            align("send it to Hex please", "send it to Hanks Smith", false),
            Alignment::NotFound
        );
    }
}
