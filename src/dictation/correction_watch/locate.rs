//! Where a pasted dictation sits in the field now. Pure: the watcher reads
//! the field, this decides whether the text is still there, and if not,
//! which span of the field replaced it.
//!
//! Two ways to look, by what is known about where the paste went:
//!
//! - **Anchored** ([`landing`], [`align_at`]): the caret was read before the
//!   paste, so the paste's offset in the field is known. Once the paste has
//!   been seen where it was put (`Landing::Landed`), only the field's region
//!   at that offset is judged, so text elsewhere in the field can never
//!   match, and position is evidence enough to call a one- or two-word
//!   replacement an edit. The region moves with the field's length: an edit
//!   before the paste shifts it, an edit inside stretches it; both are tried
//!   and the one that holds more of the paste wins.
//! - **Strict** ([`align`]): no anchor (the app exposes no selection range),
//!   or one that did not hold. A token-level Smith-Waterman of the inserted
//!   tokens against the whole field's. An exact token scores +2; a near one
//!   (three or more letters, one edit apart, such as "Kip" and "Kit") +1; any
//!   other pair -1; a gap -1. Only exact and near pairs count toward the
//!   floor: half the inserted tokens (all below three), at least two exact
//!   words in three or more, at most one inserted token left over at an end.
//!   An edited first or last word scores below zero, so it sits outside the
//!   core and the span extends one field token over it, never across
//!   sentence punctuation. A lone word is found whole, or is the whole field.
//!
//! Tokens split on whitespace and fold case and punctuation, so text without
//! spaces (CJK) is one token and effectively unsupported.

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
/// The most alignment cells (inserted tokens x field tokens) worth filling.
const MAX_CELLS: usize = 4_000_000;

/// The most characters an edited span keeps for an insert of `inserted_chars`.
pub const fn span_cap(inserted_chars: usize) -> usize {
    2 * inserted_chars + 64
}

struct Token {
    fold: Vec<char>,
    start: usize,
    end: usize,
    /// Where the token sits in the text, in UTF-16 units (what Accessibility
    /// counts in).
    u16_start: usize,
    u16_end: usize,
}

fn tokens(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    let mut at16 = 0;
    let flush = |from: (usize, usize), to: (usize, usize), out: &mut Vec<Token>| {
        let fold: Vec<char> = text[from.0..to.0]
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        if !fold.is_empty() {
            out.push(Token {
                fold,
                start: from.0,
                end: to.0,
                u16_start: from.1,
                u16_end: to.1,
            });
        }
    };
    for (at, c) in text.char_indices() {
        match (c.is_whitespace(), open) {
            (false, None) => open = Some((at, at16)),
            (true, Some(from)) => {
                flush(from, (at, at16), &mut out);
                open = None;
            }
            _ => {}
        }
        at16 += c.len_utf16();
    }
    if let Some(from) = open {
        flush(from, (text.len(), at16), &mut out);
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
    !want.is_empty()
        && have
            .windows(want.len())
            .any(|run| run.iter().zip(want).all(|(a, b)| a.fold == b.fold))
}

/// The best local alignment of `want` in `have`: where it ends, and the
/// scoring matrix to walk back through.
struct Matrix {
    h: Vec<i32>,
    width: usize,
    at: (usize, usize),
}

fn fill(want: &[Token], have: &[Token]) -> Option<Matrix> {
    if want.len().saturating_mul(have.len()) > MAX_CELLS {
        return None;
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
    (best > 0).then_some(Matrix { h, width, at })
}

/// The aligned core found by walking back from the best cell.
struct Core {
    /// Exact and near pairs: the ones that are evidence the text is there.
    aligned: usize,
    exact: usize,
    /// Field token indices of the first and last pair.
    first: usize,
    last: usize,
    /// Inserted tokens left over before the core and after it.
    leading: usize,
    trailing: usize,
    gaps: bool,
}

fn traceback(m: &Matrix, want: &[Token], have: &[Token]) -> Option<Core> {
    let (mut i, mut j) = m.at;
    let trailing = want.len() - i;
    let (mut aligned, mut exact) = (0, 0);
    let (mut first, mut last) = (None, None);
    let mut gaps = false;
    while i > 0 && j > 0 && m.h[i * m.width + j] > 0 {
        let here = m.h[i * m.width + j];
        let s = score(&want[i - 1], &have[j - 1]);
        if here == m.h[(i - 1) * m.width + j - 1] + s {
            // A mismatch is inside the core but is not evidence.
            aligned += usize::from(s > 0);
            exact += usize::from(s == MATCH);
            last.get_or_insert(j - 1);
            first = Some(j - 1);
            i -= 1;
            j -= 1;
        } else if here == m.h[(i - 1) * m.width + j] + GAP {
            gaps = true;
            i -= 1;
        } else {
            gaps = true;
            j -= 1;
        }
    }
    Some(Core {
        aligned,
        exact,
        first: first?,
        last: last?,
        leading: i,
        trailing,
        gaps,
    })
}

fn capped(field: &str, from: usize, to: usize, inserted: &str) -> String {
    field[from..to]
        .chars()
        .take(span_cap(inserted.chars().count()))
        .collect()
}

/// Strict, whole-field alignment of `inserted` in `field`. U+FFFC is
/// stripped from both. A `windowed` field is only part of the field (it held
/// more than a read takes): it can show the paste whole, or not at all,
/// never an edit of it.
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
    let Some(matrix) = fill(&want, &have) else {
        return Alignment::NotFound;
    };
    let Some(core) = traceback(&matrix, &want, &have) else {
        return Alignment::NotFound;
    };
    let needed = if want.len() < 3 {
        want.len()
    } else {
        want.len().div_ceil(2)
    };
    if !accepted(&core, &want, needed) {
        return Alignment::NotFound;
    }
    if core.exact == want.len() && !core.gaps {
        return Alignment::Unchanged;
    }
    let (mut first, mut last) = (core.first, core.last);
    // One inserted token left over at an end stands for the field token the
    // person typed over it.
    if core.leading > 0 && first > 0 && !ends_sentence(&field, &have[first - 1], Some(&have[first]))
    {
        first -= 1;
    }
    if core.trailing > 0
        && last + 1 < have.len()
        && !ends_sentence(&field, &have[last], Some(&have[last + 1]))
    {
        last += 1;
    }
    Alignment::Edited(capped(&field, have[first].start, have[last].end, inserted))
}

/// The strict floor: the core alone covers half the text by exact and near
/// pairs, with two exact words in a text of three or more, and at most one
/// inserted token is left over at an end: more is a different text that
/// shares the rest.
fn accepted(core: &Core, want: &[Token], needed: usize) -> bool {
    core.aligned >= needed
        && !(want.len() >= 3 && core.exact < 2)
        && core.leading <= 1
        && core.trailing <= 1
}

/// A read of the field with where it sits in the whole field.
pub struct View<'a> {
    pub text: &'a str,
    /// UTF-16 units of the field before `text` (a window around the caret
    /// starts after the beginning).
    pub offset: usize,
    /// UTF-16 units in the whole field.
    pub total: usize,
    pub windowed: bool,
}

/// Where the paste went, read before it: the caret's offset, the paste's
/// length, and how long the field then was, all in UTF-16 units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Anchor {
    pub start: usize,
    pub len: usize,
    /// The field's length with the paste in it; set from the first read that
    /// finds the paste where it was put, and the reference for how far the
    /// region has moved since.
    pub total: usize,
}

/// Whether a read shows the paste where it was put.
#[derive(Debug, Eq, PartialEq)]
pub enum Landing {
    /// The paste, whole, at the anchor.
    Landed,
    /// The paste is in the field, but not at the anchor: the offset is
    /// not in this app's units, or the paste went elsewhere.
    Elsewhere,
    /// Not there (yet).
    Absent,
}

/// The two places the paste's region can be: edits before it shift it, edits
/// inside it stretch it. `(from, to)` in UTF-16 units of the whole field.
fn regions(anchor: &Anchor, total: usize) -> [(i64, i64); 2] {
    let delta = total as i64 - anchor.total as i64;
    let (start, end) = (anchor.start as i64, (anchor.start + anchor.len) as i64);
    [
        (start, (end + delta).max(start)),
        ((start + delta).max(0), (end + delta).max(0)),
    ]
}

/// The field tokens inside `region`, if a window holds all of it.
fn tokens_in<'a>(
    have: &'a [Token],
    view: &View,
    text_u16: usize,
    region: (i64, i64),
) -> Option<&'a [Token]> {
    let offset = view.offset as i64;
    if view.windowed && (region.0 < offset || region.1 > offset + text_u16 as i64) {
        return None;
    }
    let inside = |t: &Token| {
        (t.u16_end as i64 + offset) > region.0 && (t.u16_start as i64 + offset) < region.1
    };
    let from = have.iter().position(inside)?;
    let to = have.iter().rposition(inside)?;
    Some(&have[from..=to])
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// Whether `view` shows `inserted` where `anchor` says it went.
pub fn landing(view: &View, inserted: &str, anchor: &Anchor) -> Landing {
    let (text, inserted) = (strip(view.text), strip(inserted));
    let (want, have) = (tokens(inserted.trim()), tokens(&text));
    if !contains_run(&have, &want) {
        return Landing::Absent;
    }
    let view = View {
        text: &text,
        ..*view
    };
    let u16s = utf16_len(&text);
    let landed = regions(
        &Anchor {
            total: view.total,
            ..*anchor
        },
        view.total,
    )
    .into_iter()
    .filter_map(|region| tokens_in(&have, &view, u16s, region))
    .any(|region| contains_run(region, &want));
    if landed {
        Landing::Landed
    } else {
        Landing::Elsewhere
    }
}

/// What sits at the anchor now, once the paste has landed there: the field's
/// region, not the whole field. An empty region is the paste deleted (nothing
/// to learn from); text there that is not the paste is what it was changed to.
pub fn align_at(view: &View, inserted: &str, anchor: &Anchor) -> Alignment {
    let (text, inserted) = (strip(view.text), strip(inserted));
    let inserted = inserted.trim();
    let (want, have) = (tokens(inserted), tokens(&text));
    if want.is_empty() {
        return Alignment::NotFound;
    }
    let u16s = utf16_len(&text);
    let view = View {
        text: &text,
        ..*view
    };
    let mut best: Option<Judged> = None;
    for (n, region) in regions(anchor, view.total).into_iter().enumerate() {
        let Some(found) = tokens_in(&have, &view, u16s, region) else {
            continue;
        };
        // The first region is an edit inside the paste, the second one before.
        let Some(judged) = judge(&text, found, &want, inserted, n == 0) else {
            continue;
        };
        // Ties go to the smaller region, then to the first.
        if best.as_ref().is_none_or(|held| judged.beats(held)) {
            best = Some(judged);
        }
    }
    best.map_or(Alignment::NotFound, |judged| judged.alignment)
}

/// One region against the paste.
struct Judged {
    /// How much of the paste the region holds.
    rank: usize,
    tokens: usize,
    alignment: Alignment,
}

impl Judged {
    fn beats(&self, other: &Judged) -> bool {
        (self.rank, std::cmp::Reverse(self.tokens)) > (other.rank, std::cmp::Reverse(other.tokens))
    }
}

/// What a region says of the paste. `None` for an empty region, or one that
/// shows nothing of the paste and is not the paste's own place.
fn judge(
    text: &str,
    region: &[Token],
    want: &[Token],
    inserted: &str,
    inside: bool,
) -> Option<Judged> {
    if region.is_empty() {
        return None;
    }
    let judged = |rank, tokens: &[Token]| Judged {
        rank,
        tokens: tokens.len(),
        alignment: Alignment::Edited(capped(
            text,
            tokens[0].start,
            tokens[tokens.len() - 1].end,
            inserted,
        )),
    };
    if contains_run(region, want) {
        return Some(Judged {
            rank: usize::MAX,
            tokens: region.len(),
            alignment: Alignment::Unchanged,
        });
    }
    let core = (want.len() > 1)
        .then(|| fill(want, region).and_then(|m| traceback(&m, want, region)))
        .flatten();
    let Some(core) = core else {
        // Position is evidence: a lone word, or text with nothing in common,
        // now sitting where the paste went is what it became. Only for an
        // edit inside the paste, and only if the region did not grow far
        // past it.
        return (inside && region.len() <= want.len() + 2).then(|| judged(0, region));
    };
    if inside {
        // The region is the paste, however far its edits stretched it.
        return Some(judged(core.aligned, region));
    }
    // Before it, the region also holds text the shift brought in: keep the
    // core and what the paste had left over at an end.
    let first = if core.leading > 0 { 0 } else { core.first };
    let last = if core.trailing > 0 {
        region.len() - 1
    } else {
        core.last
    };
    Some(judged(core.aligned, &region[first..=last]))
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

    // ---- anchored ----

    fn view(text: &str) -> View<'_> {
        View {
            text,
            offset: 0,
            total: utf16_len(text),
            windowed: false,
        }
    }

    /// The anchor of `pasted` in the field as it first stood, `base`.
    fn anchor_of(base: &str, pasted: &str) -> Anchor {
        let at = base.rfind(pasted).expect("the paste is in the base field");
        Anchor {
            start: utf16_len(&base[..at]),
            len: utf16_len(pasted),
            total: utf16_len(base),
        }
    }

    /// What the field `now` says of `pasted`, which first stood in `base`.
    fn at(base: &str, pasted: &str, now: &str) -> Alignment {
        let anchor = anchor_of(base, pasted);
        assert_eq!(
            landing(&view(base), pasted, &anchor),
            Landing::Landed,
            "{base:?}"
        );
        align_at(&view(now), pasted, &anchor)
    }

    fn edited_at(base: &str, pasted: &str, now: &str) -> String {
        match at(base, pasted, now) {
            Alignment::Edited(span) => span,
            other => panic!("{base:?} -> {now:?}: {other:?}"),
        }
    }

    #[test]
    fn short_pastes_are_judged_by_what_now_sits_at_the_anchor() {
        assert_eq!(
            edited_at("I met Hanks yesterday", "Hanks", "I met Hex yesterday"),
            "Hex"
        );
        assert_eq!(
            edited_at("Is it ready? Yes", "Yes", "Is it ready? No"),
            "No"
        );
        assert_eq!(
            edited_at(
                "see Hanks Smith today",
                "Hanks Smith",
                "see Hex Jones today"
            ),
            "Hex Jones"
        );
        assert_eq!(
            edited_at("go home now", "go home", "go back now"),
            "go back"
        );
        assert_eq!(
            edited_at("go home now", "go home", "stay put now"),
            "stay put"
        );
    }

    #[test]
    fn a_trailing_two_word_replacement_keeps_both_words() {
        assert_eq!(
            edited_at(
                "send it to Kip",
                "send it to Kip",
                "send it to Kit and Sarah"
            ),
            "send it to Kit and Sarah"
        );
        assert_eq!(
            edited_at(
                "send it to Kip",
                "send it to Kip",
                "send it to Hex and Sarah"
            ),
            "send it to Hex and Sarah"
        );
    }

    #[test]
    fn the_paste_deleted_whole_is_nothing_to_learn_from() {
        assert_eq!(
            at("hello Hanks bye", "Hanks", "hello bye"),
            Alignment::NotFound
        );
        assert_eq!(at("Hanks", "Hanks", ""), Alignment::NotFound);
    }

    #[test]
    fn the_same_text_elsewhere_in_the_document_is_not_the_paste() {
        let base = "send it to Kip. Later: send it to Kip";
        // The first copy is intact; the pasted one (at the end) was edited.
        assert_eq!(
            edited_at(
                base,
                "send it to Kip",
                "send it to Kip. Later: send it to Kit"
            ),
            "send it to Kit"
        );
        // Intact at the anchor stays unchanged, whatever the rest says.
        assert_eq!(
            at(
                base,
                "send it to Kip",
                "send it to Hex. Later: send it to Kip"
            ),
            Alignment::Unchanged
        );
    }

    #[test]
    fn edits_before_the_anchor_shift_the_region() {
        let base = "A. send it to Kip";
        assert_eq!(
            at(base, "send it to Kip", "Hello A. send it to Kip"),
            Alignment::Unchanged
        );
        assert_eq!(
            edited_at(base, "send it to Kip", "Hello there A. send it to Kit"),
            "send it to Kit"
        );
        // Text removed before it.
        let base = "Dear Bob, send it to Kip";
        assert_eq!(
            at(base, "send it to Kip", "Bob, send it to Kip"),
            Alignment::Unchanged
        );
        assert_eq!(
            edited_at(base, "send it to Kip", "Bob, send it to Kit"),
            "send it to Kit"
        );
    }

    #[test]
    fn an_anchor_that_is_not_where_the_paste_is_does_not_land() {
        // Offsets in other units (or a paste that went elsewhere) never validate.
        let anchor = Anchor {
            start: 0,
            len: 14,
            total: 40,
        };
        let field = "some other words first then send it to Kip";
        assert_eq!(
            landing(&view(field), "send it to Kip", &anchor),
            Landing::Elsewhere
        );
        assert_eq!(
            landing(&view("nothing like it"), "send it to Kip", &anchor),
            Landing::Absent
        );
    }

    #[test]
    fn offsets_count_utf16_units() {
        let base = "caf\u{e9} \u{1F600} send it to Kip";
        assert_eq!(
            edited_at(base, "send it to Kip", "caf\u{e9} \u{1F600} send it to Kit"),
            "send it to Kit"
        );
    }

    #[test]
    fn a_window_that_misses_the_region_judges_nothing() {
        let anchor = Anchor {
            start: 9_000,
            len: 14,
            total: 10_000,
        };
        let view = View {
            text: "words in a window far from it",
            offset: 100,
            total: 10_000,
            windowed: true,
        };
        assert_eq!(
            align_at(&view, "send it to Kip", &anchor),
            Alignment::NotFound
        );
    }

    const CHAT: [&str; 36] = [
        "thanks for the update I will review it tomorrow",
        "can you send me the report by Friday",
        "I think we should move the meeting to Monday",
        "let me know if you have any questions",
        "sounds good to me see you then",
        "I will be there in ten minutes",
        "please send it to Kip tomorrow",
        "can you send it to Bob today",
        "we need to talk about the budget for next quarter",
        "thanks for your help with this",
        "I can't make it to the meeting today",
        "let me check and get back to you",
        "the report is attached to this email",
        "hope you are doing well",
        "I will send it over tomorrow morning",
        "can we talk about this tomorrow",
        "looking forward to seeing you there",
        "please let me know what you think",
        "I think it is fine to send it to him",
        "what time works for you tomorrow",
        "thanks I will look at it today",
        "I will send it to you tomorrow",
        "did you send it to Kip yet",
        "I went to the store and bought some milk",
        "can you review the pull request when you get a chance",
        "I will review it tomorrow if that works",
        "that works for me thanks",
        "we should send it to the team",
        "let me know when you are free to talk",
        "I will get back to you tomorrow",
        "please review the document and send it back",
        "the meeting is at noon tomorrow",
        "do you want to go to lunch today",
        "he said he would send it to Kip",
        "I will talk to him about it tomorrow",
        "it is going to be a good day",
    ];

    /// Unrelated chat sentences, every ordered pair: one in the field, the
    /// other the paste. The fallback's false-edit count (the strict rules,
    /// with no anchor) must not rise past what it was; anchored it is none.
    #[test]
    fn the_unrelated_pair_corpus() {
        let (mut pairs, mut strict) = (0, 0);
        for field in CHAT {
            for pasted in CHAT {
                if field == pasted {
                    continue;
                }
                pairs += 1;
                strict += usize::from(matches!(align(field, pasted, false), Alignment::Edited(_)));
                // Anchored: the paste landed at the end of the field, beside
                // the unrelated text, and that text is never part of an edit.
                let base = format!("{field} {pasted}");
                let anchor = anchor_of(&base, pasted);
                assert_eq!(landing(&view(&base), pasted, &anchor), Landing::Landed);
                assert_eq!(
                    align_at(&view(&base), pasted, &anchor),
                    Alignment::Unchanged
                );
                let words: Vec<&str> = pasted.split(' ').collect();
                let edited = format!("{field} {} Hex", words[..words.len() - 1].join(" "));
                match align_at(&view(&edited), pasted, &anchor) {
                    Alignment::Edited(span) => assert!(
                        !span.contains(field) && span.ends_with("Hex"),
                        "{field:?} leaked into {span:?}"
                    ),
                    other => panic!("{field:?} / {pasted:?}: {other:?}"),
                }
                // Never validated: a field that does not hold the paste at
                // all (the unrelated sentence alone) is not a landing.
                assert_ne!(landing(&view(field), pasted, &anchor), Landing::Landed);
            }
        }
        assert_eq!(pairs, 1260);
        println!("strict false edits: {strict} of {pairs}");
        assert!(strict <= STRICT_FALSE_EDITS, "{strict} of {pairs}");
    }

    const STRICT_FALSE_EDITS: usize = 14;
}
