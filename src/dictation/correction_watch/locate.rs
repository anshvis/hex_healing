//! Where a pasted dictation sits in the field now. Pure: the watcher reads
//! the field, this decides whether the text is still there, and if not,
//! which span of the field replaced it.
//!
//! The invariant is to fail closed: only text inside the paste's own region
//! in the current field is ever reported (what now occupies the place the
//! pasted text occupied, plus a token that touches its edge), never text from
//! before or after it. When unsure the answer is `NotFound` or `Unchanged`.
//!
//! Two ways to look, by what is known about where the paste went:
//!
//! - **Anchored** ([`landing`], [`follow`], [`judge`]): the caret was read
//!   before the paste, so the paste's offset in the field is known. Once a
//!   read shows the paste whole there ([`landing`] gives its [`Region`]), the
//!   region is followed through every later read by diffing the field against
//!   the read before ([`follow`]): text typed before it shifts it, typing
//!   inside it stretches it, typing after it leaves it alone, and a change
//!   that crosses its edge loses it. Only the field's text in that region is
//!   judged ([`judge`]), so text anywhere else can never match, and for a
//!   paste of one or two words position is evidence enough to call whatever
//!   now sits there what the paste became. Longer pastes need the strict
//!   floor below, inside the region. An empty region (the paste deleted) is
//!   `NotFound`.
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
//! Neither reports an edit until the paste has been seen whole (the caller
//! checks). Tokens split on whitespace and on U+FFFC (attachments keep their
//! place in the text, so offsets match the app's) and fold case and
//! punctuation, so text without spaces (CJK) is one token and effectively
//! unsupported.

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
        match (is_space(c), open) {
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

/// An attachment's placeholder in the text.
const ATTACHMENT: char = '\u{FFFC}';

fn is_space(c: char) -> bool {
    c.is_whitespace() || c == ATTACHMENT
}

/// Whether `text` holds nothing a person typed.
pub fn is_blank(text: &str) -> bool {
    text.chars().all(is_space)
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
    let inserted = inserted.trim();
    let (want, have) = (tokens(inserted), tokens(field));
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
    if core.leading > 0 && first > 0 && !ends_sentence(field, &have[first - 1], Some(&have[first]))
    {
        first -= 1;
    }
    if core.trailing > 0
        && last + 1 < have.len()
        && !ends_sentence(field, &have[last], Some(&have[last + 1]))
    {
        last += 1;
    }
    Alignment::Edited(capped(field, have[first].start, have[last].end, inserted))
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

/// Where the paste went, read before it: the caret's offset and the paste's
/// length, in UTF-16 units (what Accessibility counts in).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Anchor {
    pub start: usize,
    pub len: usize,
}

/// The paste's place in the field, `[from, to)` in UTF-16 units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Region {
    pub from: usize,
    pub to: usize,
}

/// Whether a read shows the paste where it was put.
#[derive(Debug, Eq, PartialEq)]
pub enum Landing {
    /// The paste, whole, at the anchor: its region.
    Landed(Region),
    /// The paste is in the field, but not at the anchor: the offset is not in
    /// this app's units, or the read came before the app applied the paste.
    Elsewhere,
    /// Not there (yet).
    Absent,
}

/// The tokens that overlap `region`.
fn tokens_in(have: &[Token], region: Region) -> &[Token] {
    let overlaps = |t: &Token| t.u16_end > region.from && t.u16_start < region.to;
    match (
        have.iter().position(overlaps),
        have.iter().rposition(overlaps),
    ) {
        (Some(from), Some(to)) => &have[from..=to],
        _ => &[],
    }
}

fn same_words(a: &[Token], b: &[Token]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.fold == b.fold)
}

/// Whether `text` shows `inserted` where `anchor` says it went.
pub fn landing(text: &str, inserted: &str, anchor: &Anchor) -> Landing {
    let (have, want) = (tokens(text), tokens(inserted.trim()));
    let region = Region {
        from: anchor.start,
        to: anchor.start + anchor.len,
    };
    if want.is_empty() {
        Landing::Absent
    } else if same_words(tokens_in(&have, region), &want) {
        Landing::Landed(region)
    } else if contains_run(&have, &want) {
        Landing::Elsewhere
    } else {
        Landing::Absent
    }
}

/// Where `region` of `before` is in `after`, from the one change between
/// them (their common prefix and suffix bound it): text inserted or removed
/// before the region moves it, a change inside it stretches or shrinks it,
/// one after it leaves it, and an insertion at its start counts as before it
/// and one at its end as after. `None` when the change crosses an edge, or
/// when the same change could as well have been made on either side of one
/// (text typed before the paste that begins with what the paste begins with
/// is also an insertion a few characters in): the text that went with the
/// paste is then not known.
pub fn follow(before: &str, after: &str, region: Region) -> Option<Region> {
    if before == after {
        return Some(region);
    }
    let (b, a): (Vec<u16>, Vec<u16>) = (
        before.encode_utf16().collect(),
        after.encode_utf16().collect(),
    );
    let shorter = b.len().min(a.len());
    let prefix = b.iter().zip(&a).take_while(|(x, y)| x == y).count();
    let suffix_alone = b
        .iter()
        .rev()
        .zip(a.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let suffix = suffix_alone.min(shorter - prefix);
    // The change could have been made this many units earlier, as well.
    let slide = (prefix + suffix_alone).saturating_sub(shorter);
    let (from, to) = (region.from, region.to);
    let shift = |at: usize| (at + a.len()).saturating_sub(b.len());
    let moved = Region {
        from: shift(from),
        to: shift(to),
    };
    let place = |changed_from: usize, changed_to: usize| {
        if changed_from == changed_to {
            // Pure insertion.
            return Some(match changed_from {
                at if at <= from => moved,
                at if at >= to => region,
                _ => Region { from, to: moved.to },
            });
        }
        if changed_to <= from {
            Some(moved)
        } else if changed_from >= to {
            Some(region)
        } else if from <= changed_from && changed_to <= to {
            Some(Region { from, to: moved.to })
        } else {
            None
        }
    };
    let (changed_from, changed_to) = (prefix, b.len() - suffix);
    let placed = place(changed_from, changed_to);
    (0..=slide)
        .all(|back| place(changed_from - back, changed_to - back) == placed)
        .then_some(placed)
        .flatten()
}

/// What `text` says of the paste now, from its region alone. An empty region
/// is the paste deleted: nothing to learn from.
pub fn judge(text: &str, inserted: &str, region: Region) -> Alignment {
    let inserted = inserted.trim();
    let (have, want) = (tokens(text), tokens(inserted));
    if want.is_empty() || region.from >= region.to {
        return Alignment::NotFound;
    }
    let found = tokens_in(&have, region);
    if found.is_empty() {
        return Alignment::NotFound;
    }
    if contains_run(found, &want) {
        return Alignment::Unchanged;
    }
    let edited = || {
        Alignment::Edited(capped(
            text,
            found[0].start,
            found[found.len() - 1].end,
            inserted,
        ))
    };
    if want.len() <= 2 {
        // Position is evidence: one or two words now sitting where the paste
        // went are what it became, if they did not grow far past it.
        return if found.len() <= want.len() + 4 {
            edited()
        } else {
            Alignment::NotFound
        };
    }
    // A longer paste must still resemble itself: half its words (exact or
    // near) in order, two of them exact, inside the region. Where the edited
    // words sit does not matter here: the region is already the paste's.
    if want.len().saturating_mul(found.len()) > MAX_CELLS {
        return Alignment::NotFound;
    }
    let same = |a: &Token, b: &Token| score(a, b) > 0;
    let exact = |a: &Token, b: &Token| a.fold == b.fold;
    if common_run(&want, found, same) >= want.len().div_ceil(2)
        && common_run(&want, found, exact) >= 2
    {
        edited()
    } else {
        Alignment::NotFound
    }
}

/// The most words of `want` that follow each other, in order, among `have`
/// (the longest common subsequence under `same`).
fn common_run(want: &[Token], have: &[Token], same: impl Fn(&Token, &Token) -> bool) -> usize {
    let mut row = vec![0_usize; have.len() + 1];
    for w in want {
        let mut diagonal = 0;
        for (j, h) in have.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if same(w, h) {
                diagonal + 1
            } else {
                above.max(row[j])
            };
            diagonal = above;
        }
    }
    row[have.len()]
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

    fn utf16_len(text: &str) -> usize {
        text.encode_utf16().count()
    }

    /// The anchor of `pasted` (its last copy) in the field as it first stood.
    fn anchor_of(base: &str, pasted: &str) -> Anchor {
        let at = base.rfind(pasted).expect("the paste is in the base field");
        Anchor {
            start: utf16_len(&base[..at]),
            len: utf16_len(pasted),
        }
    }

    /// What the field `now` says of `pasted`, which first stood in `base`,
    /// the way a watch reads it: land it, follow its region, judge it.
    fn at(base: &str, pasted: &str, now: &str) -> Alignment {
        let anchor = anchor_of(base, pasted);
        let Landing::Landed(region) = landing(base, pasted, &anchor) else {
            panic!("{pasted:?} did not land in {base:?}");
        };
        match follow(base, now, region) {
            Some(region) => judge(now, pasted, region),
            None => Alignment::NotFound,
        }
    }

    /// Like [`at`] over several reads, each followed from the last.
    fn edited_in_steps(base: &str, pasted: &str, reads: &[&str]) -> String {
        let anchor = anchor_of(base, pasted);
        let Landing::Landed(mut region) = landing(base, pasted, &anchor) else {
            panic!("{pasted:?} did not land in {base:?}");
        };
        let mut before = base;
        for read in reads {
            region = follow(before, read, region).expect("followed");
            before = read;
        }
        match judge(before, pasted, region) {
            Alignment::Edited(span) => span,
            other => panic!("{reads:?}: {other:?}"),
        }
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
    fn text_typed_after_an_edit_is_not_the_edit() {
        let base = "ok. send it to Kip";
        // Typed after, in a later read: the region does not grow to hold it.
        let edited = "ok. send it to Kit";
        let typed = "ok. send it to Kit please also tell Bob";
        let anchor = anchor_of(base, "send it to Kip");
        let Landing::Landed(region) = landing(base, "send it to Kip", &anchor) else {
            panic!()
        };
        let region = follow(base, edited, region).unwrap();
        let region = follow(edited, typed, region).unwrap();
        assert_eq!(
            judge(typed, "send it to Kip", region),
            Alignment::Edited("send it to Kit".into())
        );
    }

    #[test]
    fn the_paste_deleted_whole_is_nothing_to_learn_from() {
        assert_eq!(
            at("hello Hanks bye", "Hanks", "hello bye"),
            Alignment::NotFound
        );
        assert_eq!(
            at("report to HR. Hanks", "Hanks", "report to HR."),
            Alignment::NotFound
        );
        assert_eq!(at("Hanks", "Hanks", ""), Alignment::NotFound);
        // The paste and the text before it deleted: the region is lost.
        assert_eq!(
            at("report to HR. ok Hanks", "Hanks", "bye"),
            Alignment::NotFound
        );
        // An empty region selects no token, even one it touches.
        let text = "abcxyz";
        let region = Region { from: 3, to: 3 };
        assert_eq!(judge(text, "Hanks", region), Alignment::NotFound);
    }

    #[test]
    fn the_same_text_elsewhere_in_the_document_is_not_the_paste() {
        let base = "send it to Kip. Later: send it to Kip";
        assert_eq!(
            edited_at(
                base,
                "send it to Kip",
                "send it to Kip. Later: send it to Kit"
            ),
            "send it to Kit"
        );
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
        // Two changes in one read (before it and inside it) cross no edge a
        // diff can place: the region is lost and nothing is reported.
        assert_eq!(
            at(base, "send it to Kip", "Hello there A. send it to Kit"),
            Alignment::NotFound
        );
        // One read at a time, both are followed.
        assert_eq!(
            edited_in_steps(
                base,
                "send it to Kip",
                &[
                    "Hello there A. send it to Kip",
                    "Hello there A. send it to Kit"
                ]
            ),
            "send it to Kit"
        );
        let base = "Dear Bob, send it to Kip";
        assert_eq!(
            at(base, "send it to Kip", "Bob, send it to Kip"),
            Alignment::Unchanged
        );
        assert_eq!(
            edited_in_steps(
                base,
                "send it to Kip",
                &["Bob, send it to Kip", "Bob, send it to Kit"]
            ),
            "send it to Kit"
        );
    }

    #[test]
    fn a_paste_wholly_replaced_by_unrelated_words_is_not_an_edit() {
        // Two words or fewer: position is evidence. Longer: it must resemble.
        assert_eq!(
            edited_at("ok. Hanks Smith", "Hanks Smith", "ok. it rains"),
            "it rains"
        );
        assert_eq!(
            at(
                "ok. send it to Kip tomorrow",
                "send it to Kip tomorrow",
                "ok. the weather is nice"
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn an_anchor_that_is_not_where_the_paste_is_does_not_land() {
        let anchor = Anchor { start: 0, len: 14 };
        let field = "some other words first then send it to Kip";
        assert_eq!(
            landing(field, "send it to Kip", &anchor),
            Landing::Elsewhere
        );
        assert_eq!(
            landing("nothing like it", "send it to Kip", &anchor),
            Landing::Absent
        );
    }

    #[test]
    fn offsets_count_utf16_units_and_attachments_keep_their_place() {
        let base = "caf\u{e9} \u{1F600} send it to Kip";
        assert_eq!(
            edited_at(base, "send it to Kip", "caf\u{e9} \u{1F600} send it to Kit"),
            "send it to Kit"
        );
        let base = "see \u{FFFC} attached. send it to Kip";
        assert_eq!(
            edited_at(
                base,
                "send it to Kip",
                "see \u{FFFC} attached. send it to Kit"
            ),
            "send it to Kit"
        );
        let base = "\u{FFFC}\u{FFFC}send it to Kip";
        assert_eq!(
            edited_at(base, "send it to Kip", "\u{FFFC}\u{FFFC}send it to Kit"),
            "send it to Kit"
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
    /// other the paste. The strict rules' false-edit count (no anchor) must
    /// not rise past what it was; anchored it is none, and an edit at the
    /// anchor never carries the neighbouring text.
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
                let base = format!("{field} {pasted}");
                let anchor = anchor_of(&base, pasted);
                let Landing::Landed(region) = landing(&base, pasted, &anchor) else {
                    panic!("{field:?} / {pasted:?} did not land");
                };
                assert_eq!(judge(&base, pasted, region), Alignment::Unchanged);
                let words: Vec<&str> = pasted.split(' ').collect();
                let edited = format!("{field} {} Hex", words[..words.len() - 1].join(" "));
                let region = follow(&base, &edited, region).unwrap();
                match judge(&edited, pasted, region) {
                    Alignment::Edited(span) => assert!(
                        !span.contains(field) && span.ends_with("Hex"),
                        "{field:?} leaked into {span:?}"
                    ),
                    other => panic!("{field:?} / {pasted:?}: {other:?}"),
                }
                // A field without the paste is no landing at all.
                assert_ne!(landing(field, pasted, &anchor), Landing::Landed(region));
            }
        }
        assert_eq!(pairs, 1260);
        println!("strict false edits: {strict} of {pairs}");
        assert!(strict <= STRICT_FALSE_EDITS, "{strict} of {pairs}");
    }

    const STRICT_FALSE_EDITS: usize = 14;

    #[test]
    fn a_huge_paste_is_not_aligned() {
        let field = "word ".repeat(3_000);
        let pasted = "other ".repeat(40_000);
        assert_eq!(align(&field, &pasted, false), Alignment::NotFound);
    }
}
