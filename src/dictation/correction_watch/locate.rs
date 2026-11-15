//! Where a pasted dictation sits in the field now, and what became of it.
//! Pure: the watcher reads the field, a [`Tracker`] says what the paste's
//! own place in it holds.
//!
//! The invariant is to fail closed: an edit is stored only from a region
//! Hex is following, only text inside it (what now occupies the place the
//! pasted text occupied, plus a token that touches its edge), never text from
//! before or after it. When unsure the answer is `NotFound` or `Unchanged`,
//! and a region once lost stays lost.
//!
//! How a region is found:
//!
//! - **Anchored**: the caret was read before the paste, so the paste's offset
//!   is known. The first read that shows the paste's words as a run starting
//!   within [`DRIFT`] units of it lands there. The region is snapped to the
//!   field's own tokens (first matched token's start to the last one's end),
//!   so an app that stores fewer units than were inserted, or an anchor a
//!   little off, cannot leave it hanging past the paste. The paste found
//!   elsewhere only (another copy; the app has not applied it yet) is not a
//!   landing, and waits.
//! - **No caret**: the field was read before the paste. When it held the
//!   paste's text nowhere, the first read with it exactly once adopts that
//!   occurrence. Held before, or twice now, or unknown: the watch can only
//!   report `Unchanged` or `NotFound`.
//!
//! How a region is followed ([`follow`]): every read is diffed against the
//! last. Text inserted or removed before the region moves it, a change inside
//! it stretches or shrinks it, one after it leaves it, and a change that
//! crosses an edge, or could as well sit on either side of one, loses it. A
//! change that shortens the region at an edge (backspace, or typing over a
//! selection) leaves that edge open: typing right at it counts as inside, up
//! to the words deleted plus one, and a longer change at an edge in one read
//! is lost. A blank read, or one too large to follow (a window around the
//! caret), retires the region. Only the field's text in the region is judged
//! ([`judge`]): for a paste of one or two words position is evidence enough to
//! call whatever sits there what it became; a longer paste must still
//! resemble itself.
//!
//! Tokens split on whitespace and on U+FFFC (attachments keep their place in
//! the text, so offsets match the app's) and fold case and punctuation, so
//! text without spaces (CJK) is one token and effectively unsupported.

use crate::accessibility::FieldText;

/// How the inserted text sits in the field.
#[derive(Debug, Eq, PartialEq)]
pub enum Alignment {
    /// The inserted text is still in the field as pasted.
    Unchanged,
    /// The field's span that replaced it, at most [`span_cap`] characters.
    Edited(String),
    NotFound,
}

/// How far, in UTF-16 units, the paste may sit from the caret the app
/// reported (a dropped space or newline, an anchor a unit or two off).
const DRIFT: usize = 8;
/// How far, in UTF-16 units, a change's edge may have shifted by chance:
/// the matching units around it that are the same words by coincidence.
const SNAP: usize = 32;
/// A paste of one or two words that grew to more than this many extra words
/// is no longer the paste's place.
const SHORT_PASTE_GROWTH: usize = 4;
/// The most comparisons (inserted tokens x region tokens) worth making.
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

/// An attachment's placeholder in the text.
const ATTACHMENT: char = '\u{FFFC}';

fn is_space(c: char) -> bool {
    c.is_whitespace() || c == ATTACHMENT
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

/// Whether `text` holds nothing a person typed.
pub fn is_blank(text: &str) -> bool {
    text.chars().all(is_space)
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

/// Whether two words are the same word, or a near one: three or more
/// letters, one edit apart ("Kip" and "Kit").
fn alike(a: &Token, b: &Token) -> bool {
    a.fold == b.fold
        || (a.fold.len() >= 3 && b.fold.len() >= 3 && edit_within_one(&a.fold, &b.fold))
}

fn same(a: &Token, b: &Token) -> bool {
    a.fold == b.fold
}

/// Where in `have` the words `want` occur as a run.
fn occurrences(have: &[Token], want: &[Token]) -> Vec<usize> {
    if want.is_empty() {
        return Vec::new();
    }
    have.windows(want.len())
        .enumerate()
        .filter(|(_, run)| run.iter().zip(want).all(|(a, b)| a.fold == b.fold))
        .map(|(at, _)| at)
        .collect()
}

fn contains_run(have: &[Token], want: &[Token]) -> bool {
    !occurrences(have, want).is_empty()
}

fn capped(field: &str, from: usize, to: usize, inserted: &str) -> String {
    field[from..to]
        .chars()
        .take(span_cap(inserted.chars().count()))
        .collect()
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

/// One edge of a region that a change shortened: typing right at it is still
/// the replacement, for `allow` words. For the end edge `mark` is where the
/// typed text starts; for the start edge, where it ends so far.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Edge {
    allow: usize,
    mark: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Open {
    start: Option<Edge>,
    end: Option<Edge>,
}

fn words_in(units: &[u16]) -> usize {
    tokens(&String::from_utf16_lossy(units)).len()
}

/// Where `region` of `before` is in `after`, from the one change between
/// them (their common prefix and suffix bound it). See the module doc.
/// `None` when it is lost: the change crosses an edge, could as well sit on
/// either side of one (text typed before the paste that begins with what the
/// paste begins with is also an insertion a few characters in), or grows an
/// edge by more than the words deleted there plus one.
fn follow(before: &str, after: &str, region: Region, open: Open) -> Option<(Region, Open)> {
    if before == after {
        return Some((region, open));
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
        let removed = changed_to - changed_from;
        let new_len = (a.len() + removed) - b.len();
        if removed == 0 {
            // Pure insertion. Typing at an open edge goes on being inside.
            if let Some(edge) = open.end.filter(|_| changed_from == to) {
                let new_to = to + new_len;
                return Some(if words_in(&a[edge.mark..new_to]) <= edge.allow {
                    let end = Some(edge);
                    (Region { from, to: new_to }, Open { start: None, end })
                } else {
                    (region, Open::default())
                });
            }
            if let Some(edge) = open.start.filter(|e| changed_from == e.mark) {
                let typed_to = changed_from + new_len;
                if from > typed_to || words_in(&a[from..typed_to]) > edge.allow {
                    return None;
                }
                let start = Some(Edge {
                    mark: typed_to,
                    ..edge
                });
                return Some((Region { from, to: moved.to }, Open { start, end: None }));
            }
            return Some((
                match changed_from {
                    at if at <= from => moved,
                    at if at >= to => region,
                    _ => Region { from, to: moved.to },
                },
                Open::default(),
            ));
        }
        if changed_to <= from {
            Some((moved, Open::default()))
        } else if changed_from >= to {
            Some((region, Open::default()))
        } else if from <= changed_from && changed_to <= to {
            let deleted = words_in(&b[changed_from..changed_to]);
            let at_start = changed_from == from;
            let at_end = changed_to == to;
            // At an edge, a change may add the words it deleted and one more.
            if (at_start || at_end)
                && words_in(&a[changed_from..changed_from + new_len]) > deleted + 1
            {
                return None;
            }
            let shortened = new_len < removed;
            let allow = |old: Option<Edge>| old.map_or(0, |e| e.allow).max(deleted + 1);
            let new_to = to + new_len - removed;
            let start = (at_start && shortened).then(|| Edge {
                allow: allow(open.start),
                mark: from + new_len,
            });
            let end = (at_end && shortened).then(|| Edge {
                allow: allow(open.end),
                mark: new_to - new_len,
            });
            Some((Region { from, to: new_to }, Open { start, end }))
        } else {
            None
        }
    };
    let (changed_from, changed_to) = (prefix, b.len() - suffix);
    // Typing at an edge a change left open goes on being the replacement, a
    // space either side of the first key notwithstanding.
    let continues = changed_from == changed_to
        && (open.end.is_some() && changed_from == to
            || open.start.is_some_and(|edge| changed_from == edge.mark));
    if continues {
        return place(changed_from, changed_to);
    }
    // Every way the change could have been made, as `(from, to)` windows of
    // `before`: this many units earlier, as well; and, for a change inside the
    // region that stops a few units short of an edge, one that ran on to it,
    // with the units between matching what follows by chance ("Friday" and
    // "today" both end in "y"). At an edge a change is bounded, so each is
    // read, and if they disagree the region is lost.
    let mut windows: Vec<(usize, usize)> = (0..=slide)
        .map(|back| (changed_from - back, changed_to - back))
        .collect();
    let inside = (from < changed_from && changed_from < to)
        || (changed_from < changed_to && from <= changed_from && changed_to <= to);
    if inside {
        let starts = [changed_from]
            .into_iter()
            .chain((from <= changed_from && changed_from - from <= SNAP).then_some(from));
        for start in starts.collect::<Vec<_>>() {
            let reaches =
                to >= changed_to && to - changed_to <= SNAP && b.len() - to <= suffix_alone;
            let ends = [changed_to].into_iter().chain(reaches.then_some(to));
            windows.extend(ends.map(|end| (start, end)));
        }
    }
    // A window that crosses an edge only over whitespace deletes nothing of
    // the paste's on that side: the same text gone as "word " or " word".
    let blank = |units: &[u16]| {
        units
            .iter()
            .all(|&u| char::from_u32(u32::from(u)).is_some_and(is_space))
    };
    let crosses_only_blanks = |(start, end): (usize, usize)| {
        (start < from || end > to)
            && (start >= from || blank(&b[start..from.min(end)]))
            && (end <= to || blank(&b[to.max(start)..end]))
    };
    let mut placements = Vec::new();
    for window in windows {
        match place(window.0, window.1) {
            None if crosses_only_blanks(window) => {}
            placed => placements.push(placed),
        }
    }
    let first = placements.first().cloned().flatten()?;
    placements
        .iter()
        .all(|placed| placed.as_ref().map(|(region, _)| *region) == Some(first.0))
        .then_some(first)
}

/// What `text` says of the paste now, from its region alone. An empty region
/// is the paste deleted: nothing to learn from.
fn judge(text: &str, have: &[Token], want: &[Token], inserted: &str, region: Region) -> Alignment {
    if want.is_empty() || region.from >= region.to {
        return Alignment::NotFound;
    }
    let found = tokens_in(have, region);
    if found.is_empty() {
        return Alignment::NotFound;
    }
    if contains_run(found, want) {
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
        return if found.len() <= want.len() + SHORT_PASTE_GROWTH {
            edited()
        } else {
            Alignment::NotFound
        };
    }
    if resembles(want, found) {
        edited()
    } else {
        Alignment::NotFound
    }
}

/// The floor for a longer paste: half its words (exact or near) in order, two
/// of them exact. Where the edited words sit does not matter: the region is
/// already the paste's.
fn resembles(want: &[Token], found: &[Token]) -> bool {
    want.len().saturating_mul(found.len()) <= MAX_CELLS
        && in_order(want, found, alike) >= want.len().div_ceil(2)
        && in_order(want, found, same) >= 2
}

/// The most words of `want` that follow each other, in order, among `have`
/// (the longest common subsequence under `matches`).
fn in_order(want: &[Token], have: &[Token], matches: impl Fn(&Token, &Token) -> bool) -> usize {
    let mut row = vec![0_usize; have.len() + 1];
    for w in want {
        let mut diagonal = 0;
        for (j, h) in have.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if matches(w, h) {
                diagonal + 1
            } else {
                above.max(row[j])
            };
            diagonal = above;
        }
    }
    row[have.len()]
}

enum Tracking {
    /// The paste has not been seen where it will be followed. `known` is how
    /// many times the field held its text before the paste (no caret only).
    Waiting {
        anchor: Option<Anchor>,
        known: Option<usize>,
    },
    /// Followed: `text` is the read the region was last placed in.
    Following {
        text: String,
        region: Region,
        open: Open,
    },
    /// Never an edit again: only `Unchanged` or `NotFound`.
    Lost,
}

/// One pasted text's watch over the field: finds the paste's region, follows
/// it, and says what the region holds.
pub struct Tracker {
    inserted: String,
    want: Vec<Token>,
    state: Tracking,
}

impl Tracker {
    /// `before` is the field's text before the paste, read when there was no
    /// caret to anchor on.
    pub fn new(inserted: String, anchor: Option<Anchor>, before: Option<&str>) -> Self {
        let want = tokens(inserted.trim());
        let known = before.map(|before| occurrences(&tokens(before), &want).len());
        Self {
            inserted,
            want,
            state: Tracking::Waiting { anchor, known },
        }
    }

    #[cfg(test)]
    pub fn region(&self) -> Option<Region> {
        match &self.state {
            Tracking::Following { region, .. } => Some(*region),
            _ => None,
        }
    }

    /// What the field says of the paste now. A blank field retires the
    /// region; so does a window, which cannot be followed.
    pub fn read(&mut self, field: &FieldText) -> Alignment {
        let FieldText { text, windowed } = field;
        let have = tokens(text);
        let unchanged_only = |have: &[Token], want: &[Token]| {
            if contains_run(have, want) {
                Alignment::Unchanged
            } else {
                Alignment::NotFound
            }
        };
        if self.want.is_empty() {
            return Alignment::NotFound;
        }
        if is_blank(text) {
            self.state = Tracking::Lost;
            return Alignment::NotFound;
        }
        match &self.state {
            Tracking::Lost => unchanged_only(&have, &self.want),
            Tracking::Waiting { .. } if *windowed => unchanged_only(&have, &self.want),
            Tracking::Following { .. } if *windowed => {
                self.state = Tracking::Lost;
                unchanged_only(&have, &self.want)
            }
            Tracking::Waiting { anchor, known } => {
                let found = occurrences(&have, &self.want);
                let landing = match (anchor, known) {
                    (Some(anchor), _) => {
                        let near: Vec<_> = found
                            .iter()
                            .filter(|&&at| have[at].u16_start.abs_diff(anchor.start) <= DRIFT)
                            .collect();
                        (near.len() == 1).then(|| *near[0])
                    }
                    (None, Some(0)) => (found.len() == 1).then(|| found[0]),
                    (None, _) => None,
                };
                match landing {
                    Some(at) => {
                        let region = Region {
                            from: have[at].u16_start,
                            to: have[at + self.want.len() - 1].u16_end,
                        };
                        self.state = Tracking::Following {
                            text: text.clone(),
                            region,
                            open: Open::default(),
                        };
                        Alignment::Unchanged
                    }
                    // Not there yet, or only a copy that was there before.
                    None if anchor.is_some() || found.is_empty() => {
                        unchanged_only(&have, &self.want)
                    }
                    // No caret: the text twice, or held before, or unknown.
                    None => {
                        self.state = Tracking::Lost;
                        Alignment::Unchanged
                    }
                }
            }
            Tracking::Following {
                text: before,
                region,
                open,
            } => match follow(before, text, *region, *open) {
                None => {
                    self.state = Tracking::Lost;
                    unchanged_only(&have, &self.want)
                }
                Some((region, open)) => {
                    // The region starts at its first word: whitespace and
                    // attachments a coalesced change brought in before it are
                    // not the paste's, and text typed before it must land
                    // outside.
                    let from = tokens_in(&have, region)
                        .first()
                        .map_or(region.from, |first| region.from.max(first.u16_start));
                    let open = Open {
                        start: open.start.filter(|_| from == region.from),
                        ..open
                    };
                    let region = Region { from, ..region };
                    let alignment = judge(text, &have, &self.want, &self.inserted, region);
                    self.state = Tracking::Following {
                        text: text.clone(),
                        region,
                        open,
                    };
                    alignment
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(text: &str) -> FieldText {
        FieldText {
            text: text.to_string(),
            windowed: false,
        }
    }

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

    /// A tracker anchored on the last copy of `pasted` in `base`, having read
    /// `base`: the paste has landed.
    fn landed(base: &str, pasted: &str) -> Tracker {
        let mut tracker = Tracker::new(pasted.to_string(), Some(anchor_of(base, pasted)), None);
        assert_eq!(tracker.read(&field(base)), Alignment::Unchanged, "{base:?}");
        assert!(
            matches!(tracker.state, Tracking::Following { .. }),
            "{base:?}"
        );
        tracker
    }

    /// What the last of `reads` says of `pasted`, which first stood in `base`.
    fn last(base: &str, pasted: &str, reads: &[&str]) -> Alignment {
        let mut tracker = landed(base, pasted);
        let mut last = Alignment::Unchanged;
        for read in reads {
            last = tracker.read(&field(read));
        }
        last
    }

    fn edited(base: &str, pasted: &str, reads: &[&str]) -> String {
        match last(base, pasted, reads) {
            Alignment::Edited(span) => span,
            other => panic!("{base:?} -> {reads:?}: {other:?}"),
        }
    }

    /// Keys typed one at a time after `prefix`: every prefix of `typed`.
    fn typing(prefix: &str, typed: &str) -> Vec<String> {
        typed
            .char_indices()
            .map(|(at, c)| format!("{prefix}{}", &typed[..at + c.len_utf8()]))
            .collect()
    }

    fn refs(owned: &[String]) -> Vec<&str> {
        owned.iter().map(String::as_str).collect()
    }

    #[test]
    fn short_pastes_are_judged_by_what_now_sits_in_the_region() {
        assert_eq!(
            edited("I met Hanks yesterday", "Hanks", &["I met Hex yesterday"]),
            "Hex"
        );
        assert_eq!(
            edited("Is it ready? Yes", "Yes", &["Is it ready? No"]),
            "No"
        );
        assert_eq!(
            edited(
                "see Hanks Smith today",
                "Hanks Smith",
                &["see Hex Jones today"]
            ),
            "Hex Jones"
        );
        assert_eq!(
            edited("go home now", "go home", &["go back now"]),
            "go back"
        );
        assert_eq!(
            edited("go home now", "go home", &["stay put now"]),
            "stay put"
        );
    }

    #[test]
    fn one_edit_in_a_longer_paste_is_stored_whole() {
        assert_eq!(
            edited(
                "ok. send it to Kip",
                "send it to Kip",
                &["ok. send it to Kit"]
            ),
            "send it to Kit"
        );
        assert_eq!(
            edited(
                "ok. send it to Kip",
                "send it to Kip",
                &["ok. please it to Kip"]
            ),
            "please it to Kip"
        );
    }

    #[test]
    fn a_trailing_replacement_longer_than_the_words_it_replaced_plus_one_is_lost() {
        let base = "send it to Kip";
        // One read: the same as an edit with text appended, so it fails closed.
        assert_eq!(
            last(base, base, &["send it to Kit and Sarah"]),
            Alignment::NotFound
        );
        // Within the allowance (the word and one more) it is stored.
        assert_eq!(
            edited(base, base, &["send it to Kit and"]),
            "send it to Kit and"
        );
        // Key by key past the allowance stops at it; the rest is outside.
        let keys = typing("send it to ", "Kit and Sarah");
        let mut reads = vec!["send it to Ki", "send it to K", "send it to "];
        reads.extend(refs(&keys));
        assert_eq!(edited(base, base, &reads), "send it to Kit and");
    }

    #[test]
    fn backspace_and_retype_at_an_edge_keeps_the_new_word() {
        let base = "ok. send it to Kip";
        let mut reads = vec!["ok. send it to Ki", "ok. send it to K", "ok. send it to "];
        let keys = typing("ok. send it to ", "Kit");
        reads.extend(refs(&keys));
        assert_eq!(edited(base, "send it to Kip", &reads), "send it to Kit");
        // The first word, from the front.
        let base = "ok. Hanks Smith said hi";
        let mut reads = vec![
            "ok. anks Smith said hi",
            "ok. nks Smith said hi",
            "ok. Smith said hi",
        ];
        let keys: Vec<String> = typing("ok. ", "Hex")
            .into_iter()
            .map(|t| format!("{t} Smith said hi"))
            .collect();
        reads.extend(refs(&keys));
        assert_eq!(
            edited(base, "Hanks Smith said hi", &reads),
            "Hex Smith said hi"
        );
        // A lone word.
        let mut reads = vec!["Hank", "Han", "Ha", "H", ""[..0].trim()];
        let _ = &mut reads;
        let base = "ok. Hanks";
        let keys = typing("ok. ", "Hex");
        let mut reads = vec!["ok. Hank", "ok. Han", "ok. Ha", "ok. H", "ok. "];
        reads.extend(refs(&keys));
        assert_eq!(edited(base, "Hanks", &reads), "Hex");
    }

    #[test]
    fn selecting_a_word_and_typing_over_it_is_followed_key_by_key() {
        // "tomorrow" is selected and "Kit and Sarah" typed: the first key
        // replaces it, the rest are typed at the edge it left open.
        let base = "please send it to Kip tomorrow";
        let keys = typing("please send it to Kip ", "Kit and Sarah");
        assert_eq!(
            edited(base, "please send it to Kip tomorrow", &refs(&keys)),
            "please send it to Kip Kit and"
        );
        // Replacing "ok thanks" by typing "yes sure thing".
        let base = "report. ok thanks";
        let keys = typing("report. ", "yes sure thing");
        assert_eq!(edited(base, "ok thanks", &refs(&keys)), "yes sure thing");
        // One word more than the two it replaced plus one is cut at that.
        let keys = typing("report. ", "yes sure thing today");
        assert_eq!(edited(base, "ok thanks", &refs(&keys)), "yes sure thing");
    }

    #[test]
    fn text_typed_after_an_edit_is_not_the_edit() {
        let base = "ok. send it to Kip";
        assert_eq!(
            edited(
                base,
                "send it to Kip",
                &[
                    "ok. send it to Kit",
                    "ok. send it to Kit please also tell Bob"
                ]
            ),
            "send it to Kit"
        );
        // A coalesced edit and appended text is lost, not stretched.
        assert_eq!(
            last(
                base,
                "send it to Kip",
                &["ok. send it to Kit my private unrelated text"]
            ),
            Alignment::NotFound
        );
        // Typed after an untouched paste, key by key.
        let keys = typing("ok. send it to Kip ", "my private words");
        assert_eq!(
            last(base, "send it to Kip", &refs(&keys)),
            Alignment::Unchanged
        );
    }

    #[test]
    fn text_typed_before_a_coalesced_edit_is_lost_not_stretched() {
        let base = "ok. send it to Kip";
        assert_eq!(
            last(
                base,
                "send it to Kip",
                &["ok. my private words send it to Kit"]
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn the_paste_deleted_whole_is_nothing_to_learn_from() {
        assert_eq!(
            last("hello Hanks bye", "Hanks", &["hello bye"]),
            Alignment::NotFound
        );
        assert_eq!(
            last("report to HR. Hanks", "Hanks", &["report to HR."]),
            Alignment::NotFound
        );
        // The paste and the text before it deleted: the region is lost.
        assert_eq!(
            last("report to HR. ok Hanks", "Hanks", &["bye"]),
            Alignment::NotFound
        );
        // An empty region selects no token, even one it touches.
        let text = "abcxyz";
        let region = Region { from: 3, to: 3 };
        assert_eq!(
            judge(text, &tokens(text), &tokens("Hanks"), "Hanks", region),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_blank_field_retires_the_region_for_good() {
        let base = "ok. send it to Kip";
        let mut tracker = landed(base, "send it to Kip");
        assert_eq!(tracker.read(&field("")), Alignment::NotFound);
        // Other text pasted back into the emptied field is never an edit.
        assert_eq!(
            tracker.read(&field("my private unrelated text")),
            Alignment::NotFound
        );
        assert_eq!(
            tracker.read(&field("ok. send it to Kit")),
            Alignment::NotFound,
            "a lost region stays lost"
        );
        assert_eq!(
            tracker.read(&field("ok. send it to Kip")),
            Alignment::Unchanged
        );
    }

    #[test]
    fn the_same_text_elsewhere_in_the_document_is_not_the_paste() {
        let base = "send it to Kip. Later: send it to Kip";
        assert_eq!(
            edited(
                base,
                "send it to Kip",
                &["send it to Kip. Later: send it to Kit"]
            ),
            "send it to Kit"
        );
        assert_eq!(
            last(
                base,
                "send it to Kip",
                &["send it to Hex. Later: send it to Kip"]
            ),
            Alignment::Unchanged
        );
    }

    #[test]
    fn edits_before_the_anchor_shift_the_region() {
        let base = "A. send it to Kip";
        assert_eq!(
            last(base, "send it to Kip", &["Hello A. send it to Kip"]),
            Alignment::Unchanged
        );
        // Two changes in one read cross no edge a diff can place: lost.
        assert_eq!(
            last(base, "send it to Kip", &["Hello there A. send it to Kit"]),
            Alignment::NotFound
        );
        assert_eq!(
            edited(
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
            edited(
                base,
                "send it to Kip",
                &["Bob, send it to Kip", "Bob, send it to Kit"]
            ),
            "send it to Kit"
        );
    }

    #[test]
    fn a_paste_wholly_replaced_by_unrelated_words_is_not_an_edit() {
        assert_eq!(
            edited("ok. Hanks Smith", "Hanks Smith", &["ok. it rains"]),
            "it rains"
        );
        assert_eq!(
            last(
                "ok. send it to Kip tomorrow",
                "send it to Kip tomorrow",
                &["ok. the weather is nice"]
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn the_region_is_snapped_to_the_fields_own_tokens() {
        // The app kept one newline of two: the paste is a unit shorter than
        // inserted, and what is typed after it is not part of it.
        let pasted = "alpha beta\n\ngamma delta";
        let base = "ok. alpha beta\ngamma delta";
        let mut tracker = Tracker::new(
            pasted.to_string(),
            Some(anchor_of("ok. alpha beta\n\ngamma delta", pasted)),
            None,
        );
        assert_eq!(tracker.read(&field(base)), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("ok. alpha beta\ngamma Rune")),
            Alignment::Edited("alpha beta\ngamma Rune".into())
        );
        assert_eq!(
            tracker.read(&field("ok. alpha beta\ngamma Rune typed after text")),
            Alignment::Edited("alpha beta\ngamma Rune".into())
        );
        // An anchor a few units off still lands, on the paste.
        for drift in [-3_i32, -2, -1, 1, 2, 3] {
            let mut anchor = anchor_of("ok. send it to Kip", "send it to Kip");
            anchor.start = (anchor.start as i32 + drift) as usize;
            let mut tracker = Tracker::new("send it to Kip".into(), Some(anchor), None);
            assert_eq!(
                tracker.read(&field("ok. send it to Kip")),
                Alignment::Unchanged
            );
            assert_eq!(
                tracker.read(&field("ok. send it to Kit")),
                Alignment::Edited("send it to Kit".into()),
                "drift {drift}"
            );
        }
    }

    #[test]
    fn a_paste_only_found_elsewhere_does_not_land_until_it_is_at_the_anchor() {
        // The first read comes before the app applied the paste, in a field
        // with another copy: nothing lands, nothing is an edit.
        let before = "send it to Kip. ok";
        let after = "send it to Kip. ok send it to Kip";
        let mut tracker = Tracker::new(
            "send it to Kip".into(),
            Some(Anchor { start: 19, len: 14 }),
            None,
        );
        assert_eq!(tracker.read(&field(before)), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("send it to Kit. ok")),
            Alignment::NotFound
        );
        assert_eq!(tracker.read(&field(after)), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("send it to Kip. ok send it to Kit")),
            Alignment::Edited("send it to Kit".into())
        );
        // An anchor that is nowhere near the paste never lands.
        let mut tracker = Tracker::new(
            "send it to Kip".into(),
            Some(Anchor {
                start: 500,
                len: 14,
            }),
            None,
        );
        for text in ["some words send it to Kip", "some words send it to Kit"] {
            assert!(!matches!(tracker.read(&field(text)), Alignment::Edited(_)));
        }
    }

    #[test]
    fn nothing_is_an_edit_before_the_paste_was_seen() {
        for anchor in [None, Some(Anchor { start: 0, len: 6 })] {
            let mut tracker = Tracker::new("Kip".into(), anchor, Some(""));
            assert_eq!(tracker.read(&field("secret")), Alignment::NotFound);
        }
    }

    #[test]
    fn without_a_caret_the_one_new_copy_is_adopted() {
        let before = "ok. thanks";
        let mut tracker = Tracker::new("send it to Kip".into(), None, Some(before));
        assert_eq!(tracker.read(&field(before)), Alignment::NotFound);
        assert_eq!(
            tracker.read(&field("ok. thanks send it to Kip")),
            Alignment::Unchanged
        );
        assert_eq!(
            tracker.read(&field("ok. thanks send it to Kit")),
            Alignment::Edited("send it to Kit".into())
        );
        assert_eq!(
            tracker.read(&field("ok. thanks send it to Kit and typed after")),
            Alignment::Edited("send it to Kit".into())
        );
    }

    #[test]
    fn without_a_caret_a_copy_held_before_or_twice_or_unknown_is_never_an_edit() {
        // Held before the paste: the new one cannot be told from the old.
        let mut tracker = Tracker::new("send it to Kip".into(), None, Some("send it to Kip. ok"));
        assert_eq!(
            tracker.read(&field("send it to Kip. ok send it to Kip")),
            Alignment::Unchanged
        );
        assert_eq!(
            tracker.read(&field("send it to Kip. ok send it to Kit")),
            Alignment::Unchanged,
            "the old copy is all that is left: never an edit"
        );
        // Twice at once, then one is edited.
        let mut tracker = Tracker::new("send it to Kip".into(), None, Some("ok"));
        assert_eq!(
            tracker.read(&field("ok send it to Kip send it to Kip")),
            Alignment::Unchanged
        );
        assert_eq!(
            tracker.read(&field("ok send it to Kip send it to Kit")),
            Alignment::Unchanged
        );
        // The field before the paste was never read.
        let mut tracker = Tracker::new("send it to Kip".into(), None, None);
        assert_eq!(
            tracker.read(&field("ok send it to Kip")),
            Alignment::Unchanged
        );
        assert_eq!(
            tracker.read(&field("ok send it to Kit")),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_window_cannot_be_followed() {
        let mut tracker = landed("ok. send it to Kip", "send it to Kip");
        let window = FieldText {
            text: "words around the caret send it to Kit".into(),
            windowed: true,
        };
        assert_eq!(tracker.read(&window), Alignment::NotFound);
        assert_eq!(
            tracker.read(&field("ok. send it to Kit")),
            Alignment::NotFound
        );
    }

    #[test]
    fn offsets_count_utf16_units_and_attachments_keep_their_place() {
        let base = "caf\u{e9} \u{1F600} send it to Kip";
        assert_eq!(
            edited(
                base,
                "send it to Kip",
                &["caf\u{e9} \u{1F600} send it to Kit"]
            ),
            "send it to Kit"
        );
        let base = "see \u{FFFC} attached. send it to Kip";
        assert_eq!(
            edited(
                base,
                "send it to Kip",
                &["see \u{FFFC} attached. send it to Kit"]
            ),
            "send it to Kit"
        );
        let base = "\u{FFFC}\u{FFFC}send it to Kip";
        assert_eq!(
            edited(base, "send it to Kip", &["\u{FFFC}\u{FFFC}send it to Kit"]),
            "send it to Kit"
        );
    }

    #[test]
    fn follow_places_the_region_by_where_the_change_is() {
        let region = Region { from: 6, to: 10 };
        let open = Open::default();
        // (before, after, the region after)
        let table = [
            ("ab    wxyz tail", "ab    wxyz tail", Some((6, 10))),
            ("ab    wxyz tail", "ZZab    wxyz tail", Some((8, 12))),
            ("ab    wxyz tail", "ab    wxyz tail more", Some((6, 10))),
            ("ab    wxyz tail", "ab    wQxyz tail", Some((6, 11))),
            ("ab    wxyz tail", "ab    wz tail", Some((6, 8))),
            ("ab    wxyz tail", "ab    tail", None),
            ("ab    wxyz tail", "ab", None),
            ("ab    wxyz tail", "ab    wxyzQ tail", Some((6, 10))),
        ];
        for (before, after, expected) in table {
            let got = follow(before, after, region, open).map(|(r, _)| (r.from, r.to));
            assert_eq!(got, expected, "{before:?} -> {after:?}");
        }
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
    /// other the paste. On every path (anchored, and with no caret) a field
    /// that does not hold the paste stores no edit, and an edit of the paste
    /// beside the unrelated text never carries that text.
    #[test]
    fn the_unrelated_pair_corpus_stores_no_edit_on_any_path() {
        let mut pairs = 0;
        for field_text in CHAT {
            for pasted in CHAT {
                if field_text == pasted {
                    continue;
                }
                pairs += 1;
                let base = format!("{field_text} {pasted}");
                let words: Vec<&str> = pasted.split(' ').collect();
                let edited_base =
                    format!("{field_text} {} Hex", words[..words.len() - 1].join(" "));
                let trackers = [
                    Tracker::new(pasted.into(), Some(anchor_of(&base, pasted)), None),
                    Tracker::new(pasted.into(), None, Some(field_text)),
                ];
                for mut tracker in trackers {
                    // The field alone, without the paste: nothing.
                    let mut alone = Tracker::new(pasted.into(), None, Some(field_text));
                    assert!(
                        !matches!(alone.read(&field(field_text)), Alignment::Edited(_)),
                        "{field_text:?} / {pasted:?}"
                    );
                    assert_eq!(tracker.read(&field(&base)), Alignment::Unchanged);
                    match tracker.read(&field(&edited_base)) {
                        Alignment::Edited(span) => assert!(
                            !span.contains(field_text) && span.ends_with("Hex"),
                            "{field_text:?} leaked into {span:?}"
                        ),
                        other => panic!("{field_text:?} / {pasted:?}: {other:?}"),
                    }
                }
            }
        }
        assert_eq!(pairs, 1260);
    }

    #[test]
    fn a_huge_paste_is_not_compared() {
        let pasted = "other ".repeat(40_000);
        let mut tracker = Tracker::new(pasted.clone(), None, Some(""));
        let long = "word ".repeat(3_000);
        assert!(!matches!(tracker.read(&field(&long)), Alignment::Edited(_)));
    }
}
