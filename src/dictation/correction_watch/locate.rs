//! Where a pasted dictation sits in the field now, and what became of it.
//! Pure: the watcher reads the field, a [`Tracker`] says what the paste's
//! own place in it holds.
//!
//! The invariant is to fail closed: an edit is stored only from a region
//! Hex is following, only text inside it (what now occupies the place the
//! pasted text occupied), never text from before or after it. When unsure the
//! answer is `NotFound` or `Unchanged`, and a region once lost stays lost.
//!
//! How a region is found:
//!
//! - **Anchored**: the caret was read before the paste, so the paste's offset
//!   is known. The first read that shows exactly one copy of the paste's words
//!   starting within [`DRIFT`] units of it lands there, provided no other near
//!   copy was ever seen (two near copies, or one from before the paste, lose
//!   the tracker). Text typed before the paste arrives moves the anchor first,
//!   through the change locator the region uses. The region is snapped to the
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
//! How a region is followed (`Followed::after`): every read is diffed against
//! the last. Text inserted or removed before the region moves it, a change
//! inside it stretches or shrinks it, one after it leaves it, and a change
//! that crosses an edge, or could as well sit on either side of one, loses it.
//! So do two changes in one read, however far apart (the words of the two reads
//! make more than one hunk, or one replacement that could be read as two). A
//! change that deletes words at an edge (backspace, or typing over a
//! selection) leaves that edge open: typing right at it counts as inside only
//! as far as the words deleted there; one more word loses the region. A blank
//! read, or one too large to follow (a window around the caret), retires the
//! region; before the paste lands a blank read is only noted, a window retires
//! the tracker at once. Only the field's text in the region is judged
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
/// The most comparisons (inserted tokens x region tokens) worth making.
const MAX_CELLS: usize = 4_000_000;
/// The most words (before x after, once their common ends are set aside) a
/// diff of two reads is worth working out: past it the change is called
/// several.
const MAX_HUNK_CELLS: usize = 250_000;

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

/// Where the paste went, read before it: the caret's offset, in UTF-16 units
/// (what Accessibility counts in).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Anchor {
    pub start: usize,
}

/// The paste's place in the field, `[from, to)` in UTF-16 units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Region {
    from: usize,
    to: usize,
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

/// A window of the old text: what a change replaced, `[from, to)`.
#[derive(Clone, Copy, Debug)]
struct Change {
    from: usize,
    to: usize,
}

impl Change {
    fn len(self) -> usize {
        self.to - self.from
    }
}

/// The one change between two reads, as their common prefix and suffix bound
/// it, and every other window it could as well have been.
struct Diff {
    b: Vec<u16>,
    a: Vec<u16>,
    base: Change,
    slide: usize,
}

impl Diff {
    fn new(before: &str, after: &str) -> Self {
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
        Self {
            base: Change {
                from: prefix,
                to: b.len() - suffix,
            },
            slide: (prefix + suffix_alone).saturating_sub(shorter),
            b,
            a,
        }
    }

    /// What the change put in its place: its length.
    fn new_len(&self, change: Change) -> usize {
        (self.a.len() + change.len()) - self.b.len()
    }

    /// `at` of the old text, in the new, for a change before it.
    fn shift(&self, at: usize) -> usize {
        (at + self.a.len()).saturating_sub(self.b.len())
    }

    /// The change could have been made this many units earlier, as well.
    fn slid(&self) -> impl Iterator<Item = Change> + '_ {
        (0..=self.slide).map(|back| Change {
            from: self.base.from - back,
            to: self.base.to - back,
        })
    }
}

/// How many separate word-level hunks turn `before` into `after`: edits in
/// two places are two changes in one read, however far apart they are. Two
/// when the words are too tangled to work out.
fn hunk_count(before: &[Token], after: &[Token]) -> usize {
    let prefix = before
        .iter()
        .zip(after)
        .take_while(|(x, y)| x.fold == y.fold)
        .count();
    let suffix = before[prefix..]
        .iter()
        .rev()
        .zip(after[prefix..].iter().rev())
        .take_while(|(x, y)| x.fold == y.fold)
        .count();
    let (b, a) = (
        &before[prefix..before.len() - suffix],
        &after[prefix..after.len() - suffix],
    );
    if b.is_empty() && a.is_empty() {
        return 0;
    }
    if b.is_empty() || a.is_empty() {
        return 1;
    }
    if b.len().saturating_mul(a.len()) > MAX_HUNK_CELLS {
        return 2;
    }
    let width = a.len() + 1;
    let lcs = lcs_table(b, a);
    let (mut i, mut j) = (0, 0);
    let mut found = 0;
    let mut open = false;
    while i < b.len() || j < a.len() {
        let same = i < b.len()
            && j < a.len()
            && b[i].fold == a[j].fold
            && lcs[i * width + j] == lcs[(i + 1) * width + j + 1] + 1;
        if same {
            found += usize::from(std::mem::take(&mut open));
            (i, j) = (i + 1, j + 1);
            continue;
        }
        open = true;
        if j == a.len() || (i < b.len() && lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) {
            i += 1;
        } else {
            j += 1;
        }
    }
    found + usize::from(open)
}

/// lcs[i * (a.len() + 1) + j]: the common words of b[i..] and a[j..].
fn lcs_table(b: &[Token], a: &[Token]) -> Vec<u16> {
    let width = a.len() + 1;
    let mut lcs = vec![0_u16; (b.len() + 1) * width];
    for i in (0..b.len()).rev() {
        for j in (0..a.len()).rev() {
            lcs[i * width + j] = if b[i].fold == a[j].fold {
                lcs[(i + 1) * width + j + 1] + 1
            } else {
                lcs[(i + 1) * width + j].max(lcs[i * width + j + 1])
            };
        }
    }
    lcs
}

/// The edges of a region, one flag each.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Edges {
    start: bool,
    end: bool,
}

impl Edges {
    fn or(self, other: Self) -> Self {
        Self {
            start: self.start || other.start,
            end: self.end || other.end,
        }
    }
}

/// How many more words may still be put back at each open edge: the whole
/// words taken away there, less the words typed there since.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Room {
    start: usize,
    end: usize,
}

/// A region and what is known of its edges: what is followed.
///
/// An edge is `open` once a change took or replaced its whole edge word, by
/// backspace, selection, typing over or forward delete: the keys typed right
/// at it are the replacement, so they are inside, but only as many words as
/// were taken away at that edge (`room`; the region's word count also stays
/// within the paste's). Whitespace typed at an open edge once its words are
/// back closes it and is never part of the region. An edge is `fringed` when
/// words were typed right outside it (on the same line): it can never open,
/// since which of the words around it is the replacement is not known.
#[derive(Clone, Copy)]
struct Followed {
    region: Region,
    open: Edges,
    room: Room,
    fringed: Edges,
}

fn is_ws_unit(unit: u16) -> bool {
    char::from_u32(u32::from(unit)).is_some_and(is_space)
}

fn is_blank_units(units: &[u16]) -> bool {
    units.iter().all(|&u| is_ws_unit(u))
}

/// Blank without a line break: words typed this far from the region are right
/// beside it.
fn is_gap(units: &[u16]) -> bool {
    units.iter().all(|&u| {
        char::from_u32(u32::from(u)).is_some_and(|c| is_space(c) && c != '\n' && c != '\r')
    })
}

/// The words `typed` puts in, put between the units `left` and `right` of the
/// text around it: a word that runs on from either is not a new one.
fn new_words(typed: &[u16], left: Option<u16>, right: Option<u16>) -> usize {
    let runs = typed
        .iter()
        .enumerate()
        .filter(|&(at, &u)| !is_ws_unit(u) && (at == 0 || is_ws_unit(typed[at - 1])))
        .count();
    let glued_left =
        typed.first().is_some_and(|&u| !is_ws_unit(u)) && left.is_some_and(|u| !is_ws_unit(u));
    let glued_right =
        typed.last().is_some_and(|&u| !is_ws_unit(u)) && right.is_some_and(|u| !is_ws_unit(u));
    runs.saturating_sub(usize::from(glued_left) + usize::from(glued_right))
}

impl Followed {
    /// Where the region of `before` is in `after`, from the one change
    /// between them. See the module doc. `None` when it is lost: edits in
    /// more than one place, a change that crosses an edge, one that could as
    /// well sit on either side of one (text typed before the paste that begins
    /// with what the paste begins with is also an insertion a few characters
    /// in), one that could as well have been typed beyond a separator the
    /// region keeps, an edge that opens beside words typed there, words
    /// put back at an edge beyond those taken from it, a word that now runs
    /// across the region's edge into the text around it, or a region that
    /// would hold more words than the paste did (`budget`).
    fn after(
        self,
        (before, before_words): (&str, &[Token]),
        (after, words): (&str, &[Token]),
        budget: usize,
    ) -> Option<Self> {
        if before == after {
            return Some(self);
        }
        if hunk_count(before_words, words) > 1 {
            return None;
        }
        let edge = tokens_in(before_words, self.region);
        let diff = Diff::new(before, after);
        // Where no word is left (attachments, emoji), only more keys at an
        // open edge are followed: a change that takes anything away is not
        // known to be the paste's.
        if diff.base.len() > 0 && edge.is_empty() {
            return None;
        }
        let followed = self.placed(&diff, edge)?;
        // The region starts at its first word: whitespace and attachments a
        // coalesced change brought in before it are not the paste's, and text
        // typed before it must land outside. While the start is open, typing
        // goes where the old word was, so it stays where it is.
        let region = followed.region;
        let from = match (followed.open.start, tokens_in(words, region).first()) {
            (false, Some(first)) => region.from.max(first.u16_start),
            _ => region.from,
        };
        let region = Region { from, ..region };
        let held = tokens_in(words, region);
        // A word that runs out of the region is the region's text joined to
        // the text beside it: none of it is known to be the paste's.
        if held
            .first()
            .zip(held.last())
            .is_some_and(|(first, last)| first.u16_start < region.from || last.u16_end > region.to)
        {
            return None;
        }
        // A region of attachments or emoji alone has no word to replace: what
        // is typed beside it is not the replacement.
        let open = if held.is_empty() && from < region.to {
            Edges::default()
        } else {
            followed.open
        };
        (held.len() <= budget).then_some(Self {
            region,
            open,
            ..followed
        })
    }

    /// Where `diff` puts the region, read every way it could have been made.
    fn placed(self, diff: &Diff, edge: &[Token]) -> Option<Self> {
        let Region { from, to } = self.region;
        // A window that crosses an edge only over whitespace deletes nothing of
        // the paste's on that side: the same text gone as "word " or " word".
        let crosses_only_blanks = |c: Change| {
            (c.from < from || c.to > to)
                && (c.from >= from || is_blank_units(&diff.b[c.from..from.min(c.to)]))
                && (c.to <= to || is_blank_units(&diff.b[to.max(c.from)..c.to]))
        };
        // Every way the change could have been made: slid earlier, as well.
        let mut placements = Vec::new();
        for window in diff.slid() {
            let placed = self.place(diff, window, edge);
            if placed.is_some() || !crosses_only_blanks(window) {
                placements.push(placed);
                continue;
            }
            // Such a window is a reading too, and must agree with the others:
            // what it takes of the region's own text is gone from the region.
            // One that takes all of it, or puts text in its place, leaves
            // nothing to agree on.
            let clipped = (diff.new_len(window) == 0 && !(window.from <= from && window.to >= to))
                .then(|| Region {
                    from: window.from.min(from),
                    to: if window.to > to {
                        window.from
                    } else {
                        to - window.len()
                    },
                });
            placements.push(clipped.map(|region| Self { region, ..self }));
        }
        let first = placements.first().copied().flatten()?;
        // Readings agree on the region's words, not on the blanks around them.
        let trimmed = |region: Region| {
            let blank = |at: usize| is_ws_unit(diff.a[at]);
            let (mut from, mut to) = (region.from, region.to);
            while from < to && blank(from) {
                from += 1;
            }
            while to > from && blank(to - 1) {
                to -= 1;
            }
            Region { from, to }
        };
        placements
            .iter()
            .all(|placed| placed.map(|p| trimmed(p.region)) == Some(trimmed(first.region)))
            .then(|| {
                // An edge is open only if every reading leaves it open (a
                // line break typed where it could as well have gone either
                // side of another closes it), with the least room any reading
                // leaves, and fringed if any reading fringes it.
                let both = Edges {
                    start: true,
                    end: true,
                };
                let open = placements.iter().flatten().fold(both, |all, p| Edges {
                    start: all.start && p.open.start,
                    end: all.end && p.open.end,
                });
                let room = placements.iter().flatten().fold(
                    Room {
                        start: usize::MAX,
                        end: usize::MAX,
                    },
                    |all, p| Room {
                        start: all.start.min(p.room.start),
                        end: all.end.min(p.room.end),
                    },
                );
                let fringed = placements
                    .iter()
                    .flatten()
                    .fold(Edges::default(), |all, p| all.or(p.fringed));
                Self {
                    region: first.region,
                    open,
                    room,
                    fringed,
                }
            })
    }

    /// Where one window of the change puts the region. `edge` is the words
    /// the region held before it.
    fn place(self, diff: &Diff, change: Change, edge: &[Token]) -> Option<Self> {
        let Region { from, to } = self.region;
        let (a, b) = (&diff.a, &diff.b);
        let new_len = diff.new_len(change);
        let moved = Region {
            from: diff.shift(from),
            to: diff.shift(to),
        };
        if change.len() == 0 {
            return self.inserted(
                change.from,
                &a[change.from..change.from + new_len],
                b,
                moved,
                edge,
            );
        }
        if change.to <= from {
            return Some(Self {
                region: moved,
                ..self
            });
        }
        if change.from >= to {
            return Some(self);
        }
        if change.from < from || change.to > to {
            return None;
        }
        self.replaced(diff, change, edge)
    }

    /// A window wholly inside the region: edge words included.
    fn replaced(self, diff: &Diff, change: Change, edge: &[Token]) -> Option<Self> {
        let Region { from, to } = self.region;
        let (a, b) = (&diff.a, &diff.b);
        let new_len = diff.new_len(change);
        let typed = &a[change.from..change.from + new_len];
        // The edge words the window took whole (a part of one is an edit of
        // it, not its replacement).
        let covers = |t: &&Token| change.from <= t.u16_start && t.u16_end <= change.to;
        let touch = Edges {
            start: edge.first().iter().any(covers),
            end: edge.last().iter().any(covers),
        };
        // (An edge words were typed beside is never open, so a fringed edge
        // that a change takes the word of is one that would open.)
        if (touch.start && self.fringed.start) || (touch.end && self.fringed.end) {
            return None;
        }
        // Whether the window runs up to an edge: no word of the region lies
        // beyond it.
        let abut = Edges {
            start: is_blank_units(&b[from..change.from]),
            end: is_blank_units(&b[change.to..to]),
        };
        let words = !is_blank_units(typed);
        // Typed words that begin with the separator the region keeps after it
        // (or end with the one before it) could as well have been typed past
        // that separator, outside the region: which it was is not known.
        if words
            && ((abut.end && b.get(change.to) == typed.first())
                || (abut.start && change.from > 0 && b.get(change.from - 1) == typed.last()))
        {
            return None;
        }
        let put = new_words(
            typed,
            change.from.checked_sub(1).map(|at| b[at]),
            b.get(change.to).copied(),
        );
        let gone_start = if touch.start {
            edge.iter().take_while(covers).count()
        } else {
            0
        };
        let gone_end = if touch.end {
            edge.iter().rev().take_while(covers).count()
        } else {
            0
        };
        // Words put back at an edge come out of the words taken from it.
        let step = |abut: bool, touch: bool, gone: usize, open: bool, room: usize| {
            if !abut {
                return Some((open, room));
            }
            let left = (if open { room } else { 0 } + gone).checked_sub(put)?;
            Some((open || touch, left))
        };
        let (start_open, start_room) = step(
            abut.start,
            touch.start,
            gone_start,
            self.open.start,
            self.room.start,
        )?;
        let (end_open, end_room) =
            step(abut.end, touch.end, gone_end, self.open.end, self.room.end)?;
        let mut followed = Self {
            region: Region {
                from,
                to: to + new_len - change.len(),
            },
            open: Edges {
                start: start_open,
                end: end_open,
            },
            room: Room {
                start: start_room,
                end: end_room,
            },
            ..self
        };
        // Words put in the middle of the region while an edge is open may be
        // that edge's, typed a little in.
        if !abut.start && !abut.end {
            let taken = edge.iter().filter(covers).count();
            followed.room = self.spend(edge, change.from, put.saturating_sub(taken), self.room)?;
        }
        // Whitespace typed past the replacement closes its edge, once the
        // words taken there are back, and is never part of the region. Before
        // that, whether the replacement goes on is not known.
        if words {
            let lead = typed.iter().take_while(|&&u| is_ws_unit(u)).count();
            let tail = typed.iter().rev().take_while(|&&u| is_ws_unit(u)).count();
            if abut.end && tail > 0 {
                if followed.open.end && followed.room.end > 0 {
                    return None;
                }
                followed.region.to -= tail;
                followed.open.end = false;
                followed.room.end = 0;
            }
            if abut.start && lead > 0 {
                if followed.open.start && followed.room.start > 0 {
                    return None;
                }
                followed.region.from = change.from + lead;
                followed.open.start = false;
                followed.room.start = 0;
            }
        }
        Some(followed)
    }

    /// `room` less `put` words typed at `at` of the region's text, taken from
    /// the open edge nearest to it (by words): none open, nothing to spend.
    fn spend(self, edge: &[Token], at: usize, put: usize, room: Room) -> Option<Room> {
        if put == 0 {
            return Some(room);
        }
        let before = edge.iter().filter(|t| t.u16_start < at).count();
        let after = edge.iter().filter(|t| t.u16_end > at).count();
        match (self.open.start, self.open.end) {
            (false, false) => Some(room),
            (true, end) if !end || before <= after => Some(Room {
                start: room.start.checked_sub(put)?,
                ..room
            }),
            _ => Some(Room {
                end: room.end.checked_sub(put)?,
                ..room
            }),
        }
    }

    /// `typed` put in at `at` of the old text `b`.
    fn inserted(
        self,
        at: usize,
        typed: &[u16],
        b: &[u16],
        moved: Region,
        edge: &[Token],
    ) -> Option<Self> {
        let Region { from, to } = self.region;
        let is_ws = |u: &u16| is_ws_unit(*u);
        let words = !is_blank_units(typed);
        let lead = typed.iter().take_while(|u| is_ws(u)).count();
        let tail = typed.len() - typed.iter().rev().take_while(|u| is_ws(u)).count();
        let breaks = !is_gap(&typed[..lead]);
        let tail_breaks = !is_gap(&typed[tail..]);
        let put = new_words(typed, at.checked_sub(1).map(|at| b[at]), b.get(at).copied());
        let fringed = |edges: Edges| Edges {
            start: self.fringed.start || edges.start,
            end: self.fringed.end || edges.end,
        };
        // A region with no word left, both edges open, is one place: words
        // typed in it come out of the room of each.
        let shared = edge.is_empty() && self.open.start && self.open.end;
        // Typing at an open edge, across spaces on its line, goes on being the
        // replacement while the words taken there are not all back; a line
        // break, or whitespace once they are, closes the edge and is never
        // part of the region.
        if self.open.end && at >= to && is_gap(&b[to..at]) {
            let full = self.room.end == 0;
            // Spaces before the words taken are back: whether the replacement
            // goes on is not known.
            if !full && !breaks && (lead > 0 || tail < typed.len() || at > to) {
                return None;
            }
            let closing = breaks || (full && (lead > 0 || at > to));
            let region = if closing || !words {
                self.region
            } else {
                Region {
                    from,
                    to: at + tail,
                }
            };
            let room = if closing {
                0
            } else {
                self.room.end.checked_sub(put)?
            };
            return Some(Self {
                region,
                open: Edges {
                    end: !closing,
                    ..self.open
                },
                room: Room {
                    end: room,
                    start: if shared && !closing {
                        self.room.start.checked_sub(put)?
                    } else {
                        self.room.start
                    },
                },
                fringed: fringed(Edges {
                    end: closing && words,
                    ..Edges::default()
                }),
            });
        }
        if self.open.start && at <= from && is_gap(&b[at..from]) {
            // While no replacement word is typed there, keys glued to the word
            // after it are typed outside the region, as before an edge that is
            // not open, or are the replacement run on to it: which is not known.
            if self.room.start > 0
                && typed.last().is_some_and(|u| !is_ws(u))
                && b.get(at).is_some_and(|u| !is_ws(u))
            {
                return None;
            }
            let full = self.room.start == 0;
            if !full && !breaks && (lead > 0 || tail < typed.len() || at < from) {
                return None;
            }
            let closing = breaks || (full && (lead > 0 || tail < typed.len()));
            let region = if closing || !words {
                moved
            } else {
                Region {
                    from: at + lead,
                    to: moved.to,
                }
            };
            let room = if closing {
                0
            } else {
                self.room.start.checked_sub(put)?
            };
            return Some(Self {
                region,
                open: Edges {
                    start: !closing,
                    ..self.open
                },
                room: Room {
                    start: room,
                    end: if shared && !closing {
                        self.room.end.checked_sub(put)?
                    } else {
                        self.room.end
                    },
                },
                fringed: fringed(Edges {
                    start: closing && words,
                    ..Edges::default()
                }),
            });
        }
        // Keys glued to the last word are that word, longer: inside.
        if at == to && to > from && to > 0 && !is_ws(&b[to - 1]) && lead == 0 && !typed.is_empty() {
            let run = typed.iter().take_while(|u| !is_ws(u)).count();
            return Some(Self {
                region: Region { from, to: to + run },
                ..self
            });
        }
        // Keys right before the first word, glued to it or across spaces, are
        // typed outside the region and fringe it.
        let beside_start = words && !tail_breaks && at <= from && is_gap(&b[at..from]);
        let beside_end = words && !breaks && at >= to && is_gap(&b[to..at]);
        if at <= from {
            return Some(Self {
                region: moved,
                fringed: fringed(Edges {
                    start: beside_start,
                    ..Edges::default()
                }),
                ..self
            });
        }
        if at >= to {
            return Some(Self {
                fringed: fringed(Edges {
                    end: beside_end,
                    ..Edges::default()
                }),
                ..self
            });
        }
        Some(Self {
            region: Region { from, to: moved.to },
            room: self.spend(edge, at, put, self.room)?,
            ..self
        })
    }
}

/// Where `start` of `before` is in `after`, text inserted or removed before it
/// moving it. `None` when a change crosses it, or could as well have been made
/// on either side of it. Text inserted exactly at `start` goes after it for
/// the start of a paste not yet seen (the paste itself arrives there) and
/// before it for the start of a copy already in the field (`copy`): that copy
/// moves on.
fn shift_start(before: &str, after: &str, start: usize, copy: bool) -> Option<usize> {
    if before == after {
        return Some(start);
    }
    let diff = Diff::new(before, after);
    let place = |change: Change| {
        if change.len() == 0 {
            Some(if change.from < start || (copy && change.from == start) {
                diff.shift(start)
            } else {
                start
            })
        } else if change.to <= start {
            Some(diff.shift(start))
        } else if change.from >= start {
            Some(start)
        } else {
            None
        }
    };
    let base = place(diff.base)?;
    diff.slid().all(|c| place(c) == Some(base)).then_some(base)
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
        // went are what it became (the region holds no more words than the
        // paste did).
        return edited();
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
    /// The paste has not been seen where it will be followed. `expect` is
    /// where it should start (moved through every change since the paste),
    /// `known` how many times the field held its text before it (no caret
    /// only), `last` the read before.
    Waiting {
        expect: Option<usize>,
        known: Option<usize>,
        /// Where copies of the paste's words that the field held before it
        /// start now (moved through every change since): never the paste.
        old: Vec<usize>,
        last: Option<String>,
        /// Whether a read has been seen without the paste in place.
        seen: bool,
    },
    /// Followed: `text` is the read the region was last placed in.
    Following { followed: Followed, text: String },
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
    /// `before` is the field's text before the paste, when it could be read.
    pub fn new(inserted: String, anchor: Option<Anchor>, before: Option<&str>) -> Self {
        let want = tokens(inserted.trim());
        let before_words = before.map(tokens);
        let known = before_words
            .as_ref()
            .map(|words| occurrences(words, &want).len());
        // The paste's text already within reach of where it went: which copy
        // is the paste could never be told.
        let old = before_words
            .as_ref()
            .map(|words| {
                occurrences(words, &want)
                    .into_iter()
                    .map(|at| words[at].u16_start)
                    .collect()
            })
            .unwrap_or_default();
        let ambiguous = anchor
            .zip(before_words.as_ref())
            .is_some_and(|(anchor, words)| !near(words, &want, anchor.start).is_empty());
        // Without the field as it stood, no copy of the paste's words nearby
        // can be told from the paste: nothing is followed.
        let state = if ambiguous || before.is_none() {
            Tracking::Lost
        } else {
            Tracking::Waiting {
                expect: anchor.map(|anchor| anchor.start),
                known,
                old,
                last: before.map(str::to_string),
                seen: false,
            }
        };
        Self {
            inserted,
            want,
            state,
        }
    }

    #[cfg(test)]
    fn region(&self) -> Option<Region> {
        match &self.state {
            Tracking::Following { followed, .. } => Some(followed.region),
            _ => None,
        }
    }

    /// What the field says of the paste now. A blank field retires a
    /// followed region (before the paste lands it is only noted), and so does
    /// a window, which cannot be followed.
    pub fn read(&mut self, field: &FieldText) -> Alignment {
        if self.want.is_empty() {
            return Alignment::NotFound;
        }
        let have = tokens(&field.text);
        let state = std::mem::replace(&mut self.state, Tracking::Lost);
        let (state, alignment) = self.advance(state, field, &have);
        self.state = state;
        alignment
    }

    fn advance(&self, state: Tracking, field: &FieldText, have: &[Token]) -> (Tracking, Alignment) {
        let FieldText { text, windowed } = field;
        let unchanged_only = || {
            if contains_run(have, &self.want) {
                Alignment::Unchanged
            } else {
                Alignment::NotFound
            }
        };
        // A window's offsets are not the field's: nothing is followed or
        // landed from one.
        if *windowed {
            return (Tracking::Lost, unchanged_only());
        }
        if is_blank(text) {
            let state = match state {
                Tracking::Waiting {
                    expect,
                    known,
                    old,
                    seen,
                    ..
                } => Tracking::Waiting {
                    expect,
                    known,
                    old,
                    last: None,
                    seen,
                },
                _ => Tracking::Lost,
            };
            return (state, Alignment::NotFound);
        }
        match state {
            Tracking::Lost => (Tracking::Lost, unchanged_only()),
            Tracking::Waiting {
                expect,
                known,
                old,
                last,
                seen,
            } => self.wait(expect, known, old, (last, seen), text, have),
            Tracking::Following {
                followed,
                text: before,
            } => match followed.after((&before, &tokens(&before)), (text, have), self.want.len()) {
                None => (Tracking::Lost, unchanged_only()),
                Some(next) => {
                    let alignment = judge(text, have, &self.want, &self.inserted, next.region);
                    let state = Tracking::Following {
                        followed: next,
                        text: text.clone(),
                    };
                    (state, alignment)
                }
            },
        }
    }

    /// A read while the paste is waited for.
    fn wait(
        &self,
        expect: Option<usize>,
        known: Option<usize>,
        old: Vec<usize>,
        (last, seen): (Option<String>, bool),
        text: &str,
        have: &[Token],
    ) -> (Tracking, Alignment) {
        let found = occurrences(have, &self.want);
        let unchanged_only = |found: &[usize]| {
            if found.is_empty() {
                Alignment::NotFound
            } else {
                Alignment::Unchanged
            }
        };
        let waiting = |expect, old| Tracking::Waiting {
            expect,
            known,
            old,
            last: Some(text.to_string()),
            seen: true,
        };
        let landing = match (expect, known) {
            (Some(start), _) => {
                // The anchor, and the copies the field held before the paste,
                // move with text typed before the paste arrived. The change
                // that brought the paste in is not typing before it, wherever
                // the anchor put it.
                let moved = match last.as_deref() {
                    None => Some((start, old)),
                    Some(last) => {
                        let start = if arrival(last, text, have, &self.want) {
                            Some(start)
                        } else {
                            shift_start(last, text, start, false)
                        };
                        let old: Option<Vec<usize>> = old
                            .iter()
                            .map(|&at| shift_start(last, text, at, true))
                            .collect();
                        start.zip(old)
                    }
                };
                let Some((start, old)) = moved else {
                    return (Tracking::Lost, unchanged_only(&found));
                };
                let (near, copies): (Vec<usize>, Vec<usize>) = near(have, &self.want, start)
                    .into_iter()
                    .partition(|&at| !old.contains(&have[at].u16_start));
                match (near.as_slice(), copies.is_empty()) {
                    // Nothing new there: a copy brought near is never the paste.
                    ([], _) => return (waiting(Some(start), old), unchanged_only(&found)),
                    ([only], true) => *only,
                    // Two copies within reach: which is the paste?
                    _ => return (Tracking::Lost, Alignment::Unchanged),
                }
            }
            (None, Some(0)) if found.len() == 1 => found[0],
            (None, _) if found.is_empty() => return (waiting(None, old), Alignment::NotFound),
            // No caret: the text twice, or held before, or unknown.
            (None, _) => return (Tracking::Lost, Alignment::Unchanged),
        };
        // A paste not in place at the first read, then found, may be the same
        // words typed: it lands only as one change that brings in more than a
        // character of them (a keystroke of one cannot be told from it).
        let arrived = last
            .as_deref()
            .is_none_or(|last| arrival(last, text, have, &self.want));
        if seen && (!arrived || self.inserted.trim().chars().count() < 2) {
            return (Tracking::Lost, unchanged_only(&found));
        }
        let region = Region {
            from: have[landing].u16_start,
            to: have[landing + self.want.len() - 1].u16_end,
        };
        let state = Tracking::Following {
            followed: Followed {
                region,
                open: Edges::default(),
                room: Room::default(),
                fringed: Edges::default(),
            },
            text: text.to_string(),
        };
        (state, Alignment::Unchanged)
    }
}

/// Whether the change from `before` to `after` is the paste arriving: a copy
/// of its words starts inside what the change put there.
fn arrival(before: &str, after: &str, words: &[Token], want: &[Token]) -> bool {
    if before == after {
        return false;
    }
    let diff = Diff::new(before, after);
    let (from, to) = (diff.base.from, diff.base.from + diff.new_len(diff.base));
    occurrences(words, want)
        .into_iter()
        .any(|at| (from..to).contains(&words[at].u16_start))
}

/// Where in `have` the words `want` occur starting within [`DRIFT`] units of
/// `start`.
fn near(have: &[Token], want: &[Token], start: usize) -> Vec<usize> {
    occurrences(have, want)
        .into_iter()
        .filter(|&at| have[at].u16_start.abs_diff(start) <= DRIFT)
        .collect()
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
        }
    }

    /// `base` as it stood before its last copy of `pasted` was pasted.
    fn without(base: &str, pasted: &str) -> String {
        let at = base.rfind(pasted).expect("the paste is in the base field");
        format!("{}{}", &base[..at], &base[at + pasted.len()..])
    }

    /// A tracker anchored on the last copy of `pasted` in `base`, having read
    /// `base`: the paste has landed.
    fn landed(base: &str, pasted: &str) -> Tracker {
        let before = without(base, pasted);
        let mut tracker = Tracker::new(
            pasted.to_string(),
            Some(anchor_of(base, pasted)),
            Some(&before),
        );
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
    fn a_trailing_replacement_may_put_back_only_the_words_it_deleted() {
        let base = "send it to Kip";
        // One word for one: stored.
        assert_eq!(edited(base, base, &["send it to Kit"]), "send it to Kit");
        // One word for two is lost: it is the replacement and what comes
        // after it, and which is which is not known.
        assert_eq!(
            last(base, base, &["send it to Kit and"]),
            Alignment::NotFound
        );
        assert_eq!(
            last(base, base, &["send it to Kit and Sarah"]),
            Alignment::NotFound
        );
        // Key by key: the space after "Kit" closes the edge, so what is typed
        // on is outside the region and the fixed word is kept.
        let keys = typing("send it to ", "Kit and Sarah");
        let mut reads = vec!["send it to Ki", "send it to K", "send it to "];
        reads.extend(refs(&keys));
        assert_eq!(edited(base, base, &reads), "send it to Kit");
        // Two words typed over two words are stored in one read. Key by key
        // the space after the first is lost, not cut short: with a word of the
        // paste still unused, whether the replacement goes on is not known.
        let base = "report. ok thanks";
        assert_eq!(edited(base, "ok thanks", &["report. yes sure"]), "yes sure");
        let keys = typing("report. ", "yes sure");
        assert_eq!(last(base, "ok thanks", &refs(&keys)), Alignment::NotFound);
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
            "ok.  Smith said hi",
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
        let base = "ok. Hanks";
        let keys = typing("ok. ", "Hex");
        let mut reads = vec!["ok. Hank", "ok. Han", "ok. Ha", "ok. H", "ok. "];
        reads.extend(refs(&keys));
        assert_eq!(edited(base, "Hanks", &reads), "Hex");
    }

    #[test]
    fn selecting_a_word_and_typing_over_it_is_followed_key_by_key() {
        // One word selected and one typed over it.
        let base = "please send it to Kip tomorrow";
        let keys = typing("please send it to Kip ", "Friday");
        assert_eq!(
            edited(base, "please send it to Kip tomorrow", &refs(&keys)),
            "please send it to Kip Friday"
        );
        // Three typed over one: the first is the replacement; what follows the
        // space after it is outside.
        let keys = typing("please send it to Kip ", "Kit and Sarah");
        assert_eq!(
            edited(base, "please send it to Kip tomorrow", &refs(&keys)),
            "please send it to Kip Kit"
        );
        // Two typed over two, by key: lost at the space, never cut short.
        let base = "report. ok thanks";
        let keys = typing("report. ", "yes sure");
        never_holds(&answers(base, "ok thanks", &refs(&keys)), &["sure"]);
        assert_eq!(last(base, "ok thanks", &refs(&keys)), Alignment::NotFound);
    }

    #[test]
    fn the_start_edge_backspaced_from_after_the_word_and_retyped() {
        // Backspace "Hanks" from after it, leaving the space, then type "Hex".
        let base = "ok. Hanks Smith said hi";
        let mut reads = vec![
            "ok. Hank Smith said hi".to_string(),
            "ok. Han Smith said hi".to_string(),
            "ok. Ha Smith said hi".to_string(),
            "ok. H Smith said hi".to_string(),
            "ok.  Smith said hi".to_string(),
        ];
        reads.extend(
            typing("ok. ", "Hex")
                .into_iter()
                .map(|k| format!("{k} Smith said hi")),
        );
        assert_eq!(
            edited(base, "Hanks Smith said hi", &refs(&reads)),
            "Hex Smith said hi"
        );
        // A lone word, the same way.
        let mut reads = vec!["ok. Hank", "ok. Han", "ok. Ha", "ok. H", "ok. "]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        reads.extend(typing("ok. ", "Hex"));
        assert_eq!(edited("ok. Hanks", "Hanks", &refs(&reads)), "Hex");
    }

    #[test]
    fn two_changes_in_one_read_lose_the_region_however_far_apart() {
        // "it" to "Hex" and a word appended, together.
        assert_eq!(
            last("send it to Kip", "send it to Kip", &["send Hex to Kip zz"]),
            Alignment::NotFound
        );
        // Long identical context either side (past any window) does not let a
        // coalesced edit and appended sentence in.
        let prefix = "the quick brown fox jumps over the lazy dog ";
        let suffix = " and then it went home with all of the other animals";
        let before = format!("{prefix}send it to Kip{suffix}");
        let after = format!("{prefix}send it to Kit and an unrelated appended sentence{suffix}");
        assert_eq!(
            last(&before, "send it to Kip", &[&after]),
            Alignment::NotFound
        );
        // Two edited words that are not next to each other.
        assert_eq!(
            last(
                "a. one two three four five",
                "one two three four five",
                &["a. one XX three YY five"]
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_blank_field_before_the_paste_lands_is_only_noted() {
        // The field is empty until the paste arrives: that is not a loss.
        let mut tracker =
            Tracker::new("send it to Kip".into(), Some(Anchor { start: 0 }), Some(""));
        assert_eq!(tracker.read(&field("")), Alignment::NotFound);
        assert_eq!(tracker.read(&field("send it to Kip")), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("send it to Kit")),
            Alignment::Edited("send it to Kit".into())
        );
        // No caret, the same.
        let mut tracker = Tracker::new("send it to Kip".into(), None, Some(""));
        assert_eq!(tracker.read(&field("")), Alignment::NotFound);
        assert_eq!(tracker.read(&field("send it to Kip")), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("send it to Kit")),
            Alignment::Edited("send it to Kit".into())
        );
    }

    #[test]
    fn a_copy_of_the_paste_within_reach_is_never_told_from_it() {
        let never = |a: Alignment| assert!(!matches!(a, Alignment::Edited(_)), "{a:?}");
        // "Yes Yes": the paste "Yes" sits beside the "Yes" that was there.
        let mut tracker = Tracker::new("Yes".into(), Some(Anchor { start: 3 }), Some("ok Yes"));
        never(tracker.read(&field("ok Yes Yes")));
        never(tracker.read(&field("ok Yes No")));
        never(tracker.read(&field("ok No No")));
        // Not told by the field before it: the old copy was already near.
        let mut tracker = Tracker::new(
            "Hanks".into(),
            Some(Anchor { start: 0 }),
            Some("Hanks went"),
        );
        never(tracker.read(&field("Hanks Hanks went")));
        never(tracker.read(&field("Hex Hanks went")));
        never(tracker.read(&field("Hex Hex went")));
        // Both near on one read, with no read before the paste.
        let mut tracker = Tracker::new("Hanks".into(), Some(Anchor { start: 0 }), None);
        never(tracker.read(&field("Hanks Hanks went")));
        never(tracker.read(&field("Hex Hanks went")));
        never(tracker.read(&field("Hex Hex went")));
        // An older copy far away is no matter.
        let filler = "x ".repeat(35);
        let mut tracker = Tracker::new(
            "send it to Kip".into(),
            Some(Anchor { start: 85 }),
            Some(&format!("send it to Kip {filler}")),
        );
        let text = format!("send it to Kip {filler}send it to Kip");
        assert_eq!(tracker.read(&field(&text)), Alignment::Unchanged);
        let edited = format!("send it to Kip {filler}send it to Kit");
        assert_eq!(
            tracker.read(&field(&edited)),
            Alignment::Edited("send it to Kit".into())
        );
    }

    #[test]
    fn without_a_caret_two_copies_at_once_are_never_adopted_first() {
        let never = |a: Alignment| assert!(!matches!(a, Alignment::Edited(_)), "{a:?}");
        let both = "ok send it to Kip send it to Kip";
        // The first of the two edited, then the second.
        let mut tracker = Tracker::new("send it to Kip".into(), None, Some("ok"));
        assert_eq!(tracker.read(&field(both)), Alignment::Unchanged);
        never(tracker.read(&field("ok send it to Kit send it to Kip")));
        let mut tracker = Tracker::new("send it to Kip".into(), None, Some("ok"));
        assert_eq!(tracker.read(&field(both)), Alignment::Unchanged);
        never(tracker.read(&field("ok send it to Kip send it to Kit")));
    }

    #[test]
    fn deleting_a_first_word_with_its_space_is_one_change_either_way_it_is_read() {
        // "Hanks " or " Hanks" gone: the same text, the space before the
        // region deleted or the one after the word.
        assert_eq!(
            edited(
                "ok. Hanks Smith said hi",
                "Hanks Smith said hi",
                &["ok. Smith said hi"]
            ),
            "Smith said hi"
        );
    }

    #[test]
    fn a_deletion_that_may_have_cut_into_the_region_loses_it() {
        // "aa " gone from "aa aa bb": the first copy, or "a a" across the
        // region's start. Which is not known, so an edit after it is not
        // taken for the paste's.
        assert_eq!(
            last("aa aa bb", "aa bb", &["aa bb", "aa cc"]),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_region_with_no_word_left_is_lost_and_what_is_typed_beside_it_is_not_it() {
        // The paste replaced by attachments, then words typed where it was.
        assert_eq!(
            last(
                "ok. please send it to Kip tomorrow",
                "please send it to Kip tomorrow",
                &[
                    "ok. \u{fffc}\u{fffc}",
                    "ok. banana please review the document and send it back"
                ]
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_word_edited_and_more_typed_after_that_ends_as_the_paste_does_is_two_changes() {
        // "for" to "fot", then " I will get back to you tomorrow" at the end:
        // read as one replacement of "for" it would hold the text typed after.
        assert_eq!(
            last(
                "ok. what time works for you tomorrow",
                "what time works for you tomorrow",
                &["ok. what time works fot you tomorrow I will get back to you tomorrow"]
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_first_word_retyped_as_the_word_before_the_paste_is_still_followed() {
        // "I" backspaced and "Friday" typed where it was, right after the
        // "Friday" that sits before the paste: as many words in as out is one
        // change however the words line up.
        let base = "ok. Friday I will be there in ten minutes";
        let tail = " will be there in ten minutes";
        let mut reads = vec![format!("ok. Friday {tail}")];
        reads.extend(
            typing("ok. Friday ", "Friday")
                .into_iter()
                .map(|r| format!("{r}{tail}")),
        );
        let reads: Vec<&str> = reads.iter().map(String::as_str).collect();
        assert_eq!(
            edited(base, "I will be there in ten minutes", &reads),
            "Friday will be there in ten minutes"
        );
    }

    #[test]
    fn a_long_paste_whose_words_are_all_near_but_none_exact_is_not_it() {
        // Every word one letter off: half the words, but not two exact.
        assert_eq!(
            last(
                "ok. please send Kip tomorrow today",
                "please send Kip tomorrow today",
                &["ok. pleased sends Kit tomorrows todays"]
            ),
            Alignment::NotFound
        );
    }

    #[test]
    fn whitespace_and_attachments_a_change_brought_in_before_the_region_stay_outside_it() {
        // An edit and two attachments typed before it, in one read; then
        // words typed right before the paste: they land outside it.
        let base = "x. what time works for you";
        let read2 = "x. \u{FFFC}\u{FFFC} what time works Rune you";
        let read3 = "x. \u{FFFC}\u{FFFC} let me what time works Rune you";
        assert_eq!(
            edited(base, "what time works for you", &[read2, read3]),
            "what time works Rune you"
        );
    }

    #[test]
    fn the_anchor_moves_with_text_typed_before_the_paste_arrives() {
        let mut tracker = Tracker::new(
            "send it to Kip".into(),
            Some(Anchor { start: 3 }),
            Some("ok. and send it to Kip tonight"),
        );
        // Typed before the paste's place, then the paste arrives (twice
        // over, as an older copy lies further on).
        assert_eq!(
            tracker.read(&field("ok. and send it to Kip tonight")),
            Alignment::Unchanged
        );
        assert_eq!(
            tracker.read(&field("hello ok. and send it to Kip tonight")),
            Alignment::Unchanged
        );
        assert!(
            tracker.region().is_none(),
            "the copy is an old one, not the paste"
        );
    }

    #[test]
    fn a_window_while_waiting_retires_the_tracker() {
        let mut tracker = Tracker::new(
            "send it to Kip".into(),
            Some(Anchor { start: 3 }),
            Some("ok."),
        );
        let window = FieldText {
            text: "ok. send it to Kip".into(),
            windowed: true,
        };
        assert_eq!(tracker.read(&window), Alignment::Unchanged);
        // Nothing lands after a window, and nothing is an edit.
        assert_eq!(
            tracker.read(&field("ok. send it to Kip")),
            Alignment::Unchanged
        );
        assert_eq!(
            tracker.read(&field("ok. send it to Kit")),
            Alignment::NotFound
        );
    }

    #[test]
    fn a_long_paste_replaced_by_a_few_shared_words_is_not_an_edit() {
        // The floor: half the words, two of them exact.
        assert_eq!(
            last(
                "ok. send it to Kip tomorrow",
                "send it to Kip tomorrow",
                &["ok. send the weather is nice"]
            ),
            Alignment::NotFound
        );
        // And a short paste grown far past its words is no longer it.
        let grown = "ok. Hanks one two three four five six seven Smith";
        assert_eq!(
            last("ok. Hanks Smith", "Hanks Smith", &[grown]),
            Alignment::NotFound
        );
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
            Some("ok. "),
        );
        assert_eq!(tracker.read(&field(base)), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("ok. alpha beta\ngamma Rune")),
            Alignment::Edited("alpha beta\ngamma Rune".into())
        );
        // Typing on after a space is outside the region.
        assert_eq!(
            tracker.read(&field("ok. alpha beta\ngamma Rune typed after text")),
            Alignment::Edited("alpha beta\ngamma Rune".into())
        );
        // An anchor a few units off still lands, on the paste.
        for drift in [-3_i32, -2, -1, 1, 2, 3] {
            let mut anchor = anchor_of("ok. send it to Kip", "send it to Kip");
            anchor.start = (anchor.start as i32 + drift) as usize;
            let mut tracker = Tracker::new("send it to Kip".into(), Some(anchor), Some("ok. "));
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
            Some(Anchor { start: 19 }),
            Some(before),
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
            Some(Anchor { start: 500 }),
            Some("some words "),
        );
        for text in ["some words send it to Kip", "some words send it to Kit"] {
            assert!(!matches!(tracker.read(&field(text)), Alignment::Edited(_)));
        }
    }

    #[test]
    fn nothing_is_an_edit_before_the_paste_was_seen() {
        for anchor in [None, Some(Anchor { start: 0 })] {
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
        // (before, after, the region after)
        let table = [
            ("ab    wxyz tail", "ab    wxyz tail", Some((6, 10))),
            ("ab    wxyz tail", "ZZab    wxyz tail", Some((8, 12))),
            ("ab    wxyz tail", "ab    wxyz tail more", Some((6, 10))),
            ("ab    wxyz tail", "ab    wQxyz tail", Some((6, 11))),
            ("ab    wxyz tail", "ab    wz tail", Some((6, 8))),
            ("ab    wxyz tail", "ab    tail", None),
            ("ab    wxyz tail", "ab", None),
            ("ab    wxyz tail", "ab    wxyzQ tail", Some((6, 11))),
            ("ab    wxyz tail", "ab    wxyz Q tail", Some((6, 10))),
        ];
        for (before, after, expected) in table {
            let got = Followed {
                region,
                open: Edges::default(),
                room: Room::default(),
                fringed: Edges::default(),
            }
            .after((before, &tokens(before)), (after, &tokens(after)), 99)
            .map(|f| (f.region.from, f.region.to));
            assert_eq!(got, expected, "{before:?} -> {after:?}");
        }
    }

    /// The paste read as landed, then `reads`, every one's answer.
    fn answers(base: &str, pasted: &str, reads: &[&str]) -> Vec<Alignment> {
        let mut tracker = landed(base, pasted);
        reads.iter().map(|r| tracker.read(&field(r))).collect()
    }

    fn never_holds(answers: &[Alignment], words: &[&str]) {
        for answer in answers {
            if let Alignment::Edited(span) = answer {
                for word in words {
                    assert!(!span.contains(word), "{span:?} holds {word:?}");
                }
            }
        }
    }

    #[test]
    fn a_region_never_holds_more_words_than_the_paste_had() {
        // One word for two, in one read, is lost (Hanks to "Hank Smith").
        assert_eq!(
            last("ok. Hanks", "Hanks", &["ok. Hank Smith"]),
            Alignment::NotFound
        );
        // Key by key the fixed word is kept and what follows the space is not.
        let a = answers("ok. Hanks", "Hanks", &["ok. Hank", "ok. Hank Smith"]);
        assert_eq!(a[1], Alignment::Edited("Hank".into()));
        // "Kit chen" fixed to "Kitchen", then typing on: past the paste's words
        // it is lost, and no word of it is stored.
        let base = "send it to Kit chen";
        assert_eq!(
            last(
                base,
                base,
                &["send it to Kitchen", "send it to Kitchen and then"]
            ),
            Alignment::NotFound
        );
        let keys = typing("send it to Kitchen", " and then");
        let mut reads = vec!["send it to Kitchen"];
        reads.extend(refs(&keys));
        never_holds(&answers(base, base, &reads), &["then"]);
        // A word fixed and a password typed on, in one read or by key.
        let base = "send it to Kip";
        assert_eq!(
            last(base, base, &["send it to Kit my password is hunter2"]),
            Alignment::NotFound
        );
        let keys = typing("send it to Kit", " my password is hunter2");
        never_holds(&answers(base, base, &refs(&keys)), &["password", "hunter2"]);
    }

    #[test]
    fn a_one_character_edge_word_typed_over_is_not_stored_with_what_follows() {
        for (base, pasted, after) in [
            ("ok. call me at 5", "call me at 5", "ok. call me at six pm"),
            ("ok. plan B", "plan B", "ok. plan C instead"),
            ("ok. room 4", "room 4", "ok. room 5 on the left"),
        ] {
            assert_eq!(last(base, pasted, &[after]), Alignment::NotFound, "{after}");
        }
        // By key: the fixed word is kept, the rest is outside.
        let keys = typing("ok. call me at ", "six pm");
        let a = answers("ok. call me at 5", "call me at 5", &refs(&keys));
        never_holds(&a, &["pm"]);
        assert_eq!(a.last(), Some(&Alignment::Edited("call me at six".into())));
    }

    #[test]
    fn an_edge_that_opens_is_checked_for_ambiguous_insertions_too() {
        // "Hanks Smith" cut back to "Hanks", then "Hanks " put before it: the
        // new word is either before the region or inside it.
        let a = answers(
            "ok. Hanks Smith",
            "Hanks Smith",
            &["ok. Hanks", "ok. Hanks Hanks"],
        );
        assert_eq!(a[1], Alignment::NotFound);
    }

    #[test]
    fn a_new_first_word_then_the_old_one_forward_deleted_is_not_the_paste_minus_a_word() {
        // Typed before the paste's first word, then the first word deleted:
        // the edit is "Hanks" to "Hex", and the region holds only one of them.
        let base = "ok. Hanks Smith said hi";
        let a = answers(
            base,
            "Hanks Smith said hi",
            &["ok. Hex Hanks Smith said hi", "ok. Hex Smith said hi"],
        );
        assert_eq!(a[1], Alignment::NotFound);
        // The same typed into the word (no space) and the old letters deleted
        // forward: keys glued before the first word are outside it, and the
        // word cannot then be replaced, so it is lost, not cut short.
        let a = answers(
            base,
            "Hanks Smith said hi",
            &[
                "ok. HHanks Smith said hi",
                "ok. HeHanks Smith said hi",
                "ok. HexHanks Smith said hi",
                "ok. Hexanks Smith said hi",
                "ok. Hexnks Smith said hi",
                "ok. Hexks Smith said hi",
                "ok. Hexs Smith said hi",
                "ok. Hex Smith said hi",
            ],
        );
        assert_eq!(a.last(), Some(&Alignment::NotFound));
    }

    #[test]
    fn whitespace_typed_at_an_open_edge_closes_it_and_is_outside_the_region() {
        // The last word fixed, a space and more typed after: the fixed word
        // stays, and nothing typed after it is in.
        let a = answers(
            "ok. send it to Kip",
            "send it to Kip",
            &[
                "ok. send it to Kit",
                "ok. send it to Kit ",
                "ok. send it to Kit and",
            ],
        );
        assert_eq!(a[2], Alignment::Edited("send it to Kit".into()));
        // A newline too, and typing before the newline: outside.
        let a = answers(
            "ok. send it to Kip",
            "send it to Kip",
            &[
                "ok. send it to Kit",
                "ok. send it to Kit\n",
                "ok. send it to Kit secret words\n",
            ],
        );
        never_holds(&a, &["secret"]);
        // A space typed while the paste's words are not all used is lost: the
        // replacement may go on, and a cut-short span is never stored.
        let a = answers(
            "report. ok thanks",
            "ok thanks",
            &["report. fine", "report. fine ", "report. fine sure"],
        );
        assert_eq!(a[1], Alignment::NotFound);
        assert_eq!(a[2], Alignment::NotFound);
    }

    #[test]
    fn a_closed_edge_beside_typed_words_never_opens_again() {
        // Words typed right after the paste, then its last word taken away: the
        // span would be the paste minus a word beside the typed one.
        let a = answers(
            "ok. Hanks Smith",
            "Hanks Smith",
            &["ok. Hanks Smith Hex", "ok. Hanks Hex"],
        );
        assert_eq!(a[1], Alignment::NotFound);
    }

    #[test]
    fn a_copy_that_was_there_before_the_paste_is_never_it_even_moved_near_the_anchor() {
        // Pre-paste text: the old copy, and filler. The paste arrives at the end;
        // then everything after the old copy is deleted before the first read.
        let before = "Hanks 01234567890123";
        let mut tracker = Tracker::new(
            "Hanks".into(),
            Some(Anchor {
                start: utf16_len(before),
            }),
            Some(before),
        );
        assert_eq!(tracker.read(&field("Hanks ")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("Hex ")), Alignment::NotFound);
        assert_eq!(tracker.read(&field("Hex there")), Alignment::NotFound);
    }

    #[test]
    fn a_blank_window_retires_a_waiting_tracker() {
        let mut tracker = Tracker::new("Hanks".into(), Some(Anchor { start: 0 }), Some(""));
        let blank = FieldText {
            text: "   ".into(),
            windowed: true,
        };
        assert_eq!(tracker.read(&blank), Alignment::NotFound);
        assert_eq!(tracker.read(&field("Hanks")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("Hex")), Alignment::NotFound);
    }

    #[test]
    fn without_a_caret_a_copy_held_once_before_is_not_adopted_when_the_old_one_goes() {
        // Held once before the paste: even when only one copy is left, which
        // one it is is not known.
        let mut tracker = Tracker::new("Hanks".into(), None, Some("Hanks x"));
        assert_eq!(tracker.read(&field("x Hanks")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("x Hex")), Alignment::NotFound);
        // Held nowhere before: the one new copy is adopted.
        let mut tracker = Tracker::new("Hanks".into(), None, Some("x"));
        assert_eq!(tracker.read(&field("x Hanks")), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("x Hex")),
            Alignment::Edited("Hex".into())
        );
    }

    #[test]
    fn deleting_a_one_word_paste_and_typing_words_is_the_replacement_and_no_more() {
        // "Yes" deleted whole, then words typed where it was.
        let a = answers(
            "Yes u1x u2x",
            "Yes",
            &[
                " u1x u2x",
                "u u1x u2x",
                "u3x  u1x u2x",
                "u3x  let me know u1x u2x",
            ],
        );
        assert_eq!(a[3], Alignment::Edited("u3x".into()));
    }

    #[test]
    fn the_anchor_moves_by_text_typed_before_the_paste_arrives_beyond_the_drift() {
        // Nothing within reach of the anchor before the paste, so only the shift
        // of the anchor through the typed text lets the paste land.
        let before = "aaaa bbbb cccc dddd eeee ";
        let mut tracker = Tracker::new(
            "Hanks".into(),
            Some(Anchor {
                start: utf16_len(before),
            }),
            Some(before),
        );
        let typed = format!("{} {before}", "z".repeat(20));
        assert_eq!(tracker.read(&field(&typed)), Alignment::NotFound);
        let pasted = format!("{typed}Hanks");
        assert_eq!(tracker.read(&field(&pasted)), Alignment::Unchanged);
        let fixed = format!("{typed}Hex");
        assert_eq!(
            tracker.read(&field(&fixed)),
            Alignment::Edited("Hex".into())
        );
    }

    /// The tracker anchored `offset` units off where `pasted` lands in `base`.
    fn drifted(base: &str, pasted: &str, offset: isize) -> Tracker {
        let anchor = anchor_of(base, pasted);
        Tracker::new(
            pasted.to_string(),
            Some(Anchor {
                start: anchor.start.saturating_add_signed(offset),
            }),
            Some(&without(base, pasted)),
        )
    }

    #[test]
    fn the_paste_lands_within_eight_units_of_the_anchor_and_not_nine() {
        let base = "twelve chars. send it to Kip";
        for offset in [-8_isize, 8] {
            let mut tracker = drifted(base, "send it to Kip", offset);
            assert_eq!(tracker.read(&field(base)), Alignment::Unchanged);
            assert_eq!(
                tracker.read(&field("twelve chars. send it to Kit")),
                Alignment::Edited("send it to Kit".into()),
                "{offset}"
            );
        }
        for offset in [-9_isize, 9] {
            let mut tracker = drifted(base, "send it to Kip", offset);
            assert_eq!(tracker.read(&field(base)), Alignment::Unchanged);
            assert_eq!(
                tracker.read(&field("twelve chars. send it to Kit")),
                Alignment::NotFound,
                "{offset}"
            );
        }
    }

    #[test]
    fn a_blank_field_retires_a_followed_region_for_good() {
        let mut tracker = landed("ok. Hanks", "Hanks");
        assert_eq!(tracker.read(&field("")), Alignment::NotFound);
        assert_eq!(tracker.read(&field("ok. Hanks")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("ok. Hex")), Alignment::NotFound);
    }

    #[test]
    fn a_deletion_that_could_as_well_have_cut_into_real_text_is_lost() {
        // "ab " gone from "x b ab zz": the paste's own "ab ", or "b a" across
        // its start (a "b" that is not the paste's): which is not known.
        assert_eq!(last("x b ab zz", "ab zz", &["x b zz"]), Alignment::NotFound);
    }

    #[test]
    fn a_longer_paste_needs_half_its_words_and_two_of_them_exact() {
        // All five words near (one letter off), one of them exact: not it.
        assert_eq!(
            last(
                "ok. alpha beta gamma delta epsilon",
                "alpha beta gamma delta epsilon",
                &["ok. alpha betb gammx delt epsilo"]
            ),
            Alignment::NotFound
        );
        // Six words, two exact in order and the rest replaced: under half.
        assert_eq!(
            last(
                "ok. one two three four five six",
                "one two three four five six",
                &["ok. one two xxxx yyyy zzzz wwww"]
            ),
            Alignment::NotFound
        );
        // Five words, two exact and the rest replaced: under half of five.
        assert_eq!(
            last(
                "ok. one two three four five",
                "one two three four five",
                &["ok. one two xxxx yyyy zzzz"]
            ),
            Alignment::NotFound
        );
        // Half in order, two of them exact: it.
        assert_eq!(
            edited(
                "ok. one two three four five six",
                "one two three four five six",
                &["ok. one two three xxxx yyyy zzzz"]
            ),
            "one two three xxxx yyyy zzzz"
        );
    }

    #[test]
    fn a_change_too_big_to_work_out_word_by_word_is_lost() {
        // 520 of 1,400 words replaced in one hunk: past what the word diff
        // works out (250,000 cells), so it is called several changes.
        let words: Vec<String> = (0..1_400).map(|n| format!("w{n}x")).collect();
        let base = format!("ok. {}", words.join(" "));
        let mut changed = words.clone();
        for (n, word) in changed.iter_mut().enumerate().skip(400).take(520) {
            *word = format!("v{n}y");
        }
        let after = format!("ok. {}", changed.join(" "));
        assert_eq!(
            last(&base, &words.join(" "), &[&after]),
            Alignment::NotFound
        );
        // A small one is read as the one change it is.
        let mut small = words.clone();
        small[700] = "zzz".into();
        let after = format!("ok. {}", small.join(" "));
        assert!(matches!(
            last(&base, &words.join(" "), &[&after]),
            Alignment::Edited(_)
        ));
    }

    #[test]
    fn typing_at_the_start_edge_once_the_paste_words_are_used_closes_it() {
        // The first word replaced by one word: the paste's two words are used.
        // A word and a space typed before it are outside; the edit is kept.
        let a = answers("ok. ab cd", "ab cd", &["ok. x cd", "ok. y x cd"]);
        assert_eq!(a[1], Alignment::Edited("x cd".into()));
        // The same with the space first, and then the word.
        let a = answers(
            "ok. ab cd",
            "ab cd",
            &["ok. x cd", "ok.  x cd", "ok. y x cd"],
        );
        assert_eq!(a[2], Alignment::Edited("x cd".into()));
        // A line break at the start edge closes it before the words are used.
        let a = answers(
            "ok. ab cd ef",
            "ab cd ef",
            &["ok. cd ef", "ok. \n cd ef", "ok. \nx cd ef"],
        );
        never_holds(&a, &["x "]);
    }

    #[test]
    fn typing_at_the_start_edge_across_a_space_before_the_words_are_used_is_lost() {
        // "ab" deleted (one of three words), then a word and a space typed in
        // one read: the replacement may go on, so the region is lost.
        let a = answers("ok. ab cd ef", "ab cd ef", &["ok.  cd ef", "ok. x  cd ef"]);
        assert_eq!(a[1], Alignment::NotFound);
    }

    #[test]
    fn a_line_break_then_words_typed_on_the_next_line_is_not_beside_the_paste() {
        // Return, then a new message key by key, then the last word fixed.
        let mut reads = vec!["ok. send it to Kip\n".to_string()];
        reads.extend(typing("ok. send it to Kip\n", "thanks"));
        reads.push("ok. send it to Kit\nthanks".into());
        assert_eq!(
            edited("ok. send it to Kip", "send it to Kip", &refs(&reads)),
            "send it to Kit"
        );
    }

    #[test]
    fn a_region_of_attachments_has_no_edge_to_open() {
        let a = answers(
            "ok. Hanks",
            "Hanks",
            &["ok. \u{FFFC}\u{FFFC}", "ok. \u{FFFC}\u{FFFC}hello"],
        );
        assert_eq!(a[1], Alignment::NotFound);
        // Nor does a deletion elsewhere let a word typed where the paste was be
        // taken for its replacement.
        let a = answers("ok. one Hanks", "Hanks", &["ok. one ", "ok. ", "ok. Hex"]);
        assert_eq!(a[2], Alignment::NotFound);
    }

    #[test]
    fn an_anchor_that_could_as_well_have_moved_either_way_is_lost() {
        // "a" typed where it could be before or after the anchor ("ab" to
        // "aab"): which side is not known, so nothing lands after.
        let mut tracker = Tracker::new("Hanks".into(), Some(Anchor { start: 1 }), Some("ab"));
        assert_eq!(tracker.read(&field("aab")), Alignment::NotFound);
        assert_eq!(tracker.read(&field("aa Hanks b")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("aa Hex b")), Alignment::NotFound);
    }

    #[test]
    fn text_typed_at_the_anchor_itself_before_the_paste_arrives_is_not_shifted_past() {
        let before = "start ";
        let mut tracker = Tracker::new(
            "Hanks".into(),
            Some(Anchor {
                start: utf16_len(before),
            }),
            Some(before),
        );
        let typed = format!("{before}{} ", "y".repeat(20));
        assert_eq!(tracker.read(&field(&typed)), Alignment::NotFound);
        let pasted = format!("{typed}Hanks");
        assert_eq!(tracker.read(&field(&pasted)), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field(&format!("{typed}Hex"))),
            Alignment::NotFound
        );
    }

    #[test]
    fn an_old_copy_is_moved_with_the_text_so_it_cannot_land_after_a_deletion() {
        // The old copy is brought near the anchor by a deletion after it, and
        // then text before both is deleted: it moves with the text, and is
        // still the old copy.
        let before = format!("{} Hanks 0 {}", "A".repeat(20), "b".repeat(10));
        let mut tracker = Tracker::new(
            "Hanks".into(),
            Some(Anchor {
                start: utf16_len(&before),
            }),
            Some(&before),
        );
        let near = format!("{} Hanks 0 ", "A".repeat(20));
        assert_eq!(tracker.read(&field(&near)), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("Hanks 0 ")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("Hex 0 ")), Alignment::NotFound);
    }

    #[test]
    fn an_old_copy_beside_the_new_one_is_never_told_from_it() {
        // The old copy is brought near the anchor by a deletion (ignored: it is
        // the old one), and then the paste arrives beside it.
        let before = "Hanks 0 abcdefghij";
        let mut tracker = Tracker::new(
            "Hanks".into(),
            Some(Anchor {
                start: utf16_len(before),
            }),
            Some(before),
        );
        assert_eq!(tracker.read(&field("Hanks 0 ")), Alignment::Unchanged);
        assert_eq!(tracker.read(&field("Hanks 0 Hanks")), Alignment::Unchanged);
        assert!(!matches!(
            tracker.read(&field("Hanks 0 Hex")),
            Alignment::Edited(_)
        ));
    }

    #[test]
    fn a_line_break_typed_where_it_could_have_gone_either_side_closes_the_edge() {
        // The last word backspaced; a line break typed beside the one already
        // there (before or after it, the text is the same); then a word typed
        // before the break: outside, since the edge is not known to be open.
        let a = answers(
            "ok. send it to Kip\nnext",
            "send it to Kip",
            &[
                "ok. send it to \nnext",
                "ok. send it to \n\nnext",
                "ok. send it to x\n\nnext",
            ],
        );
        assert_eq!(a[2], Alignment::Edited("send it to".into()));
    }

    #[test]
    fn whitespace_typed_beside_the_paste_does_not_fringe_an_edge() {
        // A space typed after the last word, then the word fixed.
        let a = answers(
            "ok. send it to Kip",
            "send it to Kip",
            &["ok. send it to Kip ", "ok. send it to Kit "],
        );
        assert_eq!(a[1], Alignment::Edited("send it to Kit".into()));
        // A space typed before the first word, then the word fixed.
        let a = answers("ok. ab cd", "ab cd", &["ok.  ab cd", "ok.  xy cd"]);
        assert_eq!(a[1], Alignment::Edited("xy cd".into()));
        // Words typed before it on the line above, then the word fixed.
        let a = answers("ok. ab cd", "ab cd", &["ok. Hi\nab cd", "ok. Hi\nxy cd"]);
        assert_eq!(a[1], Alignment::Edited("xy cd".into()));
        // Words typed far before it, past a line break, then the word fixed.
        let a = answers("ok.\nab cd", "ab cd", &["Hi ok.\nab cd", "Hi ok.\nxy cd"]);
        assert_eq!(a[1], Alignment::Edited("xy cd".into()));
        // Words typed right before it do fringe it: the word cannot then be
        // told from them.
        let a = answers("ok. ab cd", "ab cd", &["ok. Hi ab cd", "ok. Hi xy cd"]);
        assert_eq!(a[1], Alignment::NotFound);
    }

    #[test]
    fn a_line_break_typed_at_the_start_edge_closes_it() {
        // "ab" deleted (one of three words), a line break typed where it was,
        // then a word typed after the break: outside the region.
        let a = answers(
            "ok. ab cd ef",
            "ab cd ef",
            &["ok.  cd ef", "ok. \n cd ef", "ok. \nx cd ef"],
        );
        assert_eq!(a[2], Alignment::Edited("cd ef".into()));
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
                    Tracker::new(
                        pasted.into(),
                        Some(anchor_of(&base, pasted)),
                        Some(&format!("{field_text} ")),
                    ),
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
