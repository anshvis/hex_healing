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
/// A paste of one or two words that grew to more than this many extra words
/// is no longer the paste's place.
const SHORT_PASTE_GROWTH: usize = 4;
/// The most comparisons (inserted tokens x region tokens) worth making.
const MAX_CELLS: usize = 4_000_000;
/// Words of the unchanged text either side of a change that `most_hunks` reads
/// too: the old tail may match inside the new text.
const CONTEXT_WORDS: usize = 32;
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

/// One edge of a region that a change shortened: typing right at it is still
/// the replacement, for `allow` words (the words deleted there, no more). For
/// the end edge `mark` is where the typed text starts; for the start edge,
/// where it ends so far.
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

/// What one hunk of a word diff replaced: `old` words of the read before by
/// `new` words of the read after.
struct Hunk {
    old: std::ops::Range<usize>,
    new: std::ops::Range<usize>,
}

/// The separate word-level hunks that turn `before` into `after`: edits in
/// two places are two changes in one read, however far apart they are. Two
/// when the words are too tangled to work out.
fn hunks(before: &[Token], after: &[Token]) -> Vec<Hunk> {
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
    let shifted = |old: std::ops::Range<usize>, new: std::ops::Range<usize>| Hunk {
        old: prefix + old.start..prefix + old.end,
        new: prefix + new.start..prefix + new.end,
    };
    if b.is_empty() && a.is_empty() {
        return Vec::new();
    }
    if b.is_empty() || a.is_empty() {
        return vec![shifted(0..b.len(), 0..a.len())];
    }
    if b.len().saturating_mul(a.len()) > MAX_HUNK_CELLS {
        return vec![shifted(0..b.len(), 0..a.len()), shifted(0..0, 0..0)];
    }
    let width = a.len() + 1;
    let lcs = lcs_table(b, a);
    let (mut i, mut j) = (0, 0);
    let mut found: Vec<Hunk> = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    while i < b.len() || j < a.len() {
        let same = i < b.len()
            && j < a.len()
            && b[i].fold == a[j].fold
            && lcs[i * width + j] == lcs[(i + 1) * width + j + 1] + 1;
        if same {
            if let Some((i0, j0)) = open.take() {
                found.push(shifted(i0..i, j0..j));
            }
            (i, j) = (i + 1, j + 1);
            continue;
        }
        open.get_or_insert((i, j));
        if j == a.len() || (i < b.len() && lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) {
            i += 1;
        } else {
            j += 1;
        }
    }
    if let Some((i0, j0)) = open {
        found.push(shifted(i0..b.len(), j0..a.len()));
    }
    // One replacement of words that could as well have been two changes (a
    // word edited, then more typed after it that ends as the old words did)
    // is not known to be one.
    if found.len() == 1 && found[0].new.len() > found[0].old.len() {
        // The words either side of the change count too, a few of them: the
        // old tail may match inside the new text instead. (Words swapped for
        // as many are one change however they line up.)
        let (ep, es) = (prefix.min(CONTEXT_WORDS), suffix.min(CONTEXT_WORDS));
        let wide_b = &before[prefix - ep..before.len() - suffix + es];
        let wide_a = &after[prefix - ep..after.len() - suffix + es];
        if most_hunks(wide_b, wide_a) > 1 {
            found.push(shifted(0..0, 0..0));
        }
    }
    found
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

/// The most separate hunks any alignment keeping as many words as possible
/// makes of `b` and `a`.
fn most_hunks(b: &[Token], a: &[Token]) -> usize {
    let width = a.len() + 1;
    let lcs = lcs_table(b, a);
    // most[(i * width + j) * 2 + open]: from b[i..], a[j..], where `open` is
    // whether the step before was already inside a hunk.
    let mut most = vec![0_usize; (b.len() + 1) * width * 2];
    for i in (0..=b.len()).rev() {
        for j in (0..=a.len()).rev() {
            for open in 0..2 {
                let here = lcs[i * width + j];
                let mut best = 0;
                if i < b.len()
                    && j < a.len()
                    && b[i].fold == a[j].fold
                    && here == lcs[(i + 1) * width + j + 1] + 1
                {
                    best = best.max(most[((i + 1) * width + j + 1) * 2]);
                }
                let starts = 1 - open;
                if i < b.len() && here == lcs[(i + 1) * width + j] {
                    best = best.max(starts + most[((i + 1) * width + j) * 2 + 1]);
                }
                if j < a.len() && here == lcs[i * width + j + 1] {
                    best = best.max(starts + most[(i * width + j + 1) * 2 + 1]);
                }
                most[(i * width + j) * 2 + open] = best;
            }
        }
    }
    most[0]
}

/// The indices of the words of `have` that overlap `region`.
fn word_range(have: &[Token], region: Region) -> std::ops::Range<usize> {
    let overlaps = |t: &Token| t.u16_end > region.from && t.u16_start < region.to;
    match (
        have.iter().position(overlaps),
        have.iter().rposition(overlaps),
    ) {
        (Some(first), Some(last)) => first..last + 1,
        _ => 0..0,
    }
}

/// Which edge words a change replaced, and how many words it deleted: what
/// the read of the words says of the change the units say.
#[derive(Clone, Copy, Debug, Default)]
struct Touch {
    start: bool,
    end: bool,
    deleted: usize,
}

/// A region and the edges of it left open: what is followed.
#[derive(Clone, Copy)]
struct Followed {
    region: Region,
    open: Open,
}

impl Followed {
    /// Where the region of `before` is in `after`, from the one change
    /// between them. See the module doc. `None` when it is lost: edits in
    /// more than one place, a change that crosses an edge, could as well sit
    /// on either side of one (text typed before the paste that begins with
    /// what the paste begins with is also an insertion a few characters in),
    /// or grows an edge by more than the words deleted there.
    fn after(
        self,
        (before, before_words): (&str, &[Token]),
        (after, words): (&str, &[Token]),
    ) -> Option<Self> {
        if before == after {
            return Some(self);
        }
        let changes = hunks(before_words, words);
        if changes.len() > 1 {
            return None;
        }
        // At an edge a replacement may put back as many words as it deleted,
        // no more: more is the replacement and what is typed after it, and
        // which is which is not known. (Inside, a replacement may grow.)
        let edge = word_range(before_words, self.region);
        let touch = changes
            .first()
            .filter(|hunk| !hunk.old.is_empty())
            .map(|hunk| Touch {
                start: hunk.old.start <= edge.start && hunk.old.end > edge.start,
                end: hunk.old.end >= edge.end && hunk.old.start < edge.end,
                deleted: hunk.old.len(),
            });
        if let (Some(hunk), Some(touch)) = (changes.first(), touch)
            && hunk.new.len() > hunk.old.len()
            && (touch.start || touch.end)
        {
            return None;
        }
        let diff = Diff::new(before, after);
        // Where no word is left (attachments, emoji), only more keys at an
        // open edge are followed: a change that takes anything away is not
        // known to be the paste's.
        if diff.base.len() > 0 && word_range(before_words, self.region).is_empty() {
            return None;
        }
        let followed = self.placed(&diff, touch.unwrap_or_default())?;
        // The region starts at its first word: whitespace and attachments a
        // coalesced change brought in before it are not the paste's, and text
        // typed before it must land outside. While the start is open, typing
        // goes where the old word was, so it stays where it is.
        let region = followed.region;
        let from = match (followed.open.start, tokens_in(words, region).first()) {
            (None, Some(first)) => region.from.max(first.u16_start),
            _ => region.from,
        };
        Some(Self {
            region: Region { from, ..region },
            open: followed.open,
        })
    }

    /// Where `diff` puts the region, read every way it could have been made.
    fn placed(self, diff: &Diff, touch: Touch) -> Option<Self> {
        let Region { from, to } = self.region;
        let base = diff.base;
        // Typing at an edge a change left open goes on being the replacement,
        // a space either side of the first key notwithstanding.
        let continues = base.len() == 0
            && (self.open.end.is_some() && base.from == to
                || self.open.start.is_some_and(|edge| base.from == edge.mark));
        if continues {
            return self.place(diff, base, touch);
        }
        // Every way the change could have been made: slid earlier, as well.
        let windows: Vec<Change> = diff.slid().collect();
        // A window that crosses an edge only over whitespace deletes nothing of
        // the paste's on that side: the same text gone as "word " or " word".
        let blank = |units: &[u16]| {
            units
                .iter()
                .all(|&u| char::from_u32(u32::from(u)).is_some_and(is_space))
        };
        let crosses_only_blanks = |c: Change| {
            (c.from < from || c.to > to)
                && (c.from >= from || blank(&diff.b[c.from..from.min(c.to)]))
                && (c.to <= to || blank(&diff.b[to.max(c.from)..c.to]))
        };
        let mut placements = Vec::new();
        for window in windows {
            match self.place(diff, window, touch) {
                None if crosses_only_blanks(window) => {}
                placed => placements.push(placed),
            }
        }
        let first = placements.first().copied().flatten()?;
        placements
            .iter()
            .all(|placed| placed.map(|p| p.region) == Some(first.region))
            .then(|| {
                // An edge any reading left open is open: the keys typed there
                // next are the replacement, as far as the words deleted.
                let open = |pick: fn(&Open) -> Option<Edge>| {
                    placements.iter().flatten().find_map(|p| pick(&p.open))
                };
                Self {
                    region: first.region,
                    open: Open {
                        start: open(|o| o.start),
                        end: open(|o| o.end),
                    },
                }
            })
    }

    /// Where one window of the change puts the region.
    fn place(self, diff: &Diff, change: Change, touch: Touch) -> Option<Self> {
        let Region { from, to } = self.region;
        let (a, b) = (&diff.a, &diff.b);
        let new_len = diff.new_len(change);
        let moved = Region {
            from: diff.shift(from),
            to: diff.shift(to),
        };
        let unchanged = |region: Region| Self {
            region,
            open: Open::default(),
        };
        if change.len() == 0 {
            // Pure insertion. Typing at an open edge goes on being inside.
            if let Some(edge) = self.open.end.filter(|_| change.from == to) {
                let new_to = to + new_len;
                // Typing past the words deleted loses the region: whether it
                // is the replacement or what comes after is not known.
                if words_in(&a[edge.mark..new_to]) > edge.allow {
                    return None;
                }
                return Some(Self {
                    region: Region { from, to: new_to },
                    open: Open {
                        start: None,
                        end: Some(edge),
                    },
                });
            }
            if let Some(edge) = self.open.start.filter(|e| change.from == e.mark) {
                let typed_to = change.from + new_len;
                if from > typed_to || words_in(&a[from..typed_to]) > edge.allow {
                    return None;
                }
                return Some(Self {
                    region: Region { from, to: moved.to },
                    open: Open {
                        start: Some(Edge {
                            mark: typed_to,
                            ..edge
                        }),
                        end: None,
                    },
                });
            }
            return Some(unchanged(match change.from {
                at if at <= from => moved,
                at if at >= to => self.region,
                _ => Region { from, to: moved.to },
            }));
        }
        if change.to <= from {
            Some(unchanged(moved))
        } else if change.from >= to {
            Some(unchanged(self.region))
        } else if from <= change.from && change.to <= to {
            let deleted = words_in(&b[change.from..change.to]).max(touch.deleted);
            let at_start = change.from == from || touch.start;
            let at_end = change.to == to || touch.end;
            let shortened = new_len < change.len();
            let allow = |old: Option<Edge>| old.map_or(0, |e| e.allow).max(deleted);
            let new_to = to + new_len - change.len();
            Some(Self {
                region: Region { from, to: new_to },
                open: Open {
                    start: (at_start && shortened).then(|| Edge {
                        allow: allow(self.open.start),
                        mark: from + new_len,
                    }),
                    end: (at_end && shortened).then(|| Edge {
                        allow: allow(self.open.end),
                        mark: new_to - new_len,
                    }),
                },
            })
        } else {
            None
        }
    }
}

/// Where the start of a paste not yet seen is in `after`, given where it was
/// in `before`: text inserted or removed before it moves it, a change at or
/// after it (the paste itself arrives at it) does not. `None` when a change
/// crosses it, or could as well have been made on either side of it.
fn shift_start(before: &str, after: &str, start: usize) -> Option<usize> {
    if before == after {
        return Some(start);
    }
    let diff = Diff::new(before, after);
    let place = |change: Change| {
        if change.len() == 0 {
            Some(if change.from < start {
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
    /// The paste has not been seen where it will be followed. `expect` is
    /// where it should start (moved through every change since the paste),
    /// `known` how many times the field held its text before it (no caret
    /// only), `last` the read before.
    Waiting {
        expect: Option<usize>,
        known: Option<usize>,
        last: Option<String>,
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
        let ambiguous = anchor
            .zip(before_words.as_ref())
            .is_some_and(|(anchor, words)| !near(words, &want, anchor.start).is_empty());
        let state = if ambiguous {
            Tracking::Lost
        } else {
            Tracking::Waiting {
                expect: anchor.map(|anchor| anchor.start),
                known,
                last: before.map(str::to_string),
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
        if is_blank(text) {
            let state = match state {
                Tracking::Waiting { expect, known, .. } => Tracking::Waiting {
                    expect,
                    known,
                    last: None,
                },
                _ => Tracking::Lost,
            };
            return (state, Alignment::NotFound);
        }
        // A window's offsets are not the field's: nothing is followed or
        // landed from one.
        if *windowed {
            return (Tracking::Lost, unchanged_only());
        }
        match state {
            Tracking::Lost => (Tracking::Lost, unchanged_only()),
            Tracking::Waiting {
                expect,
                known,
                last,
            } => self.wait(expect, known, last, text, have),
            Tracking::Following {
                followed,
                text: before,
            } => match followed.after((&before, &tokens(&before)), (text, have)) {
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
        last: Option<String>,
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
        let waiting = |expect| Tracking::Waiting {
            expect,
            known,
            last: Some(text.to_string()),
        };
        let landing = match (expect, known) {
            // The anchor moves with text typed before the paste arrived.
            (Some(start), _) => {
                // The change that brought the paste in is not typing before it,
                // wherever the anchor put it.
                let arrived = |last: &str| arrival(last, text, have, &self.want);
                let Some(start) = last.as_deref().map_or(Some(start), |last| {
                    if arrived(last) {
                        Some(start)
                    } else {
                        shift_start(last, text, start)
                    }
                }) else {
                    return (Tracking::Lost, unchanged_only(&found));
                };
                match near(have, &self.want, start).as_slice() {
                    [] => return (waiting(Some(start)), unchanged_only(&found)),
                    [only] => *only,
                    // Two copies within reach: which is the paste?
                    _ => return (Tracking::Lost, Alignment::Unchanged),
                }
            }
            (None, Some(0)) if found.len() == 1 => found[0],
            (None, _) if found.is_empty() => return (waiting(None), Alignment::NotFound),
            // No caret: the text twice, or held before, or unknown.
            (None, _) => return (Tracking::Lost, Alignment::Unchanged),
        };
        let region = Region {
            from: have[landing].u16_start,
            to: have[landing + self.want.len() - 1].u16_end,
        };
        let state = Tracking::Following {
            followed: Followed {
                region,
                open: Open::default(),
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
        // Key by key: past the words deleted the region is lost, not cut short.
        let keys = typing("send it to ", "Kit and Sarah");
        let mut reads = vec!["send it to Ki", "send it to K", "send it to "];
        reads.extend(refs(&keys));
        assert_eq!(last(base, base, &reads), Alignment::NotFound);
        // Two words typed over two words are stored.
        let base = "report. ok thanks";
        let keys = typing("report. ", "yes sure");
        assert_eq!(edited(base, "ok thanks", &refs(&keys)), "yes sure");
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
        // One word selected and one typed over it.
        let base = "please send it to Kip tomorrow";
        let keys = typing("please send it to Kip ", "Friday");
        assert_eq!(
            edited(base, "please send it to Kip tomorrow", &refs(&keys)),
            "please send it to Kip Friday"
        );
        // Three typed over one: lost, however far it is typed.
        let keys = typing("please send it to Kip ", "Kit and Sarah");
        assert_eq!(
            last(base, "please send it to Kip tomorrow", &refs(&keys)),
            Alignment::NotFound
        );
        // Two typed over two.
        let base = "report. ok thanks";
        let keys = typing("report. ", "yes sure");
        assert_eq!(edited(base, "ok thanks", &refs(&keys)), "yes sure");
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
        let mut reads = vec![format!("ok. Friday {}", tail.trim_start())];
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
            None,
        );
        assert_eq!(tracker.read(&field(base)), Alignment::Unchanged);
        assert_eq!(
            tracker.read(&field("ok. alpha beta\ngamma Rune")),
            Alignment::Edited("alpha beta\ngamma Rune".into())
        );
        // Typing on past the word it shortened is the replacement or what
        // comes after it, and which is not known.
        assert_eq!(
            tracker.read(&field("ok. alpha beta\ngamma Rune typed after text")),
            Alignment::NotFound
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
        let mut tracker = Tracker::new("send it to Kip".into(), Some(Anchor { start: 19 }), None);
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
        let mut tracker = Tracker::new("send it to Kip".into(), Some(Anchor { start: 500 }), None);
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
            let got = Followed { region, open }
                .after((before, &tokens(before)), (after, &tokens(after)))
                .map(|f| (f.region.from, f.region.to));
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
