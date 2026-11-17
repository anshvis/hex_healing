//! Cell-origin fuzz of the tracker over fields of unusual words (accents,
//! CJK, emoji, punctuation, attachments), with the paste put in beside
//! neighbours with or without a separator, and keystroke, deletion, rewrite
//! and re-paste edits after it. Every character of the field remembers where
//! it came from (the field before, the paste, or typing); an edit the tracker
//! stores leaks when a word of it is wholly the field's own, or (with `nop`,
//! the paste's words never deleted by the fuzz) wholly typing that lies
//! outside the paste's hull.
//!
//! The brief's `fz4`/`fz5` runs: `unfiltered` counts a leak whether or not the
//! tracker ever landed (a glued paste that is never followed must still never
//! store anything).

use super::locate::{Alignment, Anchor, Tracker};
use crate::accessibility::FieldText;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap_or(0)
    }
}

/// Where a character of the field came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Origin {
    Before,
    Paste,
    Typed,
}

type Cells = Vec<(char, Origin)>;

const BEFORE_WORDS: [&str; 22] = [
    "alpha",
    "bravo",
    "caf\u{e9}",
    "na\u{ef}ve",
    "\u{65e5}\u{672c}\u{8a9e}",
    "stra\u{df}e",
    "\u{130}stanbul",
    "a",
    "I",
    "it's",
    "(hello)",
    "world,",
    "end.",
    "\u{1F600}",
    "ok\u{1F600}",
    "Zo\u{eb}",
    "\u{fb01}nal",
    "x",
    "alpha",
    "alpha",
    "\u{2014}",
    "note:",
];
const PASTE_WORDS: [&str; 14] = [
    "north",
    "south",
    "east",
    "na\u{ef}ve",
    "caf\u{e9}",
    "up",
    "down",
    "it's",
    "a",
    "I",
    "\u{1F600}",
    "Stra\u{df}e",
    "left,",
    "(right)",
];
const TYPED_WORDS: [&str; 16] = [
    "red",
    "green",
    "blue",
    "c\u{f4}t\u{e9}",
    "a",
    "I",
    "north",
    "alpha",
    "\u{1F600}",
    "grey,",
    "(teal)",
    "navy.",
    "\u{df}",
    "\u{f1}and\u{fa}",
    "x",
    "it's",
];

fn is_blank(c: char) -> bool {
    c.is_whitespace() || c == '\u{FFFC}'
}

/// The words of `cells` as `[start, end)` cell ranges.
fn words(cells: &[(char, Origin)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut open = None;
    for (at, (c, _)) in cells.iter().enumerate() {
        match (is_blank(*c), open) {
            (false, None) => open = Some(at),
            (true, Some(start)) => {
                out.push((start, at));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(start) = open {
        out.push((start, cells.len()));
    }
    out
}

fn text(cells: &[(char, Origin)]) -> String {
    cells.iter().map(|cell| cell.0).collect()
}

fn blank(rng: &mut Rng) -> char {
    match rng.below(8) {
        0 => '\n',
        1 => '\t',
        2 => '\u{FFFC}',
        _ => ' ',
    }
}

fn typed_word(rng: &mut Rng) -> String {
    let word = TYPED_WORDS[rng.below(TYPED_WORDS.len())];
    match rng.below(5) {
        0 => format!(" {word}"),
        1 => format!("{word} "),
        2 => format!("{word}\n"),
        _ => word.to_string(),
    }
}

fn put(cells: &mut Cells, at: usize, s: &str) -> usize {
    for (k, c) in s.chars().enumerate() {
        cells.insert(at + k, (c, Origin::Typed));
    }
    at + s.chars().count()
}

#[derive(Clone, Copy)]
struct Options {
    /// The fuzz never deletes the paste's characters, and typing outside the
    /// paste's hull is outside text.
    nop: bool,
    /// Leaks count even when the tracker never landed.
    unfiltered: bool,
    /// Text is typed or deleted between the read before the paste and the
    /// paste arriving.
    pretype: bool,
}

/// Whether a stored `span` of `cells` could only be text that is not the
/// paste's.
fn leaks(cells: &[(char, Origin)], span: &str, options: Options) -> bool {
    let all = text(cells);
    let (span_chars, chars): (Vec<char>, Vec<char>) =
        (span.chars().collect(), all.chars().collect());
    if span_chars.len() > chars.len() {
        return false;
    }
    let hull = {
        let at: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| cell.1 == Origin::Paste && !cell.0.is_whitespace())
            .map(|(at, _)| at)
            .collect();
        at.first().zip(at.last()).map(|(a, b)| (*a, *b))
    };
    let mut found = false;
    for start in 0..=chars.len() - span_chars.len() {
        if chars[start..start + span_chars.len()] != span_chars[..] {
            continue;
        }
        found = true;
        let bad = words(&cells[start..start + span_chars.len()])
            .iter()
            .any(|&(a, b)| {
                let word = &cells[start + a..start + b];
                if word.iter().all(|c| c.1 == Origin::Before) {
                    return true;
                }
                options.nop
                    && hull.is_some_and(|(h0, h1)| {
                        word.iter().all(|c| c.1 == Origin::Typed)
                            && (start + b - 1 < h0 || start + a > h1)
                    })
            });
        if !bad {
            return false;
        }
    }
    found
}

/// One generated case; the description of its leak, if it has one.
fn run_case(seed: u64, options: Options) -> Option<String> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..3 {
        rng.next();
    }
    let mut base: Cells = Vec::new();
    for i in 0..rng.below(6) {
        if i > 0 {
            base.push((blank(&mut rng), Origin::Before));
            if rng.below(8) == 0 {
                base.push((' ', Origin::Before));
            }
        }
        for c in BEFORE_WORDS[rng.below(BEFORE_WORDS.len())].chars() {
            base.push((c, Origin::Before));
        }
    }
    if rng.below(6) == 0 {
        base.push((' ', Origin::Before));
    }
    if rng.below(10) == 0 {
        base.insert(0, (' ', Origin::Before));
    }
    let mut caret = match rng.below(4) {
        0 => 0,
        1 => base.len(),
        _ => rng.below(base.len() + 1),
    };
    let paste_words: Vec<String> = (0..1 + rng.below(4))
        .map(|_| {
            let tokens = words(&base);
            if rng.below(4) == 0 && !tokens.is_empty() {
                let (a, b) = tokens[rng.below(tokens.len())];
                base[a..b].iter().map(|c| c.0).collect()
            } else {
                PASTE_WORDS[rng.below(PASTE_WORDS.len())].to_string()
            }
        })
        .collect();
    let paste = paste_words.join(if rng.below(6) == 0 { "\n" } else { " " });
    let anchored = rng.below(3) != 0;
    let drift = if rng.below(5) == 0 {
        i64::try_from(rng.below(5)).unwrap_or(0) - 2
    } else {
        0
    };
    let before_text = text(&base);
    let pasted_before = rng.below(5) == 0;
    let no_before = rng.below(15) == 0;
    let units = |cells: &[(char, Origin)]| cells.iter().map(|c| c.0.len_utf16()).sum::<usize>();
    let anchor = anchored.then(|| Anchor {
        start: usize::try_from((i64::try_from(units(&base[..caret])).unwrap_or(0) + drift).max(0))
            .unwrap_or(0),
    });
    let mut tracker = Tracker::new(
        paste.clone(),
        anchor,
        if no_before { None } else { Some(&before_text) },
    );
    let mut cells = base.clone();
    let mut log: Vec<String> = Vec::new();
    let mut leaked = false;
    let read = |tracker: &mut Tracker, cells: &Cells, log: &mut Vec<String>, window: bool| {
        let all = text(cells);
        let shown = if window {
            let chars: Vec<char> = all.chars().collect();
            let a = chars.len() / 4;
            let b = (chars.len() * 3 / 4).max(a);
            chars[a..b].iter().collect()
        } else {
            all
        };
        let alignment = tracker.read(&FieldText {
            text: shown.clone(),
            windowed: window,
        });
        log.push(format!(
            "{}{shown:?} -> {alignment:?}",
            if window { "W " } else { "" }
        ));
        match alignment {
            Alignment::Edited(span) if !window => leaks(cells, &span, options),
            _ => false,
        }
    };
    if pasted_before {
        leaked |= read(&mut tracker, &cells, &mut log, false);
    }
    if options.pretype && rng.below(3) == 0 {
        // Typing between the field read and the paste landing.
        for _ in 0..1 + rng.below(2) {
            let q = rng.below(cells.len() + 1);
            if rng.below(3) == 0 && q > 0 {
                let a = q.saturating_sub(1 + rng.below(4));
                cells.drain(a..q);
                if caret >= q {
                    caret -= q - a;
                } else if caret > a {
                    caret = a;
                }
            } else {
                let word = typed_word(&mut rng);
                let end = put(&mut cells, q, &word);
                if q <= caret {
                    caret += end - q;
                }
            }
            if rng.below(2) == 0 {
                leaked |= read(&mut tracker, &cells, &mut log, false);
            }
        }
    }
    // The paste lands, with a space beside it where it would run into a word
    // (most of the time: a third of those run in).
    let mut pasted = String::new();
    let word_before = caret > 0 && !is_blank(cells[caret - 1].0);
    let word_after = caret < cells.len() && !is_blank(cells[caret].0);
    if word_before && rng.below(3) != 0 {
        pasted.push(' ');
    }
    pasted.push_str(&paste);
    if word_after && rng.below(3) != 0 {
        pasted.push(' ');
    }
    for (k, c) in pasted.chars().enumerate() {
        cells.insert(caret + k, (c, Origin::Paste));
    }
    let mut cur = caret + pasted.chars().count();
    leaked |= read(&mut tracker, &cells, &mut log, false);
    let landed = log
        .last()
        .is_some_and(|line| line.ends_with("-> Unchanged"));
    for _ in 0..1 + rng.below(6) {
        let place: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| cell.1 == Origin::Paste)
            .map(|(at, _)| at)
            .collect();
        let (pf, pt) = match (place.first(), place.last()) {
            (Some(a), Some(b)) => (*a, *b + 1),
            _ => (cur, cur),
        };
        let choice = rng.below(16);
        let options_at = [
            pf,
            pt,
            cur,
            rng.below(cells.len() + 1),
            pf.saturating_sub(1),
            (pt + 1).min(cells.len()),
            pf + 1,
            pt.saturating_sub(1),
        ];
        let p = options_at[rng.below(options_at.len())].min(cells.len());
        let may_delete = |cells: &Cells, a: usize, b: usize| {
            !options.nop || cells[a..b].iter().all(|c| c.1 != Origin::Paste)
        };
        match choice {
            0 | 1 => {
                let c = match rng.below(6) {
                    0 => ' ',
                    1 => ',',
                    2 => '.',
                    3 => '\n',
                    4 => '\u{e9}',
                    _ => char::from(b'a' + u8::try_from(rng.below(26)).unwrap_or(0)),
                };
                cells.insert(p, (c, Origin::Typed));
                cur = p + 1;
            }
            2 | 3 => {
                let word = typed_word(&mut rng);
                cur = put(&mut cells, p, &word);
            }
            4 | 5 => {
                let a = p.saturating_sub(1 + rng.below(8));
                if may_delete(&cells, a, p) {
                    cells.drain(a..p);
                    cur = a;
                }
            }
            6 => {
                let b = (p + 1 + rng.below(8)).min(cells.len());
                if may_delete(&cells, p, b) {
                    cells.drain(p..b);
                    cur = p;
                }
            }
            7 | 8 => {
                let b = (p + 1 + rng.below(10)).min(cells.len());
                if may_delete(&cells, p, b) {
                    let word = typed_word(&mut rng);
                    cells.drain(p..b);
                    cur = put(&mut cells, p, &word);
                }
            }
            9 | 10 => {
                let tokens = words(&cells);
                if let Some(&(a, b)) = tokens.iter().find(|&&(a, b)| a <= p && p <= b)
                    && may_delete(&cells, a, b)
                {
                    cells.drain(a..b);
                    cur = a;
                    if choice == 10 {
                        let word = TYPED_WORDS[rng.below(TYPED_WORDS.len())];
                        cur = put(&mut cells, a, word);
                    }
                }
            }
            11 => {
                let c = blank(&mut rng);
                cells.insert(p, (c, Origin::Typed));
                cur = p + 1;
            }
            12 => {
                let again = if rng.below(2) == 0 {
                    format!(" {paste}")
                } else {
                    format!("{paste} ")
                };
                cur = put(&mut cells, p, &again);
            }
            13 => {
                if p > 0 && cells[p - 1].0.is_whitespace() && may_delete(&cells, p - 1, p) {
                    cells.remove(p - 1);
                    cur = p - 1;
                }
            }
            14 => {
                if p < cells.len() && cells[p].0.is_alphabetic() && cells[p].1 != Origin::Paste {
                    let upper: String = cells[p].0.to_uppercase().collect();
                    if let (1, Some(c)) = (upper.chars().count(), upper.chars().next()) {
                        cells[p] = (c, Origin::Typed);
                    }
                }
            }
            _ => {
                let word = TYPED_WORDS[rng.below(TYPED_WORDS.len())];
                cur = put(&mut cells, p, word);
            }
        }
        let window = rng.below(14) == 0;
        leaked |= read(&mut tracker, &cells, &mut log, window);
    }
    (leaked && (landed || options.unfiltered)).then(|| {
        format!(
            "seed {seed}: paste {paste:?} in {before_text:?} caret {caret} \
             anchored {anchored} drift {drift}\n  {}",
            log.join("\n  ")
        )
    })
}

fn fuzz(cases: u64, first: u64, options: Options) {
    let leaks: Vec<String> = (first..first + cases)
        .filter_map(|seed| run_case(seed, options))
        .collect();
    assert!(
        leaks.is_empty(),
        "{} leaks in {cases} cases; first:\n{}",
        leaks.len(),
        leaks.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn a_glued_or_spaced_paste_among_unusual_words_stores_nothing_outside() {
    fuzz(
        20_000,
        1,
        Options {
            nop: true,
            unfiltered: true,
            pretype: false,
        },
    );
    fuzz(
        20_000,
        1,
        Options {
            nop: false,
            unfiltered: true,
            pretype: false,
        },
    );
}

#[test]
fn text_typed_between_the_read_and_the_paste_stores_nothing_outside() {
    fuzz(
        20_000,
        1,
        Options {
            nop: true,
            unfiltered: true,
            pretype: true,
        },
    );
}

#[test]
#[ignore = "long: 300,000 cases per run"]
fn a_long_unfiltered_run_with_the_paste_kept_stores_nothing_outside() {
    fuzz(
        300_000,
        1,
        Options {
            nop: true,
            unfiltered: true,
            pretype: false,
        },
    );
}

#[test]
#[ignore = "long: 300,000 cases per run"]
fn a_long_unfiltered_run_with_the_paste_edited_stores_nothing_outside() {
    fuzz(
        300_000,
        1_000_000,
        Options {
            nop: false,
            unfiltered: true,
            pretype: false,
        },
    );
}

#[test]
#[ignore = "long: 300,000 cases per run"]
fn a_long_run_with_text_typed_before_the_paste_arrives_stores_nothing_outside() {
    fuzz(
        300_000,
        2_000_000,
        Options {
            nop: true,
            unfiltered: true,
            pretype: true,
        },
    );
}
