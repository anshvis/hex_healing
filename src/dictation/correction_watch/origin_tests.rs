//! Leaks by origin. Every character of a generated field carries where it came
//! from (the text before the paste, the paste, or typing), so a stored span is
//! judged by what it is made of, never by where the tracker says it sits: it
//! leaks when every place it could be read from holds a word made only of text
//! that was there before the paste. The fuzzers are the reviewer's, ported;
//! the named tests are the sequences they and the review found.
//!
//! The fuzzers are ignored (`cargo test --release -- --ignored origin_fuzz`);
//! each runs 300,000 cases and must find no leak.

use super::locate::{Alignment, Anchor, Tracker};
use crate::accessibility::FieldText;

fn field(text: &str) -> FieldText {
    FieldText {
        text: text.to_string(),
        windowed: false,
    }
}

/// What the tracker says of each read, for a paste at `anchor` in a field that
/// read `before`.
fn reads(
    paste: &str,
    anchor: Option<usize>,
    before: Option<&str>,
    reads: &[&str],
) -> Vec<Alignment> {
    let mut tracker = Tracker::new(
        paste.to_string(),
        anchor.map(|start| Anchor { start }),
        before,
    );
    reads
        .iter()
        .map(|text| tracker.read(&field(text)))
        .collect()
}

fn edited(text: &str) -> Alignment {
    Alignment::Edited(text.to_string())
}

/// The last read of the sequence says nothing was learned (`NotFound`), and no
/// read says an edit holding `outside`.
fn learns_nothing(paste: &str, anchor: usize, before: &str, sequence: &[&str], outside: &str) {
    let seen = reads(paste, Some(anchor), Some(before), sequence);
    assert_eq!(seen.last(), Some(&Alignment::NotFound), "{sequence:?}");
    for read in seen {
        if let Alignment::Edited(span) = read {
            assert!(!span.contains(outside), "{span:?} holds {outside:?}");
        }
    }
}

// 1. A word that runs on into a neighbour is no longer the paste's.

#[test]
fn a_word_joined_to_the_text_after_the_region_is_not_stored() {
    learns_nothing(
        "send it to Kip",
        4,
        "ok. my secret is hunter2",
        &[
            "ok. send it to Kip my secret is hunter2",
            "ok. send it to Kipmy secret is hunter2",
        ],
        "secret",
    );
    learns_nothing(
        "Hanks",
        0,
        " tail",
        &["Hanks tail", "Hankstail"],
        "Hankstail",
    );
}

#[test]
fn a_word_joined_to_the_text_before_the_region_is_not_stored() {
    learns_nothing(
        "send it to Kip",
        8,
        "hunter2 ",
        &["hunter2 send it to Kip", "hunter2send it to Kip"],
        "hunter2",
    );
    learns_nothing(
        "Hanks",
        8,
        "hunter2 ",
        &["hunter2 Hanks", "hunter2Hanks"],
        "hunter2",
    );
}

// 2. A region of one character cannot slide onto a neighbour.

#[test]
fn a_one_character_region_deleted_with_a_space_does_not_slide_onto_a_neighbour() {
    learns_nothing(
        "a",
        10,
        "great idea",
        &["great idea a", "great idea"],
        "idea",
    );
    learns_nothing("I", 0, "It works", &["I It works", "It works"], "It");
    learns_nothing(
        "buy a",
        0,
        "apple pie",
        &["buy a  apple pie", "buy apple pie"],
        "apple",
    );
}

#[test]
fn a_word_backspaced_to_one_character_and_deleted_stores_nothing() {
    learns_nothing(
        "Hanks",
        7,
        "ok. OH ",
        &[
            "ok. OH Hanks",
            "ok. OH Hank",
            "ok. OH Han",
            "ok. OH Ha",
            "ok. OH H",
            "ok. OH",
        ],
        "OH",
    );
}

// 3. An old copy moves on when text is inserted at its start.

#[test]
fn an_old_copy_pushed_on_by_an_insertion_at_its_start_is_still_old() {
    let seen = reads(
        "Hanks 0",
        Some(18),
        Some("Hanks 0 abcdefghij"),
        &["Hanks 0 ", " Hanks 0 ", " Hex 0 "],
    );
    assert!(
        seen.iter()
            .all(|read| !matches!(read, Alignment::Edited(_))),
        "{seen:?}"
    );
}

// 4. A partial change of a word opens no edge.

#[test]
fn a_part_of_a_word_deleted_does_not_open_its_edge() {
    let seen = reads(
        "Hanks Smith",
        Some(0),
        Some(""),
        &["Hanks Smith", "Hanks", "Hank", "secretHank", "secret Hank"],
    );
    assert_eq!(seen.last(), Some(&Alignment::NotFound), "{seen:?}");
    for read in seen {
        if let Alignment::Edited(span) = read {
            assert!(!span.contains("secret"), "{span:?}");
        }
    }
}

// 5. A replacement that ends in whitespace closes its edge.

#[test]
fn a_replacement_ending_in_whitespace_closes_its_edge() {
    let seen = reads(
        "Hanks Smith",
        Some(0),
        Some(""),
        &["Hanks Smith", "Hanks Rune ", "Rune ", "Rune secret"],
    );
    assert_eq!(seen[1], edited("Hanks Rune"));
    assert_eq!(seen[3], edited("Rune"));
}

// 6. Words put back at an edge are no more than the words taken from it.

#[test]
fn words_put_back_at_an_edge_are_no_more_than_were_taken_there() {
    learns_nothing(
        "one two three four five six",
        4,
        "ok. ",
        &[
            "ok. one two three four five six",
            "ok. three four five six",
            "ok. three four five seven and Sarah",
        ],
        "Sarah",
    );
    learns_nothing(
        "Hanks Smith",
        4,
        "ok. ",
        &["ok. Hanks Smith", "ok. Smith", "ok. Jones and"],
        "and",
    );
}

#[test]
fn the_same_text_gives_the_same_answer_key_by_key_or_in_one_read() {
    let one = reads(
        "one two three four five six",
        Some(4),
        Some("ok. "),
        &[
            "ok. one two three four five six",
            "ok. three four five six",
            "ok. three four five seven and Sarah",
        ],
    );
    let keys = reads(
        "one two three four five six",
        Some(4),
        Some("ok. "),
        &[
            "ok. one two three four five six",
            "ok. three four five six",
            "ok. three four five ",
            "ok. three four five s",
            "ok. three four five se",
            "ok. three four five sev",
            "ok. three four five seve",
            "ok. three four five seven",
            "ok. three four five seven ",
            "ok. three four five seven a",
            "ok. three four five seven an",
            "ok. three four five seven and",
            "ok. three four five seven and ",
            "ok. three four five seven and S",
        ],
    );
    for read in one.iter().chain(&keys) {
        if let Alignment::Edited(span) = read {
            assert!(!span.contains("and"), "{span:?}");
        }
    }
    assert_eq!(one.last(), Some(&Alignment::NotFound));
    assert_eq!(keys.last(), Some(&edited("three four five seven")));
}

// 7. Without the field as it stood, nothing is followed.

#[test]
fn a_field_that_could_not_be_read_before_the_paste_follows_nothing() {
    let seen = reads("Hanks", Some(5), None, &["Hanks", "Hex"]);
    assert!(
        seen.iter()
            .all(|read| !matches!(read, Alignment::Edited(_))),
        "{seen:?}"
    );
    let seen = reads("Hanks", None, None, &["Hanks", "Hex"]);
    assert!(
        seen.iter()
            .all(|read| !matches!(read, Alignment::Edited(_))),
        "{seen:?}"
    );
}

// 8. A change that could as well be typing past the region's separator.

#[test]
fn text_that_could_have_been_typed_past_the_separator_is_not_stored() {
    learns_nothing(
        "Hanks Smith",
        0,
        " tail",
        &["Hanks Smith tail", "Hanks  secret tail"],
        "secret",
    );
    // The mirror: typed before the separator that precedes the region.
    learns_nothing(
        "Hanks Smith",
        5,
        "tail ",
        &["tail Hanks Smith", "tail secret  Smith"],
        "secret",
    );
}

// The fuzzers.

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

    fn seeded(seed: u64) -> Self {
        let mut rng = Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..3 {
            rng.next();
        }
        rng
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Origin {
    Before,
    Paste,
    Typed,
}

type Cells = Vec<(char, Origin)>;

fn text_of(cells: &Cells) -> String {
    cells.iter().map(|(c, _)| *c).collect()
}

fn words_of(cells: &[(char, Origin)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut open = None;
    for (at, (c, _)) in cells.iter().enumerate() {
        match (c.is_whitespace(), open) {
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

/// Whether an edit stored from `cells` leaks: every place `span` could be read
/// from holds a word made only of text from before the paste.
fn leaks(cells: &Cells, alignment: &Alignment) -> bool {
    let Alignment::Edited(span) = alignment else {
        return false;
    };
    let span: Vec<char> = span.chars().collect();
    let text: Vec<char> = cells.iter().map(|(c, _)| *c).collect();
    if span.len() > text.len() {
        return false;
    }
    let mut any = false;
    for start in 0..=text.len() - span.len() {
        if text[start..start + span.len()] != span[..] {
            continue;
        }
        any = true;
        let outside = words_of(&cells[start..start + span.len()])
            .iter()
            .any(|&(a, b)| {
                cells[start + a..start + b]
                    .iter()
                    .all(|(_, origin)| *origin == Origin::Before)
            });
        if !outside {
            return false;
        }
    }
    any
}

fn log_line(cells: &Cells, alignment: &Alignment) -> String {
    let origins: String = cells
        .iter()
        .map(|(_, origin)| match origin {
            Origin::Before => 'b',
            Origin::Paste => 'p',
            Origin::Typed => 't',
        })
        .collect();
    format!("{:?} -> {alignment:?}   [{origins}]", text_of(cells))
}

/// One fuzzed case: reads every change, and says how it leaked, if it did.
struct Watch {
    tracker: Tracker,
    log: Vec<String>,
    leaked: bool,
}

impl Watch {
    fn read(&mut self, cells: &Cells) {
        let alignment = self.tracker.read(&field(&text_of(cells)));
        self.log.push(log_line(cells, &alignment));
        self.leaked |= leaks(cells, &alignment);
    }
}

/// Pastes `paste` at `caret`, with the spaces the app's paste would add.
fn paste_in(cells: &mut Cells, caret: usize, paste: &str) -> usize {
    let mut pasted = String::new();
    if caret > 0 && !cells[caret - 1].0.is_whitespace() {
        pasted.push(' ');
    }
    pasted.push_str(paste);
    if caret < cells.len() && !cells[caret].0.is_whitespace() {
        pasted.push(' ');
    }
    for (k, c) in pasted.chars().enumerate() {
        cells.insert(caret + k, (c, Origin::Paste));
    }
    pasted.chars().count()
}

fn type_in(cells: &mut Cells, at: usize, typed: &str) {
    for (k, c) in typed.chars().enumerate() {
        cells.insert(at + k, (c, Origin::Typed));
    }
}

fn random_string(rng: &mut Rng, alphabet: &[char], longest: usize) -> String {
    (0..1 + rng.below(longest))
        .map(|_| alphabet[rng.below(alphabet.len())])
        .collect()
}

fn finish(name: &str, cases: u64, leaks: &[String]) {
    for case in leaks.iter().take(5) {
        println!("{case}");
    }
    assert!(
        leaks.is_empty(),
        "{name}: {} leaks in {cases} cases",
        leaks.len()
    );
}

/// Tiny alphabets, so that words collide with the text around them: a few
/// letters, spaces and newlines, a paste of a word or two, random changes.
fn letters(mode: &str, cases: u64, first: u64) -> Vec<String> {
    let alphabet: Vec<char> = match mode {
        "ab" => vec!['a', 'b', ' ', 'a', 'b'],
        "abc" => vec!['a', 'b', 'c', ' ', ' '],
        _ => vec!['a', ' ', 'b', '\n'],
    };
    let mut found = Vec::new();
    for seed in first..first + cases {
        let mut rng = Rng::seeded(seed);
        let before_text = random_string(&mut rng, &alphabet, 8);
        let mut cells: Cells = before_text.chars().map(|c| (c, Origin::Before)).collect();
        let caret = rng.below(cells.len() + 1);
        let mut paste = random_string(&mut rng, &['a', 'b', 'c'], 3);
        if rng.below(2) == 0 {
            paste.push(' ');
            paste.push_str(&random_string(&mut rng, &['a', 'b', 'c'], 2));
        }
        let anchored = rng.below(2) == 0;
        let tracker = Tracker::new(
            paste.clone(),
            anchored.then_some(Anchor { start: caret }),
            Some(&before_text),
        );
        paste_in(&mut cells, caret, &paste);
        let mut watch = Watch {
            tracker,
            log: Vec::new(),
            leaked: false,
        };
        watch.read(&cells);
        for _ in 0..1 + rng.below(4) {
            let from = rng.below(cells.len() + 1);
            let to = (from + rng.below(4)).min(cells.len());
            let typed = if rng.below(3) == 0 {
                String::new()
            } else {
                random_string(&mut rng, &alphabet, 4)
            };
            cells.drain(from..to);
            type_in(&mut cells, from, &typed);
            watch.read(&cells);
        }
        if watch.leaked {
            found.push(format!(
                "seed {seed} paste {paste:?} before {before_text:?} caret {caret} anchored {anchored}\n   {}",
                watch.log.join("\n   ")
            ));
        }
    }
    found
}

#[test]
#[ignore = "long: 300,000 cases"]
fn origin_fuzz_letters_a_b() {
    finish("ab", 300_000, &letters("ab", 300_000, 1));
}

#[test]
#[ignore = "long: 300,000 cases"]
fn origin_fuzz_letters_a_b_c() {
    finish("abc", 300_000, &letters("abc", 300_000, 777));
}

#[test]
#[ignore = "long: 300,000 cases"]
fn origin_fuzz_letters_with_newlines() {
    finish("newlines", 300_000, &letters("lines", 300_000, 4_000_001));
}

const BEFORE_WORDS: [&str; 12] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima",
];
const PASTE_WORDS: [&str; 8] = [
    "north", "south", "east", "west", "up", "down", "left", "right",
];
const TYPED_WORDS: [&str; 8] = [
    "red", "green", "blue", "pink", "gray", "gold", "teal", "navy",
];

fn spaced(words: &[&str]) -> Cells {
    let mut cells = Cells::new();
    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            cells.push((' ', Origin::Before));
        }
        cells.extend(word.chars().map(|c| (c, Origin::Before)));
    }
    cells
}

/// Whole words: a field of words, a paste of one to five, and a stream of
/// keystrokes, deletions, selections, replacements and second dictations,
/// aimed at the paste's edges.
fn words(cases: u64, first: u64) -> Vec<String> {
    let mut found = Vec::new();
    for seed in first..first + cases {
        let mut rng = Rng::seeded(seed);
        let before: Vec<&str> = (0..rng.below(5))
            .map(|_| BEFORE_WORDS[rng.below(BEFORE_WORDS.len())])
            .collect();
        let mut base = spaced(&before);
        for at in 0..base.len() {
            if rng.below(40) == 0 && base[at].0 == ' ' {
                base[at].0 = '\n';
            }
        }
        let mut caret = rng.below(base.len() + 1);
        if rng.below(10) != 0 {
            while caret < base.len() && !base[caret].0.is_whitespace() {
                caret += 1;
            }
            caret = (caret + 1).min(base.len());
        }
        let mut paste_words: Vec<&str> = Vec::new();
        while paste_words.len() < 1 + rng.below(5) {
            let word = PASTE_WORDS[rng.below(PASTE_WORDS.len())];
            if !paste_words.contains(&word) {
                paste_words.push(word);
            }
        }
        let paste = paste_words.join(" ");
        if rng.below(8) == 0 && !base.is_empty() {
            // An old copy of the paste, already in the field.
            let mut old = vec![(' ', Origin::Before)];
            old.extend(spaced(&paste_words));
            old.push((' ', Origin::Before));
            let mut at = rng.below(base.len() + 1);
            while at < base.len() && !base[at].0.is_whitespace() {
                at += 1;
            }
            at = (at + 1).min(base.len());
            for (k, cell) in old.into_iter().enumerate() {
                base.insert(at + k, cell);
            }
        }
        let caret = caret.min(base.len());
        let before_text = text_of(&base);
        let drift = if rng.below(4) == 0 { rng.below(5) } else { 2 };
        let anchored = rng.below(3) != 0;
        let anchor = Anchor {
            start: (caret + drift).saturating_sub(2),
        };
        let mut watch = Watch {
            tracker: Tracker::new(
                paste.clone(),
                anchored.then_some(anchor),
                Some(&before_text),
            ),
            log: Vec::new(),
            leaked: false,
        };
        let mut cells = base.clone();
        if rng.below(4) == 0 {
            watch.read(&cells);
        }
        let pasted = paste_in(&mut cells, caret, &paste);
        let mut cursor = caret + pasted;
        watch.read(&cells);
        for _ in 0..1 + rng.below(6) {
            for _ in 0..1 + if rng.below(4) == 0 { rng.below(3) } else { 0 } {
                let pasted_at: Vec<usize> = cells
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, origin))| *origin == Origin::Paste)
                    .map(|(at, _)| at)
                    .collect();
                let (from, to) = match (pasted_at.first(), pasted_at.last()) {
                    (Some(&first), Some(&last)) => (first, last + 1),
                    _ => (cursor, cursor),
                };
                let places = [
                    from,
                    to,
                    cursor,
                    rng.below(cells.len() + 1),
                    from.saturating_sub(1),
                    (to + 1).min(cells.len()),
                ];
                let at = places[rng.below(places.len())].min(cells.len());
                let word = TYPED_WORDS[rng.below(TYPED_WORDS.len())];
                let phrase = if rng.below(3) == 0 {
                    format!("{word} {}", TYPED_WORDS[rng.below(TYPED_WORDS.len())])
                } else {
                    word.to_string()
                };
                match rng.below(14) {
                    0 | 1 => {
                        let c = if rng.below(5) == 0 {
                            ' '
                        } else {
                            char::from(b'a' + u8::try_from(rng.below(26)).unwrap_or(0))
                        };
                        cells.insert(at, (c, Origin::Typed));
                        cursor = at + 1;
                    }
                    2 => {
                        let typed = if rng.below(2) == 0 {
                            format!(" {word}")
                        } else {
                            word.to_string()
                        };
                        type_in(&mut cells, at, &typed);
                        cursor = at + typed.chars().count();
                    }
                    3 | 4 => {
                        let start = at.saturating_sub(1 + rng.below(8));
                        cells.drain(start..at);
                        cursor = start;
                    }
                    5 => {
                        let end = (at + 1 + rng.below(8)).min(cells.len());
                        cells.drain(at..end);
                        cursor = at;
                    }
                    6 | 7 => {
                        let end = (at + 1 + rng.below(10)).min(cells.len());
                        cells.drain(at..end);
                        type_in(&mut cells, at, &phrase);
                        cursor = at + phrase.chars().count();
                    }
                    8 => {
                        if let Some(&(a, b)) =
                            words_of(&cells).iter().find(|&&(a, b)| a <= at && at <= b)
                        {
                            cells.drain(a..b);
                            cursor = a;
                        }
                    }
                    9 => {
                        if let Some(&(a, b)) =
                            words_of(&cells).iter().find(|&&(a, b)| a <= at && at <= b)
                        {
                            cells.drain(a..b);
                            type_in(&mut cells, a, &phrase);
                            cursor = a + phrase.chars().count();
                        }
                    }
                    10 => {
                        cells.insert(at, ('\n', Origin::Typed));
                        cursor = at + 1;
                    }
                    11 => {
                        let typed = if rng.below(2) == 0 {
                            format!(" {paste}")
                        } else {
                            format!("{paste} ")
                        };
                        type_in(&mut cells, at, &typed);
                        cursor = at + typed.chars().count();
                    }
                    12 => {
                        if at > 0 && cells[at - 1].0 == ' ' {
                            cells.remove(at - 1);
                            cursor = at - 1;
                        }
                    }
                    _ => {
                        type_in(&mut cells, at, word);
                        cursor = at + word.chars().count();
                    }
                }
            }
            watch.read(&cells);
        }
        if watch.leaked {
            found.push(format!(
                "seed {seed} paste {paste:?} before {before_text:?} caret {caret} anchored {anchored}\n   {}",
                watch.log.join("\n   ")
            ));
        }
    }
    found
}

#[test]
#[ignore = "long: 300,000 cases"]
fn origin_fuzz_words() {
    finish("words", 300_000, &words(300_000, 1));
}
