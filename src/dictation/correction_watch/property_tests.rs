//! Generated user actions against the watch, with the paste's region tracked
//! by construction. The invariant is to fail closed: what is stored is only
//! what now occupies the place the pasted text occupied, never text from
//! before or after it, and a deleted paste stores nothing. Deterministic:
//! every case comes from a seed.
//!
//! The oracle is by provenance, with no slack: every word of a stored span
//! must lie wholly inside the model's region (a word that merely touches it,
//! or is joined to the text beside it, is not inside), and the unique-word
//! cases make every word's place exact. Edge
//! edits that come a key at a time must store exactly the region or nothing.
//!
//! Every case runs with the caret known, and with no caret, and with one more
//! of: the caret a few units off, an app that collapses blank lines, and a
//! field too large to read whole. A field that grows past what a read takes
//! turns to windows mid-run. Some cases paste with no separator from the text
//! on one side or both (`Setup::Glued`), and some paste a repeated or periodic
//! text beside the same words (`Setup::Periodic`).
//!
//! The one waiver is a leak that first shows at a typed line break in an app
//! that collapses blank lines (`waivable_step`); a deleted paste's is never
//! waived.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::locate::{Alignment, Anchor, Tracker};
use super::session::{Event, Field, Outcome, READ_CHARS, Session, Step};
use crate::accessibility::{FieldText, ReadError};

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

/// Text around a paste that is not plain English words.
const ODD: [&str; 14] = [
    "great idea",
    "It works",
    "apple pie",
    "ok. my secret is hunter2",
    "tail",
    "I a",
    "a",
    "I",
    "see \u{FFFC} attached",
    "ok \u{1F600} great",
    "\u{597d}\u{7684}\u{8c22}\u{8c22}",
    "\u{FFFC}\u{FFFC}",
    "caf\u{e9} \u{1F600}\u{1F600} to the",
    "to the it",
];

/// Pastes of one or two words, with emoji or CJK, and with blank lines; a one
/// character word or region, and words that share a first or last letter with
/// the text around them.
const SHORT: [&str; 20] = [
    "a",
    "I",
    "buy a",
    "I a",
    "Hanks tail",
    "idea",
    "It",
    "a idea",
    "send it to Kip",
    "Kip kip",
    "Hanks",
    "Yes",
    "Kip",
    "Hanks Smith",
    "go home",
    "Kip tomorrow",
    "report",
    "ok thanks",
    "ok \u{1F600} thanks",
    "alpha beta\n\ngamma delta",
];

const WORDS: [&str; 12] = [
    "Hex", "Rune", "Friday", "banana", "Sarah", "Monday", "Kit", "a", "I", "idea", "Hanks", "tail",
];

/// Words no two of which are alike (never one letter apart), so that every
/// word in a case has a place of its own.
fn pool() -> &'static [String] {
    static POOL: OnceLock<Vec<String>> = OnceLock::new();
    POOL.get_or_init(|| {
        let mut rng = Rng(0x5EED_1234_ABCD_9876);
        let mut words: Vec<String> = Vec::new();
        while words.len() < 600 {
            let len = 5 + rng.below(3);
            let word: String = (0..len)
                .map(|_| char::from(b'a' + u8::try_from(rng.below(26)).unwrap_or(0)))
                .collect();
            let alike = |other: &String| {
                other.len().abs_diff(word.len()) <= 1
                    && (other
                        .chars()
                        .zip(word.chars())
                        .filter(|(a, b)| a != b)
                        .count()
                        <= 1
                        || other.starts_with(&word[..word.len() - 1])
                        || word.starts_with(&other[..other.len() - 1]))
            };
            if !words.iter().any(alike) {
                words.push(word);
            }
        }
        words
    })
}

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

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[self.below(from.len())]
    }
}

/// Where a case gets its words: a corpus (words repeat), or a pool where no
/// word is used twice.
struct Gen {
    rng: Rng,
    unique: Option<Vec<usize>>,
}

impl Gen {
    fn new(seed: u64) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let unique = (rng.below(2) == 0).then(|| {
            let mut order: Vec<usize> = (0..pool().len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
            order
        });
        Self { rng, unique }
    }

    /// `n` words, each used once in a unique case.
    fn words(&mut self, n: usize) -> String {
        match &mut self.unique {
            Some(order) => (0..n)
                .filter_map(|_| order.pop())
                .map(|at| pool()[at].as_str())
                .collect::<Vec<_>>()
                .join(" "),
            None => self.rng.pick(&CHAT).to_string(),
        }
    }

    fn sentence(&mut self) -> String {
        match self.unique {
            Some(_) => {
                let n = 2 + self.rng.below(5);
                self.words(n)
            }
            None if self.rng.below(5) == 0 => self.rng.pick(&ODD).to_string(),
            None => self.words(0),
        }
    }

    fn word(&mut self) -> String {
        match self.unique {
            Some(_) => self.words(1),
            None => self.rng.pick(&WORDS).to_string(),
        }
    }

    /// Words to type over a word: more than one in a unique case.
    fn phrase(&mut self, n: usize) -> String {
        match self.unique {
            Some(_) => self.words(n),
            None => ["Kit", "Kit and", "Kit and Sarah"][n.min(3) - 1].to_string(),
        }
    }
}

/// The field, and where the paste is in it: a byte range.
#[derive(Clone, Debug)]
struct Model {
    text: String,
    from: usize,
    to: usize,
}

impl Model {
    /// Where typing after the region goes: past a newline a Return typed
    /// there, as the caret moved on with it.
    fn after_region(&self) -> usize {
        self.to
            + self.text[self.to..]
                .bytes()
                .take_while(|&b| b == b'\n')
                .count()
    }

    fn splice(&mut self, a: usize, b: usize, with: &str) {
        self.text.replace_range(a..b, with);
    }

    /// The words (byte ranges) inside the region.
    fn words(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut open = None;
        for (at, c) in self.text[self.from..self.to].char_indices() {
            match (c.is_whitespace(), open) {
                (false, None) => open = Some(at),
                (true, Some(start)) => {
                    out.push((self.from + start, self.from + at));
                    open = None;
                }
                _ => {}
            }
        }
        if let Some(start) = open {
            out.push((self.from + start, self.to));
        }
        out
    }

    fn region(&self) -> &str {
        &self.text[self.from..self.to]
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Action {
    EditWord,
    EditTwo,
    TrailingTwo,
    TrailingLong,
    ReplaceAll,
    DeletePaste,
    DeleteBefore,
    TypeAfter,
    TypeBefore,
    KeysAfter,
    KeysBefore,
    SecondDictation,
    PasteBefore,
    ReturnThenEdit,
    Clear,
    BackspaceRetypeEnd,
    BackspaceRetypeStart,
    SelectTypeEnd,
    SelectTypeStart,
    SelectTypeOneEnd,
    SelectTypeOneStart,
    EditAndAppend,
    PrependAndEdit,
    BlankRefill,
    Ime,
    Autocorrect,
    UndoRedo,
    MultiHunk,
    AppendWord,
    Grow,
    ForwardDeleteStart,
    WsThenInsertBefore,
    OneCharEdge,
    SpaceMerge,
    MergeDeleteTyped,
    TailCutThenSpace,
}

const ALL: [Action; 36] = [
    Action::EditWord,
    Action::EditTwo,
    Action::TrailingTwo,
    Action::TrailingLong,
    Action::ReplaceAll,
    Action::DeletePaste,
    Action::DeleteBefore,
    Action::TypeAfter,
    Action::TypeBefore,
    Action::KeysAfter,
    Action::KeysBefore,
    Action::SecondDictation,
    Action::PasteBefore,
    Action::ReturnThenEdit,
    Action::Clear,
    Action::BackspaceRetypeEnd,
    Action::BackspaceRetypeStart,
    Action::SelectTypeEnd,
    Action::SelectTypeStart,
    Action::SelectTypeOneEnd,
    Action::SelectTypeOneStart,
    Action::EditAndAppend,
    Action::PrependAndEdit,
    Action::BlankRefill,
    Action::Ime,
    Action::Autocorrect,
    Action::UndoRedo,
    Action::MultiHunk,
    Action::AppendWord,
    Action::Grow,
    Action::ForwardDeleteStart,
    Action::WsThenInsertBefore,
    Action::OneCharEdge,
    Action::SpaceMerge,
    Action::MergeDeleteTyped,
    Action::TailCutThenSpace,
];

/// A word that sounds nearly like `word`: one letter changed.
fn near(word: &str) -> String {
    let mut chars: Vec<char> = word.chars().collect();
    if let Some(last) = chars.last_mut() {
        *last = if *last == 't' { 'p' } else { 't' };
    }
    chars.into_iter().collect()
}

/// How a run reaches the field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    /// The caret is known.
    Anchored,
    /// The caret is known, a few units off.
    Drifting(isize),
    /// No caret: the field was read before the paste.
    NoCaret,
    /// The app collapses blank lines (the paste is shorter than inserted).
    Collapsing,
    /// The field is too long to read whole: a window is all there is.
    Windowed,
}

/// What happens before the first action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Setup {
    Normal,
    /// The paste is edited before the first read ever sees it whole.
    EditedBeforeSeen,
    /// The field is empty until the paste arrives.
    StartsEmpty,
    /// The paste's own words sit right next to it already.
    DuplicateShort,
    /// The paste is put in with no separator from the text on one side or
    /// both: its edge words run into the words beside it.
    Glued,
    /// A repeated (or periodic) paste with the same words beside it, sometimes
    /// glued: the same text can be read as inserted at more than one place.
    Periodic,
}

struct Live {
    text: RefCell<String>,
    windowed: bool,
}

impl Field for &Live {
    fn read(&self) -> Result<FieldText, ReadError> {
        let text = self.text.borrow().clone();
        // A field that grows past what a read takes is read as a window.
        let windowed = self.windowed || text.len() > READ_CHARS;
        Ok(FieldText { text, windowed })
    }
}

fn u16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// The session under test, over `live`, for a paste at `anchor` (or with no
/// caret), over a field that read `before` before the paste.
fn session_for<'a>(
    live: &'a Live,
    paste: &str,
    anchor: Option<Anchor>,
    before: &str,
    t0: Instant,
) -> Session<&'a Live> {
    Session::new(
        live,
        t0,
        Tracker::new(paste.to_string(), anchor, Some(before)),
    )
}

/// One generated case: the field, the actions, each with the reads it made.
#[derive(Clone)]
struct Case {
    paste: String,
    initial: Model,
    before: String,
    setup: Setup,
    unique: bool,
    steps: Vec<(Action, Vec<Model>)>,
}

fn generate(seed: u64) -> Case {
    let mut words_gen = Gen::new(seed);
    let short = words_gen.rng.below(3) == 0;
    let paste = if short {
        match words_gen.unique {
            Some(_) => {
                let n = 1 + words_gen.rng.below(2);
                words_gen.words(n)
            }
            None => words_gen.rng.pick(&SHORT).to_string(),
        }
    } else {
        match words_gen.unique {
            Some(_) => {
                let n = 3 + words_gen.rng.below(5);
                words_gen.words(n)
            }
            None => words_gen.rng.pick(&CHAT).to_string(),
        }
    };
    let setup = match words_gen.rng.below(14) {
        0 => Setup::EditedBeforeSeen,
        1 => Setup::StartsEmpty,
        2 if short => Setup::DuplicateShort,
        3 | 4 => Setup::Glued,
        5 | 6 => Setup::Periodic,
        _ => Setup::Normal,
    };
    let mut paste = paste;
    let (mut prefix, mut suffix) = (String::new(), String::new());
    if setup != Setup::StartsEmpty {
        if words_gen.rng.below(4) != 0 {
            prefix = format!("{} ", words_gen.sentence());
        }
        if words_gen.rng.below(4) != 0 {
            suffix = format!(" {}", words_gen.sentence());
        }
    }
    if setup == Setup::DuplicateShort {
        if words_gen.rng.below(2) == 0 {
            prefix = format!("{paste} ");
        } else {
            suffix = format!(" {paste}");
        }
    }
    if setup == Setup::Glued {
        // At least one side runs straight into the text beside it.
        let sides = 1 + words_gen.rng.below(3);
        prefix = words_gen.sentence();
        suffix = words_gen.sentence();
        if sides & 1 == 0 {
            prefix.push(' ');
        }
        if sides & 2 == 0 {
            suffix.insert(0, ' ');
        }
    }
    if setup == Setup::Periodic {
        let word = words_gen.word();
        let unit = if words_gen.rng.below(2) == 0 {
            word.clone()
        } else {
            format!("{word} {}", words_gen.word())
        };
        let times = |n: usize| vec![unit.as_str(); n].join(" ");
        paste = times(2 + words_gen.rng.below(2));
        let around = |rng: &mut Rng, joined: String, space_first: bool| {
            let joiner = if rng.below(3) == 0 { "" } else { " " };
            if space_first {
                format!("{joiner}{joined}")
            } else {
                format!("{joined}{joiner}")
            }
        };
        let count = words_gen.rng.below(3);
        if count > 0 {
            prefix = around(&mut words_gen.rng, times(count), false);
        }
        let count = words_gen.rng.below(3);
        if count > 0 {
            suffix = around(&mut words_gen.rng, times(count), true);
        }
    }
    let text = format!("{prefix}{paste}{suffix}");
    let (from, to) = (prefix.len(), prefix.len() + paste.len());
    let initial = Model { text, from, to };
    let before = format!("{prefix}{suffix}");
    let mut model = initial.clone();
    let mut steps = Vec::new();
    for _ in 0..1 + words_gen.rng.below(3) {
        let action = ALL[words_gen.rng.below(ALL.len())];
        let reads = apply(&mut model, action, &mut words_gen);
        if !reads.is_empty() {
            steps.push((action, reads));
        }
        if model.from == model.to {
            break;
        }
    }
    Case {
        paste,
        initial,
        before,
        setup,
        unique: words_gen.unique.is_some() && setup != Setup::Periodic,
        steps,
    }
}

/// Keys typed one at a time (or two or three together) at `at`: the models
/// after each, the region as `region` says for the bytes typed so far.
fn typed(
    model: &mut Model,
    at: usize,
    keys: &str,
    rng: &mut Rng,
    region: impl Fn(&Model, usize) -> (usize, usize),
) -> Vec<Model> {
    let chars: Vec<char> = keys.chars().collect();
    let mut reads = Vec::new();
    let (mut done, mut inserted_bytes) = (0, 0);
    while done < chars.len() {
        let take = 1 + rng.below(3) * usize::from(rng.below(3) == 0);
        let chunk: String = chars[done..(done + take).min(chars.len())].iter().collect();
        done += chunk.chars().count();
        model.splice(at + inserted_bytes, at + inserted_bytes, &chunk);
        inserted_bytes += chunk.len();
        (model.from, model.to) = region(model, inserted_bytes);
        reads.push(model.clone());
    }
    reads
}

/// Which word of the region, and which edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Edge {
    Start,
    End,
}

/// A word of the region replaced by `new`, typed from `at` after the word is
/// first deleted (a backspace a key at a time, or one key over a selection).
fn rewrite_edge(
    model: &mut Model,
    edge: Edge,
    new: &str,
    backspace: bool,
    rng: &mut Rng,
) -> Vec<Model> {
    let words = model.words();
    let Some(&(a, b)) = (match edge {
        Edge::Start => words.first(),
        Edge::End => words.last(),
    }) else {
        return Vec::new();
    };
    if model.text[a..b] == *new {
        return Vec::new();
    }
    let mut reads = Vec::new();
    if backspace {
        // Backspace from after the word, a key at a time, leaving the space.
        let mut cut = b;
        for c in model.text[a..b].chars().rev().collect::<Vec<_>>() {
            cut -= c.len_utf8();
            model.splice(cut, cut + c.len_utf8(), "");
            model.to -= c.len_utf8();
            reads.push(model.clone());
        }
    } else {
        // The first key replaces the selected word.
        let first: String = new.chars().take(1).collect();
        model.splice(a, b, &first);
        model.to = model.to + first.len() - (b - a);
        reads.push(model.clone());
    }
    let rest: String = new.chars().skip(usize::from(!backspace)).collect();
    let at = if backspace {
        a
    } else {
        a + new.chars().next().map_or(0, char::len_utf8)
    };
    let to_before = model.to;
    reads.extend(typed(model, at, &rest, rng, move |m, typed| match edge {
        Edge::End => (m.from, at + typed),
        Edge::Start => (m.from, to_before + typed),
    }));
    reads
}

/// Applies `action` to the model, moving the region as the action does by
/// construction. The models after each read it makes; none when it does not
/// apply.
fn apply(model: &mut Model, action: Action, words_gen: &mut Gen) -> Vec<Model> {
    let words = model.words();
    let one = |model: &Model| vec![model.clone()];
    let replace_word = |model: &mut Model, words_gen: &mut Gen, k: usize| -> bool {
        let (a, b) = words[k];
        let old = model.text[a..b].to_string();
        let new = if words_gen.rng.below(2) == 0 && old.chars().count() >= 3 {
            near(&old)
        } else {
            words_gen.word()
        };
        if new == old {
            return false;
        }
        model.splice(a, b, &new);
        model.to = model.to + new.len() - old.len();
        true
    };
    match action {
        // The newline a Return typed; the edit comes later, as its own action.
        Action::ReturnThenEdit => {
            let to = model.to;
            model.splice(to, to, "\n");
            one(model)
        }
        Action::EditWord | Action::Autocorrect if !words.is_empty() => {
            let k = words_gen.rng.below(words.len());
            if replace_word(model, words_gen, k) {
                one(model)
            } else {
                Vec::new()
            }
        }
        Action::EditTwo if words.len() >= 2 => {
            let k = words_gen.rng.below(words.len() - 1);
            let (a, b) = (words[k].0, words[k + 1].1);
            let new = words_gen.phrase(2);
            model.splice(a, b, &new);
            model.to = model.to + new.len() - (b - a);
            one(model)
        }
        Action::TrailingTwo | Action::TrailingLong if !words.is_empty() => {
            let (a, b) = words[words.len() - 1];
            let new = words_gen.phrase(if action == Action::TrailingTwo { 2 } else { 4 });
            model.splice(a, b, &new);
            model.to = model.to + new.len() - (b - a);
            one(model)
        }
        Action::ReplaceAll => {
            let new = words_gen.sentence();
            let (from, to) = (model.from, model.to);
            model.splice(from, to, &new);
            model.to = from + new.len();
            one(model)
        }
        Action::DeletePaste => {
            let (from, to) = (model.from, model.to);
            model.splice(from, to, "");
            model.to = from;
            one(model)
        }
        Action::DeleteBefore => {
            let (from, to) = (model.from, model.to);
            let start = model.text[..from]
                .trim_end()
                .rfind(char::is_whitespace)
                .map_or(0, |at| at + 1);
            model.splice(start, to, "");
            (model.from, model.to) = (start, start);
            one(model)
        }
        Action::TypeAfter => {
            let new = format!(" {}", words_gen.sentence());
            let to = model.after_region();
            model.splice(to, to, &new);
            one(model)
        }
        Action::AppendWord => {
            // One word right at the end: spaced, or glued to the last word.
            let glue = if words_gen.rng.below(2) == 0 { "" } else { " " };
            let new = format!("{glue}{}", words_gen.word());
            let to = model.after_region();
            model.splice(to, to, &new);
            if glue.is_empty() && to == model.to {
                // Glued to the last word, it is that word, longer.
                model.to += new.len();
            }
            one(model)
        }
        Action::SecondDictation => {
            // The same words dictated again, after.
            let again = format!(" {}", model.region());
            let to = model.after_region();
            model.splice(to, to, &again);
            one(model)
        }
        Action::PasteBefore => {
            // The same words again, before it.
            let again = format!("{} ", model.region());
            let from = model.from;
            model.splice(from, from, &again);
            model.from += again.len();
            model.to += again.len();
            one(model)
        }
        Action::TypeBefore => {
            let new = format!("{} ", words_gen.sentence());
            let from = model.from;
            model.splice(from, from, &new);
            model.from += new.len();
            model.to += new.len();
            one(model)
        }
        Action::KeysAfter => {
            let keys = format!(" {}", words_gen.sentence());
            let at = model.after_region();
            typed(model, at, &keys, &mut words_gen.rng, |m, _| (m.from, m.to))
        }
        Action::KeysBefore => {
            let keys = format!("{} ", words_gen.sentence());
            let (at, to) = (model.from, model.to);
            typed(model, at, &keys, &mut words_gen.rng, move |_, typed| {
                (at + typed, to + typed)
            })
        }
        Action::Clear => {
            model.text.clear();
            (model.from, model.to) = (0, 0);
            one(model)
        }
        Action::BlankRefill => {
            let blank = Model {
                text: String::new(),
                from: 0,
                to: 0,
            };
            model.text = words_gen.sentence();
            (model.from, model.to) = (0, 0);
            vec![blank, model.clone()]
        }
        Action::EditAndAppend if !words.is_empty() => {
            let k = words_gen.rng.below(words.len());
            if !replace_word(model, words_gen, k) {
                return Vec::new();
            }
            let new = format!(" {}", words_gen.sentence());
            let to = model.to;
            model.splice(to, to, &new);
            one(model)
        }
        Action::PrependAndEdit if !words.is_empty() => {
            let k = words_gen.rng.below(words.len());
            if !replace_word(model, words_gen, k) {
                return Vec::new();
            }
            let new = format!("{} ", words_gen.sentence());
            let from = model.from;
            model.splice(from, from, &new);
            model.from += new.len();
            model.to += new.len();
            one(model)
        }
        Action::MultiHunk if words.len() >= 3 => {
            // Two words, not next to each other, in one read.
            let first = words_gen.rng.below(words.len() - 2);
            let second = first + 2 + words_gen.rng.below(words.len() - first - 2);
            for k in [second, first] {
                let (a, b) = words[k];
                let new = words_gen.word();
                model.splice(a, b, &new);
                model.to = model.to + new.len() - (b - a);
            }
            one(model)
        }
        Action::UndoRedo if !words.is_empty() => {
            let original = model.clone();
            let k = words_gen.rng.below(words.len());
            if !replace_word(model, words_gen, k) {
                return Vec::new();
            }
            let edited = model.clone();
            vec![edited.clone(), original, edited]
        }
        Action::Ime if !words.is_empty() => {
            // Composition: the word is replaced by a longer and longer
            // composing text, and then by what it settles on.
            let (a, b) = words[words.len() - 1];
            let new = words_gen.word();
            let old = model.text[a..b].to_string();
            if new == old {
                return Vec::new();
            }
            let mut reads = Vec::new();
            let mut end = b;
            for n in 1..=new.chars().count() {
                let composing: String = new.chars().take(n).collect();
                model.splice(a, end, &composing);
                model.to = model.to + composing.len() - (end - a);
                end = a + composing.len();
                reads.push(model.clone());
            }
            reads
        }
        Action::Grow => {
            // More than a read takes, typed after.
            let filler = (0..2_000)
                .map(|n| format!("filler{n}"))
                .collect::<Vec<_>>()
                .join(" ");
            let to = model.to;
            model.splice(to, to, &format!(" {filler}"));
            one(model)
        }
        Action::BackspaceRetypeEnd | Action::BackspaceRetypeStart => {
            let edge = if action == Action::BackspaceRetypeEnd {
                Edge::End
            } else {
                Edge::Start
            };
            let new = words_gen.word();
            rewrite_edge(model, edge, &new, true, &mut words_gen.rng)
        }
        Action::SelectTypeOneEnd | Action::SelectTypeOneStart => {
            let edge = if action == Action::SelectTypeOneEnd {
                Edge::End
            } else {
                Edge::Start
            };
            let new = words_gen.word();
            rewrite_edge(model, edge, &new, false, &mut words_gen.rng)
        }
        Action::SelectTypeEnd | Action::SelectTypeStart => {
            let edge = if action == Action::SelectTypeEnd {
                Edge::End
            } else {
                Edge::Start
            };
            let new = words_gen.phrase(3);
            rewrite_edge(model, edge, &new, false, &mut words_gen.rng)
        }
        Action::ForwardDeleteStart if words.len() >= 2 => {
            // A word typed right before the paste's first word, which is then
            // forward-deleted a key at a time.
            let new = format!("{} ", words_gen.word());
            let from = model.from;
            model.splice(from, from, &new);
            model.from += new.len();
            model.to += new.len();
            let mut reads = vec![model.clone()];
            let (a, b) = model.words()[0];
            for c in model.text[a..=b].chars().collect::<Vec<_>>() {
                model.splice(a, a + c.len_utf8(), "");
                model.to -= c.len_utf8();
                reads.push(model.clone());
            }
            reads
        }
        Action::WsThenInsertBefore if words.len() >= 2 => {
            // The last word backspaced, a space or newline typed after it, then
            // words typed before that whitespace.
            let (a, b) = words[words.len() - 1];
            let mut reads = Vec::new();
            let mut cut = b;
            for c in model.text[a..b].chars().rev().collect::<Vec<_>>() {
                cut -= c.len_utf8();
                model.splice(cut, cut + c.len_utf8(), "");
                model.to -= c.len_utf8();
                reads.push(model.clone());
            }
            let ws = if words_gen.rng.below(2) == 0 {
                "\n"
            } else {
                " "
            };
            let to = model.to;
            model.splice(to, to, ws);
            reads.push(model.clone());
            let more = words_gen.sentence();
            reads.extend(typed(model, to, &more, &mut words_gen.rng, |m, _| {
                (m.from, m.to)
            }));
            reads
        }
        Action::OneCharEdge if !words.is_empty() => {
            // The last word becomes one character; that is typed over by a
            // phrase of two words.
            let (a, b) = words[words.len() - 1];
            model.splice(a, b, "x");
            model.to = model.to + 1 - (b - a);
            let mut reads = vec![model.clone()];
            let phrase = words_gen.phrase(2);
            reads.extend(rewrite_edge(
                model,
                Edge::End,
                &phrase,
                false,
                &mut words_gen.rng,
            ));
            reads
        }
        Action::SpaceMerge => {
            // The space between the paste and a word beside it deleted.
            let after = model.text[model.to..].starts_with(' ')
                && model.text[model.to + 1..]
                    .chars()
                    .next()
                    .is_some_and(|c| !c.is_whitespace());
            let before = model.from > 0
                && model.text[..model.from].ends_with(' ')
                && model.text[..model.from - 1]
                    .chars()
                    .next_back()
                    .is_some_and(|c| !c.is_whitespace());
            if after && (words_gen.rng.below(2) == 0 || !before) {
                let to = model.to;
                let rest = &model.text[to + 1..];
                let word = rest.split(char::is_whitespace).next().unwrap_or("").len();
                model.splice(to, to + 1, "");
                model.to += word;
                one(model)
            } else if before {
                let from = model.from;
                let head = &model.text[..from - 1];
                let word = head.rsplit(char::is_whitespace).next().unwrap_or("").len();
                model.splice(from - 1, from, "");
                model.from = from - 1 - word;
                model.to -= 1;
                one(model)
            } else {
                Vec::new()
            }
        }
        Action::MergeDeleteTyped if !words.is_empty() => {
            // The last word's end deleted, then one more character deleted and
            // two typed in one read.
            let (a, b) = words[words.len() - 1];
            if model.text[a..b].chars().count() < 3 {
                return Vec::new();
            }
            let mut reads = Vec::new();
            let last = model.text[a..b]
                .chars()
                .next_back()
                .map_or(1, char::len_utf8);
            model.splice(b - last, b, "");
            model.to -= last;
            reads.push(model.clone());
            let b = b - last;
            let prev = model.text[a..b]
                .chars()
                .next_back()
                .map_or(1, char::len_utf8);
            model.splice(b - prev, b, "QZ");
            model.to = model.to + 2 - prev;
            reads.push(model.clone());
            reads
        }
        Action::TailCutThenSpace if !words.is_empty() => {
            // The last word's tail deleted, then the space after the paste, so
            // what is left runs into the next word.
            let (a, b) = words[words.len() - 1];
            if model.text[a..b].chars().count() < 2
                || !model.text[model.to..].starts_with(' ')
                || model.text[model.to + 1..].trim().is_empty()
            {
                return Vec::new();
            }
            let mut reads = Vec::new();
            let last = model.text[a..b]
                .chars()
                .next_back()
                .map_or(1, char::len_utf8);
            model.splice(b - last, b, "");
            model.to -= last;
            reads.push(model.clone());
            let to = model.to;
            let word = model.text[to + 1..]
                .split(char::is_whitespace)
                .next()
                .unwrap_or("")
                .len();
            model.splice(to, to + 1, "");
            model.to += word;
            reads.push(model.clone());
            reads
        }
        _ => Vec::new(),
    }
}

fn at(t0: Instant, ms: u64) -> Instant {
    t0 + Duration::from_millis(ms)
}

/// What the app stores of the text: blank lines collapse, or nothing changes.
fn stored(collapses: bool, text: &str) -> String {
    if collapses {
        text.replace("\n\n", "\n")
    } else {
        text.to_string()
    }
}

/// The observation a run of the case ends with.
struct Run {
    outcome: Outcome,
    /// The region at the end, in the field as the app stores it.
    from: usize,
    to: usize,
}

fn run(case: &Case, mode: Mode) -> Run {
    let collapses = mode == Mode::Collapsing;
    let t0 = Instant::now();
    let first_read = match (case.setup, case.steps.first()) {
        (Setup::EditedBeforeSeen, Some((_, reads))) => &reads[0],
        _ => &case.initial,
    };
    let live = Live {
        text: RefCell::new(if case.setup == Setup::StartsEmpty {
            String::new()
        } else {
            stored(collapses, &first_read.text)
        }),
        windowed: mode == Mode::Windowed,
    };
    let start = u16_len(&case.initial.text[..case.initial.from]);
    let anchor = match mode {
        Mode::NoCaret => None,
        Mode::Drifting(k) => Some(Anchor {
            start: start.saturating_add_signed(k),
        }),
        _ => Some(Anchor { start }),
    };
    let mut session = session_for(
        &live,
        &case.paste,
        anchor,
        &stored(collapses, &case.before),
        t0,
    );
    let open: &dyn Fn() -> bool = &|| true;
    let finished = |step: Step| match step {
        Step::Done(seen) => Some(seen.outcome),
        _ => None,
    };
    let last = case
        .steps
        .last()
        .and_then(|(_, reads)| reads.last())
        .unwrap_or(&case.initial);
    let end = |outcome: Outcome| Run {
        outcome,
        from: stored(collapses, &last.text[..last.from]).len(),
        to: stored(collapses, &last.text[..last.to]).len(),
    };
    let mut ms = 50;
    if let Some(done) = finished(session.step(at(t0, ms), Event::Tick, open)) {
        return end(done);
    }
    if case.setup == Setup::StartsEmpty {
        ms += 100;
        *live.text.borrow_mut() = stored(collapses, &case.initial.text);
        if let Some(done) = finished(session.step(at(t0, ms), Event::ValueChanged, open)) {
            return end(done);
        }
    }
    let mut after_return = false;
    for (n, (action, reads)) in case.steps.iter().enumerate() {
        for (k, model) in reads.iter().enumerate() {
            // The edit the first read already showed was read at the start.
            if case.setup == Setup::EditedBeforeSeen && n == 0 && k == 0 {
                continue;
            }
            // An edit after a Return comes well after it, as a later keystroke.
            ms += if after_return && k == 0 { 1_500 } else { 100 };
            *live.text.borrow_mut() = stored(collapses, &model.text);
            if *action == Action::ReturnThenEdit
                && let Some(done) = finished(session.step(at(t0, ms), Event::Return, open))
            {
                return end(done);
            }
            if let Some(done) = finished(session.step(at(t0, ms + 50), Event::ValueChanged, open)) {
                return end(done);
            }
        }
        after_return = *action == Action::ReturnThenEdit;
    }
    match session.step(at(t0, ms + 500), Event::FocusOut, open) {
        Step::Done(seen) => end(seen.outcome),
        other => panic!("the closing read ends the watch: {other:?}"),
    }
}

/// The byte ranges of the words of `text`, split as the tracker splits them.
fn words_of(text: &str) -> Vec<(usize, usize)> {
    let space = |c: char| c.is_whitespace() || c == '\u{FFFC}';
    let mut out = Vec::new();
    let mut open = None;
    for (at, c) in text.char_indices() {
        match (space(c), open) {
            (false, None) => open = Some(at),
            (true, Some(start)) => {
                out.push((start, at));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(start) = open {
        out.push((start, text.len()));
    }
    out
}

/// The fewest separate changes that turn the words of `before` into the words
/// of `after`, keeping as many words as can be kept (the same words in order,
/// ignoring case). Written apart from the tracker's own diff.
fn fewest_hunks(before: &str, after: &str) -> usize {
    hunks_between(before, after, false)
}

/// The most, over the same ways of keeping as many words.
fn most_hunks(before: &str, after: &str) -> usize {
    hunks_between(before, after, true)
}

fn hunks_between(before: &str, after: &str, most: bool) -> usize {
    let fold = |text: &str| -> Vec<String> {
        words_of(text)
            .iter()
            .map(|&(a, b)| text[a..b].to_lowercase())
            .collect()
    };
    let (x, y) = (fold(before), fold(after));
    let (n, m) = (x.len(), y.len());
    // best[i][j][c]: (words kept, minus changes, or plus them for the most)
    // for the tails x[i..], y[j..], with c set when the change before them is
    // still open.
    let step = if most { 1 } else { -1 };
    let mut best = vec![vec![[(0i64, 0i64); 2]; m + 1]; n + 1];
    for i in (0..=n).rev() {
        for j in (0..=m).rev() {
            for open in 0..2 {
                let start = |value: (i64, i64)| -> (i64, i64) {
                    if open == 1 {
                        value
                    } else {
                        (value.0, value.1 + step)
                    }
                };
                let mut pick = if i == n && j == m {
                    (0, 0)
                } else {
                    (i64::MIN, 0)
                };
                if i < n {
                    pick = pick.max(start(best[i + 1][j][1]));
                }
                if j < m {
                    pick = pick.max(start(best[i][j + 1][1]));
                }
                if i < n && j < m && x[i] == y[j] {
                    let kept = best[i + 1][j + 1][0];
                    pick = pick.max((kept.0 + 1, kept.1));
                }
                best[i][j][open] = pick;
            }
        }
    }
    usize::try_from(best[0][0][0].1 * step).unwrap_or(0)
}

/// How a stored span leaks, if it does: from a deleted paste, or from outside
/// the region of every read of the field.
fn leak(
    run: &Run,
    every_read: &[&Model],
    span: &str,
    collapses: bool,
    unique: bool,
) -> Option<&'static str> {
    if run.from == run.to {
        return Some("deleted");
    }
    let inside_a_read = every_read.iter().any(|m| {
        inside(
            &stored(collapses, &m.text),
            stored(collapses, &m.text[..m.from]).len(),
            stored(collapses, &m.text[..m.to]).len(),
            span,
            unique,
        )
    });
    (!inside_a_read).then_some("outside")
}

/// A span that stops where the paste's words were used up, with the rest of
/// what the edit typed beyond it: the replacement kept as far as the paste had
/// words, whitespace then outside the region (Known limits in corrections.md).
fn kept_the_fixed_words(span: &str, expected: &str, paste: &str) -> bool {
    expected.starts_with(span)
        && expected[span.len()..].starts_with(char::is_whitespace)
        && words_of(span).len() == words_of(paste.trim()).len()
}

/// Whether step `k` of the case is one whose stored text the words cannot tell
/// from the paste's own, so that a leak that first shows at it is counted in
/// `waived`, not as a leak. A waiver belongs to its step alone: a leak that
/// first shows at any other step is never waived, and a deleted paste's leak
/// is never waived (the caller checks it first).
///
/// The one waiver: an app that collapses blank lines shows a typed line break
/// as nothing at all, so what is typed after it cannot be told from typing at
/// the edge. (The clauses for two changes in one read, an edit before the paste
/// was first read, and keys after a replaced paste were each measured dead over
/// 857,700 cases and removed in round 10.)
fn waivable_step(case: &Case, k: usize, mode: Mode) -> bool {
    mode == Mode::Collapsing && case.steps[k].0 == Action::WsThenInsertBefore
}

/// The case cut after its first `steps` steps.
fn cut(case: &Case, steps: usize) -> Case {
    Case {
        steps: case.steps[..steps].to_vec(),
        ..case.clone()
    }
}

/// How the case's stored text leaks after its first `steps` steps, if it does.
fn leak_after(case: &Case, steps: usize, mode: Mode) -> Option<&'static str> {
    let case = cut(case, steps);
    let run = run(&case, mode);
    let Outcome::Edited(span) = &run.outcome else {
        return None;
    };
    let every_read: Vec<&Model> = std::iter::once(&case.initial)
        .chain(case.steps.iter().flat_map(|(_, reads)| reads))
        .collect();
    leak(
        &run,
        &every_read,
        span,
        mode == Mode::Collapsing,
        case.unique && case.setup != Setup::DuplicateShort,
    )
}

/// Whether the leak the whole case ends with first showed at a step that is
/// waivable: the first step after which the case, cut there, already leaks.
fn leak_is_waived(case: &Case, mode: Mode) -> bool {
    let first = (1..=case.steps.len())
        .find(|&steps| leak_after(case, steps, mode).is_some())
        .unwrap_or(case.steps.len());
    first > 0 && waivable_step(case, first - 1, mode)
}

/// Whether the first or last word of the paste, taken out by an edit, shares
/// an edge character with a word it put in (a word typed over that ends as the
/// new one begins, or the other way about), or a word it took out stands
/// elsewhere in the field as well. The
/// same text can then be read as the edit having been made beside the paste or
/// beside the word that shares the character, so the tracker, which fails
/// closed on it, is not held to storing the edit.
fn edge_ambiguous(before: &str, after: &str, paste: &str) -> bool {
    let fold = |text: &str| -> Vec<String> {
        words_of(text)
            .iter()
            .map(|&(a, b)| text[a..b].to_lowercase())
            .collect()
    };
    let (old, new) = (fold(before), fold(after));
    let count = |words: &[String], word: &String| words.iter().filter(|w| *w == word).count();
    let taken: Vec<&String> = old
        .iter()
        .filter(|w| count(&old, w) > count(&new, w))
        .collect();
    let put: Vec<&String> = new
        .iter()
        .filter(|w| count(&new, w) > count(&old, w))
        .collect();
    let paste = fold(paste);
    taken
        .iter()
        .filter(|t| paste.first() == Some(**t) || paste.last() == Some(**t))
        .any(|t| {
            count(&old, t) > 1
                || put.iter().any(|p| {
                    let (t, p) = (t.as_bytes(), p.as_bytes());
                    t.first() == p.first()
                        || t.last() == p.last()
                        || t.first() == p.last()
                        || t.last() == p.first()
                })
        })
}

/// Whether `span` sits in the final field with every word of it wholly inside
/// the region: a word that only touches it, or runs out of it into the text
/// beside it, is not. With `every` (unique words) all its places must, else
/// one.
fn inside(text: &str, from: usize, to: usize, span: &str, every: bool) -> bool {
    let words = words_of(text);
    let mut ok = Vec::new();
    let mut at = 0;
    while let Some(found) = text[at..].find(span) {
        let (start, end) = (at + found, at + found + span.len());
        let all = words
            .iter()
            .filter(|(a, b)| *a < end && *b > start)
            .all(|(a, b)| *a >= from && *b <= to);
        ok.push(all);
        at = start + text[start..].chars().next().map_or(1, char::len_utf8);
    }
    if every {
        !ok.is_empty() && ok.iter().all(|&all| all)
    } else {
        ok.iter().any(|&all| all)
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct Counts {
    cases: usize,
    stored: usize,
    outside: usize,
    deleted_stored: usize,
    good: usize,
    missed: usize,
    /// Known-good edits lost where `edge_ambiguous` says the text has a second
    /// reading: the price of failing closed on it.
    ambiguous_missed: usize,
    inexact: usize,
    /// Stored text the words could not tell from the paste's own: see
    /// `one_hunk_reading`.
    waived: usize,
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        self.cases += other.cases;
        self.stored += other.stored;
        self.outside += other.outside;
        self.deleted_stored += other.deleted_stored;
        self.good += other.good;
        self.missed += other.missed;
        self.ambiguous_missed += other.ambiguous_missed;
        self.inexact += other.inexact;
        self.waived += other.waived;
    }
}

#[derive(Default)]
struct Report {
    by_action: BTreeMap<Action, Counts>,
    by_mode: BTreeMap<String, Counts>,
    examples: Vec<String>,
}

impl Report {
    fn total(&self) -> Counts {
        let mut total = Counts::default();
        for counts in self.by_mode.values() {
            total.add(counts);
        }
        total
    }
}

/// Whether the two words an `EditTwo` changed are both inside the paste, not
/// at its edge.
fn two_inside(case: &Case) -> bool {
    let before = case.initial.words();
    let model = &case.steps[0].1[0];
    let after = model.words();
    let kept_front = before
        .iter()
        .zip(after.iter())
        .take_while(|(a, b)| case.initial.text[a.0..a.1] == model.text[b.0..b.1])
        .count();
    kept_front >= 1 && before.len() - kept_front >= 3
}

fn modes(seed: u64) -> Vec<Mode> {
    let mut rng = Rng(seed ^ 0xABCD_EF01_2345_6789 | 1);
    let extra = match rng.below(4) {
        0 => Mode::Drifting([-3, -2, -1, 1, 2, 3][rng.below(6)]),
        1 => Mode::Collapsing,
        2 => Mode::Windowed,
        _ => Mode::Drifting([-2, 2][rng.below(2)]),
    };
    vec![Mode::Anchored, Mode::NoCaret, extra]
}

fn mode_name(mode: Mode) -> String {
    match mode {
        Mode::Drifting(_) => "drifting".to_string(),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// Edits that change the paste's words (not just what is around them).
fn rewrites(action: Action) -> bool {
    !matches!(
        action,
        Action::TypeAfter
            | Action::TypeBefore
            | Action::KeysAfter
            | Action::KeysBefore
            | Action::SecondDictation
            | Action::PasteBefore
            | Action::ReturnThenEdit
            | Action::AppendWord
            | Action::Grow
    )
}

fn report(first: u64, last: u64) -> Report {
    let mut report = Report::default();
    for seed in first..=last {
        let case = generate(seed);
        let Some(kind) = case.steps.last().map(|(a, _)| *a) else {
            continue;
        };
        let final_model = case.steps.last().and_then(|(_, r)| r.last()).unwrap();
        // What a span can have come from: any read of the field.
        let every_read: Vec<&Model> = std::iter::once(&case.initial)
            .chain(case.steps.iter().flat_map(|(_, reads)| reads))
            .collect();
        // The paste, whole, in any read (the first included: an edit that
        // happens to put the same words back): a landing like any.
        let later_shows_paste = case
            .steps
            .iter()
            .flat_map(|(_, reads)| reads.iter())
            .any(|m| m.text.to_lowercase().contains(&case.paste.to_lowercase()));
        for mode in modes(seed) {
            let run = run(&case, mode);
            let collapses = mode == Mode::Collapsing;
            let mut mine = Counts {
                cases: 1,
                ..Counts::default()
            };
            let describe = || {
                let steps: Vec<_> = case.steps.iter().map(|(a, _)| a).collect();
                format!(
                    "seed {seed} {mode:?} {:?}: paste {:?} in {:?}, {steps:?} -> {:?}",
                    case.setup, case.paste, case.initial.text, final_model.text
                )
            };
            let mut note = |why: &str, extra: String| {
                report
                    .examples
                    .push(format!("{why}: {} => {extra}", describe()));
            };
            if let Outcome::Edited(span) = &run.outcome {
                mine.stored += 1;
                if mode == Mode::Windowed {
                    mine.outside += 1;
                    note("windowed", format!("{span:?}"));
                } else if let Some(leak) = leak(
                    &run,
                    &every_read,
                    span,
                    collapses,
                    case.unique && case.setup != Setup::DuplicateShort,
                ) {
                    if leak == "deleted" {
                        mine.deleted_stored += 1;
                        note(leak, format!("{span:?}"));
                    } else if leak_is_waived(&case, mode) {
                        mine.waived += 1;
                    } else {
                        mine.outside += 1;
                        note(leak, format!("{span:?}"));
                    }
                } else if case.setup == Setup::EditedBeforeSeen
                    && case
                        .steps
                        .first()
                        .is_some_and(|(a, _)| rewrites(*a) && *a != Action::UndoRedo)
                    && !later_shows_paste
                {
                    mine.outside += 1;
                    note("edit before the paste was seen", format!("{span:?}"));
                }
            }
            let single = case.steps.len() == 1;
            let held_before = case
                .before
                .to_lowercase()
                .contains(&case.paste.to_lowercase());
            // A word typed before the first, then that first word deleted: the
            // edit is of two words, and the region holds only one of them.
            if case.steps.len() == 1
                && kind == Action::ForwardDeleteStart
                && case.setup == Setup::Normal
                && mode != Mode::Windowed
                && matches!(run.outcome, Outcome::Edited(_))
            {
                mine.outside += 1;
                note(
                    "typed beside, then the word deleted",
                    format!("{:?}", run.outcome),
                );
            }
            if single && case.setup != Setup::EditedBeforeSeen {
                let (action, reads) = &case.steps[0];
                let model = &reads[reads.len() - 1];
                let expected = stored(mode == Mode::Collapsing, model.region().trim());
                // A field backspaced to empty is delete-all, not an edit.
                let blanked = reads
                    .iter()
                    .any(|m| m.text.chars().all(|c| c.is_whitespace() || c == '\u{FFFC}'));
                let words_only = case
                    .paste
                    .split_whitespace()
                    .all(|w| w.chars().any(char::is_alphanumeric));
                let landed = words_only
                    && matches!(mode, Mode::Anchored | Mode::Drifting(_) | Mode::NoCaret)
                    && !held_before
                    && !blanked
                    && !matches!(
                        case.setup,
                        Setup::DuplicateShort | Setup::Glued | Setup::Periodic
                    )
                    // A single character that shows up after a read of the
                    // empty field cannot be told from a key typed.
                    && !(case.setup == Setup::StartsEmpty
                        && case.paste.trim().chars().count() < 2);
                let n = case.initial.words().len();
                // One read, one word or two for as many: stored exactly.
                let good_one = reads.len() == 1
                    && fewest_hunks(&case.initial.text, &model.text) == 1
                    && most_hunks(&case.initial.text, &model.text) == 1
                    && match action {
                        Action::EditWord | Action::Autocorrect | Action::UndoRedo => true,
                        Action::EditTwo => n == 2 || (n >= 4 && two_inside(&case)),
                        _ => false,
                    };
                // A word backspaced and retyped, or typed over: the region.
                let edge_one = matches!(
                    action,
                    Action::BackspaceRetypeEnd
                        | Action::BackspaceRetypeStart
                        | Action::SelectTypeOneEnd
                        | Action::SelectTypeOneStart
                        | Action::MergeDeleteTyped
                );
                if landed && (good_one || edge_one) {
                    mine.good += 1;
                    if run.outcome != Outcome::Edited(expected.clone()) {
                        if edge_ambiguous(&case.initial.text, &model.text, &case.paste) {
                            mine.ambiguous_missed += 1;
                        } else {
                            mine.missed += 1;
                            note("missed", format!("{:?}, wanted {expected:?}", run.outcome));
                        }
                    }
                }
                // Typed over a selection with more words than it held, or
                // an edge edit of any kind: the region, or nothing.
                if matches!(
                    action,
                    Action::SelectTypeEnd
                        | Action::SelectTypeStart
                        | Action::BackspaceRetypeEnd
                        | Action::BackspaceRetypeStart
                        | Action::SelectTypeOneEnd
                        | Action::SelectTypeOneStart
                        | Action::OneCharEdge
                        | Action::MergeDeleteTyped
                        | Action::Ime
                ) && let Outcome::Edited(span) = &run.outcome
                    && mode != Mode::Windowed
                    && *span != expected
                    && !kept_the_fixed_words(span, &expected, &case.paste)
                {
                    mine.inexact += 1;
                    note("inexact", format!("{span:?}, wanted {expected:?}"));
                }
                // Two separate edits in one read are two changes: nothing.
                // (Only where the words are plain: emoji and attachments are not
                // words, so an edit around one is one change to a word diff.)
                if matches!(
                    action,
                    Action::MultiHunk | Action::EditAndAppend | Action::PrependAndEdit
                ) && case.initial.text.is_ascii()
                    && model.text.is_ascii()
                    && fewest_hunks(&case.initial.text, &model.text) >= 2
                    && matches!(run.outcome, Outcome::Edited(_))
                {
                    mine.outside += 1;
                    note("two changes in one read", format!("{:?}", run.outcome));
                }
            }
            report.by_action.entry(kind).or_default().add(&mine);
            report
                .by_mode
                .entry(mode_name(mode))
                .or_default()
                .add(&mine);
        }
    }
    report
}

fn print(report: &Report) {
    println!("total {:?}", report.total());
    for (mode, counts) in &report.by_mode {
        println!("mode {mode}: {counts:?}");
    }
    for (action, counts) in &report.by_action {
        println!("action {action:?}: {counts:?}");
    }
    for line in report
        .examples
        .iter()
        .filter(|l| std::env::var("SHOW").map_or(true, |s| l.contains(&s)))
        .take(30)
    {
        println!("{line}");
    }
}

fn check(report: &Report, runs: usize, good: usize) {
    let total = report.total();
    assert!(total.cases >= runs, "{}", total.cases);
    assert!(total.good >= good, "{}", total.good);
    assert_eq!(total.outside, 0, "stored outside the region");
    assert_eq!(total.deleted_stored, 0, "stored from a deleted paste");
    assert_eq!(
        total.inexact, 0,
        "an edge edit stored other than the region"
    );
    assert_eq!(total.missed, 0, "known-good edits missed");
    assert!(
        total.ambiguous_missed * 500 <= total.good,
        "{} of {} known-good edits lost to a second reading",
        total.ambiguous_missed,
        total.good
    );
}

#[test]
fn what_is_stored_is_the_pastes_own_region_and_nothing_else() {
    // The fast subset: it fits nextest's 30 s limit in a debug build.
    let report = report(1, 1_200);
    print(&report);
    check(&report, 2_000, 100);
}

/// The long run: over 800,000 cases. `cargo test -- --ignored`.
#[test]
#[ignore = "long: 290,000 cases in three modes each"]
fn what_is_stored_holds_over_a_long_run() {
    let report = report(1_000_000, 1_290_000);
    print(&report);
    check(&report, 800_000, 30_000);
}

/// Before the paste has been seen whole, nothing is an edit: not anchored,
/// not without a caret.
#[test]
fn nothing_is_an_edit_before_the_paste_was_seen() {
    for mode in [Mode::Anchored, Mode::NoCaret] {
        for paste in ["send it to Kip", "Kip", "Hanks Smith"] {
            let case = Case {
                paste: paste.to_string(),
                initial: Model {
                    text: "secret".to_string(),
                    from: 0,
                    to: 6,
                },
                before: String::new(),
                setup: Setup::Normal,
                unique: false,
                steps: Vec::new(),
            };
            let outcome = run(&case, mode).outcome;
            assert!(
                !matches!(outcome, Outcome::Edited(_)),
                "{paste:?} {mode:?}: {outcome:?}"
            );
        }
    }
}

/// Prints what the tracker made of each read of one seed: `SEED=n cargo test
/// -- --ignored debug_one_seed --nocapture`; `DRIFT=n` moves the caret n units,
/// `NOCARET=1` reads no caret.
#[test]
#[ignore = "a debugging aid"]
fn debug_one_seed() {
    let Ok(seed) = std::env::var("SEED") else {
        return;
    };
    let case = generate(seed.parse().unwrap());
    let drift: isize = std::env::var("DRIFT").map_or(0, |d| d.parse().unwrap_or(0));
    let anchor = Anchor {
        start: u16_len(&case.initial.text[..case.initial.from]).saturating_add_signed(drift),
    };
    let anchor = std::env::var("NOCARET").map_or(Some(anchor), |_| None);
    let mut tracker = Tracker::new(case.paste.clone(), anchor, Some(&case.before));
    // What the field reads, in order: an edit made before the paste was seen
    // is the first read, and an empty field is read first when it starts so.
    let mut all = match case.setup {
        Setup::EditedBeforeSeen => Vec::new(),
        Setup::StartsEmpty => vec![Model {
            text: String::new(),
            from: 0,
            to: 0,
        }],
        _ => vec![case.initial.clone()],
    };
    if case.setup == Setup::StartsEmpty {
        all.push(case.initial.clone());
    }
    all.extend(
        case.steps
            .iter()
            .flat_map(|(_, reads)| reads.iter().cloned()),
    );
    for model in &all {
        let read = FieldText {
            text: model.text.clone(),
            windowed: false,
        };
        println!("{:?} -> {:?}", model.text, tracker.read(&read));
    }
}

#[test]
fn every_action_keeps_its_model_in_bounds() {
    for seed in 1..=3_000 {
        let case = generate(seed);
        for (action, reads) in &case.steps {
            for m in reads {
                assert!(
                    m.from <= m.to
                        && m.to <= m.text.len()
                        && m.text.is_char_boundary(m.from)
                        && m.text.is_char_boundary(m.to),
                    "seed {seed} {action:?}: {m:?}"
                );
            }
        }
    }
}

/// A copy of the paste that was in the field before it, brought near the
/// anchor by deleting what lay between (before the paste is first read), is
/// never taken for the paste: nothing stored comes from it.
#[test]
fn an_old_copy_brought_near_the_anchor_is_never_the_paste() {
    for seed in 1..=2_000_u64 {
        let mut words_gen = Gen::new(seed);
        let (a, b) = (1 + words_gen.rng.below(2), 2 + words_gen.rng.below(4));
        let paste = words_gen.words(a);
        let filler = words_gen.words(b);
        let after = words_gen.word();
        // The old copy, filler, then where the paste will go.
        let before = format!("{paste} {filler} {after} ");
        let anchor = Anchor {
            start: u16_len(&before[..before.len() - after.len() - 1]),
        };
        let mut tracker = Tracker::new(paste.clone(), Some(anchor), Some(&before));
        let read = |tracker: &mut Tracker, text: &str| {
            tracker.read(&FieldText {
                text: text.to_string(),
                windowed: false,
            })
        };
        // The paste arrives, then everything after the old copy is deleted
        // before the first read: the old copy is now where the anchor is.
        assert!(!matches!(
            read(&mut tracker, &format!("{paste} ")),
            Alignment::Edited(_)
        ));
        // Whatever is read of the old copy afterwards is no edit of the paste.
        for later in [
            format!("{} ", words_gen.word()),
            format!("{} tail", words_gen.word()),
        ] {
            let seen = read(&mut tracker, &later);
            assert!(
                !matches!(seen, Alignment::Edited(_)),
                "seed {seed}: {paste:?} -> {later:?}: {seen:?}"
            );
        }
    }
}
