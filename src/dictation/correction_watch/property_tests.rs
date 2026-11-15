//! Generated user actions against the watch, with the paste's region tracked
//! by construction. The invariant is to fail closed: what is stored is only
//! what now occupies the place the pasted text occupied (plus at most one
//! token at each edge), never text from before or after it, and a deleted
//! paste stores nothing. Deterministic: every case comes from a seed.
//!
//! Every case runs with the caret known, and with no caret (the field read
//! before the paste instead), and with one more of: the caret a few units
//! off, an app that collapses blank lines, and a field too large to read
//! whole. Edits come a key at a time (a read per key, or per two or three),
//! backspaced and retyped at both edges, typed over a selection, coalesced
//! with typing before or after, wiped and refilled, and pasted twice.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::locate::{Anchor, Tracker};
use super::session::{Event, Field, Outcome, Session, Step};
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
const ODD: [&str; 6] = [
    "see \u{FFFC} attached",
    "ok \u{1F600} great",
    "\u{597d}\u{7684}\u{8c22}\u{8c22}",
    "\u{FFFC}\u{FFFC}",
    "caf\u{e9} \u{1F600}\u{1F600} to the",
    "to the it",
];

/// Pastes of one or two words, with emoji or CJK, and with blank lines.
const SHORT: [&str; 10] = [
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

const WORDS: [&str; 7] = ["Hex", "Rune", "Friday", "banana", "Sarah", "Monday", "Kit"];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[self.below(from.len())]
    }
}

/// The field, and where the paste is in it: a byte range.
#[derive(Clone)]
struct Model {
    text: String,
    from: usize,
    to: usize,
}

impl Model {
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
    SecondSameWords,
    ReturnThenEdit,
    Clear,
    BackspaceRetypeEnd,
    BackspaceRetypeStart,
    SelectTypeEnd,
    SelectTypeStart,
    EditAndAppend,
    PrependAndEdit,
    BlankRefill,
}

const ALL: [Action; 22] = [
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
    Action::SecondSameWords,
    Action::ReturnThenEdit,
    Action::Clear,
    Action::BackspaceRetypeEnd,
    Action::BackspaceRetypeStart,
    Action::SelectTypeEnd,
    Action::SelectTypeStart,
    Action::EditAndAppend,
    Action::PrependAndEdit,
    Action::BlankRefill,
];

/// A word that sounds nearly like `word`: one letter changed.
fn near(word: &str) -> String {
    let mut chars: Vec<char> = word.chars().collect();
    if let Some(last) = chars.last_mut() {
        *last = if *last == 't' { 'p' } else { 't' };
    }
    chars.into_iter().collect()
}

/// What the app stores of the text: nothing changes, or blank lines collapse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Storage {
    Exact,
    CollapsesBlankLines,
}

fn stored(storage: Storage, text: &str) -> String {
    match storage {
        Storage::Exact => text.to_string(),
        Storage::CollapsesBlankLines => text.replace("\n\n", "\n"),
    }
}

/// How a run reaches the field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    /// The caret is known.
    Anchored,
    /// The caret is known, a few units off.
    Drifting(i32),
    /// No caret: the field was read before the paste.
    NoCaret,
    /// The app collapses blank lines (the paste is shorter than inserted).
    Collapsing,
    /// The field is too long to read whole: a window is all there is.
    Windowed,
}

struct Live {
    text: RefCell<String>,
    windowed: bool,
}

impl Field for &Live {
    fn read(&self) -> Result<FieldText, ReadError> {
        Ok(FieldText {
            text: self.text.borrow().clone(),
            windowed: self.windowed,
        })
    }
}

fn u16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// The session under test, over `live`, for a paste at `anchor` (or with no
/// caret, over a field that read `before`).
fn session_for<'a>(
    live: &'a Live,
    paste: &str,
    anchor: Option<Anchor>,
    before: &str,
    t0: Instant,
) -> Session<&'a Live> {
    let before = anchor.is_none().then_some(before);
    Session::new(live, t0, Tracker::new(paste.to_string(), anchor, before))
}

/// The range the stored span may occupy: the region plus one word at each
/// edge, as bytes of the final field.
fn allowed(text: &str, from: usize, to: usize) -> (usize, usize) {
    let start = text[..from]
        .trim_end()
        .rfind(char::is_whitespace)
        .map_or(0, |at| at + 1);
    let after = &text[to..];
    let lead = after.len() - after.trim_start().len();
    let end = after[lead..]
        .find(char::is_whitespace)
        .map_or(text.len(), |at| to + lead + at);
    // A word that touches the region (no space between) is part of it.
    let touching_start = text[..from]
        .rfind(char::is_whitespace)
        .map_or(0, |at| at + 1);
    (start.min(touching_start), end)
}

/// One generated case: the field, the actions, each with the reads it made.
struct Case {
    paste: String,
    initial: Model,
    before: String,
    steps: Vec<(Action, Vec<Model>)>,
}

fn sentence(rng: &mut Rng) -> String {
    match rng.below(10) {
        0 => rng.pick(&ODD).to_string(),
        _ => rng.pick(&CHAT).to_string(),
    }
}

fn generate(seed: u64) -> Case {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let paste = if rng.below(3) == 0 {
        rng.pick(&SHORT).to_string()
    } else {
        rng.pick(&CHAT).to_string()
    };
    let prefix = match rng.below(4) {
        0 => String::new(),
        _ => format!("{} ", sentence(&mut rng)),
    };
    let suffix = match rng.below(4) {
        0 => String::new(),
        _ => format!(" {}", sentence(&mut rng)),
    };
    let text = format!("{prefix}{paste}{suffix}");
    let (from, to) = (prefix.len(), prefix.len() + paste.len());
    let initial = Model { text, from, to };
    let before = format!("{prefix}{suffix}");
    let mut model = initial.clone();
    let mut steps = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let action = ALL[rng.below(ALL.len())];
        let reads = apply(&mut model, action, &mut rng);
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
        steps,
    }
}

/// Keys typed one at a time (or two or three together) at `at`: the models
/// after each, the region as `region` says for the text typed so far.
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

/// Applies `action` to the model, moving the region as the action does by
/// construction. The models after each read it makes; none when it does not
/// apply.
fn apply(model: &mut Model, action: Action, rng: &mut Rng) -> Vec<Model> {
    let words = model.words();
    let one = |model: &Model| vec![model.clone()];
    let replace_word = |model: &mut Model, rng: &mut Rng, k: usize| -> bool {
        let (a, b) = words[k];
        let old = model.text[a..b].to_string();
        let new = if rng.below(2) == 0 && old.chars().count() >= 3 {
            near(&old)
        } else {
            rng.pick(&WORDS).to_string()
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
        Action::EditWord if !words.is_empty() => {
            let k = rng.below(words.len());
            if replace_word(model, rng, k) {
                one(model)
            } else {
                Vec::new()
            }
        }
        Action::EditTwo if words.len() >= 2 => {
            let k = rng.below(words.len() - 1);
            let (a, b) = (words[k].0, words[k + 1].1);
            model.splice(a, b, "Rune Sarah");
            model.to = model.to + "Rune Sarah".len() - (b - a);
            one(model)
        }
        Action::TrailingTwo | Action::TrailingLong if !words.is_empty() => {
            let (a, b) = words[words.len() - 1];
            let new = if action == Action::TrailingTwo {
                "Kit and"
            } else {
                "Kit and Sarah Smith"
            };
            model.splice(a, b, new);
            model.to = model.to + new.len() - (b - a);
            one(model)
        }
        Action::ReplaceAll => {
            let new = sentence(rng);
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
        Action::TypeAfter | Action::SecondDictation => {
            let new = format!(" {}", sentence(rng));
            let to = model.to;
            model.splice(to, to, &new);
            one(model)
        }
        Action::TypeBefore => {
            let new = format!("{} ", sentence(rng));
            let from = model.from;
            model.splice(from, from, &new);
            model.from += new.len();
            model.to += new.len();
            one(model)
        }
        Action::KeysAfter => {
            let keys = format!(" {}", sentence(rng));
            let at = model.to;
            typed(model, at, &keys, rng, |m, _| (m.from, m.to))
        }
        Action::KeysBefore => {
            let keys = format!("{} ", sentence(rng));
            let (at, to) = (model.from, model.to);
            typed(model, at, &keys, rng, move |_, typed| {
                (at + typed, to + typed)
            })
        }
        Action::SecondSameWords => {
            let paste = model.text[model.from..model.to].to_string();
            let to = model.to;
            model.splice(to, to, &format!(" {paste}"));
            one(model)
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
            model.text = sentence(rng);
            (model.from, model.to) = (0, 0);
            vec![blank, model.clone()]
        }
        Action::EditAndAppend if !words.is_empty() => {
            let k = rng.below(words.len());
            if !replace_word(model, rng, k) {
                return Vec::new();
            }
            let new = format!(" {}", sentence(rng));
            let to = model.to;
            model.splice(to, to, &new);
            one(model)
        }
        Action::PrependAndEdit if !words.is_empty() => {
            let k = rng.below(words.len());
            if !replace_word(model, rng, k) {
                return Vec::new();
            }
            let new = format!("{} ", sentence(rng));
            let from = model.from;
            model.splice(from, from, &new);
            model.from += new.len();
            model.to += new.len();
            one(model)
        }
        Action::BackspaceRetypeEnd | Action::SelectTypeEnd if !words.is_empty() => {
            let (a, b) = words[words.len() - 1];
            let old = model.text[a..b].to_string();
            let new = if action == Action::SelectTypeEnd {
                "Kit and Sarah".to_string()
            } else {
                rng.pick(&WORDS).to_string()
            };
            if new == old {
                return Vec::new();
            }
            let mut reads = Vec::new();
            if action == Action::BackspaceRetypeEnd {
                // Backspace the word a key at a time.
                let mut cut = b;
                for c in old.chars().rev() {
                    cut -= c.len_utf8();
                    let len = c.len_utf8();
                    model.splice(cut, cut + len, "");
                    model.to -= len;
                    reads.push(model.clone());
                }
            } else {
                // The first key replaces the selected word.
                let first: String = new.chars().take(1).collect();
                model.splice(a, b, &first);
                model.to = model.to + first.len() - (b - a);
                reads.push(model.clone());
            }
            let skip = if action == Action::SelectTypeEnd {
                1
            } else {
                0
            };
            let rest: String = new.chars().skip(skip).collect();
            let at = model.to;
            reads.extend(typed(model, at, &rest, rng, move |m, typed| {
                (m.from, at + typed)
            }));
            reads
        }
        Action::BackspaceRetypeStart | Action::SelectTypeStart if !words.is_empty() => {
            let (a, b) = words[0];
            let old = model.text[a..b].to_string();
            let new = if action == Action::SelectTypeStart {
                "Kit and Sarah".to_string()
            } else {
                rng.pick(&WORDS).to_string()
            };
            if new == old {
                return Vec::new();
            }
            let mut reads = Vec::new();
            if action == Action::BackspaceRetypeStart {
                let mut cut = b;
                for c in old.chars().rev() {
                    cut -= c.len_utf8();
                    let len = c.len_utf8();
                    model.splice(cut, cut + len, "");
                    model.to -= len;
                    reads.push(model.clone());
                }
            } else {
                let first: String = new.chars().take(1).collect();
                model.splice(a, b, &first);
                model.to = model.to + first.len() - (b - a);
                reads.push(model.clone());
            }
            let skip = if action == Action::SelectTypeStart {
                1
            } else {
                0
            };
            let rest: String = new.chars().skip(skip).collect();
            let at = if action == Action::SelectTypeStart {
                a + 1
            } else {
                a
            };
            let to_before = model.to;
            reads.extend(typed(model, at, &rest, rng, move |m, typed| {
                (m.from, to_before + typed)
            }));
            reads
        }
        _ => Vec::new(),
    }
}

fn at(t0: Instant, ms: u64) -> Instant {
    t0 + Duration::from_millis(ms)
}

/// The observation a run of the case ends with.
struct Run {
    outcome: Outcome,
    /// The field as the app stores it at the end.
    text: String,
    from: usize,
    to: usize,
}

fn run(case: &Case, mode: Mode) -> Run {
    let storage = if mode == Mode::Collapsing {
        Storage::CollapsesBlankLines
    } else {
        Storage::Exact
    };
    let t0 = Instant::now();
    let live = Live {
        text: RefCell::new(stored(storage, &case.initial.text)),
        windowed: mode == Mode::Windowed,
    };
    let start = u16_len(&case.initial.text[..case.initial.from]);
    let anchor = match mode {
        Mode::NoCaret => None,
        Mode::Drifting(k) => Some(Anchor {
            start: (start as i32 + k).max(0) as usize,
            len: u16_len(&case.paste),
        }),
        _ => Some(Anchor {
            start,
            len: u16_len(&case.paste),
        }),
    };
    let mut session = session_for(
        &live,
        &case.paste,
        anchor,
        &stored(storage, &case.before),
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
        text: stored(storage, &last.text),
        from: stored(storage, &last.text[..last.from]).len(),
        to: stored(storage, &last.text[..last.to]).len(),
    };
    let mut ms = 50;
    if let Some(done) = finished(session.step(at(t0, ms), Event::Tick, open)) {
        return end(done);
    }
    let mut after_return = false;
    for (action, reads) in &case.steps {
        for (n, model) in reads.iter().enumerate() {
            // An edit after a Return comes well after it, as a later keystroke.
            ms += if after_return && n == 0 { 1_500 } else { 100 };
            *live.text.borrow_mut() = stored(storage, &model.text);
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

/// Whether `span` sits inside `allowed` somewhere in the final field.
fn inside(text: &str, from: usize, to: usize, span: &str) -> bool {
    let (from, to) = allowed(text, from, to);
    let mut at = 0;
    while let Some(found) = text[at..].find(span) {
        let start = at + found;
        if start >= from && start + span.len() <= to {
            return true;
        }
        at = start + text[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

#[derive(Default, Debug)]
struct Counts {
    cases: usize,
    stored: usize,
    outside: usize,
    deleted_stored: usize,
    good: usize,
    missed: usize,
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
        for c in self.by_mode.values() {
            total.cases += c.cases;
            total.stored += c.stored;
            total.outside += c.outside;
            total.deleted_stored += c.deleted_stored;
            total.good += c.good;
            total.missed += c.missed;
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

fn report(cases: u64) -> Report {
    let mut report = Report::default();
    for seed in 1..=cases {
        let case = generate(seed);
        if case.steps.is_empty() {
            continue;
        }
        let kind = case.steps.last().map(|(a, _)| *a).unwrap();
        for mode in modes(seed) {
            let outcome_run = run(&case, mode);
            let last = case
                .steps
                .last()
                .and_then(|(_, reads)| reads.last())
                .unwrap();
            let describe = || {
                let steps: Vec<_> = case.steps.iter().map(|(a, _)| a).collect();
                format!(
                    "seed {seed} {mode:?}: paste {:?} in {:?}, {steps:?} -> {:?}",
                    case.paste, case.initial.text, last.text
                )
            };
            let by_action = report.by_action.entry(kind).or_default();
            let name = match mode {
                Mode::Drifting(_) => "drifting".to_string(),
                other => format!("{other:?}").to_lowercase(),
            };
            let by_mode = report.by_mode.entry(name).or_default();
            for counts in [by_action, by_mode] {
                counts.cases += 1;
            }
            let mut tally = |f: &dyn Fn(&mut Counts)| {
                f(report.by_action.get_mut(&kind).unwrap());
                let name = match mode {
                    Mode::Drifting(_) => "drifting".to_string(),
                    other => format!("{other:?}").to_lowercase(),
                };
                f(report.by_mode.get_mut(&name).unwrap());
            };
            if let Outcome::Edited(span) = &outcome_run.outcome {
                tally(&|c| c.stored += 1);
                if mode == Mode::Windowed {
                    tally(&|c| c.outside += 1);
                    report
                        .examples
                        .push(format!("{} => {span:?} (windowed)", describe()));
                } else if outcome_run.from == outcome_run.to {
                    tally(&|c| c.deleted_stored += 1);
                    report
                        .examples
                        .push(format!("{} => {span:?} (deleted)", describe()));
                } else if !inside(&outcome_run.text, outcome_run.from, outcome_run.to, span) {
                    tally(&|c| c.outside += 1);
                    report.examples.push(format!("{} => {span:?}", describe()));
                }
            }
            // Known good: one edit, nothing else, to a paste the floor can
            // judge, where the caret is known or the paste is the one new copy.
            if let [(action, reads)] = &case.steps[..]
                && reads.len() == 1
                && matches!(mode, Mode::Anchored | Mode::Drifting(_) | Mode::NoCaret)
            {
                let n = case.initial.words().len();
                let judged = match action {
                    Action::EditWord | Action::TrailingTwo => true,
                    Action::EditTwo => n == 2 || (n >= 4 && two_inside(&case)),
                    _ => false,
                };
                let held_before = mode == Mode::NoCaret
                    && case
                        .before
                        .to_lowercase()
                        .contains(&case.paste.to_lowercase());
                if judged && !held_before {
                    tally(&|c| c.good += 1);
                    let model = &reads[0];
                    let expected = model.text[model.from..model.to].trim().to_string();
                    if outcome_run.outcome != Outcome::Edited(expected.clone()) {
                        tally(&|c| c.missed += 1);
                        report.examples.push(format!(
                            "{} => {:?}, wanted {expected:?}",
                            describe(),
                            outcome_run.outcome
                        ));
                    }
                }
            }
        }
    }
    report
}

#[test]
fn what_is_stored_is_the_pastes_own_region_and_nothing_else() {
    let report = report(4_000);
    let total = report.total();
    println!("total {total:?}");
    for (mode, counts) in &report.by_mode {
        println!("mode {mode}: {counts:?}");
    }
    for (action, counts) in &report.by_action {
        println!("action {action:?}: {counts:?}");
    }
    for line in report.examples.iter().take(10) {
        println!("{line}");
    }
    assert!(total.cases > 9_000, "{}", total.cases);
    assert!(total.good > 300, "{}", total.good);
    assert_eq!(total.outside, 0, "stored outside the region");
    assert_eq!(total.deleted_stored, 0, "stored from a deleted paste");
    assert_eq!(total.missed, 0, "known-good edits missed");
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

#[test]
fn debug_one_seed() {
    let Ok(seed) = std::env::var("SEED") else {
        return;
    };
    let seed: u64 = seed.parse().unwrap();
    let case = generate(seed);
    println!("paste {:?}\ninitial {:?}", case.paste, case.initial.text);
    for (action, reads) in &case.steps {
        println!("{action:?}");
        for read in reads {
            println!(
                "   {:?}  region {:?}",
                read.text,
                &read.text[read.from..read.to]
            );
        }
    }
    let mut tracker = Tracker::new(
        case.paste.clone(),
        Some(Anchor {
            start: u16_len(&case.initial.text[..case.initial.from]),
            len: u16_len(&case.paste),
        }),
        None,
    );
    let mut all = vec![case.initial.clone()];
    all.extend(
        case.steps
            .iter()
            .flat_map(|(_, reads)| reads.iter().cloned()),
    );
    for model in &all {
        let a = tracker.read(&FieldText {
            text: model.text.clone(),
            windowed: false,
        });
        let region = tracker.region();
        let units: Vec<u16> = model.text.encode_utf16().collect();
        let shown = region.map(|r| {
            String::from_utf16_lossy(&units[r.from.min(units.len())..r.to.min(units.len())])
        });
        println!("  {a:?} region {shown:?}");
    }
    for mode in modes(seed) {
        let run = run(&case, mode);
        println!("{mode:?}: {:?}", run.outcome);
    }
}
