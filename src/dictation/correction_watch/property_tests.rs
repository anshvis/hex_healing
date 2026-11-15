//! Generated user actions against the watch, with the paste's region tracked
//! by construction. The invariant is to fail closed: what is stored is only
//! what now occupies the place the pasted text occupied (plus at most one
//! token at each edge), never text from before or after it, and a deleted
//! paste stores nothing. Deterministic: every case comes from a seed.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use super::locate::Anchor;
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

const SHORT: [&str; 8] = [
    "Hanks",
    "Yes",
    "Kip",
    "Hanks Smith",
    "go home",
    "Kip tomorrow",
    "report",
    "ok thanks",
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    EditWord,
    EditTwo,
    TrailingTwo,
    ReplaceAll,
    DeletePaste,
    DeleteBefore,
    TypeAfter,
    TypeBefore,
    SecondDictation,
    ReturnThenEdit,
    Clear,
}

const ALL: [Action; 11] = [
    Action::EditWord,
    Action::EditTwo,
    Action::TrailingTwo,
    Action::ReplaceAll,
    Action::DeletePaste,
    Action::DeleteBefore,
    Action::TypeAfter,
    Action::TypeBefore,
    Action::SecondDictation,
    Action::ReturnThenEdit,
    Action::Clear,
];

/// A word that sounds nearly like `word`: one letter changed.
fn near(word: &str) -> String {
    let mut chars: Vec<char> = word.chars().collect();
    if let Some(last) = chars.last_mut() {
        *last = if *last == 't' { 'p' } else { 't' };
    }
    chars.into_iter().collect()
}

struct Live(RefCell<String>);

impl Field for &Live {
    fn read(&self) -> Result<FieldText, ReadError> {
        Ok(field_text(self.0.borrow().clone()))
    }
}

fn field_text(text: String) -> FieldText {
    FieldText {
        text,
        windowed: false,
    }
}

fn u16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn anchor(model: &Model, paste: &str) -> Anchor {
    Anchor {
        start: u16_len(&model.text[..model.from]),
        len: u16_len(paste),
    }
}

/// The range the stored span may occupy: the region plus one word at each
/// edge, as bytes of the final field.
fn allowed(model: &Model) -> (usize, usize) {
    let text = &model.text;
    let start = text[..model.from]
        .trim_end()
        .rfind(char::is_whitespace)
        .map_or(0, |at| at + 1);
    let after = &text[model.to..];
    let lead = after.len() - after.trim_start().len();
    let end = after[lead..]
        .find(char::is_whitespace)
        .map_or(text.len(), |at| model.to + lead + at);
    // A word that touches the region (no space between) is part of it.
    let touching_start = text[..model.from]
        .rfind(char::is_whitespace)
        .map_or(0, |at| at + 1);
    (start.min(touching_start), end)
}

/// One generated case: the field, the actions, what came of them.
struct Case {
    paste: String,
    initial: Model,
    actions: Vec<(Action, Model)>,
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
    let mut model = initial.clone();
    let mut actions = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let action = ALL[rng.below(ALL.len())];
        if apply(&mut model, action, &mut rng) {
            actions.push((action, model.clone()));
            if action == Action::ReturnThenEdit && apply(&mut model, Action::EditWord, &mut rng) {
                actions.push((Action::EditWord, model.clone()));
            }
        }
        if model.from == model.to {
            break;
        }
    }
    Case {
        paste,
        initial,
        actions,
    }
}

/// Applies `action` to the model, moving the region as the action does by
/// construction. `false` when it does not apply.
fn apply(model: &mut Model, action: Action, rng: &mut Rng) -> bool {
    let words = model.words();
    match action {
        // The newline a Return typed; the edit comes later, as its own action.
        Action::ReturnThenEdit => {
            let to = model.to;
            model.splice(to, to, "\n");
            true
        }
        Action::EditWord => {
            if words.is_empty() {
                return false;
            }
            let (a, b) = words[rng.below(words.len())];
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
        }
        Action::EditTwo => {
            if words.len() < 2 {
                return false;
            }
            let k = rng.below(words.len() - 1);
            let (a, b) = (words[k].0, words[k + 1].1);
            model.splice(a, b, "Rune Sarah");
            model.to = model.to + "Rune Sarah".len() - (b - a);
            true
        }
        Action::TrailingTwo => {
            let Some(&(a, b)) = words.last() else {
                return false;
            };
            model.splice(a, b, "Kit and Sarah");
            model.to = model.to + "Kit and Sarah".len() - (b - a);
            true
        }
        Action::ReplaceAll => {
            let new = sentence(rng);
            let (from, to) = (model.from, model.to);
            model.splice(from, to, &new);
            model.to = from + new.len();
            true
        }
        Action::DeletePaste => {
            let (from, to) = (model.from, model.to);
            model.splice(from, to, "");
            model.to = from;
            true
        }
        Action::DeleteBefore => {
            let (from, to) = (model.from, model.to);
            let start = model.text[..from]
                .trim_end()
                .rfind(char::is_whitespace)
                .map_or(0, |at| at + 1);
            model.splice(start, to, "");
            (model.from, model.to) = (start, start);
            true
        }
        Action::TypeAfter | Action::SecondDictation => {
            let new = format!(" {}", sentence(rng));
            let to = model.to;
            model.splice(to, to, &new);
            true
        }
        Action::TypeBefore => {
            let new = format!("{} ", sentence(rng));
            let from = model.from;
            model.splice(from, from, &new);
            model.from += new.len();
            model.to += new.len();
            true
        }
        Action::Clear => {
            model.text.clear();
            (model.from, model.to) = (0, 0);
            true
        }
    }
}

fn at(t0: Instant, ms: u64) -> Instant {
    t0 + Duration::from_millis(ms)
}

/// Runs the session over the case; the observation it ends with.
fn run(case: &Case, anchored: bool) -> Outcome {
    let t0 = Instant::now();
    let live = Live(RefCell::new(case.initial.text.clone()));
    let first = anchor(&case.initial, &case.paste);
    let mut session = Session::new(&live, case.paste.clone(), t0, anchored.then_some(first));
    let open: &dyn Fn() -> bool = &|| true;
    let mut ms = 50;
    let finished = |step: Step| match step {
        Step::Done(seen) => Some(seen.outcome),
        _ => None,
    };
    if let Some(done) = finished(session.step(at(t0, ms), Event::Tick, open)) {
        return done;
    }
    let mut after_return = false;
    for (action, model) in &case.actions {
        // An edit after a Return comes well after it, as a later keystroke.
        ms += if after_return { 1_500 } else { 100 };
        after_return = *action == Action::ReturnThenEdit;
        *live.0.borrow_mut() = model.text.clone();
        if after_return && let Some(done) = finished(session.step(at(t0, ms), Event::Return, open))
        {
            return done;
        }
        if let Some(done) = finished(session.step(at(t0, ms + 50), Event::ValueChanged, open)) {
            return done;
        }
    }
    match session.step(at(t0, ms + 500), Event::FocusOut, open) {
        Step::Done(seen) => seen.outcome,
        other => panic!("the closing read ends the watch: {other:?}"),
    }
}

/// Whether `span` sits inside `allowed` somewhere in the final field.
fn inside(model: &Model, span: &str) -> bool {
    let (from, to) = allowed(model);
    let mut at = 0;
    while let Some(found) = model.text[at..].find(span) {
        let start = at + found;
        if start >= from && start + span.len() <= to {
            return true;
        }
        at = start + model.text[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

#[derive(Default, Debug)]
struct Report {
    cases: usize,
    stored: usize,
    outside: Vec<String>,
    deleted_stored: Vec<String>,
    good_missed: Vec<String>,
    good: usize,
}

fn report(cases: u64) -> Report {
    let mut report = Report::default();
    for seed in 1..=cases {
        let case = generate(seed);
        let Some((_, last)) = case.actions.last() else {
            continue;
        };
        report.cases += 1;
        let outcome = run(&case, true);
        let describe = || {
            let steps: Vec<_> = case.actions.iter().map(|(a, _)| a).collect();
            format!(
                "seed {seed}: paste {:?} in {:?}, {steps:?} -> {:?}, region {:?}",
                case.paste,
                case.initial.text,
                last.text,
                &last.text[last.from..last.to]
            )
        };
        if let Outcome::Edited(span) = &outcome {
            report.stored += 1;
            if last.from == last.to {
                report
                    .deleted_stored
                    .push(format!("{} => {span:?}", describe()));
            } else if !inside(last, span) {
                report.outside.push(format!("{} => {span:?}", describe()));
            }
        }
        // Known good: one edit, nothing else, to a paste the floor can judge.
        if let [(action, model)] = &case.actions[..] {
            let n = case.initial.words().len();
            let judged = match action {
                Action::EditWord | Action::TrailingTwo => true,
                Action::EditTwo => n == 2 || (n >= 4 && two_inside(&case)),
                _ => false,
            };
            if judged {
                report.good += 1;
                let expected = model.text[model.from..model.to].trim().to_string();
                if outcome != Outcome::Edited(expected.clone()) {
                    report.good_missed.push(format!(
                        "{} => {outcome:?}, wanted {expected:?}",
                        describe()
                    ));
                }
            }
        }
    }
    report
}

/// Whether the two words an `EditTwo` changed are both inside the paste, not
/// at its edge (the strict floor leaves at most one token over at an end).
fn two_inside(case: &Case) -> bool {
    let before = case.initial.words();
    let after = case.actions[0].1.words();
    // "Rune Sarah" replaced two adjacent words: the words kept at the front
    // and back say where.
    let kept_front = before
        .iter()
        .zip(after.iter())
        .take_while(|(a, b)| case.initial.text[a.0..a.1] == case.actions[0].1.text[b.0..b.1])
        .count();
    kept_front >= 1 && before.len() - kept_front >= 3
}

#[test]
fn what_is_stored_is_the_pastes_own_region_and_nothing_else() {
    let report = report(4_000);
    println!(
        "cases {} stored {} outside {} deleted-but-stored {} known-good {} missed {}",
        report.cases,
        report.stored,
        report.outside.len(),
        report.deleted_stored.len(),
        report.good,
        report.good_missed.len()
    );
    for line in report
        .outside
        .iter()
        .chain(&report.deleted_stored)
        .chain(&report.good_missed)
        .take(8)
    {
        println!("{line}");
    }
    assert!(report.cases > 3_000, "{}", report.cases);
    assert!(report.good > 300, "{}", report.good);
    assert!(
        report.outside.is_empty(),
        "{} outside",
        report.outside.len()
    );
    assert!(report.deleted_stored.is_empty());
    assert!(
        report.good_missed.is_empty(),
        "{} missed",
        report.good_missed.len()
    );
}

/// Before the paste has been seen whole, nothing is an edit: not anchored,
/// not strict.
#[test]
fn nothing_is_an_edit_before_the_paste_was_seen() {
    for anchored in [true, false] {
        for paste in ["send it to Kip", "Kip", "Hanks Smith"] {
            let case = Case {
                paste: paste.to_string(),
                initial: Model {
                    text: "secret".to_string(),
                    from: 0,
                    to: 6,
                },
                actions: Vec::new(),
            };
            let outcome = run(&case, anchored);
            assert!(
                !matches!(outcome, Outcome::Edited(_)),
                "{paste:?} anchored {anchored}: {outcome:?}"
            );
        }
    }
}
