//! Recall of realistic single-word fixes through the real `Session`: how often
//! a person's fix of one word of a pasted dictation is stored, exactly, across
//! the ways of making it (selecting and typing, backspacing, typing first and
//! forward-deleting, one coalesced read), the first, middle or last word, and
//! what follows (nothing, more typing, a newline). A fix that is stored but
//! not exactly the fixed paste is counted apart: it is a precision miss, and
//! there must be none. Ported from a reviewer's measurement.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::locate::{Anchor, Tracker};
use super::session::{Event, Field, Outcome, Session, Step};
use crate::accessibility::{FieldText, ReadError};

struct Live {
    text: RefCell<String>,
}

impl Field for &Live {
    fn read(&self) -> Result<FieldText, ReadError> {
        Ok(FieldText {
            text: self.text.borrow().clone(),
            windowed: false,
        })
    }
}

const SENT: [&str; 12] = [
    "Kip",
    "Hanks",
    "ask Kip",
    "Hanks Smith",
    "send it to Kip",
    "Hanks is the best",
    "please send it to Kip tomorrow",
    "I think their house is nice",
    "Kip said hello to the whole team today",
    "the meeting with Hanks is at noon on Friday",
    "can you please send the report to Kip before the end of the day tomorrow so we can review",
    "we should ask Hanks about the budget for the next quarter and then decide what to do",
];

const METHODS: [&str; 5] = [
    "select+type",
    "oneRead",
    "bsWord+retype",
    "bsTail+type",
    "typeFirst+delete",
];
const TAILS: [&str; 3] = ["stop", "keepTyping", "newline"];

/// What the word becomes: the last letter changed, a different word, or an
/// appended letter.
fn replacement(old: &str, how: usize) -> String {
    match how {
        0 => {
            let mut chars: Vec<char> = old.chars().collect();
            if let Some(last) = chars.last_mut() {
                *last = if *last == 't' { 'p' } else { 't' };
            }
            chars.into_iter().collect()
        }
        1 => "Hex".to_string(),
        _ => format!("{old}s"),
    }
}

/// The successive texts of the one word as the keys that fix it are typed.
fn word_states(old: &str, new: &str, method: usize) -> Vec<String> {
    let o: Vec<char> = old.chars().collect();
    let n: Vec<char> = new.chars().collect();
    let common = o.iter().zip(&n).take_while(|(a, b)| a == b).count();
    let text = |chars: &[char]| chars.iter().collect::<String>();
    let mut out = Vec::new();
    match method {
        // Select the word, type the new one key by key.
        0 => out.extend((1..=n.len()).map(|i| text(&n[..i]))),
        // One read: pasted over the selection, or the spelling menu.
        1 => out.push(new.to_string()),
        // Backspace the whole word, retype it.
        2 => {
            out.extend((0..o.len()).rev().map(|i| text(&o[..i])));
            out.extend((1..=n.len()).map(|i| text(&n[..i])));
        }
        // Backspace only the differing tail, type the new tail.
        3 => {
            out.extend((common..o.len()).rev().map(|i| text(&o[..i])));
            out.extend((common + 1..=n.len()).map(|i| text(&n[..i])));
        }
        // Type the new tail first (inserted after the common prefix), then
        // forward-delete the old tail.
        _ => {
            let mut cur = o.clone();
            for (i, &c) in n.iter().enumerate().skip(common) {
                cur.insert(i, c);
                out.push(text(&cur));
            }
            while cur.len() > n.len() {
                cur.remove(n.len());
                out.push(text(&cur));
            }
        }
    }
    out
}

#[derive(Default)]
struct Tally {
    n: usize,
    ok: usize,
    bad: Vec<String>,
}

/// One way of fixing one word: what, how, and around what.
struct Scenario<'a> {
    paste: &'a str,
    idx: usize,
    how: usize,
    method: usize,
    coalesce: usize,
    prefix: &'a str,
    suffix: &'a str,
    tail: usize,
}

impl Scenario<'_> {
    fn describe(&self, why: &str) -> String {
        format!(
            "{:?} idx {} how {} {} c{} pre {:?} suf {:?} tail {}: {why}",
            self.paste,
            self.idx,
            self.how,
            METHODS[self.method],
            self.coalesce,
            self.prefix,
            self.suffix,
            TAILS[self.tail]
        )
    }

    /// Whether the fix was stored exactly, and what was stored.
    fn run(&self) -> (bool, String) {
        let words: Vec<&str> = self.paste.split(' ').collect();
        let new_word = replacement(words[self.idx], self.how);
        if new_word == words[self.idx] {
            return (true, String::new());
        }
        let with = |word: &str, extra: &str| -> String {
            let mut all: Vec<&str> = words.clone();
            all[self.idx] = word;
            format!("{}{}{extra}{}", self.prefix, all.join(" "), self.suffix)
        };
        let states: Vec<String> = word_states(words[self.idx], &new_word, self.method)
            .iter()
            .map(|word| with(word, ""))
            .collect();
        // Keystrokes arrive coalesced, a few to a read.
        let mut reads: Vec<String> = states
            .chunks(self.coalesce)
            .filter_map(|chunk| chunk.last().cloned())
            .collect();
        match self.tail {
            // Keeps typing a new sentence right after the paste, key by key.
            1 => {
                let more = " and then we can go";
                let mut typed = String::new();
                for c in more.chars() {
                    typed.push(c);
                    reads.push(with(&new_word, &typed));
                }
            }
            2 => reads.push(with(&new_word, "\n")),
            _ => {}
        }
        let live = Live {
            text: RefCell::new(format!("{}{}{}", self.prefix, self.paste, self.suffix)),
        };
        let t0 = Instant::now();
        let anchor = Anchor {
            start: self.prefix.encode_utf16().count(),
        };
        let mut session = Session::new(
            &live,
            t0,
            Tracker::new(self.paste.to_string(), Some(anchor), None),
        );
        let open: &dyn Fn() -> bool = &|| true;
        let mut result = None;
        let mut ms = 50_u64;
        let mut feed = |session: &mut Session<&Live>, event: Event, ms: u64| {
            if result.is_none()
                && let Step::Done(seen) = session.step(t0 + Duration::from_millis(ms), event, open)
            {
                result = Some(seen.outcome);
            }
        };
        feed(&mut session, Event::Tick, ms);
        for read in &reads {
            ms += 50;
            live.text.borrow_mut().clone_from(read);
            feed(&mut session, Event::ValueChanged, ms);
            feed(&mut session, Event::Tick, ms);
        }
        ms += 1_000;
        feed(&mut session, Event::FocusOut, ms);
        let expect = with(&new_word, "")
            .strip_prefix(self.prefix)
            .and_then(|rest| rest.strip_suffix(self.suffix))
            .unwrap_or_default()
            .to_string();
        match result {
            Some(Outcome::Edited(span)) => {
                (span == expect, format!("edited {span:?} want {expect:?}"))
            }
            Some(other) => (false, format!("{other:?} want {expect:?}")),
            None => (false, "none".into()),
        }
    }
}

/// The measurement: `(fixes stored exactly, fixes tried, stored but not
/// exactly the fix, lost with nothing typed after)`.
fn measure(full: bool) -> (usize, usize, usize, usize) {
    let prefixes: &[&str] = if full {
        &["", "ok. ", "Hi Sarah,\n"]
    } else {
        &["ok. "]
    };
    let suffixes: &[&str] = if full {
        &["", " thanks", "\nBest, Ann"]
    } else {
        &["", " thanks"]
    };
    let coalesces: &[usize] = if full { &[1, 2, 3] } else { &[1, 2] };
    let mut by: BTreeMap<String, Tally> = BTreeMap::new();
    let (mut ok, mut tried, mut wrong, mut lost) = (0, 0, 0, 0);
    for paste in SENT {
        let n = paste.split(' ').count();
        for idx in 0..n {
            let position = match idx {
                _ if n == 1 => "only",
                0 => "first",
                i if i == n - 1 => "last",
                _ => "middle",
            };
            for how in 0..3 {
                for (method, method_name) in METHODS.iter().enumerate() {
                    for &coalesce in coalesces {
                        for &prefix in prefixes {
                            for &suffix in suffixes {
                                for (tail, tail_name) in TAILS.iter().enumerate() {
                                    // The one-read method has no keys to coalesce, and
                                    // typing on only makes sense at the end of the text.
                                    if (method == 1 && coalesce > 1)
                                        || (tail == 1 && !suffix.is_empty())
                                    {
                                        continue;
                                    }
                                    let scenario = Scenario {
                                        paste,
                                        idx,
                                        how,
                                        method,
                                        coalesce,
                                        prefix,
                                        suffix,
                                        tail,
                                    };
                                    let (exact, why) = scenario.run();
                                    if !exact && why.starts_with("edited") {
                                        wrong += 1;
                                        println!("WRONG {}", scenario.describe(&why));
                                    }
                                    if !exact && why.starts_with("Lost") && tail == 0 {
                                        lost += 1;
                                    }
                                    for key in [
                                        format!("pos {position}"),
                                        format!("method {method_name}"),
                                        format!("tail {tail_name}"),
                                        format!("pos+tail {position}/{tail_name}"),
                                        "ALL".to_string(),
                                    ] {
                                        let entry = by.entry(key).or_default();
                                        entry.n += 1;
                                        if exact {
                                            entry.ok += 1;
                                        } else if entry.bad.len() < 3 {
                                            entry.bad.push(scenario.describe(&why));
                                        }
                                    }
                                    tried += 1;
                                    ok += usize::from(exact);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (key, tally) in &by {
        println!(
            "{key:34} {:>7}/{:<7} = {:5.1}%",
            tally.ok,
            tally.n,
            100.0 * tally.ok as f64 / tally.n as f64
        );
    }
    for tally in by.values() {
        for bad in &tally.bad {
            println!("  MISS {bad}");
        }
    }
    println!(
        "TOTAL {ok}/{tried}, stored but not exactly the fix {wrong}, lost with nothing typed after {lost}"
    );
    (ok, tried, wrong, lost)
}

#[test]
fn a_fix_is_stored_exactly_or_not_at_all() {
    let (ok, tried, wrong, _) = measure(false);
    assert_eq!(wrong, 0, "a fix was stored as something other than itself");
    assert!(ok * 100 >= tried * 97, "{ok}/{tried} fixes stored exactly");
}

/// The full measurement: `cargo test --bin hex recall_tests -- --ignored
/// --nocapture`. Reviewers measured 96.5% before the word budget and 37.7% of
/// the fixes that were typed on after.
#[test]
#[ignore = "long: every way of fixing every word"]
fn recall_of_every_way_of_fixing_a_word() {
    let (ok, tried, wrong, _) = measure(true);
    assert_eq!(wrong, 0, "a fix was stored as something other than itself");
    assert!(ok * 100 >= tried * 98, "{ok}/{tried} fixes stored exactly");
}
