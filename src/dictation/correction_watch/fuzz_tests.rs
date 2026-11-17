//! Reviewer fuzz: provenance-model oracle, keystroke / IME / undo / multi-change.
use super::locate::{Anchor, Tracker};
use super::session::{Event, Field, Outcome, Session, Step};
use crate::accessibility::{FieldText, ReadError};
use std::cell::RefCell;
use std::time::{Duration, Instant};

static HASH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
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
    fn chance(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }
}

const CHAT: [&str; 14] = [
    "thanks for the update I will review it tomorrow",
    "can you send me the report by Friday",
    "please send it to Kip tomorrow",
    "send it to Kip",
    "we need to talk about the budget for next quarter",
    "I will send it over tomorrow morning",
    "did you send it to Kip yet",
    "the meeting is at noon tomorrow",
    "let me know when you are free",
    "ok thanks",
    "Hanks",
    "Yes",
    "Hanks Smith",
    "go home",
];
const WORDS: [&str; 9] = [
    "Hex",
    "Rune",
    "Friday",
    "banana",
    "Sarah",
    "Monday",
    "Kit",
    "caf\u{e9}",
    "\u{1F600}",
];
const IME: [&str; 5] = [
    "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}",
    "kit",
    "hex",
    "\u{597d}\u{7684}",
    "na\u{ef}ve",
];

fn is_sp(c: char) -> bool {
    c.is_whitespace() || c == '\u{FFFC}'
}

#[derive(Clone)]
struct M {
    cells: Vec<char>,
    from: usize,
    to: usize,
    open_s: Option<usize>, // index where typing continues (start edge): typed inserted at this index goes inside
    open_e: bool,
    tainted: bool,
    caret: usize,
}
impl M {
    fn text(&self) -> String {
        self.cells.iter().collect()
    }
    /// Replace [a,b) with `with`; update the region by construction.
    fn replace(&mut self, a: usize, b: usize, with: &str) {
        let w: Vec<char> = with.chars().collect();
        let (n, removed) = (w.len(), b - a);
        self.cells.splice(a..b, w.iter().copied());
        self.caret = a + n;
        let (from, to) = (self.from, self.to);
        let mut open_s = None;
        let mut open_e = false;
        if removed == 0 {
            if open_e_continues(self.open_e, a, to) {
                self.to = to + n;
                open_e = true;
            } else if self.open_s == Some(a) && a >= from && a <= to {
                // typed at an open start edge: inside
                self.to = to + n;
                open_s = Some(a + n);
            } else if a == to
                && to > from
                && a > 0
                && !is_sp(self.cells[a - 1])
                && w.first().is_some_and(|&c| !is_sp(c))
            {
                // Keys glued to the last word are that word, longer.
                self.to = to + w.iter().take_while(|&&c| !is_sp(c)).count();
            } else if a <= from {
                self.from += n;
                self.to += n;
            } else if a >= to {
            } else {
                self.to += n;
            }
        } else if b <= from {
            self.from = from + n - removed;
            self.to = to + n - removed;
            if from + n < removed {
                self.from = 0;
            }
        } else if a >= to {
        } else if from <= a && b <= to {
            let shortened = n < removed;
            self.to = to + n - removed;
            if a == from && shortened {
                open_s = Some(a + n);
            }
            if b == to && shortened {
                open_e = true;
            }
            if (a == from || b == to) && n > removed {
                /* growth at edge: ambiguous */
                self.tainted = self.tainted || false;
            }
            if from == to { /* empty region */ }
        } else {
            // crosses an edge or encloses the region
            self.tainted = true;
            let nf = a.min(from);
            let nt = b.max(to) + n - removed;
            self.from = nf;
            self.to = nt;
        }
        self.open_s = open_s;
        self.open_e = open_e;
    }
}
fn open_e_continues(open: bool, a: usize, to: usize) -> bool {
    open && a == to
}

fn tokens_idx(cells: &[char]) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut open = None;
    for (i, &c) in cells.iter().enumerate() {
        match (is_sp(c), open) {
            (false, None) => open = Some(i),
            (true, Some(s)) => {
                out.push((s, i));
                open = None
            }
            _ => {}
        }
    }
    if let Some(s) = open {
        out.push((s, cells.len()));
    }
    out
}

/// Positions allowed for stored chars: tokens overlapping the region.
fn allowed(m: &M) -> Vec<bool> {
    let mut ok = vec![false; m.cells.len()];
    for (s, e) in tokens_idx(&m.cells) {
        if e > m.from && s < m.to {
            ok[s..e].fill(true);
        }
    }
    let end = m.to.min(m.cells.len());
    if m.from < end {
        ok[m.from..end].fill(true);
    }
    ok
}

fn outside_tokens(m: &M, span: &str) -> usize {
    let s: Vec<char> = span.chars().collect();
    if s.is_empty() || s.len() > m.cells.len() {
        return 99;
    }
    let ok = allowed(m);
    let mut best = 99;
    for start in 0..=m.cells.len() - s.len() {
        if m.cells[start..start + s.len()] == s[..] {
            let toks = tokens_idx(&m.cells[start..start + s.len()]);
            let n = toks
                .iter()
                .filter(|(a, b)| !(start + a..start + b).any(|i| ok[i]))
                .count();
            best = best.min(n);
        }
    }
    best
}

fn valid_span(m: &M, span: &str) -> bool {
    let s: Vec<char> = span.chars().collect();
    if s.is_empty() || s.len() > m.cells.len() {
        return false;
    }
    let ok = allowed(m);
    for start in 0..=m.cells.len().saturating_sub(s.len()) {
        if m.cells[start..start + s.len()] == s[..]
            && (start..start + s.len()).all(|i| ok[i] || is_sp(m.cells[i]))
        {
            return true;
        }
    }
    false
}

struct Live {
    text: RefCell<String>,
    caret: RefCell<usize>,
}
impl Field for &Live {
    fn read(&self) -> Result<FieldText, ReadError> {
        let t = self.text.borrow().clone();
        let units: Vec<u16> = t.encode_utf16().collect();
        if units.len() > 8192 {
            let caret = *self.caret.borrow();
            let start = caret.saturating_sub(4096).min(units.len());
            let len = (units.len() - start).min(8192);
            Ok(FieldText {
                text: String::from_utf16_lossy(&units[start..start + len]),
                windowed: true,
            })
        } else {
            Ok(FieldText {
                text: t,
                windowed: false,
            })
        }
    }
}

fn u16len(c: &[char]) -> usize {
    c.iter().map(|c| c.len_utf16()).sum()
}

struct Gen {
    rng: Rng,
    uniq: usize,
}
impl Gen {
    fn uword(&mut self) -> String {
        self.uniq += 1;
        format!("u{}x", self.uniq)
    }
    fn usentence(&mut self) -> String {
        let n = 1 + self.rng.below(5);
        (0..n).map(|_| self.uword()).collect::<Vec<_>>().join(" ")
    }
    fn text(&mut self, coincide: bool) -> String {
        if coincide && std::env::var("NODUP").is_err() && self.rng.chance(2) {
            CHAT[self.rng.below(CHAT.len())].to_string()
        } else {
            self.usentence()
        }
    }
    fn typed(&mut self) -> String {
        match self.rng.below(5) {
            0 => self.uword(),
            1 => WORDS[self.rng.below(WORDS.len())].to_string(),
            2 => format!("{} {}", self.uword(), self.uword()),
            3 => IME[self.rng.below(IME.len())].to_string(),
            _ => format!("{} ", self.uword()),
        }
    }
}

/// A random position class in the model.
fn pick_pos(g: &mut Gen, m: &M) -> usize {
    let n = m.cells.len();
    let toks = tokens_idx(&m.cells);
    match g.rng.below(11) {
        0 => m.from.min(n),
        1 => m.to.min(n),
        2 => m.from.saturating_sub(1),
        3 => (m.to + 1).min(n),
        4 => 0,
        5 => n,
        6 => {
            if m.to > m.from {
                m.from + g.rng.below(m.to - m.from + 1)
            } else {
                m.from.min(n)
            }
        }
        7 => {
            if toks.is_empty() {
                0
            } else {
                let t = toks[g.rng.below(toks.len())];
                if g.rng.chance(2) { t.0 } else { t.1 }
            }
        }
        8 => m.from.saturating_sub(2).min(n),
        9 => (m.to + 2).min(n),
        _ => g.rng.below(n + 1),
    }
}

fn chunks(g: &mut Gen, m: &mut M, a: usize, s: &str, out: &mut Vec<M>) {
    // key by key, coalescing 1..3
    let cs: Vec<char> = s.chars().collect();
    let mut pos = a;
    let mut i = 0;
    while i < cs.len() {
        let take = if g.rng.chance(4) {
            1 + g.rng.below(3)
        } else {
            1
        };
        let chunk: String = cs[i..(i + take).min(cs.len())].iter().collect();
        i += take;
        m.replace(pos, pos, &chunk);
        pos += chunk.chars().count();
        out.push(m.clone());
    }
}

/// Apply one op; returns the reads (field texts) it makes. `compose` ops return text only at the end.
fn op(
    g: &mut Gen,
    m: &mut M,
    hist: &mut Vec<M>,
    redo: &mut Vec<M>,
    name: &mut Vec<&'static str>,
) -> Vec<M> {
    let n = m.cells.len();
    let mut out = vec![];
    let k = if std::env::var("MORE").is_ok() && g.rng.chance(2) {
        14 + g.rng.below(4)
    } else {
        g.rng.below(14)
    };
    match k {
        0 => {
            name.push("type");
            let a = pick_pos(g, m);
            let s = g.typed();
            chunks(g, m, a, &s, &mut out);
        }
        1 => {
            name.push("backspace"); // delete k chars before a position, per key
            let a = pick_pos(g, m);
            let cnt = 1 + g.rng.below(6);
            let mut pos = a;
            for _ in 0..cnt {
                if pos == 0 {
                    break;
                }
                m.replace(pos - 1, pos, "");
                pos -= 1;
                out.push(m.clone());
            }
        }
        2 => {
            name.push("select-replace");
            let a = pick_pos(g, m);
            let b = (a + 1 + g.rng.below(14)).min(n);
            if a < b {
                let s = g.typed();
                let first: String = s.chars().take(1).collect();
                m.replace(a, b, &first);
                out.push(m.clone());
                let rest: String = s.chars().skip(1).collect();
                chunks(g, m, a + first.chars().count(), &rest, &mut out);
            }
        }
        2.. if k == 3 => {
            name.push("autocorrect-word");
            let toks = tokens_idx(&m.cells);
            if !toks.is_empty() {
                let t = toks[g.rng.below(toks.len())];
                let s = g.typed();
                m.replace(t.0, t.1, s.trim());
                out.push(m.clone());
            }
        }
        4 => {
            name.push("autocorrect-span"); // replace across a few words, incl. across edges
            let toks = tokens_idx(&m.cells);
            if toks.len() >= 2 {
                let i = g.rng.below(toks.len() - 1);
                let j = (i + 1 + g.rng.below(2)).min(toks.len() - 1);
                let s = g.typed();
                m.replace(toks[i].0, toks[j].1, s.trim());
                out.push(m.clone());
            }
        }
        5 => {
            name.push("ime");
            let a = pick_pos(g, m);
            let fin = IME[g.rng.below(IME.len())];
            let cs: Vec<char> = fin.chars().collect();
            let mut comp_len = 0; // composition buffer replaced each time
            for i in 1..=cs.len() {
                let buf: String = if i < cs.len() && g.rng.chance(2) {
                    format!("{}~", cs[..i].iter().collect::<String>())
                } else {
                    cs[..i].iter().collect()
                };
                m.replace(a, a + comp_len, &buf);
                comp_len = buf.chars().count();
                out.push(m.clone());
            }
        }
        6 => {
            name.push("multi-change");
            let cnt = 2 + g.rng.below(2);
            for _ in 0..cnt {
                let toks = tokens_idx(&m.cells);
                if toks.is_empty() {
                    break;
                }
                let t = toks[g.rng.below(toks.len())];
                match g.rng.below(3) {
                    0 => {
                        let s = g.typed();
                        m.replace(t.0, t.1, s.trim());
                    }
                    1 if std::env::var("NOZZ").is_err() => {
                        m.replace(t.1, t.1, " zz");
                    }
                    _ => {
                        let c = t.0.min(m.cells.len());
                        m.replace(c, c, "Q");
                    }
                }
            }
            out.push(m.clone());
        }
        14 => {
            name.push("ws-then-insert-before"); // whitespace at an edge, then words typed before it
            let a = if g.rng.chance(2) {
                m.to.min(n)
            } else {
                m.from.min(n)
            };
            let ws = if g.rng.chance(2) { "\n" } else { " " };
            m.replace(a, a, ws);
            out.push(m.clone());
            let at = a;
            let s = g.typed();
            let at = at.min(m.cells.len());
            chunks(g, m, at, &s, &mut out);
        }
        15 => {
            name.push("merge-ws");
            let toks = tokens_idx(&m.cells);
            if !toks.is_empty() {
                let t = toks[g.rng.below(toks.len())];
                let side = g.rng.chance(2);
                let p = if side { t.1 } else { t.0.saturating_sub(1) };
                if p < m.cells.len() && is_sp(m.cells[p]) {
                    m.replace(p, p + 1, "");
                    out.push(m.clone());
                }
            }
        }
        16 => {
            name.push("tail-cut-then-merge"); // delete the last few chars of the region then the following space
            if m.to > m.from + 1 {
                let cnt = 1 + g.rng.below(3);
                let mut pos = m.to;
                for _ in 0..cnt {
                    if pos <= m.from + 1 {
                        break;
                    }
                    m.replace(pos - 1, pos, "");
                    pos -= 1;
                    out.push(m.clone());
                }
                if pos < m.cells.len() && is_sp(m.cells[pos]) && g.rng.chance(2) {
                    m.replace(pos, pos + 1, "");
                    out.push(m.clone());
                }
            }
        }
        17 => {
            name.push("fwd-delete");
            let a = pick_pos(g, m);
            let cnt = 1 + g.rng.below(4);
            for _ in 0..cnt {
                if a >= m.cells.len() {
                    break;
                }
                m.replace(a, a + 1, "");
                out.push(m.clone());
            }
        }
        7 => {
            name.push("undo");
            if let Some(prev) = hist.pop() {
                redo.push(m.clone());
                *m = prev;
                out.push(m.clone());
            }
        }
        8 => {
            name.push("redo");
            if let Some(next) = redo.pop() {
                hist.push(m.clone());
                *m = next;
                out.push(m.clone());
            }
        }
        9 => {
            name.push("paste-at");
            let a = pick_pos(g, m);
            let s = g.text(true);
            m.replace(a, a, &format!(" {s}"));
            out.push(m.clone());
        }
        10 => {
            name.push("select-all-paste");
            let s = g.text(true);
            let len = m.cells.len();
            m.replace(0, len, &s);
            out.push(m.clone());
        }
        11 => {
            name.push("delete-word-back");
            let a = pick_pos(g, m);
            let mut s = a;
            while s > 0 && is_sp(m.cells[s - 1]) {
                s -= 1;
            }
            while s > 0 && !is_sp(m.cells[s - 1]) {
                s -= 1;
            }
            if s < a {
                m.replace(s, a, "");
                out.push(m.clone());
            }
        }
        12 => {
            name.push("delete-region-then-type"); // select paste, delete, then type a word
            if m.to > m.from {
                let (a, b) = (m.from, m.to);
                m.replace(a, b, "");
                out.push(m.clone());
                let s = g.typed();
                chunks(g, m, a, &s, &mut out);
            }
        }
        _ => {
            name.push("newline");
            let a = pick_pos(g, m);
            m.replace(a, a, "\n");
            out.push(m.clone());
        }
    }
    out
}

#[derive(Debug, Default)]
struct Tally {
    cases: usize,
    edited: usize,
    outside: usize,
    after_taint: usize,
    empty_edited: usize,
    lost_then_edited: usize,
}

fn run_case(seed: u64, tally: &mut Tally) -> Option<String> {
    let dbg = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        == Some(seed);
    let mut g = Gen {
        rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
        uniq: 0,
    };
    let paste = CHAT[g.rng.below(CHAT.len())].to_string();
    let coincide = g.rng.chance(3);
    let large = g.rng.chance(8);
    let mut prefix = if g.rng.chance(4) {
        String::new()
    } else {
        format!("{} ", g.text(coincide))
    };
    let mut suffix = if g.rng.chance(4) {
        String::new()
    } else {
        format!(" {}", g.text(coincide))
    };
    if large {
        let target = 8192 - 3 + g.rng.below(8);
        let mut f = String::new();
        while f.encode_utf16().count()
            + prefix.encode_utf16().count()
            + paste.encode_utf16().count()
            + suffix.encode_utf16().count()
            < target
        {
            f.push_str(&g.uword());
            f.push(' ');
        }
        if g.rng.chance(2) {
            prefix = format!("{f}{prefix}");
        } else {
            suffix = format!("{suffix} {f}");
        }
    }
    let no_caret = g.rng.chance(3);
    let drift: i32 = if g.rng.chance(4) {
        [-3, -2, -1, 1, 2, 3][g.rng.below(6)]
    } else {
        0
    };
    let pre: Vec<char> = prefix.chars().collect();
    let m0 = M {
        cells: format!("{prefix}{paste}{suffix}").chars().collect(),
        from: pre.len(),
        to: pre.len() + paste.chars().count(),
        open_s: None,
        open_e: false,
        tainted: false,
        caret: pre.len() + paste.chars().count(),
    };
    let before = format!("{prefix}{suffix}");
    let before_arg: Option<String> = if no_caret {
        if before.encode_utf16().count() > 8192 {
            None
        } else {
            Some(before.clone())
        }
    } else {
        None
    };
    let anchor = if no_caret {
        None
    } else {
        Some(Anchor {
            start: (u16len(&pre) as i32 + drift).max(0) as usize,
        })
    };
    let live = Live {
        text: RefCell::new(m0.text()),
        caret: RefCell::new(m0.caret),
    };
    let t0 = Instant::now();
    let mut session = Session::new(
        &live,
        t0,
        Tracker::new(paste.clone(), anchor, before_arg.as_deref()),
    );
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let open: &dyn Fn() -> bool = &|| true;
    let mut ms = 50u64;
    let mut names: Vec<&'static str> = vec![];
    let mut result: Option<(Outcome, u64)> = None;
    let mut snaps: std::collections::HashMap<u64, M> = Default::default();
    let mut cur = m0.clone();
    macro_rules! feed {
        ($ev:expr, $t:expr) => {{
            *live.text.borrow_mut() = cur.text();
            *live.caret.borrow_mut() = cur.caret;
            snaps.insert($t, cur.clone());
            if dbg {
                println!("  @{} {:?}", $t, cur.text());
            }
            if result.is_none() {
                if let Step::Done(s) = session.step(at($t), $ev, open) {
                    result = Some((s.outcome, s.observed_at_ms));
                }
            }
        }};
    }
    feed!(Event::Tick, ms);
    let mut hist: Vec<M> = vec![];
    let mut redo: Vec<M> = vec![];
    let mut m = m0.clone();
    let nops = 1 + g.rng.below(5);
    'ops: for _ in 0..nops {
        if result.is_some() {
            break;
        }
        let snap = m.clone();
        let reads = op(&mut g, &mut m, &mut hist, &mut redo, &mut names);
        if reads.is_empty() {
            continue;
        }
        if names.last() != Some(&"undo") && names.last() != Some(&"redo") {
            hist.push(snap);
            redo.clear();
        }
        let mut i = 0;
        while i < reads.len() {
            let take = if g.rng.chance(5) {
                1 + g.rng.below(3)
            } else {
                1
            };
            let j = (i + take).min(reads.len());
            ms += 50 + 50 * g.rng.below(2) as u64;
            cur = reads[j - 1].clone();
            feed!(Event::ValueChanged, ms);
            if result.is_some() {
                break 'ops;
            }
            feed!(Event::Tick, ms);
            if result.is_some() {
                break 'ops;
            }
            i = j;
        }
        if g.rng.chance(6) {
            ms += 60;
            let clear = g.rng.chance(2);
            let mut blank = m.clone();
            blank.cells.clear();
            blank.from = 0;
            blank.to = 0;
            blank.tainted = true;
            let mut nl = m.clone();
            let l = nl.cells.len();
            nl.replace(l, l, "\n");
            let after = if clear { blank.clone() } else { nl.clone() };
            if g.rng.chance(2) {
                cur = after.clone();
                feed!(Event::Return, ms);
            } else {
                feed!(Event::Return, ms);
                cur = after.clone();
            }
            if result.is_none() {
                ms += 50;
                feed!(Event::ValueChanged, ms);
            }
            if clear {
                break;
            }
            m = nl;
        }
    }
    if result.is_none() {
        ms += 500;
        cur = m.clone();
        feed!(Event::FocusOut, ms);
        assert!(result.is_some(), "no done");
    }
    tally.cases += 1;
    if dbg {
        println!(
            "paste {paste:?} no_caret {no_caret} anchor {anchor:?} before {before_arg:?} -> {:?}",
            result
        );
    }
    let (outcome, obs_ms) = result.unwrap();
    {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        format!("{:?}{}", outcome, obs_ms).hash(&mut h);
        HASH.fetch_xor(h.finish(), std::sync::atomic::Ordering::Relaxed);
    }
    if let Outcome::Edited(span) = &outcome {
        tally.edited += 1;
        let Some(model) = snaps.get(&obs_ms) else {
            return Some(format!("seed {seed}: no snapshot for ms {obs_ms}"));
        };
        if model.tainted {
            tally.after_taint += 1;
        }
        if model.from >= model.to {
            tally.empty_edited += 1;
        }
        let extra = if valid_span(model, span) {
            0
        } else {
            outside_tokens(model, span)
        };
        if extra == 1 {
            tally.after_taint += 0;
            tally.lost_then_edited += 1;
        }
        if extra >= 2 {
            tally.outside += 1;
            return Some(format!(
                "seed {seed} paste {paste:?} no_caret {no_caret} drift {drift} ops {names:?} tainted {} region {:?} field {:?} span {span:?}",
                model.tainted,
                model
                    .text()
                    .chars()
                    .skip(model.from)
                    .take(model.to.saturating_sub(model.from))
                    .collect::<String>(),
                model.text()
            ));
        }
    }
    None
}

fn fuzz(n: u64, base: u64) {
    let mut t = Tally::default();
    let mut bad = vec![];
    for seed in base..base + n {
        if let Some(violation) = run_case(seed, &mut t) {
            bad.push(violation);
        }
    }
    println!(
        "{t:?} HASH {}",
        HASH.load(std::sync::atomic::Ordering::Relaxed)
    );
    for b in bad.iter().take(25) {
        println!("VIOLATION {b}");
    }
    assert!(bad.is_empty(), "{} violations", bad.len());
}

/// The fast subset (a debug build must finish inside nextest's limit).
#[test]
fn a_reviewers_provenance_fuzz_finds_nothing() {
    fuzz(1_500, 1);
}

/// The long run, with the seeds the reviewer used and then some: `cargo test
/// --bin hex fuzz_tests -- --ignored --nocapture`.
#[test]
#[ignore = "long: 300,000 generated cases"]
fn a_reviewers_provenance_fuzz_finds_nothing_over_a_long_run() {
    fuzz(300_000, 1);
}
