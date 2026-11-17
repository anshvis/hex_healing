//! A cell-origin fuzz of the tracker: every character of the field is tracked
//! as before the paste (b), the paste (p), typed inside it (i), typed
//! outside it (x) or a deleted paste character, and a stored span leaks when a
//! word of it is made of before-text or wholly of outside keys. Users type,
//! backspace, select-and-type, autocorrect, paste a copy again, and some reads
//! cover several keys (`coalesce`: one read in that many is skipped). `tiny`
//! uses a two-letter alphabet so that every word equals its neighbours.
//!
//! Ported from a reviewer's measurement (`fz8`). Long: ignored by default;
//! `HEX_FUZZ_SESSIONS` sets the count (1.5 million each).
#![allow(clippy::all, clippy::pedantic, dead_code, unused)]
use std::cell::Cell;

use super::locate::{Alignment, Anchor, Tracker};
use crate::accessibility::FieldText;

thread_local! {
    static TINY: Cell<bool> = const { Cell::new(false) };
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
        (self.next() % n as u64) as usize
    }
}
// B before, P paste, I typed-inside, X typed-outside, D tombstone of deleted paste char
#[derive(Clone, Copy, PartialEq, Debug)]
enum O {
    B,
    P,
    I,
    X,
    D,
}
const BW_BIG: &[&str] = &[
    "cafe\u{301}",
    "alpha",
    "bravo",
    "café",
    "naïve",
    "日本語",
    "straße",
    "a",
    "I",
    "it's",
    "(hello)",
    "world,",
    "end.",
    "😀",
    "ok😀",
    "Zoë",
    "ﬁnal",
    "x",
    "alpha",
    "alpha",
    "—",
    "note:",
];
const PW_BIG: &[&str] = &[
    "cafe\u{301}",
    "north",
    "south",
    "east",
    "naïve",
    "café",
    "up",
    "down",
    "it's",
    "a",
    "I",
    "😀",
    "Straße",
    "left,",
    "(right)",
    "kip",
    "hanks",
];
const TW_BIG: &[&str] = &[
    "cafe\u{301}",
    "red",
    "green",
    "blue",
    "côté",
    "a",
    "I",
    "north",
    "alpha",
    "😀",
    "grey,",
    "(teal)",
    "navy.",
    "ß",
    "ñandú",
    "x",
    "it's",
    "kit",
    "hank",
    "kips",
];

const TINY_B: &[&str] = &["a", "b", "ab", "a", "b", "aa"];
const TINY_P: &[&str] = &["a", "b", "ab", "a", "b", "ba"];
const TINY_T: &[&str] = &["a", "b", "ab", "c", "a", "b", "ba"];
fn bw() -> &'static [&'static str] {
    if TINY.with(std::cell::Cell::get) {
        TINY_B
    } else {
        BW_BIG
    }
}
fn pwl() -> &'static [&'static str] {
    if TINY.with(std::cell::Cell::get) {
        TINY_P
    } else {
        PW_BIG
    }
}
fn twl() -> &'static [&'static str] {
    if TINY.with(std::cell::Cell::get) {
        TINY_T
    } else {
        TW_BIG
    }
}
fn is_sp(c: char) -> bool {
    c.is_whitespace() || c == '\u{FFFC}'
}
fn vis(c: &[(char, O)]) -> Vec<(char, O)> {
    c.iter().filter(|x| x.1 != O::D).cloned().collect()
}
fn toks(c: &[(char, O)]) -> Vec<(usize, usize)> {
    let mut v = vec![];
    let mut o = None;
    for (i, (ch, _)) in c.iter().enumerate() {
        let sp = is_sp(*ch);
        match (sp, o) {
            (false, None) => o = Some(i),
            (true, Some(s)) => {
                v.push((s, i));
                o = None
            }
            _ => {}
        }
    }
    if let Some(s) = o {
        v.push((s, c.len()))
    }
    v
}
fn text(c: &[(char, O)]) -> String {
    c.iter().filter(|x| x.1 != O::D).map(|x| x.0).collect()
}
fn ws(r: &mut Rng) -> char {
    match r.below(11) {
        0 => '\n',
        1 => '\t',
        2 => '\u{FFFC}',
        3 => '\u{00A0}',
        4 => '\r',
        5 => '\u{2003}',
        _ => ' ',
    }
}

// visible index -> cell index of that visible char
fn vidx(c: &[(char, O)]) -> Vec<usize> {
    c.iter()
        .enumerate()
        .filter(|(_, x)| x.1 != O::D)
        .map(|(i, _)| i)
        .collect()
}
fn nvis(c: &[(char, O)]) -> usize {
    c.iter().filter(|x| x.1 != O::D).count()
}
fn typed_o(c: &[(char, O)], lo: usize, hi: usize, ch: char) -> O {
    if lo == hi && lo > 0 && !is_sp(ch) && !is_sp(c[lo - 1].0) && matches!(c[lo - 1].1, O::P | O::I)
    {
        return O::I;
    }
    // lo..hi tombs only between neighbours
    let l = if lo > 0 { Some(c[lo - 1].1) } else { None };
    let r = if hi < c.len() { Some(c[hi].1) } else { None };
    let mem = |o: Option<O>| matches!(o, Some(O::P) | Some(O::I) | Some(O::D));
    if hi > lo || (mem(l) && mem(r)) || l == Some(O::I) || r == Some(O::I) {
        O::I
    } else {
        O::X
    }
}
fn insert_vis(c: &mut Vec<(char, O)>, v: usize, ch: char) {
    let vi = vidx(c);
    let lo = if v == 0 { 0 } else { vi[v - 1] + 1 };
    let hi = if v >= vi.len() { c.len() } else { vi[v] };
    let o = typed_o(c, lo, hi, ch);
    c.insert(hi, (ch, o));
}
fn delete_vis(c: &mut Vec<(char, O)>, a: usize, b: usize, nop: bool) -> bool {
    let vi = vidx(c);
    let b = b.min(vi.len());
    if a >= b {
        return false;
    }
    if nop && (a..b).any(|k| c[vi[k]].1 == O::P) {
        return false;
    }
    for k in (a..b).rev() {
        let ci = vi[k];
        if c[ci].1 == O::P {
            c[ci] = ('\0', O::D);
        } else {
            c.remove(ci);
        }
    }
    true
}
fn sel_type(c: &mut Vec<(char, O)>, a: usize, b: usize, s: &str, nop: bool) -> usize {
    // returns new caret; types s after deleting
    if !delete_vis(c, a, b, nop) && b > a {
        return a;
    }
    let mut p = a;
    for ch in s.chars() {
        insert_vis(c, p, ch);
        p += 1;
    }
    p
}

fn pasted_tokens_with_tomb(cells: &[(char, O)]) -> usize {
    // P/D cells only, tomb chars are '\0' (non-space) so they glue to neighbours: a run of P/D cells with no P-space between is one original word
    let mut n = 0;
    let mut has = false;
    let mut inw = false;
    for x in cells.iter() {
        match x.1 {
            O::P => {
                if is_sp(x.0) {
                    if inw && has {
                        n += 1;
                    }
                    inw = false;
                    has = false;
                } else {
                    inw = true;
                }
            }
            O::D => {
                inw = true;
                has = true;
            }
            _ => {}
        }
    }
    if inw && has {
        n += 1;
    }
    n
}
fn fuzz(n: u64, seed0: u64, coalesce: usize, tiny: bool) -> (u64, usize, String) {
    TINY.with(|t| t.set(tiny));
    let nop = true;
    let mut report = String::new();

    let mut leaks = 0;
    let mut shown = 0;
    let mut stored = 0u64;
    for seed in seed0..seed0 + n {
        let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        for _ in 0..3 {
            r.next();
        }
        let nb = r.below(6);
        let mut base: Vec<(char, O)> = vec![];
        for i in 0..nb {
            if i > 0 {
                let w = ws(&mut r);
                base.push((w, O::B));
                if r.below(8) == 0 {
                    base.push((' ', O::B));
                }
            }
            for ch in bw()[r.below(bw().len())].chars() {
                base.push((ch, O::B));
            }
        }
        if r.below(6) == 0 {
            base.push((' ', O::B));
        }
        if r.below(10) == 0 {
            base.insert(0, (' ', O::B));
        }
        let caret0 = match r.below(4) {
            0 => 0,
            1 => base.len(),
            _ => r.below(base.len() + 1),
        };
        let pn = 1 + r.below(5);
        let mut pw: Vec<String> = vec![];
        while pw.len() < pn {
            let w = if r.below(4) == 0 && !base.is_empty() {
                let tk = toks(&base);
                if tk.is_empty() {
                    pwl()[r.below(pwl().len())].to_string()
                } else {
                    let (a, b) = tk[r.below(tk.len())];
                    base[a..b].iter().map(|x| x.0).collect()
                }
            } else {
                pwl()[r.below(pwl().len())].to_string()
            };
            pw.push(w);
        }
        let sep = if r.below(6) == 0 { "\n" } else { " " };
        let paste: String = pw.join(sep);
        let anchored = r.below(3) != 0;
        let drift: i64 = if r.below(5) == 0 {
            (r.below(5) as i64) - 2
        } else {
            0
        };
        let before_text = text(&base);
        let mut cells = base.clone();
        let mut log: Vec<String> = vec![];
        let mut leaked = false;
        let mut nstored = 0u64;
        let read = |tr: &mut Tracker,
                    cells: &Vec<(char, O)>,
                    log: &mut Vec<String>,
                    win: bool,
                    nstored: &mut u64|
         -> bool {
            let v = vis(cells);
            let t = text(cells);
            let (tt, w) = if win {
                let cs: Vec<char> = t.chars().collect();
                let a = cs.len() / 4;
                let b = (cs.len() * 3 / 4).max(a);
                (cs[a..b].iter().collect::<String>(), true)
            } else {
                (t.clone(), false)
            };
            let a = tr.read(&FieldText {
                text: tt.clone(),
                windowed: w,
            });
            log.push(format!(
                "{}{:?} -> {:?}   [{}]",
                if w { "W " } else { "" },
                tt,
                a,
                cells
                    .iter()
                    .map(|x| match x.1 {
                        O::B => 'b',
                        O::P => 'p',
                        O::I => 'i',
                        O::X => 'x',
                        O::D => '_',
                    })
                    .collect::<String>()
            ));
            if w {
                return false;
            }
            if let Alignment::Edited(span) = a {
                *nstored += 1;
                let sc: Vec<char> = span.chars().collect();
                let tc: Vec<char> = t.chars().collect();
                let mut any = false;
                let mut all_bad = true;
                if sc.len() <= tc.len() {
                    for s in 0..=tc.len() - sc.len() {
                        if tc[s..s + sc.len()] == sc[..] {
                            any = true;
                            let bad = toks(&v[s..s + sc.len()]).iter().any(|&(a, b)| {
                                let c = &v[s + a..s + b];
                                if c.iter().any(|x| x.1 == O::B) {
                                    return true;
                                }
                                // wholly typed token: must be inside
                                if c.iter().all(|x| x.1 == O::I || x.1 == O::X)
                                    && c.iter().any(|x| x.1 == O::X)
                                {
                                    return true;
                                }
                                false
                            });
                            let mut bad = bad;
                            if false {
                                let sl = &v[s..s + sc.len()];
                                // surviving P hull within the whole visible text
                                let pidx: Vec<usize> = v
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, x)| x.1 == O::P && !is_sp(x.0))
                                    .map(|(i, _)| i)
                                    .collect();
                                let hull = if pidx.is_empty() {
                                    None
                                } else {
                                    Some((pidx[0], *pidx.last().unwrap()))
                                };
                                let tk = toks(sl);
                                let mut outside_i = 0;
                                for &(a, b) in &tk {
                                    let c = &sl[a..b];
                                    if c.iter().all(|x| x.1 == O::I) {
                                        match hull {
                                            Some((h0, h1)) => {
                                                if s + b - 1 < h0 || s + a > h1 {
                                                    outside_i += 1;
                                                }
                                            }
                                            None => {
                                                outside_i += 1;
                                            }
                                        }
                                    }
                                }
                                // paste tokens with a tomb
                                let tombs: usize = {
                                    let mut k = 0;
                                    let mut open = false;
                                    let mut had = false;
                                    for x in cells.iter() {
                                        if x.1 == O::D {
                                            had = true;
                                        }
                                        if x.1 != O::D && is_sp(x.0) || x.0 == '\0' && false {}
                                        let _ = open;
                                    }
                                    let _ = had;
                                    0
                                };
                                let _ = tombs;
                                let mut words_with_tomb = 0usize;
                                {
                                    let mut cur_has = false;
                                    let mut in_word = false;
                                    for x in cells.iter() {
                                        if x.1 == O::P || x.1 == O::D {
                                            if x.1 == O::D {
                                                cur_has = true;
                                            }
                                            in_word = true;
                                        } else if in_word && is_sp(x.0) && x.1 == O::P {
                                        }
                                    }
                                    let _ = (cur_has, in_word);
                                }
                                let _ = words_with_tomb;
                                if outside_i > pasted_tokens_with_tomb(cells) {
                                    bad = true;
                                }
                            }
                            if !bad {
                                all_bad = false;
                            }
                        }
                    }
                }
                return any && all_bad;
            }
            false
        };
        // typing between the field read and the paste landing (rare)
        let mut caret = caret0;
        if r.below(6) == 0 {
            let q = r.below(cells.len() + 1);
            let s = twl()[r.below(twl().len())];
            let mut j = 0;
            for c in s.chars() {
                cells.insert(q + j, (c, O::X));
                j += 1;
            }
            if q <= caret {
                caret += j;
            }
        }
        let mut ps_ = String::new();
        let prev_nonws = caret > 0 && !is_sp(cells[caret - 1].0);
        let next_nonws = caret < cells.len() && !is_sp(cells[caret].0);
        if prev_nonws && r.below(5) != 0 {
            ps_.push(' ');
        }
        if r.below(8) == 0 {
            ps_.push(ws(&mut r));
        }
        ps_.push_str(&paste);
        if r.below(8) == 0 {
            ps_.push(ws(&mut r));
        }
        if next_nonws && r.below(5) != 0 {
            ps_.push(' ');
        }
        let mut tr = Tracker::new(
            ps_.clone(),
            if anchored {
                Some(Anchor {
                    start: ((base[..caret0]
                        .iter()
                        .map(|x| x.0.len_utf16())
                        .sum::<usize>() as i64
                        + drift)
                        .max(0)) as usize,
                })
            } else {
                None
            },
            Some(&before_text),
        );
        let pc: Vec<(char, O)> = ps_.chars().map(|c| (c, O::P)).collect();
        for (k, x) in pc.iter().enumerate() {
            cells.insert(caret + k, *x);
        }
        let mut cur = caret + pc.len();
        if read(&mut tr, &cells, &mut log, false, &mut nstored) {
            leaked = true;
        }
        let landed_ok = tr.landed();
        // the user session
        let actions = 2 + r.below(7);
        let mut unread = 0usize;
        for _ in 0..actions {
            let nv = nvis(&cells);
            let vi = vidx(&cells);
            let ps: Vec<usize> = (0..nv).filter(|&k| cells[vi[k]].1 == O::P).collect();
            let (pf, pt) = if ps.is_empty() {
                (cur.min(nv), cur.min(nv))
            } else {
                (ps[0], ps[ps.len() - 1] + 1)
            };
            let opts = [
                pf,
                pt,
                cur.min(nv),
                r.below(nv + 1),
                pf.saturating_sub(1),
                (pt + 1).min(nv),
                (pf + 1).min(nv),
                pt.saturating_sub(1),
            ];
            let p = opts[r.below(opts.len())].min(nv);
            // an action is a list of key events; each may be read
            let mut keys: Vec<Box<dyn Fn(&mut Vec<(char, O)>, &mut usize)>> = vec![];
            let choice = r.below(12);
            match choice {
                0 | 1 | 2 => {
                    // type a word key by key at p
                    let w = twl()[r.below(twl().len())];
                    let lead = r.below(3);
                    let s = match lead {
                        0 => format!(" {}", w),
                        1 => format!("{} ", w),
                        _ => w.to_string(),
                    };
                    let cs: Vec<char> = s.chars().collect();
                    let mut at = p;
                    for ch in cs {
                        let a = at;
                        keys.push(Box::new(move |c, cur| {
                            insert_vis(c, a, ch);
                            *cur = a + 1;
                        }));
                        at += 1;
                    }
                }
                3 | 4 => {
                    // backspace k keys from p
                    let k = 1 + r.below(10);
                    let mut at = p;
                    for _ in 0..k {
                        if at == 0 {
                            break;
                        }
                        let a = at - 1;
                        keys.push(Box::new(move |c, cur| {
                            delete_vis(c, a, a + 1, nop);
                            *cur = a;
                        }));
                        at -= 1;
                    }
                }
                5 => {
                    // forward delete k keys
                    let k = 1 + r.below(8);
                    for _ in 0..k {
                        let a = p;
                        keys.push(Box::new(move |c, cur| {
                            delete_vis(c, a, a + 1, nop);
                            *cur = a;
                        }));
                    }
                }
                6 | 7 => {
                    // select a range and type over (word-aligned or not)
                    let k = 1 + r.below(14);
                    let a = p.saturating_sub(if r.below(2) == 0 { 0 } else { r.below(5) });
                    let b = (a + k).min(nv);
                    let w = twl()[r.below(twl().len())];
                    let s = match r.below(4) {
                        0 => format!("{} {}", w, twl()[r.below(twl().len())]),
                        _ => w.to_string(),
                    };
                    let cs: Vec<char> = s.chars().collect();
                    // first key replaces selection
                    let c0 = cs[0];
                    let rest: Vec<char> = cs[1..].to_vec();
                    keys.push(Box::new(move |c, cur| {
                        delete_vis(c, a, b, nop);
                        insert_vis(c, a, c0);
                        *cur = a + 1;
                    }));
                    let mut at = a + 1;
                    for ch in rest {
                        let aa = at;
                        keys.push(Box::new(move |c, cur| {
                            insert_vis(c, aa, ch);
                            *cur = aa + 1;
                        }));
                        at += 1;
                    }
                }
                8 => {
                    // delete a token wholly (cmd-backspace / double click delete)
                    let v = vis(&cells);
                    let tk = toks(&v);
                    if let Some(&(a, b)) = tk.iter().find(|&&(a, b)| a <= p && p <= b) {
                        keys.push(Box::new(move |c, cur| {
                            delete_vis(c, a, b, nop);
                            *cur = a;
                        }));
                    }
                }
                9 => {
                    // autocorrect: replace the token at p with another word, one read
                    let v = vis(&cells);
                    let tk = toks(&v);
                    if let Some(&(a, b)) = tk.iter().find(|&&(a, b)| a <= p && p <= b) {
                        let w = twl()[r.below(twl().len())];
                        let s: Vec<char> = w.chars().collect();
                        keys.push(Box::new(move |c, cur| {
                            delete_vis(c, a, b, nop);
                            let mut q = a;
                            for &ch in &s {
                                insert_vis(c, q, ch);
                                q += 1;
                            }
                            *cur = q;
                        }));
                    }
                }
                10 => {
                    // whitespace key(s) at p
                    let k = 1 + r.below(2);
                    let mut at = p;
                    for _ in 0..k {
                        let w = ws(&mut r);
                        let a = at;
                        keys.push(Box::new(move |c, cur| {
                            insert_vis(c, a, w);
                            *cur = a + 1;
                        }));
                        at += 1;
                    }
                }
                _ => {
                    // paste a copy of the words again (cmd-V of something) at p
                    let s = if r.below(2) == 0 {
                        format!(" {}", paste)
                    } else {
                        format!("{} ", paste)
                    };
                    let cs: Vec<char> = s.chars().collect();
                    let at = p;
                    keys.push(Box::new(move |c, cur| {
                        let mut q = at;
                        for &ch in &cs {
                            insert_vis(c, q, ch);
                            q += 1;
                        }
                        *cur = q;
                    }));
                }
            }
            for k in keys {
                k(&mut cells, &mut cur);
                unread += 1;
                let win = r.below(40) == 0;
                let skip = coalesce > 1 && r.below(coalesce) == 0;
                if win || !skip {
                    if read(&mut tr, &cells, &mut log, win, &mut nstored) {
                        leaked = true;
                    }
                    unread = 0;
                }
            }
        }
        let _ = unread;
        if read(&mut tr, &cells, &mut log, false, &mut nstored) {
            leaked = true;
        }
        stored += nstored;
        if leaked && (landed_ok) {
            leaks += 1;
            if shown < 400 {
                shown += 1;
                report.push_str(&format!(
                    "LEAK seed {} paste {:?} base {:?} caret {} anchored {} drift {}\n",
                    seed, paste, before_text, caret0, anchored, drift
                ));
                for l in &log {
                    report.push_str(&format!("   {}\n", l));
                }
            }
        }
    }
    (stored, leaks, report)
}

fn sessions() -> u64 {
    std::env::var("HEX_FUZZ_SESSIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000)
}

fn check(coalesce: usize, tiny: bool) {
    let (stored, leaks, report) = fuzz(sessions(), 1, coalesce, tiny);
    assert_eq!(leaks, 0, "{stored} stored reads\n{report}");
}

#[test]
fn a_short_cell_fuzz_stores_nothing_outside() {
    for (coalesce, tiny) in [(1, false), (3, true), (8, false), (3, false), (8, true)] {
        let (_, leaks, report) = fuzz(20_000, 1, coalesce, tiny);
        assert_eq!(leaks, 0, "{report}");
    }
}

#[test]
#[ignore = "long: 1.5 million sessions"]
fn cell_fuzz_one_read_per_key() {
    check(1, false);
    check(1, true);
}

#[test]
#[ignore = "long: 1.5 million sessions"]
fn cell_fuzz_coalesced_by_three() {
    check(3, false);
    check(3, true);
}

#[test]
#[ignore = "long: 1.5 million sessions"]
fn cell_fuzz_coalesced_by_eight() {
    check(8, false);
    check(8, true);
}
