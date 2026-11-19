"""`healing tidy habits [--json]`: which fillers and verbal habits you accept removing (T8.2, D-38).

Labs › Tidy after paste suggests removing fillers; you accept or skip each suggestion, and T8.5 records
every decision in the dataset's `tidy_events` table (`clip_id, kind, token, accepted, source`, the token
alone, no surrounding text; the decision's time is its clip's `timestamp_ms`). This module turns those
decisions into a habit list for the suggester, never a speech-model label:

- **Candidates** are a fixed starter list plus phrases you removed by hand after pastes (M2's `edits`
  rows with nothing typed in their place) and often say.
- **Each candidate's table row** counts its filler removals shown, accepted and skipped; a decision older
  than 90 days counts half, so the rates follow what you do now (D-64).
- **Status:** `habit` at an accept rate of at least 0.7 over at least 5 (weighted) showings; `dropped`
  below that rate once shown 5 times, or sooner when even accepting every showing up to 5 couldn't reach
  it; else `trial`, so a candidate is suggested while there is too little evidence (D-65).

The export, `$HEALING_HOME/tidy/habits.json` (atomic, owner-only), lists the habits ranked by accept rate
and the trial phrases; Hex reads it from `tidy.habits_file` (D-66).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from healing.paths import healing_home
from healing.store import now_ms

STARTERS = ("like", "you know", "basically", "actually", "kind of", "sort of", "i guess")
# Hex's own fillers, which its rule pass always suggests: never a habit (`tidy/rules.rs` FILLERS).
BUILT_IN_FILLERS = frozenset({"um", "umm", "uh", "uhh", "erm", "er", "hmm", "hm", "ah"})
# The decision kinds that are a removal of the phrase as a filler. A model-pass removal of a habit word
# is classified a self-correction by T8.1 (it isn't a built-in filler), and it is the same decision.
REMOVAL_KINDS = frozenset({"filler", "self_correction"})

MIN_ACCEPT_RATE = 0.7
MIN_SHOWN = 5.0
DECAY_AFTER_MS = 90 * 86_400_000
OLD_WEIGHT = 0.5

# A phrase you removed by hand is a candidate once removed in this many clips, said in this many, and
# removed in at least this share of the clips that said it.
MIN_HAND_REMOVALS = 3
MIN_RAW_CLIPS = 5
MIN_HAND_SHARE = 0.2

# What Hex reads (`src/dictation/tidy/habits.rs`): the export never holds more.
MAX_PHRASES = 64
MAX_PHRASE_BYTES = 32
MAX_PHRASE_WORDS = 4
FILE_VERSION = 1

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


@dataclass(frozen=True)
class HabitRow:
    phrase: str
    shown: float
    accepted: float
    skipped: float
    last_seen_ms: int | None
    accept_rate: float | None
    status: str  # habit | trial | dropped


def normalize(text: str) -> str | None:
    """A phrase as Hex matches it: lowercase words joined by one space; None if it has anything but
    letters and apostrophes, or is too long for Hex to read."""
    words = _WORD.findall(text.lower().replace("’", "'"))
    rest = _WORD.sub("", text.lower().replace("’", "'"))
    if not words or re.search(r"[^\s,.;:!?]", rest):
        return None
    phrase = " ".join(words)
    if len(words) > MAX_PHRASE_WORDS or len(phrase.encode()) > MAX_PHRASE_BYTES:
        return None
    return phrase


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def decisions(db: sqlite3.Connection) -> list[tuple[str, str, bool, int]]:
    """Every recorded tidy decision as `(kind, token, accepted, at_ms)`. None before T8.5 writes them."""
    if not _table_exists(db, "tidy_events"):
        return []
    rows = db.execute(
        "SELECT t.kind, t.token, t.accepted, c.timestamp_ms FROM tidy_events t JOIN clips c ON c.clip_id = t.clip_id"
    ).fetchall()
    return [(str(kind), str(token), bool(accepted), int(at)) for kind, token, accepted, at in rows]


def hand_removed(db: sqlite3.Connection) -> set[str]:
    """Phrases you removed by hand after pastes (M2's edits that typed nothing in their place) often
    enough, and say often enough, to be worth suggesting."""
    removed: dict[str, set[str]] = defaultdict(set)
    for clip_id, final_from in db.execute("SELECT clip_id, final_from FROM edits WHERE to_text = '' AND final_from != ''"):
        phrase = normalize(final_from)
        if phrase and phrase not in BUILT_IN_FILLERS:
            removed[phrase].add(clip_id)
    frequent = {phrase for phrase, clips in removed.items() if len(clips) >= MIN_HAND_REMOVALS}
    if not frequent:
        return set()
    said: dict[str, int] = defaultdict(int)
    for (raw,) in db.execute("SELECT raw_text FROM clips"):
        padded = f" {' '.join(_WORD.findall(raw.lower().replace('’', "'")))} "
        for phrase in frequent:
            if f" {phrase} " in padded:
                said[phrase] += 1
    return {
        phrase for phrase in frequent
        if said[phrase] >= MIN_RAW_CLIPS and len(removed[phrase]) >= MIN_HAND_SHARE * said[phrase]
    }


def _status(shown: float, accepted: float) -> tuple[float | None, str]:
    rate = accepted / shown if shown else None
    if shown >= MIN_SHOWN:
        return rate, "habit" if rate >= MIN_ACCEPT_RATE else "dropped"
    # Even accepting every showing until there are enough couldn't reach the rate.
    if accepted + (MIN_SHOWN - shown) < MIN_ACCEPT_RATE * MIN_SHOWN:
        return rate, "dropped"
    return rate, "trial"


def habit_table(store, *, now: int | None = None) -> list[HabitRow]:
    """One row per candidate phrase, sorted by status (habits first), then accept rate."""
    now = now_ms() if now is None else now
    candidates = set(STARTERS) | hand_removed(store.db)
    counts: dict[str, list] = {phrase: [0.0, 0.0, None] for phrase in candidates}  # accepted, skipped, last
    for kind, token, accepted, at in decisions(store.db):
        phrase = normalize(token)
        if kind not in REMOVAL_KINDS or phrase not in counts:
            continue
        weight = OLD_WEIGHT if now - at > DECAY_AFTER_MS else 1.0
        entry = counts[phrase]
        entry[0 if accepted else 1] += weight
        entry[2] = at if entry[2] is None else max(entry[2], at)
    rows = []
    for phrase, (accepted, skipped, last) in counts.items():
        rate, status = _status(accepted + skipped, accepted)
        rows.append(HabitRow(phrase, accepted + skipped, accepted, skipped, last, rate, status))
    order = {"habit": 0, "trial": 1, "dropped": 2}
    return sorted(rows, key=lambda r: (order[r.status], -(r.accept_rate or 0.0), -r.shown, r.phrase))


def habits_path() -> Path:
    return healing_home() / "tidy" / "habits.json"


def export(rows: list[HabitRow], *, now: int | None = None) -> Path:
    """Write `habits.json` atomically, owner-only: habits ranked by accept rate, then trial phrases, at
    most `MAX_PHRASES` in all and only phrases Hex can read."""
    from healing.vocab.apply import write_atomic

    readable = [r for r in rows if normalize(r.phrase) == r.phrase]
    habits = sorted((r for r in readable if r.status == "habit"), key=lambda r: (-(r.accept_rate or 0.0), r.phrase))
    habits = habits[:MAX_PHRASES]
    trial = sorted(r.phrase for r in readable if r.status == "trial")[: MAX_PHRASES - len(habits)]
    document = {
        "version": FILE_VERSION,
        "generated_at_ms": now_ms() if now is None else now,
        "habits": [{"phrase": r.phrase, "accept_rate": round(r.accept_rate, 4)} for r in habits],
        "trial": trial,
    }
    path = habits_path()
    write_atomic(path, json.dumps(document, indent=2) + "\n")
    path.parent.chmod(0o700)
    return path


def main(args) -> int:
    import sys

    from healing.store import Store, StoreError

    try:
        store = Store(sweep=False)
        try:
            rows = habit_table(store)
        finally:
            store.close()
        path = export(rows)
    except (StoreError, sqlite3.Error, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"file": str(path), "phrases": [asdict(r) for r in rows]}, indent=2))
        return 0
    print(f"{'phrase':<16} {'status':<8} {'shown':>6} {'accepted':>8} {'skipped':>7} {'rate':>5}")
    for r in rows:
        rate = "-" if r.accept_rate is None else f"{r.accept_rate:.2f}"
        print(f"{r.phrase:<16} {r.status:<8} {r.shown:>6g} {r.accepted:>8g} {r.skipped:>7g} {rate:>5}")
    print(f"wrote {path}")
    return 0
