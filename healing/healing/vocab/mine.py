"""`healing vocab mine`: acoustic corrections you made more than once become candidate terms.

Evidence is the `edits` table (T2.4's classifier, no second one) of `strong` clips (`labels` owns that
label: an acoustic edit, no unmapped one, no rewrite, no review overruling it). The same
`raw_from -> to`, compared on `corrections.tokens` norms so case, punctuation and number spelling
don't split it, seen in at least `min_pairs` clips is a pair. Per term:

- `term` is `to` as typed, in its most frequent casing; casing variants are one term;
- `sounds_like` are the heard spellings of its pairs;
- `score` is the term's use in your last `window` of dictations (a clip counts once, when its
  `final_text` or `reference_text` holds the term), each use halved every `HALF_LIFE_DAYS`.

`mine` reads only. `save` writes the ranked result to `vocab_candidates` (T4.2 reads it with
`load_candidates`; T4.3 calls `mine(train_only=True)` and keeps it in memory, so its leak-free run never
replaces the table `apply` ranks from). A spelling with two targets that each reach `min_pairs` has
neither (`conflicts`); a common word needs `COMMON_MIN_PAIRS` (`refused`).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from healing import corrections, splits
from healing.store import Store, now_ms

# Hex's own limits (`src/dictation/vocabulary.rs`; a test reads them from the source).
MAX_TERM_CHARS = 64
MAX_SOUNDS_LIKE = 16

MIN_PAIRS = 2
COMMON_MIN_PAIRS = 4
WINDOW = "30d"
HALF_LIFE_DAYS = 14.0
DAY_MS = 86_400_000
COMMON_WORDS_FILE = Path(__file__).with_name("common_words.txt")


def common_words() -> frozenset[str]:
    """Frequent English words, one lowercase word each in `common_words.txt` (swap the file for a
    longer list; nothing else reads its size)."""
    words: set[str] = set()
    for line in COMMON_WORDS_FILE.read_text().splitlines():
        if not line.startswith("#"):
            words.update(line.lower().split())
    return frozenset(words)


@dataclass(frozen=True)
class Candidate:
    term: str
    spellings: tuple[tuple[str, int], ...]  # (heard spelling, clips), most frequent first
    pair_count: int
    recent_uses: int
    score: float
    first_seen_ms: int
    last_seen_ms: int
    source_split: str  # 'train' or 'all'

    @property
    def sounds_like(self) -> list[str]:
        return [spelling for spelling, _ in self.spellings]


@dataclass
class Mined:
    candidates: list[Candidate] = field(default_factory=list)  # ranked: score, pairs, recency, term
    conflicts: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def norm(text: str) -> str:
    return " ".join(t.norm for t in corrections.tokens(text))


def _most_common(counter: Counter) -> str:
    """Highest count; a tie goes to the form with more capitals (a proper noun beats its lowercase),
    then to alphabetical order, so a run never depends on row order."""
    return min(counter, key=lambda s: (-counter[s], -sum(c.isupper() for c in s), s))


def rank_key(candidate: Candidate) -> tuple:
    return (-candidate.score, -candidate.pair_count, -candidate.last_seen_ms, candidate.term)


def mine(
    store: Store, *, min_pairs: int = MIN_PAIRS, window_ms: int | None = None, train_only: bool = False,
    now: int | None = None,
) -> Mined:
    """The ranked candidates. Nothing is written. `train_only` counts only the clips a training run
    may take (`splits.Zones.may_train`: outside the test window and every test cluster), for pairs and
    for use alike, so a term learned from the rolling clips never reaches a measurement on them (T3.2)."""
    if min_pairs < MIN_PAIRS:
        raise ValueError(f"--min-pairs must be at least {MIN_PAIRS}")
    now = now_ms() if now is None else now
    window_ms = 30 * DAY_MS if window_ms is None else window_ms
    allowed = splits.Zones.load(store.db, now=now).may_train if train_only else (lambda clip: True)
    source = "train" if train_only else "all"
    result = Mined()

    # (heard norm, term norm) -> clips; and each form as typed, for display and casing
    clips_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    heard_forms: dict[str, Counter] = defaultdict(Counter)
    term_forms: dict[str, Counter] = defaultdict(Counter)
    seen_ms: dict[tuple[str, str], list[int]] = defaultdict(list)
    rows = store.db.execute(
        "SELECT e.raw_from, e.to_text, c.clip_id, c.timestamp_ms FROM edits e JOIN clips c ON c.clip_id = e.clip_id"
        " WHERE e.kind = 'acoustic' AND c.label = 'strong' AND c.kind = 'dictation' ORDER BY c.timestamp_ms, c.clip_id, e.idx"
    ).fetchall()
    for row in rows:
        if not allowed(row["clip_id"]):
            continue
        heard, term = norm(row["raw_from"]), norm(row["to_text"])
        if not heard or not term:
            continue
        key = (heard, term)
        if row["clip_id"] in clips_of[key]:
            continue  # one dictation is one piece of evidence, however often it repeats the edit
        clips_of[key].add(row["clip_id"])
        seen_ms[key].append(row["timestamp_ms"])
        heard_forms[heard][row["raw_from"]] += 1
        term_forms[term][row["to_text"]] += 1

    # a spelling with two targets that each recur has neither; a target equal to the spelling (a casing
    # only fix) is no candidate but still contests it
    targets: dict[str, list[str]] = defaultdict(list)
    for (heard, term), clips in clips_of.items():
        if len(clips) >= min_pairs:
            targets[heard].append(term)
    kept: dict[str, dict[str, int]] = defaultdict(dict)  # term norm -> heard norm -> clips
    for heard, terms in sorted(targets.items()):
        if len(terms) > 1:
            result.conflicts.append(f"{heard!r} is corrected to " + " and ".join(sorted(repr(_most_common(term_forms[t])) for t in terms)))
            continue
        if terms[0] != heard:
            kept[terms[0]][heard] = len(clips_of[(heard, terms[0])])

    common = common_words()
    uses = _uses(store, allowed, now, window_ms)
    for term_norm, pairs in kept.items():
        term = _most_common(term_forms[term_norm])
        pair_count = sum(pairs.values())
        if len(term) > MAX_TERM_CHARS:
            result.skipped.append(f"{term[:20]}… is longer than {MAX_TERM_CHARS} characters")
            continue
        if " " not in term_norm and term_norm in common and pair_count < COMMON_MIN_PAIRS:
            result.refused.append(f"{term} is a common word: {pair_count} of {COMMON_MIN_PAIRS} pairs")
            continue
        spellings = []
        for heard, count in pairs.items():
            form = _most_common(heard_forms[heard])
            if len(form) <= MAX_TERM_CHARS and form.lower() != term.lower():
                spellings.append((form, count))
        spellings.sort(key=lambda s: (-s[1], s[0]))
        stamps = [t for heard in pairs for t in seen_ms[(heard, term_norm)]]
        recent, score = uses(term_norm)
        result.candidates.append(Candidate(
            term, tuple(spellings[:MAX_SOUNDS_LIKE]), pair_count, recent, score, min(stamps), max(stamps), source,
        ))
    result.candidates.sort(key=rank_key)
    return result


def _uses(store: Store, allowed, now: int, window_ms: int):
    """`term norm -> (clips in the window that hold it, their decayed weight)`."""
    texts: list[tuple[str, float]] = []
    for row in store.db.execute(
        "SELECT clip_id, timestamp_ms, final_text, reference_text FROM clips WHERE kind = 'dictation'"
        " AND timestamp_ms >= ? AND timestamp_ms <= ?", (now - window_ms, now),
    ):
        if allowed(row["clip_id"]):
            body = " " + " ".join(norm(t) for t in (row["final_text"], row["reference_text"]) if t) + " "
            texts.append((body, 0.5 ** ((now - row["timestamp_ms"]) / DAY_MS / HALF_LIFE_DAYS)))

    def uses(term_norm: str) -> tuple[int, float]:
        needle = f" {term_norm} "
        weights = [w for body, w in texts if needle in body]
        return len(weights), sum(weights)

    return uses


def save(store: Store, mined: Mined, source: str) -> None:
    """Replace `vocab_candidates` with this run (it holds only the latest). In one transaction."""
    with store.immediate() as db:
        db.execute("DELETE FROM vocab_candidates")
        for c in mined.candidates:
            db.execute(
                "INSERT INTO vocab_candidates (term, sounds_like, pair_count, recent_uses, score, first_seen_ms,"
                " last_seen_ms, source_split) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (c.term, json.dumps(c.sounds_like), c.pair_count, c.recent_uses, c.score, c.first_seen_ms,
                 c.last_seen_ms, source),
            )


def load_candidates(store: Store) -> list[Candidate]:
    """The saved candidates in rank order (spelling counts are not stored)."""
    out = [
        Candidate(r["term"], tuple((s, 0) for s in json.loads(r["sounds_like"])), r["pair_count"], r["recent_uses"],
                  r["score"], r["first_seen_ms"], r["last_seen_ms"], r["source_split"])
        for r in store.db.execute("SELECT * FROM vocab_candidates")
    ]
    return sorted(out, key=rank_key)


def line(c: Candidate) -> str:
    """`Kit ← kip ×3, kipp ×2` with its use and last-seen date."""
    from datetime import datetime, timezone

    heard = ", ".join(f"{s} ×{n}" for s, n in c.spellings) or "(no spelling)"
    last = datetime.fromtimestamp(c.last_seen_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    return f"{c.term} ← {heard}  [{c.pair_count} pairs, {c.recent_uses} recent, score {c.score:.2f}, last {last}]"


def main(args) -> int:
    import sys

    from healing import labels
    from healing.collect import since_to_ms
    from healing.store import StoreError, process_lock

    now = now_ms()
    start = since_to_ms(args.window, now)
    if start is None:
        print(f"error: cannot read --window {args.window!r}", file=sys.stderr)
        return 2
    try:
        with process_lock():
            store = Store()
            try:
                labels.label_all(store)  # the evidence is derived: bring it up to date, as `label all` does
                mined = mine(store, min_pairs=args.min_pairs, window_ms=now - start, train_only=args.train_only, now=now)
                save(store, mined, "train" if args.train_only else "all")
            finally:
                store.close()
    except (StoreError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2 if isinstance(error, ValueError) else 1
    if args.json:
        print(json.dumps([
            {"term": c.term, "sounds_like": c.sounds_like, "pair_count": c.pair_count, "recent_uses": c.recent_uses,
             "score": c.score, "first_seen_ms": c.first_seen_ms, "last_seen_ms": c.last_seen_ms,
             "source_split": c.source_split} for c in mined.candidates], indent=2))
    else:
        for c in mined.candidates:
            print(line(c))
        print(f"{len(mined.candidates)} candidate(s)" + (" (train clips only)" if args.train_only else ""))
    for note in mined.conflicts:
        print(f"conflict, no candidate: {note}", file=sys.stderr)
    for note in mined.refused + mined.skipped:
        print(f"not a candidate: {note}", file=sys.stderr)
    return 0
