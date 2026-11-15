"""`healing golden review`: the frozen golden set (D-17, G-10).

You confirm clips in a terminal review (play, see the text, accept or fix); each confirmed clip
joins `golden-v1`. At `splits.GOLDEN_SIZE` (200) the set freezes on its own: its manifest and WAVs
are written, its SHA-256 is pinned in `sets/golden-v1/manifest.sha256` (the registry, once M5
lands) and in the `splits` row, and the schema refuses every later change to that row or its
members. Until then it is pending and evaluates as n/a.

Candidates are random clips, one per split-cluster and spread across apps and months, so 200 clips
are not 200 sayings of one phrase.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from datetime import datetime, timezone

from healing import review, splits
from healing.store import Store, now_ms
from healing.wer import normalized_words


class GoldenError(RuntimeError):
    pass


def _stratum(row) -> tuple[str, str]:
    month = datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc).strftime("%Y-%m")
    return (row["application"] or "(unknown)", month)


def row(store: Store):
    return store.db.execute("SELECT * FROM splits WHERE split_id = ?", (splits.GOLDEN_ID,)).fetchone()


def member_ids(store: Store) -> list[str]:
    return [r["clip_id"] for r in store.db.execute(
        "SELECT clip_id FROM split_members WHERE split_id = ? ORDER BY clip_id", (splits.GOLDEN_ID,))]


def check_open(store: Store) -> None:
    current = row(store)
    if current is not None and current["frozen"]:
        raise GoldenError(
            f"golden is frozen at {current['n_clips']} clips ({str(current['manifest_sha256'])[:12]}) and never changes"
        )


def stratified_pick(rows: list, taken: Counter, taken_clusters: set[str], clusters: dict[str, str],
                    n: int, rng: random.Random) -> list:
    """`n` clips, each from a split-cluster not yet used, always from the least-covered app and month."""
    pools: dict[tuple[str, str], list] = {}
    for candidate in rows:
        if clusters[candidate["clip_id"]] not in taken_clusters:
            pools.setdefault(_stratum(candidate), []).append(candidate)
    chosen = []
    used = set(taken_clusters)
    counts = Counter(taken)
    while len(chosen) < n:
        for key in list(pools):
            pools[key] = [c for c in pools[key] if clusters[c["clip_id"]] not in used]
            if not pools[key]:
                del pools[key]
        if not pools:
            break
        fewest = min(counts[key] for key in pools)
        key = rng.choice(sorted(k for k in pools if counts[k] == fewest))
        pick = rng.choice(pools[key])
        chosen.append(pick)
        used.add(clusters[pick["clip_id"]])
        counts[key] += 1
    return chosen


def add(store: Store, clip_id: str) -> bool:
    """Put a confirmed clip in golden; freezes the set when it is full. False if it can't hold one."""
    db = store.db
    clip = db.execute("SELECT * FROM clips WHERE clip_id = ?", (clip_id,)).fetchone()
    if clip["label"] != "verified" or not normalized_words(clip["reference_text"] or ""):
        return False
    check_open(store)
    if row(store) is None:
        db.execute(
            "INSERT INTO splits (split_id, kind, created_at_ms) VALUES (?, 'golden', ?)",
            (splits.GOLDEN_ID, now_ms()),
        )
    db.execute("INSERT OR IGNORE INTO split_members (split_id, clip_id) VALUES (?, ?)", (splits.GOLDEN_ID, clip_id))
    db.execute("UPDATE clips SET golden = 1 WHERE clip_id = ?", (clip_id,))
    count = len(member_ids(store))
    db.execute("UPDATE splits SET n_clips = ? WHERE split_id = ?", (count, splits.GOLDEN_ID))
    db.commit()
    if count >= splits.GOLDEN_SIZE:
        freeze(store)
    return True


def freeze(store: Store) -> None:
    db = store.db
    ids = member_ids(store)
    rows = [db.execute("SELECT * FROM clips WHERE clip_id = ?", (i,)).fetchone() for i in ids]
    clusters, sessions = splits.clip_clusters(db), splits.sessions(db)
    items = splits.entries(store, rows, clusters, sessions)
    if len(items) != len(ids):
        raise GoldenError("a golden clip has no usable reference; refusing to freeze")
    path, digest, words = splits.write_and_pin(store, splits.GOLDEN_ID, items)
    db.execute(
        "UPDATE splits SET manifest_path = ?, manifest_sha256 = ?, n_clips = ?, n_words = ?, frozen = 1"
        " WHERE split_id = ?",
        (str(path), digest, len(ids), words, splits.GOLDEN_ID),
    )
    db.commit()


def run(store: Store, n: int = 20, *, player=review.afplay, ask=input, out=print, seed: int | None = None) -> review.ReviewResult:
    check_open(store)
    db = store.db
    clusters = splits.clip_clusters(db)
    members = member_ids(store)
    taken_clusters = {clusters[m] for m in members}
    taken = Counter(_stratum(db.execute("SELECT * FROM clips WHERE clip_id = ?", (m,)).fetchone()) for m in members)
    pool = review.candidates(store, "random")
    rows = stratified_pick(pool, taken, taken_clusters, clusters, min(n, splits.GOLDEN_SIZE - len(members)),
                           random.Random(seed))
    if not rows:
        out("nothing to review: no unreviewed clips with audio in a new cluster")
        return review.ReviewResult()

    def after(clip, action: str) -> bool:
        if action != "reject":
            add(store, clip["clip_id"])
        current = row(store)
        return current is not None and bool(current["frozen"])

    result = review.review_clips(store, rows, "random", player=player, ask=ask, out=out, after=after)
    current = row(store)
    out(f"golden: {current['n_clips'] if current else 0} of {splits.GOLDEN_SIZE}"
        + (" (frozen)" if current and current["frozen"] else " (pending)"))
    return result


def main(args) -> int:
    from healing.store import StoreError

    try:
        store = Store(sweep=False)
        try:
            result = run(store, args.n)
        finally:
            store.close()
    except (GoldenError, review.ReviewError, StoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result.text())
    return 0
