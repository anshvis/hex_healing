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


def check_open(store: Store) -> None:
    current = splits.golden_row(store.db)
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
    if splits.golden_row(db) is None:
        db.execute(
            "INSERT INTO splits (split_id, kind, created_at_ms) VALUES (?, 'golden', ?)",
            (splits.GOLDEN_ID, now_ms()),
        )
    db.execute("INSERT OR IGNORE INTO split_members (split_id, clip_id) VALUES (?, ?)", (splits.GOLDEN_ID, clip_id))
    db.execute("UPDATE clips SET golden = 1 WHERE clip_id = ?", (clip_id,))
    count = len(splits.golden_members(store.db))
    db.execute("UPDATE splits SET n_clips = ? WHERE split_id = ?", (count, splits.GOLDEN_ID))
    db.commit()
    freeze_if_full(store)
    return True


def freeze_if_full(store: Store) -> bool:
    """Freeze a full, unfrozen golden set. Idempotent, so a freeze that failed is retried by the next
    `add` or `run` and the confirmed 200th clip never strands the set."""
    current = splits.golden_row(store.db)
    if current is None or current["frozen"] or len(splits.golden_members(store.db)) < splits.GOLDEN_SIZE:
        return False
    freeze(store)
    return True


def freeze(store: Store) -> None:
    db = store.db
    ids = splits.golden_members(store.db)
    rows = [db.execute("SELECT * FROM clips WHERE clip_id = ?", (i,)).fetchone() for i in ids]
    items = splits.entries(store, rows, splits.Zones.load(db, None))
    if len(items) != len(ids):
        raise GoldenError("a golden clip has no usable reference; refusing to freeze")
    info = splits.write_and_pin(store, splits.GOLDEN_ID, items)
    db.execute(
        "UPDATE splits SET manifest_path = ?, manifest_sha256 = ?, sidecar_sha256 = ?, n_clips = ?,"
        " n_words = ?, frozen = 1 WHERE split_id = ?",
        (str(info.path), info.manifest_sha256, info.sidecar_sha256, len(ids), info.n_words, splits.GOLDEN_ID),
    )
    db.commit()


def run(store: Store, n: int = 20, *, player=review.afplay, ask=input, out=print, seed: int | None = None) -> review.ReviewResult:
    freeze_if_full(store)  # a freeze that failed last time
    check_open(store)
    db = store.db
    zones = splits.Zones.load(db)
    clusters = zones.clusters
    members = splits.golden_members(store.db)
    taken_clusters = {clusters[m] for m in members}
    taken = Counter(_stratum(db.execute("SELECT * FROM clips WHERE clip_id = ?", (m,)).fetchone()) for m in members)
    # `Zones.may_test`, and clear of what a rolling or edited set holds: a golden clip and a rolling clip
    # are both test clips, and the registry keeps their clusters apart (`split verify` checks).
    pool = review.candidates(store, "random", zones=zones, apart_from_registered=True)
    rows = stratified_pick(pool, taken, taken_clusters, clusters, min(n, splits.GOLDEN_SIZE - len(members)),
                           random.Random(seed))
    if not rows:
        out("nothing to review: no unreviewed clips with audio in a new cluster")
        return review.ReviewResult()

    def after(clip, action: str) -> bool:
        if action != "reject":
            add(store, clip["clip_id"])
        current = splits.golden_row(store.db)
        return current is not None and bool(current["frozen"])

    result = review.review_clips(store, rows, "random", player=player, ask=ask, out=out, after=after)
    current = splits.golden_row(store.db)
    out(f"golden: {current['n_clips'] if current else 0} of {splits.GOLDEN_SIZE}"
        + (" (frozen)" if current and current["frozen"] else " (pending)"))
    return result


def main(args) -> int:
    from healing.store import StoreError

    try:
        # No `process_lock`, for the reason `review.main` gives: answers are short transactions.
        store = Store(sweep=False)
        try:
            result = run(store, args.n)
        finally:
            store.close()
    except (GoldenError, review.ReviewError, StoreError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result.text())
    return 0
