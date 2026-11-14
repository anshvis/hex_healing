"""`healing prune`: cluster new clips, keep k audio per cluster, hold the byte budget.

Audio is evicted per file (`audio.evicted_at_ms`); clip rows and text are never deleted. Strong,
verified and golden clips are protected: never evicted, never counted toward k. Audio younger than
7 days is never evicted (a clip can still become strong). The unit of eviction is one audio file,
so a file several clips share is protected, or young, if any of them is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import soundfile

from healing import dedupe
from healing.store import Store

DEFAULT_BUDGET = 5 * 1024**3
DEFAULT_K = 3
MIN_AGE_MS = 7 * 24 * 3600 * 1000
HALF_LIFE_DAYS = 90
PROTECTED_LABELS = ("strong", "verified")
DAY_MS = 24 * 3600 * 1000


class PruneError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


@dataclass
class Unit:
    sha256: str
    path: str
    bytes: int
    clip_ids: list[str]
    newest_id: str
    timestamp_ms: int
    cluster_id: str | None
    protected: bool
    redundancy: float | None = None  # min over its clusters; None until a cluster judges it
    cluster_ids: set[str] = field(default_factory=set)


@dataclass
class Eviction:
    clip_id: str
    sha256: str
    cluster_id: str | None
    label: str
    bytes: int
    reason: str

    def __str__(self) -> str:
        return (
            f"evict {self.clip_id} cluster={self.cluster_id or '-'} label={self.label} "
            f"bytes={self.bytes} reason={self.reason}"
        )


@dataclass
class Result:
    evictions: list[Eviction] = field(default_factory=list)
    evicted_bytes: int = 0
    total_bytes: int = 0
    budget: int = 0
    over_by: int = 0

    def summary(self) -> str:
        return (
            f"evicted {len(self.evictions)} ({self.evicted_bytes} bytes), "
            f"total {self.total_bytes} / budget {self.budget}"
        )


def parse_size(text: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]i?B?|B)?", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"not a size: {text!r}")
    unit = (match.group(2) or "B").upper().rstrip("B").rstrip("I")
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[unit]
    return int(float(match.group(1)) * scale)


def age_decay(age_ms: int) -> float:
    """How much of a clip's value has decayed: 0 when new, 0.5 at the half-life, towards 1."""
    return 1 - 0.5 ** (max(age_ms, 0) / DAY_MS / HALF_LIFE_DAYS)


def _load_units(store: Store) -> list[Unit]:
    rows = store.db.execute(
        "SELECT c.clip_id, c.timestamp_ms, c.cluster_id, c.label, c.golden, c.audio_sha256,"
        " a.path, a.bytes FROM clips c JOIN audio a ON a.sha256 = c.audio_sha256"
        " WHERE a.evicted_at_ms IS NULL ORDER BY c.timestamp_ms, c.clip_id"
    ).fetchall()
    units: dict[str, Unit] = {}
    for row in rows:
        protected = row["golden"] == 1 or row["label"] in PROTECTED_LABELS
        unit = units.get(row["audio_sha256"])
        if unit is None:
            units[row["audio_sha256"]] = Unit(
                row["audio_sha256"], row["path"], row["bytes"], [row["clip_id"]],
                row["clip_id"], row["timestamp_ms"], row["cluster_id"], protected,
                cluster_ids={row["cluster_id"]} if row["cluster_id"] else set(),
            )
            continue
        unit.clip_ids.append(row["clip_id"])
        unit.protected = unit.protected or protected
        unit.newest_id, unit.timestamp_ms = row["clip_id"], row["timestamp_ms"]  # rows ascend
        unit.cluster_id = row["cluster_id"] or unit.cluster_id
        if row["cluster_id"]:
            unit.cluster_ids.add(row["cluster_id"])
    return list(units.values())


def _release_stale(store: Store) -> None:
    """Re-open clips whose text no longer fits their cluster (Replace / Transcribe again).

    A representative whose text changed refreshes its cluster's text; members that no longer
    match that text lose their cluster and are clustered again with the new clips.
    """
    for row in store.db.execute(
        "SELECT k.cluster_id, c.raw_text FROM clusters k JOIN clips c ON c.clip_id = k.cluster_id"
    ).fetchall():
        fresh = dedupe.normalize(row["raw_text"])
        if fresh:
            store.db.execute(
                "UPDATE clusters SET norm_text = ? WHERE cluster_id = ? AND norm_text != ?",
                (fresh, row["cluster_id"], fresh),
            )
        else:  # emptied text belongs to no cluster; the clusters row stays as a key
            store.db.execute("UPDATE clips SET cluster_id = NULL WHERE clip_id = ?", (row["cluster_id"],))
    texts = {r["cluster_id"]: r["norm_text"] for r in store.db.execute("SELECT * FROM clusters")}
    for row in store.db.execute(
        "SELECT clip_id, cluster_id, raw_text FROM clips WHERE cluster_id IS NOT NULL"
    ).fetchall():
        if row["clip_id"] == row["cluster_id"]:
            continue
        if not dedupe.compatible(dedupe.normalize(row["raw_text"]), texts.get(row["cluster_id"], "")):
            store.db.execute("UPDATE clips SET cluster_id = NULL WHERE clip_id = ?", (row["clip_id"],))


def _cluster_new_clips(store: Store) -> None:
    _release_stale(store)
    pending = [
        dedupe.Clip(r["clip_id"], r["timestamp_ms"], r["raw_text"], r["audio_ms"])
        for r in store.db.execute(
            "SELECT clip_id, timestamp_ms, raw_text, audio_ms FROM clips WHERE cluster_id IS NULL"
        )
        if dedupe.normalize(r["raw_text"])
    ]
    if not pending:
        return
    reps = [
        dedupe.Clip(r["cluster_id"], r["timestamp_ms"], r["norm_text"], r["audio_ms"])
        for r in store.db.execute(
            "SELECT k.cluster_id, k.norm_text, c.timestamp_ms, c.audio_ms FROM clusters k"
            " LEFT JOIN clips c ON c.clip_id = k.cluster_id ORDER BY k.rowid"
        )
    ]
    assignments = dedupe.cluster(pending, reps)
    known = {rep.clip_id for rep in reps}
    by_id = {c.clip_id: c for c in pending}
    for clip_id, cluster_id in assignments.items():
        store.db.execute("UPDATE clips SET cluster_id = ? WHERE clip_id = ?", (cluster_id, clip_id))
        if cluster_id not in known:
            known.add(cluster_id)
            store.db.execute(
                "INSERT INTO clusters (cluster_id, norm_text, size, audio_kept, last_seen_ms)"
                " VALUES (?, ?, 0, 0, 0)",
                (cluster_id, dedupe.normalize(by_id[cluster_id].text)),
            )


def _features(store: Store, unit: Unit) -> np.ndarray:
    try:
        samples, _ = soundfile.read(store.root / unit.path, dtype="int16", always_2d=False)
        return dedupe.mfcc_vector(samples)
    except (OSError, RuntimeError):
        return np.zeros(2 * dedupe.N_MFCC)  # unreadable audio looks like every other one


def _refresh_clusters(store: Store) -> None:
    store.db.execute(
        "UPDATE clusters SET"
        " size = (SELECT COUNT(*) FROM clips c WHERE c.cluster_id = clusters.cluster_id),"
        " last_seen_ms = COALESCE((SELECT MAX(timestamp_ms) FROM clips c"
        "   WHERE c.cluster_id = clusters.cluster_id), 0)"
    )
    kept = {
        row[0]: row[1]
        for row in store.db.execute(
            "SELECT c.cluster_id, COUNT(DISTINCT c.audio_sha256) FROM clips c"
            " JOIN audio a ON a.sha256 = c.audio_sha256 WHERE a.evicted_at_ms IS NULL"
            " AND c.cluster_id IS NOT NULL AND c.golden = 0 AND c.label NOT IN ('strong','verified')"
            " AND NOT EXISTS (SELECT 1 FROM clips p WHERE p.audio_sha256 = c.audio_sha256"
            "   AND (p.golden = 1 OR p.label IN ('strong','verified')))"
            " GROUP BY c.cluster_id"
        )
    }
    for cluster_id, in store.db.execute("SELECT cluster_id FROM clusters").fetchall():
        store.db.execute(
            "UPDATE clusters SET audio_kept = ? WHERE cluster_id = ?", (kept.get(cluster_id, 0), cluster_id)
        )


def real_file_under(home, path) -> bool:
    """True when `path` is a regular file and every component from `home` down is a real directory
    (checked with lstat, never resolved), so a link at any level is refused (G-6)."""
    import stat as _stat

    try:
        relative = path.relative_to(home)
        current = home
        for part in relative.parts[:-1]:
            current = current / part
            if not _stat.S_ISDIR(current.lstat().st_mode):
                return False
        return _stat.S_ISREG(path.lstat().st_mode)
    except (OSError, ValueError):
        return False


def _unlink_inside(home, path) -> None:
    """Delete only a regular file reached through real directories under `home` (G-6)."""
    if real_file_under(home, path):
        try:
            path.unlink()
        except OSError:
            pass


def prune(
    store: Store, budget: int = DEFAULT_BUDGET, k: int = DEFAULT_K, dry_run: bool = False,
    now: int | None = None, lowering: bool = False,
) -> Result:
    from healing.store import now_ms

    now = now_ms() if now is None else now
    units = _load_units(store)
    result = Result(budget=budget, total_bytes=sum(u.bytes for u in units))
    protected_bytes = sum(u.bytes for u in units if u.protected)
    if lowering:  # never below what is protected: evict every eligible file, then report
        budget = max(budget, protected_bytes)
    if protected_bytes > budget:
        raise PruneError(
            f"protected audio is {protected_bytes} bytes, over the budget {budget} by "
            f"{protected_bytes - budget}; nothing evicted",
            4,
        )
    if units and not lowering and budget < min(u.bytes for u in units):
        raise PruneError(f"budget {budget} is smaller than one audio file", 4)

    try:
        _cluster_new_clips(store)
        units = _load_units(store)  # now with cluster ids
        young = {u.sha256 for u in units if now - u.timestamp_ms < MIN_AGE_MS}
        chosen: dict[str, str] = {}  # sha -> reason, for audio to evict

        by_cluster: dict[str, list[Unit]] = {}
        for unit in units:
            for cluster_id in unit.cluster_ids:
                by_cluster.setdefault(cluster_id, []).append(unit)
        for cluster_id, members in sorted(by_cluster.items()):
            _spread(store, members, k, young, chosen)
        for unit in units:
            if unit.redundancy is None:
                unit.redundancy = 0.0

        remaining = result.total_bytes - sum(u.bytes for u in units if u.sha256 in chosen)
        if remaining > budget:
            candidates = [
                u for u in units
                if not u.protected and u.sha256 not in young and u.sha256 not in chosen
            ]
            candidates.sort(
                key=lambda u: (
                    -u.redundancy * age_decay(now - u.timestamp_ms), u.timestamp_ms, u.newest_id
                )
            )
            for unit in candidates:
                if remaining <= budget:
                    break
                chosen[unit.sha256] = "budget"
                remaining -= unit.bytes
        result.over_by = max(remaining - budget, 0)

        labels = {
            r["clip_id"]: r["label"]
            for r in store.db.execute("SELECT clip_id, label FROM clips")
        }
        for unit in units:
            if unit.sha256 in chosen:
                result.evictions.append(
                    Eviction(unit.newest_id, unit.sha256, unit.cluster_id, labels[unit.newest_id],
                             unit.bytes, chosen[unit.sha256])
                )
        result.evicted_bytes = sum(e.bytes for e in result.evictions)
        result.total_bytes -= result.evicted_bytes

        for unit in units:
            for clip_id in unit.clip_ids:
                store.db.execute(
                    "UPDATE clips SET redundancy = ? WHERE clip_id = ?", (unit.redundancy, clip_id)
                )
        for eviction in result.evictions:
            store.db.execute(
                "UPDATE audio SET evicted_at_ms = ? WHERE sha256 = ?", (now, eviction.sha256)
            )
        _refresh_clusters(store)
    except BaseException:
        store.db.rollback()
        raise
    if dry_run:
        store.db.rollback()
        return result
    store.db.commit()
    for row in store.db.execute("SELECT path FROM audio WHERE evicted_at_ms IS NOT NULL").fetchall():
        _unlink_inside(store.root.parent, store.root / row["path"])
    if result.over_by:
        raise PruneError(
            f"still over the budget by {result.over_by} bytes: what is left is protected or "
            "younger than 7 days",
            4,
        )
    return result


def _spread(store: Store, members: list[Unit], k: int, young: set[str], chosen: dict[str, str]) -> None:
    """Keep k unprotected files per cluster by acoustic spread; set every member's redundancy."""
    if len(members) < 2:
        for unit in members:
            unit.redundancy = 0.0  # a sole member is not redundant
        return
    members.sort(key=lambda u: (u.timestamp_ms, u.newest_id))
    dist = dedupe.distances(np.array([_features(store, u) for u in members]))
    # Fixed units always survive and take no k slot: protected, young (can still become strong),
    # and audio another cluster also needs. Survivors are fixed + at most k, so a rerun agrees.
    fixed = [
        i for i, u in enumerate(members)
        if u.protected or u.sha256 in young or len(u.cluster_ids) > 1
    ]
    newest_first = sorted(
        (i for i in range(len(members)) if i not in fixed),
        key=lambda i: (-members[i].timestamp_ms, members[i].newest_id),
    )
    keep = fixed + dedupe.spread_pick(dist, newest_first, k, fixed)
    for i in newest_first:
        if i not in keep:
            chosen[members[i].sha256] = "k"
    for i, unit in enumerate(members):
        others = [j for j in keep if j != i]
        value = 1 / (1 + min(dist[i][j] for j in others)) if others else 0.0
        unit.redundancy = value if unit.redundancy is None else min(unit.redundancy, value)
