"""`healing split build|verify`: test sets with no leakage into training.

Sets (T3.2):
- rolling: sessions after the parent version's cut-off, clips sampled at random and verified by you;
- golden: 200 verified clips, frozen and pinned (`healing/golden.py`);
- edited: the rolling sessions' strong clips, scored only at the edited words (T3.3);
- general: T0.3's LibriSpeech and AMI manifest (not made from the dataset);
- train: trainable clips sharing no split-cluster with any test set.

A split-cluster is a connected component of near-duplicate `reference_text` (union-find, token
3-gram Jaccard >= 0.8, no duration condition). It is deliberately not T1.3's storage cluster:
those assign each clip to one representative and gate on duration, so "send it to kip" said slowly
and "send it to kit" said fast would land apart and straddle train and test. A clip also joins
through its `raw_text`, so a clip whose correction is "kit" meets another clip that was merely
heard as "kit".
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile

from healing import dedupe, labels, manifest
from healing.paths import healing_home
from healing.sets import general
from healing.store import Store, now_ms
from healing.wer import normalized_words

SESSION_GAP_MS = 10 * 60 * 1000  # D-10: a session is a run of dictations with no gap over 10 minutes
DEFAULT_CUTOFF_MS = 14 * 24 * 3600 * 1000  # v0's parent has no snapshot: the last 14 days
ROLLING_MIN_WORDS = 1500  # under this the rolling set is too small to gate (D-12)
EXCLUDED_WARN = 0.30
GOLDEN_SIZE = 200  # frozen once it reaches this many confirmed clips
GOLDEN_ID = "golden-v1"
GENERAL_ID = "general-v1"


class SplitError(RuntimeError):
    pass


def sets_dir() -> Path:
    return healing_home() / "sets"


# --- split-clusters ------------------------------------------------------------------------


def split_clusters(clip_texts: dict[str, list[str]]) -> dict[str, str]:
    """clip_id -> its component's root (the smallest clip_id in it) for every clip given."""
    parent = {clip: clip for clip in clip_texts}

    def find(clip: str) -> str:
        while parent[clip] != clip:
            parent[clip] = parent[parent[clip]]
            clip = parent[clip]
        return clip

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owners: dict[str, list[str]] = {}
    for clip, texts in clip_texts.items():
        for text in texts:
            key = dedupe.normalize(text)
            if key:
                owners.setdefault(key, []).append(clip)
    for clips in owners.values():
        for other in clips[1:]:
            union(clips[0], other)
    grams: dict[str, frozenset[str]] = {}
    buckets: dict[tuple[int, bytes], list[str]] = {}
    for key in sorted(owners):
        found = dedupe.shingles(key)  # texts under 3 tokens match by equality only
        if found is None:
            continue
        grams[key] = found
        for band, value in enumerate(dedupe._bands(dedupe.signature(found))):
            buckets.setdefault((band, value), []).append(key)
    for keys in buckets.values():
        for index, key in enumerate(keys):
            for other in keys[index + 1 :]:
                if find(owners[key][0]) != find(owners[other][0]) and dedupe.jaccard(grams[key], grams[other]) >= dedupe.JACCARD:
                    union(owners[key][0], owners[other][0])
    return {clip: find(clip) for clip in clip_texts}


def clip_clusters(db: sqlite3.Connection) -> dict[str, str]:
    texts = {
        row["clip_id"]: [t for t in (row["reference_text"] or row["raw_text"], row["raw_text"]) if t]
        for row in db.execute("SELECT clip_id, reference_text, raw_text FROM clips")
    }
    return split_clusters(texts)


# --- sessions and the cut-off -----------------------------------------------------------------


def sessions(db: sqlite3.Connection) -> dict[str, tuple[str, int]]:
    """clip_id -> (session id, session start ms): runs of dictations with no gap over 10 minutes."""
    result: dict[str, tuple[str, int]] = {}
    previous, start = None, 0
    for row in db.execute("SELECT clip_id, timestamp_ms FROM clips ORDER BY timestamp_ms, clip_id"):
        if previous is None or row["timestamp_ms"] - previous > SESSION_GAP_MS:
            start = row["timestamp_ms"]
        result[row["clip_id"]] = (f"s{start}", start)
        previous = row["timestamp_ms"]
    return result


def ensure_cutoff(store: Store, parent: str, override_ms: int | None = None, now: int | None = None) -> int:
    """The parent version's snapshot cut-off, in UTC ms, kept so it never drifts between calls.

    `override_ms` replaces it. Otherwise the cut-off of the parent's rolling set, else 14 days
    before now (v0's parent has no snapshot). A fresh one is recorded in a pending rolling row so
    reviews done before the first `split build` stay on the right side of it.
    """
    db = store.db
    row = db.execute(
        "SELECT split_id, cutoff_ms FROM splits WHERE kind = 'rolling' AND parent_version = ?"
        " ORDER BY created_at_ms DESC, split_id DESC LIMIT 1",
        (parent,),
    ).fetchone()
    if override_ms is None and row is not None:
        return row["cutoff_ms"]
    when = now_ms() if now is None else now
    cutoff = override_ms if override_ms is not None else when - DEFAULT_CUTOFF_MS
    if row is not None:
        # A built rolling set keeps the cut-off it was built with; `split build` replaces it.
        if row["split_id"].endswith("-pending"):
            db.execute("UPDATE splits SET cutoff_ms = ? WHERE split_id = ?", (cutoff, row["split_id"]))
    else:
        db.execute(
            "INSERT INTO splits (split_id, kind, parent_version, cutoff_ms, created_at_ms)"
            " VALUES (?, 'rolling', ?, ?, ?)",
            (f"rolling-{parent}-pending", parent, cutoff, when),
        )
    db.commit()
    return cutoff


def golden_clusters(db: sqlite3.Connection, clusters: dict[str, str]) -> set[str]:
    members = db.execute(
        "SELECT clip_id FROM split_members WHERE split_id = ?", (GOLDEN_ID,)
    ).fetchall()
    return {clusters[m["clip_id"]] for m in members if m["clip_id"] in clusters}


def rolling_eligible_ids(store: Store, cutoff_ms: int) -> set[str]:
    """Clips that could sit in the rolling set by position alone: their whole session started after
    the cut-off, and no golden clip is a near-duplicate."""
    db = store.db
    clusters = clip_clusters(db)
    blocked = golden_clusters(db, clusters)
    return {
        clip for clip, (_, start) in sessions(db).items()
        if start > cutoff_ms and clusters[clip] not in blocked
    }


# --- build ---------------------------------------------------------------------------------


@dataclass
class SetInfo:
    split_id: str
    n_clips: int
    n_words: int
    manifest_sha256: str | None = None


@dataclass
class BuildResult:
    parent: str
    cutoff_ms: int
    rolling: SetInfo
    edited: SetInfo
    train: SetInfo
    golden: SetInfo | None
    golden_pending: int
    general: SetInfo | None
    trainable: int
    excluded_train: int
    warnings: list[str] = field(default_factory=list)

    @property
    def rolling_small(self) -> bool:
        return self.rolling.n_words < ROLLING_MIN_WORDS

    def text(self) -> str:
        lines = [f"parent {self.parent}, cut-off {self.cutoff_ms}"]
        for label, info in (("rolling", self.rolling), ("edited-span", self.edited), ("train", self.train)):
            lines.append(f"{label}: {info.n_clips} clips, {info.n_words} words ({info.split_id})")
        if self.golden:
            lines.append(f"golden: {self.golden.n_clips} clips, {self.golden.n_words} words, frozen")
        else:
            lines.append(f"golden: n/a, {self.golden_pending} of {GOLDEN_SIZE} confirmed (pending)")
        lines.append(
            f"general: {self.general.n_clips} clips ({self.general.split_id})" if self.general
            else "general: n/a (run `healing sets build-general`)"
        )
        lines.append(f"excluded_train_clips: {self.excluded_train} of {self.trainable} trainable")
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _pcm(store: Store, sha256: str) -> np.ndarray:
    data, _ = soundfile.read(store.audio_path(sha256), dtype="int16", always_2d=False)
    return data


def _terms(db: sqlite3.Connection, clip_id: str, reference: str) -> list[str]:
    """The corrected words (acoustic edits) that survive in the reference."""
    found: list[str] = []
    for edit in db.execute(
        "SELECT to_text FROM edits WHERE clip_id = ? AND kind = 'acoustic' ORDER BY idx", (clip_id,)
    ):
        term = edit["to_text"].strip()
        if term and manifest.term_count(reference, term) > 0 and term not in found:
            found.append(term)
    return found


def entries(store: Store, rows: list[sqlite3.Row], clusters: dict[str, str], sess: dict, *, edited: bool = False) -> list[dict]:
    """Manifest entries for `rows`, with the sidecar fields every set carries."""
    result = []
    for row in rows:
        reference = row["reference_text"] or ""
        if not normalized_words(reference):
            continue
        entry = {
            "id": row["clip_id"],
            "expected": reference,
            "terms": _terms(store.db, row["clip_id"], reference),
            "application": row["application"],
            "session": sess[row["clip_id"]][0],
            "source": "history",
            "timestamp_ms": row["timestamp_ms"],
            "label": row["label"],
            "split_cluster": clusters[row["clip_id"]],
        }
        if edited:
            entry["edited_positions"] = json.loads(row["edited_positions"])
        result.append(entry)
    return result


def write_and_pin(store: Store, split_id: str, items: list[dict]) -> tuple[Path, str, int]:
    out = sets_dir() / split_id
    if out.exists():
        shutil.rmtree(out)
    rows = {i["id"]: store.db.execute("SELECT audio_sha256 FROM clips WHERE clip_id = ?", (i["id"],)).fetchone() for i in items}
    digest, words = manifest.write_set(out, items, lambda clip: _pcm(store, rows[clip]["audio_sha256"]))
    return out / "manifest.json", digest, words


def _register(db: sqlite3.Connection, split_id: str, kind: str, parent: str | None, cutoff: int | None,
              created: int, path: Path | None, digest: str | None, members: list[str], words: int, frozen: int = 0) -> None:
    db.execute(
        "INSERT INTO splits (split_id, kind, parent_version, cutoff_ms, created_at_ms, manifest_path,"
        " manifest_sha256, frozen, n_clips, n_words) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (split_id, kind, parent, cutoff, created, str(path) if path else None, digest, frozen, len(members), words),
    )
    db.executemany("INSERT INTO split_members (split_id, clip_id) VALUES (?, ?)", [(split_id, m) for m in members])


def build(store: Store, parent: str = "current", cutoff_ms: int | None = None, now: int | None = None) -> BuildResult:
    db = store.db
    when = now_ms() if now is None else now
    cutoff = ensure_cutoff(store, parent, cutoff_ms, when)
    clusters = clip_clusters(db)
    sess = sessions(db)
    blocked = golden_clusters(db, clusters)
    stamp = datetime.fromtimestamp(when / 1000, timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    after = {clip for clip, (_, start) in sess.items() if start > cutoff}
    rolling_rows, edited_rows = [], []
    for row in db.execute(
        f"SELECT c.* FROM clips c WHERE {labels.HAS_AUDIO} AND c.kind = 'dictation' AND c.golden = 0"
        " ORDER BY c.timestamp_ms, c.clip_id"
    ):
        if row["clip_id"] not in after or clusters[row["clip_id"]] in blocked:
            continue
        if row["label"] == "verified" and row["review_strategy"] == "random":
            rolling_rows.append(row)
        elif row["label"] == "strong" and row["edited_positions"]:
            edited_rows.append(row)

    rolling_items = entries(store, rolling_rows, clusters, sess)
    edited_items = entries(store, edited_rows, clusters, sess, edited=True)
    test_clusters = {i["split_cluster"] for i in rolling_items + edited_items} | blocked
    trainable = labels.trainable(store)
    train_ids = [r["clip_id"] for r in trainable if clusters[r["clip_id"]] not in test_clusters]
    excluded = len(trainable) - len(train_ids)

    rolling_id, edited_id = f"rolling-{parent}-{stamp}", f"edited-{parent}-{stamp}"
    train_id = f"train-{parent}-{stamp}"
    old = db.execute(
        "SELECT split_id FROM splits WHERE frozen = 0 AND parent_version = ? AND kind IN ('rolling', 'edited', 'train')",
        (parent,),
    ).fetchall()
    # Write the folders first: a failure leaves the registry as it was.
    rolling_path, rolling_sha, rolling_words = (None, None, 0)
    edited_path, edited_sha, edited_words = (None, None, 0)
    if rolling_items:
        rolling_path, rolling_sha, rolling_words = write_and_pin(store, rolling_id, rolling_items)
    if edited_items:
        edited_path, edited_sha, edited_words = write_and_pin(store, edited_id, edited_items)

    for row in old:
        db.execute("DELETE FROM split_members WHERE split_id = ?", (row["split_id"],))
        db.execute("DELETE FROM splits WHERE split_id = ?", (row["split_id"],))
    _register(db, rolling_id, "rolling", parent, cutoff, when, rolling_path, rolling_sha,
              [i["id"] for i in rolling_items], rolling_words)
    _register(db, edited_id, "edited", parent, cutoff, when, edited_path, edited_sha,
              [i["id"] for i in edited_items], edited_words)
    kept = set(train_ids)
    _register(db, train_id, "train", parent, cutoff, when, None, None, train_ids,
              sum(len(normalized_words(r["reference_text"] or r["raw_text"])) for r in trainable if r["clip_id"] in kept))
    db.executemany("UPDATE clips SET split_cluster = ? WHERE clip_id = ?", [(c, k) for k, c in clusters.items()])
    general_info = _register_general(db, when)
    db.commit()
    for row in old:
        if row["split_id"] not in (rolling_id, edited_id):
            shutil.rmtree(sets_dir() / row["split_id"], ignore_errors=True)

    golden_row = db.execute("SELECT * FROM splits WHERE split_id = ?", (GOLDEN_ID,)).fetchone()
    frozen = golden_row is not None and golden_row["frozen"] == 1
    pending = db.execute("SELECT COUNT(*) FROM split_members WHERE split_id = ?", (GOLDEN_ID,)).fetchone()[0]
    result = BuildResult(
        parent, cutoff,
        SetInfo(rolling_id, len(rolling_items), rolling_words, rolling_sha),
        SetInfo(edited_id, len(edited_items), edited_words, edited_sha),
        SetInfo(train_id, len(train_ids), db.execute("SELECT n_words FROM splits WHERE split_id = ?", (train_id,)).fetchone()[0]),
        SetInfo(GOLDEN_ID, golden_row["n_clips"], golden_row["n_words"], golden_row["manifest_sha256"]) if frozen else None,
        pending, general_info, len(trainable), excluded,
    )
    if result.rolling_small:
        result.warnings.append(f"rolling_small: {rolling_words} reference words, under {ROLLING_MIN_WORDS}; the gate refuses")
    if trainable and excluded / len(trainable) > EXCLUDED_WARN:
        result.warnings.append(f"{excluded / len(trainable):.0%} of trainable clips share a cluster with a test set")
    return result


def _register_general(db: sqlite3.Connection, when: int) -> SetInfo | None:
    folder = sets_dir() / "general"
    if not (folder / "manifest.json").is_file():
        return None
    digest = general.verify(folder)
    clips = json.loads((folder / "manifest.json").read_text())["clips"]
    words = sum(len(normalized_words(c["expected"])) for c in clips)
    db.execute("DELETE FROM splits WHERE split_id = ?", (GENERAL_ID,))
    _register(db, GENERAL_ID, "general", None, None, when, folder / "manifest.json", digest, [], words)
    db.execute("UPDATE splits SET n_clips = ? WHERE split_id = ?", (len(clips), GENERAL_ID))
    return SetInfo(GENERAL_ID, len(clips), words, digest)


# --- verify --------------------------------------------------------------------------------


def verify(store: Store) -> list[str]:
    """Every violation, or an empty list. Reads only (clusters are recomputed from the clips)."""
    db = store.db
    problems: list[str] = []
    clusters = clip_clusters(db)

    def members(split_id: str) -> list[str]:
        return [r["clip_id"] for r in db.execute("SELECT clip_id FROM split_members WHERE split_id = ?", (split_id,))]

    split_rows = db.execute("SELECT * FROM splits ORDER BY split_id").fetchall()
    tests = [r for r in split_rows if r["kind"] in ("rolling", "edited", "golden")]
    trains = [r for r in split_rows if r["kind"] == "train"]
    test_clusters: dict[str, str] = {}
    for row in tests:
        for clip in members(row["split_id"]):
            test_clusters.setdefault(clusters.get(clip, clip), row["split_id"])
    for row in trains:
        for clip in members(row["split_id"]):
            hit = test_clusters.get(clusters.get(clip, clip))
            if hit:
                problems.append(f"{row['split_id']}: clip {clip} shares a split-cluster with {hit}")

    golden_frozen = [r for r in split_rows if r["kind"] == "golden" and r["frozen"]]
    if len(golden_frozen) > 1:
        problems.append("more than one golden set is frozen")
    for row in tests:
        gating = row["kind"] in ("rolling", "golden")
        for clip in members(row["split_id"]):
            c = db.execute("SELECT * FROM clips WHERE clip_id = ?", (clip,)).fetchone()
            if c is None:
                problems.append(f"{row['split_id']}: clip {clip} is gone")
                continue
            if gating and c["label"] in ("weak", "strong"):
                problems.append(f"{row['split_id']}: clip {clip}'s gating reference comes from a {c['label']} label")
            elif gating and (c["label"] != "verified" or c["review_strategy"] != "random"):
                problems.append(f"{row['split_id']}: clip {clip} is not a randomly sampled, verified clip")
            if row["kind"] == "rolling" and c["timestamp_ms"] <= row["cutoff_ms"]:
                problems.append(f"{row['split_id']}: clip {clip} predates the cut-off")
            if row["kind"] == "edited" and c["label"] != "strong":
                problems.append(f"{row['split_id']}: clip {clip} is not a strong clip")
            if row["kind"] == "golden" and not c["golden"]:
                problems.append(f"{row['split_id']}: clip {clip} is not marked golden")
        if row["manifest_path"]:
            problems += _check_set(row, db)
    return problems


def _check_set(row: sqlite3.Row, db: sqlite3.Connection) -> list[str]:
    folder = Path(row["manifest_path"]).parent
    try:
        digest = general.verify(folder)
    except general.BuildError as error:
        return [f"{row['split_id']}: {error}"]
    problems = []
    if digest != row["manifest_sha256"]:
        problems.append(f"{row['split_id']}: manifest hash {digest[:12]} differs from its pin {str(row['manifest_sha256'])[:12]}")
    if row["kind"] != "golden":
        return problems
    expected = {c["id"]: c["expected"] for c in json.loads((folder / "manifest.json").read_text())["clips"]}
    for clip, text in expected.items():
        current = db.execute("SELECT reference_text FROM clips WHERE clip_id = ?", (clip,)).fetchone()
        if current is None or current["reference_text"] != text:
            problems.append(f"{row['split_id']}: clip {clip}'s reference changed since the golden set froze")
    latest = labels.latest_reviews(db)
    for clip in expected:
        review = latest.get(clip)
        if review is None or review["action"] == "reject" or review["text"] != expected[clip]:
            problems.append(f"{row['split_id']}: clip {clip}'s latest review no longer matches the frozen reference")
    return problems
