"""`healing split build|verify`: test sets with no leakage into training.

Sets (T3.2):
- rolling: sessions after the cut-off, clips sampled at random and verified by you. Parent
  "current" has no snapshot, so its cut-off slides: the last 14 days at each build (D-40);
- golden: verified clips, frozen and pinned (the review-and-freeze flow is `healing/golden.py`);
- edited: the window's strong clips, scored only at the edited words (T3.3);
- general: T0.3's LibriSpeech and AMI manifest (not made from the dataset);
- train: what a training run may use, kept as history (D-41): every build adds a train split and
  freezes the one before it, so "what a model trained on" is a row.

The window is the test zone (D-41). `train_ids`, the leak-free entry point, takes no clip from a
session inside it, none that shares a split-cluster with an in-window clip, a registered test set
or the general set, and reads live data, so it is right between builds. `eligible_ids` is the one
rule for what may enter rolling or edited (and the random review's `--since-cutoff` frame): in the
window, and in no cluster of golden, the general set or any retained train split, so a clip a model
trained on can never become a test clip.

A split-cluster is a connected component of near-duplicate `reference_text` (union-find, token
3-gram Jaccard >= 0.8, no duration condition). It is deliberately not T1.3's storage cluster:
those assign each clip to one representative and gate on duration, so "send it to kip" said slowly
and "send it to kit" said fast would land apart and straddle train and test. A clip also joins
through its `raw_text`, so a clip whose correction is "kit" meets another clip that was merely
heard as "kit".
"""

from __future__ import annotations

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
WINDOW_MS = 14 * 24 * 3600 * 1000  # D-40: parent "current" (v0) has no snapshot, so a sliding 14-day window
ROLLING_MIN_WORDS = 1500  # under this the rolling set is too small to gate (D-12)
EXCLUDED_WARN = 0.30
GOLDEN_ID = "golden-v1"
GOLDEN_SIZE = 200  # the golden set freezes once it holds this many confirmed clips
GENERAL_PREFIX = "general:"


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
    for left, right in dedupe.near_duplicate_pairs(owners):  # texts under 3 tokens match by equality only
        union(owners[left][0], owners[right][0])
    return {clip: find(clip) for clip in clip_texts}


def general_references() -> dict[str, str]:
    """`general:<id>` -> reference, for the general set when it is built (it is not made from the dataset)."""
    path = general.folder() / "manifest.json"
    if not path.is_file():
        return {}
    return {GENERAL_PREFIX + c["id"]: c["expected"] for c in json.loads(path.read_text())["clips"]}


def clip_clusters(db: sqlite3.Connection) -> dict[str, str]:
    """clip_id -> split-cluster, over the dataset's clips and the general set's references.

    The general references join as `general:<id>` entries so a training clip that near-duplicates
    one lands in a test cluster; callers read clips by id and `general_clusters` reads the rest.
    """
    texts = {
        row["clip_id"]: [t for t in (row["reference_text"] or row["raw_text"], row["raw_text"]) if t]
        for row in db.execute("SELECT clip_id, reference_text, raw_text FROM clips")
    }
    texts.update({key: [text] for key, text in general_references().items()})
    return split_clusters(texts)


def general_clusters(clusters: dict[str, str]) -> set[str]:
    return {root for key, root in clusters.items() if key.startswith(GENERAL_PREFIX)}


# --- sessions -------------------------------------------------------------------------------


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


# --- sessions, the window and what is reserved ---------------------------------------------------


def cutoff_for(override_ms: int | None = None, now: int | None = None) -> int:
    """The window's start, in UTC ms: `override_ms`, else the last 14 days (D-40).

    It slides: nothing stores it, so a strong clip that ages past the window leaves the edited
    diagnostic and becomes trainable. M5's registry replaces it, for a parent with a snapshot, with
    that snapshot's time.
    """
    when = now_ms() if now is None else now
    return override_ms if override_ms is not None else when - WINDOW_MS


def golden_row(db: sqlite3.Connection) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM splits WHERE split_id = ?", (GOLDEN_ID,)).fetchone()


def golden_members(db: sqlite3.Connection) -> list[str]:
    return [r["clip_id"] for r in db.execute(
        "SELECT clip_id FROM split_members WHERE split_id = ? ORDER BY clip_id", (GOLDEN_ID,))]


def _member_clusters(db: sqlite3.Connection, clusters: dict[str, str], where: str, parameters: tuple = ()) -> dict[str, str]:
    """split-cluster -> the split holding it, for the members of splits matching `where`."""
    found: dict[str, str] = {}
    for row in db.execute(
        f"SELECT s.split_id, m.clip_id FROM splits s JOIN split_members m ON m.split_id = s.split_id WHERE {where}",
        parameters,
    ):
        found.setdefault(clusters.get(row["clip_id"], row["clip_id"]), row["split_id"])
    return found


def trained_clusters(db: sqlite3.Connection, clusters: dict[str, str]) -> set[str]:
    """Clusters of every clip in any retained train split: what a model has been trained on."""
    return set(_member_clusters(db, clusters, "s.kind = 'train'"))


def registered_test_clusters(db: sqlite3.Connection, clusters: dict[str, str]) -> set[str]:
    """Clusters of every registered rolling, edited and golden member, whatever the parent."""
    return set(_member_clusters(db, clusters, "s.kind IN ('rolling', 'edited', 'golden')"))


def reserved_clusters(db: sqlite3.Connection, clusters: dict[str, str]) -> set[str]:
    """Clusters no clip may be newly sampled for a test set from: golden, the general set's
    references, and everything a retained train split holds."""
    golden = {clusters[c] for c in golden_members(db) if c in clusters}
    return golden | general_clusters(clusters) | trained_clusters(db, clusters)


def eligible_ids(db: sqlite3.Connection, clusters: dict[str, str], sess: dict, cutoff_ms: int) -> set[str]:
    """The one rule for what may enter rolling or edited: the clip's session starts after the
    cut-off, and its cluster is reserved by no golden, general or trained clip."""
    reserved = reserved_clusters(db, clusters)
    return {clip for clip, (_, start) in sess.items() if start > cutoff_ms and clusters.get(clip) not in reserved}


def test_clusters_of(db: sqlite3.Connection, clusters: dict[str, str], parent: str) -> dict[str, str]:
    """split-cluster -> the test split holding it, for what `parent`'s train sets must not touch:
    the parent's rolling and edited sets, the golden set and the general set's references."""
    found = _member_clusters(
        db, clusters,
        "s.kind IN ('rolling', 'edited', 'golden') AND (s.parent_version IS NULL OR s.parent_version = ?)",
        (parent,),
    )
    for root in general_clusters(clusters):
        found.setdefault(root, general.GENERAL_ID)
    return found


def _train_rows(store: Store, parent: str, cutoff_ms: int, model_id: str | None, include_weak: bool) -> list[sqlite3.Row]:
    db = store.db
    clusters, sess = clip_clusters(db), sessions(db)
    held = set(test_clusters_of(db, clusters, parent))
    held |= {clusters[clip] for clip, (_, start) in sess.items() if start > cutoff_ms}  # the window is a test zone
    return [
        r for r in labels.trainable(store, model_id, include_weak)
        if clusters[r["clip_id"]] not in held and sess[r["clip_id"]][1] <= cutoff_ms
    ]


def train_ids(
    store: Store, parent: str = "current", cutoff_ms: int | None = None, now: int | None = None,
    model_id: str | None = None, include_weak: bool = False,
) -> list[str]:
    """The clips a training run may use: `labels.trainable` outside the test zone (D-41).

    Taken from live data, so it is right after a review that no build has seen yet: no clip of a
    session inside the window, none sharing a split-cluster with an in-window clip, a registered
    test set of `parent`, or the general set. Raw `labels.trainable` is not leak-free.
    """
    return [r["clip_id"] for r in _train_rows(store, parent, cutoff_for(cutoff_ms, now), model_id, include_weak)]


# --- build ---------------------------------------------------------------------------------


@dataclass
class SetInfo:
    split_id: str
    n_clips: int
    n_words: int
    manifest_sha256: str | None = None
    sidecar_sha256: str | None = None
    path: Path | None = None  # manifest.json; None for a set with no folder (train, or an empty one)


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


def write_and_pin(store: Store, split_id: str, items: list[dict]) -> SetInfo:
    """Write a set folder and return what it pinned."""
    out = sets_dir() / split_id
    if out.exists():
        shutil.rmtree(out)
    rows = {i["id"]: store.db.execute("SELECT audio_sha256 FROM clips WHERE clip_id = ?", (i["id"],)).fetchone() for i in items}
    digest, words = manifest.write_set(out, items, lambda clip: _pcm(store, rows[clip]["audio_sha256"]))
    return SetInfo(split_id, len(items), words, digest, manifest.sidecar_digest(out), out / "manifest.json")


def verified_set(row: sqlite3.Row) -> tuple[Path, str]:
    """(folder, manifest sha256) of a registered set, or `manifest.ManifestError` naming what changed.

    The one pin check, shared by `split verify` and `eval`: the folder is whole, its manifest and
    sidecar still match their pin files and each WAV its recorded hash, and both hashes equal the
    registry's (a folder rewritten consistently, pin files included, still fails here).
    """
    folder = Path(row["manifest_path"]).parent
    digest = manifest.verify(folder)
    if digest != row["manifest_sha256"]:
        raise manifest.ManifestError(
            f"{row['split_id']}: manifest hash {digest[:12]} differs from its pin {str(row['manifest_sha256'])[:12]}"
        )
    sidecar = manifest.sidecar_digest(folder)
    if row["sidecar_sha256"] and sidecar != row["sidecar_sha256"]:
        raise manifest.ManifestError(
            f"{row['split_id']}: clips.jsonl hash {sidecar[:12]} differs from its pin {row['sidecar_sha256'][:12]}"
        )
    return folder, digest


def _register(db: sqlite3.Connection, info: SetInfo, kind: str, parent: str | None, cutoff: int | None,
              created: int, members: list[str], frozen: int = 0) -> None:
    db.execute(
        "INSERT INTO splits (split_id, kind, parent_version, cutoff_ms, created_at_ms, manifest_path,"
        " manifest_sha256, sidecar_sha256, frozen, n_clips, n_words) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (info.split_id, kind, parent, cutoff, created, str(info.path) if info.path else None,
         info.manifest_sha256, info.sidecar_sha256, frozen, info.n_clips, info.n_words),
    )
    db.executemany("INSERT INTO split_members (split_id, clip_id) VALUES (?, ?)", [(info.split_id, m) for m in members])


def _written(store: Store, split_id: str, items: list[dict]) -> SetInfo:
    return write_and_pin(store, split_id, items) if items else SetInfo(split_id, 0, 0)


def _free_id(db: sqlite3.Connection, split_id: str) -> str:
    candidate, n = split_id, 1
    while db.execute("SELECT 1 FROM splits WHERE split_id = ?", (candidate,)).fetchone():
        n += 1
        candidate = f"{split_id}-{n}"
    return candidate


def build(store: Store, parent: str = "current", cutoff_ms: int | None = None, now: int | None = None) -> BuildResult:
    db = store.db
    labels.label_all(store)  # a fresh collect or migration has no labels yet: sets would come out empty
    when = now_ms() if now is None else now
    cutoff = cutoff_for(cutoff_ms, when)
    clusters, sess = clip_clusters(db), sessions(db)
    eligible = eligible_ids(db, clusters, sess, cutoff)
    stamp = datetime.fromtimestamp(when / 1000, timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rolling_rows, edited_rows = [], []
    for row in db.execute(f"SELECT c.* FROM clips c WHERE {labels.USABLE} ORDER BY c.timestamp_ms, c.clip_id"):
        if row["clip_id"] not in eligible:
            continue
        if row["label"] == "verified" and row["review_strategy"] == "random":
            rolling_rows.append(row)
        elif row["label"] == "strong" and row["edited_positions"]:
            edited_rows.append(row)
    rolling_items = entries(store, rolling_rows, clusters, sess)
    edited_items = entries(store, edited_rows, clusters, sess, edited=True)

    rolling_id, edited_id = f"rolling-{parent}-{stamp}", f"edited-{parent}-{stamp}"
    old = db.execute(
        "SELECT split_id FROM splits WHERE frozen = 0 AND parent_version = ? AND kind IN ('rolling', 'edited')",
        (parent,),
    ).fetchall()
    # Write the folders first: a failure leaves the registry as it was.
    rolling, edited = _written(store, rolling_id, rolling_items), _written(store, edited_id, edited_items)
    for row in old:
        db.execute("DELETE FROM split_members WHERE split_id = ?", (row["split_id"],))
        db.execute("DELETE FROM splits WHERE split_id = ?", (row["split_id"],))
    _register(db, rolling, "rolling", parent, cutoff, when, [i["id"] for i in rolling_items])
    _register(db, edited, "edited", parent, cutoff, when, [i["id"] for i in edited_items])

    # Train is history (D-41): the new split is computed after the test sets are registered, by the
    # rule `train_ids` and `verify` use, and the one before it freezes rather than goes.
    trainable = labels.trainable(store)
    train_rows = _train_rows(store, parent, cutoff, None, False)
    words = sum(len(normalized_words(r["reference_text"] or r["raw_text"])) for r in train_rows)
    train = SetInfo(_free_id(db, f"train-{parent}-{stamp}"), len(train_rows), words)
    earlier = [r["split_id"] for r in db.execute(
        "SELECT split_id FROM splits WHERE kind = 'train' AND parent_version = ? AND frozen = 0", (parent,))]
    _register(db, train, "train", parent, cutoff, when, [r["clip_id"] for r in train_rows])
    db.executemany("UPDATE splits SET frozen = 1 WHERE split_id = ?", [(i,) for i in earlier])
    general_info = _register_general(db, when)
    db.commit()
    for row in old:
        if row["split_id"] not in (rolling_id, edited_id):
            shutil.rmtree(sets_dir() / row["split_id"], ignore_errors=True)

    golden = golden_row(db)
    frozen = golden is not None and golden["frozen"] == 1
    result = BuildResult(
        parent, cutoff, rolling, edited, train,
        SetInfo(GOLDEN_ID, golden["n_clips"], golden["n_words"], golden["manifest_sha256"]) if frozen else None,
        len(golden_members(db)), general_info, len(trainable), len(trainable) - len(train_rows),
    )
    if result.rolling_small:
        result.warnings.append(f"rolling_small: {rolling.n_words} reference words, under {ROLLING_MIN_WORDS}; the gate refuses")
    excluded = result.excluded_train
    if trainable and excluded / len(trainable) > EXCLUDED_WARN:
        result.warnings.append(f"{excluded / len(trainable):.0%} of trainable clips share a cluster with a test set or sit in the window")
    return result


def _register_general(db: sqlite3.Connection, when: int) -> SetInfo | None:
    """Register the general set once. A different digest later is refused: it never changes."""
    folder = general.folder()
    if not (folder / "manifest.json").is_file():
        return None
    digest = manifest.verify(folder)
    clips = json.loads((folder / "manifest.json").read_text())["clips"]
    words = sum(len(normalized_words(c["expected"])) for c in clips)
    info = SetInfo(general.GENERAL_ID, len(clips), words, digest, manifest.sidecar_digest(folder), folder / "manifest.json")
    existing = db.execute("SELECT manifest_sha256 FROM splits WHERE split_id = ?", (general.GENERAL_ID,)).fetchone()
    if existing is None:
        _register(db, info, "general", None, None, when, [])
    elif existing["manifest_sha256"] != digest:
        raise SplitError(
            f"the general set is pinned at {existing['manifest_sha256'][:12]} and the folder now holds {digest[:12]}; it never changes"
        )
    return info


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
    for train in (r for r in split_rows if r["kind"] == "train"):  # every retained one, not just the newest
        held = test_clusters_of(db, clusters, train["parent_version"])
        for clip in members(train["split_id"]):
            hit = held.get(clusters.get(clip, clip))
            if hit:
                problems.append(f"{train['split_id']}: clip {clip} shares a split-cluster with {hit}")

    golden_frozen = [r for r in split_rows if r["kind"] == "golden" and r["frozen"]]
    if len(golden_frozen) > 1:
        problems.append("more than one golden set is frozen")
    golden_roots = {clusters.get(c, c) for c in golden_members(db)}
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
            if row["kind"] in ("rolling", "edited") and clusters.get(clip, clip) in golden_roots:
                problems.append(f"{row['split_id']}: clip {clip} shares a split-cluster with the golden set")
    for row in split_rows:
        if row["manifest_path"]:
            problems += _check_set(row, db, members(row["split_id"]))
    return problems


def _check_set(row: sqlite3.Row, db: sqlite3.Connection, member_ids: list[str]) -> list[str]:
    try:
        folder, _ = verified_set(row)
    except manifest.ManifestError as error:
        return [str(error)]
    if row["kind"] != "golden":
        return []
    problems = []
    expected = {c["id"]: c["expected"] for c in json.loads((folder / "manifest.json").read_text())["clips"]}
    if set(expected) != set(member_ids):
        problems.append(f"{row['split_id']}: the manifest's clips differ from the registry's members")
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
