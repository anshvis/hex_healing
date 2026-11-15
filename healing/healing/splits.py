"""`healing split build|verify`: test sets with no leakage into training.

Sets (T3.2):
- rolling: sessions after the cut-off, clips sampled at random and verified by you. Parent
  "current" has no snapshot, so its cut-off slides: the last 14 days at each build (D-40);
- golden: verified clips, frozen and pinned (the review-and-freeze flow is `healing/golden.py`);
- edited: the window's strong clips, scored only at the edited words (T3.3);
- general: T0.3's LibriSpeech and AMI manifest (not made from the dataset);
- train: not a set a build makes. `snapshot_train` is the only way training gets clips; it returns
  them and records them as a frozen `train` row, so "what a model trained on" is a row (D-41).

The window is the test zone (D-41), and `Zones` is the one leak policy: `may_test` for what a rolling
or edited set (or a golden review) may draw, `may_train` for what a run may take. A clip any recorded
snapshot holds can never be drawn into golden, the random-review frame or eligibility.

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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile

from healing import dedupe, labels, manifest
from healing.paths import sets_dir
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
TRAIN_PREFIX = "train:"


class SplitError(RuntimeError):
    pass


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


def snapshot_texts(db: sqlite3.Connection) -> dict[str, list[str]]:
    """`train:<split>:<clip>` -> the texts a recorded snapshot stored for each member when it handed it out."""
    found: dict[str, list[str]] = {}
    for row in db.execute("SELECT split_id, detail FROM splits WHERE kind = 'train' AND detail IS NOT NULL"):
        for clip, texts in json.loads(row["detail"]).get("members", {}).items():
            found[f"{TRAIN_PREFIX}{row['split_id']}:{clip}"] = list(texts)
    return found


def clip_clusters(db: sqlite3.Connection) -> dict[str, str]:
    """clip_id -> split-cluster, over the dataset's clips, the general set's references and the texts
    recorded snapshots stored.

    The general references join as `general:<id>` entries and snapshot members as `train:<split>:<clip>`
    entries, so a clip that near-duplicates what a test set holds, or what a model saw (as it read then),
    lands in that cluster; callers read clips by id and `general_clusters` reads the rest.
    """
    texts = {
        row["clip_id"]: _texts(row)
        for row in db.execute("SELECT clip_id, reference_text, raw_text FROM clips")
    }
    texts.update({key: [text] for key, text in general_references().items()})
    texts.update(snapshot_texts(db))
    return split_clusters(texts)


def general_clusters(clusters: dict[str, str]) -> set[str]:
    return {root for key, root in clusters.items() if key.startswith(GENERAL_PREFIX)}


# --- sessions and zones ------------------------------------------------------------------------


class Session(NamedTuple):
    id: str
    start: int  # ms of the session's first clip


def sessions(db: sqlite3.Connection) -> dict[str, Session]:
    """clip_id -> its session: runs of dictations with no gap over 10 minutes."""
    result: dict[str, Session] = {}
    previous, start = None, 0
    for row in db.execute("SELECT clip_id, timestamp_ms FROM clips ORDER BY timestamp_ms, clip_id"):
        if previous is None or row["timestamp_ms"] - previous > SESSION_GAP_MS:
            start = row["timestamp_ms"]
        result[row["clip_id"]] = Session(f"s{start}", start)
        previous = row["timestamp_ms"]
    return result


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


@dataclass
class Zones:
    """Who may test and who may train, loaded once per command: the one leak policy (D-41).

    The cut-off is taken at load. Everything asks the same object through named frames:

    - `rolling(clip)`: may enter a rolling or edited set (or a `--since-cutoff` review): its session
      starts after the cut-off, no golden, general or recorded-snapshot cluster holds it, and it is not
      already in another registered set;
    - `golden(clip)`: may join golden, in any age: as above without the window, and no registered set
      holds its cluster either;
    - `random_review(clip)`: may be shown to a random review: not in a golden, general or snapshot
      cluster, and not already a registered member (a reviewed edited clip would stop being strong);
    - `may_train(clip)`: a training run may take it: outside the window, in no cluster of an in-window
      clip, a registered test set or the general set.

    A recorded snapshot reserves its clusters from the texts it stored when it handed its clips out
    (`train:<split>:<clip>` entries in the clustering, like `general:<id>`), so a clip whose text has
    changed since cannot hide what the model saw. `why_not_rolling`, `why_not_edited` and
    `why_not_golden` add each set's own conditions and return the reason (None: it belongs). `build`
    selects with them and `verify` re-asks them of the registered members, so there is one rule.
    """

    clusters: dict[str, str]  # clip id, `general:<id>` and `train:<split>:<clip>` -> split-cluster root
    sessions: dict[str, Session]
    cutoff_ms: int
    owners: dict[str, dict[str, str]]  # cluster -> {split_id: kind} for golden, general, rolling, edited (every parent)
    members: dict[str, dict[str, str]]  # clip id -> {split_id: kind} registered in rolling, edited, golden
    trained: set[str]  # clusters of the stored texts (and the current clips) of every recorded snapshot
    window: set[str]  # clusters of every clip in a session inside the window

    @classmethod
    def load(
        cls, db: sqlite3.Connection, cutoff_ms: int | None = None, now: int | None = None,
        ignore: Iterable[str] = (),
    ) -> Zones:
        """`ignore` names registered sets about to be replaced (a build's old rolling and edited sets
        for its parent), which must not block their own successors. Every other parent's sets count:
        a clip in any registered test set is out of every training run."""
        clusters, sess = clip_clusters(db), sessions(db)
        cutoff = cutoff_for(cutoff_ms, now)
        skipped = set(ignore)
        owners: dict[str, dict[str, str]] = {}
        members: dict[str, dict[str, str]] = {}
        trained = {root for key, root in clusters.items() if key.startswith(TRAIN_PREFIX)}
        for row in db.execute(
            "SELECT s.split_id, s.kind, m.clip_id FROM splits s JOIN split_members m ON m.split_id = s.split_id"
            " WHERE s.kind IN ('rolling', 'edited', 'golden', 'train')"
        ):
            root = clusters.get(row["clip_id"], row["clip_id"])
            if row["kind"] == "train":
                trained.add(root)
            elif row["split_id"] not in skipped:
                owners.setdefault(root, {})[row["split_id"]] = row["kind"]
                members.setdefault(row["clip_id"], {})[row["split_id"]] = row["kind"]
        for root in general_clusters(clusters):
            owners.setdefault(root, {})[general.GENERAL_ID] = "general"
        window = {clusters[clip] for clip, session in sess.items() if session.start > cutoff}
        return cls(clusters, sess, cutoff, owners, members, trained, window)

    def in_window(self, clip: str, cutoff_ms: int | None = None) -> bool:
        return self.sessions[clip].start > (self.cutoff_ms if cutoff_ms is None else cutoff_ms)

    def _others(self, clip: str, own: str | None) -> dict[str, str]:
        """{split_id: kind} of the registered sets whose clusters the clip shares, apart from `own`."""
        return {s: k for s, k in self.owners.get(self.clusters[clip], {}).items() if s != own}

    def _reason(self, clip: str, *, own: str | None, windowed: bool, cutoff_ms: int | None,
                trained: bool, registered: bool, members: bool) -> str | None:
        """Why `clip` may not be drawn into a frame, or None. `own` is the registered set the clip is
        already judged as a member of (verify), whose own rows are not held against it."""
        if windowed and not self.in_window(clip, cutoff_ms):
            return "predates the cut-off"
        if trained and self.clusters[clip] in self.trained:
            return "shares a split-cluster with a recorded train snapshot"
        others = self._others(clip, own)
        for split_id, kind in others.items():
            if kind in ("golden", "general"):
                return f"shares a split-cluster with {'the golden set' if kind == 'golden' else split_id}"
        if registered and others:
            return f"shares a split-cluster with {next(iter(others))}"
        if members:
            held = {s: k for s, k in self.members.get(clip, {}).items() if s != own}
            if held:
                return f"is already in {next(iter(held))}"
        return None

    def rolling(self, clip: str, *, own: str | None = None, cutoff_ms: int | None = None, trained: bool = True) -> bool:
        return self.rolling_reason(clip, own=own, cutoff_ms=cutoff_ms, trained=trained) is None

    def rolling_reason(self, clip: str, *, own: str | None = None, cutoff_ms: int | None = None, trained: bool = True) -> str | None:
        return self._reason(clip, own=own, windowed=True, cutoff_ms=cutoff_ms, trained=trained, registered=False, members=True)

    def golden(self, clip: str) -> bool:
        return self.golden_reason(clip) is None

    def golden_reason(self, clip: str, *, own: str | None = None, trained: bool = True) -> str | None:
        return self._reason(clip, own=own, windowed=False, cutoff_ms=None, trained=trained, registered=True, members=False)

    def random_review(self, clip: str) -> bool:
        return self._reason(clip, own=None, windowed=False, cutoff_ms=None, trained=True, registered=False, members=True) is None

    def may_train(self, clip: str) -> bool:
        # `window` holds the cluster of every in-window clip, so it also keeps those clips themselves out.
        root = self.clusters[clip]
        return root not in self.window and root not in self.owners

    # The sets' own conditions on top of the frames. `fields` is a clips row, or (for an edited set,
    # judged by what it pinned) a sidecar row: it needs `label`, plus the keys each check reads.

    def why_not_rolling(self, clip: str, fields, *, own: str | None = None, cutoff_ms: int | None = None, trained: bool = True) -> str | None:
        reason = _gating_reason(fields)
        return reason or self.rolling_reason(clip, own=own, cutoff_ms=cutoff_ms, trained=trained)

    def why_not_edited(self, clip: str, fields, *, own: str | None = None, cutoff_ms: int | None = None, trained: bool = True) -> str | None:
        if fields["label"] != "strong":
            return "is not a strong clip"
        if not fields["edited_positions"]:
            return "has no edited positions"
        return self.rolling_reason(clip, own=own, cutoff_ms=cutoff_ms, trained=trained)

    def why_not_golden(self, clip: str, fields, *, own: str | None = None, trained: bool = True) -> str | None:
        reason = _gating_reason(fields)
        if reason is None and not fields["golden"]:
            reason = "is not marked golden"
        return reason or self.golden_reason(clip, own=own, trained=trained)


def _gating_reason(fields) -> str | None:
    """A gating set takes verified, randomly sampled references only (AC-7)."""
    if fields["label"] in ("weak", "strong"):
        return f"gating reference comes from a {fields['label']} label"
    if fields["label"] != "verified" or fields["review_strategy"] != "random":
        return "is not a randomly sampled, verified clip"
    return None


def _trainable(store: Store, zones: Zones, model: str | None, include_weak: bool) -> list[sqlite3.Row]:
    return [r for r in labels.trainable(store, model, include_weak) if zones.may_train(r["clip_id"])]


def preview_train(
    store: Store, *, include_weak: bool = False, model: str | None = None,
    cutoff_ms: int | None = None, now: int | None = None,
) -> list[str]:
    """The clips `snapshot_train` would hand out now. Read-only: it records nothing, so what it
    returns is not what any model trained on."""
    zones = Zones.load(store.db, cutoff_ms, now)
    return [r["clip_id"] for r in _trainable(store, zones, model, include_weak)]


def snapshot_train(
    store: Store, *, parent: str = "current", include_weak: bool = False, model: str | None = None,
    note: str | None = None, cutoff_ms: int | None = None, now: int | None = None,
) -> list[str]:
    """The one way training gets clips (D-41): returns the ids and records them.

    Labels are refreshed first, then the clips are `labels.trainable` outside the test zone
    (`Zones.may_train`), read from live data inside one write transaction, so it is right after a
    review no build has seen and a concurrent golden review or snapshot cannot decide on stale facts.
    A frozen `train` row is written for `parent` whose members are exactly the returned clips (weak
    ones too when asked), with `model`, `note` and each member's texts in its detail: `Zones` and
    `verify` read those texts, not the clips' current ones. Every cluster in a recorded snapshot is then
    closed to golden, random review and test eligibility. Nothing is recorded when there is nothing
    to hand out.
    """
    db = store.db
    labels.refresh(store)
    when = now_ms() if now is None else now
    with store.immediate():
        zones = Zones.load(db, cutoff_ms, now)
        rows = _trainable(store, zones, model, include_weak)
        if not rows:
            return []
        stamp = datetime.fromtimestamp(when / 1000, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        split_id = _free_id(db, f"train-{parent}-{stamp}")
        detail = {
            "model": model, "note": note, "include_weak": include_weak,
            "members": {r["clip_id"]: _texts(r) for r in rows},
        }
        words = sum(len(normalized_words(r["reference_text"] or r["raw_text"])) for r in rows)
        db.execute(
            "INSERT INTO splits (split_id, kind, parent_version, cutoff_ms, created_at_ms, frozen, n_clips, n_words, detail)"
            " VALUES (?, 'train', ?, ?, ?, 0, ?, ?, ?)",
            (split_id, parent, zones.cutoff_ms, when, len(rows), words, json.dumps(detail, sort_keys=True)),
        )
        db.executemany("INSERT INTO split_members (split_id, clip_id) VALUES (?, ?)", [(split_id, r["clip_id"]) for r in rows])
        db.execute("UPDATE splits SET frozen = 1 WHERE split_id = ?", (split_id,))  # members first: a frozen row takes none
    return [r["clip_id"] for r in rows]


def _texts(row) -> list[str]:
    """The texts a clip clusters by: its reference and what Hex heard."""
    return [t for t in (row["reference_text"] or row["raw_text"], row["raw_text"]) if t]


# --- build ---------------------------------------------------------------------------------


@dataclass
class SetInfo:
    split_id: str
    n_clips: int
    n_words: int
    manifest_sha256: str | None = None
    sidecar_sha256: str | None = None
    path: Path | None = None  # manifest.json; None for a set with no folder (an empty one)


@dataclass
class BuildResult:
    parent: str
    cutoff_ms: int
    rolling: SetInfo
    edited: SetInfo
    golden: SetInfo | None
    golden_pending: int
    general: SetInfo | None
    trainable: int
    train_ready: int
    warnings: list[str] = field(default_factory=list)

    @property
    def rolling_small(self) -> bool:
        return self.rolling.n_words < ROLLING_MIN_WORDS

    @property
    def excluded_train(self) -> int:
        return self.trainable - self.train_ready

    def text(self) -> str:
        lines = [f"parent {self.parent}, cut-off {self.cutoff_ms}"]
        for label, info in (("rolling", self.rolling), ("edited-span", self.edited)):
            lines.append(f"{label}: {info.n_clips} clips, {info.n_words} words ({info.split_id})")
        if self.golden:
            lines.append(f"golden: {self.golden.n_clips} clips, {self.golden.n_words} words, frozen")
        else:
            lines.append(f"golden: n/a, {self.golden_pending} of {GOLDEN_SIZE} confirmed (pending)")
        lines.append(
            f"general: {self.general.n_clips} clips ({self.general.split_id})" if self.general
            else "general: n/a (run `healing sets build-general`)"
        )
        lines.append(
            f"train: {self.train_ready} of {self.trainable} trainable clips are outside the test zone "
            "(a preview: `snapshot_train` records what a run takes)"
        )
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


def entries(store: Store, rows: list[sqlite3.Row], clusters: dict[str, str], sess: dict[str, Session], *, edited: bool = False) -> list[dict]:
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
            "session": sess[row["clip_id"]].id,
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
    ids = [i["id"] for i in items]
    audio = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        audio.update({
            r["clip_id"]: r["audio_sha256"] for r in store.db.execute(
                f"SELECT clip_id, audio_sha256 FROM clips WHERE clip_id IN ({','.join('?' * len(chunk))})", chunk)
        })
    digest, words = manifest.write_set(out, items, lambda clip: _pcm(store, audio[clip]), sets_dir())
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
              created: int, members: list[str]) -> None:
    db.execute(
        "INSERT INTO splits (split_id, kind, parent_version, cutoff_ms, created_at_ms, manifest_path,"
        " manifest_sha256, sidecar_sha256, n_clips, n_words) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (info.split_id, kind, parent, cutoff, created, str(info.path) if info.path else None,
         info.manifest_sha256, info.sidecar_sha256, info.n_clips, info.n_words),
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
    """Rebuild the parent's rolling and edited sets and register golden and general.

    A build is not training: it creates no train rows (`snapshot_train` does).
    """
    db = store.db
    labels.label_all(store)  # a fresh collect or migration has no labels yet: sets would come out empty
    when = now_ms() if now is None else now
    stamp = datetime.fromtimestamp(when / 1000, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with store.immediate():  # decide and register on one set of facts: a snapshot or golden add waits
        old = db.execute(
            "SELECT split_id FROM splits WHERE frozen = 0 AND parent_version = ? AND kind IN ('rolling', 'edited')",
            (parent,),
        ).fetchall()
        zones = Zones.load(db, cutoff_ms, when, ignore=[r["split_id"] for r in old])
        rolling_rows, edited_rows = [], []
        for row in db.execute(f"SELECT c.* FROM clips c WHERE {labels.USABLE} ORDER BY c.timestamp_ms, c.clip_id"):
            if zones.why_not_rolling(row["clip_id"], row) is None:
                rolling_rows.append(row)
            elif zones.why_not_edited(row["clip_id"], row) is None:
                edited_rows.append(row)
        rolling_items = entries(store, rolling_rows, zones.clusters, zones.sessions)
        edited_items = entries(store, edited_rows, zones.clusters, zones.sessions, edited=True)
        # Write the folders first: a failure leaves the registry as it was.
        rolling = _written(store, f"rolling-{parent}-{stamp}", rolling_items)
        edited = _written(store, f"edited-{parent}-{stamp}", edited_items)
        for row in old:
            db.execute("DELETE FROM split_members WHERE split_id = ?", (row["split_id"],))
            db.execute("DELETE FROM splits WHERE split_id = ?", (row["split_id"],))
        _register(db, rolling, "rolling", parent, zones.cutoff_ms, when, [i["id"] for i in rolling_items])
        _register(db, edited, "edited", parent, zones.cutoff_ms, when, [i["id"] for i in edited_items])
        general_info = _register_general(db, when)
    for row in old:
        if row["split_id"] not in (rolling.split_id, edited.split_id):
            shutil.rmtree(sets_dir() / row["split_id"], ignore_errors=True)

    golden = golden_row(db)
    frozen = golden is not None and golden["frozen"] == 1
    trainable = labels.trainable(store)
    result = BuildResult(
        parent, zones.cutoff_ms, rolling, edited,
        SetInfo(GOLDEN_ID, golden["n_clips"], golden["n_words"], golden["manifest_sha256"]) if frozen else None,
        len(golden_members(db)), general_info, len(trainable),
        len(preview_train(store, cutoff_ms=zones.cutoff_ms)),  # after the sets are registered
    )
    if result.rolling_small:
        result.warnings.append(f"rolling_small: {rolling.n_words} reference words, under {ROLLING_MIN_WORDS}; the gate refuses")
    if trainable and result.excluded_train / len(trainable) > EXCLUDED_WARN:
        result.warnings.append(
            f"{result.excluded_train / len(trainable):.0%} of trainable clips share a cluster with a test set or sit in the window"
        )
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


@dataclass
class Report:
    leaks: list[str] = field(default_factory=list)  # the current sets break a rule
    snapshot_overlaps: list[str] = field(default_factory=list)  # a model trained on a clip a test set now holds

    @property
    def problems(self) -> list[str]:
        return self.leaks + self.snapshot_overlaps


def verify(store: Store) -> list[str]:
    """Every violation, or an empty list. Reads only."""
    return verify_report(store).problems


def _pinned(row: sqlite3.Row) -> dict[str, dict]:
    """An edited set's sidecar rows (clip id -> row): what the set pinned, or {} when unreadable (the
    pin check reports that)."""
    if not row["manifest_path"]:
        return {}
    try:
        path = Path(row["manifest_path"]).parent / "clips.jsonl"
        return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    except (OSError, ValueError, KeyError):
        return {}


def verify_report(store: Store) -> Report:
    """Current-set leaks and snapshot overlaps, kept apart.

    Leaks re-ask `Zones` of every registered member, the same predicates `build` selected with
    (clusters are recomputed from the clips). A rolling or golden member is judged by its live label,
    so a review that rejects it shows at once; an edited member is judged by the label and positions
    its set pinned (its sidecar, covered by the pin), so a clip whose correction changes after the
    build does not flag a set that is still exactly what was built. A recorded train snapshot is
    checked against the test clips' texts directly, from the texts the snapshot stored, with the
    near-duplicate test and no clustering: a clip trained on that a test set now holds is a true
    contamination report, and a union-find that links non-transitively could never make one fail
    permanently (so a member's live cluster is not held against a snapshot here).
    """
    db = store.db
    report = Report()
    zones = Zones.load(db)

    def members(split_id: str) -> list[str]:
        return [r["clip_id"] for r in db.execute("SELECT clip_id FROM split_members WHERE split_id = ?", (split_id,))]

    split_rows = db.execute("SELECT * FROM splits ORDER BY split_id").fetchall()
    tests = [r for r in split_rows if r["kind"] in ("rolling", "edited", "golden")]
    if len([r for r in split_rows if r["kind"] == "golden" and r["frozen"]]) > 1:
        report.leaks.append("more than one golden set is frozen")
    test_texts: dict[str, list[str]] = {}
    for row in tests:
        pinned = _pinned(row) if row["kind"] == "edited" else {}
        for clip in members(row["split_id"]):
            c = db.execute("SELECT * FROM clips WHERE clip_id = ?", (clip,)).fetchone()
            if c is None:
                report.leaks.append(f"{row['split_id']}: clip {clip} is gone")
                continue
            test_texts[f"{row['split_id']}:{clip}"] = _texts(c)
            own = row["split_id"]
            if row["kind"] == "rolling":
                reason = zones.why_not_rolling(clip, c, own=own, cutoff_ms=row["cutoff_ms"], trained=False)
            elif row["kind"] == "edited":
                fields = pinned.get(clip)
                reason = ("is not in its pinned sidecar" if fields is None else
                          zones.why_not_edited(clip, fields, own=own, cutoff_ms=row["cutoff_ms"], trained=False))
            else:
                reason = zones.why_not_golden(clip, c, own=own, trained=False)
            if reason:
                report.leaks.append(f"{row['split_id']}: clip {clip} {reason}")
    for row in split_rows:
        if row["manifest_path"]:
            report.leaks += _check_set(row, db, members(row["split_id"]))
    for key, text in general_references().items():
        test_texts[f"{general.GENERAL_ID}:{key[len(GENERAL_PREFIX):]}"] = [text]
    report.snapshot_overlaps = _snapshot_overlaps(split_rows, test_texts)
    return report


def _snapshot_overlaps(split_rows: list, test_texts: dict[str, list[str]]) -> list[str]:
    """A snapshot's stored member texts against every test text, through one index built once."""
    by_key: dict[str, list[str]] = {}
    for owner, texts in test_texts.items():
        for text in texts:
            if key := dedupe.normalize(text):
                by_key.setdefault(key, []).append(owner)
    index = dedupe.NearIndex(by_key)
    found = []
    for snapshot in (r for r in split_rows if r["kind"] == "train"):
        members = json.loads(snapshot["detail"] or "{}").get("members", {})
        seen: set[tuple[str, str]] = set()
        for clip, texts in sorted(members.items()):
            for key in filter(None, map(dedupe.normalize, texts)):
                for other in index.near(key):
                    for owner in by_key[other]:
                        if (clip, owner) not in seen:
                            seen.add((clip, owner))
                            found.append(
                                f"{snapshot['split_id']}: a model trained on clip {clip}, which near-duplicates "
                                f"{owner.split(':')[1]} in {owner.split(':')[0]}"
                            )
    return found


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
    latest = labels.latest_reviews(db)
    for clip, text in expected.items():
        current = db.execute("SELECT reference_text FROM clips WHERE clip_id = ?", (clip,)).fetchone()
        review = latest.get(clip)
        if current is None or current["reference_text"] != text:
            problems.append(f"{row['split_id']}: clip {clip}'s reference changed since the golden set froze")
        elif review is None or review["action"] == "reject" or review["text"] != text:
            problems.append(f"{row['split_id']}: clip {clip}'s latest review no longer matches the frozen reference")
    return problems
