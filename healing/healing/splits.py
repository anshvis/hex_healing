"""`healing split build|verify`: test sets with no leakage into training.

Sets (T3.2):
- rolling: sessions after the cut-off, clips sampled at random and verified by you. Parent
  "current" has no snapshot, so its cut-off slides: the last 14 days at each build (D-40);
- golden: verified clips, frozen and pinned (the review-and-freeze flow is `healing/golden.py`);
- edited: the window's strong clips, scored only at the edited words (T3.3);
- general: T0.3's LibriSpeech and AMI manifest (not made from the dataset);
- train: not a set a build makes. `snapshot_train` is the only way training gets clips; it returns
  them and records them as a frozen `train` row, so "what a model trained on" is a row (D-41).

The window is the test zone (D-41), and `Zones` is the one leak policy: the frames `rolling`, `golden`
and `random_review` for what a set (or a review) may draw, `may_train` for what a run may take. A clip
any recorded snapshot holds can never be drawn into golden, the random-review frame or eligibility.

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


def general_references(db: sqlite3.Connection, *, strict: bool = True) -> dict[str, str]:
    """`general:<id>` -> reference, for the general set (it is not made from the dataset).

    Once the set is registered its references come from the folder only while the folder's manifest
    still matches the registered pin: a missing or changed folder raises `SplitError` (`strict`),
    because silently reading nothing would stop reserving the set's clusters. `verify` passes
    `strict=False` and reports the broken pin itself.
    """
    path = general.folder() / "manifest.json"
    row = db.execute("SELECT manifest_sha256 FROM splits WHERE split_id = ?", (general.GENERAL_ID,)).fetchone()
    if row is not None:
        try:
            data = path.read_bytes()
        except OSError:
            data = None
        if data is None or hashlib.sha256(data).hexdigest() != row["manifest_sha256"]:
            if strict:
                raise SplitError(
                    f"the general set is registered but {path} is missing or no longer matches its pin; "
                    "restore it (`healing split verify` names the difference) before building or training"
                )
            if data is None:
                return {}
    elif not path.is_file():
        return {}
    return {GENERAL_PREFIX + c["id"]: c["expected"] for c in json.loads(path.read_text())["clips"]}


def snapshot_texts(db: sqlite3.Connection) -> dict[str, list[str]]:
    """`train:<split>:<clip>` -> the texts a recorded snapshot stored for each member when it handed it out.

    Members whose texts another snapshot already stored are left out: the same texts cluster
    together whichever snapshot they came from, so the reservation is the same and 40 snapshots of the
    same clips cost one pass."""
    found: dict[str, list[str]] = {}
    seen: set[tuple[str, ...]] = set()
    for row in db.execute("SELECT split_id, detail FROM splits WHERE kind = 'train' AND detail IS NOT NULL"):
        for clip, texts in json.loads(row["detail"]).get("members", {}).items():
            key = tuple(texts)
            if key not in seen:
                seen.add(key)
                found[f"{TRAIN_PREFIX}{row['split_id']}:{clip}"] = list(texts)
    return found


def clip_clusters(db: sqlite3.Connection, *, snapshots: bool = True, strict: bool = True) -> dict[str, str]:
    """clip_id -> split-cluster, over the dataset's clips, the general set's references and the texts
    recorded snapshots stored.

    The general references join as `general:<id>` entries and snapshot members as `train:<split>:<clip>`
    entries, so a clip that near-duplicates what a test set holds, or what a model saw (as it read then),
    lands in that cluster; callers read clips by id and `general_clusters` reads the rest. `verify` asks
    for neither the snapshots nor a strict general set.
    """
    texts = {
        row["clip_id"]: _texts(row)
        for row in db.execute("SELECT clip_id, reference_text, raw_text FROM clips")
    }
    texts.update({key: [text] for key, text in general_references(db, strict=strict).items()})
    if snapshots:
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


@dataclass(frozen=True)
class Facts:
    """What a set's own conditions read about a clip, from a clips row or from a sidecar entry (an
    edited set is judged by what it pinned)."""

    label: str
    review_strategy: str | None = None
    edited_positions: object = None
    golden: bool = False

    @classmethod
    def of(cls, row) -> Facts:
        return cls(row["label"], row["review_strategy"], row["edited_positions"], bool(row["golden"]))

    @classmethod
    def pinned(cls, entry: dict) -> Facts:
        return cls(entry.get("label", ""), None, entry.get("edited_positions"))


@dataclass
class Zones:
    """Who may test and who may train, loaded once per command: the one leak policy (D-41).

    The cut-off is taken at load. Everything asks the same object through named frames:

    - `rolling(clip)`: may enter a rolling or edited set (or a `--since-cutoff` review): its session
      starts after the cut-off, no golden, general or recorded-snapshot cluster holds it, and it is not
      already in a registered set of the same parent (parents are model versions with their own
      holdouts, so a clip may sit in two parents' sets);
    - `golden(clip)`: may join golden, in any age: as above without the window, and no registered set
      of any parent holds its cluster either;
    - `random_review(clip)`: may be shown to a random review: not in a golden, general or snapshot
      cluster, and not already a registered member of any set (a reviewed edited clip would stop being
      strong);
    - `may_train(clip)`: a training run may take it: outside the window, in no cluster of an in-window
      clip, a registered test set (every parent's) or the general set.

    A recorded snapshot reserves its clusters from the texts it stored when it handed its clips out
    (`train:<split>:<clip>` entries in the clustering, like `general:<id>`), so a clip whose text has
    changed since cannot hide what the model saw. `why_not(kind, ...)` adds each set's own conditions to
    the frame and returns the reason (None: it belongs). `build` selects with it, `golden.add` admits
    with it, and `verify` re-asks it of the registered members, so there is one rule.
    """

    clusters: dict[str, str]  # clip id, `general:<id>` and `train:<split>:<clip>` -> split-cluster root
    sessions: dict[str, Session]
    cutoff_ms: int
    parent: str  # whose rolling and edited sets a clip may not already be in
    owners: dict[str, dict[str, str]]  # cluster -> {split_id: kind} for golden, general, rolling, edited (every parent)
    parents: dict[str, str | None]  # split_id -> its parent version
    members: dict[str, dict[str, str]]  # clip id -> {split_id: kind} registered in rolling, edited, golden
    trained: set[str]  # clusters of the stored texts (and the current clips) of every recorded snapshot
    window: set[str]  # clusters of every clip in a session inside the window

    @classmethod
    def load(
        cls, db: sqlite3.Connection, cutoff_ms: int | None = None, now: int | None = None,
        ignore: Iterable[str] = (), *, parent: str = "current", snapshots: bool = True,
    ) -> Zones:
        """`ignore` names registered sets about to be replaced (a build's old rolling and edited sets
        for its parent), which must not block their own successors. `snapshots=False` is `verify`'s: it
        reserves nothing for recorded snapshots (it reads their stored texts directly instead) and
        tolerates a broken general set (it reports it)."""
        clusters, sess = clip_clusters(db, snapshots=snapshots, strict=snapshots), sessions(db)
        cutoff = cutoff_for(cutoff_ms, now)
        skipped = set(ignore)
        owners: dict[str, dict[str, str]] = {}
        members: dict[str, dict[str, str]] = {}
        parents: dict[str, str | None] = {}
        trained = {root for key, root in clusters.items() if key.startswith(TRAIN_PREFIX)}
        kinds = "'rolling', 'edited', 'golden'" + (", 'train'" if snapshots else "")
        for row in db.execute(
            "SELECT s.split_id, s.kind, s.parent_version, m.clip_id FROM splits s"
            f" JOIN split_members m ON m.split_id = s.split_id WHERE s.kind IN ({kinds})"
        ):
            root = clusters.get(row["clip_id"], row["clip_id"])
            if row["kind"] == "train":
                trained.add(root)
            elif row["split_id"] not in skipped:
                owners.setdefault(root, {})[row["split_id"]] = row["kind"]
                members.setdefault(row["clip_id"], {})[row["split_id"]] = row["kind"]
                parents[row["split_id"]] = row["parent_version"]
        for root in general_clusters(clusters):
            owners.setdefault(root, {})[general.GENERAL_ID] = "general"
        window = {clusters[clip] for clip, session in sess.items() if session.start > cutoff}
        return cls(clusters, sess, cutoff, parent, owners, parents, members, trained, window)

    def in_window(self, clip: str, cutoff_ms: int | None = None, session_start: int | None = None) -> bool:
        start = self.sessions[clip].start if session_start is None else session_start
        return start > (self.cutoff_ms if cutoff_ms is None else cutoff_ms)

    def _others(self, clip: str, own: str | None) -> dict[str, str]:
        """{split_id: kind} of the registered sets whose clusters the clip shares, apart from `own`."""
        return {s: k for s, k in self.owners.get(self.clusters[clip], {}).items() if s != own}

    def _reason(self, clip: str, *, own: str | None, windowed: bool, cutoff_ms: int | None = None,
                session_start: int | None = None, registered: bool = False, members: str = "none") -> str | None:
        """Why `clip` may not be drawn into a frame, or None. `own` is the registered set the clip is
        already judged as a member of (verify), whose own rows are not held against it. `members` is
        which registered sets already holding the clip close it: "parent" (the clip's own parent's, and
        golden), "any", or "none"."""
        if windowed and not self.in_window(clip, cutoff_ms, session_start):
            return "predates the cut-off"
        if self.clusters[clip] in self.trained:
            return "shares a split-cluster with a recorded train snapshot"
        others = self._others(clip, own)
        for split_id, kind in others.items():
            if kind in ("golden", "general"):
                return f"shares a split-cluster with {'the golden set' if kind == 'golden' else split_id}"
        if registered and others:
            return f"shares a split-cluster with {next(iter(others))}"
        if members != "none":
            scope = self.parents.get(own, self.parent) if own else self.parent
            held = {
                s: k for s, k in self.members.get(clip, {}).items()
                if s != own and (members == "any" or self.parents.get(s) in (None, scope))
            }
            if held:
                return f"is already in {next(iter(held))}"
        return None

    def rolling(self, clip: str) -> bool:
        return self._reason(clip, own=None, windowed=True, members="parent") is None

    def golden(self, clip: str) -> bool:
        return self._reason(clip, own=None, windowed=False, registered=True) is None

    def random_review(self, clip: str) -> bool:
        return self._reason(clip, own=None, windowed=False, members="any") is None

    def may_train(self, clip: str) -> bool:
        # `window` holds the cluster of every in-window clip, so it also keeps those clips themselves out.
        root = self.clusters[clip]
        return root not in self.window and root not in self.owners

    def why_not(self, kind: str, clip: str, facts: Facts, *, own: str | None = None,
                cutoff_ms: int | None = None, session_start: int | None = None) -> str | None:
        """The reason `clip` may not be in a `kind` set ("rolling", "edited" or "golden"), or None. The
        set's own conditions on `facts`, then the frame. `own`, `cutoff_ms` and `session_start` are
        `verify`'s: the set the clip is judged as a member of, that set's cut-off, and the session
        start it pinned."""
        if kind == "edited":
            if facts.label != "strong":
                return "is not a strong clip"
            if not facts.edited_positions:
                return "has no edited positions"
        else:  # a gating set takes verified, randomly sampled references only (AC-7)
            if facts.label in ("weak", "strong"):
                return f"gating reference comes from a {facts.label} label"
            if facts.label != "verified" or facts.review_strategy != "random":
                return "is not a randomly sampled, verified clip"
        if kind == "golden":
            return self._reason(clip, own=own, windowed=False, registered=True)
        return self._reason(clip, own=own, windowed=True, cutoff_ms=cutoff_ms, session_start=session_start, members="parent")


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
        zones = Zones.load(db, cutoff_ms, now, parent=parent)
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
            "session_start_ms": sess[row["clip_id"]].start,
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


def _written(store: Store, split_id: str, items: list[dict], written: list[SetInfo]) -> SetInfo:
    """Write a set's folder (recorded in `written`, so a failed build can remove it), or note it is empty."""
    if not items:
        return SetInfo(split_id, 0, 0)
    info = write_and_pin(store, split_id, items)
    written.append(info)
    return info


def _free_id(db: sqlite3.Connection, split_id: str) -> str:
    candidate, n = split_id, 1
    while db.execute("SELECT 1 FROM splits WHERE split_id = ?", (candidate,)).fetchone():
        n += 1
        candidate = f"{split_id}-{n}"
    return candidate


def why_empty(store: Store, kind: str, parent: str = "current") -> str:
    """Why no `kind` ("rolling" or "edited") clip qualified: each candidate clip's reason from the
    same `why_not` that `build` uses, counted."""
    from collections import Counter

    label = {"rolling": "verified", "edited": "strong"}[kind]
    rows = store.db.execute(f"SELECT c.* FROM clips c WHERE {labels.USABLE} AND c.label = ?", (label,)).fetchall()
    if not rows:
        return (f"there are no {label} clips yet" + (
            " (edits you made that sound like what Hex heard)" if kind == "edited"
            else "; run `healing review --since-cutoff`"))
    try:
        zones = Zones.load(store.db, parent=parent)
    except SplitError as error:
        return str(error)
    reasons = Counter(zones.why_not(kind, r["clip_id"], Facts.of(r)) or "qualifies (run `healing split build`)" for r in rows)
    return f"{len(rows)} {label} clips: " + "; ".join(f"{reason} ({n})" for reason, n in reasons.most_common())


def _unique_id(db: sqlite3.Connection, base: str) -> str:
    """`base`, or `base-2`, `base-3`... : an id no registered set uses and no folder holds, so a rebuild in
    the same second never rewrites the folder a still-registered set points at."""
    candidate, n = base, 1
    while db.execute("SELECT 1 FROM splits WHERE split_id = ?", (candidate,)).fetchone() or (sets_dir() / candidate).exists():
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def build(store: Store, parent: str = "current", cutoff_ms: int | None = None, now: int | None = None) -> BuildResult:
    """Rebuild the parent's rolling and edited sets and register golden and general.

    A build is not training: it creates no train rows (`snapshot_train` does). The pin hashes come from
    the files, so they are written inside the same write transaction that decides and registers (a
    shorter one would let a snapshot or a golden add change the facts between); the new folders have
    ids of their own, so the sets still registered stay intact until the swap commits, and a failed
    build removes what it wrote.
    """
    db = store.db
    labels.label_all(store)  # a fresh collect or migration has no labels yet: sets would come out empty
    when = now_ms() if now is None else now
    stamp = datetime.fromtimestamp(when / 1000, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written: list[SetInfo] = []
    try:
        with store.immediate():  # decide and register on one set of facts: a snapshot or golden add waits
            old = db.execute(
                "SELECT split_id FROM splits WHERE frozen = 0 AND parent_version = ? AND kind IN ('rolling', 'edited')",
                (parent,),
            ).fetchall()
            zones = Zones.load(db, cutoff_ms, when, ignore=[r["split_id"] for r in old], parent=parent)
            rolling_rows, edited_rows = [], []
            for row in db.execute(f"SELECT c.* FROM clips c WHERE {labels.USABLE} ORDER BY c.timestamp_ms, c.clip_id"):
                facts = Facts.of(row)
                if zones.why_not("rolling", row["clip_id"], facts) is None:
                    rolling_rows.append(row)
                elif zones.why_not("edited", row["clip_id"], facts) is None:
                    edited_rows.append(row)
            rolling_items = entries(store, rolling_rows, zones.clusters, zones.sessions)
            edited_items = entries(store, edited_rows, zones.clusters, zones.sessions, edited=True)
            rolling = _written(store, _unique_id(db, f"rolling-{parent}-{stamp}"), rolling_items, written)
            edited = _written(store, _unique_id(db, f"edited-{parent}-{stamp}"), edited_items, written)
            for row in old:
                db.execute("DELETE FROM split_members WHERE split_id = ?", (row["split_id"],))
                db.execute("DELETE FROM splits WHERE split_id = ?", (row["split_id"],))
            _register(db, rolling, "rolling", parent, zones.cutoff_ms, when, [i["id"] for i in rolling_items])
            _register(db, edited, "edited", parent, zones.cutoff_ms, when, [i["id"] for i in edited_items])
            general_info = _register_general(db, when)
            # The trainable clips outside the test zone, from the same zones: this build's own sets hold
            # only in-window clips, which the zones' window already keeps out.
            trainable = labels.trainable(store)
            train_ready = sum(1 for r in trainable if zones.may_train(r["clip_id"]))
    except BaseException:
        for info in written:
            shutil.rmtree(sets_dir() / info.split_id, ignore_errors=True)
        raise
    for row in old:
        shutil.rmtree(sets_dir() / row["split_id"], ignore_errors=True)

    golden = golden_row(db)
    frozen = golden is not None and golden["frozen"] == 1
    result = BuildResult(
        parent, zones.cutoff_ms, rolling, edited,
        SetInfo(GOLDEN_ID, golden["n_clips"], golden["n_words"], golden["manifest_sha256"]) if frozen else None,
        len(golden_members(db)), general_info, len(trainable), train_ready,
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
    """A set's sidecar rows (clip id -> row): what it pinned, or {} when unreadable (the pin check
    reports that)."""
    if not row["manifest_path"]:
        return {}
    try:
        path = Path(row["manifest_path"]).parent / "clips.jsonl"
        return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    except (OSError, ValueError, KeyError):
        return {}


def verify_report(store: Store) -> Report:
    """Current-set leaks and snapshot overlaps, kept apart.

    One read transaction, so a build or snapshot committing from another process cannot change the
    registry between the reads and make a clean dataset look broken. Leaks re-ask `Zones.why_not` of
    every registered member, the same predicate `build` selected with (clusters are recomputed from
    the clips). A rolling or golden member is judged by its live label, so a review that rejects it
    shows at once; an edited member by the label and positions its set pinned (its sidecar, covered by
    the pin), so a clip whose correction changes after the build does not flag a set that is still
    exactly what was built; a rolling member's window by the session start it pinned. A recorded train
    snapshot is checked against the test clips' texts directly, from the texts the snapshot stored,
    with the near-duplicate test and no clustering: a clip trained on that a test set now holds is a
    true contamination report, and a union-find that links non-transitively could never make one fail
    permanently (so `Zones` is loaded without snapshots here).
    """
    with store.read_transaction():
        return _verify(store)


def _verify(store: Store) -> Report:
    db = store.db
    report = Report()
    zones = Zones.load(db, snapshots=False)

    def members(split_id: str) -> list[str]:
        return [r["clip_id"] for r in db.execute("SELECT clip_id FROM split_members WHERE split_id = ?", (split_id,))]

    split_rows = db.execute("SELECT * FROM splits ORDER BY split_id").fetchall()
    tests = [r for r in split_rows if r["kind"] in ("rolling", "edited", "golden")]
    test_texts: dict[tuple[str, str], list[str]] = {}
    for row in tests:
        pinned = _pinned(row) if row["kind"] in ("rolling", "edited") else {}
        for clip in members(row["split_id"]):
            c = db.execute("SELECT * FROM clips WHERE clip_id = ?", (clip,)).fetchone()
            if c is None:
                report.leaks.append(f"{row['split_id']}: clip {clip} is gone")
                continue
            test_texts[(row["split_id"], clip)] = _texts(c)
            entry = pinned.get(clip)
            facts = Facts.of(c)
            if row["kind"] == "edited":
                reason = "is not in its pinned sidecar" if entry is None else None
                facts = Facts.pinned(entry or {})
            else:
                reason = None
            reason = reason or zones.why_not(
                row["kind"], clip, facts, own=row["split_id"], cutoff_ms=row["cutoff_ms"],
                session_start=(entry or {}).get("session_start_ms"),
            )
            if reason is None and row["kind"] == "golden" and not c["golden"]:
                reason = "is not marked golden"
            if reason:
                report.leaks.append(f"{row['split_id']}: clip {clip} {reason}")
    for row in split_rows:
        if row["manifest_path"]:
            report.leaks += _check_set(row, db, members(row["split_id"]))
    for key, text in general_references(db, strict=False).items():
        test_texts[(general.GENERAL_ID, key[len(GENERAL_PREFIX):])] = [text]
    report.snapshot_overlaps = _snapshot_overlaps(split_rows, test_texts)
    return report


def _snapshot_overlaps(split_rows: list, test_texts: dict[tuple[str, str], list[str]]) -> list[str]:
    """Every recorded snapshot's stored member texts against every test text, through one index built
    once and one query per distinct text: forty snapshots of the same clips cost one pass."""
    by_key: dict[str, list[tuple[str, str]]] = {}
    for owner, texts in test_texts.items():
        for text in texts:
            if key := dedupe.normalize(text):
                by_key.setdefault(key, []).append(owner)
    index = dedupe.NearIndex(by_key)
    holders: dict[str, list[tuple[str, str]]] = {}  # distinct snapshot text -> [(snapshot, clip)]
    for snapshot in (r for r in split_rows if r["kind"] == "train"):
        for clip, texts in sorted(json.loads(snapshot["detail"] or "{}").get("members", {}).items()):
            for key in filter(None, map(dedupe.normalize, texts)):
                holders.setdefault(key, []).append((snapshot["split_id"], clip))
    found, seen = [], set()
    for key, who in sorted(holders.items(), key=lambda item: item[1][0]):
        owners = [owner for other in index.near(key) for owner in by_key[other]]
        for snapshot_id, clip in who:
            for owner in owners:
                if (snapshot_id, clip, owner) not in seen:
                    seen.add((snapshot_id, clip, owner))
                    found.append(
                        f"{snapshot_id}: a model trained on clip {clip}, which near-duplicates {owner[1]} in {owner[0]}"
                    )
    return found


def _check_set(row: sqlite3.Row, db: sqlite3.Connection, member_ids: list[str]) -> list[str]:
    try:
        folder, _ = verified_set(row)
    except manifest.ManifestError as error:
        text = str(error)
        return [text if text.startswith(row["split_id"]) else f"{row['split_id']}: {text}"]
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
