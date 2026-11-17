"""Dataset snapshots: what a model trained on, as ids and hashes only (no audio, no text).

A snapshot is the sorted lines `[clip_id, audio_sha256, reference_text_sha256, weight]` of one frozen
`train` row (`splits.snapshot_train`, D-41), each a compact JSON array ending in a newline. Its SHA-256
is over those uncompressed lines, so the order rows arrive in never changes it and neither does the
compression. The file is `registry/snapshots/<sha>.jsonl.gz` (gzip at level 9 with no name or time in
its header, D-42: stdlib only, no new dependency), written once.

The reference text and the weight are the ones the train row stored when it handed the clip out, not
the clip's now, so a later relabel cannot change what the snapshot says the model saw.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from healing.registry.objects import fsync_dir, owner_dir, registry_dir


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    sha256: str
    rows: list[tuple[str, str | None, str, float]]
    strong_n: int = 0
    weak_n: int = 0
    split_id: str | None = None
    weights_frozen: bool = True  # False for a train row made before `snapshot_train` stored weights

    @property
    def train_n(self) -> int:
        return len(self.rows)


def snapshots_dir() -> Path:
    return registry_dir() / "snapshots"


def snapshot_path(sha256: str) -> Path:
    return snapshots_dir() / f"{sha256}.jsonl.gz"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def encode(rows) -> bytes:
    """The canonical bytes: rows sorted, one compact JSON array per line."""
    ordered = sorted((str(c), a, str(r), float(w)) for c, a, r, w in rows)
    return "".join(json.dumps(list(row), separators=(",", ":")) + "\n" for row in ordered).encode()


def digest(rows) -> str:
    return hashlib.sha256(encode(rows)).hexdigest()


def from_train(db, split_id: str) -> Snapshot:
    """The snapshot of a frozen `train` row: each member's audio hash, and the reference text and weight
    the row stored for it (D-44), so a relabel after training changes neither the lines nor the hash. A
    row from before weights were stored falls back to the live weight and says so (`weights_frozen`)."""
    row = db.execute("SELECT kind, frozen, detail FROM splits WHERE split_id = ?", (split_id,)).fetchone()
    if row is None:
        raise SnapshotError(f"no split {split_id!r}")
    if row["kind"] != "train" or not row["frozen"]:
        raise SnapshotError(f"{split_id} is not a frozen train snapshot (`splits.snapshot_train` makes those)")
    detail = json.loads(row["detail"] or "{}")
    members = detail.get("members", {})
    frozen_weights = detail.get("weights")
    rows, strong, weak = [], 0, 0
    for clip_id, texts in members.items():
        clip = db.execute("SELECT audio_sha256, weight, label, raw_text FROM clips WHERE clip_id = ?", (clip_id,)).fetchone()
        if clip is None:
            raise SnapshotError(f"{split_id}: clip {clip_id} is gone from the dataset")
        reference = texts[0] if texts else clip["raw_text"]
        weight = frozen_weights.get(clip_id) if isinstance(frozen_weights, dict) else None
        rows.append((clip_id, clip["audio_sha256"], text_sha256(reference), float(clip["weight"] if weight is None else weight)))
        strong += clip["label"] == "strong"
        weak += clip["label"] == "weak"
    if not rows:
        raise SnapshotError(f"{split_id} has no members")
    frozen = isinstance(frozen_weights, dict) and all(c in frozen_weights for c in members)
    return Snapshot(digest(rows), rows, strong, weak, split_id, frozen)


def write(snapshot: Snapshot) -> Path:
    """Store the snapshot under its hash; one already there is kept (it holds the same lines)."""
    target = snapshot_path(snapshot.sha256)
    if target.is_file():
        return target
    owner_dir(snapshots_dir())
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as packed:
        packed.write(encode(snapshot.rows))
    descriptor, scratch = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(buffer.getvalue())
            file.flush()
            os.fsync(file.fileno())
        os.chmod(scratch, 0o600)
        os.replace(scratch, target)
    except BaseException:
        Path(scratch).unlink(missing_ok=True)
        raise
    fsync_dir(target.parent)
    return target


def read(sha256: str) -> list[tuple[str, str | None, str, float]]:
    """A stored snapshot's rows, refused unless they still hash to its name."""
    try:
        data = gzip.decompress(snapshot_path(sha256).read_bytes())
    except FileNotFoundError as error:
        raise SnapshotError(f"no snapshot {sha256[:12]} in {snapshots_dir()}") from error
    except (OSError, EOFError) as error:
        raise SnapshotError(f"snapshot {sha256[:12]} is unreadable: {error}") from error
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SnapshotError(f"snapshot {sha256[:12]} no longer matches its hash")
    return [tuple(json.loads(line)) for line in data.decode().splitlines()]


def reproducible(store, rows) -> str:
    """`full` while every clip's audio is in the dataset and on disk, else `partial` (weak clips may be
    evicted; strong ones never are)."""
    for _, audio, _, _ in rows:
        if not audio:
            return "partial"
        kept = store.db.execute(
            "SELECT 1 FROM audio WHERE sha256 = ? AND evicted_at_ms IS NULL AND missing = 0 AND corrupt = 0", (audio,)
        ).fetchone()
        if kept is None or not store.audio_path(audio).is_file():
            return "partial"
    return "full"
