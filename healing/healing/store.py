"""The dataset Hex can't expire: `dataset.sqlite` plus FLAC audio, owner-only.

Layout under `$HEALING_HOME/dataset/`: `dataset.sqlite` and `audio/<aa>/<sha256>.flac`.
Mirrors Hex's own recordings: temp file, fsync, rename, mode 0600 (directories 0700).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile

from healing.paths import healing_home

SAMPLE_RATE = 16_000
SCHEMA_VERSION = 5

SCHEMA = """
CREATE TABLE audio (
    sha256 TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    samples INTEGER NOT NULL,
    evicted_at_ms INTEGER
);
CREATE TABLE clips (
    clip_id TEXT PRIMARY KEY,
    hex_id INTEGER NOT NULL,
    hex_epoch_ms INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    application TEXT,
    mode TEXT,
    kind TEXT NOT NULL,
    model_id TEXT,
    model_name TEXT,
    rewritten INTEGER NOT NULL DEFAULT 0,
    audio_ms INTEGER,
    raw_text TEXT NOT NULL,
    final_text TEXT NOT NULL,
    audio_sha256 TEXT REFERENCES audio(sha256),
    cluster_id TEXT,
    label TEXT NOT NULL DEFAULT 'unlabelled',
    golden INTEGER NOT NULL DEFAULT 0,
    collected_at_ms INTEGER NOT NULL,
    UNIQUE (hex_epoch_ms, hex_id)
);
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_ms INTEGER NOT NULL,
    since_ms INTEGER,
    seen INTEGER NOT NULL DEFAULT 0,
    new INTEGER NOT NULL DEFAULT 0,
    audio_new INTEGER NOT NULL DEFAULT 0,
    missing_audio INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
"""

# Migrations run in order from the stored `user_version`; index i upgrades i -> i + 1.
SCHEMA_V2 = """
ALTER TABLE clips ADD COLUMN redundancy REAL NOT NULL DEFAULT 0;
CREATE TABLE clusters (
    cluster_id TEXT PRIMARY KEY,
    norm_text TEXT NOT NULL,
    size INTEGER NOT NULL,
    audio_kept INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL
);
CREATE INDEX clips_cluster ON clips(cluster_id);
"""

SCHEMA_V3 = """
CREATE INDEX clips_audio ON clips(audio_sha256);
"""

SCHEMA_V4 = """
ALTER TABLE audio ADD COLUMN missing INTEGER NOT NULL DEFAULT 0;
ALTER TABLE audio ADD COLUMN corrupt INTEGER NOT NULL DEFAULT 0;
CREATE TABLE clean_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at_ms INTEGER NOT NULL,
    orphans_moved INTEGER NOT NULL,
    missing_marked INTEGER NOT NULL,
    mismatched_marked INTEGER NOT NULL,
    temp_removed INTEGER NOT NULL,
    freed_bytes INTEGER NOT NULL,
    free_bytes_after INTEGER NOT NULL,
    free_basis TEXT NOT NULL
);
"""

SCHEMA_V5 = """
ALTER TABLE clips ADD COLUMN correction_outcome TEXT;
ALTER TABLE clips ADD COLUMN correction_reason TEXT;
ALTER TABLE clips ADD COLUMN edited_text TEXT;
ALTER TABLE clips ADD COLUMN reference_text TEXT;
CREATE TABLE edits (
    clip_id TEXT NOT NULL REFERENCES clips(clip_id),
    idx INTEGER NOT NULL,
    raw_from TEXT NOT NULL,
    final_from TEXT NOT NULL,
    to_text TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('acoustic', 'rewrite', 'unmapped')),
    score_metaphone INTEGER,
    score_char REAL,
    score_phone REAL,
    PRIMARY KEY (clip_id, idx),
    CHECK (kind != 'acoustic' OR raw_from != '')
);
"""

MIGRATIONS = [SCHEMA, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5]


class StoreError(Exception):
    pass


def dataset_root() -> Path:
    return healing_home() / "dataset"


@contextlib.contextmanager
def process_lock():
    """One healing writer at a time (`collect`, `prune`): `$HEALING_HOME/.lock`."""
    home = healing_home()
    _owner_dir(home)
    descriptor = os.open(home / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise StoreError("another healing process is running") from error
        yield
    finally:
        os.close(descriptor)


def now_ms() -> int:
    return int(time.time() * 1000)


def clip_id(epoch_ms: int, hex_id: int) -> str:
    return hashlib.sha256(f"{epoch_ms}:{hex_id}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Pcm:
    sha256: str
    samples: int
    data: np.ndarray  # int16, mono


def read_wav(path: Path) -> Pcm:
    """Decode a Hex recording (16 kHz mono s16). Anything else is a StoreError."""
    try:
        info = soundfile.info(path)
        if info.samplerate != SAMPLE_RATE or info.channels != 1:
            raise StoreError("recording is not 16 kHz mono")
        data, _ = soundfile.read(path, dtype="int16", always_2d=False)
    except (OSError, RuntimeError) as error:
        raise StoreError(f"unreadable recording: {error.__class__.__name__}") from error
    digest = hashlib.sha256(data.astype("<i2").tobytes()).hexdigest()
    return Pcm(digest, int(data.shape[0]), data)


def _owner_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


class Store:
    def __init__(self, root: Path | None = None, sweep: bool = True):
        self.root = root or dataset_root()
        _owner_dir(self.root)
        _owner_dir(self.root / "audio")
        database = self.root / "dataset.sqlite"
        fresh = not database.exists()
        self.db = sqlite3.connect(database)
        self.db.row_factory = sqlite3.Row
        if fresh:
            os.chmod(database, 0o600)
        self._migrate()
        if sweep:
            self.sweep_temporaries()

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StoreError(f"dataset is schema {version}; this healing knows {SCHEMA_VERSION}")
        for step in range(version, SCHEMA_VERSION):
            self.db.executescript(MIGRATIONS[step])
            self.db.execute(f"PRAGMA user_version = {step + 1}")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def sweep_temporaries(self) -> None:
        for leftover in (self.root / "audio").glob("*/*.tmp"):
            leftover.unlink(missing_ok=True)

    def audio_path(self, sha256: str) -> Path:
        return self.root / "audio" / sha256[:2] / f"{sha256}.flac"

    def write_audio(self, pcm: Pcm) -> bool:
        """Store `pcm` as FLAC keyed by its hash. True when a file was written now."""
        target = self.audio_path(pcm.sha256)
        row = self.db.execute(
            "SELECT 1 FROM audio WHERE sha256 = ? AND evicted_at_ms IS NULL", (pcm.sha256,)
        ).fetchone()
        if row and target.is_file():
            return False
        _owner_dir(target.parent)
        temporary = target.with_suffix(".flac.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as file:
                soundfile.write(file, pcm.data, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                file.flush()
                os.fsync(file.fileno())
            os.rename(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self.db.execute(
            "INSERT INTO audio (sha256, path, bytes, samples, evicted_at_ms) VALUES (?, ?, ?, ?, NULL)"
            " ON CONFLICT(sha256) DO UPDATE SET path = excluded.path, bytes = excluded.bytes,"
            " samples = excluded.samples, evicted_at_ms = NULL, missing = 0, corrupt = 0",
            (
                pcm.sha256,
                str(target.relative_to(self.root)),
                target.stat().st_size,
                pcm.samples,
            ),
        )
        return True
