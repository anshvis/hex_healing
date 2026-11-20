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
SCHEMA_VERSION = 9

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

SCHEMA_V6 = """
ALTER TABLE clips ADD COLUMN classified_at_ms INTEGER;
"""

# v7 (M3): weights, review provenance and the split registry. Every statement is guarded, so a run
# that stopped between two of them (executescript commits per statement) finishes on the next open.
M3_COLUMNS = {
    "weight": "REAL NOT NULL DEFAULT 0",
    "edited_positions": "TEXT",
    "review_strategy": "TEXT",
    "acoustic_reference": "TEXT",  # the classifier's output; `reference_text` is derived from the label
}

SCHEMA_M3 = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id TEXT NOT NULL REFERENCES clips(clip_id),
    reviewed_at_ms INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('accept', 'fix', 'reject')),
    text TEXT,
    strategy TEXT NOT NULL CHECK (strategy IN ('random', 'uncertain'))
);
CREATE INDEX IF NOT EXISTS reviews_clip ON reviews(clip_id);
CREATE TABLE IF NOT EXISTS splits (
    split_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('rolling', 'golden', 'general', 'edited', 'train')),
    parent_version TEXT,
    cutoff_ms INTEGER,
    created_at_ms INTEGER NOT NULL,
    manifest_path TEXT,
    manifest_sha256 TEXT,
    frozen INTEGER NOT NULL DEFAULT 0,
    n_clips INTEGER NOT NULL DEFAULT 0,
    n_words INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS split_members (
    split_id TEXT NOT NULL REFERENCES splits(split_id),
    clip_id TEXT NOT NULL REFERENCES clips(clip_id),
    PRIMARY KEY (split_id, clip_id)
);
CREATE INDEX IF NOT EXISTS split_members_clip ON split_members(clip_id);
-- G-10 in the schema: exactly one golden row is ever frozen, and a frozen split never changes.
CREATE UNIQUE INDEX IF NOT EXISTS one_frozen_golden ON splits(kind) WHERE frozen = 1 AND kind = 'golden';
CREATE TRIGGER IF NOT EXISTS frozen_split_update BEFORE UPDATE ON splits WHEN OLD.frozen = 1
BEGIN SELECT RAISE(ABORT, 'a frozen split never changes'); END;
CREATE TRIGGER IF NOT EXISTS frozen_split_delete BEFORE DELETE ON splits WHEN OLD.frozen = 1
BEGIN SELECT RAISE(ABORT, 'a frozen split never changes'); END;
CREATE TRIGGER IF NOT EXISTS frozen_members_insert BEFORE INSERT ON split_members
WHEN (SELECT frozen FROM splits WHERE split_id = NEW.split_id) = 1
BEGIN SELECT RAISE(ABORT, 'a frozen split never changes'); END;
CREATE TRIGGER IF NOT EXISTS frozen_members_update BEFORE UPDATE ON split_members
WHEN (SELECT frozen FROM splits WHERE split_id = OLD.split_id) = 1
   OR (SELECT frozen FROM splits WHERE split_id = NEW.split_id) = 1
BEGIN SELECT RAISE(ABORT, 'a frozen split never changes'); END;
CREATE TRIGGER IF NOT EXISTS frozen_members_delete BEFORE DELETE ON split_members
WHEN (SELECT frozen FROM splits WHERE split_id = OLD.split_id) = 1
BEGIN SELECT RAISE(ABORT, 'a frozen split never changes'); END;
"""

# v8 (M4): the miner's latest candidates (`healing vocab mine`); a recomputed table, not evidence.
SCHEMA_V8 = """
CREATE TABLE IF NOT EXISTS vocab_candidates (
    term TEXT PRIMARY KEY,
    sounds_like TEXT NOT NULL,
    pair_count INTEGER NOT NULL CHECK (pair_count >= 2),
    recent_uses INTEGER NOT NULL,
    score REAL NOT NULL,
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL,
    source_split TEXT NOT NULL CHECK (source_split IN ('train', 'all'))
);
"""


# v9 (M8, T8.5): Tidy after paste's decisions, the tidied text a correction is measured from, and
# where a label came from. `raw_text` and `final_text` never change for a tidy (D-38).
TIDY_KINDS = ("filler", "repeat", "self_correction", "misheard")
TIDY_COLUMNS = {"tidied_text": "TEXT", "tidy_applied": "INTEGER NOT NULL DEFAULT 0", "label_source": "TEXT"}
TIDY_EVENTS = f"""
CREATE TABLE IF NOT EXISTS tidy_events (
    clip_id TEXT NOT NULL REFERENCES clips(clip_id),
    idx INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL CHECK (kind IN ({", ".join(repr(k) for k in TIDY_KINDS)})),
    token TEXT NOT NULL CHECK (length(CAST(token AS BLOB)) <= 32),
    replacement TEXT CHECK (replacement IS NULL OR length(CAST(replacement AS BLOB)) <= 32),
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    applied INTEGER NOT NULL DEFAULT 0 CHECK (applied IN (0, 1)),
    source TEXT NOT NULL CHECK (source IN ('rule', 'model'))
);
CREATE INDEX IF NOT EXISTS tidy_events_clip ON tidy_events(clip_id, idx);
"""


def _migrate_v9(db: sqlite3.Connection) -> None:
    """The T8.5 step, guarded like the M3 one, so a step cut short (or repeated) finishes."""
    have = {row[1] for row in db.execute("PRAGMA table_info(clips)")}
    for name, definition in TIDY_COLUMNS.items():
        if name not in have:
            db.execute(f"ALTER TABLE clips ADD COLUMN {name} {definition}")
    db.executescript(TIDY_EVENTS)


# `splits` columns beyond the table's base layout, defined once and always added by the guarded loop below:
# the pin of the sidecar, and a train snapshot's record (model, note, each member's texts).
M3_SPLIT_COLUMNS = {"sidecar_sha256": "TEXT", "detail": "TEXT"}


def ensure_m3_schema(db: sqlite3.Connection) -> bool:
    """Create whatever of the M3 schema is missing; True when it had to. Safe to repeat, so it runs on
    every open: a v7 file made by an earlier build of this schema (before `acoustic_reference`, the
    sidecar pin or the snapshot detail existed) is repaired instead of failing."""
    have = {row[1] for row in db.execute("PRAGMA table_info(clips)")}
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    split_have = {row[1] for row in db.execute("PRAGMA table_info(splits)")} if "splits" in tables else None
    missing = [n for n in M3_COLUMNS if n not in have]
    if split_have is not None:
        missing += [n for n in M3_SPLIT_COLUMNS if n not in split_have]
    if not missing and {"reviews", "splits", "split_members"} <= tables:
        return False
    for name in (n for n in M3_COLUMNS if n not in have):
        db.execute(f"ALTER TABLE clips ADD COLUMN {name} {M3_COLUMNS[name]}")
    db.executescript(SCHEMA_M3)
    split_have = {row[1] for row in db.execute("PRAGMA table_info(splits)")}
    for name, definition in M3_SPLIT_COLUMNS.items():
        if name not in split_have:
            db.execute(f"ALTER TABLE splits ADD COLUMN {name} {definition}")
    return True


def reset_stale_classifications(db: sqlite3.Connection) -> None:
    """Classify again (the next `label all` does) every clip classified before `acoustic_reference`
    existed: it has an acoustic edit, a classification time and no reference of its own. Before that
    column the classifier wrote `reference_text`, which labels later overwrote, and `edited_positions`
    did not exist. Safe to repeat and cheap when there is nothing to do: once reclassified, such a clip
    has its reference and is never selected again."""
    stale = (
        "classified_at_ms IS NOT NULL AND acoustic_reference IS NULL"
        " AND clip_id IN (SELECT clip_id FROM edits WHERE kind = 'acoustic')"
    )
    if db.execute(f"SELECT 1 FROM clips WHERE {stale} LIMIT 1").fetchone():
        db.execute(f"UPDATE clips SET classified_at_ms = NULL WHERE {stale}")


def _migrate_m3(db: sqlite3.Connection) -> None:
    """The M3 step: the schema, then the reset. Both are safe to repeat, so this runs on every open and
    a file stamped v7 by an earlier build of this schema is repaired and its strong clips come back."""
    ensure_m3_schema(db)
    reset_stale_classifications(db)


MIGRATIONS = [
    SCHEMA, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, _migrate_m3, SCHEMA_V8, _migrate_v9,
]


class StoreError(Exception):
    pass


def dataset_root() -> Path:
    return healing_home() / "dataset"


def lock_free(descriptor: int) -> bool:
    """Take an exclusive lock on an open file without waiting; False when another process holds it."""
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


@contextlib.contextmanager
def locked(path: Path, *, wait: bool, busy: str = "another healing process is running"):
    """Hold an exclusive `flock` on `path` (created owner-only) for the block. `wait` blocks until it
    is free; otherwise a held lock raises `StoreError(busy)`."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if wait:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        elif not lock_free(descriptor):
            raise StoreError(busy)
        yield
    finally:
        os.close(descriptor)


def process_lock():
    """One healing writer at a time (`collect`, `prune`): `$HEALING_HOME/.lock`."""
    home = healing_home()
    _owner_dir(home)
    return locked(home / ".lock", wait=False)


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


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backup_is_whole(path: Path, version: int) -> bool:
    if not path.is_file():
        return False
    try:
        copy = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return copy.execute("PRAGMA user_version").fetchone()[0] == version and (
                copy.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            ) and copy.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
        finally:
            copy.close()
    except sqlite3.Error:
        return False


class Store:
    def __init__(self, root: Path | None = None, sweep: bool = True):
        self.root = root or dataset_root()
        _owner_dir(self.root)
        _owner_dir(self.root / "audio")
        database = self.root / "dataset.sqlite"
        fresh = not database.exists()
        self.db = sqlite3.connect(database)
        self.db.row_factory = sqlite3.Row
        # INSERT OR REPLACE deletes the old row, and only fires delete triggers when this is on.
        self.db.execute("PRAGMA recursive_triggers = ON")
        self.db.execute("PRAGMA busy_timeout = 60000")  # a scheduled collect may hold the file a moment
        if fresh:
            os.chmod(database, 0o600)
        self._migrate()
        if sweep:
            self.sweep_temporaries()

    def _migrate(self) -> None:
        """Bring the file to the current schema, one process at a time.

        The exclusive lock covers reading the version, the backup and the steps, and the version is
        read again once it is held, so a second opener finds the work done instead of repeating it.
        A file already past the M3 step still has its M3 schema checked and repaired.
        """
        with locked(self.root / ".migrate.lock", wait=True):
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StoreError(f"dataset is schema {version}; this healing knows {SCHEMA_VERSION}")
            if 0 < version < SCHEMA_VERSION:
                self._back_up(version)
            for step in range(version, SCHEMA_VERSION):
                migration = MIGRATIONS[step]
                if callable(migration):
                    migration(self.db)
                else:
                    self.db.executescript(migration)
                self.db.execute(f"PRAGMA user_version = {step + 1}")
            if version >= 7:  # the M3 step did not run above: still check its schema and its reset
                _migrate_m3(self.db)
            self.db.commit()

    @contextlib.contextmanager
    def immediate(self):
        """One write transaction that takes the write lock up front (`BEGIN IMMEDIATE`), so a read
        made inside it can't be stale by the time its write lands: concurrent runs serialize here
        (waiting out the busy timeout) instead of both deciding on the same facts. Refuses to start
        with a transaction already open, because it would have to commit someone's half-done work."""
        self._no_pending()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except BaseException:
            self.db.rollback()
            raise
        self.db.commit()

    @contextlib.contextmanager
    def read_transaction(self):
        """One read-only view: every read inside sees the registry as it was at the first, whatever
        another process commits meanwhile (a writer waits at its commit until this ends). Inside a
        transaction the caller already holds, that one is the view and stays the caller's."""
        if self.db.in_transaction:
            yield self.db
            return
        self.db.execute("BEGIN")
        try:
            yield self.db
        finally:
            self.db.rollback()

    def _no_pending(self) -> None:
        if self.db.in_transaction:
            raise StoreError("a transaction is already open: commit it before starting another")

    def _back_up(self, version: int) -> None:
        """Copy the dataset before a migration touches it: its rows are the user's only copy of what
        History has since expired. One backup per starting version, owner-only.

        Written beside the target and moved into place only once complete and synced, so a crash
        leaves no half file under the real name. A backup already there is kept only if it opens at
        that version and passes a quick check; an empty or damaged one is made again.
        """
        target = self.root / f"dataset.sqlite.pre-v{version}"
        if _backup_is_whole(target, version):
            return
        temporary = target.with_name(target.name + ".tmp")
        try:
            temporary.unlink(missing_ok=True)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
            copy = sqlite3.connect(temporary)
            try:
                self.db.backup(copy)
            finally:
                copy.close()
            _fsync(temporary)
            os.replace(temporary, target)
            _fsync(self.root)
        except (sqlite3.Error, OSError) as error:
            temporary.unlink(missing_ok=True)
            raise StoreError(f"could not back up the dataset before migrating it: {error}") from error

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
