"""`healing clean`: keep the audio index true and the disk from filling.

Everything stays under `$HEALING_HOME`. Nothing here follows a symlink: a link is judged by `lstat`,
never opened, and a tree walk never descends through one (G-6). Orphans and corrupt files are moved
to `quarantine/<at_ms>/`, never deleted; quarantine is emptied after 7 days. The report holds counts
and bytes only, never text (G-4). Hex's own recordings are not touched (G-6).
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import soundfile

from healing.budget import PruneError, prune
from healing.paths import healing_home
from healing.store import Store, now_ms

DEFAULT_MIN_FREE = 10 * 1024**3
HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS
QUARANTINE_MS = 7 * DAY_MS
SCRATCH_MS = DAY_MS
TEMP_SUFFIXES = (".tmp", ".partial", ".part")
FREE_BASIS = "statvfs_bavail"  # blocks available to this user; not APFS purgeable space


class CleanError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def free_bytes(path: Path | None = None) -> int:
    stat_result = os.statvfs(path or healing_home())
    return stat_result.f_bavail * stat_result.f_frsize


@dataclass
class Report:
    orphans_moved: int = 0
    missing_marked: int = 0
    mismatched_marked: int = 0
    temp_removed: int = 0
    quarantine_emptied: int = 0
    scratch_removed: int = 0
    freed_bytes: int = 0
    free_bytes_after: int = 0
    shortfall: int = 0
    actions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"orphans {self.orphans_moved}, missing {self.missing_marked}, "
            f"mismatched {self.mismatched_marked}, temp {self.temp_removed}, "
            f"quarantine {self.quarantine_emptied}, scratch {self.scratch_removed}, "
            f"freed {self.freed_bytes} bytes, free {self.free_bytes_after} bytes ({FREE_BASIS})"
        )

    def changed(self) -> bool:
        return bool(
            self.orphans_moved or self.missing_marked or self.mismatched_marked or self.temp_removed
            or self.quarantine_emptied or self.scratch_removed or self.freed_bytes
        )


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError:
        return None


def _is_regular(info: os.stat_result | None) -> bool:
    return info is not None and stat.S_ISREG(info.st_mode)


def _inside(home: Path, path: Path) -> bool:
    """True when `path`, with its parent links resolved, is inside `home` (the file itself is not)."""
    try:
        return path.parent.resolve().is_relative_to(home.resolve())
    except OSError:
        return False


def _tree_bytes(path: Path) -> int:
    total = 0
    for folder, _dirs, names in os.walk(path, followlinks=False):
        for name in names:
            info = _lstat(Path(folder) / name)
            if info is not None and stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _pcm_sha256(path: Path) -> str | None:
    try:
        data, _ = soundfile.read(path, dtype="int16", always_2d=False)
    except (OSError, RuntimeError):
        return None
    return hashlib.sha256(data.astype("<i2").tobytes()).hexdigest()


def _orphans(store: Store) -> list[Path]:
    known = {row["path"] for row in store.db.execute("SELECT path FROM audio")}
    found = []
    audio = store.root / "audio"
    for shard in sorted(audio.iterdir()) if audio.is_dir() else []:
        if not stat.S_ISDIR(shard.lstat().st_mode):  # a linked shard is never entered
            continue
        for candidate in sorted(shard.glob("*.flac")):
            if str(candidate.relative_to(store.root)) in known:
                continue
            if _is_regular(_lstat(candidate)):
                found.append(candidate)
    return found


def index_problems(store: Store) -> dict[str, int]:
    """A cheap, read-only size check: how far the index is from the disk right now."""
    problems = {"orphans": len(_orphans(store)), "missing": 0, "mismatched": 0}
    for row in store.db.execute("SELECT path, bytes FROM audio WHERE evicted_at_ms IS NULL"):
        info = _lstat(store.root / row["path"])
        if not _is_regular(info):
            problems["missing"] += 1
        elif info.st_size != row["bytes"]:
            problems["mismatched"] += 1
    return problems


def _real_dir(path: Path) -> bool:
    info = _lstat(path)
    return info is not None and stat.S_ISDIR(info.st_mode)


def _quarantine(home: Path, at_ms: int, path: Path) -> bool:
    """Move `path` into quarantine; False (nothing moved) when quarantine or its day is a link."""
    root = home / "quarantine"
    target_dir = root / str(at_ms)
    if _lstat(root) is not None and not _real_dir(root):
        return False
    if _lstat(root) is None:
        root.mkdir(mode=0o700)
    if _lstat(target_dir) is not None and not _real_dir(target_dir):
        return False
    if _lstat(target_dir) is None:
        target_dir.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    os.chmod(target_dir, 0o700)
    target = target_dir / path.name
    suffix = 0
    while target.exists() or target.is_symlink():
        suffix += 1
        target = target_dir / f"{path.name}.{suffix}"
    os.rename(path, target)
    return True


def _sweep_temp(home: Path, now: int, report: Report, dry_run: bool) -> None:
    for folder, dirs, names in os.walk(home, followlinks=False):
        if Path(folder) == home:
            dirs[:] = [d for d in dirs if d != "quarantine"]
        for name in names:
            if not name.endswith(TEMP_SUFFIXES):
                continue
            path = Path(folder) / name
            info = _lstat(path)
            if info is None or not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                continue
            if now - int(info.st_mtime * 1000) < HOUR_MS:
                continue
            report.temp_removed += 1
            report.freed_bytes += info.st_size if stat.S_ISREG(info.st_mode) else 0
            report.actions.append(f"remove temp {path.relative_to(home)}")
            if not dry_run:
                path.unlink(missing_ok=True)


def _sweep_quarantine(home: Path, now: int, report: Report, dry_run: bool) -> None:
    root = home / "quarantine"
    info = _lstat(root)
    if info is None or not stat.S_ISDIR(info.st_mode):
        return
    for entry in sorted(root.iterdir()):
        entry_info = _lstat(entry)
        if entry_info is None or not stat.S_ISDIR(entry_info.st_mode):
            continue
        stamp = int(entry.name) if entry.name.isdigit() else int(entry_info.st_mtime * 1000)
        if now - stamp < QUARANTINE_MS:
            continue
        report.quarantine_emptied += 1
        report.freed_bytes += _tree_bytes(entry)
        report.actions.append(f"empty quarantine {entry.name}")
        if not dry_run:
            shutil.rmtree(entry)


def _round_running(round_dir: Path) -> bool:
    """A round holds an exclusive flock on `rounds/<id>/.lock` while it runs."""
    lock = round_dir / ".lock"
    if not _is_regular(_lstat(lock)):
        return False
    try:
        descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return True
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        return False
    finally:
        os.close(descriptor)


def _sweep_scratch(home: Path, now: int, report: Report, dry_run: bool) -> None:
    rounds = home / "rounds"
    info = _lstat(rounds)
    if info is None or not stat.S_ISDIR(info.st_mode):
        return
    for round_dir in sorted(rounds.iterdir()):
        round_info = _lstat(round_dir)
        work = round_dir / "work"
        work_info = _lstat(work)
        if round_info is None or not stat.S_ISDIR(round_info.st_mode):
            continue
        if work_info is None or not stat.S_ISDIR(work_info.st_mode):  # a link is not scratch
            continue
        if now - int(work_info.st_mtime * 1000) < SCRATCH_MS or _round_running(round_dir):
            continue
        report.scratch_removed += 1
        report.freed_bytes += _tree_bytes(work)
        report.actions.append(f"remove scratch {round_dir.name}")
        if not dry_run:
            shutil.rmtree(work)


def _check_index(store: Store, now: int, deep: bool, report: Report, dry_run: bool) -> None:
    home = store.root.parent
    for path in _orphans(store):
        if not dry_run and not _quarantine(home, now, path):
            continue
        report.orphans_moved += 1
        report.actions.append(f"quarantine orphan {path.relative_to(store.root)}")
    for row in store.db.execute(
        "SELECT sha256, path, bytes FROM audio WHERE evicted_at_ms IS NULL ORDER BY sha256"
    ).fetchall():
        path = store.root / row["path"]
        info = _lstat(path)
        if not _is_regular(info) or not _inside(home, path):
            report.missing_marked += 1
            report.actions.append(f"mark missing {row['sha256'][:12]}")
            if not dry_run:
                store.db.execute(
                    "UPDATE audio SET evicted_at_ms = ?, missing = 1 WHERE sha256 = ?",
                    (now, row["sha256"]),
                )
            continue
        bad = info.st_size != row["bytes"] or (deep and _pcm_sha256(path) != row["sha256"])
        if not bad:
            continue
        report.mismatched_marked += 1
        report.actions.append(f"mark corrupt {row['sha256'][:12]}")
        if not dry_run:
            store.db.execute(
                "UPDATE audio SET evicted_at_ms = ?, corrupt = 1 WHERE sha256 = ?", (now, row["sha256"])
            )
            _quarantine(home, now, path)  # a refused move leaves the file; the row is marked
    if not dry_run:
        store.db.commit()


def clean(
    store: Store,
    min_free: int = DEFAULT_MIN_FREE,
    deep: bool = False,
    dry_run: bool = False,
    now: int | None = None,
    free_fn: Callable[[], int] | None = None,
    k: int | None = None,
) -> Report:
    """Run every step in order; the caller holds `process_lock`. Raises CleanError (4) below the floor."""
    now = now_ms() if now is None else now
    free_fn = free_fn or (lambda: free_bytes(store.root.parent))
    home = store.root.parent
    report = Report()

    _sweep_temp(home, now, report, dry_run)
    _check_index(store, now, deep, report, dry_run)
    _sweep_quarantine(home, now, report, dry_run)
    _sweep_scratch(home, now, report, dry_run)

    free = free_fn()
    report.shortfall = max(min_free - free, 0)
    prune_error: PruneError | None = None
    if report.shortfall:
        total = store.db.execute(
            "SELECT COALESCE(SUM(bytes), 0) FROM audio WHERE evicted_at_ms IS NULL"
        ).fetchone()[0]
        try:
            result = prune(
                store, max(total - report.shortfall, 0), **({"k": k} if k else {}),
                dry_run=dry_run, now=now, lowering=True,
            )
        except PruneError as error:
            prune_error = error
        else:
            report.freed_bytes += result.evicted_bytes
            report.actions += [f"floor {eviction}" for eviction in result.evictions]
        if prune_error is not None:  # evictions made before it gave up are in the index
            left = store.db.execute(
                "SELECT COALESCE(SUM(bytes), 0) FROM audio WHERE evicted_at_ms IS NULL"
            ).fetchone()[0]
            report.freed_bytes += total - left
        free = free_fn()
    report.free_bytes_after = free

    if not dry_run:
        _record(store, report, now)
    if report.shortfall and not dry_run and free < min_free:
        raise CleanError(
            f"free space is {min_free - free} bytes under the {min_free}-byte floor and what is left "
            "is protected or younger than 7 days; collection keeps text but copies no audio",
            4,
        )
    return report


def _record(store: Store, report: Report, now: int) -> None:
    store.db.execute(
        "INSERT INTO clean_runs (at_ms, orphans_moved, missing_marked, mismatched_marked, temp_removed,"
        " freed_bytes, free_bytes_after, free_basis) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            now, report.orphans_moved, report.missing_marked, report.mismatched_marked,
            report.temp_removed, report.freed_bytes, report.free_bytes_after, FREE_BASIS,
        ),
    )
    store.db.commit()


def main_clean(min_free: int, deep: bool, dry_run: bool) -> int:
    import sys

    from healing.store import StoreError, process_lock

    try:
        with process_lock():
            store = Store(sweep=False)
            try:
                report = clean(store, min_free, deep, dry_run)
            except CleanError as error:
                print(f"error: {error}", file=sys.stderr)
                return error.code
            finally:
                store.close()
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if dry_run:
        for action in report.actions:
            print(action)
        if report.shortfall:
            print(f"free space is {report.shortfall} bytes under the floor")
    print(report.summary())
    return 0
