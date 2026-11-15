"""`healing collect`: copy Hex's History into the dataset, idempotently.

Reads `hex history --json --raw --oldest-first` (read-only, versioned) and writes only
under `$HEALING_HOME`. A clip is keyed by (Hex store epoch, entry id).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from healing.corrections import label_after
from healing.eval.bench import BenchError, hex_binary
from healing.store import Store, StoreError, process_lock, clip_id, now_ms, read_wav

PAGE_LIMIT = 10_000
BATCH = 500
RESCAN_MS = 2 * 24 * 3600 * 1000
HISTORY_VERSION = 1

# Fields Hex can change after an entry exists (a correction, Transcribe again).
MUTABLE = (
    "application",
    "mode",
    "kind",
    "model_id",
    "model_name",
    "rewritten",
    "audio_ms",
    "raw_text",
    "final_text",
    "correction_outcome",
    "correction_reason",
    "edited_text",
)

# A change to any of these makes a stored classification stale.
CLASSIFICATION_INPUTS = ("raw_text", "final_text", "edited_text", "correction_outcome")


class CollectError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


@dataclass
class Summary:
    seen: int = 0
    new: int = 0
    audio_new: int = 0
    missing_audio: int = 0
    audio_skipped: int = 0  # not copied: free space is under the floor

    def __str__(self) -> str:
        text = (
            f"seen {self.seen}, new {self.new}, audio {self.audio_new}, "
            f"missing_audio {self.missing_audio}"
        )
        return text + (f", audio_skipped {self.audio_skipped}" if self.audio_skipped else "")


def rfc3339(ms: int) -> str:
    """Whole seconds, rounded down: `--since` is inclusive and rows are deduplicated."""
    moment = datetime.fromtimestamp(ms // 1000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def check_since(value: str) -> str:
    if re.fullmatch(r"\d+", value):
        raise CollectError("--since takes 3d, a date or RFC 3339, not epoch ms", 2)
    return value


def run_history(binary: Path, since: str | None) -> dict:
    command = [str(binary), "history", "--json", "--raw", "--oldest-first", "--limit", str(PAGE_LIMIT)]
    if since:
        command += ["--since", since]
    env = {**os.environ, "HEX_DISABLE_UPDATES": "1"}
    done = subprocess.run(command, capture_output=True, env=env, check=False)
    if done.returncode != 0:
        raise CollectError(f"hex history exited {done.returncode}", 1)
    try:
        document = json.loads(done.stdout)
    except ValueError as error:
        raise CollectError("hex history printed invalid JSON", 3) from error
    if document.get("version") != HISTORY_VERSION:
        raise CollectError(
            f"hex history version {document.get('version')!r} is not {HISTORY_VERSION}; refusing", 3
        )
    return document


class Epochs:
    """Which Hex store (epoch) an entry id belongs to, surviving store resets."""

    def __init__(self, store: Store):
        self.db = store.db
        row = self.db.execute("SELECT MAX(hex_epoch_ms) FROM clips").fetchone()
        self.current: int | None = row[0]

    def assign(self, hex_id: int, timestamp_ms: int) -> int:
        """Reuse of an id is the only evidence of a reset; a clock step back is not."""
        row = self.db.execute(
            "SELECT MAX(hex_epoch_ms) FROM clips WHERE hex_id = ? AND timestamp_ms = ?",
            (hex_id, timestamp_ms),
        ).fetchone()
        if row[0] is not None:
            return row[0]
        if self.current is None:
            self.current = timestamp_ms
            return self.current
        taken = self.db.execute(
            "SELECT 1 FROM clips WHERE hex_epoch_ms = ? AND hex_id = ?", (self.current, hex_id)
        ).fetchone()
        if taken is not None:
            # The id holds a different entry in this store: Hex's store was reset.
            self.current = timestamp_ms
        return self.current


def since_to_ms(since: str | None, now: int) -> int | None:
    """The lower bound a `--since` value means, in ms, or None when unbounded."""
    if since is None:
        return None
    match = re.fullmatch(r"(\d+)([smhdw])", since)
    if match:
        unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[match.group(2)]
        return max(now - int(match.group(1)) * unit * 1000, 0)
    try:
        moment = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def collect(
    store: Store, binary: Path, since: str | None = None, copy_audio: bool = True
) -> Summary:
    summary = Summary()
    started = now_ms()
    if since is None:
        newest = store.db.execute("SELECT MAX(timestamp_ms) FROM clips").fetchone()[0]
        # Bound by this run's clock too: an entry stamped in the future must not hide later ones.
        since = rfc3339(max(min(newest, started) - RESCAN_MS, 0)) if newest is not None else None
    run = store.db.execute(
        "INSERT INTO runs (started_at_ms, since_ms) VALUES (?, ?)", (started, since_to_ms(since, started))
    ).lastrowid
    store.db.commit()
    error: str | None = None
    try:
        _collect_pages(store, binary, since, summary, copy_audio)
    except (CollectError, StoreError, OSError) as failure:
        error = str(failure)
        raise
    except BaseException as failure:
        # Name only the type: a message could carry transcript text (G-4).
        error = type(failure).__name__
        raise
    finally:
        store.db.execute(
            "UPDATE runs SET seen = ?, new = ?, audio_new = ?, missing_audio = ?, error = ?"
            " WHERE run_id = ?",
            (summary.seen, summary.new, summary.audio_new, summary.missing_audio, error, run),
        )
        store.db.commit()
    return summary


def _collect_pages(
    store: Store, binary: Path, since: str | None, summary: Summary, copy_audio: bool = True
) -> None:
    epochs = Epochs(store)
    seen: set[tuple[int, int]] = set()
    while True:
        document = run_history(binary, since)
        entries = document.get("entries", [])
        fresh = [e for e in entries if (e["id"], e["timestamp_ms"]) not in seen]
        for start in range(0, len(fresh), BATCH):
            for entry in fresh[start : start + BATCH]:
                seen.add((entry["id"], entry["timestamp_ms"]))
                _store_entry(store, epochs, entry, summary, copy_audio)
            store.db.commit()
        if not fresh or not (document.get("truncated") or len(entries) >= PAGE_LIMIT):
            return
        since = rfc3339(entries[-1]["timestamp_ms"])


def _store_entry(
    store: Store, epochs: Epochs, entry: dict, summary: Summary, copy_audio: bool = True
) -> None:
    summary.seen += 1
    hex_id, timestamp = entry["id"], entry["timestamp_ms"]
    epoch = epochs.assign(hex_id, timestamp)
    fields = {
        "application": entry.get("application"),
        "mode": entry.get("mode"),
        "kind": entry["kind"],
        "model_id": entry.get("model_id"),
        "model_name": entry.get("model_name"),
        "rewritten": 1 if entry.get("rewritten") else 0,
        "audio_ms": entry.get("audio_ms"),
        "raw_text": entry["raw_text"],
        "final_text": entry["final_text"],
    }
    correction = entry.get("correction") or {}
    fields["correction_outcome"] = correction.get("outcome")
    fields["correction_reason"] = correction.get("reason")
    fields["edited_text"] = correction.get("edited_text")
    existing = store.db.execute(
        "SELECT * FROM clips WHERE hex_epoch_ms = ? AND hex_id = ?", (epoch, hex_id)
    ).fetchone()
    audio_sha = existing["audio_sha256"] if existing else None
    if audio_sha is None and entry.get("recording"):
        if copy_audio:
            audio_sha = _store_audio(store, entry["recording"], summary)
        else:
            summary.audio_skipped += 1
    if existing is None:
        summary.new += 1
        store.db.execute(
            "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, application, mode,"
            " kind, model_id, model_name, rewritten, audio_ms, raw_text, final_text, audio_sha256,"
            " collected_at_ms, correction_outcome, correction_reason, edited_text)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                clip_id(epoch, hex_id), hex_id, epoch, timestamp, fields["application"],
                fields["mode"], fields["kind"], fields["model_id"], fields["model_name"],
                fields["rewritten"], fields["audio_ms"], fields["raw_text"], fields["final_text"],
                audio_sha, now_ms(), fields["correction_outcome"], fields["correction_reason"],
                fields["edited_text"],
            ),
        )
        return
    changed = {k: v for k, v in fields.items() if existing[k] != v}
    if audio_sha != existing["audio_sha256"]:
        changed["audio_sha256"] = audio_sha
    if not changed:
        return
    if "raw_text" in changed and existing["label"] != "verified":
        changed["label"] = "unlabelled"  # M3 recomputes labels from the new raw text
    if any(name in changed for name in CLASSIFICATION_INPUTS):
        # Hex added or changed the correction: `label corrections` classifies it again.
        store.db.execute("DELETE FROM edits WHERE clip_id = ?", (existing["clip_id"],))
        changed["reference_text"] = None
        changed["classified_at_ms"] = None
        if (label := label_after(existing["label"], False)) != existing["label"]:
            changed["label"] = label
    assignments = ", ".join(f"{name} = ?" for name in changed)
    store.db.execute(
        f"UPDATE clips SET {assignments} WHERE hex_epoch_ms = ? AND hex_id = ?",
        (*changed.values(), epoch, hex_id),
    )


def _store_audio(store: Store, recording: str, summary: Summary) -> str | None:
    try:
        pcm = read_wav(Path(recording))
        if store.write_audio(pcm):
            summary.audio_new += 1
    except (StoreError, OSError):
        summary.missing_audio += 1
        return None
    return pcm.sha256


def main_collect(since: str | None, hex_path: Path | None, min_free: int | None = None) -> int:
    import sys

    from healing.clean import DEFAULT_MIN_FREE, free_bytes
    from healing.paths import healing_home

    try:
        if since is not None:
            check_since(since)
        try:
            binary = Path(hex_path or hex_binary(prefer_installed=True)).resolve()
        except BenchError as error:
            raise CollectError(str(error), 2) from error
        if not Path(binary).is_file():
            raise CollectError(f"no hex at {binary}", 2)
        with process_lock():
            store = Store()
            try:
                floor = DEFAULT_MIN_FREE if min_free is None else min_free
                summary = collect(store, Path(binary), since, free_bytes(healing_home()) >= floor)
            finally:
                store.close()
    except CollectError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(summary)
    return 0
