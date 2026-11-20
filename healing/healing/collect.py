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

from healing import labels
from healing.corrections import forget
from healing.eval.bench import BenchError, history_hex
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
    "tidied_text",
    "tidy_applied",
)

# A change to any of these makes a stored classification stale.
CLASSIFICATION_INPUTS = ("raw_text", "final_text", "edited_text", "correction_outcome", "tidied_text")


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
        # Name only the type: a message could carry transcript text (G-4). A labelling failure is
        # one of these: it is what ended the run, and is raised as such.
        error = type(failure).__name__
        raise
    finally:
        store.db.rollback()  # a batch cut short by a failure is dropped whole; the next run stores it again
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
            touched = []
            for entry in fresh[start : start + BATCH]:
                seen.add((entry["id"], entry["timestamp_ms"]))
                if clip := _store_entry(store, epochs, entry, summary, copy_audio):
                    touched.append(clip)
            store.db.commit()
            labels.refresh(store, touched)  # labels follow the batch just committed, once, never per row
        if not fresh or not (document.get("truncated") or len(entries) >= PAGE_LIMIT):
            return
        since = rfc3339(entries[-1]["timestamp_ms"])


def tidied(final_text: str, decisions: list[dict]) -> tuple[int, str | None]:
    """(tidy_applied, tidied_text) for an entry's Tidy after paste decisions (T8.5, D-38).

    The tidied text is `final_text` with the decisions Hex edited into the field applied: the text the
    person's later corrections are measured from. `raw_text` and `final_text` stay what Hex heard and
    pasted. An applied tidy whose text can't be rebuilt (a replacement too long to keep, offsets that
    don't fit) is `(1, None)`: its correction has no known baseline and never labels. A tidy applied
    after the watch had ended (`after_watch`) is `(0, None)`: the correction was of `final_text`.
    """
    if any(d.get("applied") and d.get("after_watch") for d in decisions):
        return 0, None
    applied = sorted((d for d in decisions if d.get("applied")), key=lambda d: d["start"])
    if not applied:
        return 0, None
    chars = list(final_text)
    out: list[str] = []
    at = 0
    for decision in applied:
        start, end, replacement = decision["start"], decision["end"], decision.get("replacement")
        if replacement is None or not at <= start <= end <= len(chars):
            return 1, None
        out += chars[at:start]
        out.append(replacement)
        at = end
    out += chars[at:]
    return 1, "".join(out)


def _applied(decision: dict) -> int:
    """1 when the decision went into the field while the watch could see it; a fix applied after the
    watch ended has no evidence it survived, so it never counts toward a label."""
    return 1 if decision.get("applied") and not decision.get("after_watch") else 0


def _store_tidy_events(store: Store, clip: str, decisions: list[dict]) -> None:
    """Replace a clip's `tidy_events` with its entry's decisions: tokens only, as Hex bounded them."""
    store.db.execute("DELETE FROM tidy_events WHERE clip_id = ?", (clip,))
    for index, decision in enumerate(decisions):
        store.db.execute(
            "INSERT INTO tidy_events (clip_id, idx, kind, token, replacement, accepted, applied, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                clip, index, decision["kind"], decision["token"], decision.get("replacement"),
                1 if decision.get("accepted") else 0, _applied(decision), decision["source"],
            ),
        )


def _store_entry(
    store: Store, epochs: Epochs, entry: dict, summary: Summary, copy_audio: bool = True,
) -> str | None:
    """Store one History entry; the id of the clip it inserted or changed, else None."""
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
    # An entry without `correction` says nothing: an older Hex drops the field when it saves.
    correction = entry.get("correction")
    observed = {
        "correction_outcome": (correction or {}).get("outcome"),
        "correction_reason": (correction or {}).get("reason"),
        "edited_text": (correction or {}).get("edited_text"),
    }
    fields.update(observed)
    # Like `correction`, an entry without `tidy` says nothing: what was stored stays.
    decisions = entry.get("tidy")
    if decisions is not None:
        fields["tidy_applied"], fields["tidied_text"] = tidied(entry["final_text"], decisions)
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
        if decisions:
            store.db.execute(
                "UPDATE clips SET tidy_applied = ?, tidied_text = ? WHERE clip_id = ?",
                (fields["tidy_applied"], fields["tidied_text"], clip_id(epoch, hex_id)),
            )
            _store_tidy_events(store, clip_id(epoch, hex_id), decisions)
        return clip_id(epoch, hex_id)
    if decisions is not None and _tidy_rows(store, existing["clip_id"]) != _tidy_tuples(decisions):
        _store_tidy_events(store, existing["clip_id"], decisions)
        touched = True
    else:
        touched = False
    changed = {k: v for k, v in fields.items() if existing[k] != v}
    changed = _correction_changes(existing, correction, observed, changed)
    if audio_sha != existing["audio_sha256"]:
        changed["audio_sha256"] = audio_sha
    if not changed:
        return existing["clip_id"] if touched else None
    if any(name in changed for name in CLASSIFICATION_INPUTS):
        # Hex added or changed the correction: `label corrections` classifies it again.
        forget(store.db, existing["clip_id"], changed)
    assignments = ", ".join(f"{name} = ?" for name in changed)
    store.db.execute(
        f"UPDATE clips SET {assignments} WHERE hex_epoch_ms = ? AND hex_id = ?",
        (*changed.values(), epoch, hex_id),
    )
    return existing["clip_id"]


def _tidy_tuples(decisions: list[dict]) -> list[tuple]:
    return [
        (d["kind"], d["token"], d.get("replacement"), 1 if d.get("accepted") else 0, _applied(d), d["source"])
        for d in decisions
    ]


def _tidy_rows(store: Store, clip: str) -> list[tuple]:
    return [
        tuple(row)
        for row in store.db.execute(
            "SELECT kind, token, replacement, accepted, applied, source FROM tidy_events WHERE clip_id = ?"
            " ORDER BY idx",
            (clip,),
        )
    ]


def _correction_changes(existing, correction: dict | None, observed: dict, changed: dict) -> dict:
    """`changed`, adjusted so that only a correction Hex actually reports touches the stored one.

    An entry without `correction` says nothing (an older Hex drops the field when it saves), so
    the stored correction stays, unless the entry's text changed too (a Replace in Hex): that
    correction was seen against the old text, so it goes rather than label the new one.
    """
    if correction is not None:
        return changed
    kept = {name: value for name, value in changed.items() if name not in observed}
    if "raw_text" in kept or "final_text" in kept:
        kept.update({name: None for name in observed if existing[name] is not None})
    return kept


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
            binary = Path(hex_path or history_hex()).resolve()
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
