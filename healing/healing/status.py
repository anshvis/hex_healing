"""`healing status`: what the dataset holds and what is about to be lost.

Reads the dataset (`runs`, `clips`, `audio`) and Hex's two settings through `hex settings get`,
never `settings.json` (G-6). Exit 1 when any warning is present, so it can be scripted.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from healing.budget import DEFAULT_BUDGET
from healing.clean import DEFAULT_MIN_FREE, free_bytes, index_problems
from healing.store import Store, now_ms

DAY_MS = 24 * 3600 * 1000
RETENTION_MS = {"day": DAY_MS, "week": 7 * DAY_MS, "month": 30 * DAY_MS}
KEEP_KEY = "labs.keep_recordings"
RETENTION_KEY = "history_retention"

WARNINGS = {
    "keep_recordings_off": "Labs > Keep recordings is off: new audio is not kept and is lost for good",
    "gap_exceeds_retention": "the last successful collection is older than History's retention minus 1 day",
    "budget_over_90pct": "audio is over 90% of the byte budget",
    "strong_without_audio": "strong clips exist whose audio is missing or evicted",
    "history_retention_day": "History keeps only 24 hours",
    "recordings_hidden": "Keep recordings is off but earlier WAVs still exist; hex history hides them",
    "disk_below_floor": "free space on the dataset's volume is under the floor: no audio is being copied",
    "index_mismatch": "the audio index and the disk disagree; run healing clean",
    "settings_unreadable": "could not read Hex's settings (is Hex running?), so some warnings are unknown",
}


class SettingsUnavailable(Exception):
    pass


def hex_setting(binary: Path, key: str):
    """One setting's value through `hex settings get --json`."""
    env = {**os.environ, "HEX_DISABLE_UPDATES": "1"}
    try:
        done = subprocess.run(
            [str(binary), "settings", "get", key, "--json"],
            capture_output=True, env=env, check=False, timeout=15,
        )
        if done.returncode != 0:
            raise SettingsUnavailable(key)
        reply = json.loads(done.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SettingsUnavailable(key) from error
    for row in reply.get("settings", []):
        if row.get("key") == key:
            return row.get("value")
    raise SettingsUnavailable(key)


@dataclass
class Status:
    clips: int = 0
    with_audio: int = 0
    audio_bytes: int = 0
    budget: int = DEFAULT_BUDGET
    labels: dict[str, int] = field(default_factory=dict)
    clusters: int = 0
    last_run: dict | None = None
    oldest_uncollected_risk_ms: int | None = None
    free_bytes: int = 0
    min_free: int = DEFAULT_MIN_FREE
    last_clean: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "clips": self.clips,
            "with_audio": self.with_audio,
            "audio_bytes": self.audio_bytes,
            "budget": self.budget,
            "labels": self.labels,
            "clusters": self.clusters,
            "last_run": self.last_run,
            "oldest_uncollected_risk": self.oldest_uncollected_risk_ms,
            "free_bytes": self.free_bytes,
            "min_free": self.min_free,
            "last_clean": self.last_clean,
            "warnings": self.warnings,
        }

    def text(self) -> str:
        lines = [
            f"clips {self.clips}, with audio {self.with_audio}, clusters {self.clusters}",
            f"audio {self.audio_bytes} / {self.budget} bytes",
            "labels " + (", ".join(f"{k} {v}" for k, v in sorted(self.labels.items())) or "none"),
        ]
        run = self.last_run
        if run:
            lines.append(f"last run {run['started_at_ms']} new {run['new']} error {run['error'] or '-'}")
        else:
            lines.append("last run never")
        lines.append(f"free {self.free_bytes} / floor {self.min_free} bytes")
        if self.last_clean:
            c = self.last_clean
            lines.append(
                f"last clean {c['at_ms']} orphans {c['orphans_moved']} missing {c['missing_marked']}"
                f" mismatched {c['mismatched_marked']}"
            )
        else:
            lines.append("last clean never")
        if self.oldest_uncollected_risk_ms is not None:
            lines.append(f"oldest uncollected risk {self.oldest_uncollected_risk_ms}")
        lines += [f"warning {code}: {WARNINGS[code]}" for code in self.warnings]
        return "\n".join(lines)


def compute(
    store: Store, binary: Path, budget: int = DEFAULT_BUDGET, now: int | None = None,
    min_free: int = DEFAULT_MIN_FREE, free_fn=None,
) -> Status:
    now = now_ms() if now is None else now
    db = store.db
    status = Status(budget=budget, min_free=min_free)
    status.free_bytes = (free_fn or (lambda: free_bytes(store.root.parent)))()
    clean_run = db.execute("SELECT * FROM clean_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    if clean_run:
        status.last_clean = {
            key: clean_run[key]
            for key in ("at_ms", "orphans_moved", "missing_marked", "mismatched_marked",
                        "temp_removed", "freed_bytes", "free_bytes_after")
        }
    status.clips = db.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    status.with_audio = db.execute(
        "SELECT COUNT(*) FROM clips c JOIN audio a ON a.sha256 = c.audio_sha256"
        " WHERE a.evicted_at_ms IS NULL"
    ).fetchone()[0]
    kept = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM audio WHERE evicted_at_ms IS NULL"
    ).fetchone()
    status.audio_bytes = kept[1]
    status.labels = {
        row[0]: row[1] for row in db.execute("SELECT label, COUNT(*) FROM clips GROUP BY label")
    }
    status.clusters = db.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    run = db.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    if run:
        status.last_run = {
            "started_at_ms": run["started_at_ms"], "new": run["new"], "error": run["error"],
        }
    # A run killed mid-way leaves no error and counts as a success; the next run rescans 2 days.
    ok = db.execute("SELECT MAX(started_at_ms) FROM runs WHERE error IS NULL").fetchone()[0]

    warnings = status.warnings
    try:
        keep = hex_setting(binary, KEEP_KEY)
        retention = hex_setting(binary, RETENTION_KEY)
    except SettingsUnavailable:
        keep, retention = None, None
        warnings.append("settings_unreadable")
    if keep is False:
        warnings.append("keep_recordings_off")
        if kept[0] > 0:
            warnings.append("recordings_hidden")
    window = RETENTION_MS.get(retention) if isinstance(retention, str) else None
    if window is not None:
        if ok is not None:
            status.oldest_uncollected_risk_ms = ok + window
        if ok is None or now - ok > max(window - DAY_MS, 0):
            warnings.append("gap_exceeds_retention")
    if retention == "day":
        warnings.append("history_retention_day")
    if status.audio_bytes > 0.9 * budget:
        warnings.append("budget_over_90pct")
    missing = db.execute(
        "SELECT COUNT(*) FROM clips c LEFT JOIN audio a ON a.sha256 = c.audio_sha256"
        " WHERE c.label = 'strong' AND (a.sha256 IS NULL OR a.evicted_at_ms IS NOT NULL)"
    ).fetchone()[0]
    if missing > 0:
        warnings.append("strong_without_audio")
    if status.free_bytes < min_free:
        warnings.append("disk_below_floor")
    if any(index_problems(store).values()):
        warnings.append("index_mismatch")
    return status
