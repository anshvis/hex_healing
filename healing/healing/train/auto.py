"""`healing train auto [--dry-run]`: start a training round once there is enough new speech (T6.4).

A round starts once the strong (or verified) trainable clips that are not in the current version's
dataset snapshot hold at least 200 minutes of audio (D-33). There is no time floor: a round may start
however recently the last one ran. Otherwise it prints what it is waiting on.

- **Minutes** are the clips' stored audio sample counts at 16 kHz, never clip or word counts, and
  never Hex's own `audio_ms`.
- **New** is by clip id: trainable now (`splits.preview_train`, the clips `prepare` would hand out)
  and not a line of the parent's snapshot. v0 has no snapshot, so every trainable clip is new.
- **A failed round** registers no version, so the parent and its snapshot stay and the same minutes
  count at the next run (the train row its prepare wrote is not a version).
- **An undecided candidate** waits for M7's gate: no second round is stacked on top of it (D-53).
- **One round at a time:** `$HEALING_HOME/train-auto.lock`; a second `train auto` exits at once.

The round (D-51, D-52): `prepare` (T6.1), then the runner. `local` (the default) trains on this Mac
(T6.5), exports and registers a candidate (T6.3) and scores it with `healing eval` (M3), recording
the scores on the version; then it stops and prints `healing gate v<n>`, which M7 will add. Kaggle
(T6.2) is not built: a remote runner stops after prepare, before anything could be uploaded (G-3,
X-6). The round's `work/` scratch (about 10 GB) is removed when the round ends, pass or fail; `out/`
and what the registry recorded stay.

`$HEALING_HOME/train-auto.yaml` (optional, the `lora.yaml` subset) picks the runner:
`runner: local | kaggle` and `allow_upload: true | false` (default `local`, `false`).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from healing.export import ExportError
from healing.paths import healing_home
from healing.store import Store, StoreError, locked, lock_free

MINUTES = 200  # D-33
SAMPLE_RATE = 16_000
RUNNERS = ("local", "kaggle")
SETTINGS_KEYS = {"runner": str, "allow_upload": bool}
SETS = "rolling,golden,general,edited"

__all__ = ["ExportError", "Progress", "Settings", "Steps", "lock_path", "progress", "run", "settings_path"]


def lock_path() -> Path:
    return healing_home() / "train-auto.lock"


def settings_path() -> Path:
    return healing_home() / "train-auto.yaml"


@dataclass(frozen=True)
class Settings:
    runner: str = "local"
    allow_upload: bool = False


def load_settings(path: Path | None = None) -> Settings:
    from healing.train import config as config_module

    path = path or settings_path()
    if not path.is_file():
        return Settings()
    values = config_module.parse(path.read_text(), str(path))
    for key, value in values.items():
        kind = SETTINGS_KEYS.get(key)
        if kind is None:
            raise config_module.ConfigError(f"{path}: unknown key {key!r} (known: {', '.join(SETTINGS_KEYS)})")
        if not isinstance(value, kind):
            raise config_module.ConfigError(f"{path}: {key} must be a {kind.__name__}")
    found = Settings(**values)
    if found.runner not in RUNNERS:
        raise config_module.ConfigError(f"{path}: runner must be one of {', '.join(RUNNERS)}")
    return found


# ---- what the trigger waits on -------------------------------------------------------------------

@dataclass(frozen=True)
class Progress:
    parent: str | None  # the current version, None before `register-base`
    clips: int  # new trainable clips
    minutes: float
    pending: str | None  # an undecided candidate
    next_version: str

    @property
    def enough(self) -> bool:
        return self.minutes >= MINUTES

    @property
    def ready(self) -> bool:
        return self.enough and self.parent is not None and self.pending is None

    def amount(self) -> str:
        since = f"since {self.parent}'s snapshot" if self.parent else "with no version registered"
        return f"strong speech {int(self.minutes)}/{MINUTES} min ({self.clips} new clips {since})"

    def waiting(self) -> str:
        if self.pending:
            return (f"waiting: candidate {self.pending} awaits the gate (`healing gate {self.pending}`, M7); "
                    f"{self.amount()}")
        if self.parent is None and self.enough:
            return f"waiting: no v0 to train from (run `healing models register-base`); {self.amount()}"
        return f"waiting: {self.amount()}"


def parent_clip_ids(version) -> set[str]:
    """The clip ids of `version`'s dataset snapshot (none for v0, which trained on nothing)."""
    from healing.registry import snapshot as snapshots

    sha = (version.record.get("dataset") or {}).get("snapshot_sha256") if version else None
    return {str(row[0]) for row in snapshots.read(sha)} if sha else set()


def progress(store: Store, now: int | None = None) -> Progress:
    """Read-only: the registry, the parent's snapshot and the dataset, nothing written."""
    from healing import splits
    from healing.registry import store as registry

    state = registry.load()
    parent = state.versions.get(state.current) if state.current else None
    seen = parent_clip_ids(parent)
    new = [c for c in splits.preview_train(store, now=now) if c not in seen]
    samples = dict(store.db.execute(
        "SELECT c.clip_id, a.samples FROM clips c JOIN audio a ON a.sha256 = c.audio_sha256"
    ).fetchall())
    minutes = sum(samples.get(c) or 0 for c in new) / SAMPLE_RATE / 60
    pending = next((v.name for v in state.ordered() if v.number > 0 and v.state == "candidate"), None)
    return Progress(parent.name if parent else None, len(new), minutes, pending, state.next_name())


def _progress(now: int | None) -> Progress:
    store = Store(sweep=False)  # a read: never delete a running collect's temp files
    try:
        return progress(store, now)
    finally:
        store.close()


def running() -> bool:
    """Whether a round holds the lock, without creating the lock file (the dry run writes nothing)."""
    try:
        descriptor = os.open(lock_path(), os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    try:
        return not lock_free(descriptor)
    finally:
        os.close(descriptor)


# ---- the steps -----------------------------------------------------------------------------------

def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _prepare(parent: str):
    from healing.store import process_lock
    from healing.train import manifest as train_manifest

    with process_lock():
        store = Store()
        try:
            return train_manifest.prepare(store, parent=parent)
        finally:
            store.close()


def _open_round(bundle: Path):
    from healing.train import run as train_run

    return train_run.open_round(bundle)


def _train(round_) -> int:
    from healing.train import run as train_run

    return train_run.train_round(round_, say=_stderr)


def _export(delta: Path, config: Path, snapshot: str, parent: str) -> dict:
    from healing import export as export_module

    return export_module.export(delta, config, snapshot, parent)


def _evaluate(version: str, gguf_sha: str, parent: str) -> str:
    """`healing eval` of the candidate on every set, its scores recorded on the version."""
    from healing.eval import run as eval_run
    from healing.registry import commands, objects
    from healing.registry import store as registry

    store = Store(sweep=False)
    try:
        document = eval_run.evaluate([objects.path(gguf_sha)], SETS.split(","), store=store, parent=parent,
                                     progress=_stderr)
    finally:
        store.close()
    if not document["sets"]:
        raise eval_run.EvalError("none of the sets could be scored")
    path = eval_run.write(document)
    scores = commands.scores_from(document, gguf_sha, str(path))
    registry.record_scores(version, scores)
    return f"scored {', '.join(sorted(scores))} ({path})"


@dataclass
class Steps:
    """The heavy steps, as functions so the tests can stand in for them."""

    prepare: Callable = _prepare  # (parent) -> Prepared (.path, .train_split)
    open_round: Callable = _open_round  # (bundle) -> Round (.folder, .work, .out)
    train: Callable = _train  # (round) -> exit code
    export: Callable = _export  # (delta, config, snapshot, parent) -> {version, gguf_sha256, ...}
    evaluate: Callable = _evaluate  # (version, gguf_sha256, parent) -> a summary line


# ---- the trigger ---------------------------------------------------------------------------------

def _plan(settings: Settings) -> str:
    if settings.runner == "local":
        return "prepare, train (local), export, eval"
    return f"prepare, then stop: the {settings.runner} runner is parked (G-3, X-6)"


def _remote(settings: Settings, prepared, say) -> int:
    bundle = getattr(prepared, "path", None)
    if not settings.allow_upload:
        say(f"stopped after prepare: the {settings.runner} runner uploads the bundle, which needs "
            f"`allow_upload: true` in {settings_path()} (G-3); nothing uploaded. Bundle: {bundle}")
    else:
        say(f"parked: the {settings.runner} runner (T6.2) is not built (X-6); nothing uploaded. Bundle: {bundle}")
    return 0


def _round(steps: Steps, settings: Settings, parent: str, say) -> int:
    """One round from `parent`, the current version's name resolved once at the start: the sets eval
    reads are found by their stored parent (`--parent v0`), which never says `current`."""
    from healing.train.run import DELTA, EFFECTIVE

    step = "prepare"
    prepared = steps.prepare(parent)
    say(f"prepared {prepared.train_split}: {prepared.path}")
    if settings.runner != "local":
        return _remote(settings, prepared, say)
    step = "train"
    round_ = steps.open_round(prepared.path)
    try:
        code = steps.train(round_)
        if code != 0:
            say(f"round failed at train (exit {code}); nothing registered, the next run tries again")
            return code
        step = "export"
        result = steps.export(round_.out / DELTA, round_.out / EFFECTIVE, prepared.train_split, parent)
        version = result["version"]
        say(f"{version}: candidate registered (gguf {result['gguf_sha256'][:12]})")
        step = "eval"
        say(f"{version}: {steps.evaluate(version, result['gguf_sha256'], parent)}")
    except Exception as error:
        say(f"round failed at {step}: {error}")
        if step == "eval":
            say(f"{version} stays a candidate without scores; score it with `healing eval --model <its gguf>`")
        return 1
    finally:
        shutil.rmtree(round_.work, ignore_errors=True)  # ~10 GB of scratch; `out/` and the registry stay
    say(f"next (M7): healing gate {version}")
    return 0


def run(steps: Steps | None = None, *, dry_run: bool = False, now: int | None = None, say=print) -> int:
    steps = steps or Steps()
    settings = load_settings()
    if dry_run:
        found = _progress(now)
        if running():
            say("a round is already running; it would exit")
        if not found.ready:
            say(found.waiting())
            return 0
        say(f"would start round {found.next_version}: {found.amount()}")
        say(f"would run: {_plan(settings)}" + (f", then stop for `healing gate {found.next_version}` (M7)"
                                               if settings.runner == "local" else ""))
        return 0
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(locked(lock_path(), wait=False))
        except StoreError:
            say(f"skipped: a round is already running ({lock_path()})")
            return 0
        found = _progress(now)
        if not found.ready:
            say(found.waiting())
            return 0
        say(f"starting round {found.next_version}: {found.amount()}")
        try:
            return _round(steps, settings, found.parent, say)
        except Exception as error:  # prepare, or a round folder that could not be opened
            say(f"round failed before training: {error}")
            return 1


def main(args) -> int:
    from healing.registry.snapshot import SnapshotError
    from healing.registry.store import RegistryError
    from healing.train.config import ConfigError

    try:
        return run(dry_run=args.dry_run)
    except (ConfigError, RegistryError, SnapshotError, StoreError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
