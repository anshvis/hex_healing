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
- **An undecided candidate** (not promoted, rejected or rolled back) blocks the next round, gated or not: no second
  round is stacked on top of it (D-53). The wait names its gate state: never gated, refused for thin
  evidence (it stays a candidate, D-54), passed and awaiting promotion, or passed but its promote
  failed (`promote_failed`, T7.2; it stays a candidate).
- **One round at a time:** `$HEALING_HOME/train-auto.lock`; a second `train auto` exits at once.

The round (D-51, D-52, D-86): `prepare` (T6.1), then the runner. `local` (the default) trains on this
Mac (T6.5); `kaggle` trains on Kaggle's free GPU (T6.2), only with `allow_upload: true` (G-3: without
it the round stops after prepare and nothing is uploaded), and trains on this Mac instead when
Kaggle's weekly GPU quota is used up. Then it exports and registers a candidate (T6.3) and scores it
with `healing eval` (M3), recording the scores on the version. A local round holds its `round.lock`
from training through export, eval and cleanup, and its `work/` scratch (about 10 GB) is removed under
it when the round ends, pass or fail; `out/` and what the registry recorded stay. A round another
`healing train --local` holds fails at train and keeps its scratch. The round's parent is the current
version's name, resolved once at the start.

**Hands off** (`auto_promote: true`, D-87): every run first decides the undecided candidate. It
rebuilds the candidate's parent's rolling set from your reviews; under 1,500 reviewed words it waits
and says how many (a notification too), since only your verified text can show a regression (D-22).
Otherwise it runs `healing gate` and, on a pass, `healing promote`; a rejected candidate frees the
next round. A new round's candidate is decided the same way as soon as it is scored. Without it a
round stops at the scored candidate and prints `healing gate v<n>` (T7.1).

`$HEALING_HOME/train-auto.yaml` (optional, the `lora.yaml` subset): `runner: local | kaggle`,
`allow_upload: true | false`, `auto_promote: true | false` and `max_minutes` (Kaggle's time limit)
(defaults `local`, `false`, `false`, 120).
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
from healing.train.manifest import TrainError

MINUTES = 200  # D-33
SAMPLE_RATE = 16_000
RUNNERS = ("local", "kaggle")
SETTINGS_KEYS = {"runner": str, "allow_upload": bool, "auto_promote": bool, "max_minutes": int}
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
    auto_promote: bool = False
    max_minutes: int = 120


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
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise config_module.ConfigError(f"{path}: {key} must be a {kind.__name__}")
    found = Settings(**values)
    if found.runner not in RUNNERS:
        raise config_module.ConfigError(f"{path}: runner must be one of {', '.join(RUNNERS)}")
    if found.max_minutes < 1:
        raise config_module.ConfigError(f"{path}: max_minutes must be at least 1")
    return found


# ---- what the trigger waits on -------------------------------------------------------------------

@dataclass(frozen=True)
class Progress:
    parent: str | None  # the current version, None before `register-base`
    clips: int  # new trainable clips
    minutes: float
    pending: str | None  # an undecided candidate
    next_version: str
    pending_gate: dict | None = None  # its latest `healing gate` decision
    pending_failure: dict | None = None  # its latest `promote_failed` since that gate (T7.2)
    pending_parent: str | None = None  # the version it was trained from, whose sets gate it

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
            gate = self.pending_gate or {}
            if gate.get("pass") and self.pending_failure:
                why = (f"passed the gate, but its promote failed ({self.pending_failure.get('reason')}); "
                       f"`healing promote {self.pending}` again")
            elif gate.get("pass"):
                why = f"passed the gate; promote it (`healing promote {self.pending}`, T7.2)"
            elif gate.get("refused"):
                why = (f"the gate refused it for thin evidence; review more clips, then `healing gate {self.pending}` "
                       "again")
            else:
                why = f"awaits the gate (`healing gate {self.pending}`)"
            return f"waiting: candidate {self.pending} {why}; {self.amount()}"
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
    pending = next((v for v in state.ordered() if v.number > 0 and v.state == "candidate"), None)
    failure = None
    if pending and pending.promote_failures and pending.last_gate:
        latest = pending.promote_failures[-1]
        failure = latest if latest.get("at_ms", 0) >= pending.last_gate.get("at_ms", 0) else None
    return Progress(parent.name if parent else None, len(new), minutes, pending.name if pending else None,
                    state.next_name(), pending.last_gate if pending else None, failure,
                    (pending.record.get("parent") or (parent.name if parent else None)) if pending else None)


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


def _lock(folder: Path):
    from healing.train import run as train_run

    return train_run.round_lock(folder)


def _train(round_) -> int:
    """Train with the round's lock already held (`_round` holds it)."""
    from healing.train import run as train_run

    return train_run.train_held(round_, say=_stderr)


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


def _remote_train(bundle: Path, max_minutes: int, say) -> tuple[int, Path]:
    """The Kaggle round (T6.2): uploads the bundle to a private dataset, deleted after. Only called with
    `allow_upload: true` (G-3)."""
    from healing.remote import kaggle_runner

    code = kaggle_runner.run_remote(bundle, allow_upload=True, max_minutes=max_minutes, say=say, say_error=say)
    return code, kaggle_runner.round_folder(bundle)


def _build(parent: str) -> int:
    """Rebuild `parent`'s rolling and edited sets from the reviews so far; the rolling set's reference words."""
    from healing import splits
    from healing.store import process_lock

    with process_lock():
        store = Store()
        try:
            return splits.build(store, parent).rolling.n_words
        finally:
            store.close()


def _gate(version: str, say) -> int:
    from healing import gate

    try:
        return gate.run(version, say=say)
    except gate.GateError as error:
        say(f"the gate could not judge {version}: {error}")
        return error.code


def _promote(version: str, say) -> int:
    from healing import ship
    from healing.eval.bench import history_hex

    return ship.promote(version, binary=Path(history_hex()), say=say)


def _notify(message: str) -> None:
    from healing.notify import notify

    notify(message)


@dataclass
class Steps:
    """The heavy steps, as functions so the tests can stand in for them."""

    prepare: Callable = _prepare  # (parent) -> Prepared (.path, .train_split)
    open_round: Callable = _open_round  # (bundle) -> Round (.folder, .work, .out)
    lock: Callable = _lock  # (round folder) -> the round's lock, held from train to the scratch's removal
    train: Callable = _train  # (round) -> exit code
    export: Callable = _export  # (delta, config, snapshot, parent) -> {version, gguf_sha256, ...}
    evaluate: Callable = _evaluate  # (version, gguf_sha256, parent) -> a summary line
    remote: Callable = _remote_train  # (bundle, max_minutes, say) -> (exit code, the round's folder)
    build: Callable = _build  # (parent) -> reference words in its rebuilt rolling set
    gate: Callable = _gate  # (version, say) -> exit code (0 passed)
    promote: Callable = _promote  # (version, say) -> exit code (0 confirmed in Hex)
    notify: Callable = _notify  # (message): a macOS notification, counts and versions only (G-4)


# ---- the trigger ---------------------------------------------------------------------------------

def _plan(settings: Settings) -> str:
    then = ", then gate and promote" if settings.auto_promote else ""
    if settings.runner == "local":
        return "prepare, train (local), export, eval" + then
    if not settings.allow_upload:
        return f"prepare, then stop: the {settings.runner} runner needs `allow_upload: true` (G-3)"
    return f"prepare, train ({settings.runner}, uploads the bundle; local if its quota is used up), export, eval" + then


def _finish(steps: Steps, out: Path, prepared, parent: str, settings: Settings, say) -> int:
    """Export and score the round's delta in `out/`."""
    from healing.train.run import DELTA, EFFECTIVE

    step, version = "export", None
    try:
        result = steps.export(out / DELTA, out / EFFECTIVE, prepared.train_split, parent)
        version = result["version"]
        say(f"{version}: candidate registered (gguf {result['gguf_sha256'][:12]})")
        step = "eval"
        say(f"{version}: {steps.evaluate(version, result['gguf_sha256'], parent)}")
    except Exception as error:
        say(f"round failed at {step}: {error}")
        if step == "eval":
            say(f"{version} stays a candidate without scores; score it with `healing eval --model <its gguf>`")
        return 1
    if not settings.auto_promote:
        say(f"next: healing gate {version}")
    return 0


def _kaggle(steps: Steps, settings: Settings, prepared, parent: str, say) -> int | None:
    """The Kaggle round; None when its quota is used up and the round should train here instead."""
    from healing.remote.kaggle_runner import QUOTA

    try:
        code, folder = steps.remote(prepared.path, settings.max_minutes, say)
    except Exception as error:
        say(f"round failed at train (kaggle): {error}; nothing registered, the next run tries again")
        return 1
    if code == QUOTA:
        say("Kaggle's weekly GPU quota is used up: training this round on this Mac instead")
        return None
    if code != 0:
        say(f"round failed at train (kaggle, exit {code}); nothing registered, the next run tries again")
        return code
    try:
        return _finish(steps, folder / "out", prepared, parent, settings, say)
    finally:
        shutil.rmtree(folder / "work", ignore_errors=True)  # `out/` and what the registry recorded stay


def _round(steps: Steps, settings: Settings, parent: str, say) -> int:
    """One round from `parent`, the current version's name resolved once at the start: the sets eval
    reads are found by their stored parent (`--parent v0`), which never says `current`."""
    prepared = steps.prepare(parent)
    say(f"prepared {prepared.train_split}: {prepared.path}")
    if settings.runner != "local":
        if not settings.allow_upload:
            say(f"stopped after prepare: the {settings.runner} runner uploads the bundle, which needs "
                f"`allow_upload: true` in {settings_path()} (G-3); nothing uploaded. Bundle: {prepared.path}")
            return 0
        code = _kaggle(steps, settings, prepared, parent, say)
        if code is not None:
            return code
    round_ = steps.open_round(prepared.path)
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(steps.lock(round_.folder))
        except TrainError as error:  # a manual `healing train --local` holds it: its work/ is not ours
            say(f"round failed at train: {error}")
            return 1
        try:
            code = steps.train(round_)
            if code != 0:
                say(f"round failed at train (exit {code}); nothing registered, the next run tries again")
                return code
            return _finish(steps, round_.out, prepared, parent, settings, say)
        finally:
            shutil.rmtree(round_.work, ignore_errors=True)  # ~10 GB of scratch, under the lock; `out/` stays


# ---- hands off: gate and promote (auto_promote) ---------------------------------------------------

def _tell(steps: Steps, say, message: str) -> None:
    say(message)
    steps.notify(message)


def _decide(steps: Steps, found: Progress, say) -> bool:
    """Gate, then promote, the undecided candidate. True once it is decided (promoted or rejected), so
    a round may start; False while it still waits (on your reviews, or a promote to retry)."""
    from healing.registry import store as registry
    from healing.splits import ROLLING_MIN_WORDS

    name = found.pending
    if not (found.pending_gate or {}).get("pass"):
        words = steps.build(found.pending_parent)
        if words < ROLLING_MIN_WORDS:
            _tell(steps, say, f"{name} is trained and waits on your review: {words}/{ROLLING_MIN_WORDS} reviewed "
                              "words. Run `healing review --strategy random --since-cutoff`.")
            return False
        if steps.gate(name, say) != 0:
            after = registry.load().get(name)
            if after.state == "rejected":
                _tell(steps, say, f"{name} failed the gate and was rejected; Hex keeps its model, and the next "
                                  "round trains a new candidate.")
                return True
            if (after.last_gate or {}).get("refused"):
                _tell(steps, say, f"{name} needs more reviewed clips before the gate can judge it. Run "
                                  "`healing review --strategy random --since-cutoff`.")
            else:
                _tell(steps, say, f"the gate could not judge {name}; see {_log_hint()}")
            return False
    code = steps.promote(name, say)
    if code == 0:
        _tell(steps, say, f"{name} is now Hex's speech model. `healing watch` keeps checking it and rolls it back "
                          "if it does worse.")
        return True
    _tell(steps, say, f"{name} passed the gate, but switching Hex to it failed (exit {code}); the next run tries "
                      "again.")
    return False


def _log_hint() -> str:
    return str(healing_home() / "logs" / "train-auto.log")


def run(steps: Steps | None = None, *, dry_run: bool = False, now: int | None = None, say=print) -> int:
    steps = steps or Steps()
    settings = load_settings()
    if dry_run:
        found = _progress(now)
        if running():
            say("a round is already running; it would exit")
        if found.pending and settings.auto_promote:
            say(f"would decide {found.pending} first: rebuild its rolling set, gate it once it holds "
                "1,500 reviewed words, and promote it on a pass")
        if not found.ready:
            say(found.waiting())
            return 0
        say(f"would start round {found.next_version}: {found.amount()}")
        say(f"would run: {_plan(settings)}" + (f", then stop for `healing gate {found.next_version}`"
                                               if settings.runner == "local" and not settings.auto_promote else ""))
        return 0
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(locked(lock_path(), wait=False))
        except StoreError:
            say(f"skipped: a round is already running ({lock_path()})")
            return 0
        found = _progress(now)
        if found.pending and settings.auto_promote:
            try:
                if not _decide(steps, found, say):
                    return 0
            except Exception as error:  # the message to the log; the notification names the type only (G-4)
                say(f"deciding {found.pending} failed: {error}")
                steps.notify(f"deciding {found.pending} failed ({type(error).__name__}); see {_log_hint()}")
                return 1
            found = _progress(now)
        if not found.ready:
            say(found.waiting())
            return 0
        say(f"starting round {found.next_version}: {found.amount()}")
        try:
            code = _round(steps, settings, found.parent, say)
        except Exception as error:  # prepare, or a round folder that could not be opened
            say(f"round failed before training: {error}")
            code = 1
        if code != 0:
            steps.notify(f"training round {found.next_version} failed; see {_log_hint()}")
            return code
        if settings.auto_promote:
            found = _progress(now)
            if found.pending:
                try:
                    _decide(steps, found, say)
                except Exception as error:
                    say(f"deciding {found.pending} failed: {error}")
                    steps.notify(f"deciding {found.pending} failed ({type(error).__name__}); see {_log_hint()}")
                    return 1
        return 0


def main(args) -> int:
    from healing.registry.snapshot import SnapshotError
    from healing.registry.store import RegistryError
    from healing.train.config import ConfigError

    try:
        return run(dry_run=args.dry_run)
    except (ConfigError, RegistryError, SnapshotError, StoreError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
