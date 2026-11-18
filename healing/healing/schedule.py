"""`healing schedule install|remove|show`: the user launchd jobs.

`collect` (the default) runs `healing schedule run` (collect, then prune) at hours 0, 6, 12 and 18
and at load. `train-auto` (T6.4, optional, never installed by default) runs `healing train auto`
once a day at 03:00, so a round starts within a day of crossing 200 minutes; not at load, so a
login never starts hours of training.
StartCalendarInterval is used because launchd runs a firing missed during sleep on wake, while
StartInterval skips it. Absolute paths are stored at install because launchd's PATH has
neither `healing` nor `hex`. The log holds counts only: collect and prune print no text (G-4).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from xml.sax.saxutils import escape

from healing.eval.bench import BenchError, history_hex
from healing.paths import healing_home

LABEL = "com.hex-healing.collect"
HOURS = (0, 6, 12, 18)
TEMPLATE = Path(__file__).resolve().parents[1] / "launchd" / f"{LABEL}.plist.tmpl"


@dataclass(frozen=True)
class Job:
    label: str
    arguments: tuple[str, ...]  # after the healing command
    hours: tuple[int, ...]
    log: str
    run_at_load: bool


COLLECT = Job(LABEL, ("schedule", "run"), HOURS, "collect.log", True)
TRAIN_AUTO = Job("com.hex-healing.train-auto", ("train", "auto"), (3,), "train-auto.log", False)
JOBS = {"collect": COLLECT, "train-auto": TRAIN_AUTO}


class ScheduleError(Exception):
    pass


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(directory: Path | None = None, job: Job = COLLECT) -> Path:
    return (directory or agents_dir()) / f"{job.label}.plist"


def log_path(job: Job = COLLECT) -> Path:
    return healing_home() / "logs" / job.log


def domain() -> str:
    return f"gui/{os.getuid()}"


def launchctl(*args: str) -> subprocess.CompletedProcess:
    """The one place launchctl runs; unit tests replace it."""
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def healing_command() -> list[str]:
    found = shutil.which("healing")
    if found:
        return [str(Path(found).resolve())]
    return [sys.executable, "-m", "healing.cli"]


def render(command: list[str], hex_path: Path, home: Path, log: Path, job: Job = COLLECT) -> str:
    arguments = "\n".join(
        f"        <string>{escape(part)}</string>" for part in [*command, *job.arguments]
    )
    return Template(TEMPLATE.read_text()).substitute(
        label=job.label,
        run_at_load="<true/>" if job.run_at_load else "<false/>",
        arguments=arguments,
        home=escape(str(home)),
        hex=escape(str(hex_path)),
        calendar="\n".join(
            f"        <dict><key>Hour</key><integer>{hour}</integer>"
            "<key>Minute</key><integer>0</integer></dict>"
            for hour in job.hours
        ),
        log=escape(str(log)),
    )


def install(hex_path: Path | None = None, directory: Path | None = None, job: Job = COLLECT) -> Path:
    try:
        binary = Path(hex_path or history_hex()).resolve()
    except BenchError as error:
        raise ScheduleError(str(error)) from error
    if not binary.is_file():
        raise ScheduleError(f"no hex at {binary}")
    log = log_path(job)
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(log.parent, 0o700)
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.close(descriptor)
    os.chmod(log, 0o600)
    target = plist_path(directory, job)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(healing_command(), binary, healing_home().resolve(), log, job))
    launchctl("bootout", f"{domain()}/{job.label}")  # idempotent: reload whatever was loaded
    done = launchctl("bootstrap", domain(), str(target))
    if done.returncode != 0:
        raise ScheduleError(f"launchctl bootstrap failed ({done.returncode}): {done.stderr.strip()}")
    return target


def remove(directory: Path | None = None, job: Job = COLLECT) -> None:
    launchctl("bootout", f"{domain()}/{job.label}")  # not loaded is fine
    plist_path(directory, job).unlink(missing_ok=True)


def show(directory: Path | None = None, job: Job = COLLECT) -> str:
    target = plist_path(directory, job)
    lines = [f"plist: {target} ({'present' if target.is_file() else 'absent'})"]
    state = launchctl("print", f"{domain()}/{job.label}")
    if state.returncode == 0:
        for line in state.stdout.splitlines():
            if line.strip().startswith(("state =", "runs =", "last exit code =")):
                lines.append(line.strip())
        lines.insert(1, "loaded: yes")
    else:
        lines.insert(1, "loaded: no")
    return "\n".join(lines)


def run_once() -> int:
    """What the job runs: collect, prune, clean. Each runs even when the one before failed."""
    from healing.budget import PruneError, prune
    from healing.collect import main_collect
    from healing.store import Store, StoreError, process_lock

    try:
        code = main_collect(None, None)
    except Exception as error:  # type name only: a message could carry transcript text (G-4)
        print(f"error: collect failed ({type(error).__name__})", file=sys.stderr)
        code = 1
    try:
        with process_lock():
            store = Store()
            try:
                print(prune(store).summary())
            finally:
                store.close()
    except PruneError as error:
        print(f"error: {error}", file=sys.stderr)
        code = code or error.code
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        code = code or 1
    cleaned = _clean_after()
    return code or cleaned


def _clean_after() -> int:
    """`healing clean` after prune, under its own lock; a failure is named by type only (G-4)."""
    from healing.clean import CleanError, DEFAULT_MIN_FREE, clean
    from healing.store import Store, StoreError, process_lock

    try:
        with process_lock():
            store = Store(sweep=False)
            try:
                print(clean(store, DEFAULT_MIN_FREE).summary())
            finally:
                store.close()
    except CleanError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"error: clean failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0
