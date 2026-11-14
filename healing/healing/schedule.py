"""`healing schedule install|remove|show`: the user launchd job that runs collection.

The job runs `healing schedule run` (collect, then prune) every 6 hours and at load; launchd
runs a missed interval on wake. Absolute paths are stored at install because launchd's PATH has
neither `healing` nor `hex`. The log holds counts only: collect and prune print no text (G-4).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template
from xml.sax.saxutils import escape

from healing.eval.bench import BenchError, hex_binary
from healing.paths import healing_home

LABEL = "com.hex-healing.collect"
INTERVAL_S = 6 * 3600
TEMPLATE = Path(__file__).resolve().parents[1] / "launchd" / f"{LABEL}.plist.tmpl"


class ScheduleError(Exception):
    pass


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(directory: Path | None = None) -> Path:
    return (directory or agents_dir()) / f"{LABEL}.plist"


def log_path() -> Path:
    return healing_home() / "logs" / "collect.log"


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


def render(command: list[str], hex_path: Path, home: Path, log: Path) -> str:
    arguments = "\n".join(
        f"        <string>{escape(part)}</string>" for part in [*command, "schedule", "run"]
    )
    return Template(TEMPLATE.read_text()).substitute(
        label=LABEL,
        arguments=arguments,
        home=escape(str(home)),
        hex=escape(str(hex_path)),
        interval=INTERVAL_S,
        log=escape(str(log)),
    )


def install(hex_path: Path | None = None, directory: Path | None = None) -> Path:
    try:
        binary = Path(hex_path or hex_binary()).resolve()
    except BenchError as error:
        raise ScheduleError(str(error)) from error
    if not binary.is_file():
        raise ScheduleError(f"no hex at {binary}")
    log = log_path()
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(log.parent, 0o700)
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.close(descriptor)
    os.chmod(log, 0o600)
    target = plist_path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(healing_command(), binary, healing_home().resolve(), log))
    launchctl("bootout", f"{domain()}/{LABEL}")  # idempotent: reload whatever was loaded
    done = launchctl("bootstrap", domain(), str(target))
    if done.returncode != 0:
        raise ScheduleError(f"launchctl bootstrap failed ({done.returncode}): {done.stderr.strip()}")
    return target


def remove(directory: Path | None = None) -> None:
    launchctl("bootout", f"{domain()}/{LABEL}")  # not loaded is fine
    plist_path(directory).unlink(missing_ok=True)


def show(directory: Path | None = None) -> str:
    target = plist_path(directory)
    lines = [f"plist: {target} ({'present' if target.is_file() else 'absent'})"]
    state = launchctl("print", f"{domain()}/{LABEL}")
    if state.returncode == 0:
        for line in state.stdout.splitlines():
            if line.strip().startswith(("state =", "runs =", "last exit code =")):
                lines.append(line.strip())
        lines.insert(1, "loaded: yes")
    else:
        lines.insert(1, "loaded: no")
    return "\n".join(lines)


def run_once() -> int:
    """What the job runs: collect, then prune. Prune runs even when collect failed."""
    from healing.budget import PruneError, prune
    from healing.collect import main_collect
    from healing.store import Store, StoreError, process_lock

    code = main_collect(None, None)
    try:
        with process_lock():
            store = Store()
            try:
                print(prune(store).summary())
            finally:
                store.close()
    except PruneError as error:
        print(f"error: {error}", file=sys.stderr)
        return code or error.code
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return code or 1
    return code
