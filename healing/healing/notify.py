"""A macOS notification from the scheduled jobs, so a round that needs you says so without a terminal.

The text names versions, counts and commands only, never transcript text (G-4). `osascript` runs in
the user's GUI session, which the launchd jobs share; anywhere else (no `osascript`, a test with
`HEALING_NO_NOTIFY=1`) it does nothing. A failed notification never fails the job.
"""

from __future__ import annotations

import os
import shutil
import subprocess

TITLE = "Hex healing"


def _quoted(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(message: str, title: str = TITLE) -> None:
    if os.environ.get("HEALING_NO_NOTIFY") or shutil.which("osascript") is None:
        return
    script = f"display notification {_quoted(message)} with title {_quoted(title)}"
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False,
                       stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        pass
