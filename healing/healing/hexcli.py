"""Calling the one running Hex through its `hex` control commands (docs/cli.md) and reading `--json` replies.

Shared by `healing vocab apply` and `healing models`. Control commands never launch Hex: exit 3 means
it isn't running.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

EXIT_NOT_RUNNING = 3
DEFAULT_TIMEOUT_S = 60


class HexError(RuntimeError):
    pass


class HexNotRunning(HexError):
    pass


class HexOutcomeUnknown(HexError):
    """`hex` didn't answer in time: a write may or may not have happened."""


def reply(text: str) -> dict | None:
    """The `--json` reply in `text`, or None when there is none. Hex pretty-prints it over many lines,
    so the whole output is tried first, then the last JSON object starting at a line's beginning."""
    text = text.strip()
    try:
        found = json.loads(text)
    except ValueError:
        found = None
    if isinstance(found, dict):
        return found
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{" and (i == 0 or text[i - 1] == "\n")]
    for start in reversed(starts):
        try:
            found, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return None


def run(binary: Path, args: list[str], *, write: bool = False, timeout: float = DEFAULT_TIMEOUT_S,
        refusal_codes: frozenset[str] = frozenset()) -> subprocess.CompletedProcess:
    """Run `hex`. For a `write`, a failure is `HexError` only when Hex confirmably refused (a code in
    `refusal_codes`, or not running) or never started; any other failure is `HexOutcomeUnknown`."""
    # Not the isolated environment the benchmark uses: this talks to the one running Hex, found through
    # its own data folder's discovery file.
    env = {**os.environ, "HEX_DISABLE_UPDATES": "1"}
    name = " ".join(args[:2])
    try:
        done = subprocess.run([str(binary), *args], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise HexOutcomeUnknown(f"hex {name} did not answer in {timeout}s; its outcome is unknown") from error
    except OSError as error:  # it never started
        raise HexError(f"could not run {binary}: {error}") from error
    if done.returncode == 0:
        return done
    if done.returncode == EXIT_NOT_RUNNING:
        raise HexNotRunning("Hex isn't running")
    answer = reply(done.stdout) or reply(done.stderr) or {}
    code = answer.get("code") if answer.get("status") == "error" else None
    detail = (answer.get("message") if code else None) or (done.stderr.strip() or done.stdout.strip())[-300:]
    message = f"hex {name} exited {done.returncode}" + (f" ({code})" if code else "") + f": {detail}"
    if not write or code in refusal_codes:
        if code == "not-running":
            raise HexNotRunning("Hex isn't running")
        raise HexError(message)
    raise HexOutcomeUnknown(message + "; its outcome is unknown")
