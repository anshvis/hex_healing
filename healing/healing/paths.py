"""Where healing keeps its data. Audio and datasets never live in the repo."""

import os
from pathlib import Path


def healing_home() -> Path:
    """`$HEALING_HOME`, else `~/Library/Application Support/hex-healing`."""
    named = os.environ.get("HEALING_HOME")
    if named:
        return Path(named).expanduser()
    return Path.home() / "Library" / "Application Support" / "hex-healing"


def repo_root() -> Path:
    """The Hex checkout this package lives in (`healing/` is one level down)."""
    return Path(__file__).resolve().parents[2]


def hex_support_dir() -> Path:
    """Hex's own data folder (`$HEX_APPLICATION_SUPPORT_DIR`, else Application Support). Read-only here."""
    named = os.environ.get("HEX_APPLICATION_SUPPORT_DIR")
    return Path(named) if named else Path.home() / "Library" / "Application Support" / "hex"
