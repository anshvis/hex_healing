"""Where healing keeps its data. Audio and datasets never live in the repo."""

import os
from pathlib import Path


def healing_home() -> Path:
    """`$HEALING_HOME`, else `~/Library/Application Support/hex-healing`."""
    named = os.environ.get("HEALING_HOME")
    if named:
        return Path(named).expanduser()
    return Path.home() / "Library" / "Application Support" / "hex-healing"


def sets_dir() -> Path:
    """`$HEALING_HOME/sets`: every test set's folder (the general set is `sets/general`)."""
    return healing_home() / "sets"


def repo_root() -> Path:
    """The Hex checkout this package lives in (`healing/` is one level down)."""
    return Path(__file__).resolve().parents[2]


def hex_support_dir() -> Path:
    """Hex's own data folder (`$HEX_APPLICATION_SUPPORT_DIR`, else Application Support). Read-only here."""
    named = os.environ.get("HEX_APPLICATION_SUPPORT_DIR")
    return Path(named) if named else Path.home() / "Library" / "Application Support" / "hex"


def train_dir() -> Path:
    """`$HEALING_HOME/train`: training bundles (`<round>/`) and the replay transcript cache."""
    return healing_home() / "train"


def home_unavailable() -> str | None:
    """Why `$HEALING_HOME` can't be used right now, or None. Only a data folder on a drive that isn't
    connected counts: a link whose target is gone, or a path under an unmounted `/Volumes/<name>`.
    A folder that doesn't exist yet on the Mac is fine (the first collect creates it)."""
    home = healing_home()
    if home.is_dir():
        return None
    if home.is_symlink():
        return f"the data folder {home} links to {os.readlink(home)}, which isn't there; is its drive connected?"
    parts = home.parts
    if len(parts) > 2 and parts[1] == "Volumes" and not Path("/Volumes", parts[2]).is_dir():
        return f"the data folder {home} is on /Volumes/{parts[2]}, which isn't mounted; is its drive connected?"
    return None
