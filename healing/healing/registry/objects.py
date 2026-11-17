"""Content-addressed artifacts: `$HEALING_HOME/registry/objects/<aa>/<sha256>`, owner-only.

Every file is copied in through a temp file in the same folder, fsynced, renamed, and the folder
fsynced, so a crash leaves either the whole object or none under its name. The name is the content's
SHA-256, so the same GGUF registered twice is stored once.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from healing.paths import healing_home

CUSTOM_PREFIX = "custom-"
CUSTOM_DIGITS = 16  # `src/transcription/custom_models.rs`: `custom-` + the first 16 hex of the SHA-256


def registry_dir() -> Path:
    return healing_home() / "registry"


def objects_dir() -> Path:
    return registry_dir() / "objects"


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def owner_dir(path: Path) -> None:
    """Make `path` (0700) and any missing ancestors, fsyncing each new folder's parent so its entry is
    durable, not only the files later written inside it."""
    if path.is_dir():
        os.chmod(path, 0o700)
        return
    owner_dir(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    os.chmod(path, 0o700)
    fsync_dir(path.parent)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hex_model_id(sha256: str) -> str:
    """The id Hex gives this file when it is imported (`hex model import`)."""
    return CUSTOM_PREFIX + sha256[:CUSTOM_DIGITS]


def path(sha256: str) -> Path:
    return objects_dir() / sha256[:2] / sha256


def has(sha256: str | None) -> bool:
    return bool(sha256) and path(sha256).is_file()


def put(source: Path, sha256: str | None = None) -> str:
    """Store `source`; returns its SHA-256. A copy that does not hash to `sha256` (when given) is refused.
    An object already present is kept only when it still hashes to its name; otherwise it is atomically
    replaced."""
    digest = sha256_file(source)
    if sha256 is not None and digest != sha256:
        raise ValueError(f"{source} hashes to {digest[:12]}, not {sha256[:12]}")
    target = path(digest)
    if target.is_file() and sha256_file(target) == digest:
        return digest  # a stored copy that no longer matches its name is replaced below
    owner_dir(objects_dir())
    owner_dir(target.parent)
    descriptor, scratch = tempfile.mkstemp(prefix=digest + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as out, open(source, "rb") as file:
            shutil.copyfileobj(file, out, 1 << 20)
            out.flush()
            os.fsync(out.fileno())
        if sha256_file(Path(scratch)) != digest:  # the source changed while it was copied
            raise ValueError(f"{source} changed while it was copied")
        os.chmod(scratch, 0o600)
        os.replace(scratch, target)
    except BaseException:
        Path(scratch).unlink(missing_ok=True)
        raise
    fsync_dir(target.parent)
    return digest


def remove(sha256: str) -> bool:
    """Delete one object; True when a file was removed."""
    target = path(sha256)
    try:
        target.unlink()
    except FileNotFoundError:
        return False
    fsync_dir(target.parent)
    return True


def total_bytes() -> int:
    """What the objects store holds on disk (`healing status`)."""
    root = objects_dir()
    if not root.is_dir():
        return 0
    return sum(f.stat().st_size for f in root.glob("*/*") if f.is_file() and not f.name.endswith(".tmp"))
