"""LibriSpeech train-clean-100, the replay source `healing train prepare` reads (D-25, D-45).

`healing sets fetch-librispeech` downloads Hugging Face's parquet copy (`openslr/librispeech_asr`,
config `clean`, split `train.100`) at the revision the general set pins, into the folder
`train/replay.py` reads: `$HEALING_HOME/cache/librispeech-train-clean-100/<NNNN>.parquet`, each shard
with the `id`, `text` and `audio` columns as published. Every shard is pinned by size and SHA-256.

Downloads go to `<shard>.partial` and resume from its length with an HTTP range; a shard becomes
`<NNNN>.parquet` only once its size and hash check, so the folder never holds an unverified shard
under its final name. A complete, verified folder is a no-op. With `dest`, the shards go there and
the cache path becomes a symlink to it; a real file or folder already at the cache path is refused.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from healing.paths import healing_home
from healing.registry.objects import fsync_dir
from healing.sets.general import HF_URL, LIBRISPEECH_REPO, LIBRISPEECH_REVISION

PREFIX = "clean/train.100/"
# {repo path: (bytes, sha256)}: the Hugging Face tree at LIBRISPEECH_REVISION (LFS oids are SHA-256).
SHARDS: dict[str, tuple[int, str]] = {
    "clean/train.100/0000.parquet": (469629285, "3098c6e44d1d49f8c62bd123775f1c492bda2cabd80f3ee70fe7800572371401"),
    "clean/train.100/0001.parquet": (481204796, "011cb76e296b2650bab3ea54044c57d10293dadb05e9e96605b37c88e4bc89bd"),
    "clean/train.100/0002.parquet": (462830007, "b836ad8f9b43b2d303d81bc91201d6b0b10806ee4407c4ab06af9ddfa963bb99"),
    "clean/train.100/0003.parquet": (464400585, "f13450f202d20681ee89aab48a6b59f2dc36d5f7e1e9c36c59ba3cfae15d0049"),
    "clean/train.100/0004.parquet": (486747029, "89d08c9934fd32dc97147f3177392c1e734bb5d13c60c967523d0ba05397c5f6"),
    "clean/train.100/0005.parquet": (453059318, "fbc85ec6ddd6a0caced0e9978586f6b174392e68d09adc64a10a89679ecb713a"),
    "clean/train.100/0006.parquet": (460852251, "61cffc43952f370a65d0855e3260de4d81815fd02eac8b33f3c118fad8eebda8"),
    "clean/train.100/0007.parquet": (451701540, "cfb26522a7e15cc425714df94e874d347f5dbb3f603e348611989522d9833ca7"),
    "clean/train.100/0008.parquet": (464545879, "e41ba4f3ca0b0de4b83e343f9d6cf90418f7bb9c80ebf1a8e663d8f46e5f1c1e"),
    "clean/train.100/0009.parquet": (445148714, "ed130ea9bdaaf8d188eab0521cdd4b8aa5d507b0c3f8e8df5d90db2d10b4f5f1"),
    "clean/train.100/0010.parquet": (454357890, "a6feec58aaa6a7e4247deb3bc3f10fedbc2e6aec2840dda663dd9ecc808575fc"),
    "clean/train.100/0011.parquet": (432464858, "3196ef576660eb62055df0b1737c1e4866f1789176acc7d85a928aeb16b07f03"),
    "clean/train.100/0012.parquet": (456996471, "33b99d9a1e9ddcf411cad97379fb74d5e53c0a8b5846d6c2fe06fb1d87166bcf"),
    "clean/train.100/0013.parquet": (449987968, "5d25b68b6600db0dcf975c1cae12e3892a2ade5358731a275a2358637c2c0bd0"),
}
CHUNK = 1 << 20

# `urllib.request.urlopen` by default; tests pass a fake that serves bytes and honours `Range`.
Opener = Callable[[urllib.request.Request], object]
TIMEOUT_S = 60


def _urlopen(request: urllib.request.Request):
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)
Progress = Callable[[str, int, int], None]


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    folder: Path
    fetched: list[str]  # shard names downloaded (or finished) by this run
    total_bytes: int
    linked: bool  # the cache path was made (or re-pointed) a symlink to `dest`

    @property
    def unchanged(self) -> bool:
        return not self.fetched and not self.linked


def cache_folder() -> Path:
    """Where `healing train prepare` looks for the shards by default."""
    return healing_home() / "cache" / "librispeech-train-clean-100"


def total_bytes() -> int:
    return sum(size for size, _ in SHARDS.values())


def fetch(dest: Path | None = None, *, opener: Opener | None = None,
          progress: Progress | None = None) -> FetchResult:
    """Downloads every pinned shard not already verified in the folder, then links the cache path."""
    opener = opener or _urlopen
    cache = cache_folder()
    link = dest is not None and _absolute(dest) != _absolute(cache)
    if link:
        _check_link_target(cache)
    folder = dest if link else cache
    if folder.is_symlink() and not folder.exists():
        raise FetchError(f"{folder} is a symlink to {os.readlink(folder)}, which does not exist")
    folder.mkdir(parents=True, exist_ok=True)

    pending = [path for path in sorted(SHARDS) if not _verified(folder / _name(path), *SHARDS[path])]
    needed = sum(SHARDS[path][0] - _partial_size(folder / _name(path), SHARDS[path][0]) for path in pending)
    free = free_bytes(folder)
    if needed > free:
        raise FetchError(f"{folder} has {free:,} bytes free; the remaining shards need {needed:,}")
    for path in pending:
        size, sha256 = SHARDS[path]
        _download(HF_URL.format(repo=LIBRISPEECH_REPO, revision=LIBRISPEECH_REVISION, path=path),
                  folder / _name(path), size, sha256, opener, progress)

    linked = _link(cache, dest) if link else False
    return FetchResult(folder, [_name(path) for path in pending], total_bytes(), linked)


def free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


def _name(path: str) -> str:
    return path.removeprefix(PREFIX)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _check_link_target(cache: Path) -> None:
    if cache.exists() and not cache.is_symlink():
        raise FetchError(f"{cache} already exists and is not a symlink; move it away (or fetch without --dest)")


def _link(cache: Path, dest: Path) -> bool:
    """Points `cache` at `dest`; False when it already did."""
    target = _absolute(dest)
    if cache.is_symlink() and _absolute(cache.parent / os.readlink(cache)) == target:
        return False
    _check_link_target(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scratch = cache.with_name(cache.name + ".link")
    scratch.unlink(missing_ok=True)
    scratch.symlink_to(target, target_is_directory=True)
    os.replace(scratch, cache)
    return True


def _verified(target: Path, size: int, sha256: str) -> bool:
    if not target.exists():
        return False
    actual_size = target.stat().st_size
    if actual_size != size:
        raise FetchError(f"{target} is {actual_size:,} bytes, expected {size:,}; remove it and fetch again")
    actual = _sha256_file(target)
    if actual != sha256:
        raise FetchError(f"{target} has sha256 {actual}, expected {sha256}; remove it and fetch again")
    return True


def _partial_size(target: Path, size: int) -> int:
    partial = _partial(target)
    have = partial.stat().st_size if partial.exists() else 0
    return have if have <= size else 0


def _partial(target: Path) -> Path:
    return target.with_name(target.name + ".partial")


def _download(url: str, target: Path, size: int, sha256: str, opener: Opener,
              progress: Progress | None) -> None:
    partial = _partial(target)
    have = _partial_size(target, size)
    if have < size:
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with opener(request) as response:
                status = getattr(response, "status", 200)
                if have and status == 206:
                    start = _range_start(response)
                    if start != have:
                        raise FetchError(f"{url}: asked for bytes from {have}, the server sent from {start}")
                elif status == 200:
                    have = 0  # the server ignored the range: start over
                else:
                    raise FetchError(f"{url}: HTTP {status}")
                with open(partial, "r+b" if have else "wb") as file:
                    file.seek(have)
                    file.truncate()
                    while chunk := response.read(CHUNK):
                        file.write(chunk)
                        have += len(chunk)
                        if have > size:
                            break
                        if progress:
                            progress(target.name, have, size)
                    file.flush()
                    os.fsync(file.fileno())
        except (OSError, http.client.HTTPException) as error:
            raise FetchError(f"could not download {url}: {error!r} ({_kept(partial):,} of {size:,} bytes "
                             "kept; run again to resume)") from error
    actual_size = partial.stat().st_size
    if actual_size < size:
        raise FetchError(f"{url}: the connection ended after {actual_size:,} of {size:,} bytes "
                         "(kept; run again to resume)")
    actual = _sha256_file(partial)
    if actual_size != size or actual != sha256:
        partial.unlink()
        raise FetchError(f"{url}: got {actual_size:,} bytes with sha256 {actual}, expected {size:,} bytes "
                         f"with {sha256}; the partial download was removed")
    os.replace(partial, target)
    fsync_dir(target.parent)


def _kept(partial: Path) -> int:
    return partial.stat().st_size if partial.exists() else 0


def _range_start(response) -> int | None:
    header = response.headers.get("Content-Range", "") if hasattr(response, "headers") else ""
    # "bytes <start>-<end>/<total>"
    try:
        return int(header.split()[1].split("-")[0])
    except (IndexError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def print_progress(name: str, done: int, total: int) -> None:
    """A one-line progress meter on stderr, redrawn at each whole percent."""
    percent = done * 100 // total
    if done == total or percent != (done - CHUNK) * 100 // total:
        end = "\n" if done == total else ""
        print(f"\r{name}: {percent:3d}% of {total / 1e6:,.0f} MB", end=end, file=sys.stderr, flush=True)
