import hashlib
import io
import os

import numpy as np
import pytest
import soundfile

from healing.sets import librispeech
from healing.sets.librispeech import FetchError, fetch


def shard(ids):
    """A tiny parquet shard in Hugging Face's layout: `id`, `text`, `audio` {bytes, path}."""
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    rows = []
    for index, clip in enumerate(ids):
        buffer = io.BytesIO()
        tone = (np.sin(np.arange(1600) * (index + 1) * 0.05) * 0.3).astype(np.float32)
        soundfile.write(buffer, tone, 16_000, format="FLAC")
        rows.append({"id": clip, "text": f"WORDS {index}", "audio": {"bytes": buffer.getvalue(), "path": f"{clip}.flac"}})
    out = io.BytesIO()
    parquet.write_table(arrow.Table.from_pylist(rows), out)
    return out.getvalue()


class Response:
    def __init__(self, body, status, start, cut=None):
        self.body = body[:cut] if cut is not None else body
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{start + len(body) - 1}/*"} if status == 206 else {}
        self.position = 0
        self.fail_at = cut

    def read(self, n):
        if self.position >= len(self.body) and self.fail_at is not None and self.fail:
            raise ConnectionResetError("dropped")
        chunk = self.body[self.position : self.position + n]
        self.position += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeHub:
    """Serves the pinned shards by URL, honouring `Range`; `cut` ends the next response early."""

    def __init__(self, files, ignore_range=False):
        self.files = files
        self.ignore_range = ignore_range
        self.requests = []
        self.cut = None
        self.fail = True

    def __call__(self, request):
        path = request.full_url.split(f"/{librispeech.LIBRISPEECH_REVISION}/", 1)[1]
        header = request.get_header("Range")
        self.requests.append((path, header))
        body = self.files[path]
        start = int(header.removeprefix("bytes=").rstrip("-")) if header and not self.ignore_range else 0
        status = 206 if header and not self.ignore_range else 200
        response = Response(body[start:], status, start, self.cut)
        response.fail = self.fail
        self.cut = None
        return response


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    files = {
        "clean/train.100/0000.parquet": shard(["19-198-0000", "19-198-0001"]),
        "clean/train.100/0001.parquet": shard(["26-495-0000"]),
    }
    monkeypatch.setattr(librispeech, "SHARDS", {
        path: (len(body), hashlib.sha256(body).hexdigest()) for path, body in files.items()
    })
    monkeypatch.setattr(librispeech, "CHUNK", 64)
    return FakeHub(files)


def test_fetch_writes_the_folder_replay_reads(hub):
    from healing.train import replay

    result = fetch(opener=hub)
    assert result.folder == librispeech.cache_folder() == replay.default_source()
    assert sorted(p.name for p in result.folder.iterdir()) == ["0000.parquet", "0001.parquet"]
    utterances = sorted(replay.LocalLibriSpeech(replay.default_source()).utterances(), key=lambda u: u.id)
    assert [u.id for u in utterances] == ["ls-19-198-0000", "ls-19-198-0001", "ls-26-495-0000"]
    samples = utterances[0].load()
    assert samples.dtype == np.int16 and len(samples) == 1600


def test_a_complete_verified_folder_is_a_no_op(hub):
    fetch(opener=hub)
    count = len(hub.requests)
    again = fetch(opener=hub)
    assert again.unchanged and again.fetched == []
    assert len(hub.requests) == count


def test_a_dropped_connection_resumes_from_the_partial(hub):
    hub.cut = 300
    with pytest.raises(FetchError, match="run again to resume"):
        fetch(opener=hub)
    partial = librispeech.cache_folder() / "0000.parquet.partial"
    assert partial.stat().st_size == 300
    assert not (librispeech.cache_folder() / "0000.parquet").exists()

    result = fetch(opener=hub)
    assert result.fetched == ["0000.parquet", "0001.parquet"]
    assert ("clean/train.100/0000.parquet", "bytes=300-") in hub.requests
    assert not partial.exists()
    body = hub.files["clean/train.100/0000.parquet"]
    assert (librispeech.cache_folder() / "0000.parquet").read_bytes() == body


def test_a_short_response_without_an_error_keeps_the_partial(hub):
    hub.cut, hub.fail = 100, False
    with pytest.raises(FetchError, match="ended after 100"):
        fetch(opener=hub)
    assert (librispeech.cache_folder() / "0000.parquet.partial").stat().st_size == 100


def test_a_server_that_ignores_the_range_starts_over(hub):
    hub.cut = 200
    with pytest.raises(FetchError):
        fetch(opener=hub)
    hub.ignore_range = True
    fetch(opener=hub)
    assert (librispeech.cache_folder() / "0000.parquet").read_bytes() == hub.files["clean/train.100/0000.parquet"]


def test_a_hash_mismatch_is_refused_and_never_named_as_a_shard(hub):
    hub.files["clean/train.100/0001.parquet"] = b"x" * len(hub.files["clean/train.100/0001.parquet"])
    with pytest.raises(FetchError, match="sha256"):
        fetch(opener=hub)
    folder = librispeech.cache_folder()
    assert (folder / "0000.parquet").is_file()
    assert not (folder / "0001.parquet").exists()
    assert not (folder / "0001.parquet.partial").exists()


def test_a_changed_shard_on_disk_is_refused(hub):
    fetch(opener=hub)
    target = librispeech.cache_folder() / "0001.parquet"
    body = bytearray(target.read_bytes())
    body[-1] ^= 1
    target.write_bytes(bytes(body))
    with pytest.raises(FetchError, match="sha256"):
        fetch(opener=hub)


def test_too_little_disk_is_refused_before_downloading(hub, monkeypatch):
    monkeypatch.setattr(librispeech, "free_bytes", lambda _: 10)
    with pytest.raises(FetchError, match="bytes free"):
        fetch(opener=hub)
    assert hub.requests == []


def test_dest_writes_there_and_links_the_cache_path(hub, tmp_path):
    from healing.train import replay

    dest = tmp_path / "big-disk" / "librispeech"
    result = fetch(dest, opener=hub)
    cache = librispeech.cache_folder()
    assert result.linked and result.folder == dest
    assert cache.is_symlink() and cache.resolve() == dest.resolve()
    assert (dest / "0000.parquet").is_file()
    assert len(list(replay.LocalLibriSpeech(replay.default_source()).utterances())) == 3

    again = fetch(dest, opener=hub)
    assert again.unchanged and not again.linked
    assert fetch(opener=hub).unchanged  # through the link


def test_dest_re_points_an_existing_link(hub, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    fetch(first, opener=hub)
    result = fetch(second, opener=hub)
    assert result.linked
    assert librispeech.cache_folder().resolve() == second.resolve()


def test_dest_refuses_a_real_folder_at_the_cache_path(hub, tmp_path):
    cache = librispeech.cache_folder()
    cache.mkdir(parents=True)
    (cache / "mine.txt").write_text("keep")
    with pytest.raises(FetchError, match="not a symlink"):
        fetch(tmp_path / "elsewhere", opener=hub)
    assert hub.requests == []
    assert (cache / "mine.txt").read_text() == "keep"


def test_dest_equal_to_the_cache_path_writes_in_place(hub):
    cache = librispeech.cache_folder()
    result = fetch(cache, opener=hub)
    assert not result.linked and not cache.is_symlink()
    assert (cache / "0000.parquet").is_file()


def test_a_dangling_cache_link_is_refused(hub, tmp_path):
    cache = librispeech.cache_folder()
    cache.parent.mkdir(parents=True)
    os.symlink(tmp_path / "gone", cache)
    with pytest.raises(FetchError, match="does not exist"):
        fetch(opener=hub)


def test_pins_cover_the_whole_split():
    assert len(librispeech.SHARDS) == 14
    assert all(path.startswith(librispeech.PREFIX) for path in librispeech.SHARDS)
    assert all(len(sha) == 64 for _, sha in librispeech.SHARDS.values())
    assert librispeech.total_bytes() == 6_433_926_591


def test_cli_fetch_prints_the_folder(hub, monkeypatch, capsys):
    from healing import cli

    monkeypatch.setattr(librispeech, "_urlopen", hub)
    assert cli.main(["sets", "fetch-librispeech"]) == 0
    out = capsys.readouterr().out
    assert "fetched 2 shard(s)" in out and str(librispeech.cache_folder()) in out
    assert cli.main(["sets", "fetch-librispeech"]) == 0
    assert capsys.readouterr().out.startswith("unchanged")
