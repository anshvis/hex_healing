import hashlib
import os
import time

import numpy as np
import pytest

from healing.budget import DEFAULT_BUDGET
from healing.clean import CleanError, clean, index_problems
from healing.cli import main
from healing.store import Pcm, Store, clip_id

DAY = 24 * 3600 * 1000
NOW = int(time.time() * 1000)
TEXT = "secret transcript words that must never reach a log"


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    monkeypatch.setenv("HEALING_HOME", str(path))
    return path


@pytest.fixture
def store(home):
    store = Store()
    yield store
    store.close()


def tone(seed: int, seconds=1.0) -> Pcm:
    rng = np.random.RandomState(seed)
    data = (rng.randn(int(16_000 * seconds)) * 3000).astype(np.int16)
    return Pcm(hashlib.sha256(data.astype("<i2").tobytes()).hexdigest(), data.size, data)


def add(store, n, age_days=30, label="weak", text=None):
    pcm = tone(n)
    store.write_audio(pcm)
    text = text or f"unique sentence number {n} about topic {n * 7}"
    store.db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text,"
        " audio_sha256, label, golden, audio_ms, collected_at_ms) VALUES (?,?,?,?,?,?,?,?,?,0,?,?)",
        (clip_id(1, n), n, 1, NOW - int(age_days * DAY) + n, "dictation", text, text,
         pcm.sha256, label, 1000, NOW),
    )
    store.db.commit()
    return pcm


def age(path, hours):
    stamp = time.time() - hours * 3600
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def ample():
    return 10**15


def row(store, pcm):
    return store.db.execute("SELECT * FROM audio WHERE sha256 = ?", (pcm.sha256,)).fetchone()


def quarantined(home):
    return sorted(p.name for p in (home / "quarantine").rglob("*") if p.is_file())


def test_orphan_is_quarantined_not_deleted(store, home):
    pcm = add(store, 1)
    orphan = tone(2)
    path = store.audio_path(orphan.sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile
    soundfile.write(path, orphan.data, 16_000, format="FLAC", subtype="PCM_16")
    report = clean(store, 0, now=NOW)
    assert report.orphans_moved == 1
    assert not path.exists()
    assert quarantined(home) == [path.name]
    assert store.audio_path(pcm.sha256).is_file()


def test_missing_file_marks_row_and_keeps_text(store):
    pcm = add(store, 1, text=TEXT)
    store.audio_path(pcm.sha256).unlink()
    report = clean(store, 0, now=NOW)
    assert report.missing_marked == 1
    audio = row(store, pcm)
    assert audio["missing"] == 1 and audio["evicted_at_ms"] == NOW
    kept = store.db.execute("SELECT raw_text, audio_sha256 FROM clips").fetchone()
    assert (kept["raw_text"], kept["audio_sha256"]) == (TEXT, pcm.sha256)


def corrupt_same_size(path):
    data = bytearray(path.read_bytes())
    for i in range(len(data) // 2, len(data) // 2 + 64):
        data[i] ^= 0xFF
    path.write_bytes(bytes(data))


def test_deep_finds_corrupt_file(store, home):
    good, bad = add(store, 1), add(store, 2)
    corrupt_same_size(store.audio_path(bad.sha256))
    assert index_problems(store) == {"orphans": 0, "missing": 0, "mismatched": 0}  # size-only sees nothing
    assert clean(store, 0, now=NOW).mismatched_marked == 0
    report = clean(store, 0, deep=True, now=NOW)
    assert report.mismatched_marked == 1
    assert row(store, bad)["corrupt"] == 1 and row(store, bad)["evicted_at_ms"] == NOW
    assert row(store, good)["corrupt"] == 0
    assert quarantined(home) == [store.audio_path(bad.sha256).name]


def test_size_mismatch_is_caught_without_deep(store):
    pcm = add(store, 1)
    with open(store.audio_path(pcm.sha256), "ab") as file:
        file.write(b"x")
    assert clean(store, 0, now=NOW).mismatched_marked == 1


def test_stale_temp_removed_fresh_kept(store, home):
    stale = store.root / "audio" / "ab" / "x.flac.tmp"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"1234")
    other = home / "cache" / "download.partial"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"12")
    fresh = home / "cache" / "new.tmp"
    fresh.write_bytes(b"1")
    age(stale, 2), age(other, 3), age(fresh, 0.2)
    report = clean(store, 0, now=NOW)
    assert report.temp_removed == 2
    assert not stale.exists() and not other.exists() and fresh.exists()


def test_old_quarantine_emptied(store, home):
    old = home / "quarantine" / str(NOW - 8 * DAY)
    young = home / "quarantine" / str(NOW - 2 * DAY)
    for folder in (old, young):
        folder.mkdir(parents=True)
        (folder / "a.flac").write_bytes(b"abc")
    report = clean(store, 0, now=NOW)
    assert report.quarantine_emptied == 1
    assert not old.exists() and young.exists()


def test_stale_round_scratch_removed_running_kept(store, home):
    import fcntl

    def make(name, hours):
        work = home / "rounds" / name / "work"
        work.mkdir(parents=True)
        (work / "big.bin").write_bytes(b"0" * 100)
        age(work, hours)
        return work

    stale, fresh, running = make("r1", 30), make("r2", 2), make("r3", 30)
    lock = home / "rounds" / "r3" / ".lock"
    lock.write_text("")
    held = os.open(lock, os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        report = clean(store, 0, now=NOW)
    finally:
        os.close(held)
    assert report.scratch_removed == 1
    assert not stale.exists() and fresh.exists() and running.exists()


def disk(home, capacity):
    """A fake disk: free space is capacity minus what is stored under the home."""
    def free():
        used = sum(p.stat().st_size for p in home.rglob("*") if p.is_file() and not p.is_symlink())
        return capacity - used
    return free


def test_below_floor_prunes_weak_until_floor_holds(store, home):
    pcms = [add(store, i) for i in range(6)]
    size = store.audio_path(pcms[0].sha256).stat().st_size
    free = disk(home, capacity=0)
    free_now = free()  # negative: set capacity so free is one file short of the floor
    floor = 3 * size
    capacity = -free_now + floor - size // 2  # free starts half a file under the floor
    report = clean(store, floor, now=NOW, free_fn=disk(home, capacity))
    left = store.db.execute("SELECT COUNT(*) FROM audio WHERE evicted_at_ms IS NULL").fetchone()[0]
    assert left == 5
    assert report.free_bytes_after >= floor
    assert store.db.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 6


def test_floor_never_evicts_protected(store, home):
    strong = [add(store, i, label="strong") for i in range(2)]
    weak = [add(store, 10 + i) for i in range(3)]
    size = store.audio_path(weak[0].sha256).stat().st_size
    floor = 10 * size
    base = disk(home, 0)()
    free = disk(home, -base + floor - size * 3 // 2)  # a file and a half short
    clean(store, floor, now=NOW, free_fn=free)
    assert all(row(store, s)["evicted_at_ms"] is None for s in strong)
    assert sum(row(store, w)["evicted_at_ms"] is not None for w in weak) == 2


def test_protected_over_floor_exits_4(store, home, monkeypatch, capsys):
    strong = [add(store, i, label="strong") for i in range(2)]
    size = store.audio_path(strong[0].sha256).stat().st_size
    floor = 10 * size
    base = disk(home, 0)()
    with pytest.raises(CleanError) as raised:
        clean(store, floor, now=NOW, free_fn=disk(home, -base + floor - size // 2))
    assert raised.value.code == 4 and "under" in str(raised.value)
    assert all(row(store, s)["evicted_at_ms"] is None for s in strong)
    monkeypatch.setattr("healing.clean.free_bytes", lambda *_: floor - 1)
    assert main(["clean", "--min-free", str(floor)]) == 4
    assert TEXT not in capsys.readouterr().err


def test_second_run_is_a_no_op(store, home):
    pcms = [add(store, i) for i in range(4)]
    store.audio_path(pcms[0].sha256).unlink()
    corrupt_same_size(store.audio_path(pcms[1].sha256))
    orphan = store.root / "audio" / "ff" / ("f" * 64 + ".flac")
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    stale = home / "a.tmp"
    stale.write_bytes(b"1")
    age(stale, 3)
    first = clean(store, 0, deep=True, now=NOW)
    assert (first.orphans_moved, first.missing_marked, first.mismatched_marked, first.temp_removed) == (1, 1, 1, 1)
    second = clean(store, 0, deep=True, now=NOW + 1000)
    assert not second.changed() and second.actions == []


def test_dry_run_changes_nothing(store, home):
    pcm = add(store, 1)
    store.audio_path(pcm.sha256).unlink()
    orphan = store.root / "audio" / "ff" / ("f" * 64 + ".flac")
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    report = clean(store, 0, dry_run=True, now=NOW)
    assert report.orphans_moved == 1 and report.missing_marked == 1
    assert orphan.exists() and row(store, pcm)["missing"] == 0
    assert store.db.execute("SELECT COUNT(*) FROM clean_runs").fetchone()[0] == 0


def test_never_follows_symlink_out(store, home, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    precious = outside / "keep.tmp"
    precious.write_bytes(b"precious")
    age(precious, 5)
    scratch = outside / "work"
    scratch.mkdir()
    (scratch / "data.bin").write_bytes(b"data")
    age(scratch, 50)
    pcm = add(store, 1)
    os.symlink(outside, home / "link")  # a folder link: no temp sweep through it
    os.symlink(precious, home / "dangling.tmp")
    age(home / "dangling.tmp", 5)
    (home / "rounds" / "r1").mkdir(parents=True)
    os.symlink(scratch, home / "rounds" / "r1" / "work")
    os.symlink(outside, home / "quarantine")  # quarantine itself is a link
    stray = outside / "stray.flac"
    stray.write_bytes(b"stray")
    os.symlink(stray, store.root / "audio" / "ab-link.flac")
    snapshot = sorted((p.name, p.stat().st_size, p.stat().st_mtime) for p in outside.rglob("*"))
    clean(store, 0, deep=True, now=NOW + 60 * DAY)
    assert sorted((p.name, p.stat().st_size, p.stat().st_mtime) for p in outside.rglob("*")) == snapshot
    assert store.audio_path(pcm.sha256).is_file()


def test_symlinked_audio_row_is_missing_not_read(store, tmp_path):
    pcm = add(store, 1)
    target = tmp_path / "elsewhere.flac"
    path = store.audio_path(pcm.sha256)
    path.rename(target)
    os.symlink(target, path)
    assert clean(store, 0, now=NOW).missing_marked == 1
    assert target.exists()


def test_log_has_counts_only(store, home, capsys):
    add(store, 1, text=TEXT)
    store.close()
    assert main(["clean", "--min-free", "0", "--deep"]) == 0
    seen = capsys.readouterr()
    assert TEXT not in seen.out + seen.err and "orphans 0" in seen.out
    check = Store()
    columns = [r[1] for r in check.db.execute("PRAGMA table_info(clean_runs)")]
    assert not any("text" in c for c in columns)
    check.close()


def test_rewritten_audio_resets_the_marks(store):
    pcm = add(store, 1)
    store.audio_path(pcm.sha256).unlink()
    clean(store, 0, now=NOW)
    assert row(store, pcm)["missing"] == 1
    store.write_audio(pcm)
    store.db.commit()
    assert (row(store, pcm)["missing"], row(store, pcm)["evicted_at_ms"]) == (0, None)


def test_collect_skips_audio_below_floor(home, tmp_path, monkeypatch):
    from healing import collect

    monkeypatch.setattr("healing.clean.free_bytes", lambda *_: 1)
    seen = {}
    monkeypatch.setattr(collect, "collect", lambda store, binary, since, copy: seen.setdefault("copy", copy) or collect.Summary())
    binary = tmp_path / "hex"
    binary.write_text("")
    collect.main_collect(None, binary, 10)
    assert seen["copy"] is False


def test_status_warns_below_floor_and_mismatch(store, tmp_path):
    from healing import status

    pcm = add(store, 1)
    store.audio_path(pcm.sha256).unlink()
    result = status.compute(store, tmp_path / "no-hex", DEFAULT_BUDGET, min_free=100, free_fn=lambda: 5)
    assert "disk_below_floor" in result.warnings and "index_mismatch" in result.warnings
    assert result.as_dict()["free_bytes"] == 5
    clean(store, 0, now=NOW)
    healed = status.compute(store, tmp_path / "no-hex", DEFAULT_BUDGET, min_free=1, free_fn=lambda: 5)
    assert "index_mismatch" not in healed.warnings and healed.last_clean["missing_marked"] == 1


def _flac(path, seed):
    import soundfile
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, tone(seed).data, 16_000, format="FLAC", subtype="PCM_16")


def test_linked_shard_orphan_untouched(store, home, tmp_path):
    outside = tmp_path / "out"
    _flac(outside / "x.flac", 50)
    os.symlink(outside, store.root / "audio" / "zz")
    report = clean(store, 0, now=NOW)
    assert report.orphans_moved == 0
    assert (outside / "x.flac").is_file()


def test_linked_quarantine_not_used_or_chmodded(store, home, tmp_path):
    outside = tmp_path / "out"
    outside.mkdir()
    os.chmod(outside, 0o755)
    os.symlink(outside, home / "quarantine")
    orphan = tone(51)
    path = store.audio_path(orphan.sha256)
    _flac(path, 51)
    clean(store, 0, now=NOW)
    assert path.is_file()
    assert list(outside.iterdir()) == []
    assert (outside.stat().st_mode & 0o777) == 0o755


def test_floor_prune_never_deletes_through_linked_shard(store, home, tmp_path):
    pcm = add(store, 1)
    path = store.audio_path(pcm.sha256)
    outside = tmp_path / "out"
    outside.mkdir()
    shard = path.parent
    os.rename(shard, outside / "moved")
    os.symlink(outside / "moved", shard)
    with pytest.raises(CleanError):
        clean(store, 10, now=NOW, free_fn=lambda: 0)
    assert (outside / "moved" / path.name).is_file()


def test_floor_shortfall_larger_than_weak_audio_evicts_all_weak(store, home):
    strong = [add(store, i, label="strong") for i in range(2)]
    weak = [add(store, 10 + i) for i in range(3)]
    size = store.audio_path(weak[0].sha256).stat().st_size
    floor = 10 * size
    base = disk(home, 0)()
    with pytest.raises(CleanError) as raised:
        clean(store, floor, now=NOW, free_fn=disk(home, -base + floor - 4 * size))
    assert raised.value.code == 4
    assert all(row(store, w)["evicted_at_ms"] is not None for w in weak)
    assert all(row(store, s)["evicted_at_ms"] is None for s in strong)


def _shard_temp(store, name="x.flac.tmp", hours=0):
    shard = store.root / "audio" / "ab"
    shard.mkdir(parents=True, exist_ok=True)
    tmp = shard / name
    tmp.write_bytes(b"x")
    if hours:
        age(tmp, hours)
    return tmp


def test_cli_dry_run_keeps_fresh_temp(store, home):
    tmp = _shard_temp(store)
    store.close()
    assert main(["clean", "--dry-run", "--min-free", "0"]) == 0
    assert tmp.exists()


def test_cli_dry_run_lists_but_keeps_old_temp(store, home, capsys):
    tmp = _shard_temp(store, hours=2)
    store.close()
    assert main(["clean", "--dry-run", "--min-free", "0"]) == 0
    assert tmp.exists()
    assert "remove temp" in capsys.readouterr().out


def test_linked_dataset_changes_nothing(tmp_path, home):
    store = Store()
    pcm = add(store, 1)
    store.close()
    external = tmp_path / "external"
    (home / "dataset").rename(external)
    os.symlink(external, home / "dataset")
    before = sorted(str(p) for p in external.rglob("*"))
    store = Store(sweep=False)
    with pytest.raises(CleanError):
        clean(store, 0)
    from healing.budget import prune

    prune(store, 10**12)
    store.close()
    assert sorted(str(p) for p in external.rglob("*")) == before
    assert (external / "audio" / pcm.sha256[:2] / f"{pcm.sha256}.flac").exists()


def test_linked_audio_dir_changes_nothing(tmp_path, store, home):
    out = tmp_path / "out"
    (out / "ab").mkdir(parents=True)
    victim = out / "ab" / "x.flac"
    victim.write_bytes(b"x")
    (store.root / "audio").rmdir()
    os.symlink(out, store.root / "audio")
    with pytest.raises(CleanError):
        clean(store, 0)
    assert victim.exists()
    assert not (home / "quarantine").exists()
