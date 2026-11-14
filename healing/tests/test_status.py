"""`healing status` warnings from a fixture dataset and a fake `hex settings get`."""

import json
import stat
import sys

import pytest

from healing.cli import main
from healing.status import WARNINGS, compute
from healing.store import Store, now_ms

DAY = 24 * 3600 * 1000

FAKE_HEX = """#!{python}
import json, os, sys
args = sys.argv[1:]
assert args[:2] == ["settings", "get"] and "--json" in args, args
values = json.load(open(os.environ["FAKE_SETTINGS"]))
key = args[2]
if key not in values:
    sys.exit(1)
json.dump({{"version": 4, "status": "settings", "settings": [{{"key": key, "value": values[key]}}]}}, sys.stdout)
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    settings = tmp_path / "settings.json"
    hex_bin = tmp_path / "hex"
    hex_bin.write_text(FAKE_HEX.format(python=sys.executable))
    hex_bin.chmod(hex_bin.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_SETTINGS", str(settings))

    class World:
        pass

    w = World()
    w.hex = hex_bin
    w.store = Store()
    w.settings = lambda **v: settings.write_text(json.dumps(
        {"labs.keep_recordings": True, "history_retention": "week", **v}))
    w.run_at = lambda age_ms, error=None: (
        w.store.db.execute(
            "INSERT INTO runs (started_at_ms, new, error) VALUES (?, 1, ?)", (now_ms() - age_ms, error)),
        w.store.db.commit(),
    )
    w.settings()
    w.run_at(60_000)
    yield w
    w.store.close()


def codes(w, **kw):
    return compute(w.store, w.hex, **kw).warnings


def add_audio(w, name, size, label="unlabelled", evicted=None):
    if evicted is None:  # the index must match the disk, or status warns index_mismatch
        (w.store.root / "audio" / name).write_bytes(b"0" * size)
    w.store.db.execute(
        "INSERT INTO audio (sha256, path, bytes, samples, evicted_at_ms) VALUES (?, ?, ?, 1, ?)",
        (name, f"audio/{name}", size, evicted))
    w.store.db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text,"
        " audio_sha256, label, collected_at_ms) VALUES (?, 1, 1, 1, 'dictation', 'x', 'x', ?, ?, 1)",
        (name, name, label))
    w.store.db.commit()


def test_clean_state_exits_0(world, capsys):
    assert codes(world) == []
    assert main(["status", "--hex", str(world.hex)]) == 0
    assert "warning" not in capsys.readouterr().out


def test_keep_recordings_off_warns(world, capsys):
    world.settings(**{"labs.keep_recordings": False})
    assert codes(world) == ["keep_recordings_off"]
    assert main(["status", "--hex", str(world.hex)]) == 1


def test_recordings_hidden_when_off_with_audio_on_disk(world):
    world.settings(**{"labs.keep_recordings": False})
    add_audio(world, "a", 10)
    assert codes(world) == ["keep_recordings_off", "recordings_hidden"]


def test_gap_over_retention_warns(world):
    world.store.db.execute("DELETE FROM runs")
    world.run_at(7 * DAY)
    assert codes(world) == ["gap_exceeds_retention"]
    world.store.db.execute("DELETE FROM runs")
    world.run_at(5 * DAY)  # inside retention minus 1 day
    assert codes(world) == []


def test_failed_runs_do_not_count_as_collection(world):
    world.store.db.execute("DELETE FROM runs")
    world.run_at(7 * DAY)
    world.run_at(60_000, error="CollectError")
    assert codes(world) == ["gap_exceeds_retention"]


def test_retention_day_warns_twice(world):
    world.settings(history_retention="day")
    assert codes(world) == ["gap_exceeds_retention", "history_retention_day"]


def test_budget_pressure_warns(world):
    add_audio(world, "a", 95)
    assert codes(world, budget=100) == ["budget_over_90pct"]
    assert codes(world, budget=1000) == []


def test_strong_without_audio_warns(world):
    add_audio(world, "a", 1, label="strong", evicted=5)
    assert codes(world) == ["strong_without_audio"]


def test_unreadable_settings_warn_instead_of_guessing(world, tmp_path):
    (tmp_path / "settings.json").write_text("{}")
    assert "settings_unreadable" in codes(world)


def test_json_shape_and_list(world, capsys):
    assert main(["status", "--json", "--hex", str(world.hex)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert set(document) == {
        "clips", "with_audio", "audio_bytes", "budget", "labels", "clusters", "last_run",
        "oldest_uncollected_risk", "warnings", "free_bytes", "min_free", "last_clean",
    }
    assert main(["status", "--list"]) == 0
    listed = capsys.readouterr().out
    assert all(code in listed for code in WARNINGS) 


def test_status_does_not_sweep_temporaries(tmp_path, monkeypatch):
    from healing.store import Store

    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "h"))
    Store().close()
    leftover = tmp_path / "h" / "dataset" / "audio" / "ab" / "x.flac.tmp"
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"x")
    Store(sweep=False).close()
    assert leftover.exists()
    Store().close()
    assert not leftover.exists()
