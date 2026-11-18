"""`healing schedule` with launchctl stubbed; the live check uses the real one."""

import os
import plistlib
import stat

import pytest

from healing import schedule


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    hex_bin = tmp_path / "bin" / "hex"
    hex_bin.parent.mkdir()
    hex_bin.write_text("#!/bin/sh\n")
    hex_bin.chmod(hex_bin.stat().st_mode | stat.S_IEXEC)
    calls = []
    monkeypatch.setattr(
        schedule, "launchctl",
        lambda *a: (calls.append(a), type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())[1],
    )

    class World:
        pass

    w = World()
    w.hex, w.calls, w.agents = hex_bin, calls, tmp_path / "agents"
    return w


def test_plist_renders_absolute_paths(world):
    target = schedule.install(world.hex, world.agents)
    document = plistlib.loads(target.read_bytes())
    assert document["Label"] == schedule.LABEL
    assert document["RunAtLoad"] is True
    assert "StartInterval" not in document  # a sleeping Mac skips StartInterval firings
    assert [d["Hour"] for d in document["StartCalendarInterval"]] == [0, 6, 12, 18]
    assert document["ProgramArguments"][-2:] == ["schedule", "run"]
    assert os.path.isabs(document["ProgramArguments"][0])
    assert document["EnvironmentVariables"]["HEX_BIN"] == str(world.hex.resolve())
    assert os.path.isabs(document["StandardOutPath"])
    assert stat.S_IMODE(os.stat(document["StandardOutPath"]).st_mode) == 0o600


def test_install_twice_is_idempotent(world):
    first = schedule.install(world.hex, world.agents).read_text()
    second = schedule.install(world.hex, world.agents).read_text()
    assert first == second
    assert [c[0] for c in world.calls] == ["bootout", "bootstrap"] * 2


def test_remove_deletes_and_is_idempotent(world):
    target = schedule.install(world.hex, world.agents)
    schedule.remove(world.agents)
    schedule.remove(world.agents)
    assert not target.exists()
    assert [c[0] for c in world.calls].count("bootout") == 3


def test_install_refuses_missing_hex(world, tmp_path):
    with pytest.raises(schedule.ScheduleError):
        schedule.install(tmp_path / "nope", world.agents)


def test_run_once_survives_unexpected_collect_error(monkeypatch, capsys):
    from healing import budget, collect

    monkeypatch.setattr(collect, "main_collect", lambda *_: (_ for _ in ()).throw(ValueError("SECRET TRANSCRIPT TEXT")))
    pruned = []

    class Fake:
        def summary(self):
            return "pruned"

    class FakeStore:
        def close(self):
            pass

    import contextlib
    from healing import store

    monkeypatch.setattr(store, "Store", lambda *a, **k: FakeStore())
    monkeypatch.setattr(store, "process_lock", contextlib.nullcontext)
    monkeypatch.setattr(budget, "prune", lambda s: pruned.append(1) or Fake())
    assert schedule.run_once() == 1
    seen = capsys.readouterr()
    assert pruned and "SECRET" not in seen.out + seen.err and "ValueError" in seen.err


def test_train_auto_job_runs_daily_and_not_at_load(world):
    target = schedule.install(world.hex, world.agents, job=schedule.TRAIN_AUTO)
    document = plistlib.loads(target.read_bytes())
    assert target.name == "com.hex-healing.train-auto.plist"
    assert document["Label"] == "com.hex-healing.train-auto"
    assert document["ProgramArguments"][-2:] == ["train", "auto"]
    assert [d["Hour"] for d in document["StartCalendarInterval"]] == [3]
    assert document["RunAtLoad"] is False  # a login never starts hours of training
    assert document["StandardOutPath"].endswith("train-auto.log")
    assert not schedule.plist_path(world.agents).exists()  # the collect job is untouched
    schedule.remove(world.agents, job=schedule.TRAIN_AUTO)
    assert not target.exists()
