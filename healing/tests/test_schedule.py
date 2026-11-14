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
    assert document["StartInterval"] == 21_600 and document["RunAtLoad"] is True
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
