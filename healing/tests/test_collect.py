"""`healing collect` against a fake `hex` that serves a fixture history.json."""

import copy
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

from healing import collect as collect_module
from healing.cli import main
from healing.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
WAV = FIXTURES / "one-second.wav"

FAKE_HEX = """#!{python}
import json, os, sys
from datetime import datetime, timezone
args = sys.argv[1:]
assert args[:3] == ["history", "--json", "--raw"] and "--oldest-first" in args, args
doc = json.load(open(os.environ["FAKE_HISTORY"]))
limit = int(args[args.index("--limit") + 1])
if "--since" in args:
    value = args[args.index("--since") + 1]
    floor = int(datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)
    doc["entries"] = [e for e in doc["entries"] if e["timestamp_ms"] >= floor]
doc["entries"].sort(key=lambda e: (e["timestamp_ms"], e["id"]))
doc["truncated"] = len(doc["entries"]) > limit
doc["entries"] = doc["entries"][:limit]
json.dump(doc, sys.stdout)
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "healing-home"
    monkeypatch.setenv("HEALING_HOME", str(home))
    history = tmp_path / "history.json"
    hex_bin = tmp_path / "hex"
    hex_bin.write_text(FAKE_HEX.format(python=sys.executable))
    hex_bin.chmod(hex_bin.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_HISTORY", str(history))
    document = json.loads((FIXTURES / "history.json").read_text().replace("{wav}", str(WAV)))

    class World:
        pass

    w = World()
    w.home, w.history, w.hex, w.doc = home, history, hex_bin, document
    w.save = lambda: history.write_text(json.dumps(w.doc))
    w.run = lambda *extra: main(["collect", "--hex", str(hex_bin), *extra])
    w.save()
    w.db = lambda: Store().db
    return w


def clips(w):
    return [dict(r) for r in w.db().execute("SELECT * FROM clips ORDER BY hex_epoch_ms, hex_id")]


def flacs(w):
    return sorted((w.home / "dataset" / "audio").glob("*/*.flac"))


def test_first_run_stores_rows_and_audio(world, capsys):
    assert world.run() == 0
    assert capsys.readouterr().out.strip() == "seen 3, new 3, audio 1, missing_audio 0"
    rows = clips(world)
    assert len(rows) == 3
    assert rows[0]["raw_text"] == "hello there" and rows[0]["model_id"] == "parakeet-unified"
    assert rows[1]["rewritten"] == 1 and rows[1]["application"] == "Mail"
    assert rows[2]["audio_sha256"] is None and rows[2]["kind"] == "raw"
    # Both recordings are the same PCM, so they share one FLAC (the same PCM is one row).
    assert len(flacs(world)) == 1
    assert rows[0]["audio_sha256"] == rows[1]["audio_sha256"] is not None


def test_two_distinct_recordings_make_two_flacs(world, tmp_path, capsys):
    other = tmp_path / "other.wav"
    data, rate = soundfile.read(WAV, dtype="int16")
    soundfile.write(other, data[::-1].copy(), rate, subtype="PCM_16")
    world.doc["entries"][1]["recording"] = str(other)
    world.save()
    world.run()
    assert capsys.readouterr().out.strip() == "seen 3, new 3, audio 2, missing_audio 0"
    assert len(flacs(world)) == 2


def test_second_run_is_a_no_op(world, capsys):
    world.run()
    before = clips(world)
    capsys.readouterr()
    assert world.run() == 0
    assert capsys.readouterr().out.strip() == "seen 3, new 0, audio 0, missing_audio 0"
    assert clips(world) == before


def test_correction_arriving_after_first_collect_is_picked_up(world):
    world.run()
    world.doc["entries"][0]["final_text"] = "Hello, there."
    world.save()
    world.run()
    rows = clips(world)
    assert len(rows) == 3 and rows[0]["final_text"] == "Hello, there."
    assert rows[0]["raw_text"] == "hello there"


def test_replace_updates_raw_text_and_model_id_and_marks_labels_stale(world):
    world.run()
    store = Store()
    store.db.execute("UPDATE clips SET label = 'strong', cluster_id = 'c1', golden = 1 WHERE hex_id = 1")
    store.db.execute("UPDATE clips SET label = 'verified' WHERE hex_id = 2")
    store.db.execute(  # a verified label is a review: collect leaves it to `labels.refresh`
        "INSERT INTO reviews (clip_id, reviewed_at_ms, action, text, strategy)"
        " SELECT clip_id, 1, 'accept', raw_text, 'random' FROM clips WHERE hex_id = 2"
    )
    store.db.commit()
    store.close()
    world.doc["entries"][0].update(raw_text="hello their", model_id="whisper-large")
    world.doc["entries"][1].update(raw_text="thanks anne")
    world.save()
    world.run()
    rows = clips(world)
    assert rows[0]["raw_text"] == "hello their" and rows[0]["model_id"] == "whisper-large"
    assert rows[0]["label"] == "unlabelled"
    assert rows[0]["cluster_id"] == "c1" and rows[0]["golden"] == 1
    assert rows[1]["label"] == "verified"


def test_expired_entry_stays(world, capsys):
    world.run()
    world.doc["entries"].pop(0)
    world.save()
    world.run()
    assert capsys.readouterr().out.strip().endswith("missing_audio 0")
    assert len(clips(world)) == 3 and len(flacs(world)) == 1
    store = Store()
    assert store.db.execute("SELECT COUNT(*) FROM audio WHERE evicted_at_ms IS NULL").fetchone()[0] == 1


def test_flac_decodes_to_identical_pcm(world):
    world.run()
    flac = flacs(world)[0]
    original, _ = soundfile.read(WAV, dtype="int16")
    decoded, rate = soundfile.read(flac, dtype="int16")
    assert rate == 16000 and np.array_equal(original, decoded)
    assert stat.S_IMODE(flac.stat().st_mode) == 0o600
    assert stat.S_IMODE((world.home / "dataset").stat().st_mode) == 0o700


def test_missing_wav_counts_missing_audio(world, capsys):
    world.doc["entries"][1]["recording"] = str(world.home / "gone.wav")
    world.save()
    assert world.run() == 0
    assert capsys.readouterr().out.strip() == "seen 3, new 3, audio 1, missing_audio 1"
    rows = clips(world)
    assert rows[1]["audio_sha256"] is None
    # The file turns up on the next scan: the row gains its audio.
    world.doc["entries"][1]["recording"] = str(WAV)
    world.save()
    world.run()
    assert clips(world)[1]["audio_sha256"] is not None


def test_store_reset_gets_new_epoch(world):
    world.run()
    reset = copy.deepcopy(world.doc)
    for index, entry in enumerate(reset["entries"], start=1):
        entry["id"] = index
        entry["timestamp_ms"] += 10_000_000
        entry["final_text"] = f"after reset {index}"
        entry["raw_text"] = entry["final_text"]
    world.doc = reset
    world.save()
    world.run()
    rows = clips(world)
    assert len(rows) == 6
    assert len({r["hex_epoch_ms"] for r in rows}) == 2
    assert len({r["clip_id"] for r in rows}) == 6


def test_paging_across_the_limit_keeps_equal_timestamp_rows(world, monkeypatch):
    monkeypatch.setattr(collect_module, "PAGE_LIMIT", 4)
    stamp = 1760000000000
    world.doc["entries"] = [
        {"id": i, "timestamp_ms": stamp + (i // 3) * 5000, "kind": "dictation",
         "final_text": f"t{i}", "raw_text": f"t{i}"}
        for i in range(1, 11)
    ]
    world.save()
    world.run()
    assert [r["hex_id"] for r in clips(world)] == list(range(1, 11))


def test_refuses_history_version_2(world, capsys):
    world.doc["version"] = 2
    world.save()
    assert world.run() == 3
    assert "refusing" in capsys.readouterr().err
    assert clips(world) == []
    row = world.db().execute("SELECT error FROM runs").fetchone()
    assert row["error"]


def test_hex_not_found_exits_2(world, tmp_path):
    assert main(["collect", "--hex", str(tmp_path / "nope")]) == 2


def test_epoch_ms_since_is_rejected(world):
    assert world.run("--since", "1760000000000") == 2


def tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*"))


def test_read_only_hex_data_root(world, tmp_path, monkeypatch):
    support = tmp_path / "hex-support"
    (support / "history-audio").mkdir(parents=True)
    wav = support / "history-audio" / "a.wav"
    wav.write_bytes(WAV.read_bytes())
    for entry in world.doc["entries"]:
        if entry.get("recording"):
            entry["recording"] = str(wav)
    (support / "history.json").write_text(json.dumps(world.doc))
    monkeypatch.setenv("FAKE_HISTORY", str(support / "history.json"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(support))
    before = tree(support)
    for path in [support, *support.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        assert world.run() == 0
        assert world.run() == 0
        assert tree(support) == before
    finally:
        for path in [support, *support.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert len(clips(world)) == 3 and len(flacs(world)) == 1


def test_clock_rollback_is_idempotent(world, capsys):
    stamp = 1760000000000
    world.doc["entries"] = [
        {"id": i, "timestamp_ms": ts, "kind": "dictation", "final_text": f"t{i}", "raw_text": f"t{i}"}
        for i, ts in [(1, stamp + 1000), (2, stamp + 2000), (3, stamp)]
    ]
    world.save()
    world.run()
    capsys.readouterr()
    world.run()
    assert capsys.readouterr().out.startswith("seen 3, new 0")
    rows = clips(world)
    assert len(rows) == 3 and len({r["hex_epoch_ms"] for r in rows}) == 1


def test_runs_record_since_ms(world):
    world.run()
    world.run()
    world.run("--since", "3d")
    values = [r[0] for r in world.db().execute("SELECT since_ms FROM runs ORDER BY run_id")]
    assert values[0] is None and values[1] is not None and values[2] is not None


def test_relative_hex_path_runs_that_file(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["collect", "--hex", "./hex"]) == 0
    assert len(clips(world)) == 3


def test_future_stamped_entry_does_not_hide_later_ones(world):
    now = int(__import__("time").time() * 1000)
    day = 24 * 3600 * 1000

    def entry(i, ts):
        return {"id": i, "timestamp_ms": ts, "kind": "dictation", "final_text": f"t{i}", "raw_text": f"t{i}"}

    world.doc["entries"] = [entry(1, now + 5 * day)]
    world.save()
    world.run()
    world.doc["entries"].append(entry(2, now))
    world.save()
    world.run()
    assert sorted(r["hex_id"] for r in clips(world)) == [1, 2]


def test_unexpected_error_is_recorded_not_masked(world):
    del world.doc["entries"][0]["kind"]
    world.save()
    with pytest.raises(KeyError):
        world.run()
    row = world.db().execute("SELECT error FROM runs").fetchone()
    assert row["error"] == "KeyError"


def test_log_has_no_transcript_text(world, capsys, monkeypatch):
    """What launchd writes to collect.log is collect's and prune's output: counts only."""
    from healing import schedule

    texts = [t for e in world.doc["entries"] for t in (e["raw_text"], e["final_text"]) if t]
    assert texts
    monkeypatch.setenv("HEX_BIN", str(world.hex))
    assert schedule.run_once() == 0
    assert len(clips(world)) == 3  # the fixture really went through collect
    seen = capsys.readouterr()
    for text in texts:
        assert text not in seen.out + seen.err


def test_labels_follow_a_run_once_not_per_row(world, monkeypatch):
    from healing import labels

    calls = []
    real = labels.refresh
    monkeypatch.setattr(labels, "refresh", lambda store, ids=None: (calls.append(list(ids or [])), real(store, ids))[1])
    world.run()
    assert len(calls) == 1 and len(calls[0]) == 3  # one refresh, every inserted clip
    calls.clear()
    world.doc["entries"][0].update(raw_text="hello their")
    world.save()
    world.run()
    assert len(calls) == 1 and len(calls[0]) == 1  # one refresh, the one clip it changed


def _last_run(world):
    store = Store()
    try:
        return dict(store.db.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT 1").fetchone())
    finally:
        store.close()


def test_a_labelling_failure_still_writes_the_runs_row_and_is_raised(world, monkeypatch):
    from healing import labels

    monkeypatch.setattr(labels, "refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("derive broke")))
    with pytest.raises(RuntimeError, match="derive broke"):
        world.run()
    last = _last_run(world)
    assert last["seen"] == 3 and last["new"] == 3 and last["error"] == "labelling: RuntimeError"


def test_a_labelling_failure_does_not_mask_what_ended_the_run(world, monkeypatch):
    from healing import collect as collect_module
    from healing import labels
    from healing.collect import CollectError

    monkeypatch.setattr(labels, "refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("derive broke")))
    monkeypatch.setattr(collect_module, "_collect_pages", lambda *a, **k: (_ for _ in ()).throw(CollectError("hex is gone", 2)))
    store = Store()
    try:
        with pytest.raises(CollectError, match="hex is gone"):
            collect_module.collect(store, Path("hex"))
    finally:
        store.close()
    assert _last_run(world)["error"] == "hex is gone"
