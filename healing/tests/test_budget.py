import numpy as np
import pytest

from healing import dedupe
from healing.budget import PruneError, parse_size, prune
from healing.cli import main
from healing.store import Pcm, Store, clip_id
import hashlib

DAY = 24 * 3600 * 1000
NOW = 400 * DAY
PHRASE = "please send the quarterly report to kit before the end of the day"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    store = Store()
    yield store
    store.close()


def tone(seed: int, freq: float, seconds=1.0) -> Pcm:
    rng = np.random.RandomState(seed)
    t = np.arange(int(16_000 * seconds)) / 16_000
    wave = 0.3 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.randn(t.size)
    data = (wave * 32767).astype(np.int16)
    return Pcm(hashlib.sha256(data.tobytes()).hexdigest(), data.size, data)


def add(store, n, text, freq, age_days=30, label="weak", golden=0):
    pcm = tone(n, freq)
    store.write_audio(pcm)
    store.db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text,"
        " audio_sha256, label, golden, audio_ms, collected_at_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (clip_id(1, n), n, 1, NOW - int(age_days * DAY) + n, "dictation", text, text,
         pcm.sha256, label, golden, 1000, NOW),
    )
    store.db.commit()
    return clip_id(1, n)


def live(store):
    return {r[0] for r in store.db.execute(
        "SELECT c.clip_id FROM clips c JOIN audio a ON a.sha256=c.audio_sha256 WHERE a.evicted_at_ms IS NULL")}


def files(store):
    return list((store.root / "audio").glob("*/*.flac"))


def fixture_50(store):
    repeats = [add(store, i, PHRASE, 150 + 90 * i, label="strong" if i < 2 else "weak") for i in range(20)]
    uniques = [add(store, 100 + i, f"unique sentence number {i} about topic {i * 3}", 300 + 40 * i) for i in range(30)]
    return repeats, uniques


def test_k_per_cluster_keeps_spread(store):
    groups = {200: [], 800: [], 1600: []}
    n = 0
    for freq, count in ((200, 3), (800, 2), (1600, 1)):
        for _ in range(count):
            groups[freq].append(add(store, n, PHRASE, freq))
            n += 1
    prune(store, k=3, now=NOW)
    kept = live(store)
    assert len(kept) == 3
    assert {f for f, ids in groups.items() for i in ids if i in kept} == {200, 800, 1600}
    gone = store.db.execute("SELECT COUNT(*) FROM audio WHERE evicted_at_ms IS NOT NULL").fetchone()[0]
    assert gone == 3
    assert store.db.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 6
    assert len(files(store)) == 3
    row = store.db.execute("SELECT size, audio_kept FROM clusters").fetchone()
    assert (row["size"], row["audio_kept"]) == (6, 3)


def test_ac4_k_keeps_three_plus_protected_and_all_unique(store):
    repeats, uniques = fixture_50(store)
    result = prune(store, k=3, now=NOW)
    kept = live(store)
    assert len([r for r in repeats if r in kept]) == 5
    assert set(repeats[:2]) <= kept
    assert set(uniques) <= kept
    assert len(result.evictions) == 15
    assert prune(store, k=3, now=NOW).evictions == []  # idempotent


def test_protected_never_evicted_even_under_tight_budget(store):
    repeats, uniques = fixture_50(store)
    size = max(f.stat().st_size for f in files(store))
    prune(store, budget=size * 25, k=3, now=NOW)
    kept = live(store)
    assert set(repeats[:2]) <= kept


def test_budget_evicts_redundant_oldest_first(store):
    repeats, uniques = fixture_50(store)
    prune(store, k=3, now=NOW)
    live_bytes = sum(f.stat().st_size for f in files(store))
    budget = live_bytes - 2 * max(f.stat().st_size for f in files(store))
    result = prune(store, budget=budget, k=3, now=NOW)
    assert 2 <= len(result.evictions) <= 3
    assert result.total_bytes <= budget
    weak_repeats = set(repeats[2:])
    assert all(e.reason == "budget" and e.label == "weak" for e in result.evictions)
    assert {e.clip_id for e in result.evictions} <= weak_repeats  # redundant before unique
    assert set(repeats[:2]) <= live(store) and set(uniques) <= live(store)


def test_budget_ties_go_oldest_first(store):
    ids = [add(store, i, f"distinct sentence {i} about unrelated matter {i * 5}", 300 + 40 * i,
               age_days=100 - i) for i in range(6)]
    size = max(f.stat().st_size for f in files(store))
    prune(store, budget=size * 5, k=3, now=NOW)
    assert ids[0] not in live(store) and set(ids[1:]) <= live(store)


def test_budget_of_25_files_ends_under_budget(store):
    repeats, uniques = fixture_50(store)
    sizes = [f.stat().st_size for f in files(store)]
    budget = max(sizes) * 25
    # protected (2 files) fits; evicted order: redundant cluster first, then oldest uniques
    result = prune(store, budget=budget, k=3, now=NOW)
    assert result.total_bytes <= budget
    assert set(repeats[:2]) <= live(store)
    assert sum(f.stat().st_size for f in files(store)) <= budget


def test_new_unique_clips_admitted(store):
    fixture_50(store)
    prune(store, k=3, now=NOW)
    new = [add(store, 200 + i, f"brand new topic {i} with distinct words {i * 11}", 500 + 30 * i, age_days=0) for i in range(10)]
    prune(store, k=3, now=NOW)
    assert set(new) <= live(store)
    assert store.db.execute("SELECT COUNT(*) FROM clips WHERE cluster_id IS NULL").fetchone()[0] == 0


def test_under_seven_days_never_evicted(store):
    ids = [add(store, i, PHRASE, 200 + 100 * i, age_days=2) for i in range(6)]
    prune(store, k=1, now=NOW)
    assert live(store) == set(ids)
    size = max(f.stat().st_size for f in files(store))
    with pytest.raises(PruneError) as error:
        prune(store, budget=size * 2, k=1, now=NOW)
    assert error.value.code == 4
    assert live(store) == set(ids)


def test_protected_over_budget_exits_4_and_evicts_nothing(store):
    ids = [add(store, i, f"strong clip {i} spoken words here", 200 + 50 * i, label="verified") for i in range(3)]
    weak = add(store, 9, PHRASE, 700)
    size = max(f.stat().st_size for f in files(store))
    with pytest.raises(PruneError) as error:
        prune(store, budget=size * 2, k=3, now=NOW)
    assert error.value.code == 4 and "over the budget" in str(error.value)
    assert live(store) == set(ids) | {weak}
    assert len(files(store)) == 4


def test_budget_smaller_than_one_file_exits_4(store):
    add(store, 1, PHRASE, 300)
    with pytest.raises(PruneError) as error:
        prune(store, budget=10, now=NOW)
    assert error.value.code == 4


def test_dry_run_changes_nothing(store):
    fixture_50(store)
    before = live(store)
    result = prune(store, k=3, dry_run=True, now=NOW)
    assert len(result.evictions) == 15
    assert live(store) == before and len(files(store)) == 50
    assert store.db.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 0


def test_golden_is_protected(store):
    ids = [add(store, i, PHRASE, 200 + 100 * i, golden=1 if i < 4 else 0) for i in range(8)]
    prune(store, k=1, now=NOW)
    assert set(ids[:4]) <= live(store)
    assert len(live(store)) == 5


def test_cli_prune(store, capsys, tmp_path):
    fixture_50(store)
    store.close()
    assert main(["prune", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "evict " in out and PHRASE not in out
    assert main(["prune", "--budget", "1"]) == 4
    assert main(["prune"]) == 0
    assert "evicted 15" in capsys.readouterr().out
    Store().close()  # reopen for the fixture's teardown


def test_parse_size():
    assert parse_size("5GiB") == 5 * 1024**3
    assert parse_size("10") == 10
    assert parse_size("2k") == 2048
