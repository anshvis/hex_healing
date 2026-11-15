import json
import sqlite3

import pytest
from dataset import add_clip, review, row

from healing import labels, store as store_module
from healing.cli import main
from healing.store import MIGRATIONS, Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    s = Store()
    yield s
    s.close()


def label_of(store, cid):
    r = row(store, cid)
    return r["label"], r["weight"]


def test_fixture_transitions_and_weights(store):
    unchanged = add_clip(store, 1, outcome="unchanged")
    strong = add_clip(store, 2, outcome="edited", edited="Send it to Kit.")
    lost = add_clip(store, 3, outcome="lost")
    never = add_clip(store, 4)
    verified = add_clip(store, 5)
    review(store, verified, "fix", "send it to kit")
    assert labels.label_all(store)["verified"] == 1
    assert label_of(store, unchanged) == ("weak", 0.2)
    assert label_of(store, strong) == ("strong", 1.0)
    assert label_of(store, lost) == ("unlabelled", 0)
    assert label_of(store, never) == ("unlabelled", 0)
    assert label_of(store, verified) == ("verified", 1.0)
    assert row(store, strong)["reference_text"] == "send it to kit"
    assert json.loads(row(store, strong)["edited_positions"]) == [[3, 4]]
    assert row(store, unchanged)["reference_text"] == "send it to kip"
    assert row(store, verified)["reference_text"] == "send it to kit"
    assert row(store, verified)["review_strategy"] == "random"


def test_label_all_is_idempotent(store):
    add_clip(store, 1, outcome="unchanged")
    add_clip(store, 2, outcome="edited", edited="Send it to Kit.")
    labels.label_all(store)
    first = [tuple(r) for r in store.db.execute("SELECT * FROM clips ORDER BY clip_id")]
    labels.label_all(store)
    assert [tuple(r) for r in store.db.execute("SELECT * FROM clips ORDER BY clip_id")] == first


def test_transitions_never_downgrade(store):
    # unlabelled -> weak -> strong, as the evidence arrives
    cid = add_clip(store, 1, outcome="unchanged")
    labels.label_all(store)
    assert label_of(store, cid) == ("weak", 0.2)
    # what `collect` does when Hex reports a new correction: store it and clear the stale reference
    store.db.execute("UPDATE clips SET correction_outcome = 'edited', edited_text = 'Send it to Kit.',"
                     " reference_text = NULL WHERE clip_id = ?", (cid,))
    labels.label_all(store)
    assert label_of(store, cid) == ("strong", 1.0)
    # derived labels follow their evidence (collect demotes a changed correction before this runs)
    store.db.execute("UPDATE clips SET correction_outcome = 'unchanged' WHERE clip_id = ?", (cid,))
    labels.label_all(store)
    assert label_of(store, cid) == ("weak", 0.2)
    # any -> verified, and verified is never downgraded by the corrections
    review(store, cid, "accept", "send it to kit")
    labels.label_all(store)
    assert label_of(store, cid) == ("verified", 1.0)
    store.db.execute("UPDATE clips SET correction_outcome = 'lost' WHERE clip_id = ?", (cid,))
    labels.label_all(store)
    assert row(store, cid)["label"] == "verified" and row(store, cid)["reference_text"] == "send it to kit"
    # only an explicit rejection lowers it
    review(store, cid, "reject", at=2_000_000_000_000)
    labels.label_all(store)
    assert label_of(store, cid) == ("unlabelled", 0)


def test_a_later_review_replaces_an_earlier_one(store):
    cid = add_clip(store, 1)
    review(store, cid, "reject", at=1)
    review(store, cid, "fix", "send it to kit", at=2)
    labels.label_all(store)
    assert row(store, cid)["label"] == "verified"


def test_rewrite_only_stays_weak(store):
    cid = add_clip(store, 1, raw="see you tomorrow", final="See you tomorrow.", outcome="edited", edited="See you Friday.")
    labels.label_all(store)
    assert label_of(store, cid) == ("weak", 0.2)
    assert row(store, cid)["reference_text"] == "see you tomorrow"
    assert row(store, cid)["edited_positions"] is None


def test_unmapped_is_unlabelled(store):
    cid = add_clip(store, 1, raw="use open code today", final="Use OpenCode today.", outcome="edited", edited="Use Open Code today.")
    labels.label_all(store)
    assert label_of(store, cid) == ("unlabelled", 0)


def test_an_opencode_rewrite_is_never_trusted(store):
    cid = add_clip(store, 1, outcome="unchanged", rewritten=1)
    labels.label_all(store)
    assert label_of(store, cid) == ("unlabelled", 0)


def test_rejected_never_trainable(store):
    good = add_clip(store, 1, outcome="edited", edited="Send it to Kit.")
    bad = add_clip(store, 2, outcome="edited", edited="Send it to Kit.")
    review(store, bad, "reject")
    labels.label_all(store)
    assert [r["clip_id"] for r in labels.trainable(store)] == [good]


def test_golden_never_trainable(store):
    kept = add_clip(store, 1, outcome="edited", edited="Send it to Kit.")
    gold = add_clip(store, 2, outcome="edited", edited="Send it to Kit.", golden=1)
    labels.label_all(store)
    assert [r["clip_id"] for r in labels.trainable(store)] == [kept]
    assert gold not in {r["clip_id"] for r in labels.trainable(store, include_weak=True)}


def test_weak_excluded_by_default(store):
    weak = add_clip(store, 1, outcome="unchanged")
    strong = add_clip(store, 2, outcome="edited", edited="Send it to Kit.")
    labels.label_all(store)
    assert {r["clip_id"] for r in labels.trainable(store)} == {strong}
    assert {r["clip_id"] for r in labels.trainable(store, include_weak=True)} == {weak, strong}


def test_no_audio_and_non_dictation_are_not_trainable(store):
    add_clip(store, 1, outcome="edited", edited="Send it to Kit.", audio=False)
    add_clip(store, 2, outcome="edited", edited="Send it to Kit.", kind="raw")
    evicted = add_clip(store, 3, outcome="edited", edited="Send it to Kit.")
    store.db.execute("UPDATE audio SET evicted_at_ms = 1 WHERE sha256 = (SELECT audio_sha256 FROM clips WHERE clip_id = ?)", (evicted,))
    labels.label_all(store)
    assert labels.trainable(store) == []
    assert label_of(store, evicted) == ("strong", 1.0)  # still counts for Tier 1


def test_model_condition(store):
    weak_other = add_clip(store, 1, outcome="unchanged", model_id="old")
    weak_same = add_clip(store, 2, outcome="unchanged", model_id="m")
    weak_none = add_clip(store, 3, outcome="unchanged", model_id=None)
    strong_other = add_clip(store, 4, outcome="edited", edited="Send it to Kit.", model_id="old")
    strong_none = add_clip(store, 5, outcome="edited", edited="Send it to Kit.", model_id=None)
    labels.label_all(store)
    got = {r["clip_id"] for r in labels.trainable(store, model_id="m", include_weak=True)}
    # a human reference is model-independent; a weak one is the model's own output
    assert got == {weak_same, strong_other, strong_none}
    assert weak_other not in got and weak_none not in got


def test_label_all_command(store, capsys):
    add_clip(store, 1, outcome="unchanged")
    store.close()
    assert main(["label", "all"]) == 0
    assert capsys.readouterr().out.strip() == "strong 0, verified 0, weak 1, unlabelled 0"


def test_a_v5_dataset_migrates(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HEALING_HOME", str(home))
    root = home / "dataset"
    (root / "audio").mkdir(parents=True)
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(5):
        db.executescript(MIGRATIONS[step])
    db.execute("PRAGMA user_version = 5")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, collected_at_ms)"
        " VALUES ('c1', 1, 1, 1, 'dictation', 'a b c', 'A b c.', 1)"
    )
    db.commit()
    db.close()
    migrated = Store()
    columns = {r[1] for r in migrated.db.execute("PRAGMA table_info(clips)")}
    assert {"weight", "edited_positions", "review_strategy"} <= columns
    tables = {r[0] for r in migrated.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"reviews", "splits", "split_members"} <= tables
    assert migrated.db.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION == 7
    kept = migrated.db.execute("SELECT raw_text, weight, label FROM clips").fetchone()
    assert tuple(kept) == ("a b c", 0, "unlabelled")
    # idempotent: a run that stopped part-way finishes on the next open
    migrated.db.execute("DROP TABLE split_members")
    migrated.db.execute("PRAGMA user_version = 6")
    migrated.db.commit()
    migrated.close()
    again = Store()
    assert again.db.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION
    assert again.db.execute("SELECT COUNT(*) FROM split_members").fetchone()[0] == 0
    again.close()


def test_acoustic_and_unmapped_together_stay_unlabelled(store):
    cid = add_clip(store, 1, outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to Kit.' WHERE clip_id = ?", (cid,))
    from healing import corrections

    corrections.label_corrections(store)
    store.db.execute(
        "INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind) VALUES (?, 9, '', 'x', 'y', 'unmapped')",
        (cid,),
    )
    labels.refresh(store)
    assert label_of(store, cid) == ("unlabelled", 0)


def test_label_corrections_stores_positions_and_never_the_label(store):
    from healing import corrections

    cid = add_clip(store, 1, outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to Kit.' WHERE clip_id = ?", (cid,))
    corrections.label_corrections(store)
    r = row(store, cid)
    assert r["label"] == "unlabelled" and r["reference_text"] == "send it to kit" and r["edited_positions"] == "[[3, 4]]"
    labels.refresh(store)
    assert label_of(store, cid) == ("strong", 1.0)


def test_frozen_member_updates_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    store = Store()
    store.db.execute("INSERT INTO splits (split_id, kind, created_at_ms, frozen) VALUES ('g', 'golden', 1, 1)")
    store.db.execute("INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, collected_at_ms)"
                     " VALUES ('c1', 1, 1, 1, 'dictation', 'a', 'A', 1)")
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("INSERT OR REPLACE INTO split_members (split_id, clip_id) VALUES ('g', 'c1')")
    store.close()


def test_a_rejected_clip_loses_its_reference(store):
    cid = add_clip(store, 1, outcome="unchanged")
    labels.label_all(store)
    assert row(store, cid)["reference_text"] == "send it to kip"
    review(store, cid, "reject")
    labels.label_all(store)
    assert label_of(store, cid) == ("unlabelled", 0) and row(store, cid)["reference_text"] is None


@pytest.mark.parametrize("start", [5, 6])
def test_datasets_at_each_earlier_version_migrate_to_current(tmp_path, monkeypatch, start):
    """A v5 and a v6 dataset (whatever the base put at 6) both reach the current schema with their rows."""
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    root = tmp_path / "home" / "dataset"
    (root / "audio").mkdir(parents=True)
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(start):
        migration = MIGRATIONS[step]
        migration(db) if callable(migration) else db.executescript(migration)
    db.execute(f"PRAGMA user_version = {start}")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, collected_at_ms)"
        " VALUES ('c1', 1, 1, 1, 'dictation', 'a b c', 'A b c.', 1)"
    )
    db.commit()
    db.close()
    store = Store()
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION
    columns = {r[1] for r in store.db.execute("PRAGMA table_info(clips)")}
    assert {"classified_at_ms", "weight", "edited_positions", "review_strategy"} <= columns
    assert "sidecar_sha256" in {r[1] for r in store.db.execute("PRAGMA table_info(splits)")}
    assert store.db.execute("SELECT raw_text FROM clips").fetchone()[0] == "a b c"
    store.close()


def test_label_all_backfills_positions_for_clips_classified_before_they_existed(store):
    from healing import corrections

    cid = add_clip(store, 1, outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to Kit.' WHERE clip_id = ?", (cid,))
    corrections.label_corrections(store)
    labels.refresh(store)
    # a dataset migrated from v5/v6: classified (reference set), but no edited_positions column value
    store.db.execute("UPDATE clips SET edited_positions = NULL WHERE clip_id = ?", (cid,))
    labels.label_all(store)
    assert row(store, cid)["edited_positions"] == "[[3, 4]]" and label_of(store, cid) == ("strong", 1.0)
    before = row(store, cid)["classified_at_ms"]
    labels.label_all(store)  # settled: nothing is classified again
    assert row(store, cid)["classified_at_ms"] == before
