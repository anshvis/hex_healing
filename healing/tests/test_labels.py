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
    assert r["label"] == "unlabelled" and r["acoustic_reference"] == "send it to kit" and r["edited_positions"] == "[[3, 4]]"
    assert r["reference_text"] is None  # derived by refresh, not written by the classifier
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


def _snapshot(store):
    return [tuple(r) for r in store.db.execute(
        "SELECT clip_id, label, weight, reference_text, acoustic_reference, edited_positions, classified_at_ms,"
        " review_strategy FROM clips ORDER BY clip_id")]


def test_label_all_twice_changes_nothing(store):
    from healing import corrections

    strong = add_clip(store, 1, outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to Kit.' WHERE clip_id = ?", (strong,))
    verified_edited = add_clip(store, 2, raw="send it to kip please", final="Send it to kip please.", outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to Kit please.' WHERE clip_id = ?", (verified_edited,))
    rewrite_only = add_clip(store, 3, raw="send it to kip", final="Send it to kip.", outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to kip, and then go home.' WHERE clip_id = ?", (rewrite_only,))
    store.db.commit()
    labels.label_all(store)
    review(store, verified_edited, "fix", "send it to kit please")
    labels.label_all(store)
    assert label_of(store, strong)[0] == "strong" and label_of(store, verified_edited)[0] == "verified"
    assert label_of(store, rewrite_only)[0] in ("weak", "unlabelled")
    first = _snapshot(store)
    labels.label_all(store)
    assert _snapshot(store) == first
    assert corrections.label_corrections(store)["clips"] == 0  # nothing is classified again


def test_the_migration_clears_classifications_that_lack_the_classifiers_own_output(tmp_path, monkeypatch):
    """A v6 dataset (classified before `acoustic_reference` and positions) re-classifies once, and
    its strong clip reaches the edited set."""
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    root = tmp_path / "home" / "dataset"
    (root / "audio").mkdir(parents=True)
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(6):
        migration = MIGRATIONS[step]
        migration(db) if callable(migration) else db.executescript(migration)
    db.execute("PRAGMA user_version = 6")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, collected_at_ms,"
        " correction_outcome, edited_text, reference_text, classified_at_ms) VALUES"
        " ('c1', 1, 1, 1, 'dictation', 'send it to kip', 'Send it to kip.', 1, 'edited', 'Send it to Kit.',"
        " 'send it to kit', 5)"
    )
    db.execute("INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind) VALUES ('c1', 0, 'kip', 'kip', 'Kit', 'acoustic')")
    db.commit()
    db.close()
    store = Store()
    r = row(store, "c1")
    assert r["classified_at_ms"] is None and r["acoustic_reference"] is None
    labels.label_all(store)
    r = row(store, "c1")
    assert (r["label"], r["acoustic_reference"], r["edited_positions"]) == ("strong", "send it to kit", "[[3, 4]]")
    settled = _snapshot(store)
    labels.label_all(store)
    assert _snapshot(store) == settled


def test_an_existing_dataset_is_copied_before_it_migrates(tmp_path, monkeypatch):
    import stat

    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    root = tmp_path / "home" / "dataset"
    (root / "audio").mkdir(parents=True)
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(5):
        migration = MIGRATIONS[step]
        migration(db) if callable(migration) else db.executescript(migration)
    db.execute("PRAGMA user_version = 5")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, collected_at_ms)"
        " VALUES ('c1', 1, 1, 1, 'dictation', 'a b c', 'A b c.', 1)"
    )
    db.commit()
    db.close()
    Store().close()
    backup = root / "dataset.sqlite.pre-v5"
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    old = sqlite3.connect(backup)
    assert old.execute("PRAGMA user_version").fetchone()[0] == 5
    assert old.execute("SELECT raw_text FROM clips").fetchone()[0] == "a b c"
    assert "reviews" not in {r[0] for r in old.execute("SELECT name FROM sqlite_master")}
    old.close()
    Store().close()  # current: no new copy, and the first backup is untouched
    assert sorted(p.name for p in root.glob("dataset.sqlite.pre-*")) == ["dataset.sqlite.pre-v5"]


def test_a_fresh_dataset_makes_no_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    Store().close()
    assert not list((tmp_path / "home" / "dataset").glob("dataset.sqlite.pre-*"))


def _v5_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    root = tmp_path / "home" / "dataset"
    (root / "audio").mkdir(parents=True)
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(5):
        migration = MIGRATIONS[step]
        migration(db) if callable(migration) else db.executescript(migration)
    db.execute("PRAGMA user_version = 5")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, collected_at_ms)"
        " VALUES ('c1', 1, 1, 1, 'dictation', 'a b c', 'A b c.', 1)"
    )
    db.commit()
    db.close()
    return root


def test_an_empty_or_damaged_backup_is_made_again(tmp_path, monkeypatch):
    root = _v5_dataset(tmp_path, monkeypatch)
    (root / "dataset.sqlite.pre-v5").write_bytes(b"")  # an interrupted earlier attempt
    (root / "dataset.sqlite.pre-v5.tmp").write_bytes(b"half")  # and its leftover
    Store().close()
    backup = sqlite3.connect(root / "dataset.sqlite.pre-v5")
    assert backup.execute("PRAGMA user_version").fetchone()[0] == 5
    assert backup.execute("SELECT raw_text FROM clips").fetchone()[0] == "a b c"
    backup.close()
    assert not (root / "dataset.sqlite.pre-v5.tmp").exists()


def test_a_wrong_version_backup_is_not_trusted(tmp_path, monkeypatch):
    root = _v5_dataset(tmp_path, monkeypatch)
    other = sqlite3.connect(root / "dataset.sqlite.pre-v5")
    other.execute("CREATE TABLE x (y)")
    other.execute("PRAGMA user_version = 3")
    other.commit()
    other.close()
    Store().close()
    backup = sqlite3.connect(root / "dataset.sqlite.pre-v5")
    assert backup.execute("PRAGMA user_version").fetchone()[0] == 5
    backup.close()


def test_a_failed_backup_stops_the_migration_with_a_store_error(tmp_path, monkeypatch):
    root = _v5_dataset(tmp_path, monkeypatch)
    def boom(self, target, **kwargs):
        raise sqlite3.OperationalError("disk full")

    class Broken(Store):
        def _back_up(self, version):
            real = self.db

            class Wrapper:
                def backup(self, target):
                    boom(self, target)

            self.db = Wrapper()
            try:
                super()._back_up(version)
            finally:
                self.db = real

    from healing.store import StoreError

    with pytest.raises(StoreError, match="could not back up"):
        Broken()
    assert not (root / "dataset.sqlite.pre-v5").exists() and not (root / "dataset.sqlite.pre-v5.tmp").exists()
    check = sqlite3.connect(root / "dataset.sqlite")
    assert check.execute("PRAGMA user_version").fetchone()[0] == 5  # not migrated without a backup
    check.close()


def test_concurrent_first_opens_migrate_once(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    import time

    root = _v5_dataset(tmp_path, monkeypatch)
    go = tmp_path / "go"
    code = (
        "import os, time\nfrom healing.store import Store\n"
        f"while not os.path.exists({str(go)!r}): time.sleep(0.0005)\n"
        "s = Store(); print(s.db.execute('PRAGMA user_version').fetchone()[0]); s.close()"
    )
    env = {**os.environ, "HEALING_HOME": str(tmp_path / "home")}
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(24)]
    time.sleep(2.5)  # every process has imported and is waiting at the gate
    go.write_text("go")
    results = [(p.communicate(), p.returncode) for p in procs]
    assert all(rc == 0 for _, rc in results), [r[0][1][-200:] for r in results if r[1]][:2]
    assert {out.strip() for (out, _), _ in results} == {str(store_module.SCHEMA_VERSION)}
    assert sorted(p.name for p in root.glob("dataset.sqlite.pre-*")) == ["dataset.sqlite.pre-v5"]


def test_a_v7_dataset_missing_columns_is_repaired_on_open(tmp_path, monkeypatch):
    """An intermediate build stamped v7 before `acoustic_reference`, the sidecar pin and the snapshot
    detail existed: opening it repairs the file (and does not classify anything again)."""
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    Store().close()
    path = tmp_path / "home" / "dataset" / "dataset.sqlite"
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE clips DROP COLUMN acoustic_reference")
    db.execute("ALTER TABLE splits DROP COLUMN detail")
    db.execute("ALTER TABLE splits DROP COLUMN sidecar_sha256")
    db.commit()
    assert db.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION
    db.close()
    store = Store()
    assert "acoustic_reference" in {r[1] for r in store.db.execute("PRAGMA table_info(clips)")}
    assert {"detail", "sidecar_sha256"} <= {r[1] for r in store.db.execute("PRAGMA table_info(splits)")}
    assert store_module.ensure_m3_schema(store.db) is False  # nothing left to repair
    store.close()


def test_the_process_lock_and_the_lock_probe_share_one_flock(tmp_path):
    import os

    from healing.store import StoreError, lock_free, locked

    path = tmp_path / "x.lock"
    with locked(path, wait=False):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            assert lock_free(descriptor) is False
        finally:
            os.close(descriptor)
        with pytest.raises(StoreError, match="busy now"):
            with locked(path, wait=False, busy="busy now"):
                pass
    descriptor = os.open(path, os.O_RDONLY)
    try:
        assert lock_free(descriptor) is True
    finally:
        os.close(descriptor)


def test_refresh_labels_every_clip_past_the_chunk_size(store):
    ids = []
    for n in range(1, 531):  # more than one chunk of 500
        ids.append(add_clip(store, n, raw=f"clip {n} says hello there friend", outcome="unchanged", audio=False))
    counts = labels.refresh(store, ids)
    assert counts["weak"] == 530
    assert store.db.execute("SELECT COUNT(*) FROM clips WHERE label = 'weak' AND weight = 0.2").fetchone()[0] == 530
    assert labels.refresh(store, []) == {}  # none when empty


def test_refresh_never_touches_edited_positions(store):
    from healing import corrections

    cid = add_clip(store, 1, outcome="edited")
    store.db.execute("UPDATE clips SET edited_text = 'Send it to Kit.' WHERE clip_id = ?", (cid,))
    corrections.label_corrections(store)
    review(store, cid, "fix", "send it to kit")  # verified: not strong, but the classifier's positions stay
    labels.refresh(store)
    assert label_of(store, cid)[0] == "verified" and row(store, cid)["edited_positions"] == "[[3, 4]]"
