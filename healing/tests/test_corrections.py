import json
from pathlib import Path

from healing import corrections
from healing.cli import main
from healing.corrections import find_edits, label_corrections, observation_report, reference_text
from healing.store import Store, clip_id

from test_collect import world  # noqa: F401  (the fake-hex fixture)


def test_maps_final_edit_to_raw():
    edits = find_edits("send it to kip", "Send it to Kip.", "Send it to Kit.")
    assert len(edits) == 1
    edit = edits[0]
    assert (edit.raw_from, edit.final_from, edit.to_text, edit.kind) == ("kip", "Kip", "Kit", "acoustic")
    assert reference_text("send it to kip", edits) == "send it to kit"


def test_rewrite_changes_no_reference():
    edits = find_edits("see you tomorrow", "See you tomorrow.", "See you Friday.")
    assert [e.kind for e in edits] == ["rewrite"]
    assert reference_text("see you tomorrow", edits) is None


def test_casing_only_is_dropped():
    assert find_edits("send it to kip", "Send it to Kip.", "send it to KIP") == []
    assert find_edits("kit", "Kit.", "kit") == []


def test_numbers_normalized():
    assert find_edits("meet at five", "Meet at 5.", "Meet at five.") == []
    assert find_edits("it costs twenty five", "It costs 25.", "It costs 25 dollars")[0].kind == "rewrite"


def test_replacement_span_unmapped():
    # A replacement made "open code" into "OpenCode": the edit lands on a word raw never had.
    edits = find_edits("use open code today", "Use OpenCode today.", "Use Open Code today.")
    assert [e.kind for e in edits] == ["unmapped"]
    assert edits[0].raw_from == ""


def test_reference_applies_only_acoustic():
    raw = "send it to kip tomorrow"
    edits = find_edits(raw, "Send it to Kip tomorrow.", "Send it to Kit Friday.")
    assert [e.kind for e in edits] == ["acoustic", "rewrite"]
    assert reference_text(raw, edits) == "send it to kit tomorrow"


def test_insertion_and_deletion_are_rewrites():
    assert [e.kind for e in find_edits("a b", "A b.", "A very b.")] == ["rewrite"]
    assert [e.kind for e in find_edits("hello big world", "Hello big world.", "Hello world.")] == ["rewrite"]


def test_reference_keeps_what_the_person_typed():
    raw = "i said can not meet at five"
    # Apostrophes and number spelling survive; only the raw text's lower case is followed.
    edit = corrections.Edit("kip", "Kip", "Can't", "acoustic", None, (6, 9))
    assert reference_text("it is kip", [edit]) == "it is can't"
    edits = find_edits("meet at fife", "Meet at fife.", "Meet at five.")
    assert [e.kind for e in edits] == ["acoustic"]
    assert reference_text("meet at fife", edits) == "meet at five"
    assert reference_text(raw, []) is None


def _entry(i, **extra):
    base = {"id": i, "timestamp_ms": 1760000000000 + i * 1000, "application": "Notes", "kind": "dictation",
            "raw_text": "send it to kip", "final_text": "Send it to Kip.", "model_id": "m"}
    return {**base, **extra}


def test_collect_stores_correction_and_label_corrections(world, capsys):  # noqa: F811
    world.doc["entries"] = [
        _entry(1, correction={"outcome": "edited", "observed_at_ms": 4200, "edited_text": "Send it to Kit."}),
        _entry(2, rewritten=True, correction={"outcome": "edited", "observed_at_ms": 9, "edited_text": "Send it to Kit."}),
        _entry(3, correction={"outcome": "lost", "reason": "span_not_found", "observed_at_ms": 5}),
        _entry(4, correction={"outcome": "unchanged", "observed_at_ms": 5}),
        _entry(5),
    ]
    world.save()
    assert world.run() == 0
    assert main(["label", "corrections"]) == 0
    out = capsys.readouterr().out
    assert "acoustic 1" in out
    db = world.db()
    rows = {r["hex_id"]: dict(r) for r in db.execute("SELECT * FROM clips")}
    assert rows[1]["label"] == "strong" and rows[1]["reference_text"] == "send it to kit"
    assert rows[2]["reference_text"] is None and rows[2]["label"] == "unlabelled"  # rewrite mode skipped
    assert rows[1]["classified_at_ms"] is not None and rows[2]["classified_at_ms"] is None
    assert rows[3]["correction_reason"] == "span_not_found" and rows[5]["correction_outcome"] is None
    assert db.execute("SELECT COUNT(*) FROM edits").fetchone()[0] == 1

    # Hex adds a correction to an entry later: collect updates the row and invalidates labels.
    world.doc["entries"][3]["correction"] = {"outcome": "edited", "observed_at_ms": 5, "edited_text": "Send it to Kit."}
    world.doc["entries"][0]["correction"]["edited_text"] = "Send it to Kip."
    world.save()
    assert world.run() == 0
    rows = {r["hex_id"]: dict(r) for r in world.db().execute("SELECT * FROM clips")}
    assert rows[4]["edited_text"] == "Send it to Kit." and rows[4]["reference_text"] is None
    assert rows[1]["label"] == "unlabelled" and rows[1]["reference_text"] is None
    assert main(["label", "corrections"]) == 0
    rows = {r["hex_id"]: dict(r) for r in world.db().execute("SELECT * FROM clips")}
    assert rows[4]["label"] == "strong" and rows[1]["label"] == "weak"  # edited back to what Hex pasted
    # Processed, nothing acoustic: weak, its reference what Hex heard, and not classified again.
    assert rows[1]["reference_text"] == "send it to kip" and rows[1]["classified_at_ms"] is not None
    assert label_corrections(Store())["clips"] == 0


def test_report_observation(world, capsys):  # noqa: F811
    world.doc["entries"] = [
        _entry(1, correction={"outcome": "edited", "observed_at_ms": 1, "edited_text": "x"}),
        _entry(2, correction={"outcome": "lost", "reason": "span_not_found", "observed_at_ms": 1}),
        _entry(3, application="Slack", correction={"outcome": "unchanged", "observed_at_ms": 1}),
    ]
    world.save()
    world.run()
    report = {r["application"]: r for r in observation_report(Store())}
    assert report["Notes"]["pastes"] == 2 and report["Notes"]["observed"] == 1
    assert report["Notes"]["span_not_found_rate"] == 0.5 and report["Notes"]["edited_rate"] == 1.0
    assert main(["report", "observation"]) == 0
    assert "Slack" in capsys.readouterr().out


def test_ten_thousand_rows_fast(world):  # noqa: F811
    import time

    store = Store()
    for i in range(10_000):
        store.db.execute(
            "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text,"
            " collected_at_ms, correction_outcome, edited_text) VALUES (?, ?, 1, ?, 'dictation', ?, ?, 1, 'edited', ?)",
            (clip_id(1, i), i, i, "send it to kip", "Send it to Kip.", "Send it to Kit."),
        )
    store.db.commit()
    started = time.time()
    counts = label_corrections(store)
    assert counts["acoustic"] == 10_000 and time.time() - started < 10


def test_v4_dataset_with_rows_migrates_to_the_current_schema(tmp_path):
    import sqlite3

    from healing import store as store_module

    root = tmp_path / "dataset"
    root.mkdir()
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(4):
        db.executescript(store_module.MIGRATIONS[step])
    db.execute("PRAGMA user_version = 4")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text, label,"
        " collected_at_ms) VALUES ('a', 1, 1, 10, 'dictation', 'send it to kip', 'Send it to Kip.', 'verified', 5)"
    )
    db.commit()
    db.close()

    store = Store(root)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION
    row = store.db.execute("SELECT * FROM clips").fetchone()
    assert (row["clip_id"], row["raw_text"], row["label"], row["collected_at_ms"]) == (
        "a", "send it to kip", "verified", 5
    )
    assert all(row[c] is None for c in (
        "correction_outcome", "correction_reason", "edited_text", "reference_text", "classified_at_ms"
    ))
    assert store.db.execute("SELECT COUNT(*) FROM edits").fetchone()[0] == 0
    # The migrated row takes a correction and classifies like any other.
    store.db.execute(
        "UPDATE clips SET correction_outcome = 'edited', edited_text = 'Send it to Kit.' WHERE clip_id = 'a'"
    )
    assert label_corrections(store)["acoustic"] == 1
    row = store.db.execute("SELECT label, acoustic_reference FROM clips").fetchone()
    assert (row["label"], row["acoustic_reference"]) == ("verified", "send it to kit")  # the label is not the classifier's


def test_v5_dataset_made_before_the_classified_marker_migrates_to_v6(tmp_path):
    import sqlite3

    from healing import store as store_module

    root = tmp_path / "dataset"
    root.mkdir()
    db = sqlite3.connect(root / "dataset.sqlite")
    for step in range(5):
        db.executescript(store_module.MIGRATIONS[step])
    assert "classified_at_ms" not in {r[1] for r in db.execute("PRAGMA table_info(clips)")}  # a v5 dataset
    db.execute("PRAGMA user_version = 5")
    db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, kind, raw_text, final_text,"
        " collected_at_ms, correction_outcome, edited_text, reference_text)"
        " VALUES ('a', 1, 1, 10, 'dictation', 'send it to kip', 'Send it to Kip.', 5, 'edited',"
        " 'Send it to Kit.', 'send it to kit')"
    )
    db.commit()
    db.close()

    store = Store(root)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == store_module.SCHEMA_VERSION
    row = store.db.execute("SELECT * FROM clips").fetchone()
    assert row["classified_at_ms"] is None and row["reference_text"] == "send it to kit"
    Store(root).close()  # opening again is a no-op
    assert label_corrections(store)["acoustic"] == 1  # the unmarked row classifies once
    assert label_corrections(store)["clips"] == 0


def test_an_entry_without_a_correction_keeps_the_stored_one(world):  # noqa: F811
    world.doc["entries"] = [
        _entry(1, correction={"outcome": "edited", "observed_at_ms": 4200, "edited_text": "Send it to Kit."}),
    ]
    world.save()
    assert world.run() == 0
    assert main(["label", "corrections"]) == 0
    # An older Hex saved the History file and dropped the field.
    del world.doc["entries"][0]["correction"]
    world.save()
    assert world.run() == 0
    row = dict(world.db().execute("SELECT * FROM clips").fetchone())
    assert (row["correction_outcome"], row["edited_text"]) == ("edited", "Send it to Kit.")
    assert row["label"] == "strong" and row["classified_at_ms"] is not None
    assert world.db().execute("SELECT COUNT(*) FROM edits").fetchone()[0] == 1


def test_a_replace_in_hex_drops_the_stored_correction(world):  # noqa: F811
    world.doc["entries"] = [
        _entry(1, correction={"outcome": "edited", "observed_at_ms": 1, "edited_text": "Send it to Kit."}),
    ]
    world.save()
    world.run()
    main(["label", "corrections"])
    # Hex replaced the text (and cleared its correction): the old edit is not this text's.
    entry = world.doc["entries"][0]
    entry.pop("correction")
    entry["raw_text"], entry["final_text"] = "send it to kit", "Send it to Kit."
    world.save()
    world.run()
    row = dict(world.db().execute("SELECT * FROM clips").fetchone())
    assert row["correction_outcome"] is None and row["edited_text"] is None
    assert row["label"] == "unlabelled" and row["classified_at_ms"] is None
    assert main(["label", "corrections"]) == 0
    assert world.db().execute("SELECT COUNT(*) FROM edits").fetchone()[0] == 0
