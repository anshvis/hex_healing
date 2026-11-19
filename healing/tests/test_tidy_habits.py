"""`healing tidy habits` (T8.2): which fillers and habits you accept removing."""

import json
import stat

import pytest
from dataset import BASE_MS, DAY, add_clip

from healing import tidy
from healing.cli import main
from healing.store import Store

NOW = BASE_MS + 400 * DAY

# The shape T8.5 writes (its `tidy_events` table): the dataset has no such table until then.
TIDY_EVENTS = "CREATE TABLE IF NOT EXISTS tidy_events (clip_id TEXT NOT NULL, kind TEXT NOT NULL, token TEXT NOT NULL, accepted INTEGER NOT NULL, source TEXT NOT NULL)"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    s = Store()
    yield s
    s.close()


_n = iter(range(1, 100_000))


def decide(store, token, accepted, skipped, *, age_days=1, kind="filler", source="rule"):
    """`accepted` accepted and `skipped` skipped decisions on `token`, each on its own clip `age_days` old."""
    store.db.execute(TIDY_EVENTS)
    for verdict in [1] * accepted + [0] * skipped:
        n = next(_n)
        cid = add_clip(store, n, raw=f"so {token} we ship", ts=NOW - int(age_days * DAY) + n, audio=False)
        store.db.execute("INSERT INTO tidy_events (clip_id, kind, token, accepted, source) VALUES (?,?,?,?,?)",
                         (cid, kind, token, verdict, source))
    store.db.commit()


def by_phrase(rows):
    return {row.phrase: row for row in rows}


def test_accept_rate_threshold(store):
    """AC-16: accepted 8 of 10 is a habit; 1 of 10 is not, and stops being suggested."""
    decide(store, "basically", 8, 2)
    decide(store, "actually", 1, 9)
    decide(store, "kind of", 7, 3)  # exactly 0.7
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    assert rows["basically"].status == "habit"
    assert rows["basically"].accept_rate == pytest.approx(0.8)
    assert (rows["basically"].shown, rows["basically"].accepted, rows["basically"].skipped) == (10, 8, 2)
    assert rows["actually"].status == "dropped"
    assert rows["kind of"].status == "habit"

    path = tidy.export(tidy.habit_table(store, now=NOW), now=NOW)
    exported = json.loads(path.read_text())
    phrases = [habit["phrase"] for habit in exported["habits"]]
    assert "basically" in phrases and "actually" not in phrases
    assert "actually" not in exported["trial"]
    # Ranked: the higher accept rate first.
    assert phrases.index("basically") < phrases.index("kind of")


def test_minimum_showings(store):
    """Under 5 showings a phrase is still on trial, never a habit, however often it was accepted."""
    decide(store, "basically", 4, 0)
    decide(store, "you know", 5, 0)
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    assert rows["basically"].status == "trial"
    assert rows["you know"].status == "habit"
    exported = json.loads(tidy.export(list(rows.values()), now=NOW).read_text())
    assert [habit["phrase"] for habit in exported["habits"]] == ["you know"]
    assert "basically" in exported["trial"]


def test_trial_drops_once_it_cannot_reach_the_rate(store):
    """A trial phrase skipped so often that even accepting every showing up to 5 couldn't reach 0.7 is dropped."""
    decide(store, "like", 0, 2)
    decide(store, "sort of", 0, 1)
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    assert rows["like"].status == "dropped"
    assert rows["sort of"].status == "trial"
    # A starter phrase never shown is on trial, so it can be learned at all.
    assert rows["i guess"].status == "trial"
    assert rows["i guess"].shown == 0


def test_decay_halves_old_decisions(store):
    """Decisions older than 90 days count half: rates follow what you do now."""
    decide(store, "basically", 0, 6, age_days=120)  # old skips: 3 weighted
    decide(store, "basically", 8, 0, age_days=10)
    decide(store, "actually", 10, 0, age_days=200)  # 5 weighted showings: still enough
    decide(store, "like", 8, 0, age_days=200)  # 4 weighted showings: back on trial
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    basically = rows["basically"]
    assert (basically.shown, basically.accepted, basically.skipped) == (11, 8, 3)
    assert basically.accept_rate == pytest.approx(8 / 11)
    assert basically.status == "habit"
    assert rows["actually"].shown == 5 and rows["actually"].status == "habit"
    assert rows["like"].shown == 4 and rows["like"].status == "trial"
    assert basically.last_seen_ms > NOW - 11 * DAY


def test_only_candidate_phrases_and_removal_kinds_count(store):
    """Self-corrections, repeats and misheard words are not habits; a model removal of a habit word is."""
    decide(store, "scratch that", 9, 0, kind="self_correction")
    decide(store, "like", 6, 0, kind="repeat")
    decide(store, "Basically,", 5, 0, kind="self_correction", source="model")
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    assert "scratch that" not in rows
    assert rows["like"].shown == 0
    assert rows["basically"].status == "habit"


def test_hand_removals_become_candidates(store):
    """A phrase you removed by hand after pastes (M2's edits) and often say joins the candidates."""
    for i in range(6):
        cid = add_clip(store, next(_n), raw="so right we ship it today", ts=NOW - DAY + i, audio=False,
                       outcome="edited", edited="So we ship it today.")
        if i < 3:
            store.db.execute("INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind) VALUES (?, 0, ?, ?, '', 'rewrite')",
                             (cid, "right", "right,"))
    # Removed by hand twice only: not a habit candidate.
    for i in range(2):
        cid = add_clip(store, next(_n), raw="the plan is final", ts=NOW - DAY + 10 + i, audio=False)
        store.db.execute("INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind) VALUES (?, 0, 'final', 'final', '', 'rewrite')", (cid,))
    store.db.commit()
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    assert rows["right"].status == "trial"
    assert "final" not in rows


def test_no_decisions_yet_is_not_an_error(store):
    """Before T8.5 writes decisions there is no `tidy_events` table: starters on trial, no habits."""
    rows = tidy.habit_table(store, now=NOW)
    assert {row.phrase for row in rows} == set(tidy.STARTERS)
    assert all(row.status == "trial" for row in rows)
    exported = json.loads(tidy.export(rows, now=NOW).read_text())
    assert exported["habits"] == []


def test_export_is_atomic_owner_only_and_bounded(store, tmp_path):
    decide(store, "basically", 8, 2)
    path = tidy.export(tidy.habit_table(store, now=NOW), now=NOW)
    assert path == tmp_path / "home" / "tidy" / "habits.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not [p for p in path.parent.iterdir() if p.name != "habits.json"]
    exported = json.loads(path.read_text())
    assert exported["version"] == 1
    assert exported["habits"] == [{"phrase": "basically", "accept_rate": 0.8}]
    # Never more than Hex reads, and never a phrase Hex would refuse.
    many = [tidy.HabitRow(f"word{chr(97 + i // 26)}{chr(97 + i % 26)}", 10, 10, 0, NOW, 1.0, "habit") for i in range(100)]
    many.append(tidy.HabitRow("x" * 40, 10, 10, 0, NOW, 1.0, "habit"))
    exported = json.loads(tidy.export(many, now=NOW).read_text())
    assert len(exported["habits"]) + len(exported["trial"]) <= tidy.MAX_PHRASES
    assert all(len(habit["phrase"].encode()) <= tidy.MAX_PHRASE_BYTES for habit in exported["habits"])


def test_cli_prints_and_writes(store, capsys):
    decide(store, "basically", 8, 2)
    store.close()
    assert main(["tidy", "habits", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    row = next(r for r in printed["phrases"] if r["phrase"] == "basically")
    assert set(row) >= {"phrase", "shown", "accepted", "skipped", "last_seen_ms", "accept_rate", "status"}
    assert printed["file"].endswith("tidy/habits.json")
    assert main(["tidy", "habits"]) == 0
    assert "basically" in capsys.readouterr().out


def test_ah_is_learnable_since_tidy_does_not_suggest_it_built_in(store):
    """Only Hex's own rule fillers are excluded; "ah" isn't one of them (`tidy/rules.rs` FILLERS)."""
    for i in range(5):
        cid = add_clip(store, next(_n), raw="ah we ship today", ts=NOW - DAY + 20 + i, audio=False,
                       outcome="edited", edited="We ship today.")
        if i < 3:
            store.db.execute("INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind) VALUES (?, 0, 'ah', 'Ah,', '', 'rewrite')",
                             (cid,))
    store.db.commit()
    rows = by_phrase(tidy.habit_table(store, now=NOW))
    assert rows["ah"].status == "trial"
    assert not tidy.BUILT_IN_FILLERS & {"ah"}
