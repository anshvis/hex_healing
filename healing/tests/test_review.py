import pytest
from dataset import BASE_MS, DAY, add_clip, row

from healing import labels, review, splits
from healing.cli import main
from healing.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    s = Store()
    yield s
    s.close()


class Script:
    """A stand-in for the terminal and the speaker."""

    def __init__(self, *answers):
        self.answers, self.played, self.prompts = list(answers), [], []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def play(self, path):
        assert path.is_file()
        self.played.append(path)


def drive(store, script, **kwargs):
    return review.run(store, player=script.play, ask=script.ask, out=lambda *_: None, seed=1, **kwargs)


def test_enter_accepts_typing_replaces_r_rejects(store):
    ids = [add_clip(store, n, raw=f"clip number {n} says hello there") for n in (1, 2, 3)]
    script = Script("", "", "")
    drive(store, script, n=1)
    first = store.db.execute("SELECT * FROM reviews").fetchone()
    assert first["action"] == "accept" and first["strategy"] == "random"
    done = first["clip_id"]
    assert row(store, done)["label"] == "verified" and row(store, done)["weight"] == 1.0

    script = Script("typed text here", "r")
    result = drive(store, script, n=2)
    actions = sorted(r["action"] for r in store.db.execute("SELECT * FROM reviews"))
    assert actions == ["accept", "fix", "reject"]
    assert (result.fixed, result.rejected) == (1, 1)
    fixed = store.db.execute("SELECT * FROM reviews WHERE action = 'fix'").fetchone()
    assert fixed["text"] == "typed text here" and row(store, fixed["clip_id"])["reference_text"] == "typed text here"
    rejected = store.db.execute("SELECT * FROM reviews WHERE action = 'reject'").fetchone()
    assert rejected["text"] is None
    assert row(store, rejected["clip_id"])["label"] == "unlabelled"
    assert sorted([done, fixed["clip_id"], rejected["clip_id"]]) == sorted(ids)


def test_a_reviewed_clip_is_not_shown_again(store):
    add_clip(store, 1)
    drive(store, Script(""), n=5)
    script = Script()
    result = drive(store, script, n=5)
    assert result.shown == 0 and script.prompts == []


def test_replay_and_quit_keep_what_was_saved(store):
    for n in (1, 2, 3):
        add_clip(store, n)
    script = Script("p", "p", "", "q")
    result = drive(store, script, n=3)
    assert len(script.played) == 4  # shown once, replayed twice, second clip once
    assert store.db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
    assert result.stopped


def test_end_of_input_stops_cleanly(store):
    add_clip(store, 1)
    result = drive(store, Script(), n=1)
    assert result.stopped and store.db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_strong_clip_shows_its_correction_and_accepting_verifies_it(store):
    cid = add_clip(store, 1, outcome="edited", edited="Send it to Kit.")
    labels.label_all(store)
    drive(store, Script(""), n=1)
    assert row(store, cid)["label"] == "verified" and row(store, cid)["reference_text"] == "send it to kit"


def test_uncertain_takes_newest_weak_only_and_never_feeds_gating(store):
    old = add_clip(store, 1, outcome="unchanged")
    new = add_clip(store, 2, outcome="unchanged")
    other = add_clip(store, 3)
    labels.label_all(store)
    drive(store, Script(""), n=1, strategy="uncertain")
    assert row(store, new)["label"] == "verified" and row(store, new)["review_strategy"] == "uncertain"
    assert row(store, old)["label"] == "weak" and row(store, other)["label"] == "unlabelled"


def test_since_cutoff_takes_only_clips_of_later_sessions(store):
    cutoff = BASE_MS
    before = add_clip(store, 1, ts=cutoff - 3 * DAY)
    straddle_a = add_clip(store, 2, ts=cutoff - 1000)
    straddle_b = add_clip(store, 3, ts=cutoff + 1000)  # same session as the one before the cut-off
    after = add_clip(store, 4, ts=cutoff + DAY)
    script = Script("", "", "")
    result = drive(store, script, n=10, since_cutoff=True, cutoff_ms=cutoff)
    assert result.shown == 1
    assert [r["clip_id"] for r in store.db.execute("SELECT clip_id FROM reviews")] == [after]
    assert before and straddle_a and straddle_b


def test_since_cutoff_skips_clips_near_a_golden_one(store):
    cutoff = BASE_MS
    gold = add_clip(store, 1, ts=cutoff - DAY, raw="send the quarterly report to kit before friday", golden=1)
    twin = add_clip(store, 2, ts=cutoff + DAY, raw="send the quarterly report to kit before friday")
    free = add_clip(store, 3, ts=cutoff + 2 * DAY, raw="what is the weather in berlin tomorrow")
    store.db.execute("INSERT INTO splits (split_id, kind, created_at_ms) VALUES ('golden-v1', 'golden', 1)")
    store.db.execute("INSERT INTO split_members VALUES ('golden-v1', ?)", (gold,))
    store.db.commit()
    drive(store, Script("", ""), n=10, since_cutoff=True, cutoff_ms=cutoff)
    assert [r["clip_id"] for r in store.db.execute("SELECT clip_id FROM reviews")] == [free]
    assert twin


def test_since_cutoff_with_uncertain_is_refused(store):
    with pytest.raises(review.ReviewError, match="random"):
        drive(store, Script(), strategy="uncertain", since_cutoff=True)


def test_the_command_reads_stdin(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    s = Store()
    add_clip(s, 1)
    s.close()
    monkeypatch.setattr(review, "afplay", lambda path: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(review, "run", lambda *a, **k: review.ReviewResult(available=1, accepted=1))
    assert main(["review", "--n", "1"]) == 0
    assert "accepted 1" in capsys.readouterr().out


def test_an_uncertain_review_does_not_shrink_the_random_frame(store):
    a = add_clip(store, 1, raw="clip number 1 says hello there", outcome="unchanged")
    b = add_clip(store, 2, raw="clip number 2 says hello there")
    labels_refresh = __import__("healing.labels", fromlist=["refresh"]).refresh
    labels_refresh(store)
    drive(store, Script(""), n=1, strategy="uncertain")
    assert {r["clip_id"] for r in review.candidates(store, "random")} == {a, b}
    store.db.execute("INSERT INTO reviews (clip_id, reviewed_at_ms, action, text, strategy) VALUES (?, 1, 'reject', NULL, 'random')", (b,))
    assert {r["clip_id"] for r in review.candidates(store, "random")} == {a}
