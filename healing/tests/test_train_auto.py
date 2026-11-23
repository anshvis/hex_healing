"""T6.4: `healing train auto`, the trigger (AC-12, D-33).

The dataset and the registry are real (temp `HEALING_HOME`); the heavy steps (prepare's bundle, the
LoRA round, the export and the eval) are a fake that records what ran. Minutes come from the stored
audio's sample counts, which the fixture sets directly.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
from dataset import BASE_MS, DAY, add_clip, review

from healing import labels, splits
from healing.registry import snapshot as snapshots
from healing.registry import store as registry
from healing.store import Store, locked
from healing.train import auto
from healing.train import run as train_run
from healing.train.manifest import TrainError

NOW = BASE_MS + 60 * DAY  # every fixture clip is well outside the 14-day test window
RATE = 16_000


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))  # never the real one
    registry.append({"event": "registered", "version": "v0", "parent": None, "dataset": None,
                     "artifacts": {"gguf": None}}, {"event": "promoted", "version": "v0", "reason": "base"})
    return tmp_path / "home"


@pytest.fixture
def store(home):
    s = Store()
    yield s
    s.close()


_next = [0]


def speech(store, minutes: float, clips: int = 1, **kw) -> list[str]:
    """`clips` verified clips holding `minutes` of audio between them (by their stored sample count)."""
    ids = []
    for _ in range(clips):
        _next[0] += 1
        n = _next[0]
        cid = add_clip(store, n, raw=f"phrase number {n} about the thing {n * 7}", **kw)
        review(store, cid, "accept")
        store.db.execute("UPDATE audio SET samples = ? WHERE sha256 = (SELECT audio_sha256 FROM clips WHERE clip_id = ?)",
                         (round(minutes * 60 * RATE / clips), cid))
        ids.append(cid)
    store.db.commit()
    labels.refresh(store)
    return ids


def promote_with_snapshot(store) -> str:
    """Register and promote a version trained on every clip trainable now; returns its name."""
    split_id, clips = splits.record_train(store, now=NOW)
    assert clips
    snap = snapshots.from_train(store.db, split_id)
    snapshots.write(snap)
    name = registry.register({"parent": "v0", "dataset": {"snapshot_sha256": snap.sha256, "train_split": split_id}})
    registry.promote(name, "test")
    return name


def locked_now(folder) -> bool:
    """Whether someone holds `folder`'s round lock (taking it, and letting go at once, if not)."""
    try:
        with train_run.round_lock(folder):
            return False
    except TrainError:
        return True


class Fake:
    """The heavy steps. `prepare` writes a real train row, as `healing train prepare` does."""

    def __init__(self, rounds, fail=None, busy=False, remote_code=0, words=2000, verdict="pass", promote_code=0,
                 register=False):
        self.rounds, self.fail, self.calls, self.parents = rounds, fail, [], {}
        self.busy, self.held = busy, contextlib.ExitStack()  # busy: a manual `healing train --local` holds the round
        self.remote_code, self.words, self.verdict, self.promote_code = remote_code, words, verdict, promote_code
        self.notes, self.register = [], register  # register: export registers a real candidate

    def prepare(self, parent):
        self.calls.append("prepare")
        self.parents["prepare"] = parent
        s = Store()
        try:
            split_id, _ = splits.record_train(s, parent=parent, now=NOW)
        finally:
            s.close()
        return SimpleNamespace(path=self.rounds.parent / "bundle", train_split=split_id)

    def open_round(self, bundle):
        folder = self.rounds / f"r{len(self.calls)}"
        (folder / "work").mkdir(parents=True)
        (folder / "out").mkdir()
        (folder / "work" / "checkpoint.pt").write_bytes(b"x" * 1024)
        (folder / "out" / "run.json").write_text("{}")
        self.calls.append("open")
        if self.busy:
            self.held.enter_context(train_run.round_lock(folder))
        return SimpleNamespace(folder=folder, work=folder / "work", out=folder / "out")

    lock = staticmethod(auto.Steps.lock)  # the real round lock: these tests are about it

    def train(self, round_):
        self.calls.append("train")
        return 1 if self.fail == "train" else 0

    def export(self, delta, config, snapshot, parent):
        self.calls.append("export")
        self.export_locked = locked_now(delta.parent.parent)
        self.parents["export"] = parent
        if self.fail == "export":
            raise auto.ExportError("convert failed (exit 1): boom")
        version = registry.register({"parent": parent}) if self.register else "v9"
        return {"version": version, "gguf_sha256": "ab" * 32}

    def evaluate(self, version, gguf_sha, parent):
        self.calls.append("eval")
        self.parents["eval"] = parent
        return "scored rolling, general"

    def remote(self, bundle, max_minutes, say):
        self.calls.append("remote")
        self.max_minutes = max_minutes
        folder = self.rounds / "remote"
        (folder / "work").mkdir(parents=True, exist_ok=True)
        (folder / "out").mkdir(exist_ok=True)
        (folder / "work" / "env.json").write_text("{}")
        return self.remote_code, folder

    def build(self, parent):
        self.calls.append("build")
        self.parents["build"] = parent
        return self.words

    def gate(self, version, say):
        self.calls.append(f"gate {version}")
        if self.verdict == "pass":
            registry.record_gate(version, {"against": "v0", "pass": True, "refused": False}, None)
            return 0
        if self.verdict == "refused":
            registry.record_gate(version, {"against": "v0", "pass": False, "refused": True}, None)
        else:
            registry.record_gate(version, {"against": "v0", "pass": False, "refused": False}, "gate: rolling_wer")
        return 7

    def promote(self, version, say):
        self.calls.append(f"promote {version}")
        if self.promote_code == 0:
            registry.promote(version, "test")
        return self.promote_code

    def notify(self, message):
        self.notes.append(message)


def run(home, fake=None, **kw):
    fake = fake or Fake(home / "rounds")
    lines = []
    code = auto.run(fake, now=NOW, say=lines.append, **kw)
    return code, "\n".join(lines), fake


FULL = ["prepare", "open", "train", "export", "eval"]


def test_waits_under_200_minutes(home, store):
    speech(store, 150, clips=30)
    code, out, fake = run(home)
    assert code == 0
    assert "waiting: strong speech 150/200 min" in out
    assert fake.calls == []


def test_counts_minutes_of_audio_not_clips(home, store):
    # 300 short clips (Hex's own audio_ms claims hours) are 2.5 minutes of audio: still waiting
    speech(store, 2.5, clips=300, audio_ms=10**9)
    code, out, fake = run(home)
    assert "waiting: strong speech 2/200 min" in out and fake.calls == []
    # three long clips are 210 minutes: a round starts
    speech(store, 207.5, clips=3)
    code, out, fake = run(home)
    assert code == 0, out
    assert "starting round v1" in out
    assert fake.calls == FULL


def test_starts_at_200_minutes_with_no_time_floor(home, store):
    speech(store, 20, clips=4)
    parent = promote_with_snapshot(store)  # a round just finished and was promoted
    speech(store, 200, clips=40)
    code, out, fake = run(home)
    assert code == 0, out
    assert f"200/200 min (40 new clips since {parent}'s snapshot)" in out
    assert fake.calls == FULL
    assert "healing gate v9" in out  # M7's gate is the next step; nothing decides here


def test_new_means_not_in_parent_snapshot(home, store):
    speech(store, 150, clips=10)
    promote_with_snapshot(store)  # those 150 minutes are in v1's snapshot
    speech(store, 100, clips=10)
    code, out, fake = run(home)
    assert "waiting: strong speech 100/200 min" in out and fake.calls == []  # 250 trainable, 100 new
    speech(store, 100, clips=10)
    code, out, fake = run(home)
    assert fake.calls == FULL, out


def test_lock_prevents_double_run(home, store):
    speech(store, 250, clips=5)
    with locked(auto.lock_path(), wait=False):
        code, out, fake = run(home)
    assert code == 0
    assert "already running" in out
    assert fake.calls == []


def test_scratch_removed_after_round(home, store):
    speech(store, 250, clips=5)
    code, out, fake = run(home)
    assert code == 0
    round_ = home / "rounds" / "r1"
    assert not (round_ / "work").exists() and (round_ / "out" / "run.json").is_file()
    # a failed round removes its scratch too
    fake = Fake(home / "rounds2", fail="export")
    code, out, fake = run(home, fake)
    assert code == 1 and "export" in out
    assert not (home / "rounds2" / "r1" / "work").exists()


def test_failed_round_does_not_reset_the_clock(home, store):
    speech(store, 250, clips=5)
    code, out, fake = run(home, Fake(home / "a", fail="train"))
    assert code == 1 and fake.calls == ["prepare", "open", "train"]
    assert not (home / "a" / "r1" / "work").exists()
    code, out, fake = run(home, Fake(home / "b"))  # its train row is not a version: still 250 new
    assert "250/200 min" in out and fake.calls == FULL


def test_undecided_candidate_waits_for_the_gate(home, store):
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})  # v1: exported, not yet gated
    code, out, fake = run(home)
    assert code == 0 and fake.calls == []
    assert "candidate v1 awaits the gate" in out and "healing gate v1" in out


@pytest.mark.parametrize("decision,says", [
    ({"pass": False, "refused": True}, "refused it for thin evidence"),
    ({"pass": True, "refused": False}, "passed the gate"),
])
def test_gated_candidate_still_waits_and_says_why(home, store, decision, says):
    """A refusal (thin evidence) and a pass both leave v1 a candidate (T7.1): still no second round."""
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})
    registry.record_gate("v1", {"against": "v0", **decision}, None)
    code, out, fake = run(home)
    assert code == 0 and fake.calls == []
    assert says in out and "v1" in out


def test_remote_runner_needs_allow_upload(home, store):
    """G-3: without `allow_upload: true` a Kaggle round stops after prepare; nothing is uploaded."""
    speech(store, 250, clips=5)
    auto.settings_path().write_text("runner: kaggle\n")
    code, out, fake = run(home)
    assert code == 0 and fake.calls == ["prepare"]
    assert "allow_upload" in out and "G-3" in out and "nothing uploaded" in out


def test_kaggle_round_exports_what_it_brought_back(home, store):
    speech(store, 250, clips=5)
    auto.settings_path().write_text("runner: kaggle\nallow_upload: true\nmax_minutes: 90\n")
    code, out, fake = run(home)
    assert code == 0 and fake.calls == ["prepare", "remote", "export", "eval"]
    assert fake.max_minutes == 90 and "next: healing gate v9" in out
    assert not (fake.rounds / "remote" / "work").exists() and (fake.rounds / "remote" / "out").is_dir()


def test_kaggle_quota_used_up_trains_here(home, store):
    from healing.remote.kaggle_runner import QUOTA

    speech(store, 250, clips=5)
    auto.settings_path().write_text("runner: kaggle\nallow_upload: true\n")
    code, out, fake = run(home, Fake(home / "rounds", remote_code=QUOTA))
    assert code == 0 and fake.calls == ["prepare", "remote", "open", "train", "export", "eval"]
    assert "quota is used up" in out


def test_a_failed_kaggle_round_registers_nothing_and_says_so(home, store):
    speech(store, 250, clips=5)
    auto.settings_path().write_text("runner: kaggle\nallow_upload: true\n")
    code, out, fake = run(home, Fake(home / "rounds", remote_code=1))
    assert code == 1 and fake.calls == ["prepare", "remote"]
    assert "round failed at train (kaggle" in out and fake.notes and "failed" in fake.notes[0]
    assert registry.load().current == "v0"


def test_settings_are_checked(home):
    for text in ("max_minutes: true\n", "max_minutes: 0\n", "auto_promote: 1\n", "runner: colab\n"):
        auto.settings_path().parent.mkdir(parents=True, exist_ok=True)
        auto.settings_path().write_text(text)
        with pytest.raises(Exception):
            auto.load_settings()


HANDS_OFF = "auto_promote: true\n"


def test_hands_off_waits_on_reviews_and_says_how_many(home, store):
    """Under 1,500 reviewed words nothing is gated (D-22): it waits, and a notification says so."""
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})
    auto.settings_path().write_text(HANDS_OFF)
    code, out, fake = run(home, Fake(home / "rounds", words=600))
    assert code == 0 and fake.calls == ["build"] and fake.parents["build"] == "v0"
    assert "600/1500 reviewed words" in out and "healing review" in fake.notes[0]


def test_hands_off_gates_promotes_then_trains_the_next_round(home, store):
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})  # v1, waiting
    auto.settings_path().write_text(HANDS_OFF)
    code, out, fake = run(home, Fake(home / "rounds", register=True))
    assert code == 0
    assert fake.calls[:3] == ["build", "gate v1", "promote v1"]
    assert fake.calls[3:8] == FULL  # v1 decided, so the round starts from it
    assert fake.parents["prepare"] == "v1"
    assert fake.calls[8:] == ["build", "gate v2", "promote v2"]  # and its candidate is decided at once
    state = registry.load()
    assert state.current == "v2" and state.get("v1").state == "promoted"
    assert any("v2 is now Hex's speech model" in n for n in fake.notes)


def test_hands_off_a_rejected_candidate_frees_the_next_round(home, store):
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})
    auto.settings_path().write_text(HANDS_OFF)
    code, out, fake = run(home, Fake(home / "rounds", verdict="reject"))
    assert fake.calls[:2] == ["build", "gate v1"] and "promote v1" not in fake.calls
    assert registry.load().get("v1").state == "rejected"
    assert fake.calls[2:7] == FULL and fake.parents["prepare"] == "v0"
    assert "rejected" in fake.notes[0]


def test_hands_off_a_refused_gate_keeps_waiting(home, store):
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})
    auto.settings_path().write_text(HANDS_OFF)
    code, out, fake = run(home, Fake(home / "rounds", verdict="refused"))
    assert code == 0 and fake.calls == ["build", "gate v1"]
    assert registry.load().get("v1").state == "candidate" and "more reviewed clips" in fake.notes[0]


def test_hands_off_retries_a_failed_promote_without_gating_again(home, store):
    speech(store, 100, clips=2)
    registry.register({"parent": "v0"})
    auto.settings_path().write_text(HANDS_OFF)
    code, out, fake = run(home, Fake(home / "a", promote_code=9))
    assert fake.calls == ["build", "gate v1", "promote v1"] and "failed" in fake.notes[0]
    code, out, fake = run(home, Fake(home / "b"))
    assert fake.calls == ["promote v1"] and registry.load().current == "v1"


def test_without_auto_promote_nothing_is_gated(home, store):
    speech(store, 250, clips=5)
    registry.register({"parent": "v0"})
    code, out, fake = run(home)
    assert fake.calls == [] and "awaits the gate" in out


def test_dry_run_prints_and_runs_nothing(home, store):
    speech(store, 250, clips=5)
    code, out, fake = run(home, dry_run=True)
    assert code == 0 and fake.calls == []
    assert "would start round v1" in out and "prepare, train (local), export, eval" in out
    assert not auto.lock_path().exists()


def test_cli_dry_run(home, store, capsys):
    from healing.cli import main

    speech(store, 120, clips=4)
    assert main(["train", "auto", "--dry-run"]) == 0
    assert "waiting: strong speech 120/200 min (4 new clips since v0's snapshot)" in capsys.readouterr().out


def test_eval_records_every_metric_the_eval_produces(home, store, monkeypatch):
    from healing.eval import run as eval_run

    name = registry.register({"parent": "v0"})
    gguf_sha = "cd" * 32
    path = f"/objects/{gguf_sha}"
    scored = {"wer": 0.1, "word_errors": 3, "ref_words": 30, "format_error_rate": 0.2, "term_error_rate": 0.25}
    edited = {"wer": None, "word_errors": None, "ref_words": None, "format_error_rate": 0.3, "term_error_rate": 0.5,
              "edited_span": {"errors": 1, "positions": 4}}
    document = {"eval_id": "eval-x", "models": [{"gguf_path": path, "gguf_sha256": gguf_sha}],
                "sets": [{"set": "rolling", "manifest_sha256": "1" * 64}, {"set": "edited", "manifest_sha256": "2" * 64}],
                "vocabulary": {"mode": "current", "terms": 0, "boost": None},
                "scores": {path: {"rolling": scored, "edited": edited}}}
    monkeypatch.setattr(eval_run, "evaluate", lambda *a, **kw: document)
    monkeypatch.setattr(eval_run, "write", lambda doc: home / "evals" / "eval-x.json")
    auto._evaluate(name, gguf_sha, "v0")
    scores = registry.load().get(name).scores
    assert set(scores) == {"rolling", "edited"}
    assert scores["rolling"]["wer"] == 0.1 and scores["rolling"]["ref_words"] == 30
    assert scores["rolling"]["format_error_rate"] == 0.2 and scores["rolling"]["term_error_rate"] == 0.25
    assert scores["edited"]["wer"] is None and scores["edited"]["edited_span"] == {"errors": 1, "positions": 4}
    assert scores["edited"]["format_error_rate"] == 0.3 and scores["edited"]["term_error_rate"] == 0.5


def test_round_passes_the_resolved_parent_not_the_alias(home, store):
    # eval's rolling and edited holdouts are looked up by their stored parent_version (`--parent v1`),
    # which never reads "current"
    speech(store, 20, clips=4)
    parent = promote_with_snapshot(store)
    speech(store, 250, clips=5)
    code, out, fake = run(home)
    assert code == 0, out
    assert fake.parents == {"prepare": parent, "export": parent, "eval": parent}
    rows = store.db.execute("SELECT parent_version FROM splits WHERE kind = 'train' ORDER BY created_at_ms").fetchall()
    assert rows[-1][0] == parent


def test_a_busy_round_keeps_its_scratch(home, store):
    # a manual `healing train --local` holds this round: train auto refuses and deletes nothing of its
    speech(store, 250, clips=5)
    fake = Fake(home / "rounds", busy=True)
    with fake.held:
        code, out, fake = run(home, fake)
        assert code == 1 and "already training" in out, out
        assert fake.calls == ["prepare", "open"]
        assert (home / "rounds" / "r1" / "work" / "checkpoint.pt").is_file()


def test_round_lock_is_held_through_export_and_cleanup(home, store):
    speech(store, 250, clips=5)
    code, out, fake = run(home)
    assert code == 0, out
    assert fake.export_locked  # no other trainer can get in between training and the scratch's removal
    assert not locked_now(home / "rounds" / "r1")  # and it is let go once the round ends
