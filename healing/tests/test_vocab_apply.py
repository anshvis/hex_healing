import json
import sys

import pytest

from healing.cli import main
from healing.store import Store
from healing.vocab import apply as vocab_apply
from healing.vocab import mine as vocab

FAKE_HEX = """#!{python}
import json, os, sys
state = os.environ["FAKE_HEX_STATE"]
log = state + ".calls"
args = sys.argv[1:]
if not os.path.exists(state):
    sys.stderr.write("Hex isn't running\\n")
    sys.exit(3)
with open(log, "a") as f:
    f.write(" ".join(args[:2]) + "\\n")
entries = json.load(open(state))
if args[:2] == ["vocabulary", "list"]:
    print(json.dumps({{"version": 4, "status": "vocabulary", "entries": entries}}))
elif args[:2] == ["vocabulary", "set"]:
    items = json.load(open(args[args.index("--file") + 1]))
    new = [{{"term": i}} if isinstance(i, str) else i for i in items]
    changed = new != entries
    json.dump(new, open(state, "w"))
    print(json.dumps({{"version": 4, "status": "done", "changed": changed}}))
else:
    sys.exit(2)
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))  # never the real one
    state = tmp_path / "hex-list.json"
    monkeypatch.setenv("FAKE_HEX_STATE", str(state))
    binary = tmp_path / "hex"
    binary.write_text(FAKE_HEX.format(python=sys.executable))
    binary.chmod(0o755)

    class Env:
        hex = str(binary)

        def set_list(self, entries):
            state.write_text(json.dumps(entries))

        def listed(self):
            return json.loads(state.read_text())

        def terms(self):
            return [e["term"] for e in self.listed()]

        def calls(self):
            path = state.with_name(state.name + ".calls")
            return path.read_text().splitlines() if path.exists() else []

        def candidates(self, *rows):
            """rows: (term, [spellings]) in rank order."""
            store = Store()
            try:
                found = [vocab.Candidate(t, tuple((s, 2) for s in sl), 2, 1, 10.0 - i, 1, 2, "all") for i, (t, sl) in enumerate(rows)]
                vocab.save(store, vocab.Mined(candidates=found), "all")
            finally:
                store.close()

        def run(self, *args):
            return main(["vocab", *args, "--hex", self.hex])

    return Env()


MINED = [("Kit", ["kip"]), ("Qorvyx", ["corvix"]), ("Rowan", [])]


def test_user_terms_first(env, capsys):
    env.set_list([{"term": "Anomaly"}, {"term": "Rune"}])
    env.candidates(*MINED)
    assert env.run("apply") == 0
    assert env.listed() == [
        {"term": "Anomaly"}, {"term": "Rune"},
        {"term": "Kit", "sounds_like": ["kip"]}, {"term": "Qorvyx", "sounds_like": ["corvix"]}, {"term": "Rowan"},
    ]
    assert env.calls().count("vocabulary set") == 1  # one atomic write
    applied = json.loads(vocab_apply.applied_path().read_text())
    assert [t["term"] for t in applied["terms"]] == ["Kit", "Qorvyx", "Rowan"] and "pending" not in applied


def test_your_term_wins_a_duplicate_and_keeps_its_place(env):
    env.set_list([{"term": "kit"}, {"term": "Rune"}])
    env.candidates(*MINED)
    env.run("apply")
    assert env.terms() == ["kit", "Rune", "Qorvyx", "Rowan"]


def test_dry_run_shows_the_order_and_writes_nothing(env, capsys):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    assert env.run("apply", "--dry-run") == 0
    out = capsys.readouterr().out
    assert out.startswith("dry run, nothing written")
    assert out.index("Anomaly") < out.index("Kit ← kip") < out.index("Qorvyx")
    assert env.terms() == ["Anomaly"] and "vocabulary set" not in env.calls()
    assert not vocab_apply.applied_path().exists()


def test_idempotent(env, capsys):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    after_first = env.listed()
    record = vocab_apply.applied_path().read_text()
    assert env.run("apply") == 0
    assert env.listed() == after_first
    assert env.calls().count("vocabulary set") == 1  # the second apply wrote nothing
    assert vocab_apply.applied_path().read_text() == record
    assert "no change" in capsys.readouterr().out


def test_revert_exact(env, capsys):
    env.set_list([{"term": "Anomaly"}, {"term": "Rune"}])
    env.candidates(*MINED)
    env.run("apply")
    assert env.run("revert") == 0
    assert env.listed() == [{"term": "Anomaly"}, {"term": "Rune"}]
    assert env.calls().count("vocabulary set") == 2
    assert "removed 3 mined" in capsys.readouterr().out
    assert env.run("revert") == 0  # nothing left to take out: no write
    assert env.calls().count("vocabulary set") == 2


def test_user_removed_never_re_added(env, capsys):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    env.set_list([e for e in env.listed() if e["term"] != "Kit"])  # you removed a mined term
    env.run("apply")
    assert "Kit" not in env.terms()
    assert "removed by you, never added again: Kit" in capsys.readouterr().out
    assert json.loads(vocab_apply.applied_path().read_text())["rejected"] == ["Kit"]
    env.run("apply")
    env.run("revert")
    env.candidates(*MINED)  # mined again, still refused
    env.run("apply")
    assert "Kit" not in env.terms()


def test_edited_mined_term_survives_revert_and_reorder(env):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    edited = [{"term": "Kit", "sounds_like": ["kip", "kitt"]} if e["term"] == "Kit" else e for e in env.listed()]
    env.set_list(edited)  # you added a spelling: it is yours now
    env.run("apply")
    assert env.terms() == ["Anomaly", "Kit", "Qorvyx", "Rowan"]  # yours (Kit included) first, in order
    assert env.listed()[1]["sounds_like"] == ["kip", "kitt"]
    env.run("revert")
    assert env.listed() == [{"term": "Anomaly"}, {"term": "Kit", "sounds_like": ["kip", "kitt"]}]


def test_a_term_you_added_later_is_not_demoted(env):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    env.set_list(env.listed() + [{"term": "Zed"}])  # appended after the mined ones
    env.run("apply")
    assert env.terms() == ["Anomaly", "Zed", "Kit", "Qorvyx", "Rowan"]


def test_a_dropped_candidate_leaves_the_list_but_is_not_rejected(env):
    env.set_list([])
    env.candidates(*MINED)
    env.run("apply")
    env.candidates(*MINED[:2])  # Rowan is no longer mined
    env.run("apply")
    assert env.terms() == ["Kit", "Qorvyx"]
    assert json.loads(vocab_apply.applied_path().read_text())["rejected"] == []


def test_hex_down_exits_nonzero_and_writes_nothing(env, capsys):
    env.candidates(*MINED)  # no state file: the fake answers exit 3
    assert env.run("apply") == 3
    assert "Hex isn't running" in capsys.readouterr().err
    assert env.run("revert") == 3
    assert not vocab_apply.applied_path().exists()


def test_cap_of_500(env):
    env.set_list([{"term": f"mine{i}"} for i in range(499)])
    env.candidates(*MINED)
    env.run("apply")
    assert len(env.listed()) == 500 and env.terms()[-1] == "Kit"


def test_per_mode_slots(env, capsys):
    env.set_list([{"term": f"mine{i}"} for i in range(60)])
    env.candidates(*MINED)
    support = vocab_apply.hex_support_dir()
    support.mkdir(parents=True)
    (support / "settings.json").write_text(json.dumps({"dictation_processing": {
        "default_mode": {"name": "Global"},
        "modes": [{"name": "Code", "vocabulary": [{"term": "useEffect"}, {"term": "tsx"}]}],
    }}))
    env.run("apply", "--dry-run")
    out = capsys.readouterr().out
    assert "Global (no mode): 3 of 3 mined terms biased" in out  # 60 + 3 fit in 64
    assert "Code (2 of its own terms): 2 of 3 mined terms biased" in out  # 2 + 60 + 2


def test_your_terms_filling_the_slots_warn(env, capsys):
    env.set_list([{"term": f"mine{i}"} for i in range(64)])
    env.candidates(*MINED)
    env.run("apply", "--dry-run")
    out = capsys.readouterr().out
    assert "no mined term is biased" in out and "Global (no mode): 0 of 3 mined terms biased" in out


def test_a_crash_before_the_set_answers_keeps_ownership(env):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    # a run died after writing its intent and after Hex took the list: the record still names `pending`
    record = json.loads(vocab_apply.applied_path().read_text())
    record["pending"], record["terms"] = record["terms"], []
    vocab_apply.applied_path().write_text(json.dumps(record))
    env.run("revert")
    assert env.listed() == [{"term": "Anomaly"}]
