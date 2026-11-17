import json
import sys
from pathlib import Path

import pytest

from healing.cli import main
from healing.store import Store
from healing.vocab import apply as vocab_apply
from healing.vocab import mine as vocab

FAKE_HEX = """#!{python}
import json, os, sys, time
state = os.environ["FAKE_HEX_STATE"]
log = state + ".calls"
args = sys.argv[1:]
if not os.path.exists(state):
    sys.stderr.write("Hex isn't running\\n")
    sys.exit(3)
with open(log, "a") as f:
    f.write(" ".join(args[:2]) + "\\n")
if args[:2] == ["settings", "get"]:
    key = args[2]
    value = os.environ.get("FAKE_BOOST", "light")
    print(json.dumps({{"version": 4, "status": "settings", "settings": [{{"key": key, "value": value}}]}}))
    sys.exit(0)
entries = json.load(open(state))
if args[:2] == ["vocabulary", "list"]:
    print(json.dumps({{"version": 4, "status": "vocabulary", "entries": entries}}))
elif args[:2] == ["vocabulary", "set"]:
    mode = os.environ.get("FAKE_SET_MODE")
    if mode == "fail":
        sys.stderr.write("invalid-vocabulary: entry 0: refused\\n")
        sys.exit(1)
    if mode == "hang-before":
        time.sleep(5)
    items = json.load(open(args[args.index("--file") + 1]))
    new = [{{"term": i}} if isinstance(i, str) else i for i in items]
    changed = new != entries
    json.dump(new, open(state, "w"))
    if mode == "hang-after":
        time.sleep(5)
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
    monkeypatch.setenv("FAKE_BOOST", "light")  # the gate (T4.3) only guards normal and strong
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


# ---- review round: journal, lock, durability, and the A/B gate ----------------------------------

def journal():
    return json.loads(vocab_apply.applied_path().read_text())


def test_a_refused_set_restores_the_journal(env, monkeypatch):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    monkeypatch.setenv("FAKE_SET_MODE", "fail")
    assert env.run("apply") == 1
    assert env.listed() == [{"term": "Anomaly"}]
    record = journal() if vocab_apply.applied_path().exists() else {}
    assert not record.get("pending") and not record.get("terms")
    monkeypatch.delenv("FAKE_SET_MODE")
    env.set_list(env.listed() + [{"term": "Kit", "sounds_like": ["kip"]}])  # you add the same term by hand
    env.run("revert")
    assert env.terms() == ["Anomaly", "Kit"]


def test_an_unknown_outcome_keeps_pending_and_revert_still_owns_it(env, monkeypatch):
    monkeypatch.setattr(vocab_apply, "HEX_TIMEOUT_S", 0.5)
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    monkeypatch.setenv("FAKE_SET_MODE", "hang-after")  # Hex took the list but never answered
    assert env.run("apply") == 1
    assert [t["term"] for t in journal()["pending"]] == ["Kit", "Qorvyx", "Rowan"]
    monkeypatch.delenv("FAKE_SET_MODE")
    env.run("revert")
    assert env.listed() == [{"term": "Anomaly"}]


def test_a_retry_keeps_the_unresolved_pending_of_a_crashed_run(env, monkeypatch):
    monkeypatch.setattr(vocab_apply, "HEX_TIMEOUT_S", 0.5)
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED[:2])
    monkeypatch.setenv("FAKE_SET_MODE", "hang-after")
    env.run("apply")  # Kit and Qorvyx are in the list, unconfirmed
    env.candidates(("Rowan", []))
    monkeypatch.setenv("FAKE_SET_MODE", "hang-before")  # the retry is unknown too, and changed nothing
    env.run("apply")
    assert {t["term"] for t in journal()["pending"]} == {"Kit", "Qorvyx", "Rowan"}
    monkeypatch.setenv("FAKE_SET_MODE", "fail")  # a refused retry puts back what was unresolved
    env.run("apply")
    assert {t["term"] for t in journal()["pending"]} == {"Kit", "Qorvyx", "Rowan"}
    monkeypatch.delenv("FAKE_SET_MODE")
    env.run("revert")
    assert env.listed() == [{"term": "Anomaly"}]


def test_a_confirmed_retry_replaces_pending(env, monkeypatch):
    monkeypatch.setattr(vocab_apply, "HEX_TIMEOUT_S", 0.5)
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED[:2])
    monkeypatch.setenv("FAKE_SET_MODE", "hang-after")
    env.run("apply")
    monkeypatch.delenv("FAKE_SET_MODE")
    env.candidates(("Rowan", []))
    assert env.run("apply") == 0
    assert env.terms() == ["Anomaly", "Rowan"]
    record = journal()
    assert [t["term"] for t in record["terms"]] == ["Rowan"] and "pending" not in record


def test_apply_and_revert_hold_the_healing_lock(env, capsys):
    from healing.store import process_lock

    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    with process_lock():
        assert env.run("apply") == 1
        assert env.run("revert") == 1
    assert "another healing process" in capsys.readouterr().err
    assert env.terms() == ["Anomaly"] and "vocabulary set" not in env.calls()


def test_the_journal_uses_unique_temp_files(env, monkeypatch):
    sources = []
    real = vocab_apply.os.replace
    monkeypatch.setattr(vocab_apply.os, "replace", lambda a, b: (sources.append(str(a)), real(a, b))[1])
    vocab_apply.save_applied(vocab_apply.Applied(1, [], [], None, 0))
    vocab_apply.save_applied(vocab_apply.Applied(2, [], [], None, 0))
    assert len(set(sources)) == 2 and all(not Path(s).exists() for s in sources)


def test_the_journal_directory_is_synced_before_hex_is_called(env, monkeypatch):
    import os
    import stat

    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(vocab_apply.os, "fsync", lambda fd: (synced.append(stat.S_ISDIR(os.fstat(fd).st_mode)), real_fsync(fd))[1])
    real_set = vocab_apply.hex_set
    seen = {}

    def spy(binary, entries):
        seen["directory_synced_first"] = True in synced
        return real_set(binary, entries)

    monkeypatch.setattr(vocab_apply, "hex_set", spy)
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    assert seen == {"directory_synced_first": True}


def verdict(boost, passes, mined=None):
    """The latest A/B's record; by default it measured all of MINED."""
    measured = [{"term": t, "sounds_like": sl} for t, sl in (MINED if mined is None else mined)]
    vocab_apply.verdict_path().parent.mkdir(parents=True, exist_ok=True)
    vocab_apply.verdict_path().write_text(json.dumps(
        {"ab_id": "vocab-ab-x", "boost": boost, "passes": passes, "mined_terms": measured}))


def refused(env, capsys, *args):
    assert env.run("apply", *args) == 4
    err = capsys.readouterr().err
    assert "hex settings set vocabulary_boost light" in err and "--force" in err
    assert env.terms() == ["Anomaly"] and "vocabulary set" not in env.calls()


def test_apply_refuses_normal_boost_without_a_passing_verdict(env, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    refused(env, capsys)  # no A/B on record
    verdict("normal", False)
    refused(env, capsys)  # a failed one
    verdict("normal", None)
    refused(env, capsys)  # an undecided one
    verdict("strong", True)
    refused(env, capsys)  # passing, but at another boost
    monkeypatch.setenv("FAKE_BOOST", "strong")
    verdict("normal", True)
    refused(env, capsys)


def test_apply_goes_ahead_with_a_passing_verdict_at_the_running_boost(env, monkeypatch):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    verdict("normal", True)
    assert env.run("apply") == 0 and env.terms()[0] == "Anomaly" and len(env.terms()) == 4


@pytest.mark.parametrize("boost", ["light", "off"])
def test_light_and_off_need_no_verdict(env, monkeypatch, boost):
    monkeypatch.setenv("FAKE_BOOST", boost)
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    assert env.run("apply") == 0 and len(env.terms()) == 4


def test_force_overrides_the_gate_and_is_logged(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    assert env.run("apply", "--force") == 0 and len(env.terms()) == 4
    log = (vocab_apply.applied_path().parent / "apply.log").read_text()
    assert "forced" in log and "normal" in log and "no `healing vocab ab` is on record" in log


def test_dry_run_says_it_would_refuse_and_still_writes_nothing(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    assert env.run("apply", "--dry-run") == 0
    assert "apply would refuse" in capsys.readouterr().out and "vocabulary set" not in env.calls()


def test_unreadable_boost_refuses_too(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    monkeypatch.setattr(vocab_apply, "running_boost", lambda binary: None)
    refused(env, capsys)


def test_revert_needs_no_verdict(env, monkeypatch):
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    env.run("apply")
    monkeypatch.setenv("FAKE_BOOST", "strong")
    assert env.run("revert") == 0 and env.terms() == ["Anomaly"]


def test_apply_refuses_a_mined_term_the_ab_did_not_measure(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    verdict("normal", True, mined=MINED[:2])  # Rowan was mined after the A/B
    assert env.run("apply") == 4
    err = capsys.readouterr().err
    assert "Rowan" in err and "healing vocab ab" in err and "vocabulary set" not in env.calls()
    verdict("normal", True, mined=[("Kit", ["kip"]), ("Qorvyx", ["other"]), ("Rowan", [])])  # same term, other spellings
    assert env.run("apply") == 4 and "Qorvyx" in capsys.readouterr().err
    assert env.run("apply", "--force") == 0


def test_a_subset_of_the_measured_terms_is_fine(env, monkeypatch):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED[:2])  # the A/B measured three; two are about to be added
    verdict("normal", True)
    assert env.run("apply") == 0 and env.terms() == ["Anomaly", "Kit", "Qorvyx"]


def test_a_term_you_already_have_need_not_be_measured(env, monkeypatch):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Rowan"}])  # yours, so it is not added
    env.candidates(*MINED)
    verdict("normal", True, mined=MINED[:2])
    assert env.run("apply") == 0 and env.terms() == ["Rowan", "Kit", "Qorvyx"]


def test_a_verdict_without_the_terms_it_measured_is_refused(env, monkeypatch):
    monkeypatch.setenv("FAKE_BOOST", "normal")
    env.set_list([{"term": "Anomaly"}])
    env.candidates(*MINED)
    vocab_apply.verdict_path().parent.mkdir(parents=True, exist_ok=True)
    vocab_apply.verdict_path().write_text(json.dumps({"boost": "normal", "passes": True}))  # an older verdict file
    assert env.run("apply") == 4
