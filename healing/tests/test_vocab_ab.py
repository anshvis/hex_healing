"""`healing vocab ab` over fake evals: a Qorvyx fixture whose term is heard only when the mined list is passed."""

import json
import sys
from pathlib import Path

import pytest

from healing.cli import main
from healing.eval import run as eval_run
from healing.store import Store
from healing.vocab import ab
from healing.vocab import mine as vocab

from test_eval import general_set, world  # noqa: F401  (the fake-hex world the eval tests use)
from test_vocab_apply import FAKE_HEX

MODEL = Path("/models/unified.gguf")
YOURS = [{"term": "Anomaly", "sounds_like": []}]
MINED = [{"term": "Qorvyx", "sounds_like": ["corvix"]}]


def fake_evaluate(term_clips=25, general_clips=40, bias_hurts_general=0, false_replacements=0):
    """An `evaluate` whose transcripts get Qorvyx right only when the list names it, and that makes
    `bias_hurts_general` general clips one word worse and `false_replacements` say Qorvyx unprompted."""

    def evaluate(models, sets, *, store, parent, progress, vocabulary_entries, vocabulary_boost):
        biased = any(e["term"] == "Qorvyx" for e in vocabulary_entries)
        key = str(models[0])
        rolling = [{
            "id": f"r{i}", "terms_expected": ["Qorvyx"] if i < term_clips else [], "terms_hit": ["Qorvyx"] if biased and i < term_clips else [],
            "transcript": "x", "expected": "x", "word_errors": 0 if biased else int(i < term_clips), "reference_words": 10,
        } for i in range(term_clips + 5)]
        general = [{
            "id": f"g{i}", "terms_expected": [], "terms_hit": [],
            "transcript": "say qorvyx" if biased and i < false_replacements else "say it", "expected": "say it",
            "word_errors": 1 if biased and i < bias_hurts_general else 0, "reference_words": 10,
        } for i in range(general_clips)]
        return {
            "eval_id": f"eval-{'with' if biased else 'without'}", "models": [{"gguf_path": key, "gguf_sha256": "abc"}],
            "unavailable_sets": {},
            "scores": {key: {"rolling": {}, "general": {"false_replacement_rate": None}}},
            "per_clip": {key: {"rolling": rolling, "general": general}},
            "_boost": vocabulary_boost,
        }

    return evaluate


def run(evaluate, **kwargs):
    return ab.run_ab(MODEL, "normal", "current", yours=YOURS, mined=MINED, store=None, evaluate=evaluate, now=1_800_000_000, **kwargs)


def test_reports_deltas_and_cis():
    report = run(fake_evaluate())
    rolling, general = report["sets"]["rolling"], report["sets"]["general"]
    terms = rolling["terms"]
    # the Qorvyx fixture: hit with the mined list, missed without it
    assert terms["error_without"] == 1.0 and terms["error_with"] == 0.0
    assert terms["wins"] == 25 and terms["losses"] == 0 and terms["favours_mined"] and terms["p_value"] < 1e-6
    assert rolling["wer"]["delta"] < 0 and rolling["wer"]["ci_low"] <= rolling["wer"]["delta"] <= rolling["wer"]["ci_high"]
    assert general["wer"]["delta"] == 0 and general["false_replacements"]["rate"] == 0
    assert report["verdict"]["passes"] is True and report["verdict"]["boost"] == "normal"
    printed = ab.text(report)
    assert "McNemar exact over clips: mined got 25 more clips right" in printed and "95% CI" in printed and "verdict: the mined terms pass" in printed


def test_a_worse_general_wer_or_false_replacement_falls_back_to_light():
    report = run(fake_evaluate(bias_hurts_general=10))  # 10 words in 400: +2.5 pt
    assert report["verdict"]["passes"] is False and report["verdict"]["recommended_boost"] == "light"
    assert report["verdict"]["gates"]["general_wer_ci_high_at_most_0.3pt"] is False
    report = run(fake_evaluate(false_replacements=2))  # 2 of 40 clips: 5%
    assert report["verdict"]["gates"]["false_replacements_at_most_0.5pct"] is False
    assert report["sets"]["general"]["false_replacements"]["examples"][0]["terms"] == ["Qorvyx"]


def test_both_arms_are_biased_as_named_and_with_the_boost():
    seen = []
    inner = fake_evaluate()

    def spy(models, sets, **kwargs):
        seen.append((kwargs["vocabulary_entries"], kwargs["vocabulary_boost"], list(sets)))
        return inner(models, sets, **kwargs)

    ab.run_ab(MODEL, "strong", "current", yours=YOURS, mined=MINED, store=None, evaluate=spy)
    assert seen == [(YOURS, "strong", ["rolling", "general"]), (YOURS + MINED, "strong", ["rolling", "general"])]


def test_too_few_terms_is_na():
    report = run(fake_evaluate(term_clips=19))
    terms = report["sets"]["rolling"]["terms"]
    assert terms["na"].startswith("n/a: 19 clip(s)") and terms["favours_mined"] is False and terms["p_value"] is None
    assert report["verdict"]["passes"] is None and report["verdict"]["recommended_boost"] == "light"
    assert "no verdict" in ab.text(report) and "term error rate: n/a" in ab.text(report)


def test_an_unavailable_set_is_na_not_a_win():
    inner = fake_evaluate()

    def evaluate(models, sets, **kwargs):
        document = inner(models, sets, **kwargs)
        document["unavailable_sets"] = {"rolling": "no rolling set for parent current"}
        return document

    report = run(evaluate)
    assert "n/a (no rolling set" in ab.text(report) and report["verdict"]["passes"] is None


def test_mcnemar_exact():
    assert ab.mcnemar_exact(0, 0) == 1.0
    assert ab.mcnemar_exact(5, 5) == 1.0
    assert ab.mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert ab.mcnemar_exact(0, 8) == ab.mcnemar_exact(8, 0)


def test_no_mined_terms_is_an_error():
    with pytest.raises(ab.AbError):
        ab.run_ab(MODEL, "normal", "current", yours=YOURS, mined=[], store=None, evaluate=fake_evaluate())


def test_train_only_mining_in_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))
    state = tmp_path / "list.json"
    state.write_text(json.dumps([{"term": "Anomaly"}]))
    monkeypatch.setenv("FAKE_HEX_STATE", str(state))
    binary = tmp_path / "hex"
    binary.write_text(FAKE_HEX.format(python=sys.executable))
    binary.chmod(0o755)
    saved = Store()
    vocab.save(saved, vocab.Mined(candidates=[vocab.Candidate("Old", (("o", 2),), 2, 1, 1.0, 1, 2, "all")]), "all")
    saved.close()

    calls = []

    def spy(store, **kwargs):
        calls.append(kwargs)
        found = vocab.Candidate("Qorvyx", (("corvix", 2),), 2, 1, 1.0, 1, 2, "train")
        return vocab.Mined(candidates=[found])

    monkeypatch.setattr(vocab, "mine", spy)
    monkeypatch.setattr(vocab, "save", lambda *a, **k: pytest.fail("the A/B must never save its candidates"))
    seen = {}

    def stub(model, boost, parent, *, yours, mined, store, **kwargs):
        seen.update(yours=yours, mined=mined)
        return {"ab_id": "vocab-ab-test", "model": {"gguf_path": str(model), "gguf_sha256": "x"}, "boost": boost,
                "your_terms": len(yours), "mined_terms": mined, "sets": {}, "verdict": {"text": "stub"}}

    monkeypatch.setattr(ab, "run_ab", stub)
    assert main(["vocab", "ab", "--hex", str(binary), "--model", str(tmp_path / "m.gguf")]) == 0
    assert calls and calls[0]["train_only"] is True
    assert [e["term"] for e in seen["yours"]] == ["Anomaly"] and [e["term"] for e in seen["mined"]] == ["Qorvyx"]
    assert (tmp_path / "home" / "evals" / "vocab-ab-test.json").is_file()
    assert (state.with_name(state.name + ".calls")).read_text().splitlines() == ["vocabulary list"]  # read only
    store = Store(sweep=False)
    try:
        assert [c.term for c in vocab.load_candidates(store)] == ["Old"]  # the saved table is untouched
    finally:
        store.close()


def test_hex_down_exits_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_HEX_STATE", str(tmp_path / "absent.json"))
    binary = tmp_path / "hex"
    binary.write_text(FAKE_HEX.format(python=sys.executable))
    binary.chmod(0o755)
    assert main(["vocab", "ab", "--hex", str(binary), "--model", str(tmp_path / "m.gguf")]) == 3
    assert "Hex isn't running" in capsys.readouterr().err


def test_evaluate_passes_the_named_list_not_hex_settings(world):  # noqa: F811
    general_set(world.home)
    world.settings({"vocabulary": ["FromSettings"], "labs": {"vocabulary_biasing": True}})
    eval_run.evaluate([world.models["a.gguf"]], ["general"], store=None, vocabulary_entries=MINED, vocabulary_boost="strong")
    eval_run.evaluate([world.models["a.gguf"]], ["general"], store=None, vocabulary_entries=[], vocabulary_boost="strong")
    first, second = world.calls()
    assert first["vocabulary"] == MINED and first["args"][first["args"].index("--boost") + 1] == "strong"
    assert "--vocabulary" not in second["args"]


def test_mcnemar_counts_clips_not_terms():
    # one clip gets 10 terms right only with the mined list; 19 others are unchanged
    without = [{"id": "c0", "terms_expected": [f"t{i}" for i in range(10)], "terms_hit": []}]
    with_ = [{"id": "c0", "terms_expected": [f"t{i}" for i in range(10)], "terms_hit": [f"t{i}" for i in range(10)]}]
    for i in range(1, 25):
        row = {"id": f"c{i}", "terms_expected": ["x"], "terms_hit": ["x"]}
        without.append(row)
        with_.append(dict(row))
    result = ab.term_result(without, with_)
    assert (result.wins, result.losses, result.p_value, result.favours_mined) == (1, 0, 1.0, False)
    assert result.terms == 34 and result.error_without == 10 / 34 and result.error_with == 0  # descriptive, per term


def test_a_clip_is_right_only_when_all_its_terms_are():
    without = [{"id": f"c{i}", "terms_expected": ["a", "b"], "terms_hit": ["a"]} for i in range(20)]
    with_ = [{"id": f"c{i}", "terms_expected": ["a", "b"], "terms_hit": ["a", "b"]} for i in range(20)]
    result = ab.term_result(without, with_)
    assert (result.wins, result.losses, result.favours_mined) == (20, 0, True)
    half = [{"id": r["id"], "terms_expected": ["a", "b"], "terms_hit": []} for r in with_]
    assert ab.term_result(half, with_).wins == 20  # a clip that lost one of two terms is still wrong


def test_the_verdict_is_specific_to_the_boost_it_measured():
    for boost in ("light", "strong"):
        report = ab.run_ab(MODEL, boost, "current", yours=YOURS, mined=MINED, store=None, evaluate=fake_evaluate())
        assert report["verdict"]["boost"] == boost and report["verdict"]["passes"] is True
        assert "ship_at_normal" not in report["verdict"] and boost in report["verdict"]["text"]


def test_the_verdict_is_recorded_for_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    report = run(fake_evaluate())
    ab.write_verdict(report)
    from healing.vocab import apply as vocab_apply

    saved = json.loads(vocab_apply.verdict_path().read_text())
    assert saved["boost"] == "normal" and saved["passes"] is True and saved["ab_id"] == report["ab_id"]
    failed = run(fake_evaluate(bias_hurts_general=10))
    ab.write_verdict(failed)
    assert json.loads(vocab_apply.verdict_path().read_text())["passes"] is False  # the latest wins
