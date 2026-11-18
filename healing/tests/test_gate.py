"""T7.1: `healing gate`, the promotion gate (AC-13, D-12).

The checks are judged on synthetic `healing eval` documents (per-clip rows, as `eval/run.py` writes
them), so each one can be made to fail alone. `run` is exercised against a real registry in a temp
`HEALING_HOME` with the eval and the streaming probe standing in; no `hex` runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from healing import gate
from healing.cli import main
from healing.registry import objects
from healing.registry import store as registry

CUR, CAND, V0 = "a" * 64, "b" * 64, "c" * 64
STREAMS = gate.StreamSupport(True)
ROLLING_PIN = "f" * 64
GOLDEN_PIN = "e" * 64


# ---- synthetic evals ------------------------------------------------------------------------------

def rows(errors, words, *, right=None, format_extra=1, formatted=True, ms=None, audio_ms=None, prefix="r"):
    """One model's per-clip rows on a set: `errors[i]` word errors of `words` reference words; `right[i]`
    True/False for a clip expecting the term `Qorvyx` (None: no term)."""
    out = []
    for i, e in enumerate(errors):
        r = None if right is None or i >= len(right) else right[i]
        out.append({
            "id": f"{prefix}{i:04d}", "word_errors": int(e), "reference_words": words,
            "terms_expected": [] if r is None else ["Qorvyx"], "terms_hit": ["Qorvyx"] if r else [],
            "format_errors": int(e) + format_extra if formatted else None, "format_words": words if formatted else None,
            "inference_ms": 300.0 if ms is None else ms[i], "audio_ms": 12_000.0 if audio_ms is None else audio_ms[i],
        })
    return out


def world(*, clips=100, words=20, gain=0.35, seed=1):
    """A true ~3-pt rolling gain on 2,000 words, 25 term clips the candidate gets right and the current
    model gets 13 of, and a general set (5,000 words) both models score identically on."""
    rng = np.random.default_rng(seed)
    cur = rng.binomial(words, 0.10, clips)
    cand = cur - rng.binomial(cur, gain) + rng.binomial(words, 0.005, clips)
    general = rng.binomial(10, 0.08, 500)
    return {
        "rolling": {CUR: rows(cur, words, right=[i % 2 == 0 for i in range(25)]),
                    CAND: rows(cand, words, right=[True] * 25)},
        "general": {CUR: rows(general, 10, formatted=False, prefix="g"),
                    CAND: rows(general, 10, formatted=False, prefix="g")},
    }


def document(sets: dict, *, golden_sha=GOLDEN_PIN) -> dict:
    shas = sorted({sha for by_model in sets.values() for sha in by_model})
    path = {sha: f"/models/{sha[:4]}.gguf" for sha in shas}
    return {
        "eval_id": "eval-fixture",
        "models": [{"gguf_path": path[s], "gguf_sha256": s} for s in shas],
        "sets": [{"set": name, "split_id": name, "n": len(next(iter(by.values()))),
                  "manifest_sha256": golden_sha if name == "golden" else ROLLING_PIN} for name, by in sets.items()],
        "unavailable_sets": {} if "golden" in sets else {"golden": "golden is pending (40 of 200 confirmed)"},
        "per_clip": {path[s]: {name: by[s] for name, by in sets.items() if s in by} for s in shas},
        "scores": {path[s]: {} for s in shas},
    }


def judge(sets, *, support=STREAMS, against=CUR, margins=None, allow_thin=False, golden_pin=None):
    v0 = V0 if V0 in sets["general"] else against  # without v0's own rows, current is v0
    return gate.judge(document(sets), candidate=CAND, against=against, v0=v0,
                      support=support, margins=margins or gate.Margins.load(), golden_pin=golden_pin,
                      allow_thin=allow_thin)


def check(verdict, name) -> dict:
    return next(c for c in verdict.as_records() if c["name"] == name)


# ---- the checks -----------------------------------------------------------------------------------

def test_real_gain_passes():
    verdict = judge(world())
    assert verdict.failing == [] and verdict.passed
    rolling = check(verdict, "rolling_wer")
    assert -0.045 < rolling["estimate"] < -0.02 and rolling["ci_high"] < 0
    assert check(verdict, "golden_wer")["status"] == "skipped" and check(verdict, "golden_wer")["pass"] is None
    assert check(verdict, "terms")["p"] < 0.05


def test_noise_sized_gain_fails():
    rng = np.random.default_rng(7)
    cur = rng.binomial(20, 0.10, 30)
    cand = cur.copy()
    nonzero = np.flatnonzero(cand)
    cand[nonzero[:6]] -= 1  # 6 clips better, 4 worse: 2 fewer errors in 600 words, a 0.33-pt "improvement"
    cand[nonzero[6:10]] += 1
    sets = world()
    current = rows(cur, 20, right=[i % 2 == 0 for i in range(25)])
    candidate = [{**r, "format_errors": c["format_errors"]} for r, c in zip(rows(cand, 20, right=[True] * 25), current)]
    sets["rolling"] = {CUR: current, CAND: candidate}
    verdict = judge(sets, allow_thin=True)  # 600 words is thin; only rolling_wer is under test here
    rolling = check(verdict, "rolling_wer")
    assert rolling["estimate"] == pytest.approx(-2 / 600)
    assert rolling["ci_low"] < 0 < rolling["ci_high"]
    assert verdict.failing == ["rolling_wer"]


def test_stripped_punctuation_fails_formatting():
    sets = world()
    sets["rolling"][CAND] = [{**r, "format_errors": r["format_errors"] + 4} for r in sets["rolling"][CAND]]
    verdict = judge(sets)
    assert verdict.failing == ["formatting"]
    assert check(verdict, "formatting")["ci_low"] > 0.005


def test_non_streaming_fails():
    verdict = judge(world(), support=gate.StreamSupport(False, "turn on Labs › Stream imported models"))
    assert verdict.failing == ["streaming"]
    assert "Stream imported models" in check(verdict, "streaming")["note"]


def test_slower_long_clips_fail_streaming():
    sets = world()
    sets["rolling"][CAND] = [{**r, "inference_ms": r["inference_ms"] + 100} for r in sets["rolling"][CAND]]
    verdict = judge(sets)
    assert verdict.failing == ["streaming"]
    assert check(verdict, "streaming")["estimate"] == pytest.approx(100)


def _fail_rolling(sets):
    sets["rolling"][CAND] = [{**r, "word_errors": c["word_errors"], "format_errors": c["format_errors"]}
                             for r, c in zip(sets["rolling"][CAND], sets["rolling"][CUR])]


def _fail_terms(sets):
    sets["rolling"][CAND] = [{**r, "terms_hit": c["terms_hit"]} for r, c in zip(sets["rolling"][CAND], sets["rolling"][CUR])]


def _fail_general(sets):
    sets["general"][CAND] = [{**r, "word_errors": r["word_errors"] + (i < 40)} for i, r in enumerate(sets["general"][CAND])]


def _fail_formatting(sets):
    sets["rolling"][CAND] = [{**r, "format_errors": r["format_errors"] + 4} for r in sets["rolling"][CAND]]


def _thin(sets):
    rng = np.random.default_rng(3)
    cur = rng.binomial(20, 0.12, 70)  # 1,400 words, a large gain
    sets["rolling"] = {CUR: rows(cur, 20, right=[i % 2 == 0 for i in range(25)]),
                       CAND: rows(cur - rng.binomial(cur, 0.6), 20, right=[True] * 25)}


def _fail_golden(sets):
    golden = np.random.default_rng(5).binomial(20, 0.08, 100)
    worse = rows(golden + (np.arange(100) < 40), 20, prefix="d", format_extra=0)  # 2 pt worse, formatting the same
    sets["golden"] = {CUR: rows(golden, 20, prefix="d"),
                      CAND: [{**r, "format_errors": c["format_errors"]} for r, c in zip(worse, rows(golden, 20, prefix="d"))]}


@pytest.mark.parametrize("name,spoil,extra", [
    ("evidence", _thin, {}),
    ("rolling_wer", _fail_rolling, {}),
    ("terms", _fail_terms, {}),
    ("general_wer", _fail_general, {}),
    ("golden_wer", _fail_golden, {"golden_pin": {"sha256": GOLDEN_PIN, "n": 200}}),
    ("formatting", _fail_formatting, {}),
    ("drift_guard", None, {}),
    ("streaming", None, {"support": gate.StreamSupport(False, "not imported from a streaming model")}),
])
def test_each_check_fails_alone(name, spoil, extra):
    sets = world()
    against = CUR
    if spoil:
        spoil(sets)
    if name == "drift_guard":
        # current is v1, 1.5 pt worse on general than v0; the candidate matches current there
        v0 = np.array([r["word_errors"] for r in sets["general"][CUR]])
        v0[np.flatnonzero(v0)[:75]] -= 1
        sets["general"][V0] = rows(v0, 10, formatted=False, prefix="g")
    verdict = judge(sets, against=against, **extra)
    assert verdict.failing == [name]
    assert not verdict.passed


def test_every_check_is_reported():
    verdict = judge(world())
    assert [c["name"] for c in verdict.as_records()] == list(gate.CHECKS)
    for record in verdict.as_records():
        assert set(record) >= {"name", "estimate", "ci_low", "ci_high", "p", "threshold", "pass"}


def test_thin_evidence_refuses():
    sets = world()
    _thin(sets)
    verdict = judge(sets)
    assert verdict.refused and not verdict.passed
    assert check(verdict, "evidence")["estimate"] == 1400
    # allowed for fixtures only: recorded as skipped, never as a pass
    allowed = judge(sets, allow_thin=True)
    assert allowed.passed and check(allowed, "evidence")["status"] == "skipped"


def test_golden_pin_mismatch_refuses():
    sets = world()
    _fail_golden(sets)
    with pytest.raises(gate.GateRefused, match="G-10"):
        judge(sets, golden_pin={"sha256": "0" * 64, "n": 200})
    with pytest.raises(gate.GateRefused, match="not pinned"):
        judge(sets, golden_pin=None)


def test_seeded_deterministic(tmp_path, monkeypatch):
    first, second = judge(world()), judge(world())
    assert first.as_records() == second.as_records()
    # every interval is drawn with gate.toml's seed and resample count, never a default
    seen = []
    real = gate.stats.paired_bootstrap
    monkeypatch.setattr(gate.stats, "paired_bootstrap",
                        lambda a, b, **kw: seen.append(kw) or real(a, b, **kw))
    text = gate.Margins.load().path.read_text().replace("seed = 13", "seed = 14").replace("resamples = 10000", "resamples = 2000")
    (tmp_path / "gate.toml").write_text(text)
    judge(world(), margins=gate.Margins.load(tmp_path / "gate.toml"))
    assert seen and all(kw == {"resamples": 2000, "seed": 14, "confidence": 0.95} for kw in seen)
    assert gate.Margins.load().resamples == 10_000


def test_margins_match_d12():
    m = gate.Margins.load()
    assert (m.min_rolling_words, m.min_term_clips, m.terms_alpha) == (1500, 20, 0.05)
    assert (m.rolling_wer_ci_high, m.general_wer_ci_high, m.golden_wer_ci_high, m.formatting_ci_high) == (0, 0.005, 0.005, 0.005)
    assert (m.drift_guard, m.streaming_p95_ms) == (0.01, 50)


# ---- the command ----------------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))  # never the real one
    shas = {}
    for name in ("v0", "v1"):
        source = tmp_path / f"{name}.gguf"
        source.write_bytes(b"GGUF" + name.encode() * 100)
        shas[name] = objects.put(source)
    registry.append(
        {"event": "registered", "version": "v0", "parent": None, "base": {"catalog_id": "parakeet-unified-en-0.6b"},
         "dataset": None, "artifacts": {"gguf": shas["v0"]}},
        {"event": "promoted", "version": "v0", "reason": "base"},
        {"event": "registered", "version": "v1", "parent": "v0", "dataset": None, "artifacts": {"gguf": shas["v1"]}},
    )
    return shas


def fake_evaluate(sets_for, calls):
    """Stands in for `healing eval`: models are matched to the fixture's CUR/CAND rows by order."""
    def evaluate(models: list[Path], set_names, parent):
        calls.append({"models": [m.name for m in models], "sets": set_names, "parent": parent})
        shas = [objects.sha256_file(m) for m in models]
        doc = document({name: {shas[0]: by[CUR], shas[1]: by[CAND]} for name, by in sets_for.items()})
        doc["eval_id"] = "eval-test"
        return doc
    return evaluate


def test_margins_sha_recorded(home):
    calls = []
    sets = world()
    _fail_terms(sets)
    code = gate.run("v1", evaluate=fake_evaluate(sets, calls), support=STREAMS, say=lambda line: None)
    assert code == gate.EXIT_FAIL == 7
    events = registry.read_events()
    kinds = [e["event"] for e in events]
    record, rejected = events[kinds.index("gate")], events[kinds.index("gate") + 1]
    assert record["event"] == "gate" and record["version"] == "v1" and record["against"] == "v0"
    assert record["margins_sha"] == hashlib.sha256(gate.Margins.load().path.read_bytes()).hexdigest()
    assert record["eval_id"] == "eval-test" and record["pass"] is False
    assert [c["name"] for c in record["checks"]] == list(gate.CHECKS)
    assert rejected == {**rejected, "event": "rejected", "version": "v1", "reason": "gate: terms"}
    assert registry.load().versions["v1"].state == "rejected"
    # v0 is offered under its catalog file name, so the benchmark pads it as Hex does
    assert calls[0]["models"] == ["parakeet-unified-en-0.6b-Q8_0.gguf", Path(objects.path(home["v1"])).name]
    assert calls[0]["parent"] == "v0" and "rolling" in calls[0]["sets"]


def test_pass_exits_zero_and_stays_candidate(home):
    code = gate.run("v1", evaluate=fake_evaluate(world(), []), support=STREAMS, say=lambda line: None)
    assert code == 0
    state = registry.load()
    assert state.versions["v1"].state == "candidate"
    assert state.versions["v1"].last_gate["pass"] is True


def test_thin_evidence_is_recorded_but_not_rejected(home):
    sets = world()
    _thin(sets)
    lines = []
    code = gate.run("v1", evaluate=fake_evaluate(sets, []), support=STREAMS, say=lines.append)
    assert code == gate.EXIT_FAIL
    version = registry.load().versions["v1"]
    assert version.state == "candidate" and version.last_gate["refused"] is True
    assert registry.read_events()[-1]["event"] == "gate"
    assert any("healing review" in line for line in lines)


def test_rejected_candidate_is_not_gated_again(home):
    sets = world()
    _fail_terms(sets)
    gate.run("v1", evaluate=fake_evaluate(sets, []), support=STREAMS, say=lambda line: None)
    with pytest.raises(gate.GateError, match="rejected"):
        gate.run("v1", evaluate=fake_evaluate(world(), []), support=STREAMS, say=lambda line: None)


def test_allow_thin_evidence_refused_by_the_cli(home, capsys):
    """The override is a library parameter the tests pass; the command line refuses it unconditionally,
    whatever the environment says (PYTEST_CURRENT_TEST is set right now)."""
    before = registry.read_events()
    assert main(["gate", "v1", "--allow-thin-evidence"]) == 2
    assert "fixtures" in capsys.readouterr().err
    assert registry.read_events() == before


def test_gate_rejudges_a_saved_eval(home, tmp_path):
    sets = world()
    doc = fake_evaluate(sets, [])([objects.path(home["v0"]), objects.path(home["v1"])], [], "v0")
    saved = tmp_path / "eval.json"
    saved.write_text(json.dumps(doc))
    code = gate.run("v1", eval_path=saved, support=STREAMS, say=lambda line: None)
    assert code == 0
    assert registry.load().versions["v1"].last_gate["eval_id"] == "eval-test"


# ---- the live streaming probe (T7.4) ---------------------------------------------------------------

FAKE_HEX = """#!{python}
import json, os, sys
if os.environ.get("FAKE_HEX_DOWN"):
    sys.exit(3)
assert sys.argv[1:] == ["model", "list", "--json"], sys.argv
print(json.dumps({{"version": 4, "status": "models", "models": json.loads(os.environ["FAKE_MODELS"])}}, indent=2))
"""


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A registry whose v0 and v1 are real GGUF headers, a fake `hex` listing models, and Hex's settings."""
    from healing import gguf

    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    support = tmp_path / "hex-support"
    support.mkdir()
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(support))  # never the real one
    binary = tmp_path / "hex"
    binary.write_text(FAKE_HEX.format(python=__import__("sys").executable))
    binary.chmod(0o755)
    shas = {}
    for name, variant in (("v0", "unified"), ("v1", "unified"), ("v2", "tdt-v2")):
        path = tmp_path / f"{name}.gguf"
        gguf.write(path, {"general.architecture": "parakeet", "stt.variant": variant, "general.name": name},
                   {"t": name.encode() * 32})
        shas[name] = objects.put(path)
    registry.append(
        {"event": "registered", "version": "v0", "parent": None, "base": {"catalog_id": "parakeet-unified-en-0.6b"},
         "dataset": None, "artifacts": {"gguf": shas["v0"]}},
        {"event": "promoted", "version": "v0", "reason": "base"},
        {"event": "registered", "version": "v1", "parent": "v0", "dataset": None, "artifacts": {"gguf": shas["v1"]}},
        {"event": "registered", "version": "v2", "parent": "v0", "dataset": None, "artifacts": {"gguf": shas["v2"]}},
    )

    def setup(*, lab: bool, models: list[dict], down: bool = False):
        (support / "settings.json").write_text(json.dumps({"labs": {"stream_imported_models": lab}}))
        monkeypatch.setenv("FAKE_MODELS", json.dumps(models))
        if down:
            monkeypatch.setenv("FAKE_HEX_DOWN", "1")
        state = registry.load()
        return lambda name: gate.stream_support(binary, state.versions[name], state.versions["v0"])

    return setup, shas


CATALOG_STREAMS = {"id": "parakeet-unified-en-0.6b", "streams": True}


def test_stream_probe_lab_off_says_turn_it_on(live):
    setup, _ = live
    found = setup(lab=False, models=[CATALOG_STREAMS])("v1")
    assert not found.streams and "hex labs on stream-imported-models" in found.note


def test_stream_probe_matching_header_streams_like_current(live):
    setup, _ = live
    found = setup(lab=True, models=[CATALOG_STREAMS])("v1")
    assert found.streams and "smoke stream runs when promote loads it" in found.note


def test_stream_probe_other_variant_does_not_stream(live):
    setup, _ = live
    assert not setup(lab=True, models=[CATALOG_STREAMS])("v2").streams


def test_stream_probe_trusts_hex_for_an_imported_candidate(live):
    setup, shas = live
    listed = {"id": objects.hex_model_id(shas["v1"]), "streams": False}
    found = setup(lab=True, models=[CATALOG_STREAMS, listed])("v1")
    assert not found.streams and "streams: false" in found.note
    listed["streams"] = True
    assert setup(lab=True, models=[CATALOG_STREAMS, listed])("v1").streams


def test_stream_probe_needs_a_running_hex(live):
    setup, _ = live
    with pytest.raises(gate.GateError, match="isn't running") as error:
        setup(lab=True, models=[], down=True)("v1")
    assert error.value.code == 3


def test_eval_keeps_per_clip_latency(tmp_path):
    from healing.eval import run as eval_run
    from healing.eval.bench import BenchRun

    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"clips": [{"id": "c1", "expected": "Hello there."}]}))
    (folder / "clips.jsonl").write_text(json.dumps({"id": "c1"}) + "\n")
    clip = {"id": "c1", "transcript": "Hello there.", "word_errors": 0, "reference_words": 2,
            "inference_ms": 412.5, "audio_ms": 12_800.0}
    summary = {"wer": 0.0, "word_errors": 0, "reference_words": 2, "median_rtf": 0.03, "p95_clip_ms": 412.5,
               "quality": {"term_recall": None}}
    ref = eval_run.SetRef("rolling", "rolling-x", folder, "0" * 64, 1)
    _, per_clip = eval_run.score_set(ref, [BenchRun({}, [clip], summary)])
    assert per_clip[0]["inference_ms"] == 412.5 and per_clip[0]["audio_ms"] == 12_800.0


def test_gate_refuses_an_eval_it_cannot_judge(home, tmp_path):
    doc = fake_evaluate(world(), [])([objects.path(home["v0"]), objects.path(home["v1"])], [], "v0")
    for change, message in (({"vocabulary": {"mode": "none"}}, "vocabulary"), ({"parent": "v3"}, "holdouts"),
                            ({"sets": []}, "none of the sets")):
        saved = tmp_path / "eval.json"
        saved.write_text(json.dumps({**doc, **change}))
        with pytest.raises(gate.GateError, match=message):
            gate.run("v1", eval_path=saved, support=STREAMS, say=lambda line: None)
    assert not registry.load().versions["v1"].gates


# ---- review findings ------------------------------------------------------------------------------


def test_pinned_golden_missing_from_the_eval_refuses():
    """Once golden is pinned, an eval without it (made before the freeze, or with fewer sets) is refused."""
    with pytest.raises(gate.GateRefused, match="pinned"):
        judge(world(), golden_pin={"sha256": GOLDEN_PIN, "n": 200})


def test_no_long_clips_fails_streaming():
    sets = world()
    for by in sets.values():
        for name in by:
            by[name] = [{**r, "audio_ms": 5_000.0} for r in by[name]]
    verdict = judge(sets)
    assert verdict.failing == ["streaming"]
    assert "unmeasured" in check(verdict, "streaming")["note"]


def test_gate_records_the_models_it_judged(home):
    gate.run("v1", evaluate=fake_evaluate(world(), []), support=STREAMS, say=lambda line: None)
    models = registry.load().versions["v1"].last_gate["models"]
    assert models == {"candidate": {"version": "v1", "gguf_sha256": home["v1"]},
                      "against": {"version": "v0", "gguf_sha256": home["v0"]},
                      "v0": {"version": "v0", "gguf_sha256": home["v0"]}}


def test_restore_between_eval_and_record_refuses(home):
    """A restore that swaps a judged model's GGUF while the eval runs leaves the decision about other
    bytes: the record is refused under the registry lock."""
    inner = fake_evaluate(world(), [])

    def evaluate(models, set_names, parent):
        doc = inner(models, set_names, parent)
        with registry.lock():
            registry.append({"event": "restored", "version": "v1", "method": "wer_parity", "rebuilt_sha256": "d" * 64})
        return doc

    with pytest.raises(registry.RegistryError, match="v1"):
        gate.run("v1", evaluate=evaluate, support=STREAMS, say=lambda line: None)
    assert not registry.load().versions["v1"].gates
