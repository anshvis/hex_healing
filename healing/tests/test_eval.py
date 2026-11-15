"""`healing eval` against a fake `hex` that transcribes by a per-model recipe and logs its calls."""

import json
import re
import stat
import sys
from pathlib import Path

import pytest
from dataset import BASE_MS, DAY, add_clip, review

from healing import labels, manifest, splits
from healing.cli import main
from healing.eval import run as eval_run
from healing.eval.bench import BenchError, run_benchmark
from healing.store import Store
from test_splits import sentences

CUTOFF = BASE_MS

FAKE = """#!{python}
import json, os, re, sys
from healing.manifest import term_count
from healing.wer import word_errors

args = sys.argv[1:]
if args == ["--version"]:
    print("hex fake 0.0")
    raise SystemExit(0)
assert args[0] == "benchmark-transcription", args
log = os.environ["FAKE_LOG"]
manifest_path = args[1]
def option(name):
    return args[args.index(name) + 1] if name in args else None
model = os.path.basename(option("--model"))
vocabulary = option("--vocabulary")
entry = {{"model": model, "manifest": manifest_path, "args": args}}
if vocabulary:
    entry["vocabulary"] = json.load(open(vocabulary))
open(log, "a").write(json.dumps(entry) + "\\n")
if vocabulary and os.environ.get("FAKE_REJECT_VOCABULARY"):
    sys.stderr.write("error: unexpected argument '--vocabulary' found\\n")
    raise SystemExit(2)
recipes = json.load(open(os.environ["FAKE_RECIPES"]))
recipe = recipes.get(model, {{}})
if recipe.get("fail"):
    sys.stderr.write("boom: " + recipe["fail"] + "\\n")
    raise SystemExit(1)
document = json.load(open(manifest_path))
clips, errors_total, words_total, expected_terms, recalled = [], 0, 0, 0, 0
for clip in document["clips"]:
    text = clip["expected"]
    for old, new in recipe.get("replace", {{}}).items():
        text = re.sub(r"\\b" + old + r"\\b", new, text)
    text += recipe.get("append", "")
    if recipe.get("strip"):
        text = re.sub(r"[^\\w\\s]", "", text).lower()
    errors, words = word_errors(clip["expected"], text)
    terms = clip.get("expected_terms", [])
    hit = sum(1 for t in terms if term_count(text, t) > 0)
    errors_total, words_total = errors_total + errors, words_total + words
    expected_terms, recalled = expected_terms + len(terms), recalled + hit
    clips.append({{"type": "clip", "id": clip["id"], "word_errors": errors, "reference_words": words,
                  "expected": clip["expected"], "transcript": text, "group": clip["group"],
                  "quality": {{"expected_terms": len(terms), "recalled_terms": hit}}}})
print(json.dumps({{"type": "model", "backend": "fake", "load_ms": 1.0}}))
for c in clips:
    print(json.dumps(c))
print(json.dumps({{"type": "summary", "clips": len(clips), "wer": errors_total / words_total,
                  "word_errors": errors_total, "reference_words": words_total, "median_rtf": 0.01,
                  "p95_clip_ms": 5.0, "load_ms": 1.0, "groups": {{}},
                  "quality": {{"term_recall": (recalled / expected_terms) if expected_terms else None}}}}))
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HEALING_HOME", str(home))
    binary = tmp_path / "hex"
    binary.write_text(FAKE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HEX_BIN", str(binary))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_RECIPES", str(tmp_path / "recipes.json"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))
    (tmp_path / "recipes.json").write_text("{}")
    models = {}
    for name in ("a.gguf", "b.gguf"):
        models[name] = tmp_path / name
        models[name].write_bytes(name.encode())

    class W:
        pass

    w = W()
    w.home, w.tmp, w.models = home, tmp_path, models
    w.recipes = lambda recipes: (tmp_path / "recipes.json").write_text(json.dumps(recipes))
    w.calls = lambda: [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    w.settings = lambda document: ((tmp_path / "hex-support").mkdir(exist_ok=True),
                                   (tmp_path / "hex-support" / "settings.json").write_text(json.dumps(document)))
    w.store = Store()
    yield w
    w.store.close()


def general_set(home):
    out = home / "sets" / "general"
    entries = [{"id": f"g{i}", "expected": text.upper(), "terms": []} for i, text in
               enumerate(["hello brave new world", "send it to the moon", "one two three four"])]
    manifest.write_set(out, entries, lambda clip: __import__("numpy").zeros(8000, dtype="int16"))


def dataset(store, strong=True, verified=True, build=True):
    """Post-cutoff clips: formatted verified references, and strong ones with an edited word."""
    phrases = sentences(10, seed=3)
    for i in range(6):
        cid = add_clip(store, i, raw=f"Please {phrases[i].split()[0]} the {phrases[i].split()[1]}, then {phrases[i].split()[2]}.",
                       ts=CUTOFF + DAY * (i + 1))
        if verified:
            review(store, cid)
    if strong:
        for i in range(10, 13):
            add_clip(store, i, raw=f"{phrases[i - 5]} kip", final=f"{phrases[i - 5]} kip.", ts=CUTOFF + DAY * (i + 1),
                     outcome="edited", edited=f"{phrases[i - 5]} kit")
    labels.label_all(store)
    if build:
        splits.build(store, cutoff_ms=CUTOFF, now=CUTOFF + 30 * DAY)


def evaluate(w, models, sets, **kwargs):
    return eval_run.evaluate([w.models[m] for m in models], sets, store=w.store, **kwargs)


def test_interleaves_models(world):
    general_set(world.home)
    dataset(world.store)
    evaluate(world, ["a.gguf", "b.gguf"], ["rolling", "general"], runs=2, vocabulary="none")
    order = [(Path(c["manifest"]).parent.name.split("-")[0], c["model"]) for c in world.calls()]
    assert order == [("rolling", "a.gguf"), ("rolling", "b.gguf")] * 2 + [("general", "a.gguf"), ("general", "b.gguf")] * 2


def test_format_error_rate_sees_stripped_punctuation(world):
    dataset(world.store)
    world.recipes({"a.gguf": {}, "b.gguf": {"strip": True}})
    document = evaluate(world, ["a.gguf", "b.gguf"], ["rolling"], vocabulary="none")
    exact, stripped = (document["scores"][str(world.models[m])]["rolling"] for m in ("a.gguf", "b.gguf"))
    assert exact["wer"] == stripped["wer"] == 0
    assert exact["format_error_rate"] == 0 and stripped["format_error_rate"] > 0.3


def test_format_is_not_scored_on_the_unformatted_general_set(world):
    general_set(world.home)
    document = evaluate(world, ["a.gguf"], ["general"], vocabulary="none")
    assert document["scores"][str(world.models["a.gguf"])]["general"]["format_error_rate"] is None


def test_edited_span_scores_only_edited_positions(world):
    dataset(world.store)
    world.recipes({"a.gguf": {"replace": {"kit": "kip"}}, "b.gguf": {}, "c.gguf": {"append": " and more"}})
    (world.tmp / "c.gguf").write_bytes(b"c")
    world.models["c.gguf"] = world.tmp / "c.gguf"
    document = evaluate(world, ["a.gguf", "b.gguf", "c.gguf"], ["edited"], vocabulary="none")
    spans = {m: document["scores"][str(world.models[m])]["edited"]["edited_span"] for m in ("a.gguf", "b.gguf", "c.gguf")}
    assert spans["a.gguf"] == {"errors": 3, "positions": 3}   # the corrected word is wrong in all three clips
    assert spans["b.gguf"] == {"errors": 0, "positions": 3}
    assert spans["c.gguf"] == {"errors": 0, "positions": 3}   # extra words elsewhere don't count here
    rows = document["per_clip"][str(world.models["a.gguf"])]["edited"]
    assert all(r["edited_errors"] == 1 and r["edited_positions"] == 1 for r in rows)


def test_the_edited_set_is_not_scored_on_its_whole_reference(world):
    """AC-7: a strong clip's untouched words are the model's own, so WER and format n/a there."""
    dataset(world.store)
    world.recipes({"a.gguf": {"append": " and more"}})
    document = evaluate(world, ["a.gguf"], ["edited", "rolling"], vocabulary="none")
    scores = document["scores"][str(world.models["a.gguf"])]
    assert all(scores["edited"][k] is None for k in ("wer", "word_errors", "ref_words", "format_error_rate"))
    assert scores["rolling"]["wer"] is not None
    for row in document["per_clip"][str(world.models["a.gguf"])]["edited"]:
        assert all(row[k] is None for k in ("word_errors", "reference_words", "format_errors", "format_words"))
    line = next(l for l in eval_run.table(document).splitlines() if l.startswith("edited"))
    assert line.split()[2:5] == ["3", "n/a", "n/a"] or line.split()[-5:-2] == ["3", "n/a", "n/a"]


def test_per_clip_records_kept_for_paired_stats(world):
    dataset(world.store)
    document = evaluate(world, ["a.gguf", "b.gguf"], ["rolling"], vocabulary="none")
    for model in world.models.values():
        rows = document["per_clip"][str(model)]["rolling"]
        assert len(rows) == document["sets"][0]["n"] == 6
        assert set(rows[0]) >= {"id", "word_errors", "reference_words", "format_errors", "format_words", "terms_hit", "transcript"}
    assert [r["id"] for r in document["per_clip"][str(world.models["a.gguf"])]["rolling"]] == \
           [r["id"] for r in document["per_clip"][str(world.models["b.gguf"])]["rolling"]]


def test_missing_set_is_na(world, capsys):
    general_set(world.home)
    document = evaluate(world, ["a.gguf"], ["rolling", "golden", "general", "edited"], vocabulary="none")
    assert [s["split_id"] for s in document["sets"]] == ["general-v1"]
    assert set(document["unavailable_sets"]) == {"rolling", "golden", "edited"}
    table = eval_run.table(document)
    lines = table.splitlines()
    assert any(l.startswith("golden    n/a") for l in lines) and any(l.startswith("rolling   n/a") for l in lines)
    assert any(l.startswith("general") and "n/a  " not in l[:14] for l in lines)
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--vocabulary", "none"]) == 0
    assert "n/a" in capsys.readouterr().out
    assert list((world.home / "evals").glob("eval-*.json"))


def test_golden_pending_is_na_with_the_count(world):
    dataset(world.store)
    document = evaluate(world, ["a.gguf"], ["golden"], vocabulary="none")
    assert "pending" in document["unavailable_sets"]["golden"]


def test_term_error_rate_from_term_recall(world):
    dataset(world.store)
    world.recipes({"a.gguf": {"replace": {"kit": "kip"}}, "b.gguf": {}})
    document = evaluate(world, ["a.gguf", "b.gguf"], ["edited", "rolling"], vocabulary="none")
    a, b = (document["scores"][str(world.models[m])] for m in ("a.gguf", "b.gguf"))
    assert a["edited"]["term_error_rate"] == 1.0 and b["edited"]["term_error_rate"] == 0.0
    assert a["rolling"]["term_error_rate"] is None  # no clip with expected terms
    assert [h["hit"] for h in a["edited"]["term_hits"]] == [False] * 3
    assert all(h["term"] == "kit" for h in b["edited"]["term_hits"]) and all(h["hit"] for h in b["edited"]["term_hits"])


def test_failure_names_model_and_set(world, capsys):
    general_set(world.home)
    dataset(world.store)
    world.recipes({"b.gguf": {"fail": "out of memory"}})
    with pytest.raises(eval_run.EvalError) as caught:
        evaluate(world, ["a.gguf", "b.gguf"], ["general", "rolling"], vocabulary="none")
    assert str(world.models["b.gguf"]) in str(caught.value) and "general" in str(caught.value) and "out of memory" in str(caught.value)
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--model", str(world.models["b.gguf"]), "--sets", "general", "--vocabulary", "none"]) == 1
    error = capsys.readouterr().err
    assert "b.gguf" in error and "general" in error


def test_the_eval_json_has_the_contract_fields(world):
    general_set(world.home)
    dataset(world.store)
    document = evaluate(world, ["a.gguf"], ["rolling", "general", "edited"], vocabulary="none")
    assert set(document) >= {"eval_id", "hex_sha", "models", "sets", "scores", "per_clip", "parent", "started_at_ms", "duration_ms"}
    assert document["models"][0]["gguf_sha256"] and document["models"][0]["gguf_path"].endswith("a.gguf")
    assert all(set(s) == {"set", "split_id", "manifest_sha256", "n"} for s in document["sets"])
    score = document["scores"][str(world.models["a.gguf"])]["rolling"]
    assert set(score) >= {"wer", "word_errors", "ref_words", "format_error_rate", "term_error_rate", "term_hits", "median_rtf", "p95_clip_ms"}
    path = eval_run.write(document)
    assert json.loads(path.read_text())["eval_id"] == document["eval_id"]


def test_the_cli_needs_a_model(capsys):
    assert main(["eval"]) == 2
    assert "--model" in capsys.readouterr().err


# --- vocabulary --------------------------------------------------------------------------------


def test_current_vocabulary_is_passed_as_a_file(world):
    general_set(world.home)
    world.settings({"vocabulary": ["Qorvyx", {"term": "Kit", "sounds_like": ["kip", "kid"]}],
                    "labs": {"vocabulary_biasing": True}, "vocabulary_boost": "strong"})
    saved = (world.tmp / "hex-support" / "settings.json").read_text()
    document = evaluate(world, ["a.gguf"], ["general"], vocabulary="current")
    call = world.calls()[0]
    assert call["vocabulary"] == [{"term": "Qorvyx", "sounds_like": []}, {"term": "Kit", "sounds_like": ["kip", "kid"]}]
    assert call["args"][call["args"].index("--boost") + 1] == "strong"
    assert document["vocabulary"] == {"mode": "current", "terms": 2, "boost": "strong"}
    assert (world.tmp / "hex-support" / "settings.json").read_text() == saved  # read-only
    assert not Path(call["args"][call["args"].index("--vocabulary") + 1]).exists()  # a temp file


@pytest.mark.parametrize("settings", [
    {"vocabulary": [], "labs": {"vocabulary_biasing": True}},
    {"vocabulary": ["Qorvyx"], "labs": {"vocabulary_biasing": False}},
    {"labs": {"vocabulary_biasing": True}},
    {"vocabulary": ["Qorvyx"], "labs": {"vocabulary_biasing": True}, "vocabulary_boost": "off"},
    None,
])
def test_nothing_is_passed_without_a_usable_vocabulary(world, settings):
    general_set(world.home)
    if settings is not None:
        world.settings(settings)
    messages = []
    evaluate(world, ["a.gguf"], ["general"], vocabulary="current", progress=messages.append)
    assert "--vocabulary" not in world.calls()[0]["args"]
    assert any(m.startswith("vocabulary: none passed") for m in messages)


def test_none_ignores_a_saved_vocabulary(world):
    general_set(world.home)
    world.settings({"vocabulary": ["Qorvyx"], "labs": {"vocabulary_biasing": True}})
    evaluate(world, ["a.gguf"], ["general"], vocabulary="none")
    assert "--vocabulary" not in world.calls()[0]["args"]


def test_a_binary_without_the_flag_suggests_none(world, monkeypatch, capsys):
    general_set(world.home)
    world.settings({"vocabulary": ["Qorvyx"], "labs": {"vocabulary_biasing": True}})
    monkeypatch.setenv("FAKE_REJECT_VOCABULARY", "1")
    with pytest.raises(eval_run.EvalError, match="--vocabulary none"):
        evaluate(world, ["a.gguf"], ["general"], vocabulary="current")
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "general"]) == 1
    error = capsys.readouterr().err
    assert "--vocabulary none" in error and "general" in error
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "general", "--vocabulary", "none"]) == 0


def test_a_real_error_is_not_mistaken_for_the_missing_flag(world):
    general_set(world.home)
    world.recipes({"a.gguf": {"fail": "something else"}})
    with pytest.raises(eval_run.EvalError) as caught:
        evaluate(world, ["a.gguf"], ["general"], vocabulary="none")
    assert "doesn't accept --vocabulary" not in str(caught.value)


def test_the_baseline_command_is_still_there(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["eval", "baseline", "--help"])
    assert stop.value.code == 0 and "--model" in capsys.readouterr().out


# --- negative paths ----------------------------------------------------------------------------


def _folder(world, kind):
    return Path(world.store.db.execute(
        "SELECT manifest_path FROM splits WHERE kind = ? ORDER BY created_at_ms DESC", (kind,)).fetchone()[0]).parent


def test_a_changed_manifest_is_refused(world):
    dataset(world.store)
    path = _folder(world, "rolling") / "manifest.json"
    path.write_text(path.read_text().replace("Please", "Pleas"))
    with pytest.raises(eval_run.EvalError, match="no longer matches its pin"):
        evaluate(world, ["a.gguf"], ["rolling"], vocabulary="none")


def test_an_incomplete_folder_is_refused(world):
    dataset(world.store)
    (_folder(world, "rolling") / "manifest.sha256").unlink()
    with pytest.raises(eval_run.EvalError, match="incomplete set"):
        evaluate(world, ["a.gguf"], ["rolling"], vocabulary="none")


def test_a_swapped_wav_with_a_rewritten_sidecar_and_pin_is_refused(world):
    import hashlib

    import numpy as np
    import soundfile

    dataset(world.store)
    folder = _folder(world, "rolling")
    rows = [json.loads(l) for l in (folder / "clips.jsonl").read_text().splitlines()]
    wav = folder / "audio" / f"{rows[0]['id']}.wav"
    soundfile.write(wav, np.ones(8000, dtype="int16"), 16_000, subtype="PCM_16")
    rows[0]["audio_sha256"] = hashlib.sha256(wav.read_bytes()).hexdigest()
    sidecar = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (folder / "clips.jsonl").write_text(sidecar)
    pin = (folder / "manifest.sha256").read_text().splitlines()
    (folder / "manifest.sha256").write_text(f"{pin[0]}\n{hashlib.sha256(sidecar.encode()).hexdigest()}  clips.jsonl\n")
    assert manifest.verify(folder)  # the folder is self-consistent...
    with pytest.raises(eval_run.EvalError, match="clips.jsonl hash"):  # ...but not the registry's pin
        evaluate(world, ["a.gguf"], ["rolling"], vocabulary="none")
    assert any("clips.jsonl hash" in p for p in splits.verify(world.store))


def test_an_empty_latest_split_is_na_not_an_older_one(world):
    dataset(world.store)
    splits.build(world.store, cutoff_ms=CUTOFF + 100 * DAY, now=CUTOFF + 31 * DAY)
    document = evaluate(world, ["a.gguf"], ["rolling"], vocabulary="none")
    assert document["sets"] == [] and "has no clips" in document["unavailable_sets"]["rolling"]


def test_a_frozen_golden_set_is_scored(world, monkeypatch):
    from healing import golden

    monkeypatch.setattr(splits, "GOLDEN_SIZE", 3)
    dataset(world.store, build=False)  # golden is confirmed first; a build then keeps clear of it
    ids = [r["clip_id"] for r in world.store.db.execute("SELECT clip_id FROM clips WHERE label = 'verified' LIMIT 3")]
    for clip in ids:
        assert golden.add(world.store, clip)
    splits.build(world.store, cutoff_ms=CUTOFF, now=CUTOFF + 30 * DAY)
    assert splits.golden_row(world.store.db)["frozen"]
    document = evaluate(world, ["a.gguf"], ["golden"], vocabulary="none")
    assert [s["split_id"] for s in document["sets"]] == ["golden-v1"] and document["sets"][0]["n"] == 3
    assert document["scores"][str(world.models["a.gguf"])]["golden"]["wer"] == 0


def test_the_cli_fails_when_no_requested_set_could_be_scored(world, capsys):
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "rolling,golden", "--vocabulary", "none"]) == 1
    assert "none of the requested sets" in capsys.readouterr().err


def test_an_empty_edited_set_without_strong_clips_says_so(world):
    dataset(world.store, strong=False)
    document = evaluate(world, ["a.gguf"], ["edited"], vocabulary="none")
    assert "no strong clips yet" in document["unavailable_sets"]["edited"]


def test_an_empty_edited_set_with_unplaced_strong_clips_says_so(world):
    dataset(world.store)
    world.store.db.execute("UPDATE clips SET edited_positions = NULL")
    world.store.db.execute("UPDATE splits SET manifest_path = NULL, n_clips = 0 WHERE kind = 'edited'")
    world.store.db.commit()
    document = evaluate(world, ["a.gguf"], ["edited"], vocabulary="none")
    assert "has no edited positions (3)" in document["unavailable_sets"]["edited"]


def test_the_general_set_has_no_format_counts(world):
    general_set(world.home)
    document = evaluate(world, ["a.gguf"], ["general"], vocabulary="none")
    rows = document["per_clip"][str(world.models["a.gguf"])]["general"]
    assert all(r["format_errors"] is None and r["format_words"] is None for r in rows)


def test_baseline_without_a_general_set_is_an_error_the_cli_reports(world, capsys):
    world.settings({})
    assert main(["eval", "baseline", "--model", str(world.models["a.gguf"])]) == 1
    assert "no general set" in capsys.readouterr().err


def test_nothing_scored_writes_no_eval_file(world, capsys):
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "rolling,golden", "--vocabulary", "none"]) == 1
    assert not list((world.home / "evals").glob("*.json")) if (world.home / "evals").exists() else True
    assert "n/a" in capsys.readouterr().out  # the table still says why


def test_runs_below_one_are_refused(world, capsys):
    general_set(world.home)
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "general", "--runs", "0", "--vocabulary", "none"]) == 2
    assert "--runs is at least 1" in capsys.readouterr().err
    with pytest.raises(eval_run.EvalError, match="at least 1"):
        evaluate(world, ["a.gguf"], ["general"], runs=0, vocabulary="none")
    assert not world.tmp.joinpath("calls.jsonl").exists()


def test_a_set_named_twice_is_scored_once(world):
    general_set(world.home)
    document = evaluate(world, ["a.gguf"], ["general", "general"], vocabulary="none")
    assert [s["set"] for s in document["sets"]] == ["general"] and len(world.calls()) == 1


def test_a_write_failure_is_an_error_not_a_traceback(world, capsys, monkeypatch):
    general_set(world.home)
    monkeypatch.setattr(eval_run, "write", lambda document: (_ for _ in ()).throw(OSError("disk full")))
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "general", "--vocabulary", "none"]) == 1
    assert "disk full" in capsys.readouterr().err


def test_an_edited_row_is_decided_by_the_scoring_policy_not_the_name(world):
    dataset(world.store)
    document = evaluate(world, ["a.gguf"], ["edited", "rolling"], vocabulary="none")
    lines = {l.split()[0]: l for l in eval_run.table(document).splitlines()[1:]}
    assert lines["edited"].split()[-1] != "n/a" and lines["rolling"].split()[-1] == "n/a"
    assert eval_run.SET_NAMES == tuple(eval_run.SCORING)


def test_repeated_words_in_the_edited_span_are_not_double_credited():
    """"kit ... kit" scored against a transcript with "kit" once credits one of the two edited positions."""
    from healing.wer import matched_positions, normalized_words

    matched = matched_positions(normalized_words("kit and the other kit"), normalized_words("kit and the other"), {0, 4})
    assert matched == [True, True, True, True, False]
    assert [matched[0], matched[4]].count(True) == 1
    both = matched_positions(normalized_words("kit kit"), normalized_words("kit"), {0, 1})
    assert both.count(True) == 1


def test_a_second_parent_over_the_same_window_holds_the_same_clips_and_eval_picks_each_its_own(world):
    dataset(world.store)
    splits.build(world.store, parent="v1", cutoff_ms=CUTOFF, now=CUTOFF + 30 * DAY)  # the same window
    members = lambda parent, kind: {r[0] for r in world.store.db.execute(  # noqa: E731
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.parent_version = ? AND s.kind = ?", (parent, kind))}
    assert members("current", "rolling") and members("current", "rolling") == members("v1", "rolling")  # a clip may sit in two parents' sets
    assert members("current", "edited") == members("v1", "edited") and splits.verify(world.store) == []
    current = evaluate(world, ["a.gguf"], ["rolling", "edited"], vocabulary="none")
    other = evaluate(world, ["a.gguf"], ["rolling", "edited"], vocabulary="none", parent="v1")
    ids = lambda document: {s["set"]: s["split_id"] for s in document["sets"]}  # noqa: E731
    assert ids(current)["rolling"].startswith("rolling-current-") and ids(other)["rolling"].startswith("rolling-v1-")
    assert ids(current)["edited"].startswith("edited-current-") and ids(other)["edited"].startswith("edited-v1-")
    assert other["parent"] == "v1"
    missing = evaluate(world, ["a.gguf"], ["rolling"], vocabulary="none", parent="v9")
    assert missing["sets"] == [] and "parent v9" in missing["unavailable_sets"]["rolling"]


def test_the_cli_scores_the_parent_it_is_given(world, capsys):
    dataset(world.store)
    splits.build(world.store, parent="v1", cutoff_ms=CUTOFF, now=CUTOFF + 30 * DAY)
    world.store.close()
    assert main(["eval", "--model", str(world.models["a.gguf"]), "--sets", "rolling", "--parent", "v1", "--vocabulary", "none"]) == 0
    document = json.loads(next((world.home / "evals").glob("eval-*.json")).read_text())
    assert document["parent"] == "v1" and document["sets"][0]["split_id"].startswith("rolling-v1-")
    world.store = Store()  # the fixture closes it


def test_one_model_under_two_spellings_is_named_twice(world):
    alias = world.tmp / "alias.gguf"
    alias.symlink_to(world.models["a.gguf"])
    with pytest.raises(eval_run.EvalUsageError, match="named twice"):
        eval_run.evaluate([world.models["a.gguf"], alias], ["general"], vocabulary="none")


def test_the_edited_span_lives_in_the_set_scores(world):
    dataset(world.store)
    document = evaluate(world, ["a.gguf"], ["edited", "rolling"], vocabulary="none")
    scores = document["scores"][str(world.models["a.gguf"])]
    assert "edited_span" in scores["edited"] and "edited_span" not in scores["rolling"] and "edited_span" not in document
