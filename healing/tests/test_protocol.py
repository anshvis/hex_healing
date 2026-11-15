import json
from pathlib import Path

from healing import protocol
from healing.cli import main


def test_plan_is_deterministic_and_balanced():
    plan = protocol.build_plan(13)
    assert plan == protocol.build_plan(13)
    assert len(plan["runs"]) == 60
    for app in protocol.APPS:
        runs = [r for r in plan["runs"] if r["app"] == app]
        seeded = [r for r in runs if r["seeded"]]
        assert len(runs) == 10 and len(seeded) == 5
        assert sum(r["send"] for r in runs) == 2 and all(r["seeded"] for r in runs if r["send"])
        assert all(r["replacement"] for r in seeded) and not any(r["replacement"] for r in runs if not r["seeded"])


def test_plan_command_writes_runbook(tmp_path, capsys):
    assert main(["protocol", "corrections", "plan", "--dir", str(tmp_path)]) == 0
    text = (tmp_path / "RUNBOOK.md").read_text()
    assert "HEX_AUDIO_FIXTURE" in text and "TextEdit" in text and "X-3" in text
    assert json.loads((tmp_path / "plan.json").read_text())["version"] == 1


class FakeDriver(protocol.Driver):
    def __init__(self):
        super().__init__(Path("hex"))
        self.calls = []
        self.now = 1000

    def sleep(self, seconds):
        self.now += int(seconds * 1000)

    def clock(self):
        return self.now

    def hex_dictation(self, verb):
        self.calls.append(("hex", verb))

    def activate(self, app):
        self.calls.append(("activate", app))

    def select_last_word(self):
        self.calls.append(("select",))

    def type_text(self, text):
        self.calls.append(("type", text))

    def press_return(self):
        self.calls.append(("return",))

    def clear_field(self):
        self.calls.append(("clear",))


def test_run_drives_and_resumes(tmp_path):
    plan = protocol.build_plan(1, {"TextEdit": "x"})
    driver = FakeDriver()
    path = tmp_path / "results.json"
    assert protocol.run_plan(plan, path, driver) == 10
    assert sum(c == ("select",) for c in driver.calls) == 5
    assert sum(c == ("return",) for c in driver.calls) == 2
    assert protocol.run_plan(plan, path, driver) == 0  # already done


def _history(plan, results, captured_every=1, false_edit=False):
    entries = []
    by_run = {(r["app"], r["index"]): r for r in plan["runs"]}
    for n, res in enumerate(results):
        run = by_run[(res["app"], res["index"])]
        correction = {"outcome": "unchanged", "observed_at_ms": 1}
        if run["seeded"] and n % captured_every == 0:
            correction = {"outcome": "edited", "observed_at_ms": 1, "edited_text": f"Send it to {run['replacement']}."}
        elif false_edit and not run["seeded"]:
            correction = {"outcome": "edited", "observed_at_ms": 1, "edited_text": "Something else."}
        entries.append({"id": n, "timestamp_ms": res["start_ms"] + 5000, "final_text": "Send it to Kip.", "correction": correction})
    return entries


def _run_all(tmp_path):
    plan = protocol.build_plan(1)
    results_path = tmp_path / "r.json"
    protocol.run_plan(plan, results_path, FakeDriver())
    return plan, json.loads(results_path.read_text())


def test_score_passes_and_fails(tmp_path):
    plan, results = _run_all(tmp_path)
    good = protocol.score(plan, results, _history(plan, results))
    assert good["recall"] == 1.0 and good["false_edit_rate"] == 0 and good["pass"]
    poor = protocol.score(plan, results, _history(plan, results, captured_every=10**6))
    assert poor["recall"] == 0 and poor["x3"] and not poor["pass"]
    noisy = protocol.score(plan, results, _history(plan, results, false_edit=True))
    assert noisy["false_edit_rate"] == 1.0 and not noisy["pass"]
    assert "TextEdit" in protocol.format_score(good)


def test_score_needs_the_seeded_word_in_the_span(tmp_path):
    plan, results = _run_all(tmp_path)
    entries = _history(plan, results)
    for e in entries:
        if e["correction"]["outcome"] == "edited":
            e["correction"]["edited_text"] = "Send it to Kip."  # edited, but not what was typed
    assert protocol.score(plan, results, entries)["recall"] == 0


def test_a_partial_run_is_reported_but_never_passes(tmp_path):
    plan, results = _run_all(tmp_path)
    one_seeded = [r for r in results if next(p for p in plan["runs"] if (p["app"], p["index"]) == (r["app"], r["index"]))["seeded"]][:1]
    partial = protocol.score(plan, one_seeded, _history(plan, one_seeded))
    assert partial["recall"] == 1.0 and partial["false_edit_rate"] == 0
    assert not partial["complete"] and not partial["pass"] and not partial["x3"]
    assert partial["missing_runs"] == 59
    assert "INCOMPLETE" in protocol.format_score(partial)

    # Poor recall on a partial run is not X-3 either.
    poor = protocol.score(plan, results[:10], _history(plan, results[:10], captured_every=10**6))
    assert not poor["pass"] and not poor["x3"]


def test_a_plan_without_every_app_never_passes(tmp_path):
    plan = protocol.build_plan(1, {"TextEdit": "x", "Notes": "y"})
    results_path = tmp_path / "r.json"
    protocol.run_plan(plan, results_path, FakeDriver())
    results = json.loads(results_path.read_text())
    result = protocol.score(plan, results, _history(plan, results))
    assert result["missing_runs"] == 0 and result["untested_apps"]
    assert not result["complete"] and not result["pass"]
    assert "untested" in protocol.format_score(result)


def test_every_app_and_control_run_done_can_pass(tmp_path):
    plan, results = _run_all(tmp_path)
    result = protocol.score(plan, results, _history(plan, results))
    assert result["complete"] and result["pass"] and result["untested_apps"] == []


def test_the_field_is_cleared_after_every_run_before_the_next_dictation(tmp_path):
    plan = protocol.build_plan(1, {"TextEdit": "x"})
    driver = FakeDriver()
    protocol.run_plan(plan, tmp_path / "r.json", driver)
    calls = driver.calls
    starts = [i for i, c in enumerate(calls) if c == ("hex", "start")]
    assert len(starts) == 10 and sum(c == ("clear",) for c in calls) == 10
    # Between one dictation and the next there is exactly one clear.
    for before, after in zip(starts, starts[1:]):
        assert calls[before:after].count(("clear",)) == 1
    # The last thing a run does is clear, so the next paste lands in an empty field.
    assert calls[-1] == ("clear",)
