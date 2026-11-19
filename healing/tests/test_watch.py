"""T7.3: `healing watch` rolls a promotion back only when it is significantly worse (AC-14, D-19).

A real registry and dataset in a temp `HEALING_HOME`, T7.2's fake `hex` (it keeps what is imported and
selected in a JSON file) and a fake `healing eval` that scores the watch set's own clips with fixed
per-clip errors, so each decision follows from numbers the test chose. v0 → v1 → v2 are promoted with
`healing promote`; the watch is about v2 against v1. No real Hex runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import add_clip, review
from healing import gate, labels, schedule, ship, stats, status, watch
from healing.cli import main
from healing.registry import objects
from healing.registry import store as registry
from healing.store import Store
from healing.train import auto
from test_ship import CATALOG, candidate, events, hex_id, hexbin, passing_gate  # noqa: F401  (hexbin: T7.2's fake-hex fixture)
from test_ship import promote as ship_promote

DAY = 86_400_000
WORDS = 20  # reference words per clip


class World:
    def __init__(self, hexbin, store: Store, t0: int):
        self.hex, self.store, self.t0 = hexbin, store, t0
        self.calls: list[list[str]] = []  # the clip ids each fake eval scored
        self.restored: list[str] = []
        self.next_clip = 1

    def clips(self, count: int, *, after: bool = True, strategy: str = "random", action: str = "accept") -> list[str]:
        """`count` clips of 20 reference words, recorded after (or before) v2's promotion, reviewed."""
        ids = []
        for _ in range(count):
            n = self.next_clip
            self.next_clip += 1
            text = " ".join(f"w{n}x{j}" for j in range(WORDS))
            when = self.t0 + n * 1000 if after else self.t0 - DAY - n * 1000
            cid = add_clip(self.store, n, raw=text, ts=when, hex_id=n)
            review(self.store, cid, action=action, text=text, strategy=strategy, at=when + 10)
            ids.append(cid)
        labels.refresh(self.store)
        return ids

    def scored(self, previous: list[int], promoted: list[int], format_extra: int = 0):
        """A fake `healing eval`: clip i of the watch set (in time order) has `previous[i]` word errors
        for v1 and `promoted[i]` for v2, of 20 words each. Formatting errors follow word errors (v2 has
        `format_extra` more on every clip)."""
        state = registry.load()
        errors = {state.get("v1").gguf: previous, state.get("v2").gguf: promoted}
        extra = {state.get("v1").gguf: 0, state.get("v2").gguf: format_extra}

        def evaluate(paths: list[Path], ref) -> dict:
            clips = json.loads(ref.manifest.read_text())["clips"]
            self.calls.append([c["id"] for c in clips])
            assert len(clips) == len(previous) == len(promoted)
            models, per_clip = [], {}
            for path in paths:
                sha = objects.sha256_file(path)
                models.append({"gguf_path": str(path), "gguf_sha256": sha})
                per_clip[str(path)] = {"rolling": [
                    {"id": c["id"], "word_errors": int(e), "reference_words": WORDS, "format_errors": int(e) + extra[sha],
                     "format_words": WORDS, "terms_expected": [], "terms_hit": [], "inference_ms": 300.0,
                     "audio_ms": 2000.0}
                    for c, e in zip(clips, errors[sha])]}
            return {"eval_id": "eval-watch-fixture", "models": models,
                    "sets": [{"set": "rolling", "split_id": ref.split_id, "manifest_sha256": ref.manifest_sha256,
                              "n": ref.n}],
                    "unavailable_sets": {}, "per_clip": per_clip, "scores": {}}
        return evaluate

    def restore(self, name: str, **_):
        """A fake `healing models restore`: put the pruned GGUF back, as a sha256-exact rebuild does."""
        self.restored.append(name)
        sha = registry.load().get(name).gguf
        source = Path(self.hex.binary).parent / f"c{name}.gguf"
        objects.put(source, sha)
        with registry.lock():
            registry.append({"event": "restored", "version": name, "method": "sha256", "rebuilt_sha256": sha,
                             "gguf_sha256": sha})
        return {}

    def run(self, evaluate=None, *, days: float = 3, dry_run: bool = False) -> tuple[int, list[str]]:
        lines: list[str] = []
        code = watch.run(binary=self.hex.binary, dry_run=dry_run, now=self.t0 + int(days * DAY), say=lines.append,
                         evaluate=evaluate or self.never, restore=self.restore, confirm_s=2.0, poll_s=0.01)
        return code, lines

    def never(self, paths, ref):
        raise AssertionError("the watch scored, but it should not have")


@pytest.fixture
def world(hexbin, tmp_path):
    for name, parent in (("v1", "v0"), ("v2", "v1")):
        candidate(tmp_path, parent=parent)
        passing_gate(name)
        code, lines = ship_promote(hexbin, name)
        assert code == 0, lines
    t0 = [e for e in events("promoted") if e["version"] == "v2"][-1]["at_ms"]
    store = Store()
    yield World(hexbin, store, t0)
    store.close()


def base(n: int, seed: int = 5) -> np.ndarray:
    """v1's per-clip errors: at least one per clip, about 15% of 20 words."""
    return np.random.default_rng(seed).binomial(WORDS, 0.10, n) + 1


def shifted(errors: np.ndarray, up: int, down: int = 0) -> list[int]:
    """`errors` with one more error on the first `up` clips and one fewer on the next `down`."""
    out = errors.copy()
    out[:up] += 1
    out[up:up + down] -= 1
    return out.tolist()


def last_watch() -> dict:
    return events("watch")[-1]


# ---- AC-14 ----------------------------------------------------------------------------------------

def test_significant_loss_rolls_back(world):
    world.clips(100)  # 2,000 words
    errors = base(100)
    code, lines = world.run(world.scored(errors.tolist(), shifted(errors, 40)))  # +40 errors: 2 pt worse
    assert code == 0, lines
    assert any(line.startswith("rolled back v2 → v1: Δ +2.0 [+") for line in lines), lines
    state = registry.load()
    assert state.current == "v1" and state.versions["v2"].state == "rolled_back"
    assert "watch" in state.versions["v2"].reason and "eval-watch-fixture" in state.versions["v2"].reason
    assert world.hex.selected == hex_id("v1")
    decision = last_watch()
    assert decision["verdict"] == "rolled_back" and decision["version"] == "v2" and decision["previous"] == "v1"
    assert decision["wer"]["delta"] == pytest.approx(0.02) and decision["wer"]["ci_low"] > 0  # the CI is logged
    assert decision["words"] == 2000 and decision["clips"] == 100 and decision["eval_id"] == "eval-watch-fixture"
    assert decision["promoted_at_ms"] == world.t0


@pytest.mark.parametrize("clips,up,down,days,verdict", [
    (30, 5, 3, 14, "insufficient"),  # the AC's 0.3 pt worse on 600 words: thin after 14 days
    (100, 16, 10, 3, "kept"),  # 0.3 pt worse on 2,000 words: a superiority test would roll this back
])
def test_noise_sized_loss_kept(world, clips, up, down, days, verdict):
    world.clips(clips)
    errors = base(clips)
    code, lines = world.run(world.scored(errors.tolist(), shifted(errors, up, down)), days=days)
    assert code == 0, lines
    decision = last_watch()
    assert decision["verdict"] == verdict
    assert decision["wer"]["delta"] == pytest.approx(0.003, abs=0.0005) and decision["wer"]["ci_low"] < 0
    assert registry.load().current == "v2" and world.hex.selected == hex_id("v2")
    prefix = "kept v2: Δ +0.3 [−" if verdict == "kept" else "insufficient evidence, kept v2"
    assert any(line.startswith(prefix) for line in lines), lines


def test_win_kept(world):
    world.clips(100)
    errors = base(100)
    code, lines = world.run(world.scored(errors.tolist(), shifted(errors, 0, 40)))
    assert code == 0, lines
    assert any(line.startswith("kept v2: Δ −2.0 [−") for line in lines), lines
    assert last_watch()["verdict"] == "kept" and registry.load().current == "v2"


def test_waits_for_evidence(world):
    world.clips(30)  # 600 words after the promotion
    world.clips(50, after=False)  # verified before it: from v1's time, never counted
    world.clips(20, strategy="uncertain")  # not a random sample
    world.clips(20, action="reject")  # no reference
    before, used = registry.read_events(), len(world.hex.calls("use"))
    code, lines = world.run(days=3.5)
    assert code == 0 and lines == ["waiting: 600/1500 words, day 3/14"]
    assert registry.read_events() == before and len(world.hex.calls("use")) == used


def test_thin_evidence_after_14_days_kept(world):
    world.clips(30)
    errors = base(30)
    code, lines = world.run(world.scored(errors.tolist(), (errors + 2).tolist()), days=14)  # 10 pt worse, but 600 words
    assert code == 0, lines
    assert any(line.startswith("insufficient evidence, kept v2: 600/1500 words after 14 days") for line in lines)
    decision = last_watch()
    assert decision["verdict"] == "insufficient" and decision["wer"]["ci_low"] > 0  # clearly worse, still kept
    assert registry.load().current == "v2" and world.hex.calls("use")[-1] == ["model", "use", hex_id("v2"), "--json"]


def test_thin_evidence_with_no_clips_kept_without_scoring(world):
    code, lines = world.run(days=15)
    assert code == 0 and lines == ["insufficient evidence, kept v2: 0/1500 words after 15 days"]
    decision = last_watch()
    assert decision["verdict"] == "insufficient" and decision["eval_id"] is None and decision["wer"] is None


def test_manual_switch_disables(world):
    world.clips(100)
    world.hex.set(selected=CATALOG)  # you picked the catalog model by hand after the promotion
    before = registry.read_events()
    code, lines = world.run()
    assert code == watch.EXIT_MANUAL
    assert any(line.startswith("warning:") and CATALOG in line and "v2" in line for line in lines), lines
    assert registry.read_events() == before and world.hex.selected == CATALOG


def test_no_re_promote_after_rollback(world, tmp_path):
    world.clips(100)
    errors = base(100)
    assert world.run(world.scored(errors.tolist(), shifted(errors, 40)))[0] == 0
    assert registry.load().versions["v2"].state == "rolled_back"
    # watch never decides again: v1 became current by a rollback, not a promotion
    code, lines = world.run(days=20)
    assert code == 0 and any("nothing to watch" in line for line in lines), lines
    assert len(events("watch")) == 1
    # and nothing promotes v2 again: not promote, the gate, a rollback to it, or train auto
    code, lines = ship_promote(world.hex, "v2")
    assert code == ship.EXIT_NO_GATE and any("never promoted again" in line for line in lines), lines
    with pytest.raises(registry.RegistryError):
        registry.promote("v2", "by hand")
    with pytest.raises(gate.GateError):
        gate.run("v2", support=gate.StreamSupport(True), say=lambda _: None)
    lines = []
    assert ship.rollback(binary=world.hex.binary, to="v2", say=lines.append, confirm_s=1, poll_s=0.01) == 1
    assert auto.progress(world.store).pending is None  # decided: it doesn't block the next round either
    assert registry.load().current == "v1" and world.hex.selected == hex_id("v1")


def test_restores_pruned_previous(world):
    world.clips(100)
    v1 = registry.load().get("v1").gguf
    with registry.lock():
        registry.append({"event": "pruned_artifact", "version": "v1", "kind": "gguf", "sha256": v1})
    objects.remove(v1)
    errors = base(100)
    evaluate = world.scored(errors.tolist(), errors.tolist())

    def checked(paths, ref):
        assert world.restored == ["v1"] and objects.has(v1)  # restored before scoring
        return evaluate(paths, ref)

    code, lines = world.run(checked)
    assert code == 0, lines
    assert world.restored == ["v1"] and len(world.calls) == 1 and last_watch()["verdict"] == "kept"


# ---- the rest of the contract ---------------------------------------------------------------------

def test_formatting_regression_rolls_back(world):
    world.clips(100)
    errors = base(100).tolist()
    code, lines = world.run(world.scored(errors, errors, format_extra=1))  # same WER, 5 pt worse formatting
    assert code == 0, lines
    assert any(line.startswith("rolled back v2 → v1: Δ +0.0") for line in lines), lines
    assert any(line.startswith("formatting regressed: Δ +5.0") for line in lines), lines
    assert last_watch()["regressed"] == ["formatting"] and registry.load().current == "v1"


def test_streaming_regression_rolls_back(world):
    world.clips(100)
    errors = base(100).tolist()
    world.hex.set(no_stream=[hex_id("v2")])  # Hex stopped streaming v2 while it streams v1
    code, lines = world.run(world.scored(errors, errors))
    assert code == 0, lines
    assert last_watch()["regressed"] == ["streaming"] and registry.load().current == "v1"
    assert any(line.startswith("streaming regressed: Hex doesn't stream v2") for line in lines), lines


def test_lab_off_is_not_a_streaming_regression(world):
    world.clips(100)
    errors = base(100).tolist()
    world.hex.set(lab=False)  # you turned the lab off: neither import streams, and it's your choice
    assert world.run(world.scored(errors, errors))[0] == 0
    assert last_watch()["verdict"] == "kept"


def test_dry_run_changes_nothing(world, capsys):
    world.clips(100)
    before = registry.read_events()
    calls = len(world.hex.calls())
    code = main(["watch", "--dry-run", "--hex", str(world.hex.binary)])
    out = capsys.readouterr().out
    assert code == 0 and "would score v2 against v1 on 100 clips (2000 words" in out, out
    assert registry.read_events() == before
    assert all(c[:2] == ["status", "--json"] for c in world.hex.read()["calls"][calls:])
    assert not (watch.healing_home() / "evals").exists()


def test_decides_once_per_promotion(world):
    world.clips(100)
    errors = base(100)
    assert world.run(world.scored(errors.tolist(), errors.tolist()))[0] == 0
    world.clips(100)  # more reviews later change nothing: the promotion is decided
    code, lines = world.run(days=10)
    assert code == 0 and lines and lines[0].startswith("decided: kept v2"), lines
    assert len(world.calls) == 1 and len(events("watch")) == 1


def test_failed_rollback_records_nothing(world):
    world.clips(100)
    errors = base(100)
    world.hex.set(ignore_use=True)  # Hex answers but keeps v2
    code, lines = world.run(world.scored(errors.tolist(), shifted(errors, 40)))
    assert code == ship.EXIT_MISMATCH
    assert events("watch") == [] and registry.load().current == "v2"
    world.hex.set(ignore_use=False)  # the next run decides again
    assert world.run(world.scored(errors.tolist(), shifted(errors, 40)))[0] == 0
    assert registry.load().current == "v1"


def test_hex_down_waits(world):
    world.hex.set(running=False)
    before = registry.read_events()
    code, lines = world.run()
    assert code == watch.EXIT_HEX_DOWN and registry.read_events() == before


def test_lower_bound_is_one_sided_95():
    """The rollback bound is the 5th percentile of the bootstrap Δ (promoted − previous): positive only
    when the promoted model is worse."""
    errors = base(100)
    previous = {f"c{i}": stats.ClipScore(f"c{i}", int(e), WORDS) for i, e in enumerate(errors)}
    worse = {f"c{i}": stats.ClipScore(f"c{i}", int(e), WORDS) for i, e in enumerate(shifted(errors, 40))}
    result = watch.paired(previous, worse, resamples=2000, seed=13)
    deltas = stats.bootstrap_deltas(previous, worse, resamples=2000, seed=13)
    assert result.ci_low == pytest.approx(float(np.quantile(deltas, 0.05)))
    assert result.ci_low > 0 and result.delta == pytest.approx(0.02)
    better = watch.paired(worse, previous, resamples=2000, seed=13)
    assert better.ci_low < better.delta < 0


def test_status_nudges_review_while_watch_waits(world):
    world.clips(30)
    result = status.compute(world.store, world.hex.binary, now=world.t0 + 3 * DAY)
    assert "watch_needs_reviews" in result.warnings
    assert "600/1500 words" in result.text()


def test_schedule_runs_watch_after_collect(monkeypatch):
    order = []
    monkeypatch.setattr("healing.collect.main_collect", lambda *_: order.append("collect") or 0)
    monkeypatch.setattr(schedule, "_prune_after", lambda: order.append("prune") or 0)
    monkeypatch.setattr(schedule, "_clean_after", lambda: order.append("clean") or 0)
    monkeypatch.setattr(schedule, "_watch_after", lambda: order.append("watch") or 0)
    assert schedule.run_once() == 0
    assert order.index("collect") < order.index("watch")
