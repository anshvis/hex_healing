"""`healing vocab ab`: do the mined terms help, measured on the benchmark?

The current GGUF is scored on the rolling and general sets twice, through `healing eval`: biased with
your vocabulary alone, then with your vocabulary plus the mined terms in the order `apply` would write
them. The terms are mined with `train_only`, in memory and never saved, so a term learned from the
rolling clips cannot be measured on them (T3.2) and the table `apply` ranks from is not replaced.
Hex's vocabulary is read through `hex vocabulary list` and never written (G-6); nothing here needs a network (G-3).

Reported, with `with` minus `without` as the delta:
- the term error rate on the rolling set, with an exact McNemar test on the per-clip term hits;
- the WER, with a paired bootstrap 95% CI (`healing.stats`), on rolling and on general;
- the false-replacement rate on general: the share of clips whose transcript holds a mined term the
  reference doesn't (the benchmark's own `false_replacement_rate`, over the manifest's vocabulary, is
  shown beside it).

Only the global list is biased: a mode's own vocabulary takes slots first in Hex, and this run has no mode.
The M4 exit gate for shipping the mined terms at `normal` boost: general WER's CI upper bound at most
+0.3 pt, false replacements at most 0.5%, and the term McNemar test favouring the mined list. A
run with fewer than 20 clips holding a term cannot decide it: that is n/a, never a win (X-4).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path

from healing import manifest as manifest_module
from healing import stats
from healing.eval import run as eval_run
from healing.paths import healing_home
from healing.vocab import apply as vocab_apply
from healing.vocab import mine as mine_module

BOOSTS = ("light", "normal", "strong")
SETS = ["rolling", "general"]
MIN_TERM_CLIPS = 20
GATE_WER_CI_HIGH = 0.003  # +0.3 pt, as a fraction
GATE_FALSE_REPLACEMENTS = 0.005
ALPHA = 0.05


class AbError(RuntimeError):
    pass


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value: the discordant pairs split evenly under the null."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** n)


@dataclass(frozen=True)
class TermResult:
    clips: int  # clips holding at least one expected term
    terms: int  # expected terms in them
    error_without: float | None
    error_with: float | None
    wins: int  # clips only the mined list got right (all their terms)
    losses: int  # clips only the list without them got right
    p_value: float | None
    favours_mined: bool
    na: str | None  # why there is no verdict

    @property
    def delta(self) -> float | None:
        return None if self.na else self.error_with - self.error_without


def term_result(without: list[dict], with_: list[dict]) -> TermResult:
    """The McNemar test counts clips, not terms: a clip is right when every term it expects was heard, so
    one clip that gets ten terms cannot stand for ten independent pieces of evidence. The term error rates
    stay per term, as description."""
    by_id = {row["id"]: row for row in with_}
    clips = terms = misses_a = misses_b = wins = losses = 0
    for row in without:
        expected = row["terms_expected"]
        if not expected:
            continue
        other = by_id[row["id"]]
        clips += 1
        hits_a = [term in row["terms_hit"] for term in expected]
        hits_b = [term in other["terms_hit"] for term in expected]
        terms += len(expected)
        misses_a, misses_b = misses_a + hits_a.count(False), misses_b + hits_b.count(False)
        right_a, right_b = all(hits_a), all(hits_b)
        wins, losses = wins + (right_b and not right_a), losses + (right_a and not right_b)
    if clips < MIN_TERM_CLIPS:
        return TermResult(clips, terms, None, None, wins, losses, None, False,
                          f"n/a: {clips} clip(s) hold a term, at least {MIN_TERM_CLIPS} are needed")
    p = mcnemar_exact(wins, losses)
    return TermResult(clips, terms, misses_a / terms, misses_b / terms, wins, losses, p, wins > losses and p < ALPHA, None)


def wer_result(without: list[dict], with_: list[dict]) -> dict:
    def scores(rows):
        return {r["id"]: stats.ClipScore(r["id"], int(r["word_errors"]), int(r["reference_words"])) for r in rows}

    return stats.paired_bootstrap(scores(without), scores(with_)).as_dict()


def false_replacements(per_clip: list[dict], mined_terms: list[str]) -> dict:
    """Clips whose transcript holds a mined term more often than the reference does."""
    bad = []
    for row in per_clip:
        extra = [t for t in mined_terms
                 if manifest_module.term_count(row["transcript"], t) > manifest_module.term_count(row["expected"], t)]
        if extra:
            bad.append({"id": row["id"], "terms": extra})
    return {"clips": len(per_clip), "clips_with_false_replacement": len(bad),
            "rate": len(bad) / len(per_clip) if per_clip else None, "examples": bad[:10]}


def verdict(boost: str, terms: TermResult | None, general_wer: dict | None, general_false: dict | None) -> dict:
    """Specific to the boost that was measured: only a `normal` run can approve normal, a `strong` run
    strong (`apply` reads this against the boost Hex runs)."""
    gates = {
        "general_wer_ci_high_at_most_0.3pt": None if general_wer is None else general_wer["ci_high"] <= GATE_WER_CI_HIGH,
        "false_replacements_at_most_0.5pct": None if not general_false or general_false["rate"] is None
        else general_false["rate"] <= GATE_FALSE_REPLACEMENTS,
        "terms_favour_mined": None if terms is None or terms.na else terms.favours_mined,
    }
    if any(value is None for value in gates.values()):
        undecided = [name for name, value in gates.items() if value is None]
        return {"gates": gates, "boost": boost, "passes": None, "recommended_boost": "light",
                "text": f"no verdict at {boost} boost (n/a: " + ", ".join(undecided) + "); keep Hex at light"}
    if all(gates.values()):
        return {"gates": gates, "boost": boost, "passes": True, "recommended_boost": boost,
                "text": f"the mined terms pass the gate at {boost} boost"}
    failed = [name for name, value in gates.items() if not value]
    return {"gates": gates, "boost": boost, "passes": False, "recommended_boost": "light",
            "text": f"the gate fails at {boost} boost (" + ", ".join(failed) + "); keep Hex at light"}


def run_ab(
    model: Path, boost: str, parent: str, *, yours: list[dict], mined: list[dict], store, evaluate=eval_run.evaluate,
    progress=lambda message: None, now: float | None = None,
) -> dict:
    """The two evals and everything computed from them. `yours` and `mined` are Hex entries."""
    if boost not in BOOSTS:
        raise AbError(f"--boost is one of {', '.join(BOOSTS)}")
    if not mined:
        raise AbError("no mined terms to measure: run `healing vocab mine` (the A/B mines the training clips itself, "
                      "but a correction needs to have been made twice)")
    documents = {}
    for arm, entries in (("without", yours), ("with", yours + mined)):
        progress(f"{arm} the mined terms ({len(entries)} vocabulary terms)")
        documents[arm] = evaluate(
            [model], SETS, store=store, parent=parent, progress=progress,
            vocabulary_entries=entries, vocabulary_boost=boost,
        )
    key = str(model)
    a, b = documents["without"], documents["with"]
    sets: dict[str, dict] = {}
    mined_terms = [m["term"] for m in mined]
    for name in SETS:
        if name in a["unavailable_sets"]:
            sets[name] = {"na": a["unavailable_sets"][name]}
            continue
        rows_a, rows_b = a["per_clip"][key][name], b["per_clip"][key][name]
        scores_a, scores_b = a["scores"][key][name], b["scores"][key][name]
        result = {"clips": len(rows_a), "wer": wer_result(rows_a, rows_b)}
        if name == "rolling":
            result["terms"] = term_result(rows_a, rows_b)
        if name == "general":
            result["false_replacements"] = false_replacements(rows_b, mined_terms)
            result["benchmark_false_replacement_rate"] = {"without": scores_a.get("false_replacement_rate"),
                                                          "with": scores_b.get("false_replacement_rate")}
        sets[name] = result
    rolling, general = sets.get("rolling", {}), sets.get("general", {})
    outcome = verdict(boost, rolling.get("terms"), general.get("wer"), general.get("false_replacements"))
    started = time.time() if now is None else now
    return {
        "ab_id": "vocab-ab-" + datetime.fromtimestamp(started, timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "model": {"gguf_path": str(model), "gguf_sha256": a["models"][0]["gguf_sha256"]},
        "boost": boost,
        "parent": parent,
        "scope": "global vocabulary only; no mode vocabularies (a mode's own terms take bias slots first in Hex)",
        "your_terms": len(yours),
        "mined_terms": [{"term": m["term"], "sounds_like": m["sounds_like"]} for m in mined],
        "mined_source": "train clips only, in memory (vocab_candidates untouched)",
        "sets": {name: {k: (asdict(v) if isinstance(v, TermResult) else v) for k, v in result.items()} for name, result in sets.items()},
        "verdict": outcome,
        "evals": {"without": a["eval_id"], "with": b["eval_id"]},
        "unavailable_sets": a["unavailable_sets"],
    }


def _pt(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f} pt"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def text(report: dict) -> str:
    lines = [
        f"vocabulary A/B: {Path(report['model']['gguf_path']).name}, boost {report['boost']}",
        f"{report['your_terms']} of yours, {len(report['mined_terms'])} mined (train clips only); "
        "scope: the global list, no mode (a mode's terms take slots first)",
    ]
    for name in SETS:
        result = report["sets"].get(name)
        if result is None or "na" in result:
            lines.append(f"{name}: n/a ({(result or {}).get('na', 'not run')})")
            continue
        wer = result["wer"]
        lines.append(
            f"{name} ({result['clips']} clips): WER {_pct(wer['wer_a'])} -> {_pct(wer['wer_b'])}, delta {_pt(wer['delta'])}, "
            f"95% CI [{_pt(wer['ci_low'])}, {_pt(wer['ci_high'])}]"
        )
        terms = result.get("terms")
        if terms:
            if terms["na"]:
                lines.append(f"  term error rate: {terms['na']}")
            else:
                lines.append(
                    f"  term error rate ({terms['terms']} terms in {terms['clips']} clips): {_pct(terms['error_without'])} -> "
                    f"{_pct(terms['error_with'])}, delta {_pt(terms['error_with'] - terms['error_without'])}; "
                    f"McNemar exact over clips: mined got {terms['wins']} more clips right, lost {terms['losses']}, p = {terms['p_value']:.4f}"
                    + (" (favours mined)" if terms["favours_mined"] else " (does not favour mined)")
                )
        fr = result.get("false_replacements")
        if fr:
            bench = result["benchmark_false_replacement_rate"]
            lines.append(
                f"  false replacements: {fr['clips_with_false_replacement']} of {fr['clips']} clips ({_pct(fr['rate'])}); "
                f"benchmark's own rate {_pct(bench['without'])} -> {_pct(bench['with'])}"
            )
    lines.append("verdict: " + report["verdict"]["text"])
    return "\n".join(lines)


def write_verdict(report: dict) -> Path:
    """The latest A/B's verdict, where `apply` reads it: which boost was measured and whether it passed."""
    path = vocab_apply.verdict_path()
    outcome = report["verdict"]
    vocab_apply.write_atomic(path, json.dumps({
        "ab_id": report["ab_id"], "measured_at_ms": int(time.time() * 1000), "boost": outcome.get("boost", report["boost"]),
        "passes": outcome.get("passes"), "recommended_boost": outcome.get("recommended_boost"),
        "gates": outcome.get("gates"), "model_sha256": report["model"]["gguf_sha256"],
    }, indent=1) + "\n")
    return path


def write(report: dict) -> Path:
    """Owner-only, and never over an earlier report of the same id."""
    folder = healing_home() / "evals"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    for attempt in range(100):
        path = folder / f"{report['ab_id']}{'' if attempt == 0 else f'-{attempt}'}.json"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w") as file:
            file.write(json.dumps(report, indent=1) + "\n")
        return path
    raise AbError(f"could not find a free file name for {report['ab_id']}")


def main(args) -> int:
    from healing import labels
    from healing.eval.bench import BenchError, history_hex
    from healing.eval.baseline import DEFAULT_GGUF
    from healing.store import Store, StoreError, dataset_root, now_ms, process_lock

    model = Path(args.model) if args.model else DEFAULT_GGUF
    try:
        binary = Path(args.hex or history_hex())
        current = vocab_apply.hex_list(binary)  # read-only (G-6); fails first when Hex isn't running
        applied = vocab_apply.load_applied()
        with process_lock():
            store = Store()
            try:
                labels.label_all(store)  # the evidence is derived: bring it up to date, as `vocab mine` does
                mined = mine_module.mine(store, train_only=True, now=now_ms())  # in memory: never saved
            finally:
                store.close()
        plan = vocab_apply.plan(current, applied, mined.candidates)
        store = Store(sweep=False) if (dataset_root() / "dataset.sqlite").is_file() else None
        try:
            report = run_ab(
                model, args.boost, args.parent, yours=plan.yours, mined=plan.mined, store=store,
                progress=lambda message: print(message, file=sys.stderr),
            )
        finally:
            if store is not None:
                store.close()
        path = write(report)
        write_verdict(report)
    except vocab_apply.HexNotRunning as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except eval_run.EvalUsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (AbError, eval_run.EvalError, vocab_apply.ApplyError, BenchError, StoreError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(text(report))
    print(f"\nwrote {path}")
    return 0
