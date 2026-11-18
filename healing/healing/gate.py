"""`healing gate <v> [--against current]`: the promotion gate (T7.1, D-12).

The candidate and the version it would replace are scored together by one `healing eval` (T3.3) with
the vocabulary Hex runs (T3.4), on the candidate's parent's rolling set, golden and general, plus v0
when it is neither. Every interval is a paired bootstrap over clips from `healing.stats` (the one
bootstrap), seeded from `healing/gate.toml`, so the same eval always gives the same decision. The
candidate passes only when every check in `CHECKS` holds; `golden_wer` is `skipped` until golden
freezes (recorded, never a pass), and the edited-span error is reported but never gates.

Each decision is a registry `gate` event carrying `gate.toml`'s SHA-256. A failure also rejects the
candidate (`rejected`, naming the failing checks), except a refusal for thin evidence: too few
reviewed rolling words or term clips cannot reject a model, so it stays a candidate until more random
reviews arrive (D-22, D-54). Exit 0 on a pass, 7 on a fail or a refusal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from healing import stats
from healing.paths import healing_home, hex_support_dir, repo_root

CHECKS = ("evidence", "rolling_wer", "terms", "general_wer", "golden_wer", "formatting", "drift_guard", "streaming")
GATE_SETS = ["rolling", "golden", "general", "edited"]
FORMATTED_SETS = ("rolling", "golden")
EXIT_FAIL = 7
STREAM_LAB = "stream-imported-models"


class GateError(RuntimeError):
    code = 1

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class GateUsageError(GateError):
    code = 2


class GateRefused(GateError):
    """The gate cannot judge at all (G-10: golden differs from its pin). Nothing is recorded."""


@dataclass(frozen=True)
class Margins:
    """`healing/gate.toml`: the D-12 thresholds, resample count and seed, and the file's SHA-256."""

    path: Path
    sha256: str
    resamples: int
    seed: int
    confidence: float
    min_rolling_words: int
    min_term_clips: int
    rolling_wer_ci_high: float
    terms_alpha: float
    general_wer_ci_high: float
    golden_wer_ci_high: float
    formatting_ci_high: float
    drift_guard: float
    streaming_p95_ms: float
    long_clip_ms: float

    @classmethod
    def load(cls, path: Path | None = None) -> Margins:
        path = path or repo_root() / "healing" / "gate.toml"
        data = path.read_bytes()
        table = tomllib.loads(data.decode())
        values = {**table["bootstrap"], **table["evidence"], **table["checks"]}
        return cls(path=path, sha256=hashlib.sha256(data).hexdigest(), **values)


@dataclass(frozen=True)
class StreamSupport:
    """Whether the candidate streams in Hex (T7.4), and why or with what caveat."""

    streams: bool
    note: str | None = None


@dataclass
class Check:
    name: str
    status: str  # pass | fail | skipped
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p: float | None = None
    threshold: float | None = None
    unit: str = "rate"  # rate (a fraction of words) | words | ms | clips
    note: str | None = None

    def record(self) -> dict:
        passed = {"pass": True, "fail": False}.get(self.status)
        return {"name": self.name, "status": self.status, "pass": passed, "estimate": self.estimate,
                "ci_low": self.ci_low, "ci_high": self.ci_high, "p": self.p, "threshold": self.threshold,
                "unit": self.unit, "note": self.note}


@dataclass
class Verdict:
    checks: list[Check]
    diagnostics: dict = field(default_factory=dict)

    @property
    def failing(self) -> list[str]:
        return [c.name for c in self.checks if c.status == "fail"]

    @property
    def passed(self) -> bool:
        return not self.failing

    @property
    def refused(self) -> bool:
        """Too little evidence to decide: never a pass, and never a rejection."""
        return "evidence" in self.failing

    def as_records(self) -> list[dict]:
        return [c.record() for c in self.checks]


# ---- judging one eval -----------------------------------------------------------------------------

def _rows(document: dict, sha: str, set_name: str) -> list[dict] | None:
    paths = [m["gguf_path"] for m in document["models"] if m["gguf_sha256"] == sha]
    if not paths:
        raise GateError(f"the eval {document.get('eval_id')} did not score GGUF {sha[:12]}")
    return document["per_clip"].get(paths[0], {}).get(set_name)


def _why_missing(document: dict, set_name: str) -> str:
    return f"no {set_name} set: " + document.get("unavailable_sets", {}).get(set_name, "not scored")


def _bootstrap(name: str, a: list[dict], b: list[dict], errors: str, words: str, limit: float, margins: Margins,
               prefix: str = "") -> Check:
    def scores(rows):
        return {prefix + r["id"]: stats.ClipScore(prefix + r["id"], int(r[errors]), int(r[words]))
                for r in rows if r.get(words) is not None}

    result = stats.paired_bootstrap(scores(a), scores(b), resamples=margins.resamples, seed=margins.seed,
                                    confidence=margins.confidence)
    return Check(name, "pass" if result.ci_high < limit else "fail", result.delta, result.ci_low, result.ci_high,
                 threshold=limit, note=f"{result.clips} clips, {result.reference_words} words")


def _wer_check(name: str, document: dict, set_name: str, candidate: str, against: str, limit: float,
               margins: Margins) -> Check:
    a, b = _rows(document, against, set_name), _rows(document, candidate, set_name)
    if not a or not b:
        return Check(name, "fail", threshold=limit, note=_why_missing(document, set_name))
    return _bootstrap(name, a, b, "word_errors", "reference_words", limit, margins)


def _evidence(rolling: list[dict] | None, margins: Margins, allow_thin: bool) -> Check:
    words = sum(r["reference_words"] or 0 for r in rolling or [])
    term_clips = len(stats.clips_right(rolling or []))
    note = (f"{words} reference words ({margins.min_rolling_words} needed), "
            f"{term_clips} term clips ({margins.min_term_clips} needed)")
    enough = words >= margins.min_rolling_words and term_clips >= margins.min_term_clips
    status = "pass" if enough else "skipped" if allow_thin else "fail"
    if status == "skipped":
        note += "; thin evidence allowed (fixtures only)"
    return Check("evidence", status, words, threshold=margins.min_rolling_words, unit="words", note=note)


def _terms(a: list[dict] | None, b: list[dict] | None, margins: Margins) -> Check:
    if not a or not b:
        return Check("terms", "fail", threshold=margins.terms_alpha, unit="clips", note="no rolling set")
    right_a, right_b = stats.clips_right(a), stats.clips_right(b)
    wins = sum(right_b[c] and not right_a[c] for c in right_a)
    losses = sum(right_a[c] and not right_b[c] for c in right_a)
    p = stats.mcnemar_exact(wins, losses)
    clips = len(right_a)
    estimate = (losses - wins) / clips if clips else None  # Δ clip term error rate, candidate − current
    ok = wins > losses and p < margins.terms_alpha
    return Check("terms", "pass" if ok else "fail", estimate, p=p, threshold=margins.terms_alpha, unit="clips",
                 note=f"{clips} term clips: {wins} only the candidate got right, {losses} only the current model")


def _formatting(document: dict, candidate: str, against: str, margins: Margins) -> Check:
    a, b = [], []
    for set_name in FORMATTED_SETS:
        rows_a, rows_b = _rows(document, against, set_name), _rows(document, candidate, set_name)
        if rows_a and rows_b:
            a += [{**r, "id": f"{set_name}:{r['id']}"} for r in rows_a]
            b += [{**r, "id": f"{set_name}:{r['id']}"} for r in rows_b]
    if not any(r.get("format_words") for r in a):
        return Check("formatting", "fail", threshold=margins.formatting_ci_high, note="no formatted set was scored")
    return _bootstrap("formatting", a, b, "format_errors", "format_words", margins.formatting_ci_high, margins)


def _wer(rows: list[dict]) -> float:
    return sum(r["word_errors"] for r in rows) / sum(r["reference_words"] for r in rows)


def _drift(document: dict, candidate: str, v0: str, margins: Margins) -> Check:
    b, base = _rows(document, candidate, "general"), _rows(document, v0, "general")
    if not b or not base:
        return Check("drift_guard", "fail", threshold=margins.drift_guard, note=_why_missing(document, "general"))
    estimate = _wer(b) - _wer(base)
    return Check("drift_guard", "pass" if estimate <= margins.drift_guard else "fail", estimate,
                 threshold=margins.drift_guard, note=f"general WER {_wer(b):.4f} against v0's {_wer(base):.4f}")


def _p95_long(document: dict, sha: str, margins: Margins) -> tuple[float | None, int, bool]:
    """p95 of the benchmark's per-clip `inference_ms` over long clips of every scored set (all clips when
    none is long): (p95, clips, whether they were long)."""
    rows = [r for s in document["sets"] for r in (_rows(document, sha, s["set"]) or [])]
    if not rows or any(r.get("inference_ms") is None or r.get("audio_ms") is None for r in rows):
        return None, 0, False
    long = [r for r in rows if r["audio_ms"] >= margins.long_clip_ms]
    chosen = long or rows
    return float(np.percentile([r["inference_ms"] for r in chosen], 95)), len(chosen), bool(long)


def _streaming(document: dict, candidate: str, against: str, support: StreamSupport, margins: Margins) -> Check:
    p95_b, n, long = _p95_long(document, candidate, margins)
    p95_a, _, _ = _p95_long(document, against, margins)
    notes = [] if support.note is None else [support.note]
    if p95_a is None or p95_b is None:
        return Check("streaming", "fail", threshold=margins.streaming_p95_ms, unit="ms",
                     note="; ".join(notes + ["the eval holds no per-clip latency, so time to text is unmeasured"]))
    estimate = p95_b - p95_a
    notes.append(f"p95 {p95_b:.0f} ms against {p95_a:.0f} ms over {n} {'long' if long else 'clips (none long)'} clips")
    ok = support.streams and estimate <= margins.streaming_p95_ms
    return Check("streaming", "pass" if ok else "fail", estimate, threshold=margins.streaming_p95_ms, unit="ms",
                 note="; ".join(notes))


def _golden(document: dict, candidate: str, against: str, margins: Margins, golden_pin: dict | None) -> Check:
    scored = next((s for s in document["sets"] if s["set"] == "golden"), None)
    if scored is None:
        return Check("golden_wer", "skipped", threshold=margins.golden_wer_ci_high,
                     note=document.get("unavailable_sets", {}).get("golden", "golden is not frozen yet"))
    if golden_pin is None:
        raise GateRefused("golden is frozen but not pinned in the registry: run `healing models pin-golden` (G-10)")
    if scored["manifest_sha256"] != golden_pin["sha256"]:
        raise GateRefused(f"G-10: the scored golden set {scored['manifest_sha256'][:12]} differs from its pin "
                          f"{golden_pin['sha256'][:12]}; run `healing split verify`")
    return _wer_check("golden_wer", document, "golden", candidate, against, margins.golden_wer_ci_high, margins)


def _edited_span(document: dict, sha: str) -> float | None:
    path = next(m["gguf_path"] for m in document["models"] if m["gguf_sha256"] == sha)
    span = (document.get("scores", {}).get(path, {}).get("edited") or {}).get("edited_span")
    return span["errors"] / span["positions"] if span and span.get("positions") else None


def judge(document: dict, *, candidate: str, against: str, v0: str, support: StreamSupport, margins: Margins,
          golden_pin: dict | None, allow_thin: bool = False) -> Verdict:
    """Every D-12 check on one eval document; the models are named by their GGUFs' SHA-256."""
    rolling_a, rolling_b = _rows(document, against, "rolling"), _rows(document, candidate, "rolling")
    checks = [
        _evidence(rolling_a, margins, allow_thin),
        _wer_check("rolling_wer", document, "rolling", candidate, against, margins.rolling_wer_ci_high, margins),
        _terms(rolling_a, rolling_b, margins),
        _wer_check("general_wer", document, "general", candidate, against, margins.general_wer_ci_high, margins),
        _golden(document, candidate, against, margins, golden_pin),
        _formatting(document, candidate, against, margins),
        _drift(document, candidate, v0, margins),
        _streaming(document, candidate, against, support, margins),
    ]
    diagnostics = {"edited_span": {"candidate": _edited_span(document, candidate), "current": _edited_span(document, against)}}
    return Verdict(checks, diagnostics)


# ---- the table ------------------------------------------------------------------------------------

def _value(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "rate":
        return f"{value * 100:+.2f} pt"
    if unit == "clips":
        return f"{value * 100:+.1f} pt"
    if unit == "ms":
        return f"{value:+.0f} ms"
    return f"{value:.0f}"


def table(verdict: Verdict) -> list[str]:
    lines = [f"{'check':<13}{'status':<9}{'estimate':>11}  {'95% CI':<22}{'p':>8}  {'threshold':>10}  note"]
    for c in verdict.checks:
        ci = "-" if c.ci_low is None else f"[{_value(c.ci_low, c.unit)}, {_value(c.ci_high, c.unit)}]"
        p = "-" if c.p is None else f"{c.p:.4f}"
        threshold = "-" if c.threshold is None else (f"{c.threshold:g}" if c.unit == "clips" else _value(c.threshold, c.unit))
        lines.append(f"{c.name:<13}{c.status:<9}{_value(c.estimate, c.unit):>11}  {ci:<22}{p:>8}  {threshold:>10}  {c.note or ''}")
    span = verdict.diagnostics.get("edited_span", {})
    if any(v is not None for v in span.values()):
        lines.append("edited-span error (diagnostic, does not gate): candidate "
                     + ", current ".join("n/a" if v is None else f"{v * 100:.2f}%" for v in (span["candidate"], span["current"])))
    return lines


# ---- the live path ---------------------------------------------------------------------------------

def _in_tests() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


@contextlib.contextmanager
def _model_files(versions: list):
    """Each version's GGUF from the registry's objects (pinned by hash). v0 is offered under its catalog
    file name, as `train prepare` does, so the benchmark recognizes it and pads it as Hex does."""
    from healing.registry import objects
    from healing.registry.objects import owner_dir
    from healing.train.manifest import catalog_filename

    work = healing_home() / "tmp"
    owner_dir(work)
    with tempfile.TemporaryDirectory(dir=work, prefix="gate-") as scratch:
        paths = []
        for version in versions:
            if not objects.has(version.gguf):
                raise GateError(f"{version.name}'s GGUF {str(version.gguf)[:12]} is not kept: "
                                f"run `healing models restore {version.name}`")
            path = objects.path(version.gguf)
            catalog_id = (version.record.get("base") or {}).get("catalog_id")
            if version.name == "v0" and catalog_filename(version.record, catalog_id):
                link = Path(scratch) / catalog_filename(version.record, catalog_id)
                link.symlink_to(path)
                path = link
            paths.append(path)
        yield paths


def _evaluate(models: list[Path], set_names: list[str], parent: str) -> dict:
    """`healing eval` of every model with Hex's current vocabulary."""
    from healing.eval import run as eval_run
    from healing.store import Store, dataset_root

    store = Store(sweep=False) if (dataset_root() / "dataset.sqlite").is_file() else None
    try:
        return eval_run.evaluate(models, set_names, store=store, parent=parent, vocabulary="current",
                                 progress=lambda message: print(message, file=sys.stderr))
    finally:
        if store is not None:
            store.close()


def _lab_on() -> bool:
    """Labs › Stream imported models, read-only from Hex's `settings.json`."""
    try:
        settings = json.loads((hex_support_dir() / "settings.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(settings.get("labs", {}).get("stream_imported_models", False))


def stream_support(binary: Path, candidate, against) -> StreamSupport:
    """Whether Hex would stream the candidate (T7.4): an import streams only with Labs › Stream imported
    models on, made from a streaming catalog model, once its smoke stream passed as it loaded. Asked of
    the running Hex (`hex model list --json`). An import Hex already lists answers with its own
    `streams`; one not imported yet streams like `against` when Hex streams `against` and the two
    GGUF headers share architecture and `stt.variant`; its smoke stream runs only when `promote` loads it."""
    from healing import gguf
    from healing.registry import commands, objects

    try:
        listed = {m.get("id"): m for m in commands.hex_models(binary)}
    except commands.CommandError as error:
        raise GateError(f"cannot ask Hex whether the candidate streams: {error}", error.code) from error
    lab = _lab_on()
    turn_on = f"turn on Labs › Stream imported models (`hex labs on {STREAM_LAB}`, D-26)"
    candidate_id = objects.hex_model_id(candidate.gguf)
    if candidate_id in listed:
        if listed[candidate_id].get("streams") is True:
            return StreamSupport(True, f"Hex streams {candidate_id}")
        return StreamSupport(False, turn_on if not lab else f"Hex lists {candidate_id} with streams: false "
                             "(made from no streaming model, or its smoke stream failed)")
    if not lab:
        return StreamSupport(False, turn_on)
    against_id = (against.record.get("base") or {}).get("catalog_id") if against.name == "v0" else objects.hex_model_id(against.gguf)
    if (listed.get(against_id) or {}).get("streams") is not True:
        return StreamSupport(False, f"Hex doesn't stream {against.name} ({against_id}), so the candidate has nothing to stream like")
    try:
        heads = [gguf.read(objects.path(v.gguf)) for v in (candidate, against)]
    except (OSError, gguf.GGUFError) as error:
        raise GateError(f"cannot read a GGUF header: {error}") from error
    keys = ("general.architecture", "stt.variant")
    if any(heads[0].string(k) != heads[1].string(k) or not heads[0].string(k) for k in keys):
        return StreamSupport(False, f"its architecture and variant differ from {against_id}'s, so Hex won't stream it")
    return StreamSupport(True, f"streams like {against_id} (same architecture and variant); "
                         "its smoke stream runs when promote loads it")


def _check_eval(document: dict, parent: str) -> None:
    """An eval the gate may judge: something scored, the candidate's parent's holdouts, your vocabulary."""
    if not document.get("sets"):
        raise GateError("the eval scored none of the sets: " + "; ".join(document.get("unavailable_sets", {}).values()))
    if document.get("parent", parent) != parent:
        raise GateError(f"the eval used {document['parent']}'s holdouts; the candidate's parent is {parent}")
    mode = (document.get("vocabulary") or {}).get("mode", "current")
    if mode != "current":
        raise GateError(f"the eval was scored with --vocabulary {mode}; the gate scores with the vocabulary Hex runs")


def run(version: str, against: str = "current", *, eval_path: Path | None = None, allow_thin: bool = False,
        evaluate: Callable | None = None, support: StreamSupport | None = None, binary: Path | None = None,
        say: Callable[[str], None] = print) -> int:
    from healing.eval import run as eval_run
    from healing.registry import store as registry

    if allow_thin and not _in_tests():
        raise GateUsageError("--allow-thin-evidence is for test fixtures only; it is refused outside tests")
    margins = Margins.load()
    state = registry.load()
    candidate, target = state.get(version), state.get(against)
    if candidate.state != "candidate":
        raise GateError(f"{candidate.name} is {candidate.state}; only a candidate is gated")
    if candidate.name == target.name:
        raise GateUsageError(f"{candidate.name} cannot be gated against itself")
    v0 = state.get("v0")
    if support is None:
        from healing.eval.bench import history_hex

        support = stream_support(Path(binary or history_hex()), candidate, target)
    parent = candidate.record.get("parent") or "current"
    if eval_path is not None:
        document = json.loads(Path(eval_path).read_text())
        source = str(eval_path)
    else:
        versions = [target, candidate] + ([v0] if v0.name not in (target.name, candidate.name) else [])
        with _model_files(versions) as paths:
            document = (evaluate or _evaluate)(paths, GATE_SETS, parent)
        source = str(eval_run.write(document)) if document.get("sets") else None
    _check_eval(document, parent)
    verdict = judge(document, candidate=candidate.gguf, against=target.gguf, v0=v0.gguf, support=support,
                    margins=margins, golden_pin=state.golden, allow_thin=allow_thin)
    decision = {"against": target.name, "eval_id": document.get("eval_id"), "eval_path": source,
                "checks": verdict.as_records(), "pass": verdict.passed, "failing": verdict.failing,
                "refused": verdict.refused, "margins_sha": margins.sha256, "thin_evidence_allowed": allow_thin,
                "diagnostics": verdict.diagnostics}
    reject = None if verdict.passed or verdict.refused else "gate: " + ", ".join(verdict.failing)
    registry.record_gate(candidate.name, decision, reject)
    say(f"{candidate.name} against {target.name} ({document.get('eval_id')}, gate.toml {margins.sha256[:12]})")
    for line in table(verdict):
        say(line)
    if verdict.passed:
        say(f"{candidate.name}: passes the gate")
        return 0
    if verdict.refused:
        say(f"{candidate.name}: refused, too little evidence to decide; it stays a candidate. Review more recent "
            "clips (`healing review --strategy random --since-cutoff`), rebuild the split and gate it again")
    else:
        say(f"{candidate.name}: rejected ({reject})")
    return EXIT_FAIL


def main(args) -> int:
    from healing.eval.bench import BenchError
    from healing.eval.run import EvalError
    from healing.registry.store import RegistryError
    from healing.store import StoreError

    try:
        return run(args.version, args.against, eval_path=args.eval, allow_thin=args.allow_thin_evidence,
                   binary=args.hex)
    except GateError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except (RegistryError, EvalError, BenchError, StoreError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
