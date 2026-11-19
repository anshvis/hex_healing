"""`healing watch [--dry-run]`: roll a promotion back only when it is significantly worse (T7.3, D-19).

After `healing promote`, the collect job runs `watch` after each collect. It waits until the clips you
dictated after the promotion that a random review verified hold 1,500 reference words, or until 14
days have passed. Then it scores the promoted and the previous version together on those clips (one
`healing eval` with the vocabulary Hex runs, each model prepared as Hex runs it) and decides once per
promotion, recorded as a registry `watch` event:

- **rolled back** when the promoted model is significantly worse: the one-sided 95% lower bound of
  Δ WER (promoted − previous) is above 0, or formatting or streaming regressed past what the gate
  accepts (D-58). `healing rollback` does it, so the ship lock, Hex's confirmation and `rolled_back`
  all apply; a rollback Hex doesn't confirm records nothing and the next run tries again;
- **kept** otherwise, with Δ and its interval;
- **insufficient evidence, kept** when 14 days pass with under 1,500 words: it never rolls back on thin
  data (`healing status` nudges you to review while it waits).

The references are your verified text, never either model's output (T3.1). Only clips recorded after
the promotion count, reviewed with the `random` strategy, and none that shares a split-cluster with any
recorded training snapshot (either model may have trained on it).

It does nothing, and warns, when Hex runs another model than the current version (you switched by
hand, X-4). It watches only a version made current by `promote`: after a rollback, nothing is watched
until the next promotion (D-59). A pruned previous GGUF is restored first (`healing models restore`).

Exit codes: 0 waiting, decided or nothing to watch; 2 Hex isn't running (nothing changes); 9 Hex didn't
confirm the rollback; 11 Hex runs a model you picked by hand; 1 anything else.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from healing import gate, labels, splits, stats
from healing.paths import healing_home
from healing.store import Store, StoreError, dataset_root, locked, now_ms
from healing.wer import normalized_words

TARGET_WORDS = 1500  # D-19: verified reference words after the promotion
MAX_DAYS = 14  # D-19: or this many days, whichever comes first
DAY_MS = 24 * 3600 * 1000
CONFIDENCE = 0.90  # two-sided 90% interval: its lower end is the one-sided 95% lower bound
EXIT_HEX_DOWN = 2
EXIT_MANUAL = 11

__all__ = ["Evidence", "Watched", "evidence", "healing_home", "paired", "pending", "run", "watched"]


class WatchError(RuntimeError):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def lock():
    """One watch at a time (the launchd job and one by hand)."""
    folder = healing_home()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    return locked(folder / "watch.lock", wait=False, busy="another healing watch is running")


# ---- what is watched -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Watched:
    version: object  # the promoted registry Version
    previous: object  # the version it replaced
    promoted_at_ms: int


def watched(state) -> tuple[Watched | None, str | None]:
    """(the promotion to watch, None) or (None, why there is none). Only a version made current by a
    promotion that replaced another version is watched; one made current by a rollback is not (D-59)."""
    if state.current is None:
        return None, "no version is current"
    by = state.current_by or {}
    if by.get("event") != "promoted":
        return None, f"{state.current} became current by a rollback, not a promotion"
    version = state.versions[state.current]
    promotion = version.promotions[-1]
    previous = promotion.get("previous")
    if not previous or previous == version.name:
        return None, f"{version.name} is the base model, promoted over nothing"
    return Watched(version, state.versions[previous], int(promotion["at_ms"])), None


def decision_for(target: Watched) -> dict | None:
    return next((w for w in target.version.watches if w.get("promoted_at_ms") == target.promoted_at_ms), None)


@dataclass
class Evidence:
    rows: list[dict] = field(default_factory=list)  # clip rows with `reference_text` = your verified text
    words: int = 0
    days: float = 0.0
    sessions: dict = field(default_factory=dict)  # what the set's sidecar records, from the same `Zones`
    clusters: dict = field(default_factory=dict)

    @property
    def enough(self) -> bool:
        return self.words >= TARGET_WORDS

    @property
    def due(self) -> bool:
        return self.enough or self.days >= MAX_DAYS

    def waiting(self) -> str:
        return f"waiting: {self.words}/{TARGET_WORDS} words, day {int(self.days)}/{MAX_DAYS}"


def evidence(store: Store | None, since_ms: int, now: int) -> Evidence:
    """Clips recorded after `since_ms` whose latest review is a random accept (verified text, T3.1), in no
    split-cluster of a recorded training snapshot. Read-only: labels are derived here as `labels.derive`
    does, not refreshed."""
    found = Evidence(days=max(0, now - since_ms) / DAY_MS)
    if store is None:
        return found
    db = store.db
    reviews = labels.latest_reviews(db)
    zones = splits.Zones.load(db)
    found.sessions, found.clusters = zones.sessions, zones.clusters
    for row in db.execute(f"SELECT c.* FROM clips c WHERE {labels.USABLE} AND c.timestamp_ms > ?"
                          " ORDER BY c.timestamp_ms, c.clip_id", (since_ms,)):
        review = reviews.get(row["clip_id"])
        if review is None or review["action"] == "reject" or review["strategy"] != "random":
            continue
        words = len(normalized_words(review["text"] or ""))
        if not words or zones.clusters.get(row["clip_id"], row["clip_id"]) in zones.trained:
            continue
        found.rows.append({**dict(row), "reference_text": review["text"], "label": "verified"})
        found.words += words
    return found


@contextlib.contextmanager
def _dataset():
    """The dataset store, or None before the first collect (no clips: nothing to count)."""
    store = Store(sweep=False) if (dataset_root() / "dataset.sqlite").is_file() else None
    try:
        yield store
    finally:
        if store is not None:
            store.close()


def pending(store: Store, now: int | None = None) -> str | None:
    """`healing status`'s nudge: the waiting line while the current promotion's watch waits for reviews."""
    from healing.registry import store as registry

    target, _ = watched(registry.load())
    if target is None or decision_for(target) is not None:
        return None
    found = evidence(store, target.promoted_at_ms, now_ms() if now is None else now)
    return None if found.due else found.waiting()


# ---- judging --------------------------------------------------------------------------------------

def paired(previous: dict[str, stats.ClipScore], promoted: dict[str, stats.ClipScore], *, resamples: int,
           seed: int) -> stats.PairedResult:
    """Δ = promoted − previous with the 90% two-sided percentile interval of the one bootstrap: `ci_low`
    is its 5th percentile, the one-sided 95% lower bound. Positive Δ means the promoted model is worse."""
    return stats.paired_bootstrap(previous, promoted, resamples=resamples, seed=seed, confidence=CONFIDENCE)


def _interval(result: stats.PairedResult | None) -> dict | None:
    if result is None:
        return None
    return {"delta": result.delta, "ci_low": result.ci_low, "ci_high": result.ci_high, "wer_previous": result.wer_a,
            "wer_promoted": result.wer_b, "clips": result.clips, "reference_words": result.reference_words,
            "confidence": "one-sided 95% lower bound (90% two-sided interval)"}


def judge(document: dict, *, promoted: str, previous: str, margins: gate.Margins,
          streams_problem: str | None) -> tuple[dict, list[str]]:
    """({wer, formatting, streaming}, what regressed) on the watch set (`rolling` in the eval). WER
    regresses when its lower bound is above 0; formatting when its lower bound is above the gate's
    formatting margin; streaming when Hex stopped streaming the promoted model or its long-clip p95
    time to text is over the gate's margin (unmeasured is never a regression)."""
    a, b = gate.rows_for(document, previous, "rolling"), gate.rows_for(document, promoted, "rolling")
    if not a or not b:
        raise WatchError("the eval did not score the watch set for both models")
    wer = paired(gate.clip_scores(a), gate.clip_scores(b), resamples=margins.resamples, seed=margins.seed)
    regressed = ["wer"] if wer.ci_low > 0 else []
    formatting = None
    format_a, format_b = (gate.clip_scores(rows, "format_errors", "format_words") for rows in (a, b))
    if sum(s.reference_words for s in format_a.values()):
        formatting = paired(format_a, format_b, resamples=margins.resamples, seed=margins.seed)
        if formatting.ci_low > margins.formatting_ci_high:
            regressed.append("formatting")
    p95_b, long_clips, why_b = gate.p95_long(document, promoted, margins)
    p95_a, _, why_a = gate.p95_long(document, previous, margins)
    streaming = {"streams_problem": streams_problem, "long_clips": long_clips,
                 "p95_delta_ms": None if p95_a is None or p95_b is None else p95_b - p95_a,
                 "threshold_ms": margins.streaming_p95_ms, "note": why_b or why_a}
    if streams_problem or (streaming["p95_delta_ms"] is not None and streaming["p95_delta_ms"] > margins.streaming_p95_ms):
        regressed.append("streaming")
    return {"wer": _interval(wer), "formatting": _interval(formatting), "streaming": streaming}, regressed


def _pt(value: float) -> str:
    return f"{value * 100:+.1f}".replace("-", "−")


def _delta(result: dict | None) -> str:
    return "" if result is None else f"Δ {_pt(result['delta'])} [{_pt(result['ci_low'])}, {_pt(result['ci_high'])}]"


# ---- talking to Hex ------------------------------------------------------------------------------

def _hex_ids(version) -> set[str]:
    """Every id Hex may run `version` under: its GGUFs' imports, and v0's catalog id."""
    from healing.registry import objects

    ids = {objects.hex_model_id(sha) for sha in version.ggufs()}
    catalog_id = (version.record.get("base") or {}).get("catalog_id")
    if catalog_id and not version.artifact("lora_delta"):
        ids.add(catalog_id)
    return ids


def _streams_problem(binary: Path, status: dict, target: Watched) -> str | None:
    """Hex no longer streams the promoted model while it streams the previous one, with the lab on (a lab
    you turned off is your choice, not the model's regression)."""
    from healing import ship

    if not ship._lab_on(status):
        return None
    listed = ship._models(binary)
    now, before = (listed.get(ship.hex_id(v)) or {} for v in (target.version, target.previous))
    if now.get("streams") is False and before.get("streams") is True:
        return f"Hex doesn't stream {target.version.name} but streams {target.previous.name}"
    return None


# ---- the run --------------------------------------------------------------------------------------

def _evaluate(paths: list[Path], ref) -> dict:
    """`healing eval` of both models on the watch set, with the vocabulary Hex runs."""
    from healing.eval import run as eval_run

    return eval_run.evaluate(paths, ["rolling"], vocabulary="current", resolved={"rolling": ref},
                             progress=lambda message: print(message, file=sys.stderr))


def _score(store: Store, target: Watched, found: Evidence, evaluate: Callable) -> tuple[dict, str | None]:
    """Write the watch set to scratch, score both models on it, and save the eval: (document, its path)."""
    from healing.eval import run as eval_run
    from healing.registry.objects import owner_dir

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    items = splits.entries(store, found.rows, found.clusters, found.sessions)
    work = healing_home() / "tmp"
    owner_dir(work)
    with tempfile.TemporaryDirectory(dir=work, prefix="watch-") as scratch:
        info = splits.write_and_pin(store, f"watch-{target.version.name}-{stamp}", items, root=Path(scratch))
        ref = eval_run.SetRef("rolling", info.split_id, info.path.parent, info.manifest_sha256, info.n_clips)
        with gate.model_files([target.previous, target.version]) as paths:
            document = evaluate(paths, ref)
    path = eval_run.write(document) if document.get("sets") else None
    return document, None if path is None else str(path)


def _decide(binary: Path, target: Watched, found: Evidence, dry_run: bool, evaluate: Callable, restore: Callable,
            say: Callable[[str], None], confirm_s: float, poll_s: float) -> int:
    from healing import ship
    from healing.registry import objects
    from healing.registry import store as registry

    name, previous = target.version.name, target.previous.name
    clips, words, days = len(found.rows), found.words, int(found.days)
    base = {"promoted_at_ms": target.promoted_at_ms, "previous": previous, "words": words, "clips": clips,
            "days": round(found.days, 2)}
    thin = not found.enough
    if dry_run:
        restoring = "" if objects.has(target.previous.gguf) else f"; would restore {previous}'s GGUF first"
        say(f"would score {name} against {previous} on {clips} clips ({words} words, day {days}/{MAX_DAYS})"
            + (", too few to roll back" if thin else "") + restoring)
        return 0
    if not clips:  # nothing to score: thin after 14 days
        registry.record_watch(name, {**base, "verdict": "insufficient", "eval_id": None, "eval_path": None,
                                     "wer": None, "formatting": None, "streaming": None, "regressed": []})
        say(f"insufficient evidence, kept {name}: {words}/{TARGET_WORDS} words after {days} days")
        return 0
    if not objects.has(target.previous.gguf):
        say(f"restoring {previous}'s pruned GGUF")
        restore(previous)
        target = Watched(registry.load().get(name), registry.load().get(previous), target.promoted_at_ms)
    status = ship._status(binary)
    problem = _streams_problem(binary, status, target)
    margins = gate.Margins.load()
    with _dataset() as store:
        document, eval_path = _score(store, target, found, evaluate)
    checks, regressed = judge(document, promoted=target.version.gguf, previous=target.previous.gguf,
                              margins=margins, streams_problem=problem)
    decision = {**base, "eval_id": document.get("eval_id"), "eval_path": eval_path, **checks, "regressed": regressed,
                "margins_sha": margins.sha256,
                "models": {"promoted": target.version.gguf, "previous": target.previous.gguf}}
    shown = _delta(checks["wer"])
    if thin:
        registry.record_watch(name, {**decision, "verdict": "insufficient"})
        say(f"insufficient evidence, kept {name}: {words}/{TARGET_WORDS} words after {days} days; {shown}")
        return 0
    if not regressed:
        registry.record_watch(name, {**decision, "verdict": "kept"})
        say(f"kept {name}: {shown}")
        return 0
    reason = (f"watch: {', '.join(regressed)} regressed; Δ WER {_pt(checks['wer']['delta'])} pt "
              f"[{_pt(checks['wer']['ci_low'])}, {_pt(checks['wer']['ci_high'])}] on {words} words ({document.get('eval_id')})")
    code = ship.rollback(binary=binary, to=previous, reason=reason, say=say, confirm_s=confirm_s, poll_s=poll_s,
                         expect_leaving=name)
    if code != 0:
        say(f"{name} is not rolled back (exit {code}); the next watch tries again")
        return code
    registry.record_watch(name, {**decision, "verdict": "rolled_back"})
    say(f"rolled back {name} → {previous}: {shown}")
    for check in regressed:
        if check == "formatting":
            say(f"formatting regressed: {_delta(checks['formatting'])}, above the gate's "
                f"{_pt(margins.formatting_ci_high)} pt")
        elif check == "streaming":
            stream = checks["streaming"]
            say("streaming regressed: " + (stream["streams_problem"] or f"p95 time to text {stream['p95_delta_ms']:+.0f} ms "
                                           f"on {stream['long_clips']} long clips"))
    return 0


def _run(binary: Path, dry_run: bool, now: int, evaluate: Callable, restore: Callable, say: Callable[[str], None],
         confirm_s: float, poll_s: float) -> int:
    from healing import ship
    from healing.registry import store as registry

    target, why = watched(registry.load())
    if target is None:
        say(f"nothing to watch: {why}")
        return 0
    decided = decision_for(target)
    name = target.version.name
    if decided is not None:
        verdict = {"kept": "kept", "insufficient": "kept (insufficient evidence)"}.get(decided["verdict"], decided["verdict"])
        say(f"decided: {verdict} {name} against {target.previous.name} " + _delta(decided.get("wer")))
        return 0
    running = ship._running(ship._status(binary))  # Hex down: exit 2, nothing changed
    if running not in _hex_ids(target.version):
        say(f"warning: Hex runs {running}, not {name} ({', '.join(sorted(_hex_ids(target.version)))}); you switched "
            "models by hand after the promotion, so watch does nothing (promote or pick it in Hex to watch again)")
        return EXIT_MANUAL
    with _dataset() as store:
        found = evidence(store, target.promoted_at_ms, now)
    if not found.due:
        say(found.waiting())
        return 0
    return _decide(binary, target, found, dry_run, evaluate, restore, say, confirm_s, poll_s)


def run(*, binary: Path, dry_run: bool = False, now: int | None = None, evaluate: Callable | None = None,
        restore: Callable | None = None, say: Callable[[str], None] = print, confirm_s: float | None = None,
        poll_s: float | None = None) -> int:
    from healing import ship
    from healing.eval.bench import BenchError
    from healing.eval.run import EvalError
    from healing.registry import restore as restore_module
    from healing.registry.store import RegistryError

    try:
        with lock():
            return _run(Path(binary), dry_run, now_ms() if now is None else now, evaluate or _evaluate,
                        restore or restore_module.restore, say, ship.CONFIRM_S if confirm_s is None else confirm_s,
                        ship.POLL_S if poll_s is None else poll_s)
    except (WatchError, ship.ShipError) as error:
        say(f"error: {error}")
        return error.code
    except (gate.GateError, RegistryError, restore_module.RestoreError, EvalError, BenchError, StoreError, OSError,
            ValueError) as error:
        say(f"error: {error}")
        return 1


def main(args) -> int:
    from healing.eval.bench import BenchError, history_hex

    try:
        binary = Path(args.hex or history_hex())
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return run(binary=binary, dry_run=args.dry_run)
