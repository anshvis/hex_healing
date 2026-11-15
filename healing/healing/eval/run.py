"""`healing eval --model <gguf> [--model ...]`: one or more GGUFs on every test set.

Each model runs through `hex benchmark-transcription` on the same machine in the same invocation,
interleaved per set (A, B, A, B) so thermal drift is shared. Scores are never compared with a
stored number. The eval JSON keeps per-clip records for the paired bootstrap and McNemar tests
(T0.1's `healing stats`, M7's gate).

Scores per set:
- `wer`: the benchmark's own (case and punctuation stripped);
- `format_error_rate`: errors over whitespace tokens that keep case and punctuation (so it sees a
  model that forgot a comma); n/a on the general set, whose references are unformatted;
- `term_error_rate`: 1 - the benchmark's `term_recall`;
- edited-span error rate, on the edited set: the share of words you corrected that the candidate
  still gets wrong.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from healing import manifest as manifest_module
from healing.eval import baseline
from healing.eval.bench import BenchError, BenchRun, hex_binary, run_benchmark
from healing.paths import healing_home
from healing.sets import general
from healing.store import Store, StoreError
from healing.wer import matched_positions, normalized_words, surface_errors

SET_NAMES = ("rolling", "golden", "general", "edited")
VOCABULARY_MODES = ("current", "none")
UNFORMATTED = {"general"}  # LibriSpeech and AMI references carry no casing or punctuation


class EvalError(RuntimeError):
    pass


@dataclass(frozen=True)
class SetRef:
    name: str
    split_id: str
    folder: Path
    manifest_sha256: str
    n: int

    @property
    def manifest(self) -> Path:
        return self.folder / "manifest.json"


def hex_support_dir() -> Path:
    named = os.environ.get("HEX_APPLICATION_SUPPORT_DIR")
    return Path(named) if named else Path.home() / "Library" / "Application Support" / "hex"


def current_vocabulary(settings_path: Path | None = None) -> tuple[list[dict], str | None, str | None]:
    """Hex's global vocabulary as dictation uses it: (entries, boost, why none).

    Read-only from `settings.json` (never through `hex`, never written). Dictation biases only
    while Labs > Vocabulary biasing is on, so with it off the answer is empty.
    """
    path = settings_path or hex_support_dir() / "settings.json"
    try:
        settings = json.loads(path.read_text())
    except FileNotFoundError:
        return [], None, f"no settings file at {path}"
    except (OSError, ValueError) as error:
        raise EvalError(f"cannot read Hex's settings at {path}: {error}") from error
    entries = []
    for item in settings.get("vocabulary", []):
        term, sounds = (item, []) if isinstance(item, str) else (item.get("term", ""), item.get("sounds_like", []))
        if term.strip():
            entries.append({"term": term, "sounds_like": list(sounds)})
    if entries and not settings.get("labs", {}).get("vocabulary_biasing", False):
        return [], None, "Labs > Vocabulary biasing is off, so dictation doesn't bias with your vocabulary"
    boost = settings.get("vocabulary_boost")
    if entries and boost == "off":
        return [], None, "the vocabulary boost is Off, so dictation leaves transcripts as heard"
    return entries, boost, None if entries else "your vocabulary is empty"


def resolve_set(name: str, store: Store | None) -> SetRef | str:
    """The set's folder and pin, or the reason it isn't available (reported as n/a)."""
    if name == "general":
        folder = healing_home() / "sets" / "general"
        if not (folder / "manifest.json").is_file():
            return "no general set: run `healing sets build-general`"
        digest = _verified(folder)
        return SetRef(name, "general-v1", folder, digest, len(json.loads((folder / "manifest.json").read_text())["clips"]))
    if store is None:
        return "no dataset yet: run `healing collect`"
    if name == "golden":
        row = store.db.execute("SELECT * FROM splits WHERE split_id = 'golden-v1'").fetchone()
        if row is None or not row["frozen"]:
            have = row["n_clips"] if row else 0
            return f"golden is pending ({have} confirmed): run `healing golden review`"
    else:
        row = store.db.execute(
            "SELECT * FROM splits WHERE kind = ? AND manifest_path IS NOT NULL ORDER BY created_at_ms DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row is None:
            return f"no {name} set: run `healing split build`" + (
                " (needs random-verified clips after the cut-off)" if name == "rolling" else ""
            )
    folder = Path(row["manifest_path"]).parent
    digest = _verified(folder)
    if digest != row["manifest_sha256"]:
        raise EvalError(f"{row['split_id']}: manifest changed since it was built; run `healing split verify`")
    return SetRef(name, row["split_id"], folder, digest, row["n_clips"])


def _verified(folder: Path) -> str:
    try:
        return general.verify(folder)
    except general.BuildError as error:
        raise EvalError(str(error)) from error


def _sidecar(folder: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (folder / "clips.jsonl").read_text().splitlines())}


def score_set(ref: SetRef, runs: list[BenchRun]) -> tuple[dict, list[dict], dict | None]:
    """(scores, per-clip records, edited-span totals) for one model on one set."""
    first = runs[0]
    manifest = json.loads(ref.manifest.read_text())
    definitions = {c["id"]: c for c in manifest["clips"]}
    sidecar = _sidecar(ref.folder)
    formatted = ref.name not in UNFORMATTED
    per_clip, term_hits = [], []
    format_errors = format_words = span_errors = span_positions = 0
    for record in first.clips:
        definition = definitions[record["id"]]
        transcript = record["transcript"]
        terms = definition.get("expected_terms", [])
        hit = [t for t in terms if manifest_module.term_count(transcript, t) > 0]
        term_hits += [{"id": record["id"], "term": t, "hit": t in hit} for t in terms]
        errors, words = surface_errors(definition["expected"], transcript) if formatted else (0, 0)
        format_errors, format_words = format_errors + errors, format_words + words
        row = {
            "id": record["id"], "word_errors": record["word_errors"], "reference_words": record["reference_words"],
            "format_errors": errors, "format_words": words, "terms_hit": hit, "terms_expected": terms,
            "transcript": transcript,
        }
        positions = sidecar.get(record["id"], {}).get("edited_positions")
        if ref.name == "edited" and positions:
            matched = matched_positions(normalized_words(definition["expected"]), normalized_words(transcript))
            wrong = sum(1 for start, end in positions for i in range(start, end) if not matched[i])
            total = sum(end - start for start, end in positions)
            row["edited_errors"], row["edited_positions"] = wrong, total
            span_errors, span_positions = span_errors + wrong, span_positions + total
        per_clip.append(row)
    recall = first.summary["quality"].get("term_recall")
    scores = {
        "wer": first.summary["wer"],
        "word_errors": first.summary["word_errors"],
        "ref_words": first.summary["reference_words"],
        "format_error_rate": format_errors / format_words if format_words else None,
        "term_error_rate": None if recall is None else 1 - recall,
        "term_hits": term_hits,
        "median_rtf": statistics.median(r.summary["median_rtf"] for r in runs),
        "p95_clip_ms": statistics.median(r.summary["p95_clip_ms"] for r in runs),
    }
    span = {"errors": span_errors, "positions": span_positions} if ref.name == "edited" else None
    return scores, per_clip, span


def evaluate(
    models: list[Path],
    set_names: list[str],
    *,
    runs: int = 1,
    vocabulary: str = "current",
    store: Store | None = None,
    settings_path: Path | None = None,
    progress: Callable[[str], None] = lambda message: None,
    bench=run_benchmark,
    now: Callable[[], float] = time.time,
) -> dict:
    if vocabulary not in VOCABULARY_MODES:
        raise EvalError(f"--vocabulary is one of {', '.join(VOCABULARY_MODES)}")
    if len({str(m) for m in models}) != len(models):
        raise EvalError("the same model was named twice")
    for model in models:
        if not model.is_file():
            raise EvalError(f"no model file at {model}")
    started = now()
    hex_path = hex_binary()
    refs: list[SetRef] = []
    missing: dict[str, str] = {}
    for name in set_names:
        found = resolve_set(name, store)
        if isinstance(found, str):
            missing[name] = found
            progress(f"{name}: n/a ({found})")
        else:
            refs.append(found)

    entries, boost, why_none = ([], None, None)
    if vocabulary == "current":
        entries, boost, why_none = current_vocabulary(settings_path)
        if not entries and why_none:
            progress(f"vocabulary: none passed ({why_none})")
    document = {
        "eval_id": "eval-" + datetime.fromtimestamp(started, timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "hex_sha": baseline._git_sha() if baseline._built_here(str(hex_path)) else None,
        "hex_version": baseline._hex_version(str(hex_path)),
        "hex_binary_sha256": baseline._sha256(hex_path),
        "models": [{"gguf_path": str(m), "gguf_sha256": baseline._sha256(m)} for m in models],
        "sets": [{"split_id": r.split_id, "manifest_sha256": r.manifest_sha256, "n": r.n} for r in refs],
        "unavailable_sets": missing,
        "vocabulary": {"mode": vocabulary, "terms": len(entries), "boost": boost if entries else None},
        "scores": {str(m): {} for m in models},
        "per_clip": {str(m): {} for m in models},
        "edited_span": {},
        "started_at_ms": int(started * 1000),
    }
    with tempfile.TemporaryDirectory(prefix="healing-vocabulary-") as folder:
        vocabulary_file = None
        if entries:
            vocabulary_file = Path(folder) / "vocabulary.json"
            vocabulary_file.write_text(json.dumps(entries, indent=1))
        for ref in refs:
            results: dict[str, list[BenchRun]] = {str(m): [] for m in models}
            for round_number in range(1, runs + 1):
                for model in models:
                    progress(f"{ref.name} ({ref.n} clips): {model.name}, run {round_number} of {runs}")
                    try:
                        run, _ = bench(
                            ref.manifest, model=model, runs=1, warmups=1,
                            vocabulary=vocabulary_file, boost=boost if boost not in (None, "normal") else None,
                        )
                    except BenchError as error:
                        raise EvalError(f"{model} on {ref.name}: {error}") from error
                    results[str(model)].append(run)
            for model in models:
                scores, per_clip, span = score_set(ref, results[str(model)])
                document["scores"][str(model)][ref.name] = scores
                document["per_clip"][str(model)][ref.name] = per_clip
                if span is not None:
                    document["edited_span"][str(model)] = span
    document["duration_ms"] = int((now() - started) * 1000)
    return document


def write(document: dict) -> Path:
    folder = healing_home() / "evals"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{document['eval_id']}.json"
    path.write_text(json.dumps(document, indent=1) + "\n")
    return path


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}"


def table(document: dict) -> str:
    """Rows are sets, columns are models; every number is a percent."""
    models = [m["gguf_path"] for m in document["models"]]
    lines = [f"{'set':<10}{'model':<34}{'clips':>6}{'WER':>8}{'format':>8}{'term err':>10}{'edited':>8}"]
    shown = {s["split_id"]: s for s in document["sets"]}
    for name in SET_NAMES:
        ref = next((s for s in document["sets"] if _set_name(s["split_id"]) == name), None)
        if ref is None:
            lines.append(f"{name:<10}n/a  {document['unavailable_sets'].get(name, 'not requested')}")
            continue
        for model in models:
            scores = document["scores"][model][name]
            span = document["edited_span"].get(model) if name == "edited" else None
            edited = _percent(span["errors"] / span["positions"]) if span and span["positions"] else "n/a"
            lines.append(
                f"{name:<10}{Path(model).name[:32]:<34}{shown[ref['split_id']]['n']:>6}{_percent(scores['wer']):>8}"
                f"{_percent(scores['format_error_rate']):>8}{_percent(scores['term_error_rate']):>10}{edited:>8}"
            )
    return "\n".join(lines)


def _set_name(split_id: str) -> str:
    return split_id.split("-", 1)[0]


def main(args) -> int:
    import sys

    if not args.models:
        print("error: healing eval needs at least one --model <gguf>", file=sys.stderr)
        return 2
    names = [n.strip() for n in args.sets.split(",") if n.strip()]
    unknown = [n for n in names if n not in SET_NAMES]
    if unknown:
        print(f"error: unknown set {unknown[0]!r}; sets are {', '.join(SET_NAMES)}", file=sys.stderr)
        return 2
    store = None
    try:
        if (healing_home() / "dataset" / "dataset.sqlite").is_file():
            store = Store(sweep=False)
        try:
            document = evaluate(
                [Path(m) for m in args.models], names, runs=args.runs, vocabulary=args.vocabulary,
                store=store, progress=lambda message: print(message, file=sys.stderr),
            )
        finally:
            if store is not None:
                store.close()
    except (EvalError, BenchError, StoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    path = write(document)
    print(table(document))
    print(f"\nwrote {path}")
    return 0
