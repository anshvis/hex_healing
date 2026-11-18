"""Paired statistics over clips: the one bootstrap and the one McNemar test every gate uses.

Both sides are scored on the same clips. Each resample draws clips with
replacement and recomputes each side's WER as total errors over total
reference words (a ratio of sums, as the benchmark does), so long clips weigh
more and words within a clip move together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClipScore:
    id: str
    word_errors: int
    reference_words: int


@dataclass(frozen=True)
class PairedResult:
    clips: int
    reference_words: int
    wer_a: float
    wer_b: float
    delta: float
    ci_low: float
    ci_high: float
    resamples: int
    seed: int

    def as_dict(self) -> dict:
        return asdict(self)


def read_clip_scores(path: Path) -> dict[str, ClipScore]:
    """Per-clip scores from benchmark JSONL (`type: clip` records)."""
    scores: dict[str, ClipScore] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "clip":
            continue
        if record["id"] in scores:
            raise ValueError(f"{path}: clip {record['id']} appears twice")
        scores[record["id"]] = ClipScore(
            record["id"], int(record["word_errors"]), int(record["reference_words"])
        )
    if not scores:
        raise ValueError(f"{path}: no clip records")
    return scores


def resample_indices(clips: int, resamples: int, seed: int) -> np.ndarray:
    """`resamples` rows of `clips` clip indices drawn with replacement."""
    return np.random.default_rng(seed).integers(0, clips, size=(resamples, clips))


def bootstrap_deltas(
    a: dict[str, ClipScore], b: dict[str, ClipScore], *, resamples: int, seed: int
) -> np.ndarray:
    """Every resample's Δ WER = WER(b) − WER(a)."""
    errors_a, errors_b, words = _paired_arrays(a, b)
    draws = resample_indices(len(words), resamples, seed)
    return (errors_b[draws].sum(axis=1) - errors_a[draws].sum(axis=1)) / words[draws].sum(axis=1)


def paired_bootstrap(
    a: dict[str, ClipScore],
    b: dict[str, ClipScore],
    *,
    resamples: int = 10_000,
    seed: int = 13,
    confidence: float = 0.95,
) -> PairedResult:
    """Δ WER = WER(b) − WER(a), with a percentile CI over clip resamples."""
    errors_a, errors_b, words = _paired_arrays(a, b)
    deltas = bootstrap_deltas(a, b, resamples=resamples, seed=seed)
    tail = (1 - confidence) / 2
    low, high = np.quantile(deltas, [tail, 1 - tail])
    wer_a = errors_a.sum() / words.sum()
    wer_b = errors_b.sum() / words.sum()
    return PairedResult(
        clips=len(words),
        reference_words=int(words.sum()),
        wer_a=float(wer_a),
        wer_b=float(wer_b),
        delta=float(wer_b - wer_a),
        ci_low=float(low),
        ci_high=float(high),
        resamples=resamples,
        seed=seed,
    )


def _paired_arrays(
    a: dict[str, ClipScore], b: dict[str, ClipScore]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if a.keys() != b.keys():
        differing = sorted(a.keys() ^ b.keys())[:5]
        raise ValueError(f"the two runs scored different clips, e.g. {differing}")
    ids = sorted(a)
    for clip in ids:
        if a[clip].reference_words != b[clip].reference_words:
            raise ValueError(f"clip {clip} has different references in the two runs")
    words = np.array([a[c].reference_words for c in ids], dtype=np.int64)
    if words.sum() == 0:
        raise ValueError("no reference words")
    errors_a = np.array([a[c].word_errors for c in ids], dtype=np.int64)
    errors_b = np.array([b[c].word_errors for c in ids], dtype=np.int64)
    return errors_a, errors_b, words


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value: the discordant pairs split evenly under the null."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def clips_right(rows: list[dict]) -> dict[str, bool]:
    """Per eval clip that expects a term: whether every term it expects was heard. McNemar counts
    clips, not terms, so one clip that gets ten terms is not ten independent pieces of evidence."""
    return {row["id"]: all(term in row["terms_hit"] for term in row["terms_expected"])
            for row in rows if row["terms_expected"]}
