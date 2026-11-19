import json

import numpy as np
import pytest

from healing.stats import (
    ClipScore,
    bootstrap_deltas,
    paired_bootstrap,
    read_clip_scores,
    resample_indices,
)


def scores(rows):
    return {cid: ClipScore(cid, e, w) for cid, e, w in rows}


def test_paired_bootstrap_matches_fixture():
    # Hand-computed reference: the same algorithm written as a plain loop.
    a = scores([("c1", 1, 10), ("c2", 0, 5), ("c3", 4, 20), ("c4", 2, 8)])
    b = scores([("c1", 0, 10), ("c2", 1, 5), ("c3", 1, 20), ("c4", 2, 8)])
    result = paired_bootstrap(a, b, resamples=2_000, seed=7)

    ids = sorted(a)
    draws = np.random.default_rng(7).integers(0, 4, size=(2_000, 4))
    by_hand = []
    for row in draws:
        picked = [ids[i] for i in row]
        words = sum(a[c].reference_words for c in picked)
        by_hand.append(
            (sum(b[c].word_errors for c in picked) - sum(a[c].word_errors for c in picked))
            / words
        )
    low, high = np.quantile(by_hand, [0.025, 0.975])

    assert result.delta == pytest.approx((4 - 7) / 43)
    assert result.wer_a == pytest.approx(7 / 43)
    assert result.ci_low == pytest.approx(low)
    assert result.ci_high == pytest.approx(high)
    assert result.ci_low <= result.delta <= result.ci_high


def test_identical_runs_have_a_zero_interval():
    a = scores([(f"c{i}", i % 3, 10) for i in range(30)])
    result = paired_bootstrap(a, a)
    assert (result.delta, result.ci_low, result.ci_high) == (0.0, 0.0, 0.0)


def test_resamples_clips_not_words():
    # One clip wrong in every word, one clip right. Resampling clips can only
    # give Δ of 0, 1/2 or 1; resampling words would give many other values.
    a = scores([("bad", 0, 10), ("good", 0, 10)])
    b = scores([("bad", 10, 10), ("good", 0, 10)])
    deltas = bootstrap_deltas(a, b, resamples=500, seed=1)
    assert set(np.round(deltas, 9)) <= {0.0, 0.5, 1.0}
    assert {0.0, 0.5, 1.0} <= set(np.round(deltas, 9))


def test_seeded_deterministic():
    a = scores([(f"c{i}", i % 4, 12) for i in range(50)])
    b = scores([(f"c{i}", (i + 1) % 4, 12) for i in range(50)])
    assert paired_bootstrap(a, b, seed=3) == paired_bootstrap(a, b, seed=3)
    assert (resample_indices(5, 4, 9) == resample_indices(5, 4, 9)).all()


def test_mismatched_clips_refused():
    with pytest.raises(ValueError, match="different clips"):
        paired_bootstrap(scores([("x", 0, 1)]), scores([("y", 0, 1)]))
    with pytest.raises(ValueError, match="different references"):
        paired_bootstrap(scores([("x", 0, 1)]), scores([("x", 0, 2)]))


def test_reads_benchmark_clip_records(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"type": "model", "load_ms": 1},
                {"type": "clip", "id": "a", "word_errors": 2, "reference_words": 9},
                {"type": "measurement", "id": "a"},
                {"type": "summary", "wer": 0.2},
            ]
        )
    )
    assert read_clip_scores(path) == {"a": ClipScore("a", 2, 9)}


def test_paired_quantile_deltas_resample_clips_together():
    import numpy as np

    from healing import stats

    a = np.arange(100, dtype=float)
    assert not stats.paired_quantile_deltas(a, a, 95, resamples=200, seed=1).any()  # same clips, same p95
    shifted = stats.paired_quantile_deltas(a, a + 50, 95, resamples=200, seed=1)
    assert np.allclose(shifted, 50)
