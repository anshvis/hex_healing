import json
from pathlib import Path

import pytest

from healing.wer import normalized_words, word_errors


def test_normalizes_case_and_punctuation_like_the_benchmark():
    # benchmark.rs: word_error_rate_normalizes_case_and_punctuation
    assert word_errors("Use OpenCode, please.", "use opencode please") == (0, 3)
    assert normalized_words("We've got 2-3 Zorblax!") == ["we", "ve", "got", "2", "3", "zorblax"]


def test_counts_substitutions_insertions_and_deletions():
    assert word_errors("a b c", "a x c") == (1, 3)
    assert word_errors("a b c", "a b c d") == (1, 3)
    assert word_errors("a b c", "a c") == (1, 3)
    assert word_errors("a b", "") == (2, 2)


BASELINE_RUNS = sorted(
    (Path.home() / "Library/Application Support/hex-healing/evals").glob("baseline-*.jsonl")
)


@pytest.mark.skipif(not BASELINE_RUNS, reason="needs a recorded benchmark run")
def test_agrees_with_the_benchmark_on_every_recorded_clip():
    clips = [
        json.loads(line)
        for line in BASELINE_RUNS[-1].read_text().splitlines()
        if json.loads(line).get("type") == "clip"
    ]
    for clip in clips:
        assert word_errors(clip["expected"], clip["transcript"]) == (
            clip["word_errors"],
            clip["reference_words"],
        ), clip["id"]
