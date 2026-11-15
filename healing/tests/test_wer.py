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


def test_one_alignment_serves_every_count():
    from healing.wer import align, matched_positions, positions_of, surface_errors, word_spans

    assert align("kitten", "sitting")[0] == 3
    assert surface_errors("Send it, Kit.", "send it Kit") == (3, 3)
    assert matched_positions(["a", "b", "c"], ["a", "x", "c"]) == [True, False, True]
    assert [w for w, _, _ in word_spans("Hello, World")] == normalized_words("Hello, World")
    assert positions_of("send it to kit", [(11, 14)]) == [[3, 4]]


def test_one_alignment_judges_every_position_so_repeats_are_not_double_credited():
    from healing.wer import matched_positions

    # "kit kit" against "kit": one word is matched, the other is not, never both
    assert sum(matched_positions(["kit", "kit"], ["kit"])) == 1
    assert sum(matched_positions(["a", "b", "a"], ["a"])) == 1
    # with the edited positions named, the alignment that matches them wins
    assert matched_positions(["a", "b", "a"], ["a"], wanted={2}) == [False, False, True]
    assert matched_positions(["a", "b", "a"], ["a"], wanted={0}) == [True, False, False]
    assert matched_positions(["kit", "kit"], ["kit"], wanted={0, 1}).count(True) == 1
    assert matched_positions(["x", "a", "y"], ["x", "y"]) == [True, False, True]
    assert matched_positions(["b", "a"], ["a", "a"]) == [False, True]


def test_distance_agrees_with_the_aligned_distance():
    from healing.wer import align, distance

    for a, b in (("kitten", "sitting"), ("", "abc"), ("abc", ""), ("flaw", "lawn"), ("same", "same")):
        assert distance(a, b) == align(a, b)[0]
    assert distance(["a", "b"], ["b", "a"]) == 2
