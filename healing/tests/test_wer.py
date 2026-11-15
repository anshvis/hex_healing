import json
from pathlib import Path

import pytest

from healing.wer import align, normalized_words, word_errors


def matched(reference, hypothesis, wanted=None):
    return align(reference, hypothesis, wanted)[1]


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
    from healing.wer import align, positions_of, surface_errors, word_spans

    assert align("kitten", "sitting")[0] == 3
    assert surface_errors("Send it, Kit.", "send it Kit") == (3, 3)
    assert matched(["a", "b", "c"], ["a", "x", "c"]) == [True, False, True]
    assert [w for w, _, _ in word_spans("Hello, World")] == normalized_words("Hello, World")
    assert positions_of("send it to kit", [(11, 14)]) == [[3, 4]]


def test_one_alignment_judges_every_position_so_repeats_are_not_double_credited():
    # "kit kit" against "kit": one word is matched, the other is not, never both
    assert sum(matched(["kit", "kit"], ["kit"])) == 1
    assert sum(matched(["a", "b", "a"], ["a"])) == 1
    # with the edited positions named, the alignment that matches them wins
    assert matched(["a", "b", "a"], ["a"], wanted={2}) == [False, False, True]
    assert matched(["a", "b", "a"], ["a"], wanted={0}) == [True, False, False]
    assert matched(["kit", "kit"], ["kit"], wanted={0, 1}).count(True) == 1
    assert matched(["x", "a", "y"], ["x", "y"]) == [True, False, True]
    assert matched(["b", "a"], ["a", "a"]) == [False, True]


def test_distance_agrees_with_the_aligned_distance():
    from healing.wer import align, distance

    for a, b in (("kitten", "sitting"), ("", "abc"), ("abc", ""), ("flaw", "lawn"), ("same", "same")):
        assert distance(a, b) == align(a, b)[0]
    assert distance(["a", "b"], ["b", "a"]) == 2


def test_the_wanted_tie_break_changes_which_positions_match():
    ref, hyp = ["b", "a", "a", "b", "a", "b"], ["a", "b", "b"]
    hits = lambda wanted: sum(matched(ref, hyp, wanted)[i] for i in wanted)  # noqa: E731
    assert sum(matched(ref, hyp)) == 3
    assert hits({1, 3, 5}) == 3  # one alignment can match all three of these
    assert hits({1, 4, 5}) == 2  # only one "a" is in the transcript: two of the three is the best any alignment does
    assert matched(ref, hyp, {1, 3, 5}) != matched(ref, hyp, {0, 2, 4}) and hits({0, 2, 4}) == 2


def test_the_chosen_alignment_is_optimal_against_an_exhaustive_search():
    import functools
    import random

    def best(reference, hypothesis, wanted):
        @functools.lru_cache(None)
        def go(i, j):  # (cost, -wanted matches) of the best alignment of the suffixes
            if i == len(reference) and j == len(hypothesis):
                return (0, 0)
            options = []
            if i < len(reference) and j < len(hypothesis):
                same = reference[i] == hypothesis[j]
                cost, gain = go(i + 1, j + 1)
                options.append((cost + (not same), gain - (1 if same and i in wanted else 0)))
            if i < len(reference):
                cost, gain = go(i + 1, j)
                options.append((cost + 1, gain))
            if j < len(hypothesis):
                cost, gain = go(i, j + 1)
                options.append((cost + 1, gain))
            return min(options)

        return go(0, 0)

    rng = random.Random(4)
    for _ in range(400):
        reference = tuple(rng.choice("abc") for _ in range(rng.randint(1, 7)))
        hypothesis = tuple(rng.choice("abc") for _ in range(rng.randint(0, 7)))
        wanted = frozenset(i for i in range(len(reference)) if rng.random() < 0.5)
        cost, flags = align(list(reference), list(hypothesis), wanted)
        assert (cost, -sum(flags[i] for i in wanted)) == best(reference, hypothesis, wanted)
