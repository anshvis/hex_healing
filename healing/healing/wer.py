"""Word errors exactly as `hex benchmark-transcription` counts them.

A port of `normalized_words` and `word_errors` in
`src/transcription/benchmark.rs`: split on anything that isn't alphanumeric,
lowercase, then word-level edit distance against the reference.

Exact for the English sets healing uses. Python's `str.isalnum` and Rust's
`char::is_alphanumeric` disagree on some combining marks (Indic, Thai), so
check a non-English set against the benchmark before trusting this port.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def word_spans(text: str) -> list[tuple[str, int, int]]:
    """`normalized_words` with each word's character span in `text`."""
    spans, start = [], None
    for index, character in enumerate(text):
        if character.isalnum():
            if start is None:
                start = index
        elif start is not None:
            spans.append((text[start:index].lower(), start, index))
            start = None
    if start is not None:
        spans.append((text[start:].lower(), start, len(text)))
    return spans


def normalized_words(text: str) -> list[str]:
    return [word for word, _, _ in word_spans(text)]


def distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Edit distance (substitution, insertion, deletion each cost 1) between two sequences."""
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j - 1] + (expected != actual), previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1]


def align(reference: Sequence, hypothesis: Sequence, wanted: Iterable[int] | None = None) -> tuple[int, list[bool]]:
    """(edit distance, matched) for ONE chosen minimum-edit alignment.

    `matched[i]` says whether that alignment pairs reference item `i` with an equal one. Several
    alignments can be equally cheap when a word repeats ("kit kit" against "kit"); the chosen one is
    the cheapest that also matches the most of the `wanted` positions (all of them when None), ties
    broken toward the earliest. A position is therefore judged by a single consistent alignment, never
    credited because some other optimal alignment would match it: "kit kit" against "kit" matches one
    of its two words, not both.
    """
    rows, cols = len(reference), len(hypothesis)
    weight = [1] * rows if wanted is None else [0] * rows
    if wanted is not None:
        for index in wanted:
            if 0 <= index < rows:
                weight[index] = 1
    cost = [[0] * (cols + 1) for _ in range(rows + 1)]
    gain = [[0] * (cols + 1) for _ in range(rows + 1)]
    move = [[""] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0], move[i][0] = i, "delete"
    for j in range(1, cols + 1):
        cost[0][j], move[0][j] = j, "insert"
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            options = [
                (cost[i - 1][j - 1] + (0 if same else 1), -(gain[i - 1][j - 1] + (weight[i - 1] if same else 0)), 0,
                 "match" if same else "substitute"),
                (cost[i - 1][j] + 1, -gain[i - 1][j], 1, "delete"),
                (cost[i][j - 1] + 1, -gain[i][j - 1], 2, "insert"),
            ]
            best = min(options)
            cost[i][j], gain[i][j], move[i][j] = best[0], -best[1], best[3]
    matched = [False] * rows
    i, j = rows, cols
    while i > 0 or j > 0:
        step = move[i][j]
        if step in ("match", "substitute"):
            matched[i - 1] = step == "match"
            i, j = i - 1, j - 1
        elif step == "delete":
            i -= 1
        else:
            j -= 1
    return cost[rows][cols], matched


def word_errors(expected: str, actual: str) -> tuple[int, int]:
    """(edit distance in words, reference words)."""
    reference = normalized_words(expected)
    return distance(reference, normalized_words(actual)), len(reference)


def surface_errors(expected: str, actual: str) -> tuple[int, int]:
    """Word errors over whitespace tokens that keep their case and punctuation.

    The benchmark's own count strips both (`normalized_words`), so only this one sees a model that
    forgot a comma or a capital.
    """
    reference = expected.split()
    return distance(reference, actual.split()), len(reference)


def positions_of(text: str, spans: list[tuple[int, int]]) -> list[list[int]]:
    """Word ranges [start, end), in `normalized_words(text)`, touched by the character `spans`."""
    words = word_spans(text)
    positions = []
    for start, end in spans:
        covered = [i for i, (_, a, b) in enumerate(words) if a < end and b > start]
        if covered:
            positions.append([covered[0], covered[-1] + 1])
    return positions
