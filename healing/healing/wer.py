"""Word errors exactly as `hex benchmark-transcription` counts them.

A port of `normalized_words` and `word_errors` in
`src/transcription/benchmark.rs`: split on anything that isn't alphanumeric,
lowercase, then word-level edit distance against the reference.

Exact for the English sets healing uses. Python's `str.isalnum` and Rust's
`char::is_alphanumeric` disagree on some combining marks (Indic, Thai), so
check a non-English set against the benchmark before trusting this port.
"""

from __future__ import annotations

from collections.abc import Sequence


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


def _table(reference: Sequence, hypothesis: Sequence) -> list[list[int]]:
    """Edit distances of every prefix of `reference` against every prefix of `hypothesis`."""
    rows, cols = len(reference), len(hypothesis)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        table[i][0] = i
    for j in range(cols + 1):
        table[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i][j] = min(table[i - 1][j - 1] + cost, table[i - 1][j] + 1, table[i][j - 1] + 1)
    return table


def align(reference: Sequence, hypothesis: Sequence) -> tuple[int, list[bool]]:
    """(edit distance, per reference item whether SOME minimum-edit alignment pairs it with an equal item).

    Several alignments can be equally good when a word repeats ("a b a" against "a"). A reference
    position counts as matched when any of them matches it, so which of two tied words an
    alignment happens to keep never decides the score: a position is wrong only if no optimal
    alignment can match it. The one edit-distance table: word errors, surface errors, matched
    positions and the phonetic scores' character distance all read it.
    """
    rows, cols = len(reference), len(hypothesis)
    forward = _table(reference, hypothesis)
    total = forward[rows][cols]
    backward = _table(reference[::-1], hypothesis[::-1])  # backward[a][b]: the last a against the last b
    matched = [
        any(
            reference[i] == hypothesis[j] and forward[i][j] + backward[rows - i - 1][cols - j - 1] == total
            for j in range(cols)
        )
        for i in range(rows)
    ]
    return total, matched


def word_errors(expected: str, actual: str) -> tuple[int, int]:
    """(edit distance in words, reference words)."""
    reference = normalized_words(expected)
    return align(reference, normalized_words(actual))[0], len(reference)


def surface_errors(expected: str, actual: str) -> tuple[int, int]:
    """Word errors over whitespace tokens that keep their case and punctuation.

    The benchmark's own count strips both (`normalized_words`), so only this one sees a model that
    forgot a comma or a capital.
    """
    reference = expected.split()
    return align(reference, actual.split())[0], len(reference)


def matched_positions(reference: list[str], hypothesis: list[str]) -> list[bool]:
    """For each reference word, whether a minimum-edit alignment pairs it with the same word."""
    return align(reference, hypothesis)[1]


def positions_of(text: str, spans: list[tuple[int, int]]) -> list[list[int]]:
    """Word ranges [start, end), in `normalized_words(text)`, touched by the character `spans`."""
    words = word_spans(text)
    positions = []
    for start, end in spans:
        covered = [i for i, (_, a, b) in enumerate(words) if a < end and b > start]
        if covered:
            positions.append([covered[0], covered[-1] + 1])
    return positions
