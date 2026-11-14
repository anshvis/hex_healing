"""Word errors exactly as `hex benchmark-transcription` counts them.

A port of `normalized_words` and `word_errors` in
`src/transcription/benchmark.rs`: split on anything that isn't alphanumeric,
lowercase, then word-level edit distance against the reference.
"""

from __future__ import annotations


def normalized_words(text: str) -> list[str]:
    words, current = [], []
    for character in text:
        if character.isalnum():
            current.append(character)
        elif current:
            words.append("".join(current).lower())
            current = []
    if current:
        words.append("".join(current).lower())
    return words


def word_errors(expected: str, actual: str) -> tuple[int, int]:
    """(edit distance in words, reference words)."""
    reference, hypothesis = normalized_words(expected), normalized_words(actual)
    previous = list(range(len(hypothesis) + 1))
    for expected_index, expected_word in enumerate(reference, 1):
        current = [expected_index]
        for actual_index, actual_word in enumerate(hypothesis, 1):
            if expected_word == actual_word:
                current.append(previous[actual_index - 1])
            else:
                current.append(
                    1
                    + min(previous[actual_index - 1], previous[actual_index], current[actual_index - 1])
                )
        previous = current
    return previous[len(hypothesis)], len(reference)
