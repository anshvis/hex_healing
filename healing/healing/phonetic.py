"""Does a corrected word sound like the word Hex heard? (D-9)

`acoustic` when the Double Metaphone codes share a code, or when both the character edit
distance (<= CHAR_MAX) and the Metaphone-code edit distance (<= PHONE_MAX) are small.
Anything else is a change of mind: `rewrite`. Spans compare as joined strings, so
"a pull" against "apple" counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from metaphone import doublemetaphone

from healing import wer

CHAR_MAX = 0.6
PHONE_MAX = 0.5


@dataclass(frozen=True)
class Scores:
    metaphone: int  # 1 when the two share a Double Metaphone code
    char: float  # normalized character edit distance
    phone: float  # normalized edit distance between the primary codes


def levenshtein(a: str, b: str) -> int:
    return wer.align(a, b)[0]


def _normalized(a: str, b: str) -> float:
    longest = max(len(a), len(b))
    return levenshtein(a, b) / longest if longest else 0.0


def _joined(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _codes(text: str) -> tuple[str, set[str]]:
    primary, alternate = doublemetaphone(text)
    return primary, {code for code in (primary, alternate) if code}


def score(old: str, new: str) -> Scores:
    a, b = _joined(old), _joined(new)
    primary_a, codes_a = _codes(a)
    primary_b, codes_b = _codes(b)
    return Scores(
        metaphone=1 if codes_a & codes_b else 0,
        char=_normalized(a, b),
        phone=_normalized(primary_a, primary_b),
    )


def classify(old: str, new: str) -> tuple[str, Scores]:
    scores = score(old, new)
    if not _joined(old) or not _joined(new):
        return "rewrite", scores
    if scores.metaphone or (scores.char <= CHAR_MAX and scores.phone <= PHONE_MAX):
        return "acoustic", scores
    return "rewrite", scores
