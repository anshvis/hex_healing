"""Cut a long dictation into training pieces at pauses, and split its reference text to match (D-23).

The pause rules are `src/transcription/pause_segmenter.rs`'s, re-implemented here with the same
thresholds: 20 ms frames against an adaptive noise floor (a frame is speech when it is 12 dB above
the floor and above -55 dBFS; the floor starts at the first frame, falls at once and rises 0.06 dB
per frame), a piece at least 3 s long, ended by a 600 ms pause (300 ms once the piece is 8 s long),
cut in the middle of the pause. Those cuts are the candidates; training keeps the fewest of them
that leave every piece under the cap (the latest candidate that fits, piece after piece), because
each cut is a boundary the text has to be split at. A clip with no candidate that fits is skipped.

A piece's text is found by transcribing the pieces with the base model and aligning the
transcripts to the reference: a boundary is accepted only where, in every minimum-edit alignment
(not one traceback's tie-break), the last word heard before the cut and the first word heard after
it match the same two adjacent reference words exactly. Repeated words ("alpha beta" three times)
make that false. Anything less certain skips the clip rather than guess (a wrong split teaches the
model words it never heard).
"""

from __future__ import annotations

import numpy as np

from healing.wer import normalized_words, word_spans

SAMPLE_RATE = 16_000
FRAME_SAMPLES = SAMPLE_RATE // 50
SPEECH_MARGIN_DB = 12.0
MINIMUM_SPEECH_DB = -55.0
LOWEST_FLOOR_DB = -90.0
FLOOR_RISE_DB_PER_FRAME = 0.06
MINIMUM_PIECE = 3 * SAMPLE_RATE
PAUSE = 600 * SAMPLE_RATE // 1000
LONG_PIECE = 8 * SAMPLE_RATE
LONG_PIECE_PAUSE = 300 * SAMPLE_RATE // 1000


def candidate_cuts(samples: np.ndarray) -> list[int]:
    """Every cut `PauseSegmenter` makes over `samples` (16 kHz, int16 or float in [-1, 1]), as sample
    offsets. A trailing partial frame is never judged, as in the Rust segmenter."""
    audio = samples.astype(np.float32) / 32768.0 if samples.dtype == np.int16 else samples.astype(np.float32)
    frames = len(audio) // FRAME_SAMPLES
    if frames == 0:
        return []
    framed = audio[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES).astype(np.float64)
    energy = ((framed - framed.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    levels = 10.0 * np.log10(np.maximum(energy, 1e-18))
    cuts: list[int] = []
    floor: float | None = None
    position = piece_start = silent = 0
    heard = False
    for level in levels:
        level = float(level) if np.isfinite(level) else 0.0
        floor = floor + FLOOR_RISE_DB_PER_FRAME if floor is not None and level >= floor else max(level, LOWEST_FLOOR_DB)
        position += FRAME_SAMPLES
        if level > max(floor + SPEECH_MARGIN_DB, MINIMUM_SPEECH_DB):
            silent, heard = 0, True
            continue
        silent += 1
        piece = position - piece_start
        pause = LONG_PIECE_PAUSE if piece >= LONG_PIECE else PAUSE
        if not heard or piece < MINIMUM_PIECE or silent * FRAME_SAMPLES < pause:
            continue
        cut = position - silent * FRAME_SAMPLES // 2
        cuts.append(cut)
        piece_start, silent, heard = cut, 0, False
    return cuts


def pieces(samples: np.ndarray, limit: int) -> list[tuple[int, int]] | None:
    """[start, end) pieces of at most `limit` samples, cut only at `candidate_cuts`, as few as
    possible (each piece ends at the latest candidate that fits). None when no cut fits."""
    total = len(samples)
    if total <= limit:
        return [(0, total)]
    cuts = candidate_cuts(samples)
    result, start = [], 0
    while total - start > limit:
        fitting = [c for c in cuts if start < c <= start + limit]
        if not fitting:
            return None
        result.append((start, fitting[-1]))
        start = fitting[-1]
    result.append((start, total))
    return result


class _Alignments:
    """Every minimum-edit alignment of `hypothesis` to `reference` at once: the cheapest cost to and
    from each cell, and how many cheapest paths reach and leave it (Python ints: no overflow)."""

    def __init__(self, reference: list[str], hypothesis: list[str]):
        self.ref, self.hyp = reference, hypothesis
        rows, cols = len(reference), len(hypothesis)
        inf = float("inf")
        self.to = [[inf] * (cols + 1) for _ in range(rows + 1)]
        self.ways_to = [[0] * (cols + 1) for _ in range(rows + 1)]
        self.to[0][0], self.ways_to[0][0] = 0, 1
        for i in range(rows + 1):
            for j in range(cols + 1):
                for pi, pj, step in self._moves(i, j, -1):
                    self._relax(self.to, self.ways_to, i, j, self.to[pi][pj] + step, self.ways_to[pi][pj])
        self.rest = [[inf] * (cols + 1) for _ in range(rows + 1)]
        self.ways_rest = [[0] * (cols + 1) for _ in range(rows + 1)]
        self.rest[rows][cols], self.ways_rest[rows][cols] = 0, 1
        for i in range(rows, -1, -1):
            for j in range(cols, -1, -1):
                for ni, nj, step in self._moves(i, j, 1):
                    self._relax(self.rest, self.ways_rest, i, j, self.rest[ni][nj] + step, self.ways_rest[ni][nj])
        self.best, self.paths = self.to[rows][cols], self.ways_to[rows][cols]

    def _moves(self, i: int, j: int, direction: int):
        """Neighbouring cells one move away (back when `direction` is -1) and that move's cost."""
        for di, dj in ((1, 1), (1, 0), (0, 1)):
            ni, nj = i + direction * di, j + direction * dj
            if 0 <= ni <= len(self.ref) and 0 <= nj <= len(self.hyp):
                if di and dj:
                    r, h = (ni, nj) if direction < 0 else (i, j)
                    yield ni, nj, int(self.ref[r] != self.hyp[h])
                else:
                    yield ni, nj, 1

    @staticmethod
    def _relax(cost, ways, i, j, candidate, count) -> None:
        if candidate < cost[i][j]:
            cost[i][j], ways[i][j] = candidate, count
        elif candidate == cost[i][j]:
            ways[i][j] += count

    def certain_pair(self, last: int) -> int | None:
        """The reference index `i` that every cheapest alignment matches exactly to hypothesis word
        `last`, with `i + 1` matched exactly to `last + 1`; None when any alignment differs."""
        for i in range(len(self.ref) - 1):
            if self.ref[i] != self.hyp[last] or self.ref[i + 1] != self.hyp[last + 1]:
                continue
            if self.to[i][last] + self.rest[i + 2][last + 2] != self.best:
                continue
            if self.ways_to[i][last] * self.ways_rest[i + 2][last + 2] == self.paths:
                return i
        return None


def split_reference(reference: str, transcripts: list[str]) -> list[str] | None:
    """`reference` cut into one text per piece, keeping its punctuation and casing, or None when a
    boundary is not certain. `transcripts` are the pieces' transcripts in order."""
    if len(transcripts) == 1:
        return [reference]
    spans = word_spans(reference)
    heard = [(piece, word) for piece, text in enumerate(transcripts) for word in normalized_words(text)]
    if any(not normalized_words(text) for text in transcripts):
        return None  # a piece with nothing heard has no boundary to place
    alignments = _Alignments([w for w, _, _ in spans], [w for _, w in heard])
    cuts = [0]
    for piece in range(len(transcripts) - 1):
        last = max(j for j, (p, _) in enumerate(heard) if p == piece)
        matched = alignments.certain_pair(last)
        if matched is None:
            return None  # no exact match across the cut, or the alignments disagree where it falls
        cuts.append(spans[matched + 1][1])
    cuts.append(len(reference))
    texts = [reference[a:b].strip() for a, b in zip(cuts, cuts[1:])]
    return texts if all(normalized_words(t) for t in texts) else None
