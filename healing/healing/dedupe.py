"""Near-duplicate clustering by raw text, and acoustic features for spread.

Text: normalize, token 3-gram MinHash (128 permutations) with LSH to find candidates, then the
exact Jaccard (>= 0.8) and a duration ratio within +-30% decide (D-6). Texts under 3 tokens have
no 3-grams and cluster by whole-text equality. Assignment is deterministic given the clip order
(`timestamp_ms`, then `clip_id`): a clip joins the best-matching existing representative (highest
Jaccard, then earliest representative), else it becomes one. A cluster id is its first clip's id.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

JACCARD = 0.8
DURATION_BAND = 0.3
NUM_PERM = 128
BANDS = 32  # 32 bands x 4 rows: a 0.8-Jaccard pair shares a band with probability ~ 1
ROWS = NUM_PERM // BANDS
_PRIME = (1 << 31) - 1
_rng = np.random.RandomState(13)
_A = _rng.randint(1, _PRIME, NUM_PERM, dtype=np.int64).astype(np.uint64)
_B = _rng.randint(0, _PRIME, NUM_PERM, dtype=np.int64).astype(np.uint64)


@dataclass(frozen=True)
class Clip:
    clip_id: str
    timestamp_ms: int
    text: str
    audio_ms: int | None = None


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def shingles(normalized: str) -> frozenset[str] | None:
    """Token 3-grams, or None when the text is too short to have any."""
    tokens = normalized.split()
    if len(tokens) < 3:
        return None
    return frozenset(" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2))


def signature(grams: frozenset[str]) -> np.ndarray:
    hashes = np.array(
        [
            int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") % _PRIME
            for g in sorted(grams)
        ],
        dtype=np.uint64,
    )
    return ((_A[:, None] * hashes[None, :] + _B[:, None]) % _PRIME).min(axis=1)


def _bands(sig: np.ndarray) -> list[bytes]:
    return [sig[b * ROWS : (b + 1) * ROWS].tobytes() for b in range(BANDS)]


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


def durations_compatible(a: int | None, b: int | None) -> bool:
    if not a or not b:
        return True  # unknown duration cannot rule a match out
    return abs(a - b) <= DURATION_BAND * max(a, b)


def compatible(a: str, b: str) -> bool:
    """Whether two normalized texts would cluster together (text only, no duration)."""
    if not a or not b:
        return False
    ga, gb = shingles(a), shingles(b)
    if ga is None or gb is None:
        return a == b
    return jaccard(ga, gb) >= JACCARD


class LshIndex:
    """MinHash bands over token 3-grams: `query` finds earlier items that may be near-duplicates."""

    def __init__(self) -> None:
        self.buckets: dict[tuple[int, bytes], list[int]] = {}

    def add(self, index: int, grams: frozenset[str]) -> None:
        for band, value in enumerate(_bands(signature(grams))):
            self.buckets.setdefault((band, value), []).append(index)

    def query(self, grams: frozenset[str]) -> set[int]:
        return {
            i for band, value in enumerate(_bands(signature(grams))) for i in self.buckets.get((band, value), [])
        }


def near_duplicate_pairs(keys: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Pairs of distinct normalized texts whose 3-gram Jaccard is at least `JACCARD`.

    Texts under 3 tokens have no 3-grams and pair with nothing here (callers match them by equality).
    """
    unique = sorted(set(keys))
    index, grams = LshIndex(), {}
    for position, key in enumerate(unique):
        found = shingles(key)
        if found is None:
            continue
        for earlier in sorted(index.query(found)):
            if jaccard(grams[earlier], found) >= JACCARD:
                yield unique[earlier], key
        grams[position] = found
        index.add(position, found)


def near_duplicate_cross(left: Iterable[str], right: Iterable[str]) -> Iterator[tuple[str, str]]:
    """(left, right) pairs of normalized texts that are equal or whose 3-gram Jaccard is at least
    `JACCARD`. A direct test between two sets, with no clustering and so nothing transitive: an
    overlap exists only when two texts really are near-duplicates. Texts under 3 tokens match by
    equality only."""
    unique_right = sorted(set(right))
    index, grams = LshIndex(), {}
    for position, key in enumerate(unique_right):
        found = shingles(key)
        if found is not None:
            grams[position] = found
            index.add(position, found)
    positions = {key: i for i, key in enumerate(unique_right)}
    for key in sorted(set(left)):
        if key in positions:
            yield key, key
        found = shingles(key)
        if found is None:
            continue
        for other in sorted(index.query(found)):
            if unique_right[other] != key and jaccard(grams[other], found) >= JACCARD:
                yield key, unique_right[other]


@dataclass
class _Rep:
    cluster_id: str
    key: str
    grams: frozenset[str] | None
    audio_ms: int | None


class Clusterer:
    def __init__(self, reps: list[Clip] = ()):
        self.reps: list[_Rep] = []
        self.index = LshIndex()
        self.short: dict[str, list[int]] = {}
        for rep in reps:
            self._add(rep.clip_id, rep.text, rep.audio_ms)

    def _add(self, cluster_id: str, text: str, audio_ms: int | None) -> None:
        key = normalize(text)
        grams = shingles(key)
        index = len(self.reps)
        self.reps.append(_Rep(cluster_id, key, grams, audio_ms))
        if grams is None:
            self.short.setdefault(key, []).append(index)
            return
        self.index.add(index, grams)

    def assign(self, clip: Clip) -> str | None:
        key = normalize(clip.text)
        if not key:
            return None
        grams = shingles(key)
        best: tuple[float, int] | None = None
        if grams is None:
            candidates = {i: 1.0 for i in self.short.get(key, [])}
        else:
            candidates = {}
            for i in self.index.query(grams):
                score = jaccard(grams, self.reps[i].grams)
                if score >= JACCARD:
                    candidates[i] = score
        for index, score in candidates.items():
            if not durations_compatible(clip.audio_ms, self.reps[index].audio_ms):
                continue
            if best is None or (score, -index) > (best[0], -best[1]):
                best = (score, index)
        if best is not None:
            return self.reps[best[1]].cluster_id
        self._add(clip.clip_id, clip.text, clip.audio_ms)
        return clip.clip_id


def cluster(clips: list[Clip], reps: list[Clip] = ()) -> dict[str, str]:
    """clip_id -> cluster_id for every clip with text; `reps` are earlier clusters' first clips."""
    clusterer = Clusterer(list(reps))
    assignments: dict[str, str] = {}
    for clip in sorted(clips, key=lambda c: (c.timestamp_ms, c.clip_id)):
        found = clusterer.assign(clip)
        if found is not None:
            assignments[clip.clip_id] = found
    return assignments


# --- acoustic features ---------------------------------------------------------------------

SAMPLE_RATE = 16_000
N_MFCC = 13
_N_MELS = 26
_N_FFT = 512
_WIN = 400
_HOP = 160


def _mel_filterbank() -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    def inverse(m):
        return 700 * (10 ** (m / 2595) - 1)

    points = inverse(np.linspace(mel(0), mel(SAMPLE_RATE / 2), _N_MELS + 2))
    bins = np.floor((_N_FFT + 1) * points / SAMPLE_RATE).astype(int)
    bank = np.zeros((_N_MELS, _N_FFT // 2 + 1))
    for m in range(1, _N_MELS + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, centre):
            bank[m - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            bank[m - 1, k] = (right - k) / max(right - centre, 1)
    return bank


_BANK = _mel_filterbank()
_DCT = np.cos(
    np.pi / _N_MELS * (np.arange(_N_MELS)[None, :] + 0.5) * np.arange(N_MFCC)[:, None]
)


def mfcc_vector(samples: np.ndarray) -> np.ndarray:
    """13 MFCC means then 13 variances over the clip (mono 16 kHz int16 or float)."""
    x = samples.astype(np.float64)
    if samples.dtype == np.int16:
        x = x / 32768.0
    x = np.append(x[0] if x.size else 0.0, x[1:] - 0.97 * x[:-1]) if x.size else x
    if x.size < _WIN:
        x = np.pad(x, (0, _WIN - x.size))
    frames = 1 + (x.size - _WIN) // _HOP
    index = np.arange(_WIN)[None, :] + _HOP * np.arange(frames)[:, None]
    windowed = x[index] * np.hanning(_WIN)
    power = np.abs(np.fft.rfft(windowed, _N_FFT)) ** 2
    coefficients = np.log(power @ _BANK.T + 1e-10) @ _DCT.T
    return np.concatenate([coefficients.mean(axis=0), coefficients.var(axis=0)])


def standardize(vectors: np.ndarray) -> np.ndarray:
    """Per-dimension z-scores across `vectors` so no coefficient dominates the distance."""
    spread = vectors.std(axis=0)
    spread[spread < 1e-9] = 1.0
    return (vectors - vectors.mean(axis=0)) / spread


def distances(vectors: np.ndarray) -> np.ndarray:
    """Pairwise distances, scaled by sqrt(dim) so a typical unit is about 1."""
    z = standardize(vectors)
    squares = (z**2).sum(axis=1)
    squared = squares[:, None] + squares[None, :] - 2 * z @ z.T
    return np.sqrt(np.maximum(squared, 0)) / np.sqrt(z.shape[1])


def spread_pick(dist: np.ndarray, order: list[int], k: int, fixed: list[int] = ()) -> list[int]:
    """Greedy max-min distance. `order` ranks candidates, newest first, and breaks ties."""
    chosen = list(fixed)
    pool = list(order)
    while pool and len(chosen) - len(fixed) < k:
        if not chosen:
            pick = pool[0]
        else:
            pick = max(pool, key=lambda i: (min(dist[i][j] for j in chosen), -pool.index(i)))
        chosen.append(pick)
        pool.remove(pick)
    return chosen[len(fixed) :]
