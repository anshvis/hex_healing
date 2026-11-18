"""`healing train prepare`: the training bundle a LoRA round trains on (T6.1).

`$HEALING_HOME/train/<round>/` holds `manifest.jsonl` (NeMo's `{audio_filepath, duration, text,
weight}`, plus `id`, `source` and `cluster`), `audio/*.wav` and `bundle.sha256`. The round is the
frozen `train` row the clips came from: `splits.record_train` is the only way training gets clips
(D-41), and the manifest's texts and weights are the ones that row froze (D-44), not the clips' now.

The data:
- weights from the labels: strong and verified 1.0, weak 0.2 and only with `include_weak`;
- each storage cluster (T1.3's `cluster_id`) capped at an effective weight of 3: the sum of its
  entries' weights, so 50 copies of one phrase count as 3 (D-45);
- clips capped at 30 s (D-23): a longer one is cut at pauses (`pauses.py`) or skipped;
- every clip padded as Hex padded it before the model heard it (`pad_gguf_samples`): zeros up to
  1.5 s, plus 400 ms for the catalog Unified model. Pause cutting inside a clip is not reproduced;
- `replay` (20%) of the sampler's weight is self-distilled replay (`replay.py`, D-25);
- no entry shares a split-cluster with a test set: a replay clip that would is dropped first, and
  anything left is a `TestClusterLeak`, which fails the run (never dropped silently).

`weight` is for a `WeightedRandomSampler` (NeMo's ASR datasets have no per-sample loss weight);
`draw` is the same draw in numpy, which the tests use to check frequencies. `audio_filepath` is
relative to the bundle, so the bundle hashes the same wherever it is copied (T6.5 resolves it).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile

from healing import manifest as bench_manifest, splits
from healing.paths import train_dir
from healing.registry.objects import fsync_dir, owner_dir
from healing.store import Store
from healing.train import pauses, replay as replay_module
from healing.train.replay import Audio, Transcriber

SAMPLE_RATE = 16_000
CAP_SECONDS = 30  # D-23
CLUSTER_CAP = 3.0
DEFAULT_REPLAY = 0.2
DEFAULT_SEED = 13
MINIMUM_SAMPLES = 24_000  # `MINIMUM_PARAKEET_DURATION`, 1.5 s (src/dictation.rs)
UNIFIED_TAIL = 6_400  # `UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES` (src/transcription.rs)
MANIFEST = "manifest.jsonl"
LISTING = "bundle.sha256"


class TrainError(RuntimeError):
    pass


class TestClusterLeak(AssertionError):
    """A manifest entry shares a split-cluster with a test set. The run fails; nothing is dropped."""

    __test__ = False  # not a pytest class


@dataclass(frozen=True)
class Base:
    """The v0 model: replay transcripts and long-clip boundaries come from it."""

    sha256: str
    gguf: Path
    catalog_id: str | None
    filename: str | None = None  # the catalog file name the benchmark recognizes it by


def catalog_filename(record: dict, catalog_id: str | None) -> str | None:
    """The catalog file name the benchmark recognizes v0 by (it matches catalog artifacts by name and
    size): `<id>-Q8_0.gguf` from the catalog id, never the basename of a file passed to `register-base
    --gguf`, which can be anything. Only with no catalog id is the recorded source name used."""
    if catalog_id:
        return f"{catalog_id}-Q8_0.gguf"
    source = record.get("source_path")
    return Path(source).name if source else None


def resolve_base() -> Base:
    from healing.registry import objects
    from healing.registry import store as registry

    v0 = registry.load().versions.get("v0")
    if v0 is None:
        raise TrainError("no v0 in the registry: run `healing models register-base` first")
    sha = v0.artifact("gguf")
    if not objects.has(sha):
        raise TrainError(f"v0's GGUF {str(sha)[:12]} is not in the registry's objects")
    catalog_id = (v0.record.get("base") or {}).get("catalog_id")
    return Base(sha, objects.path(sha), catalog_id, catalog_filename(v0.record, catalog_id))


def tail(model_id: str | None) -> int:
    """Silence Hex appends after a clip of `model_id` (`trailing_silence_samples`)."""
    return UNIFIED_TAIL if model_id == replay_module.UNIFIED_CATALOG_ID else 0


def pad(samples: np.ndarray, model_id: str | None) -> np.ndarray:
    """`pad_gguf_samples`: zeros up to 1.5 s, then the model's tail."""
    length = max(len(samples), MINIMUM_SAMPLES) + tail(model_id)
    return np.concatenate([samples.astype(np.int16), np.zeros(length - len(samples), dtype=np.int16)])


def limit(model_id: str | None) -> int:
    """The most audio a clip may hold so that, padded, it stays within the cap."""
    return CAP_SECONDS * SAMPLE_RATE - tail(model_id)


@dataclass
class Entry:
    id: str
    source: str  # "user" or "replay"
    text: str
    weight: float
    samples: np.ndarray  # padded int16
    cluster: str
    texts: list[str]  # the clip's texts; the leakage checks add the entry's own `text`
    piece_of: str | None = None  # the clip a pause-cut piece came from

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


@dataclass
class Prepared:
    path: Path
    sha256: str
    train_split: str
    clips: int
    entries: int
    replay: int
    replay_share: float
    minutes: float
    skipped: Counter = field(default_factory=Counter)

    def text(self) -> str:
        lines = [
            f"bundle {self.path}",
            f"sha256 {self.sha256}",
            f"train split {self.train_split}: {self.clips} clips as {self.entries} entries ({self.minutes:.1f} min), "
            f"{self.replay} replay clips ({self.replay_share:.0%} of the sampler's weight)",
        ]
        lines += [f"skipped {n}: {reason}" for reason, n in sorted(self.skipped.items())]
        return "\n".join(lines)


# --- test sets -----------------------------------------------------------------------------


def held_out_texts(db) -> dict[str, list[str]]:
    """`held:<split>:<clip>` and `general:<id>` -> texts, for every registered test set of every parent:
    each member's live texts and the reference its pinned manifest scores (checked against its pin)."""
    texts = {
        f"held:{r['split_id']}:{r['clip_id']}": splits._texts(r)
        for r in db.execute(
            "SELECT s.split_id, m.clip_id, c.reference_text, c.raw_text FROM splits s"
            " JOIN split_members m ON m.split_id = s.split_id JOIN clips c ON c.clip_id = m.clip_id"
            " WHERE s.kind IN ('rolling', 'edited', 'golden')"
        )
    }
    for row in db.execute("SELECT * FROM splits WHERE kind IN ('rolling', 'edited', 'golden') AND manifest_path IS NOT NULL"):
        try:  # what the set scores: its pinned references, which a later correction does not change
            folder, _ = splits.verified_set(row)
        except bench_manifest.ManifestError as error:
            raise TrainError(f"{row['split_id']}: its pinned manifest can't be read to hold it out: {error}") from error
        for clip in json.loads((folder / "manifest.json").read_text())["clips"]:
            texts.setdefault(f"held:{row['split_id']}:{clip['id']}", []).append(clip["expected"])
    texts.update({key: [text] for key, text in splits.general_references(db).items()})
    return texts


def _touching(held: dict[str, list[str]], candidates: dict[str, list[str]]) -> set[str]:
    """The candidate keys in a split-cluster with any held-out text."""
    clusters = splits.split_clusters({**held, **candidates})
    roots = {clusters[key] for key in held}
    return {key for key in candidates if clusters[key] in roots}


def assert_no_test_cluster(db, entries: list[Entry]) -> None:
    candidates = {f"{e.source}:{e.id}": [*e.texts, e.text] for e in entries}  # what it is labelled with, always
    leaks = sorted(_touching(held_out_texts(db), candidates))
    if leaks:
        raise TestClusterLeak(f"{len(leaks)} training entr{'y shares' if len(leaks) == 1 else 'ies share'} "
                              f"a split-cluster with a test set: {', '.join(leaks[:5])}")


# --- weights -------------------------------------------------------------------------------


def cap_clusters(entries: list[Entry], cap: float = CLUSTER_CAP) -> None:
    """Scale each cluster whose entries' weights sum past `cap` down to exactly `cap`."""
    totals: dict[str, float] = {}
    for e in entries:
        totals[e.cluster] = totals.get(e.cluster, 0.0) + e.weight
    for e in entries:
        if totals[e.cluster] > cap:
            e.weight *= cap / totals[e.cluster]


def draw(weights, n: int, seed: int = 0) -> np.ndarray:
    """`n` indices drawn with replacement in proportion to `weights`: `WeightedRandomSampler`'s draw."""
    p = np.asarray(weights, dtype=np.float64)
    return np.random.default_rng(seed).choice(len(p), size=n, replace=True, p=p / p.sum())


# --- prepare -------------------------------------------------------------------------------


def _stable(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def _user_entries(store: Store, split_id: str, transcriber, skipped: Counter) -> list[Entry]:
    """The train row's clips as entries (pieces for long ones).
    `transcriber()` gives the base model's transcriber, asked for only when a clip is over the cap."""
    detail = json.loads(store.db.execute("SELECT detail FROM splits WHERE split_id = ?", (split_id,)).fetchone()["detail"])
    members, weights = detail["members"], detail.get("weights") or {}
    entries: list[Entry] = []
    long: list[tuple[str, list[tuple[int, int]], np.ndarray, dict]] = []
    for clip_id in sorted(members):
        row = store.db.execute("SELECT audio_sha256, model_id, cluster_id, weight FROM clips WHERE clip_id = ?", (clip_id,)).fetchone()
        samples, rate = soundfile.read(store.audio_path(row["audio_sha256"]), dtype="int16", always_2d=False)
        if rate != SAMPLE_RATE:
            raise TrainError(f"clip {clip_id}: {rate} Hz audio; training is 16 kHz only")
        texts = members[clip_id]
        meta = {"text": texts[0], "texts": texts, "weight": float(weights.get(clip_id, row["weight"])),
                "cluster": row["cluster_id"] or clip_id, "model": row["model_id"]}
        cuts = pauses.pieces(samples, limit(row["model_id"]))
        if cuts is None:
            skipped["over 30 s with no pause to cut at"] += 1
        elif len(cuts) == 1:
            entries.append(Entry(clip_id, "user", meta["text"], meta["weight"], pad(samples, meta["model"]), meta["cluster"], texts))
        else:
            long.append((clip_id, cuts, samples, meta))
    if long:
        heard = transcriber()([Audio(f"{c}@{a}-{b}", s[a:b]) for c, cuts, s, _ in long for a, b in cuts])
        for clip_id, cuts, samples, meta in long:
            parts = pauses.split_reference(meta["text"], [heard.get(f"{clip_id}@{a}-{b}", "") for a, b in cuts])
            if parts is None:
                skipped["over 30 s and its text could not be split at the pauses"] += 1
                continue
            for k, ((a, b), text) in enumerate(zip(cuts, parts)):
                entries.append(Entry(f"{clip_id}-p{k}", "user", text, meta["weight"], pad(samples[a:b], meta["model"]),
                                     meta["cluster"], meta["texts"], clip_id))
    return entries


def _without_pieces_near_tests(db, user: list[Entry], skipped: Counter) -> list[Entry]:
    """Drop every piece of a cut clip when one piece's own text shares a split-cluster with a test
    reference (counted in `skipped`). The zones judged the clip by its whole text, which a short piece
    can match a test reference without; a whole clip near a test set is still a leak the assert fails."""
    pieces = {f"user:{e.id}": [*e.texts, e.text] for e in user if e.piece_of}
    if not pieces:
        return user
    near = _touching(held_out_texts(db), pieces)
    dropped = {e.piece_of for e in user if e.piece_of and f"user:{e.id}" in near}
    if dropped:
        skipped["over 30 s and a piece is near a test set"] += len(dropped)
    return [e for e in user if e.piece_of is None or e.piece_of not in dropped]


def _replay_entries(db, user: list[Entry], pool, heard: dict[str, str], ratio: float, base: Base,
                    split_id: str, seed: int, skipped: Counter) -> list[Entry]:
    if ratio <= 0:
        return []
    candidates = {f"replay:{u.id}": [heard[u.id]] for u in pool if replay_module.usable(heard[u.id])}
    near = _touching(held_out_texts(db), {**{f"user:{e.id}": e.texts for e in user}, **candidates})
    eligible = sorted(key for key in candidates if key not in near)
    if len(candidates) > len(eligible):
        skipped["replay clips near a test set"] += len(candidates) - len(eligible)
    target = sum(e.weight for e in user) * ratio / (1 - ratio)
    if not eligible:
        raise TrainError("no replay clip is usable (empty transcripts or all near a test set); pass --replay 0 to train without")
    count = min(len(eligible), max(1, round(target)))
    each = target / count  # so replay's share of the sampler's weight is exactly `ratio`
    picks = sorted(np.random.default_rng([seed, _stable(split_id)]).choice(len(eligible), size=count, replace=False))
    by_id = {u.id: u for u in pool}
    chosen = []
    for index in picks:
        clip = by_id[eligible[index].removeprefix("replay:")]
        chosen.append(Entry(clip.id, "replay", heard[clip.id], each, pad(clip.load(), base.catalog_id), "replay", [heard[clip.id]]))
    return chosen


class _Lazy:
    """The base model and its transcriber, resolved only when replay or a long clip needs them."""

    def __init__(self, base: Base | None, transcriber: Transcriber | None):
        self.known, self._transcriber = base, transcriber

    def base(self) -> Base:
        if self.known is None:
            self.known = resolve_base()
        return self.known

    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            base = self.base()
            self._transcriber = replay_module.benchmark_transcriber(base.gguf, base.filename)
        return self._transcriber


def prepare(
    store: Store, *, include_weak: bool = False, ratio: float = DEFAULT_REPLAY, parent: str = "current",
    seed: int = DEFAULT_SEED, source=None, base: Base | None = None, transcriber: Transcriber | None = None,
    cutoff_ms: int | None = None, now: int | None = None,
) -> Prepared:
    """Snapshot the train clips, build the bundle and return where it is and its sha256."""
    if not 0 <= ratio < 1:
        raise TrainError(f"--replay {ratio} must be at least 0 and under 1")
    skipped: Counter = Counter()
    lazy = _Lazy(base, transcriber)
    pool, heard = [], {}
    if ratio > 0:
        pool = replay_module.pool(source or replay_module.LocalLibriSpeech(replay_module.default_source()))
        heard = replay_module.transcripts(pool, lazy.base().sha256, lazy.transcriber())
    split_id, clips = splits.record_train(store, parent=parent, include_weak=include_weak,
                                          note="healing train prepare", cutoff_ms=cutoff_ms, now=now)
    if not clips:
        raise TrainError("no clip may be trained on yet (strong or verified, outside the 14-day test window)")
    user = _user_entries(store, split_id, lazy.transcriber, skipped)
    if not user:
        raise TrainError(f"{split_id}: every clip was skipped ({dict(skipped)})")
    user = _without_pieces_near_tests(store.db, user, skipped)
    if not user:
        raise TrainError(f"{split_id}: every clip was skipped ({dict(skipped)})")
    cap_clusters(user)
    replayed = _replay_entries(store.db, user, pool, heard, ratio, lazy.known, split_id, seed, skipped)
    entries = user + replayed
    assert_no_test_cluster(store.db, entries)
    total = sum(e.weight for e in entries)
    path, sha = write_bundle(split_id, entries, {
        "train_split": split_id, "parent": parent, "include_weak": include_weak, "replay_ratio": ratio,
        "base_sha256": lazy.known.sha256 if lazy.known else None, "seed": seed, "skipped": dict(skipped),
    })
    kept = len({e.piece_of or e.id for e in user})
    return Prepared(path, sha, split_id, kept, len(user), len(replayed),
                    sum(e.weight for e in replayed) / total, sum(e.duration for e in user) / 60, skipped)


# --- the bundle ----------------------------------------------------------------------------


def _write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _listing(folder: Path, names: list[str]) -> bytes:
    lines = []
    for name in sorted(names):
        lines.append(f"{hashlib.sha256((folder / name).read_bytes()).hexdigest()}  {name}\n")
    return "".join(lines).encode()


def write_bundle(round_id: str, entries: list[Entry], info: dict) -> tuple[Path, str]:
    """Write the bundle in a scratch folder and rename it into place whole. Its sha256 is over
    `bundle.sha256`, the listing of every file's hash. `bundle.json` holds counts, never text."""
    root = train_dir()
    owner_dir(root)
    target = root / round_id
    if target.exists():
        raise TrainError(f"{target} already exists")
    scratch = Path(tempfile.mkdtemp(prefix=f".{round_id}.", dir=root))
    try:
        (scratch / "audio").mkdir(mode=0o700)
        names, lines = [MANIFEST], []
        for e in sorted(entries, key=lambda e: (e.source != "user", e.id)):
            name = f"audio/{e.id}.wav"
            buffer = _wav(e.samples)
            _write(scratch / name, buffer)
            names.append(name)
            lines.append(json.dumps({"audio_filepath": name, "duration": e.duration, "text": e.text, "weight": e.weight,
                                     "id": e.id, "source": e.source, "cluster": e.cluster}, ensure_ascii=False) + "\n")
        _write(scratch / MANIFEST, "".join(lines).encode())
        listing = _listing(scratch, names)
        _write(scratch / LISTING, listing)
        sha = hashlib.sha256(listing).hexdigest()
        _write(scratch / "bundle.json", (json.dumps({**info, "sha256": sha, "entries": len(entries)}, sort_keys=True) + "\n").encode())
        fsync_dir(scratch / "audio")
        fsync_dir(scratch)
        os.chmod(scratch, 0o700)
        os.rename(scratch, target)
        fsync_dir(root)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return target, sha


def _wav(samples: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    soundfile.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def verify_bundle(path: Path) -> str:
    """The bundle's sha256, once every file it lists still hashes as listed and it holds no other audio."""
    listing_path = path / LISTING
    if not listing_path.is_file():
        raise TrainError(f"{path} is not a training bundle (no {LISTING})")
    listing = listing_path.read_bytes()
    names = [line.split("  ", 1)[1] for line in listing.decode().splitlines()]
    present = {MANIFEST} | {f"audio/{p.name}" for p in (path / "audio").glob("*")}
    if set(names) != present:
        raise TrainError(f"{path}: its files differ from {LISTING}")
    try:
        if _listing(path, names) != listing:
            raise TrainError(f"{path}: a file changed since the bundle was written")
    except FileNotFoundError as error:
        raise TrainError(f"{path}: {error.filename} is missing") from error
    return hashlib.sha256(listing).hexdigest()


def bundle_for(train_split: str) -> Path:
    return train_dir() / train_split


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / MANIFEST).read_text().splitlines()]
