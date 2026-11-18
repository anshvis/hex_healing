"""Self-distilled replay (D-25): LibriSpeech train-clean-100 clips labelled with the base model's own
transcripts, punctuation and casing included.

The pool is 2,000 utterances picked once from the source by a fixed seed. Their transcripts come from
the base (v0) model, never the current one, so drift does not compound; they are made once with
`hex benchmark-transcription --model <v0 GGUF>` and cached by the base GGUF's SHA-256
(`$HEALING_HOME/train/replay/<sha>.jsonl`), so later rounds transcribe nothing.

Offline (G-3): the source is a folder of LibriSpeech parquet shards already on disk (Hugging Face's
`openslr/librispeech_asr` layout: `id`, `text`, `audio`). Nothing here downloads them; fetching
belongs beside the general set in `healing/sets/`, where the network rule allows it.

A pool that holds a clip of a test set is refused (`ReplayError`), and a pool clip whose transcript
near-duplicates a test reference is never mixed in (`manifest.py` drops it, then asserts).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from healing import manifest as bench_manifest
from healing.paths import healing_home, train_dir
from healing.registry.objects import fsync_dir, owner_dir
from healing.sets import general
from healing.wer import normalized_words

POOL_SIZE = 2000
POOL_SEED = 13
TRANSCRIBE_BATCH = 200  # clips per `hex benchmark-transcription` run
SAMPLE_RATE = 16_000
UNIFIED_CATALOG_ID = "parakeet-unified-en-0.6b"


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class Audio:
    """One clip to transcribe: an id and its 16 kHz mono int16 samples (unpadded: the benchmark pads
    as Hex does)."""

    id: str
    samples: np.ndarray


# The seam to the base model: clips in, {id: transcript} out. `benchmark_transcriber` is the real one;
# tests pass a fake.
Transcriber = Callable[[list[Audio]], dict[str, str]]


@dataclass(frozen=True)
class Utterance:
    id: str
    load: Callable[[], np.ndarray]  # int16 samples, 16 kHz mono


class LocalLibriSpeech:
    """LibriSpeech parquet shards already on disk; ids are `ls-<id>`, as the general set names its own."""

    name = "librispeech-train-clean-100"

    def __init__(self, folder: Path):
        self.folder = folder

    def utterances(self) -> Iterable[Utterance]:
        import pyarrow.parquet as parquet

        shards = sorted(self.folder.glob("**/*.parquet")) if self.folder.is_dir() else []
        if not shards:
            raise ReplayError(
                f"no LibriSpeech train-clean-100 parquet shards in {self.folder} (Hugging Face's "
                "openslr/librispeech_asr clean/train.100); put them there, pass --replay-source, or --replay 0"
            )
        for path in shards:
            for index, row in enumerate(parquet.read_table(path, columns=["id"]).to_pylist()):
                yield Utterance(f"ls-{row['id']}", lambda path=path, index=index: _load(path, index))


def _load(path: Path, index: int) -> np.ndarray:
    samples, rate = general._parquet_audio(path, index)
    if rate != SAMPLE_RATE:
        raise ReplayError(f"{path} row {index}: {rate} Hz audio; replay is 16 kHz only")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.clip(np.round(samples * 32767.0), -32768, 32767).astype(np.int16)


def default_source() -> Path:
    return healing_home() / "cache" / "librispeech-train-clean-100"


def general_ids() -> set[str]:
    """The general set's clip ids (the only test set made from public data)."""
    path = general.folder() / "manifest.json"
    return {c["id"] for c in json.loads(path.read_text())["clips"]} if path.is_file() else set()


def pool(source, size: int = POOL_SIZE, seed: int = POOL_SEED) -> list[Utterance]:
    """`size` utterances of `source`, picked by `seed` from its sorted ids: the same pool every round.
    Refused if it holds a test clip."""
    candidates = sorted(source.utterances(), key=lambda u: u.id)
    if not candidates:
        raise ReplayError(f"{source.name} holds no utterances")
    overlap = {u.id for u in candidates} & general_ids()
    if overlap:
        raise ReplayError(f"{source.name} holds {len(overlap)} clip(s) of the general test set (e.g. {min(overlap)}); "
                          "replay must come from a training split")
    if len(candidates) > size:
        picks = sorted(np.random.default_rng(seed).choice(len(candidates), size=size, replace=False))
        candidates = [candidates[i] for i in picks]
    return candidates


def cache_path(base_sha: str) -> Path:
    return train_dir() / "replay" / f"{base_sha}.jsonl"


def read_cache(base_sha: str) -> dict[str, str]:
    path = cache_path(base_sha)
    if not path.is_file():
        return {}
    return {row["id"]: row["transcript"] for row in map(json.loads, path.read_text().splitlines())}


def _write_cache(base_sha: str, transcripts: dict[str, str]) -> None:
    path = cache_path(base_sha)
    owner_dir(path.parent)
    lines = "".join(json.dumps({"id": k, "transcript": transcripts[k]}, sort_keys=True) + "\n" for k in sorted(transcripts))
    descriptor, scratch = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(lines)
            file.flush()
            os.fsync(file.fileno())
        os.replace(scratch, path)
    except BaseException:
        Path(scratch).unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def transcripts(utterances: list[Utterance], base_sha: str, transcriber: Transcriber | None,
                batch: int = TRANSCRIBE_BATCH) -> dict[str, str]:
    """{id: base transcript} for the pool, from the cache; only clips it lacks are transcribed (once).
    They go `batch` at a time, and the cache is rewritten after each batch, so memory stays bounded and
    an interrupted run resumes where it stopped."""
    cached = read_cache(base_sha)
    missing = [u for u in utterances if u.id not in cached]
    if missing and transcriber is None:
        raise ReplayError(f"{len(missing)} replay clip(s) have no base transcript cached and no transcriber was given")
    for start in range(0, len(missing), batch):
        chunk = missing[start : start + batch]
        made = transcriber([Audio(u.id, u.load()) for u in chunk])
        lost = [u.id for u in chunk if u.id not in made]
        if lost:
            raise ReplayError(f"the base model returned no transcript for {len(lost)} replay clip(s) (e.g. {lost[0]})")
        cached.update({u.id: made[u.id] for u in chunk})
        _write_cache(base_sha, cached)
    return {u.id: cached[u.id] for u in utterances}


def usable(text: str) -> bool:
    return bool(normalized_words(text))


def benchmark_transcriber(gguf: Path, catalog_name: str | None = None) -> Transcriber:
    """Transcribe with `hex benchmark-transcription --model <gguf>` in an isolated Hex data folder.

    The benchmark recognizes a catalog model by its file name and size (`Backend::transcribe_cpp`) and
    only then appends the model's trailing silence (400 ms for Unified). v0 lives in the registry under
    its hash, so with `catalog_name` it is offered through a symlink of that name, and replay labels are
    made on the same padded audio training gets. `run` refuses zero warmups, so one is done.
    The benchmark needs a reference per clip; a placeholder stands in, and only each clip's
    `transcript` is read (the scores are meaningless here)."""
    from healing.eval import bench

    def transcribe(clips: list[Audio]) -> dict[str, str]:
        if not clips:
            return {}
        work = healing_home() / "tmp"
        owner_dir(work)
        with tempfile.TemporaryDirectory(dir=work) as scratch:
            model = gguf
            if catalog_name:
                model = Path(scratch) / catalog_name
                model.symlink_to(gguf.resolve())
            folder = Path(scratch) / "set"
            samples = {c.id: c.samples for c in clips}
            entries = [{"id": c.id, "expected": "placeholder", "terms": []} for c in clips]
            bench_manifest.write_set(folder, entries, samples.__getitem__, Path(scratch))
            run, _ = bench.run_benchmark(folder / "manifest.json", model=model, runs=1, warmups=1)
        return {record["id"]: record["transcript"] for record in run.clips}

    return transcribe
