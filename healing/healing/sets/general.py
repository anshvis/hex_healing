"""The general regression set: LibriSpeech test-other plus AMI IHM test (D-11).

It never changes once written: `manifest.sha256` pins the clip list, and
`clips.jsonl` pins each WAV (written and verified by `healing.manifest`, as every set is). Audio lives under `$HEALING_HOME`, never in the
repo. The manifest is a `hex benchmark-transcription` manifest; every clip is
group `ordinary`, and the source sits in the `clips.jsonl` sidecar.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import soundfile

from healing.manifest import ManifestError, build_manifest, verify, write_set

SAMPLE_RATE = 16_000

HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
# LibriSpeech test-other through Hugging Face's parquet copy of openslr's
# release (openslr's own mirrors serve it at tens of KB/s).
LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
LIBRISPEECH_SHARDS = {
    "other/test/0000.parquet": "38e0c86a8104585c577badd707ca4331e20fa2c645f46180af0b7bcdecff9249",
}
AMI_REPO = "edinburghcstr/ami"
AMI_REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"
AMI_SHARDS = {
    "ihm/test-00000-of-00004.parquet": "d95920dccc6924c15215239461bf5d1152fe07c9c61add073bd12da26dd602e0",
    "ihm/test-00001-of-00004.parquet": "07a2b1c407cf36bc1e66c1362b0248ed32c897edc2a116549d1dc6b456dc2325",
    "ihm/test-00002-of-00004.parquet": "83cde21e55fc884295585327b0ee334cba47ad53bdb787dd5f97437067aca5f8",
    "ihm/test-00003-of-00004.parquet": "3312bb79400ddef62d5a0097e124dfabe86a693b4643e7fef7d72a62588280d7",
}

AMI_MIN_SECONDS = 1.0
AMI_MAX_SECONDS = 20.0
AMI_MIN_WORDS = 3
# Disfluencies AMI transcribes and dictation output never contains. Real
# words that are also backchannels ("yeah", "oh", "okay") stay.
AMI_FILLERS = {"um", "umm", "uh", "uhm", "hmm", "hm", "mm", "mmm", "mm-hmm", "mhm", "uh-huh"}


BuildError = ManifestError


@dataclass(frozen=True)
class Utterance:
    id: str
    source: str
    text: str
    load: Callable[[], tuple[np.ndarray, int]]


class Source(Protocol):
    name: str

    def utterances(self) -> Iterable[Utterance]: ...


@dataclass(frozen=True)
class BuildResult:
    manifest: Path
    sha256: str
    clips: int
    unchanged: bool


def build(
    out: Path, sources: list[tuple[Source, int]], *, seed: int = 13
) -> BuildResult:
    """Writes the set to `out`, or confirms the identical set already there."""
    chosen: list[Utterance] = []
    for source, count in sources:
        candidates = sorted(source.utterances(), key=lambda utterance: utterance.id)
        if len(candidates) < count:
            raise BuildError(f"{source.name}: {len(candidates)} usable utterances, need {count}")
        rng = np.random.default_rng([seed, _stable_int(source.name)])
        picks = sorted(rng.choice(len(candidates), size=count, replace=False))
        chosen += [candidates[index] for index in picks]

    entries = [
        {"id": u.id, "expected": u.text, "terms": [], "source": u.source} for u in chosen
    ]
    digest = build_manifest(entries)[2]
    manifest_path = out / "manifest.json"
    if manifest_path.exists() or (out / "manifest.sha256").exists():
        existing = verify(out)
        if existing != digest:
            raise BuildError(
                f"{manifest_path} exists with a different clip list "
                f"(have {existing[:12]}, built {digest[:12]}); the general set never changes"
            )
        return BuildResult(manifest_path, digest, len(chosen), unchanged=True)

    by_id = {u.id: u for u in chosen}

    def pcm(clip_id: str) -> np.ndarray:
        samples, rate = by_id[clip_id].load()
        if rate != SAMPLE_RATE:
            raise BuildError(f"{clip_id}: {rate} Hz audio; the set is 16 kHz only")
        return _mono(samples)

    write_set(out, entries, pcm)
    return BuildResult(manifest_path, digest, len(chosen), unchanged=False)


def sources_by_clip(set_dir: Path) -> dict[str, str]:
    """Clip id → source name, from the sidecar."""
    rows = (json.loads(line) for line in (set_dir / "clips.jsonl").read_text().splitlines())
    return {row["id"]: row["source"] for row in rows}


def subset(out: Path, prefix: str, name: str, expected: int) -> Path:
    """Writes `<name>.json`: the set's clips whose id starts with `prefix`."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in {"manifest", "clips"}:
        raise BuildError(f"subset name {name!r} must be a plain name other than manifest or clips")
    verify(out)
    manifest = json.loads((out / "manifest.json").read_text())
    clips = [clip for clip in manifest["clips"] if clip["id"].startswith(prefix)]
    if len(clips) != expected:
        raise BuildError(f"{len(clips)} clips start with {prefix!r}, expected {expected}")
    path = out / f"{name}.json"
    path.write_text(json.dumps({"clips": clips}, indent=1, sort_keys=True) + "\n")
    return path


def _mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


# --- Sources -----------------------------------------------------------------


def download(url: str, target: Path, sha256: str | None) -> Path:
    """Fetches `url` once into `target`, checked against `sha256` when pinned."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with urllib.request.urlopen(url) as response, open(partial, "wb") as file:
                while chunk := response.read(1 << 20):
                    file.write(chunk)
        except OSError as error:
            raise BuildError(f"could not download {url}: {error}") from error
        partial.rename(target)
    if sha256 is not None:
        actual = _sha256_file(target)
        if actual != sha256:
            raise BuildError(f"{target} has sha256 {actual}, expected {sha256}")
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _shards(cache: Path, repo: str, revision: str, shards: dict[str, str]) -> list[Path]:
    return [
        download(
            HF_URL.format(repo=repo, revision=revision, path=shard),
            cache / repo.replace("/", "--") / shard,
            sha256,
        )
        for shard, sha256 in sorted(shards.items())
    ]


class LibriSpeechTestOther:
    name = "librispeech"

    def __init__(self, cache: Path):
        self.cache = cache

    def utterances(self) -> Iterable[Utterance]:
        import pyarrow.parquet as parquet

        for path in _shards(self.cache, LIBRISPEECH_REPO, LIBRISPEECH_REVISION, LIBRISPEECH_SHARDS):
            rows = parquet.read_table(path, columns=["id", "text"]).to_pylist()
            for row_index, row in enumerate(rows):
                yield Utterance(
                    f"ls-{row['id']}",
                    self.name,
                    row["text"].strip(),
                    lambda path=path, row_index=row_index: _parquet_audio(path, row_index),
                )


class AmiIhmTest:
    name = "ami"

    def __init__(self, cache: Path):
        self.cache = cache

    def utterances(self) -> Iterable[Utterance]:
        import pyarrow.parquet as parquet

        for path in _shards(self.cache, AMI_REPO, AMI_REVISION, AMI_SHARDS):
            rows = parquet.read_table(
                path, columns=["audio_id", "text", "begin_time", "end_time"]
            ).to_pylist()
            for row_index, row in enumerate(rows):
                seconds = row["end_time"] - row["begin_time"]
                text = ami_reference(row["text"])
                if not (AMI_MIN_SECONDS <= seconds <= AMI_MAX_SECONDS):
                    continue
                if len(text.split()) < AMI_MIN_WORDS:
                    continue
                yield Utterance(
                    f"ami-{row['audio_id']}",
                    self.name,
                    text,
                    lambda path=path, row_index=row_index: _parquet_audio(path, row_index),
                )


def ami_reference(text: str) -> str:
    """AMI's text with disfluency tags and fillers removed."""
    text = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", text)
    words = [word for word in text.split() if word.lower() not in AMI_FILLERS]
    return " ".join(words)


def _parquet_audio(path: Path, row_index: int) -> tuple[np.ndarray, int]:
    """Decodes row `row_index`'s `audio` cell (encoded bytes) without loading the shard."""
    import pyarrow.parquet as parquet

    file = parquet.ParquetFile(path)
    offset = 0
    for group in range(file.num_row_groups):
        rows = file.metadata.row_group(group).num_rows
        if row_index < offset + rows:
            cell = file.read_row_group(group, columns=["audio"]).column(0)[row_index - offset]
            return soundfile.read(io.BytesIO(cell["bytes"].as_py()), dtype="float32")
        offset += rows
    raise BuildError(f"{path}: no row {row_index}")
