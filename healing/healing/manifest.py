"""Benchmark manifests exactly as `src/transcription/benchmark.rs` reads them.

Fields (`Manifest`, `ClipDefinition`): `vocabulary` (scoring terms, never shown to the model) and
`clips[]` of `{id, audio, expected, expected_terms, group}`. `validate` ports `validate_manifest`
for the two groups healing writes, `target` and `ordinary`; the real binary has the last word
(`test_splits.py > manifests accepted by real benchmark`).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import soundfile

from healing.wer import normalized_words

GROUPS = ("target", "ordinary")


class ManifestError(ValueError):
    pass


def term_key(term: str) -> str:
    """How the benchmark compares terms: lowercase words with the spacing removed."""
    return "".join(normalized_words(term))


def term_count(text: str, term: str) -> int:
    """Port of `term_count`: whole-word matches, ignoring case and internal spacing."""
    target = term_key(term)
    if not target:
        return 0
    words = normalized_words(text)
    count, start = 0, 0
    while start < len(words):
        candidate, following = "", start + 1
        for end in range(start, len(words)):
            candidate += words[end]
            if candidate == target:
                count += 1
                following = end + 1
                break
            if not target.startswith(candidate):
                break
        start = following
    return count


def validate(manifest: dict) -> None:
    clips = manifest.get("clips") or []
    if not clips:
        raise ManifestError("benchmark manifest contains no clips")
    vocabulary = [term_key(term) for term in manifest.get("vocabulary", [])]
    if "" in vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ManifestError("scoring vocabulary must contain unique, nonempty terms")
    known = set(vocabulary)
    ids: set[str] = set()
    for clip in clips:
        clip_id, terms = clip["id"], clip.get("expected_terms", [])
        if not clip_id.strip():
            raise ManifestError("benchmark clip id cannot be empty")
        if clip_id in ids:
            raise ManifestError(f"benchmark clip id is duplicated: {clip_id}")
        ids.add(clip_id)
        if not normalized_words(clip["expected"]):
            raise ManifestError(f"benchmark clip reference cannot be empty: {clip_id}")
        group = clip.get("group", "ordinary")
        if group not in GROUPS:
            raise ManifestError(f"unknown benchmark group: {group}")
        if group == "target" and not terms:
            raise ManifestError(f"target clip has no expected terms: {clip_id}")
        if group == "ordinary" and terms:
            raise ManifestError(f"guard clip cannot have expected terms: {clip_id}")
        seen: set[str] = set()
        for term in terms:
            key = term_key(term)
            if key not in known or key in seen or term_count(clip["expected"], term) == 0:
                raise ManifestError(
                    f"expected term must be unique, in vocabulary and reference: {clip_id} ({term})"
                )
            seen.add(key)


def build_manifest(entries: list[dict]) -> tuple[dict, bytes, str]:
    """(manifest, its bytes, their sha256) for `entries`: `id`, `expected`, `terms`, sidecar fields."""
    vocabulary: dict[str, str] = {}
    clips = []
    for entry in entries:
        terms: list[str] = []
        for term in entry.get("terms", []):
            key = term_key(term)
            if key and key not in {term_key(t) for t in terms}:
                terms.append(vocabulary.setdefault(key, term))
        clips.append(
            {"id": entry["id"], "audio": f"audio/{entry['id']}.wav", "expected": entry["expected"],
             "group": "target" if terms else "ordinary", **({"expected_terms": terms} if terms else {})}
        )
    manifest = {"clips": clips, **({"vocabulary": sorted(vocabulary.values())} if vocabulary else {})}
    validate(manifest)
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    return manifest, data, hashlib.sha256(data).hexdigest()


def write_set(out: Path, entries: list[dict], pcm_for) -> tuple[str, int]:
    """The one set writer: `out/{manifest.json, manifest.sha256, clips.jsonl, audio/<id>.wav}`.

    Returns (manifest sha256, reference words). `pcm_for(id)` returns the clip's mono 16 kHz samples.
    `manifest.sha256` pins both the manifest and the sidecar (which holds each WAV's hash and the
    fields scoring reads, such as `edited_positions`), and goes last so a folder without it is an
    unfinished build. `verify` is its reader.
    """
    manifest, data, digest = build_manifest(entries)
    audio_dir = out / "audio"
    _private_dir(audio_dir)
    sidecar = []
    for entry in entries:
        path = audio_dir / f"{entry['id']}.wav"
        soundfile.write(path, pcm_for(entry["id"]), 16_000, subtype="PCM_16")
        path.chmod(0o600)
        extra = {k: v for k, v in entry.items() if k not in ("id", "expected", "terms")}
        sidecar.append({"id": entry["id"], "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra})
    lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in sidecar).encode()
    _private_file(out / "clips.jsonl", lines)
    _private_file(out / "manifest.json", data)
    pin = f"{digest}  manifest.json\n{hashlib.sha256(lines).hexdigest()}  clips.jsonl\n"
    _private_file(out / "manifest.sha256", pin.encode())
    return digest, sum(len(normalized_words(c["expected"])) for c in manifest["clips"])


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    for folder in (path, path.parent):
        folder.chmod(0o700)


def _private_file(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(data)


def sidecar_digest(out: Path) -> str:
    """The sha256 of the sidecar as it is on disk now."""
    return hashlib.sha256((out / "clips.jsonl").read_bytes()).hexdigest()


def sidecar_sha256(out: Path) -> str | None:
    """The sidecar hash the pin file records, or None for a set written before it was pinned."""
    for line in (out / "manifest.sha256").read_text().splitlines()[1:]:
        digest, _, name = line.partition("  ")
        if name == "clips.jsonl":
            return digest
    return None


def verify(out: Path) -> str:
    """Checks a written set against its pins and returns the manifest's sha256.

    The manifest, the sidecar (when pinned), and every WAV against the sidecar's hash.
    """
    manifest_path, pin_path, sidecar = out / "manifest.json", out / "manifest.sha256", out / "clips.jsonl"
    if not (manifest_path.is_file() and pin_path.is_file() and sidecar.is_file()):
        raise ManifestError(f"{out} holds an incomplete set (a build stopped part-way); delete it and build again")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if pin_path.read_text().split()[0] != digest:
        raise ManifestError(f"{manifest_path} no longer matches its pin in {pin_path.name}")
    pinned = sidecar_sha256(out)
    if pinned is not None and hashlib.sha256(sidecar.read_bytes()).hexdigest() != pinned:
        raise ManifestError(f"{sidecar} no longer matches its pin in {pin_path.name}")
    ids = {clip["id"] for clip in json.loads(manifest_path.read_text())["clips"]}
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    if {row["id"] for row in rows} != ids:
        raise ManifestError(f"{sidecar} doesn't list exactly the manifest's clips")
    for row in rows:
        path = out / "audio" / f"{row['id']}.wav"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["audio_sha256"]:
            raise ManifestError(f"{path} is missing or changed; rebuild the set in a new folder")
    return digest
