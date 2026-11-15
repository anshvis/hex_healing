"""Benchmark manifests exactly as `src/transcription/benchmark.rs` reads them.

Fields (`Manifest`, `ClipDefinition`): `vocabulary` (scoring terms, never shown to the model) and
`clips[]` of `{id, audio, expected, expected_terms, group}`. `validate` ports `validate_manifest`
for the two groups healing writes, `target` and `ordinary`; the real binary has the last word
(`test_splits.py > manifests accepted by real benchmark`).
"""

from __future__ import annotations

import hashlib
import json
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


def write_set(out: Path, entries: list[dict], pcm_for) -> tuple[str, int]:
    """Write `out/{manifest.json, manifest.sha256, clips.jsonl, audio/<id>.wav}`; (sha256, words).

    Each entry: `id`, `expected`, `terms` (expected terms) and any sidecar fields. `pcm_for(id)`
    returns the clip's int16 mono 16 kHz samples. The pin goes last, as in the general set, so a
    folder without it is an unfinished build. Reuses `sets.general.verify`'s layout, so one
    verifier covers every set.
    """
    vocabulary: dict[str, str] = {}
    clips = []
    for entry in entries:
        terms = []
        for term in entry["terms"]:
            key = term_key(term)
            if key and key not in {term_key(t) for t in terms}:
                terms.append(vocabulary.setdefault(key, term))
        clips.append(
            {"id": entry["id"], "audio": f"audio/{entry['id']}.wav", "expected": entry["expected"],
             "group": "target" if terms else "ordinary", **({"expected_terms": terms} if terms else {})}
        )
    manifest = {"vocabulary": sorted(vocabulary.values()), "clips": clips}
    validate(manifest)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sidecar = []
    for entry in entries:
        path = audio_dir / f"{entry['id']}.wav"
        soundfile.write(path, pcm_for(entry["id"]), 16_000, subtype="PCM_16")
        extra = {k: v for k, v in entry.items() if k not in ("id", "expected", "terms")}
        sidecar.append({"id": entry["id"], "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra})
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    (out / "clips.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sidecar))
    (out / "manifest.json").write_bytes(data)
    (out / "manifest.sha256").write_text(f"{digest}  manifest.json\n")
    return digest, sum(len(normalized_words(c["expected"])) for c in clips)
