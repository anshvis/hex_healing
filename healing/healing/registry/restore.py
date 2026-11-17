"""`healing models restore <v>`: rebuild a version's GGUF from base + LoRA delta + the pinned converter.

The version records its converter as `{repo, rev, command}`: `command` is the export invocation (the
T6.3 export), an argv whose `{base}`, `{delta}`, `{config}`, `{output}` and `{checkout}` are filled in
here. It runs in a local checkout of `repo` (`--converter-dir`, `$HEALING_CONVERTER_DIR`, or the
checkout recorded at `register`) that must be at exactly `rev` with no local changes, so the rebuild uses
the recorded converter and not whatever is checked out today.

The rebuilt file's SHA-256 is checked against the record. A converter that is not deterministic gives a
different hash; then the rebuild is scored on the golden and general sets the version has recorded
scores for, and kept only when every comparable WER is within 0.1 pt of the record (`restored` with
`method: wer_parity`, the scores and the rebuilt hash, which is then the version's GGUF). Both seams,
the converter and the parity check, are arguments, so tests can swap them.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from healing.registry import objects
from healing.registry import store as registry
from healing.registry.objects import registry_dir

PARITY_TOLERANCE = 0.001  # 0.1 pt of WER
PARITY_SETS = ("golden", "general")
CONVERTER_TIMEOUT_S = 6 * 3600
PLACEHOLDERS = ("base", "delta", "config", "output", "checkout")


class RestoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Inputs:
    base: Path
    delta: Path
    config: Path
    output: Path
    checkout: Path


Converter = Callable[[dict, Inputs], None]
Parity = Callable[[registry.Version, Path], dict]


def check_checkout(converter: dict, checkout: Path) -> None:
    """The checkout is `converter.repo` at `converter.rev` with no local changes."""
    rev = str(converter.get("rev") or "")
    if len(rev) < 7:
        raise RestoreError("the version records no converter rev")

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)
        if done.returncode != 0:
            raise RestoreError(f"{checkout} is not a git checkout of the converter: {done.stderr.strip()}")
        return done.stdout.strip()

    head = git("rev-parse", "HEAD")
    if not head.startswith(rev) and not rev.startswith(head):
        raise RestoreError(f"the converter checkout {checkout} is at {head[:12]}, not the recorded {rev[:12]}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RestoreError(f"the converter checkout {checkout} has local changes")


def run_recorded(converter: dict, inputs: Inputs) -> None:
    """The default converter: the recorded export command, in the checked checkout."""
    command = converter.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(p, str) for p in command):
        raise RestoreError("the version records no export command")
    check_checkout(converter, inputs.checkout)
    values = {name: str(getattr(inputs, name)) for name in PLACEHOLDERS}
    argv = []
    for part in command:
        for name, value in values.items():
            part = part.replace("{" + name + "}", value)
        argv.append(part)
    try:
        done = subprocess.run(argv, cwd=inputs.checkout, capture_output=True, text=True, timeout=CONVERTER_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RestoreError(f"the export command failed to run: {error}") from error
    if done.returncode != 0:
        raise RestoreError(f"the export command exited {done.returncode}: {(done.stderr or done.stdout).strip()[-300:]}")
    if not inputs.output.is_file():
        raise RestoreError(f"the export command wrote no file at {inputs.output}")


def eval_parity(version: registry.Version, gguf: Path) -> dict:
    """The default parity check: `healing eval` (no vocabulary) of the rebuild on the golden and general
    sets the version has recorded WERs for. A set whose manifest changed since it was scored is not
    comparable and is skipped; at least one set must compare."""
    from healing.eval import run as eval_run

    recorded = {s: version.scores[s] for s in PARITY_SETS if (version.scores.get(s) or {}).get("wer") is not None}
    if not recorded:
        raise RestoreError(f"{version.name} has no recorded golden or general WER to check parity against")
    document = eval_run.evaluate([gguf], list(recorded), vocabulary="none")
    manifests = {s["set"]: s["manifest_sha256"] for s in document["sets"]}
    scores = document["scores"][str(gguf)]
    return compare(recorded, {s: (scores[s]["wer"], manifests.get(s)) for s in scores})


def compare(recorded: dict, rebuilt: dict) -> dict:
    """`rebuilt` is `{set: (wer, manifest sha)}`. Passes when every comparable set is within tolerance."""
    sets, skipped = {}, {}
    for name, score in recorded.items():
        if name not in rebuilt or rebuilt[name][0] is None:
            skipped[name] = "not scored now"
            continue
        wer, manifest = rebuilt[name]
        if score.get("set_sha256") and manifest and score["set_sha256"] != manifest:
            skipped[name] = "the set changed since it was scored"
            continue
        sets[name] = {"recorded": score["wer"], "rebuilt": wer, "delta_pt": round((wer - score["wer"]) * 100, 4)}
    passed = bool(sets) and all(abs(s["rebuilt"] - s["recorded"]) <= PARITY_TOLERANCE + 1e-12 for s in sets.values())
    return {"tolerance_pt": PARITY_TOLERANCE * 100, "sets": sets, "skipped": skipped, "passed": passed}


def converter_checkout(version: registry.Version, named: Path | None) -> Path:
    found = named or (Path(os.environ["HEALING_CONVERTER_DIR"]) if os.environ.get("HEALING_CONVERTER_DIR") else None)
    found = found or (Path(version.record["converter"]["checkout"]) if (version.record.get("converter") or {}).get("checkout") else None)
    if found is None:
        raise RestoreError("no converter checkout: pass --converter-dir or set HEALING_CONVERTER_DIR")
    return found


def _object(sha: str | None, what: str) -> Path:
    if not sha:
        raise RestoreError(f"the version records no {what}")
    path = objects.path(sha)
    if not path.is_file():
        raise RestoreError(f"the {what} ({sha[:12]}) is not in {objects.objects_dir()}")
    if objects.sha256_file(path) != sha:
        raise RestoreError(f"the {what} object {sha[:12]} no longer matches its hash")
    return path


def restore(name: str, *, checkout: Path | None = None, converter: Converter = run_recorded,
            parity: Parity = eval_parity) -> dict:
    """Rebuild `name`'s GGUF into the objects store and record how it was checked. Returns the event."""
    version = registry.load().get(name)
    if not version.artifact("lora_delta"):
        raise RestoreError(f"{version.name} has no LoRA delta: it is a base model and is kept, not rebuilt")
    base = _object((version.record.get("base") or {}).get("gguf_sha256"), "base GGUF")
    delta = _object(version.artifact("lora_delta"), "LoRA delta")
    expected = version.artifact("gguf")
    scratch_root = registry_dir() / "tmp"
    objects.owner_dir(scratch_root)
    with tempfile.TemporaryDirectory(prefix=f"restore-{version.name}-", dir=scratch_root) as scratch:
        folder = Path(scratch)
        config = folder / "config.yaml"
        config.write_text((version.record.get("config") or {}).get("text") or "")
        output = folder / "model.gguf"
        converter(version.record.get("converter") or {}, Inputs(base, delta, config, output, converter_checkout(version, checkout)))
        if not output.is_file():
            raise RestoreError("the converter wrote no GGUF")
        rebuilt = objects.sha256_file(output)
        event: dict = {"event": "restored", "version": version.name, "rebuilt_sha256": rebuilt}
        if rebuilt == expected:
            event["method"] = "sha256"
        else:
            result = parity(version, output)
            if not result.get("passed"):
                raise RestoreError(
                    f"the rebuild hashes to {rebuilt[:12]}, not {str(expected)[:12]}, and fails WER parity: {result}")
            event.update(method="wer_parity", gguf_sha256=rebuilt, parity=result)
        with registry.lock():
            objects.put(output, rebuilt)
            registry.append(event)
    return event
