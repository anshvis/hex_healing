"""`healing models register-base | register | list | show | restore | current | pin-golden`.

`register-base` reads Hex's selected catalog model through `hex model list --json` and pins v0 by its
GGUF's SHA-256, never by the catalog id, so a catalog update that changes the file cannot move v0
(challenge finding 20). `current` cross-checks the registry against the same listing and only reports
a mismatch (you may have picked another model by hand); it never changes Hex.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from healing import hexcli
from healing.paths import healing_home, hex_support_dir, repo_root
from healing.registry import objects
from healing.registry import restore as restore_module
from healing.registry import snapshot as snapshots
from healing.registry import store as registry
from healing.registry.store import RegistryError

ARCH_BY_CATALOG_ID = {"parakeet-unified-en-0.6b": "parakeet-unified", "parakeet-tdt-0.6b-v2": "tdt-v2"}
ARCHES = ("parakeet-unified", "tdt-v2")
SCORE_SETS = ("rolling", "golden", "general", "edited")
SCORE_METRICS = ("word_errors", "ref_words", "reference_words", "format_error_rate", "term_error_rate", "edited_span")
EXIT_UNVERIFIABLE = 4  # `current`: Hex's file could not be checked (1 is a mismatch, 3 Hex not running)


class CommandError(RuntimeError):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# ---- Hex's model list ---------------------------------------------------------------------------

def hex_models(binary: Path) -> list[dict]:
    """`hex model list --json` (`ControlReply::Models`, src/control.rs): `{status: "models", models: [...]}`."""
    try:
        done = hexcli.run(binary, ["model", "list", "--json"])
    except hexcli.HexNotRunning as error:
        raise CommandError(str(error), hexcli.EXIT_NOT_RUNNING) from error
    except hexcli.HexError as error:
        raise CommandError(str(error)) from error
    reply = hexcli.reply(done.stdout)
    if not reply or reply.get("status") != "models" or not isinstance(reply.get("models"), list):
        raise CommandError("hex model list --json printed something this version cannot read")
    return reply["models"]


def selected_model(binary: Path) -> dict:
    chosen = [m for m in hex_models(binary) if m.get("selected")]
    if len(chosen) != 1:
        raise CommandError(f"hex model list shows {len(chosen)} selected models, not one")
    return chosen[0]


def catalog_file(model_id: str) -> Path:
    """Where Hex keeps a downloaded catalog model (`models/<filename>`, src/transcription/artifacts.rs)."""
    folder = hex_support_dir() / "models"
    exact = folder / f"{model_id}-Q8_0.gguf"
    if exact.is_file():
        return exact
    near = sorted(folder.glob(f"{model_id}*.gguf"))
    if len(near) == 1:
        return near[0]
    raise CommandError(f"no single GGUF for {model_id} in {folder}; pass --gguf")


# ---- scores -------------------------------------------------------------------------------------

def _set_score(scores: dict, set_sha: str | None, source: str, vocabulary_terms: int | None) -> dict:
    """One set's score with what identifies it: the set's manifest hash and how many vocabulary terms the
    model was biased with (0 is none; None is unknown). Only vocabulary-free scores count for parity.
    The metrics are `healing eval`'s own names (`eval/run.py`'s `score_set`); `wer` is None on the
    edited set, which is scored at its edited span instead."""
    out = {"wer": scores.get("wer"), "source": source, "vocabulary_terms": vocabulary_terms}
    for key in SCORE_METRICS:
        if scores.get(key) is not None:
            out[key] = scores[key]
    if set_sha:
        out["set_sha256"] = set_sha
    return out


def scores_from(document: dict, gguf_sha: str, source: str) -> dict:
    """The scores a JSON document holds for `gguf_sha`: a `baseline.json` (T0.3), a `healing eval`
    document (its model with that hash), or a plain `{set: {wer, ...}}` map."""
    if "wer" in document and "set_sha256" in document:  # baseline.json: the general set
        if document.get("gguf_sha256") not in (None, gguf_sha):
            return {}
        # `healing eval baseline` is `healing eval --vocabulary none`
        return {"general": _set_score({**document, "ref_words": document.get("reference_words")}, document["set_sha256"], source, 0)}
    if isinstance(document.get("models"), list) and isinstance(document.get("scores"), dict):
        paths = [m["gguf_path"] for m in document["models"] if m.get("gguf_sha256") == gguf_sha]
        if not paths:
            return {}
        manifests = {s["set"]: s.get("manifest_sha256") for s in document.get("sets", [])}
        found = document["scores"].get(paths[0], {})
        vocabulary = document.get("vocabulary")
        terms = vocabulary.get("terms") if isinstance(vocabulary, dict) else None
        return {name: _set_score(s, manifests.get(name), source, terms) for name, s in found.items()
                if any(s.get(key) is not None for key in ("wer", *SCORE_METRICS))}
    return {name: _set_score(s, s.get("set_sha256"), source, s.get("vocabulary_terms")) for name, s in document.items()
            if name in SCORE_SETS and isinstance(s, dict)}


def recorded_scores(gguf_sha: str, named: Path | None) -> dict:
    """`--scores` when given; else T0.3's `healing/eval/baseline.json` and every `healing eval` in
    `$HEALING_HOME/evals`, oldest first so the newest score of each set wins, kept only for this file."""
    def read(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise CommandError(f"cannot read scores from {path}: {error}") from error

    if named is not None:
        found = scores_from(read(named), gguf_sha, str(named))
        if not found:
            raise CommandError(f"{named} holds no scores for GGUF {gguf_sha[:12]}")
        return found
    found: dict = {}
    sources = [repo_root() / "healing" / "eval" / "baseline.json"] + sorted((healing_home() / "evals").glob("eval-*.json"))
    for path in sources:
        if path.is_file():
            try:
                found.update(scores_from(json.loads(path.read_text()), gguf_sha, str(path)))
            except (OSError, ValueError):
                continue
    return found


# ---- commands -----------------------------------------------------------------------------------

def code_shas(hex_sha: str | None) -> dict:
    from healing.eval.bench import git_sha

    return {"healing": git_sha() or None, "hex": hex_sha}


def register_base(binary: Path | None, gguf: Path | None, model_id: str | None, arch: str | None,
                  scores_file: Path | None) -> list[str]:
    if gguf is None or model_id is None:
        model = selected_model(binary)
        model_id = model_id or model["id"]
        if model_id.startswith(objects.CUSTOM_PREFIX) and gguf is None:
            raise CommandError(f"Hex runs {model_id}, an imported model; v0 is the catalog model (pass --gguf and --model-id)")
        gguf = gguf or catalog_file(model_id)
    arch = arch or ARCH_BY_CATALOG_ID.get(model_id)
    if arch not in ARCHES:
        raise CommandError(f"cannot tell {model_id}'s architecture; pass --arch ({' or '.join(ARCHES)})")
    sha = objects.sha256_file(gguf)
    with registry.lock():
        state = registry.load()
        if "v0" in state.versions:
            v0 = state.versions["v0"]
            if v0.artifact("gguf") != sha:
                raise CommandError(f"v0 is pinned at {str(v0.artifact('gguf'))[:12]}; {gguf} is {sha[:12]}. v0 never moves")
            return _resume_base(state, v0, gguf, sha, model_id, scores_file)
        objects.put(gguf, sha)
        scores = recorded_scores(sha, scores_file)
        registry.append(
            {"event": "registered", "version": "v0", "base": {"catalog_id": model_id, "gguf_sha256": sha},
             "parent": None, "arch": arch, "dataset": None, "config": None, "code_sha": code_shas(None),
             "converter": None, "quantization": "q8_0",
             "artifacts": {"lora_delta": None, "merged_nemo": None, "gguf": sha}, "source_path": str(gguf)},
            {"event": "promoted", "version": "v0", "reason": "base"},
        )
        if scores:
            registry.append({"event": "scored", "version": "v0", "scores": scores})
    lines = [f"v0: {model_id}, gguf {sha[:12]} ({objects.hex_model_id(sha)}), current"]
    lines.append("scores: " + (", ".join(f"{k} {_pct(v.get('wer'))}" for k, v in scores.items()) or "none recorded"))
    return lines


def _resume_base(state: registry.State, v0: registry.Version, gguf: Path, sha: str, model_id: str,
                 scores_file: Path | None) -> list[str]:
    """Finish an init a crash interrupted (between the `registered`, `promoted` and `scored` lines, or
    before the object was stored); with nothing left to do, say v0 is already registered. The caller
    holds the lock."""
    done = []
    whole = objects.has(sha) and objects.sha256_file(objects.path(sha)) == sha
    objects.put(gguf, sha)  # validates a stored copy and replaces a damaged one
    if not whole:
        done.append("stored its GGUF")
    if state.current is None and not state.promotions and v0.state == "candidate":
        registry.append({"event": "promoted", "version": "v0", "reason": "base"})
        done.append("made it current")
    if not v0.scores:
        scores = recorded_scores(sha, scores_file)
        if scores:
            registry.append({"event": "scored", "version": "v0", "scores": scores})
            done.append("recorded its scores")
    if done:
        return [f"v0 ({model_id} at {sha[:12]}): resumed an interrupted register-base: " + ", ".join(done)]
    return [f"v0 is already {model_id} at {sha[:12]}"]


def _dataset(train: str | None, snapshot_sha: str | None) -> dict | None:
    if train and snapshot_sha:
        raise CommandError("pass --train or --snapshot, not both", 2)
    if snapshot_sha:
        rows = snapshots.read(snapshot_sha)
        return {"snapshot_sha256": snapshot_sha, "snapshot_path": str(snapshots.snapshot_path(snapshot_sha).relative_to(objects.registry_dir())),
                "train_split": None, "train_n": len(rows), "strong_n": None, "weak_n": None}
    if not train:
        return None
    from healing.store import Store

    store = Store(sweep=False)
    try:
        snap = snapshots.from_train(store.db, train)
    finally:
        store.close()
    snapshots.write(snap)
    return {"snapshot_sha256": snap.sha256, "snapshot_path": str(snapshots.snapshot_path(snap.sha256).relative_to(objects.registry_dir())),
            "train_split": train, "train_n": snap.train_n, "strong_n": snap.strong_n, "weak_n": snap.weak_n,
            "weights_frozen": snap.weights_frozen, "bundle_sha256": _bundle(train)}


def _bundle(train: str) -> str | None:
    """The sha256 of `healing train prepare`'s bundle for this train split (T6.1), checked against its
    files, or None when none was prepared (or it was deleted)."""
    from healing.train import manifest as train_manifest

    path = train_manifest.bundle_for(train)
    if not path.exists():
        return None
    try:
        return train_manifest.verify_bundle(path)
    except train_manifest.TrainError as error:
        raise CommandError(f"the training bundle for {train} is damaged: {error}") from error


def register(args) -> list[str]:
    state = registry.load()
    parent = state.get(args.parent)
    converter = dict(parent.record.get("converter") or {})
    if args.converter_repo:
        converter["repo"] = args.converter_repo
    if args.converter_rev:
        converter["rev"] = args.converter_rev
    if args.export_command:
        converter["command"] = shlex.split(args.export_command)
    if args.converter_dir:
        converter["checkout"] = str(Path(args.converter_dir).resolve())
    missing = [k for k in ("repo", "rev", "command") if not converter.get(k)]
    if missing:
        raise CommandError(f"no converter {', '.join(missing)} recorded: pass --converter-repo, --converter-rev and --export-command", 2)
    config_text = Path(args.config).read_text()
    gguf_sha = objects.sha256_file(Path(args.gguf))
    delta_sha = objects.sha256_file(Path(args.delta))
    scores = recorded_scores(gguf_sha, args.scores) if args.scores else {}  # read and checked before any write
    record = {
        "base": parent.record.get("base"), "parent": parent.name, "arch": args.arch or parent.record.get("arch"),
        "dataset": _dataset(args.train, args.snapshot),
        "config": {"text": config_text, "sha256": snapshots.text_sha256(config_text)},
        "code_sha": code_shas(args.hex_sha), "converter": converter, "quantization": args.quantization,
        "artifacts": {"lora_delta": delta_sha, "merged_nemo": None, "gguf": gguf_sha},
    }
    name = registry.register(record, publish=[(Path(args.gguf), gguf_sha), (Path(args.delta), delta_sha)], scores=scores)
    return [f"{name}: candidate from {parent.name}, gguf {gguf_sha[:12]} ({objects.hex_model_id(gguf_sha)}), delta {delta_sha[:12]}"]


def _pct(wer) -> str:
    return "n/a" if wer is None else f"{wer * 100:.2f}"


def listing(state: registry.State) -> list[str]:
    lines = [f"{'version':<9}{'state':<13}{'arch':<18}{'parent':<8}{'gguf':<14}{'general':>9}{'golden':>8}{'rolling':>9}"]
    for v in state.ordered():
        mark = "*" if v.name == state.current else " "
        gguf = (v.gguf or "-")[:12] + ("" if objects.has(v.gguf) else " ·")
        scores = [_pct((v.scores.get(s) or {}).get("wer")) for s in ("general", "golden", "rolling")]
        lines.append(f"{mark}{v.name:<8}{v.state:<13}{str(v.record.get('arch')):<18}{str(v.record.get('parent') or '-'):<8}"
                     f"{gguf:<14}{scores[0]:>9}{scores[1]:>8}{scores[2]:>9}")
    lines.append("* current; · GGUF not kept (restore rebuilds it)")
    return lines


def show(state: registry.State, name: str) -> dict:
    version = state.get(name)
    out = version.as_dict()
    out["current"] = version.name == state.current
    out["hex_model_id"] = objects.hex_model_id(version.gguf) if version.gguf else None
    out["kept"] = {k: objects.has(sha) for k, sha in (version.record.get("artifacts") or {}).items() if sha}
    dataset = version.record.get("dataset")
    out["reproducible"] = None
    if dataset:
        from healing.store import Store

        try:
            rows = snapshots.read(dataset["snapshot_sha256"])
            store = Store(sweep=False)
            try:
                out["reproducible"] = snapshots.reproducible(store, rows)
            finally:
                store.close()
        except snapshots.SnapshotError as error:
            out["reproducible"] = f"unknown: {error}"
    return out


def running_model(binary: Path) -> dict:
    """What Hex runs, from `hex status --json`'s `model` (`ModelReport`, src/control.rs): the saved `id`,
    whether it is `installed`, and the `fallback` Hex runs in its place while a saved import is missing,
    changed or switched off (`custom_models::fallback_reason`). `hex model list` keeps such an import
    `selected`, so it can't say what runs."""
    try:
        done = hexcli.run(binary, ["status", "--json"])
    except hexcli.HexNotRunning as error:
        raise CommandError(str(error), hexcli.EXIT_NOT_RUNNING) from error
    except hexcli.HexError as error:
        raise CommandError(str(error)) from error
    reply = hexcli.reply(done.stdout)
    model = reply.get("model") if reply and reply.get("status") == "status" else None
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise CommandError("hex status --json printed no model this version can read")
    return model


def current(binary: Path) -> tuple[list[str], int]:
    """The current version against what Hex really runs. Exit 1 on a mismatch and `EXIT_UNVERIFIABLE` when
    it can't be checked (the saved model isn't installed, or v0's catalog file can't be hashed); neither is
    ever fixed here."""
    version = registry.load().get("current")
    lines = [f"current {version.name}, gguf {(version.gguf or '-')[:12]}"]
    model = running_model(binary)
    catalog_id = (version.record.get("base") or {}).get("catalog_id")
    expected = {objects.hex_model_id(sha) for sha in version.ggufs()}
    if not version.artifact("lora_delta"):
        expected.add(catalog_id)
    fallback = model.get("fallback")
    if isinstance(fallback, dict) and fallback.get("id"):
        running = fallback["id"]
        note = f"Hex saved {model['id']} but runs {running} in its place ({fallback.get('reason', 'unknown')})"
    elif model.get("installed") is not True:
        lines.append(f"unverifiable: Hex's saved model {model['id']} is not installed, so it is not running it")
        return lines, EXIT_UNVERIFIABLE
    else:
        running, note = model["id"], f"Hex runs {model['id']}"
    if running not in expected:
        lines.append(f"mismatch: {note}, but the registry's current {version.name} is "
                     + " or ".join(sorted(e for e in expected if e)) + " (not changed; pick it in Hex or promote what Hex runs)")
        return lines, 1
    if running == catalog_id:
        try:
            on_disk = objects.sha256_file(catalog_file(running))
        except (CommandError, OSError) as error:
            lines.append(f"unverifiable: {note}, but its file could not be hashed ({error}), "
                         f"so it may not be the pinned {(version.gguf or '-')[:12]}")
            return lines, EXIT_UNVERIFIABLE
        if on_disk not in version.ggufs():
            lines.append(f"mismatch: {note}, but its file is now {on_disk[:12]}, not the pinned "
                         f"{(version.gguf or '-')[:12]} (a catalog update; not changed)")
            return lines, 1
    lines.append(f"{note}: matches")
    return lines, 0


def pin_golden() -> str:
    from healing import splits
    from healing.store import Store

    store = Store(sweep=False)
    try:
        row = splits.golden_row(store.db)
    finally:
        store.close()
    if row is None or not row["frozen"]:
        raise CommandError("the golden set is not frozen yet (`healing golden review`)")
    written = registry.pin_golden(row["manifest_sha256"], row["n_clips"])
    return f"golden {'pinned' if written else 'already pinned'} at {row['manifest_sha256'][:12]} ({row['n_clips']} clips)"


def golden_pin_problem(row) -> str | None:
    """`split verify`'s G-10 check: the frozen golden row's manifest hash against the registry's pin."""
    pin = registry.load().golden
    if pin is None or row is None or not row["frozen"] or row["manifest_sha256"] == pin["sha256"]:
        return None
    return f"golden: manifest hash {str(row['manifest_sha256'])[:12]} differs from the registry's pin {pin['sha256'][:12]}"


def main(args) -> int:
    from healing.eval.bench import BenchError, history_hex
    from healing.store import StoreError

    def binary() -> Path:
        return Path(args.hex or history_hex())

    try:
        if args.command == "register-base":
            needs_hex = args.gguf is None or args.model_id is None
            lines = register_base(binary() if needs_hex else None, args.gguf, args.model_id, args.arch, args.scores)
        elif args.command == "register":
            lines = register(args)
        elif args.command == "list":
            state = registry.load()
            if args.json:
                print(json.dumps({"current": state.current, "golden": state.golden,
                                  "versions": [v.as_dict() for v in state.ordered()]}, indent=2))
                return 0
            lines = listing(state)
        elif args.command == "show":
            print(json.dumps(show(registry.load(), args.version), indent=2))
            return 0
        elif args.command == "restore":
            event = restore_module.restore(args.version, checkout=args.converter_dir)
            how = "sha256 matches" if event["method"] == "sha256" else "WER parity (recorded in the version)"
            lines = [f"{args.version}: rebuilt {event['rebuilt_sha256'][:12]}, {how}"]
        elif args.command == "current":
            lines, code = current(binary())
            print("\n".join(lines))
            return code
        else:
            lines = [pin_golden()]
    except CommandError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except (RegistryError, restore_module.RestoreError, snapshots.SnapshotError, StoreError, BenchError,
            OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0
