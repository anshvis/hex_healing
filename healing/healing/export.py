"""`healing export <delta>`: merge → GGUF → Q8_0 → check → register a candidate (T6.3).

The build is the converter record restore replays (D-43). `export` writes the record
`{repo, rev, checkout, command}` first, then runs it through `restore.run_recorded`, the same function
`healing models restore` uses, so the two can't drift. The command is
`{python} -m healing.cli export-gguf …`, and `build` below is what it runs:

1. `merge`: `train/merge.py` in NeMo's interpreter. It applies W ← W + (α/r)·B·A, as the trainer
   does, to the base `.nemo` the config pins.
2. `convert`: transcribe.cpp's `scripts/convert-parakeet.py`, from the pinned checkout, to F32. It
   takes the variant from the output folder's name, so the F32 goes in `<work>/<base repo name>/`.
3. `quantize`: that checkout's `transcribe-quantize --quant Q8_0`.
4. `general.name` is set to `hex-healing <version>`, so Hex's model list says which version it is.
5. The tensor names must equal the base GGUF's, in order (T0.4).

Then `export` refuses a header without `stt.variant` or `general.languages`. It imports the file
into a throwaway `HEX_APPLICATION_SUPPORT_DIR` (`hex model import` has no dry run) and checks the
id and the name Hex reports. Then it registers the candidate under the name it built in.

The converter is `vendor/transcribe-cpp.rev` from `REPO`. It is fetched into
`$HEALING_HOME/tools/transcribe.cpp@<rev>` and built once. `--converter-dir` or
`$HEALING_CONVERTER_DIR` names an existing checkout instead, and `$HEALING_CONVERTER_SOURCE` clones
from somewhere other than `REPO`. Either way it must be at exactly the rev and clean. Every temp
file (about 3 GB: the merged `.nemo`, the F32 and the Q8_0) is under one folder, which is removed on
success and on failure.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from healing import gguf
from healing.paths import healing_home, repo_root
from healing.registry import objects
from healing.registry import restore
from healing.registry import store as registry
from healing.train import config as config_module
from healing.train import lora

REPO = "https://github.com/anomalyco/transcribe.cpp"
REV_FILE = repo_root() / "healing" / "vendor" / "transcribe-cpp.rev"
MERGER = Path(__file__).resolve().parent / "train" / "merge.py"
CONVERT = Path("scripts") / "convert-parakeet.py"
QUANTIZE = Path("build") / "bin" / "transcribe-quantize"
STAMP = Path("build") / ".healing-built-rev"
STEP_TIMEOUT_S = 3 * 3600
STDERR_TAIL = 240  # the recorded command's whole error must fit restore's 300-character tail
IMPORTED = "imported\t"


class ExportError(RuntimeError):
    pass


def pinned_rev() -> str:
    rev = REV_FILE.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ExportError(f"{REV_FILE} does not hold a full commit SHA")
    return rev


def version_name(version: str) -> str:
    return f"hex-healing {version}"


def tail(text: str, size: int = STDERR_TAIL) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= size else "…" + text[-size:]


def step(name: str, argv: list, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run one step; a failure names the step and ends with the tail of its stderr."""
    try:
        done = subprocess.run([str(a) for a in argv], cwd=cwd, capture_output=True, text=True, timeout=STEP_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExportError(f"{name} failed to run: {tail(str(error))}") from error
    if done.returncode != 0:
        raise ExportError(f"{name} failed (exit {done.returncode}): {tail(done.stderr or done.stdout)}")
    return done


# ---- the converter checkout ---------------------------------------------------------------------

def tools_checkout(rev: str) -> Path:
    return healing_home() / "tools" / f"transcribe.cpp@{rev}"


def converter_checkout(named: Path | None, rev: str) -> Path:
    """A clean checkout at `rev` with `transcribe-quantize` built from it, fetched and built if needed."""
    chosen = named or (Path(os.environ["HEALING_CONVERTER_DIR"]) if os.environ.get("HEALING_CONVERTER_DIR") else None)
    checkout = (chosen or tools_checkout(rev)).expanduser()
    if not checkout.exists():
        if chosen is not None:
            raise ExportError(f"the converter checkout {checkout} does not exist")
        _fetch(checkout, rev)
    try:
        restore.check_checkout({"rev": rev}, checkout)
    except restore.RestoreError as error:
        raise ExportError(str(error)) from error
    stamp = checkout / STAMP
    if not (checkout / QUANTIZE).is_file() or (stamp.read_text().strip() if stamp.is_file() else "") != rev:
        step("build", ["cmake", "-S", checkout, "-B", checkout / "build", "-DCMAKE_BUILD_TYPE=Release",
                       "-DTRANSCRIBE_BUILD_TOOLS=ON", "-DTRANSCRIBE_BUILD_TESTS=OFF"])
        step("build", ["cmake", "--build", checkout / "build", "--target", "transcribe-quantize", "-j", "8"])
        stamp.write_text(rev + "\n")
    return checkout.resolve()


def _fetch(checkout: Path, rev: str) -> None:
    objects.owner_dir(checkout.parent)
    partial = checkout.with_name(checkout.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    source = os.environ.get("HEALING_CONVERTER_SOURCE") or REPO
    try:
        step("fetch", ["git", "clone", "--quiet", "--no-checkout", source, partial])
        step("fetch", ["git", "-C", partial, "checkout", "--quiet", "--detach", rev])
        partial.rename(checkout)
    finally:
        shutil.rmtree(partial, ignore_errors=True)


# ---- checks -------------------------------------------------------------------------------------

def read_config(text: str, where: str) -> dict:
    try:
        return config_module.check(config_module.parse(text, where), where)
    except config_module.ConfigError as error:
        raise ExportError(str(error)) from error


def check_delta(delta: Path, config: dict) -> None:
    """The delta is the trainer's: LoRA A/B pairs over the encoder and the joint, at the config's rank
    and alpha, trained on the base the config pins."""
    try:
        header = lora.safetensors_header(delta)
    except (OSError, ValueError) as error:
        raise ExportError(f"cannot read the delta {delta}: {error}") from error
    names = [n for n in header if n != "__metadata__"]
    modules = sorted({n.rsplit(".", 1)[0] for n in names})
    problems = lora.metadata_problems(header.get("__metadata__") or {}, config["rank"], config["alpha"],
                                      config["base"]["sha256"]) + lora.delta_problems(names, modules)
    if problems:
        raise ExportError(f"{delta}: " + "; ".join(problems[:5]))


def check_tensor_names(built: Path, base: Path) -> None:
    ours, theirs = gguf.read(built).tensors, gguf.read(base).tensors
    if ours != theirs:
        extra, missing = sorted(set(ours) - set(theirs)), sorted(set(theirs) - set(ours))
        raise ExportError(f"tensor-names: the export's {len(ours)} tensors are not the base's {len(theirs)} "
                          f"(extra {extra[:3]}, missing {missing[:3]}{', order differs' if not extra and not missing else ''})")


def check_header(path: Path, name: str) -> gguf.Header:
    """What Hex's import needs (`custom_models::profile`) and the name the build set; else refuse."""
    try:
        header = gguf.read(path)
    except (OSError, gguf.GGUFError) as error:
        raise ExportError(f"header check: {error}; not registered") from error
    languages = header.kv.get("general.languages")
    missing = [key for key, ok in (("stt.variant", bool(header.string("stt.variant"))),
                                   ("general.languages", isinstance(languages, list) and bool(languages)
                                    and all(isinstance(x, str) for x in languages))) if not ok]
    if missing:
        raise ExportError(f"header check: the GGUF has no {' or '.join(missing)}; not registered")
    if header.string("general.name") != name:
        raise ExportError(f"header check: general.name is {header.string('general.name')!r}, not {name!r}; not registered")
    return header


def isolated_import(path: Path, name: str, sha: str, binary: Path | None, scratch: Path) -> str:
    """`hex model import` into a throwaway data folder (G-8), never Hex's own. Returns its line."""
    from healing.eval.bench import history_hex

    support = scratch / "hex-support"
    support.mkdir()
    env = {**os.environ, "HEX_APPLICATION_SUPPORT_DIR": str(support), "HEX_DISABLE_UPDATES": "1"}
    try:
        done = subprocess.run([str(binary or history_hex()), "model", "import", str(path)], env=env,
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExportError(f"header check: hex model import failed to run: {error}") from error
    finally:
        shutil.rmtree(support, ignore_errors=True)
    if done.returncode != 0:
        raise ExportError(f"header check: hex model import refused it: {tail(done.stderr or done.stdout)}; not registered")
    lines = [line for line in done.stdout.splitlines() if line.startswith(IMPORTED)]
    parts = lines[-1].split("\t") if lines else []
    if len(parts) < 3 or parts[1] != objects.hex_model_id(sha) or name not in parts[2]:
        raise ExportError(f"header check: hex model import printed {done.stdout.strip()[-200:]!r}, "
                          f"not {objects.hex_model_id(sha)} named {name!r}; not registered")
    return lines[-1]


# ---- the build (the recorded command) -----------------------------------------------------------

def command(name: str) -> list[str]:
    return ["{python}", "-m", "healing.cli", "export-gguf", "--delta", "{delta}", "--config", "{config}",
            "--output", "{output}", "--checkout", "{checkout}", "--base-gguf", "{base}", "--name", name]


def build(delta: Path, config_path: Path, output: Path, checkout: Path, base_gguf: Path, name: str,
          python: Path | None = None, base: Path | None = None, say=print) -> None:
    from healing.train import run

    config = read_config(config_path.read_text(), str(config_path))
    check_delta(delta, config)
    paths = run.default_paths(config, python=python, base=base)
    if not paths.base.is_file():
        raise ExportError(f"the base checkpoint {paths.base} is missing; set HEALING_BASE_NEMO")
    if not paths.python.is_file():
        raise ExportError(f"NeMo's Python {paths.python} is missing; set HEALING_NEMO_PYTHON")
    cache = healing_home() / "tools"
    objects.owner_dir(cache)
    if run.base_sha256(paths.base, cache) != config["base"]["sha256"]:
        raise ExportError(f"{paths.base} is not the base the config pins ({config['base']['sha256'][:12]})")
    work = Path(tempfile.mkdtemp(prefix=".export-", dir=output.parent))
    try:
        # convert-parakeet.py reads the variant from its output's folder name (as roundtrip.sh lays it out)
        slug = config["base"]["repo"].split("/")[-1]
        (work / slug).mkdir()
        merged, q8, named = (work / n for n in ("merged.nemo", "model-Q8_0.gguf", "named.gguf"))
        f32 = work / slug / f"{slug}-F32.gguf"
        step("merge", [paths.python, "-I", MERGER, "--base", paths.base, "--delta", delta, "--rank", config["rank"],
                       "--alpha", config["alpha"], "--transcribe-cpp", checkout, "--out", merged])
        say(f"merged {delta.name} into {paths.base.name}")
        step("convert", [paths.python, "-I", checkout / CONVERT, merged, f32], cwd=checkout)
        merged.unlink()
        step("quantize", [checkout / QUANTIZE, f32, q8, "--quant", "Q8_0"], cwd=checkout)
        f32.unlink()
        gguf.set_string(q8, named, "general.name", name)
        q8.unlink()
        check_tensor_names(named, base_gguf)
        os.replace(named, output)
        say(f"built {output} ({name})")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---- export -------------------------------------------------------------------------------------

def _dataset(snapshot: str) -> dict | None:
    from healing.registry import commands

    sha = re.fullmatch(r"[0-9a-f]{64}", snapshot) is not None
    return commands._dataset(None if sha else snapshot, snapshot if sha else None)


def _say(message: str) -> None:
    print(message, file=sys.stderr)


def export(delta: Path, config_path: Path, snapshot: str, parent: str = "current", *, converter_dir: Path | None = None,
           hex_binary: Path | None = None, tmp_dir: Path | None = None, importer=isolated_import,
           say=_say) -> dict:
    """Build, check and register a candidate. Returns `{version, gguf_sha256, delta_sha256, import}`."""
    from healing.registry import commands

    delta, config_path = Path(delta).resolve(), Path(config_path).resolve()
    text = config_path.read_text()
    config = read_config(text, str(config_path))
    check_delta(delta, config)
    state = registry.load()
    origin = state.get(parent)
    if config["arch"] != origin.record.get("arch"):
        raise ExportError(f"the config trains {config['arch']}, but {origin.name} is {origin.record.get('arch')}")
    try:
        base_gguf = restore._object((origin.record.get("base") or {}).get("gguf_sha256"), "base GGUF")
    except restore.RestoreError as error:
        raise ExportError(str(error)) from error
    dataset = _dataset(snapshot)
    rev = pinned_rev()
    checkout = converter_checkout(converter_dir, rev)
    version = state.next_name()
    name = version_name(version)
    converter = {"repo": REPO, "rev": rev, "checkout": str(checkout), "command": command(name)}
    root = Path(tmp_dir) if tmp_dir else objects.registry_dir() / "tmp"
    objects.owner_dir(root)
    with tempfile.TemporaryDirectory(prefix=f"export-{version}-", dir=root) as scratch:
        folder = Path(scratch)
        config_copy, output = folder / "config.yaml", folder / "model.gguf"
        config_copy.write_text(text)
        say(f"building {name} from {delta} (converter {rev[:12]})")
        try:
            restore.run_recorded(converter, restore.Inputs(base_gguf, delta, config_copy, output, checkout))
        except restore.RestoreError as error:
            raise ExportError(str(error)) from error
        check_header(output, name)
        sha, delta_sha = objects.sha256_file(output), objects.sha256_file(delta)
        line = importer(output, name, sha, hex_binary, folder)
        record = {
            "base": origin.record.get("base"), "parent": origin.name, "arch": config["arch"], "dataset": dataset,
            "config": {"text": text, "sha256": config_module.text_sha256(text)},
            "code_sha": commands.code_shas(None), "converter": converter, "quantization": "q8_0",
            "artifacts": {"lora_delta": delta_sha, "merged_nemo": None, "gguf": sha},
            "gguf_name": name,
        }
        registered = registry.register(record, publish=[(output, sha), (delta, delta_sha)], expect=version)
    return {"version": registered, "gguf_sha256": sha, "delta_sha256": delta_sha, "import": line}


def main(args) -> int:
    from healing.registry.snapshot import SnapshotError
    from healing.registry.store import RegistryError
    from healing.registry.commands import CommandError

    try:
        if args.group == "export-gguf":
            build(args.delta, args.config, args.output, args.checkout, args.base_gguf, args.name)
            return 0
        result = export(args.delta, args.config, args.snapshot, args.parent, converter_dir=args.converter_dir,
                        hex_binary=args.hex, tmp_dir=args.tmp_dir)
    except (ExportError, RegistryError, SnapshotError, CommandError, gguf.GGUFError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _say(f"{result['version']}: candidate, gguf {result['gguf_sha256'][:12]} "
         f"({objects.hex_model_id(result['gguf_sha256'])}), delta {result['delta_sha256'][:12]}")
    _say(f"isolated import: {result['import']}")
    print(result["version"])
    return 0
