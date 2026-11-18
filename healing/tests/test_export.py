"""`healing export` (T6.3): merge → GGUF → Q8_0 → header check → candidate, through the converter
record restore replays.

Every test runs in temp folders: `HEALING_HOME`, Hex's data folder, a fake `hex`, a fake NeMo
interpreter (merge and convert) and a throwaway git checkout standing in for transcribe.cpp, with a
fake `transcribe-quantize` built into it. Never the real Hex data or `hex-healing` folder (G-8).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from healing import export, gguf
from healing.cli import main
from healing.registry import objects
from healing.registry import restore
from healing.registry import snapshot as snapshots
from healing.registry import store as registry

TENSORS = ["enc.w", "joint.w", "pred.w"]
BASE_KV = {"general.architecture": "parakeet", "general.name": "Parakeet Unified EN 0.6B",
           "general.languages": ["en"], "stt.variant": "unified-en-0.6b"}

# NeMo's interpreter, faked: `-I merge.py …` writes the merged checkpoint, `-I convert-parakeet.py
# <nemo> <f32>` writes a GGUF (into a folder named after the variant, as the real one requires). FAKE_FAIL=<step> fails that step with a traceback on stderr;
# FAKE_DROP=<key> leaves a header key out; FAKE_EXTRA=1 adds a tensor the base lacks.
NEMO = """import json, os, sys
from pathlib import Path
sys.path.insert(0, {healing!r})
from healing import gguf
args = sys.argv[1:]
assert args[0] == "-I", args
script, rest = Path(args[1]).name, args[2:]
step = "merge" if script == "merge.py" else "convert"
if os.environ.get("FAKE_FAIL") == step:
    print("Traceback (most recent call last):", file=sys.stderr)
    print("  " + "x" * 400, file=sys.stderr)
    print(f"RuntimeError: the {{step}} broke here", file=sys.stderr)
    sys.exit(1)
if step == "merge":
    flags = dict(zip(rest[::2], rest[1::2]))
    data = Path(flags["--base"]).read_bytes() + Path(flags["--delta"]).read_bytes() + flags["--rank"].encode() + flags["--alpha"].encode()
    Path(flags["--out"]).write_bytes(data)
else:
    merged, f32 = rest
    if Path(f32).parent.name != "parakeet-unified-en-0.6b":  # convert-parakeet.py's variant slug
        print(f"ValueError: unknown parakeet variant slug: {{Path(f32).parent.name!r}}", file=sys.stderr)
        sys.exit(1)
    kv = {kv!r}
    kv.pop(os.environ.get("FAKE_DROP", ""), None)
    names = {tensors!r} + (["extra.w"] if os.environ.get("FAKE_EXTRA") else [])
    seed = Path(merged).read_bytes()
    gguf.write(Path(f32), kv, {{n: (seed[:16] + n.encode()).ljust(64, b"\\0") for n in names}})
"""

QUANTIZE = """#!{python}
import os, shutil, sys
if os.environ.get("FAKE_FAIL") == "quantize":
    print("quantize: unsupported tensor type", file=sys.stderr)
    sys.exit(1)
assert sys.argv[3:] == ["--quant", "Q8_0"], sys.argv
shutil.copyfile(sys.argv[1], sys.argv[2])
"""

# `hex model import <path>`: records the data folder it was given, prints Hex's line.
HEX = """#!{python}
import hashlib, os, sys
from pathlib import Path
sys.path.insert(0, {healing!r})
from healing import gguf
assert sys.argv[1:3] == ["model", "import"], sys.argv
Path({seen!r}).write_text(os.environ["HEX_APPLICATION_SUPPORT_DIR"])
path = Path(sys.argv[3])
header = gguf.read(path)
if not header.string("stt.variant"):
    print("error: its header has no stt.variant", file=sys.stderr)
    sys.exit(1)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print("imported\\tcustom-" + digest[:16] + "\\t" + header.string("general.name"))
"""


def git(folder, *args):
    return subprocess.run(["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def write_delta(path: Path, rank=16, alpha=32, base_sha="0" * 64, modules=("encoder.layers.0.self_attn.linear_q", "joint.joint_net.2")):
    tensors, offset = {}, 0
    for module in modules:
        for suffix in (".lora_a", ".lora_b"):
            tensors[module + suffix] = {"dtype": "F32", "shape": [1, 1], "data_offsets": [offset, offset + 4]}
            offset += 4
    tensors["__metadata__"] = {"format": "healing-lora-v1", "rank": str(rank), "alpha": str(alpha),
                               "scale": str(alpha / rank), "base_sha256": base_sha}
    header = json.dumps(tensors).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * offset)
    return path


class Env:
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env()
    healing_src = str(Path(export.__file__).resolve().parents[1])
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    e.support = tmp_path / "hex-support"
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(e.support))  # never the real one
    monkeypatch.delenv("HEALING_CONVERTER_DIR", raising=False)
    for key in ("FAKE_FAIL", "FAKE_DROP", "FAKE_EXTRA"):
        monkeypatch.delenv(key, raising=False)

    # transcribe.cpp stand-in: a clean checkout at its own rev, with a quantizer built from it
    e.checkout = tmp_path / "transcribe.cpp"
    script(e.checkout / "scripts" / "convert-parakeet.py", "# the fake NeMo interpreter dispatches on this name\n")
    git(e.checkout, "init", "-q")
    git(e.checkout, "add", ".")
    git(e.checkout, "commit", "-q", "-m", "converter")
    e.rev = git(e.checkout, "rev-parse", "HEAD")
    script(e.checkout / "build" / "bin" / "transcribe-quantize", QUANTIZE.format(python=sys.executable))
    (e.checkout / "build" / ".healing-built-rev").write_text(e.rev + "\n")
    rev_file = tmp_path / "transcribe-cpp.rev"
    rev_file.write_text(e.rev + "\n")
    monkeypatch.setattr(export, "REV_FILE", rev_file)

    dispatcher = script(tmp_path / "nemo.py", NEMO.format(healing=healing_src, kv=dict(BASE_KV), tensors=TENSORS))
    nemo = script(tmp_path / "nemo-python", f"#!/bin/sh\nexec {sys.executable} {dispatcher} \"$@\"\n")
    monkeypatch.setenv("HEALING_NEMO_PYTHON", str(nemo))
    e.base_nemo = tmp_path / "base.nemo"
    e.base_nemo.write_bytes(b"the base NeMo checkpoint")
    monkeypatch.setenv("HEALING_BASE_NEMO", str(e.base_nemo))
    base_sha = hashlib.sha256(e.base_nemo.read_bytes()).hexdigest()

    e.config = tmp_path / "lora.yaml"
    defaults = (Path(export.__file__).resolve().parents[1] / "train" / "configs" / "lora.yaml").read_text()
    e.config.write_text(defaults.replace("ec23ed9150c8fde49072c3e2d61678ab903dbcef389d658db833420cbc1da35b", base_sha))
    e.delta = write_delta(tmp_path / "lora_delta.safetensors", base_sha=base_sha)

    e.seen = tmp_path / "hex-import-support"
    e.hex = script(tmp_path / "hex", HEX.format(python=sys.executable, healing=healing_src, seen=str(e.seen)))
    base_gguf = tmp_path / "base-Q8_0.gguf"
    gguf.write(base_gguf, BASE_KV, {n: b"\1" * 64 for n in TENSORS})
    assert main(["models", "register-base", "--gguf", str(base_gguf), "--model-id", "parakeet-unified-en-0.6b",
                 "--arch", "parakeet-unified"]) == 0
    rows = [("clip-1", "a" * 64, "b" * 64, 1.0), ("clip-2", "c" * 64, "d" * 64, 0.5)]
    e.snapshot = snapshots.write(snapshots.Snapshot(snapshots.digest(rows), rows)).name.split(".")[0]
    e.tmp = tmp_path / "export-tmp"
    return e


def run_export(env, **kwargs):
    return export.export(env.delta, env.config, env.snapshot, converter_dir=env.checkout, hex_binary=env.hex,
                         tmp_dir=env.tmp, say=lambda _: None, **kwargs)


def leftovers(env) -> list[str]:
    return sorted(p.name for p in env.tmp.iterdir()) if env.tmp.exists() else []


def test_export_registers_a_candidate_named_after_its_version(env):
    result = run_export(env)
    assert result["version"] == "v1"
    version = registry.load().get("v1")
    assert version.state == "candidate" and version.record["parent"] == "v0"
    assert version.record["dataset"]["snapshot_sha256"] == env.snapshot
    assert version.record["converter"]["rev"] == env.rev
    assert version.record["converter"]["command"][-2:] == ["--name", "hex-healing v1"]
    assert version.record["config"]["text"] == env.config.read_text()
    stored = objects.path(result["gguf_sha256"])
    header = gguf.read(stored)
    assert header.string("general.name") == "hex-healing v1"
    assert header.tensors == TENSORS
    assert objects.has(result["delta_sha256"])
    assert result["import"] == f"imported\t{objects.hex_model_id(result['gguf_sha256'])}\thex-healing v1"
    # the import ran in a throwaway data folder, not Hex's (G-8), and it is gone
    isolated = Path(env.seen.read_text())
    assert isolated != env.support and not isolated.exists() and not env.support.exists()
    assert leftovers(env) == []


def test_restore_replays_the_recorded_export(env):
    result = run_export(env)
    objects.remove(result["gguf_sha256"])
    event = restore.restore("v1", parity=lambda *_: pytest.fail("the rebuild must match by hash"))
    assert event["method"] == "sha256" and objects.has(result["gguf_sha256"])


def test_converter_failure_names_step(env, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "convert")
    with pytest.raises(export.ExportError) as caught:
        run_export(env)
    message = str(caught.value)
    assert "convert failed (exit 1)" in message
    assert message.endswith("RuntimeError: the convert broke here")
    assert "v1" not in registry.load().versions


@pytest.mark.parametrize("key", ["stt.variant", "general.languages"])
def test_header_check_refuses_missing_variant(env, monkeypatch, key):
    monkeypatch.setenv("FAKE_DROP", key)
    with pytest.raises(export.ExportError, match=f"no {key}; not registered"):
        run_export(env)
    assert "v1" not in registry.load().versions
    assert not env.seen.exists()  # refused before Hex was asked


@pytest.mark.parametrize("step", ["merge", "quantize"])
def test_cleans_temp_on_failure(env, monkeypatch, step):
    monkeypatch.setenv("FAKE_FAIL", step)
    with pytest.raises(export.ExportError, match=f"{step} failed"):
        run_export(env)
    assert leftovers(env) == []
    assert "v1" not in registry.load().versions


def test_tensor_names_must_equal_the_base(env, monkeypatch):
    monkeypatch.setenv("FAKE_EXTRA", "1")
    with pytest.raises(export.ExportError, match=r"tensor-names: .*extra \['extra.w'\]"):
        run_export(env)
    assert leftovers(env) == []


def test_a_delta_trained_at_another_rank_is_refused_before_building(env, tmp_path):
    base_sha = hashlib.sha256(env.base_nemo.read_bytes()).hexdigest()
    env.delta = write_delta(tmp_path / "other.safetensors", rank=8, base_sha=base_sha)
    with pytest.raises(export.ExportError, match="trained at rank 8"):
        run_export(env)
    assert not env.tmp.exists()


def test_a_delta_from_another_base_is_refused(env, tmp_path):
    env.delta = write_delta(tmp_path / "other.safetensors", base_sha="f" * 64)
    with pytest.raises(export.ExportError, match="trained on base ffffffffffff"):
        run_export(env)


def test_a_converter_checkout_off_the_pinned_rev_is_refused(env):
    (env.checkout / "scripts" / "convert-parakeet.py").write_text("# patched\n")
    git(env.checkout, "commit", "-qam", "newer converter")
    with pytest.raises(export.ExportError, match="not the recorded"):
        run_export(env)


def test_register_refuses_a_version_name_taken_while_building(env):
    with pytest.raises(registry.RegistryError, match="another version took v5"):
        registry.register({"parent": "v0"}, expect="v5")


def test_cli_prints_the_candidate_version(env, capsys):
    code = main(["export", str(env.delta), "--config", str(env.config), "--snapshot", env.snapshot,
                 "--converter-dir", str(env.checkout), "--hex", str(env.hex), "--tmp-dir", str(env.tmp)])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert out.out.strip() == "v1"
    assert "isolated import: imported" in out.err


def test_cli_converter_failure_exits_1_naming_the_step(env, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "convert")
    code = main(["export", str(env.delta), "--config", str(env.config), "--snapshot", env.snapshot,
                 "--converter-dir", str(env.checkout), "--hex", str(env.hex), "--tmp-dir", str(env.tmp)])
    assert code == 1
    assert "convert failed (exit 1)" in capsys.readouterr().err


def test_set_string_keeps_the_tensor_data(tmp_path):
    source, target = tmp_path / "a.gguf", tmp_path / "b.gguf"
    gguf.write(source, {**BASE_KV, "general.alignment": 64}, {n: bytes(range(64)) for n in TENSORS})
    gguf.set_string(source, target, "general.name", "hex-healing v12 with a longer name")
    before, after = gguf.read(source), gguf.read(target)
    assert after.string("general.name") == "hex-healing v12 with a longer name"
    assert list(after.kv) == list(before.kv) and after.tensors == before.tensors
    assert after.data_start % 64 == 0
    assert source.read_bytes()[before.data_start:] == target.read_bytes()[after.data_start:]
    gguf.set_string(source, target, "hex.extra", "appended")
    assert gguf.read(target).kv["hex.extra"] == "appended" and gguf.read(target).tensors == TENSORS


def test_the_pinned_rev_is_one_full_sha():
    assert len(export.pinned_rev()) == 40


def test_a_delta_replaced_during_the_build_does_not_change_what_is_registered(env, monkeypatch):
    """Export builds, hashes and publishes one frozen copy of the delta, so a file swapped in under the
    input path mid-build is never recorded against a GGUF built from the old one."""
    original = objects.sha256_file(env.delta)
    real = restore.run_recorded

    def swap_after_merge(converter, inputs):
        real(converter, inputs)
        env.delta.write_bytes(env.delta.read_bytes() + b"\0")  # replaced after the merge read it

    monkeypatch.setattr(restore, "run_recorded", swap_after_merge)
    result = run_export(env)
    assert result["delta_sha256"] == original
    assert registry.load().get("v1").record["artifacts"]["lora_delta"] == original
