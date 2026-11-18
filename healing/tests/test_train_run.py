"""T6.5: `healing train --local` around a fake trainer (no NeMo, no torch).

The fake stands in for NeMo's Python: it reads the job `run.py` wrote and acts out one trainer
behaviour (finish, stop for a resume, leak text, write a stray tensor), so these tests cover the
inputs `run.py` checks, the round folder, resume and the run record.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile

from healing import cli
from healing.train import config as config_module, run
from healing.train.manifest import LISTING, MANIFEST, TrainError, _listing

FAKE = textwrap.dedent('''
    import json, os, struct, sys
    from pathlib import Path
    args = sys.argv[1:]
    job = json.loads(Path(args[args.index("--job") + 1]).read_text())
    work, out = Path(job["work_dir"]), Path(job["out_dir"])
    mode = os.environ.get("FAKE_MODE", "ok")
    calls = work / "calls"
    calls.write_text(calls.read_text() + "x" if calls.exists() else "x")
    (work / "env.json").write_text(json.dumps({"python": "3.11.0", "torch": "9.9", "nemo": "8.8"}))
    targets = ["encoder.layers.0.self_attn.linear_q", "joint.pred"]
    (work / "targets.json").write_text(json.dumps({"groups": {"encoder_attention": targets[:1], "joint": targets[1:]},
                                                   "trainable_parameters": 64, "frozen_parameters": 1000}))
    resumed = (work / "checkpoint.pt").exists()
    if mode == "interrupt" and not resumed:
        (work / "checkpoint.pt").write_text("step 5")
        print("train: interrupted", flush=True)
        sys.exit(130)
    names = [t + s for t in targets for s in (".lora_a", ".lora_b")] + (["joint.pred.weight"] if mode == "stray" else [])
    header, offset = {}, 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header).encode()
    out.mkdir(exist_ok=True)
    (out / "lora_delta.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    log = {"entries": [{"step": 20, "train_loss": 0.5, "val_loss": 0.6, "val_wer": 0.1}], "steps": 20,
           "stopped": "max_steps", "resumes": int(resumed), "wall_seconds": 12.0, "peak_rss_bytes": 1 << 30}
    if mode == "leak":
        log["entries"][0]["note"] = "phrase number one"
    (out / "train_log.json").write_text(json.dumps(log))
''')


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    for name in ("HEALING_ROUNDS_DIR", "HEALING_NEMO_PYTHON", "HEALING_BASE_NEMO", "MODEL_CACHE", "FAKE_MODE"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def setup(home):
    """A bundle, a base checkpoint, a config pinning it and a fake NeMo Python."""
    bundle = home / "home/train/split-1"
    (bundle / "audio").mkdir(parents=True)
    lines = []
    for i in range(3):
        soundfile.write(bundle / f"audio/c{i}.wav", np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")
        lines.append(json.dumps({"audio_filepath": f"audio/c{i}.wav", "duration": 1.0, "text": "phrase number one",
                                 "weight": 1.0, "id": f"c{i}", "cluster": "k"}) + "\n")
    (bundle / MANIFEST).write_text("".join(lines))
    (bundle / LISTING).write_bytes(_listing(bundle, [MANIFEST] + [f"audio/c{i}.wav" for i in range(3)]))
    base = home / "base.nemo"
    base.write_bytes(b"base weights")
    text = config_module.DEFAULT.read_text().replace(
        "ec23ed9150c8fde49072c3e2d61678ab903dbcef389d658db833420cbc1da35b", hashlib.sha256(b"base weights").hexdigest())
    config = home / "lora.yaml"
    config.write_text(text)
    fake = home / "fake_trainer.py"
    fake.write_text(FAKE)
    python = home / "nemo-python"
    python.write_text(f"#!/bin/sh\nshift; shift\nexec {sys.executable} {fake} \"$@\"\n")  # drop `-I lora.py`
    python.chmod(0o755)
    return {"bundle": bundle, "base": base, "config": config, "python": python}


def train(setup, **kwargs):
    lines: list[str] = []
    args = {"config_path": setup["config"], "python": setup["python"], "base": setup["base"], "keep_awake": False,
            "say": lines.append, **kwargs}
    code = run.train_local(setup["bundle"], **args)
    return code, lines


def round_folder(home: Path) -> Path:
    (folder,) = [p for p in (home / "home/rounds").iterdir() if p.is_dir()]
    return folder


def test_smoke_round_writes_delta_and_record(setup, home):
    code, lines = train(setup, smoke=True)
    assert code == 0, lines
    folder = round_folder(home)
    assert folder.name.startswith("split-1-")
    record = json.loads((folder / "out/run.json").read_text())
    assert record["status"] == "complete" and record["smoke"] is True
    assert record["trainer"]["torch"] == "9.9" and record["trainer"]["nemo"] == "8.8"
    assert record["trainer"]["python"] == str(setup["python"]) and record["trainer"]["python_version"] == "3.11.0"
    assert record["base"]["sha256"] == hashlib.sha256(b"base weights").hexdigest()
    assert record["outputs"]["delta"]["tensors"] == 4 and record["targets"] == {"encoder_attention": 1, "joint": 1}
    effective = (folder / "out/lora.yaml").read_text()
    assert record["config"]["sha256"] == hashlib.sha256(effective.encode()).hexdigest()
    parsed = config_module.parse(effective)
    assert parsed["max_steps"] == 20 and parsed["max_clips"] == 50 and parsed["smoke"] is True
    job = json.loads((folder / "work/trainer.json").read_text())
    assert job["config"]["checkpoint_every"] == 5 and job["bundle"] == str(setup["bundle"].resolve())
    assert any("healing export" in line for line in lines)


def test_resumes_in_the_same_round_folder(setup, home, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "interrupt")
    code, lines = train(setup, smoke=True)
    assert code == 130 and "rerun the same command" in lines[-1]
    folder = round_folder(home)
    assert not (folder / "out/run.json").exists()
    code, lines = train(setup, smoke=True)
    assert code == 0 and "resuming" in lines[0]
    assert round_folder(home) == folder
    assert json.loads((folder / "out/run.json").read_text())["resumes"] == 1
    code, lines = train(setup, smoke=True)  # a finished round is not trained again
    assert code == 0 and lines == [f"already trained: {folder / 'out/lora_delta.safetensors'}"]
    assert (folder / "work/calls").read_text() == "xx"


def test_a_changed_config_is_a_new_round(setup, home):
    assert train(setup, smoke=True)[0] == 0
    assert train(setup, smoke=False)[0] == 0
    assert len([p for p in (home / "home/rounds").iterdir() if p.is_dir()]) == 2


def test_delta_with_a_non_lora_tensor_is_refused(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "stray")
    with pytest.raises(TrainError, match="joint.pred.weight is not a LoRA tensor"):
        train(setup)


def test_no_text_in_train_log(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "leak")
    with pytest.raises(TrainError, match="numbers only"):
        train(setup)
    assert run.log_strings({"stopped": "early_stopping", "entries": [{"step": 1, "val_wer": None}]}) == []
    assert run.log_strings({"entries": [{"x": "phrase"}]}) == ["phrase"]


def test_inputs_are_checked_before_training(setup, home):
    with pytest.raises(TrainError, match="X-1"):
        train(setup, base=home / "missing.nemo")
    with pytest.raises(TrainError, match="NeMo's Python"):
        train(setup, python=home / "no-python")
    other = home / "other.nemo"
    other.write_bytes(b"other weights")
    with pytest.raises(TrainError, match="the config pins"):
        train(setup, base=other)
    (setup["bundle"] / "audio/c0.wav").write_bytes(b"changed")
    with pytest.raises(TrainError, match="changed since the bundle was written"):
        train(setup)


def test_base_hash_is_cached(setup, home, monkeypatch):
    assert train(setup)[0] == 0
    calls = []
    monkeypatch.setattr(run, "sha256_file", lambda p: calls.append(p) or "f" * 64)
    train(setup)
    assert str(setup["base"]) not in map(str, calls)  # only outputs are hashed, never the base again


def test_default_paths_follow_the_environment(home, monkeypatch):
    config, _ = config_module.load(config_module.DEFAULT)
    paths = run.default_paths(config)
    assert paths.rounds == home / "home/rounds"
    assert paths.base.parts[-3:] == ("parakeet-unified-en-0.6b", "fe53cd885760c96b6a5f51a0bfd362cb4584a98b",
                                     "parakeet-unified-en-0.6b.nemo")
    monkeypatch.setenv("HEALING_ROUNDS_DIR", "/Volumes/x/rounds")
    monkeypatch.setenv("HEALING_NEMO_PYTHON", "/opt/nemo/python")
    monkeypatch.setenv("HEALING_BASE_NEMO", "/opt/base.nemo")
    paths = run.default_paths(config)
    assert (str(paths.rounds), str(paths.python), str(paths.base)) == ("/Volumes/x/rounds", "/opt/nemo/python", "/opt/base.nemo")


def test_cli_train_local(setup, home, capsys):
    argv = ["train", "--local", "--smoke", "--bundle", str(setup["bundle"]), "--config", str(setup["config"]),
            "--python", str(setup["python"]), "--base", str(setup["base"])]
    assert cli.main(argv) == 0
    assert "delta " in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["train"])
    with pytest.raises(SystemExit):
        cli.main(["train", "--local"])
    assert cli.main(argv[:-2] + ["--base", str(home / "nope")]) == 1
    assert "X-1" in capsys.readouterr().err


# --- the config ---------------------------------------------------------------------------


def test_default_config_is_the_spec():
    config, text = config_module.load(config_module.DEFAULT)
    assert (config["rank"], config["alpha"], config["dropout"], config["lr"]) == (16, 32, 0.05, 1e-4)
    assert (config["epochs"], config["batch_max_clips"], config["batch_seconds"]) == (3, 16, 60)
    assert config["targets"] == ["encoder_attention", "encoder_feed_forward", "joint"]
    assert config["validation_share"] == 0.05 and config["checkpoint_every"] == 200 and config["max_rss_gb"] == 12
    assert "smoke" not in config and config_module.parse(text) == config
    smoke, _ = config_module.load(config_module.DEFAULT, smoke=True)
    assert (smoke["max_steps"], smoke["max_clips"]) == (20, 50)


def test_config_parser_round_trips_and_refuses_what_it_cannot_read():
    text = 'a: 1\nb: 2.5e-3\nc: "x: y"\nd: [p, q]\ne:\n  f: true\n  g: null\nh: plain # note\n'
    parsed = config_module.parse(text)
    assert parsed == {"a": 1, "b": 0.0025, "c": "x: y", "d": ["p", "q"], "e": {"f": True, "g": None}, "h": "plain"}
    assert config_module.parse(config_module.dump(parsed)) == parsed
    for bad in ("a: 1\na: 2\n", "- item\n", "a: [1, 2\n", "a: {b: 1}\n", "\ta: 1\n"):
        with pytest.raises(config_module.ConfigError):
            config_module.parse(bad)
    with pytest.raises(config_module.ConfigError, match="lacks"):
        config_module.check({"rank": 16})
