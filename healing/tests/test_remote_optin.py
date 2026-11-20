"""T6.2: `healing train --remote` around a stubbed `kaggle` CLI (no network, no Kaggle account).

The stub replaces `subprocess.run` inside `kaggle_runner`, the one boundary the runner reaches Kaggle
through. It acts out the Kaggle CLI's commands (quota, datasets create/status/delete, kernels
push/status/output/delete) and records each call with the metadata files it was handed.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile

from healing import cli
from healing.remote import kaggle_runner
from healing.train import config as config_module
from healing.train.manifest import LISTING, MANIFEST, _listing

TEXT = "please call stella ask her to bring these things"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "kaggle"))
    monkeypatch.setenv("HEALING_KAGGLE", "/stub/kaggle")
    for name in ("HEALING_ROUNDS_DIR", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kaggle_runner, "POLL_SECONDS", 0)
    return tmp_path


@pytest.fixture
def token(home):
    folder = home / "kaggle"
    folder.mkdir()
    (folder / "kaggle.json").write_text(json.dumps({"username": "someone", "key": "KGAT_" + "0" * 32}))
    return folder


@pytest.fixture
def bundle(home):
    path = home / "home/train/split-1"
    (path / "audio").mkdir(parents=True)
    lines = []
    for i in range(3):
        soundfile.write(path / f"audio/c{i}.wav", np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")
        lines.append(json.dumps({"audio_filepath": f"audio/c{i}.wav", "duration": 1.0, "text": f"{TEXT} {i}",
                                 "weight": 1.0, "id": f"c{i}", "cluster": "k"}) + "\n")
    (path / MANIFEST).write_text("".join(lines))
    (path / LISTING).write_bytes(_listing(path, [MANIFEST] + [f"audio/c{i}.wav" for i in range(3)]))
    return path


def delta_bytes(names: list[str]) -> bytes:
    header, offset = {}, 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header).encode()
    return len(blob).to_bytes(8, "little") + blob + bytes(offset)


class StubKaggle:
    """The `kaggle` CLI at its subprocess boundary. `status` is what `kernels status` reports once the
    kernel ran; `fail_delete` makes the deletes fail (and the things stay listed)."""

    def __init__(self, status: str = "complete", fail_delete: bool = False, interrupt_poll: bool = False):
        self.status, self.fail_delete, self.interrupt_poll = status, fail_delete, interrupt_poll
        self.calls: list[list[str]] = []
        self.metadata: dict[str, dict] = {}
        self.notebook: str | None = None
        self.live: set[str] = set()

    def __call__(self, argv, **kwargs):
        assert argv[0] == "/stub/kaggle"
        args = list(argv[1:])
        self.calls.append(args)
        env = kwargs.get("env") or {}
        assert env.get("KAGGLE_API_TOKEN") == "KGAT_" + "0" * 32
        out, code = "", 0
        if args[:1] == ["quota"]:
            out = "resource,used,remaining,total,refreshAt\nGPU,1.00h,29.00h,30.00h,2026-11-21T00:00:00+00:00\n"
        elif args[:2] == ["datasets", "create"]:
            folder = Path(args[args.index("-p") + 1])
            self.metadata["dataset"] = json.loads((folder / "dataset-metadata.json").read_text())
            self.uploaded = sorted(p.name for p in folder.iterdir())
            self.live.add(self.metadata["dataset"]["id"])
        elif args[:2] == ["datasets", "status"]:
            out = "ready\n"
        elif args[:2] == ["kernels", "push"]:
            folder = Path(args[args.index("-p") + 1])
            meta = json.loads((folder / "kernel-metadata.json").read_text())
            self.metadata["kernel"] = meta
            self.notebook = (folder / meta["code_file"]).read_text()
            self.live.add(meta["id"])
        elif args[:2] == ["kernels", "status"]:
            if self.interrupt_poll:
                raise KeyboardInterrupt
            out = f'{args[2]} has status "KernelWorkerStatus.{self.status.upper()}"\n'
        elif args[:2] == ["kernels", "output"]:
            self.write_output(Path(args[args.index("-p") + 1]), args[2].split("/")[1])
        elif args[1:2] == ["delete"]:
            if self.fail_delete:
                code, out = 1, "403 Forbidden"
            else:
                self.live.discard(args[2])
        elif args[1:2] == ["list"]:
            out = "\n".join(sorted(self.live)) + "\n"
        return subprocess.CompletedProcess(argv, code, out, "")

    def write_output(self, folder: Path, slug: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        log = [{"stream_name": "stdout", "time": 1.0, "data": "train: step 1 loss 1.2345 (3s)\n"},
               {"stream_name": "stdout", "time": 2.0, "data": f"WER sample: {TEXT.upper()}!\n"},
               {"stream_name": "stderr", "time": 3.0, "data": "ref: ... ask her to bring ...\n"}]
        (folder / f"{slug}.log").write_text(json.dumps(log))
        if self.status != "complete":
            (folder / "result.json").write_text(json.dumps({"code": 1, "stage": "train", "seconds": 600.0}))
            return
        targets = ["encoder.layers.0.self_attn.linear_q", "joint.pred"]
        names = [t + s for t in targets for s in (".lora_a", ".lora_b")]
        (folder / "lora_delta.safetensors").write_bytes(delta_bytes(names))
        (folder / "train_log.json").write_text(json.dumps({
            "entries": [{"step": 20, "train_loss": 0.5, "val_loss": 0.6, "val_wer": 0.1}], "steps": 20,
            "stopped": "max_steps", "resumes": 0, "wall_seconds": 120.0, "peak_rss_bytes": 1 << 30}))
        (folder / "env.json").write_text(json.dumps({"python": "3.12.3", "torch": "2.11.0+cu129", "nemo": "2.8.0rc0"}))
        (folder / "targets.json").write_text(json.dumps({"groups": {"encoder_attention": targets[:1], "joint": targets[1:]},
                                                         "trainable_parameters": 64, "frozen_parameters": 1000}))
        (folder / "result.json").write_text(json.dumps({"code": 0, "stage": "done", "seconds": 900.0, "gpu": "Tesla T4"}))


def remote(bundle, *extra) -> int:
    return cli.main(["train", "--remote", "--smoke", "--bundle", str(bundle), *extra])


def runs(home) -> list[dict]:
    path = home / "home/remote/runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_refuses_without_flag(home, token, bundle, capsys):
    assert remote(bundle) == 5
    assert "refusing to upload personal audio without --allow-upload" in capsys.readouterr().err


def test_zero_network_without_flag(home, token, bundle, monkeypatch):
    def no(*args, **kwargs):
        raise AssertionError("network or a subprocess without --allow-upload")

    for target, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo"),
                         (subprocess, "run"), (subprocess, "Popen")):
        monkeypatch.setattr(target, name, no)
    assert remote(bundle) == 5
    assert not (home / "home/remote").exists()


def test_missing_token_names_the_token_page(home, bundle, capsys):
    stub = StubKaggle()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(kaggle_runner.subprocess, "run", stub)
        assert remote(bundle, "--allow-upload") == 6
    assert "Create New Token" in capsys.readouterr().err and stub.calls == []


def test_log_filter_drops_transcript_lines(bundle):
    grams = kaggle_runner.text_grams(kaggle_runner.manifest_texts(bundle))
    lines = ["train: step 10 loss 0.4321 (12s)",
             "train: step 10 train_loss 0.5 val_loss 0.6 val_wer 0.25",
             f"predicted: {TEXT}",
             "reference: Please, call Stella!",  # case and punctuation don't hide it
             "...bring these things 2"]
    kept, dropped = kaggle_runner.scrub(lines, grams)
    assert kept == lines[:2] and dropped == 3
    # the notebook filters the trainer's stdout with the same function before Kaggle keeps it
    notebook = kaggle_runner.render_notebook({"key": "k"})
    assert "def scrub(" in notebook and "def leaks(" in notebook


def test_dataset_and_kernel_metadata_are_private(home, token, bundle, monkeypatch, capsys):
    stub = StubKaggle()
    monkeypatch.setattr(kaggle_runner.subprocess, "run", stub)
    assert remote(bundle, "--allow-upload", "--max-minutes", "30") == 0, capsys.readouterr()
    dataset, kernel = stub.metadata["dataset"], stub.metadata["kernel"]
    assert dataset["isPrivate"] is True and kernel["is_private"] is True
    assert not any(flag in call for call in stub.calls for flag in ("-u", "--public"))
    assert kernel["dataset_sources"] == [dataset["id"]] and kernel["enable_gpu"] is True
    assert dataset["id"].startswith("someone/healing-") and kernel["id"].startswith("someone/healing-")
    assert stub.uploaded == ["bundle.tar", "dataset-metadata.json"]
    # the notebook carries the job and the trainer, never the manifest's text
    assert stub.notebook and TEXT not in stub.notebook
    job = json.loads(base64.b64decode(stub.notebook.split('JOB_B64 = "')[1].split('"')[0]))
    assert job["config"]["device"] == "cuda" and job["config"]["max_steps"] == 20
    push = next(c for c in stub.calls if c[:2] == ["kernels", "push"])
    assert push[push.index("-t") + 1] == "1800"
    # cleaned up, recorded, and the delta is in the round's folder
    assert stub.live == set()
    (row,) = runs(home)
    assert row["deleted_remote"] is True and row["error"] is None and row["provider"] == "kaggle"
    assert row["dataset"] == dataset["id"] and row["kernel"] == kernel["id"] and row["uploaded_bytes"] > 48_000
    assert row["gpu_minutes"] == 15.0 and row["versions"]["torch"] == "2.11.0+cu129"
    (folder,) = [p for p in (home / "home/rounds").iterdir() if p.is_dir()]
    record = json.loads((folder / "out/run.json").read_text())
    assert record["status"] == "complete" and record["trainer"]["runner"] == "kaggle"
    assert (folder / "out/lora_delta.safetensors").is_file()
    log = (home / f"home/remote/{row['run_id']}/notebook.log").read_text()
    assert "step 1 loss" in log and "stella" not in log.lower() and "bring" not in log
    assert not (home / f"home/remote/{row['run_id']}/upload").exists()


def test_deletes_dataset_and_kernel_on_failure(home, token, bundle, monkeypatch, capsys):
    stub = StubKaggle(status="error")
    monkeypatch.setattr(kaggle_runner.subprocess, "run", stub)
    assert remote(bundle, "--allow-upload") not in (0, 5, 6, 7)
    deletes = [c[:3] for c in stub.calls if c[1:2] == ["delete"]]
    assert {d[0] for d in deletes} == {"datasets", "kernels"} and stub.live == set()
    (row,) = runs(home)
    assert row["deleted_remote"] is True and row["error"]
    assert TEXT not in capsys.readouterr().out


def test_ctrl_c_still_deletes(home, token, bundle, monkeypatch):
    stub = StubKaggle(interrupt_poll=True)
    monkeypatch.setattr(kaggle_runner.subprocess, "run", stub)
    assert remote(bundle, "--allow-upload") == 130
    assert stub.live == set() and runs(home)[0]["deleted_remote"] is True


def test_failed_delete_prints_the_commands(home, token, bundle, monkeypatch, capsys):
    stub = StubKaggle(status="error", fail_delete=True)
    monkeypatch.setattr(kaggle_runner.subprocess, "run", stub)
    assert remote(bundle, "--allow-upload") != 0
    (row,) = runs(home)
    assert row["deleted_remote"] is False
    printed = capsys.readouterr().err
    assert f"kaggle datasets delete -y {row['dataset']}" in printed
    assert f"kaggle kernels delete -y {row['kernel']}" in printed


def test_quota_used_up_exits_7_before_upload(home, token, bundle, monkeypatch, capsys):
    stub = StubKaggle()

    def spent(argv, **kwargs):
        if argv[1:2] == ["quota"]:
            return subprocess.CompletedProcess(argv, 0, "resource,used,remaining,total,refreshAt\n"
                                               "GPU,30.00h,0.00h,30.00h,2026-11-21T00:00:00+00:00\n", "")
        return stub(argv, **kwargs)

    monkeypatch.setattr(kaggle_runner.subprocess, "run", spent)
    assert remote(bundle, "--allow-upload") == 7
    assert "2026-11-21" in capsys.readouterr().err
    assert not any(c[:2] == ["datasets", "create"] for c in stub.calls)


def test_remote_config_is_the_local_one_on_cuda(home):
    config, effective = kaggle_runner.remote_config(config_module.DEFAULT, smoke=True)
    assert config["device"] == "cuda" and config["smoke"] is True and config_module.parse(effective)["device"] == "cuda"
