"""`healing train --remote --allow-upload`: one LoRA round on Kaggle's free GPUs (T6.2, D-34).

This is the only place personal data leaves the Mac (G-3), so it is explicit, minimal and cleaned up:

1. Without `--allow-upload` it refuses with exit 5 before anything else: no credential read, no
   subprocess, no network.
2. It uploads the prepared bundle (the train split only: WAVs and the manifest) as one `bundle.tar`
   in a new **private** Kaggle dataset, and pushes a **private** script kernel rendered from
   `notebook.py.tmpl` that builds T0.4's NeMo environment, runs `healing.train.lora` on a T4 and writes
   the delta and the numbers-only train log.
3. It polls the kernel, downloads its output, and checks and records the delta exactly as a local
   round does (`run._finish`), in the same round folder the same inputs would train locally.
4. **Whatever happens** (a failure, a timeout, Ctrl-C) a `finally` deletes the dataset and the
   kernel, removes the staged copy, and appends `$HEALING_HOME/remote/runs.jsonl`. A delete that fails
   is recorded as `deleted_remote: false` and the exact `kaggle … delete` commands are printed.

Kaggle is reached only through the pinned `kaggle` CLI (`KAGGLE_COMMAND`, run with `uvx`, D-80;
`$HEALING_KAGGLE` names another executable), the subprocess boundary the tests stub. Credentials are
Kaggle's own: `$KAGGLE_API_TOKEN`, else `~/.kaggle/access_token`, else `~/.kaggle/kaggle.json`
(`$KAGGLE_CONFIG_DIR` moves the folder). They go to the CLI in its environment only; healing never
stores or prints them.

Kaggle keeps a kernel's log until the kernel is deleted, and the downloaded log is kept beside the
run. Both pass `scrub`, which drops any line holding manifest text (a 3-word run of any clip's text,
or a whole shorter text); the notebook runs the same function on the trainer's stdout before Kaggle
sees it.

Exit codes: 0 done, 5 no `--allow-upload`, 6 no Kaggle token, 7 the weekly GPU quota is used up,
130 interrupted, 1 any other failure.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from healing.paths import healing_home
from healing.registry.objects import owner_dir
from healing.train import config as config_module, lora, run as train_run
from healing.train.manifest import LISTING, MANIFEST, TrainError, verify_bundle

KAGGLE_CLI_VERSION = "2.2.5"
# The CLI imports `six` without declaring it (2.2.4 and 2.2.5), so it is pinned beside it.
KAGGLE_COMMAND = ["uvx", "-q", "--python", "3.12", "--from", f"kaggle=={KAGGLE_CLI_VERSION}", "--with", "six==1.17.0", "kaggle"]
ACCELERATOR = "NvidiaTeslaT4"
# What a round changes on Kaggle's machine (D-81): the GPU, its 4 CPUs and its ~30 GB of RAM. They are
# part of the effective config, so they are hashed and recorded like every other value.
REMOTE_OVERRIDES = {"device": "cuda", "threads": 4, "max_rss_gb": 24}
TEMPLATE = Path(__file__).with_name("notebook.py.tmpl")
TRANSCRIBE_CPP_REPO = "https://github.com/anomalyco/transcribe.cpp"
UV_VERSION = "0.11.6"
NEMO_PYTHON = "3.12"
TOKEN_PAGE = "kaggle.com › Settings › API › Create New Token"
POLL_SECONDS = 30
QUEUE_MINUTES = 60  # beyond --max-minutes: queueing, the NeMo install and the base download
DATASET_READY_SECONDS = 900
GRAM = 3

REFUSE, NO_TOKEN, QUOTA, FAILED, INTERRUPTED = 5, 6, 7, 1, 130


class RemoteError(RuntimeError):
    pass


class QuotaUsedUp(RemoteError):
    pass


# --- the log filter (also copied into the notebook: standard library only, no globals)

def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def text_grams(texts, n=3):
    """Every run of `n` consecutive words of every text, and each text shorter than `n` whole."""
    grams = set()
    for text in texts:
        w = words(text)
        k = min(n, len(w))
        grams.update(tuple(w[i:i + k]) for i in range(len(w) - k + 1) if k)
    return grams


def leaks(line, grams, n=3):
    w = words(line)
    return any(tuple(w[i:i + k]) in grams for k in range(1, n + 1) for i in range(len(w) - k + 1))


def scrub(lines, grams, n=3):
    """The lines that hold no manifest text, and how many were dropped."""
    kept = [line for line in lines if not leaks(line, grams, n)]
    return kept, len(lines) - len(kept)


def manifest_texts(bundle: Path) -> list[str]:
    return [json.loads(line)["text"] for line in (bundle / MANIFEST).read_text().splitlines() if line]


# --- credentials and the CLI

@dataclass
class Credentials:
    username: str
    env: dict = field(repr=False)  # for the CLI's environment only; never printed


def config_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")


def credentials() -> Credentials | None:
    env, folder = os.environ, config_dir()
    try:
        stored = json.loads((folder / "kaggle.json").read_text())
    except (OSError, ValueError):
        stored = {}
    username = env.get("KAGGLE_USERNAME") or stored.get("username")
    token = env.get("KAGGLE_API_TOKEN")
    if not token:
        try:
            token = (folder / "access_token").read_text().strip() or None
        except OSError:
            token = None
    key = env.get("KAGGLE_KEY") or stored.get("key")
    if not token and isinstance(key, str) and key.startswith("KGAT_"):
        token = key  # a new-style API token saved as kaggle.json's key: the CLI reads it only from the env
    if not username:
        return None
    if token:
        return Credentials(username, {"KAGGLE_API_TOKEN": token})
    if key:
        return Credentials(username, {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key})
    return None


def kaggle_command() -> list[str]:
    named = os.environ.get("HEALING_KAGGLE")
    if named:
        return [named]
    if shutil.which(KAGGLE_COMMAND[0]) is None:
        raise RemoteError("`uvx` (uv) is not on PATH; it runs the pinned kaggle CLI")
    return list(KAGGLE_COMMAND)


class Kaggle:
    def __init__(self, creds: Credentials):
        self.creds, self.command = creds, kaggle_command()

    def __call__(self, *args: str, check: bool = True, timeout: float = 1800) -> subprocess.CompletedProcess:
        done = subprocess.run([*self.command, *args], env={**os.environ, **self.creds.env}, capture_output=True,
                              text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if check and done.returncode != 0:
            tail = " ".join(f"{done.stdout or ''} {done.stderr or ''}".split())[-300:]
            raise RemoteError(f"kaggle {' '.join(args[:2])} failed (exit {done.returncode}): {tail}")
        return done


def gpu_quota(kaggle: Kaggle) -> tuple[float, str] | None:
    """Minutes of GPU left this week and when the quota resets, or None if Kaggle didn't say."""
    done = kaggle("quota", "-v", check=False, timeout=300)
    for row in csv.DictReader(io.StringIO(done.stdout or "")):
        if row.get("resource") == "GPU":
            try:
                return float(row["remaining"].rstrip("h")) * 60, row.get("refreshAt") or "unknown"
            except (KeyError, ValueError):
                return None
    return None


# --- the round

def remote_config(path: Path, smoke: bool = False) -> tuple[dict, str]:
    """The local config with `REMOTE_OVERRIDES` applied: the effective config a remote round hashes."""
    config, _ = config_module.load(path, smoke=smoke)
    config.update(REMOTE_OVERRIDES)
    config_module.check(config, str(path))
    return config, config_module.dump(config)


def transcribe_cpp_rev() -> str:
    from healing.export import pinned_rev

    return pinned_rev()


def render_notebook(job: dict) -> str:
    scrub_source = "\n\n".join(inspect.getsource(f) for f in (words, text_grams, leaks, scrub))
    text = TEMPLATE.read_text()
    for name, value in (("@@JOB_B64@@", base64.b64encode(json.dumps(job, sort_keys=True).encode()).decode()),
                        ("@@LORA_B64@@", base64.b64encode(Path(lora.__file__).read_bytes()).decode()),
                        ("@@SCRUB_SOURCE@@", scrub_source)):
        text = text.replace(name, value)
    return text


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Remote:
    run_id: str
    username: str
    folder: Path
    record: dict
    created: dict = field(default_factory=dict)  # "datasets"/"kernels" -> id, once a create was attempted

    @property
    def dataset(self) -> str:
        return f"{self.username}/{self.run_id}"

    @property
    def kernel(self) -> str:
        # its own slug: pushing a kernel whose slug is the dataset's is refused (409 Conflict)
        return f"{self.username}/{self.run_id}-run"


def _stage_dataset(r: Remote, bundle: Path) -> Path:
    upload = r.folder / "upload"
    owner_dir(upload)
    names = [MANIFEST, LISTING] + sorted(f"audio/{p.name}" for p in (bundle / "audio").iterdir() if not p.name.startswith("._"))
    with tarfile.open(upload / "bundle.tar", "w", format=tarfile.PAX_FORMAT) as archive:
        for name in names:
            info = archive.gettarinfo(bundle / name, arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(bundle / name, "rb") as handle:
                archive.addfile(info, handle)
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "title": f"healing {r.run_id[-12:]}", "id": r.dataset, "isPrivate": True, "licenses": [{"name": "other"}]}))
    r.record["uploaded_bytes"] = (upload / "bundle.tar").stat().st_size
    return upload


def _stage_kernel(r: Remote, job: dict) -> Path:
    folder = r.folder / "kernel"
    owner_dir(folder)
    (folder / "notebook.py").write_text(render_notebook(job))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": r.kernel, "title": r.kernel.split("/")[1], "code_file": "notebook.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [r.dataset], "competition_sources": [], "kernel_sources": [], "model_sources": []}))
    return folder


def _wait_dataset(kaggle: Kaggle, r: Remote) -> None:
    deadline = time.monotonic() + DATASET_READY_SECONDS
    while True:
        status = kaggle("datasets", "status", r.dataset, check=False).stdout.strip().lower()
        if "ready" in status:
            return
        if "error" in status or time.monotonic() > deadline:
            raise RemoteError(f"the dataset never became ready ({status[-80:] or 'no status'})")
        time.sleep(POLL_SECONDS)


STATUS = re.compile(r"(queued|running|complete|error|cancel\w*)", re.IGNORECASE)


def _poll(kaggle: Kaggle, r: Remote, max_minutes: int, say) -> str:
    deadline = time.monotonic() + (max_minutes + QUEUE_MINUTES) * 60
    last = None
    while True:
        out = kaggle("kernels", "status", r.kernel, check=False).stdout
        found = STATUS.findall(out or "")
        status = found[-1].lower() if found else "unknown"
        if status != last:
            say(f"kernel {status}")
            last = status
        if status in ("complete", "error") or status.startswith("cancel"):
            return status
        if time.monotonic() > deadline:
            raise RemoteError(f"the kernel did not finish within {max_minutes + QUEUE_MINUTES} minutes")
        time.sleep(POLL_SECONDS)


def _keep_log(r: Remote, output: Path, grams: set) -> list[str]:
    """Scrub the downloaded kernel log into `<run>/notebook.log` (0600); the raw copy is removed."""
    lines: list[str] = []
    for path in sorted(output.glob("*.log")):
        raw = path.read_text(errors="replace")
        try:
            lines += [line for entry in json.loads(raw) for line in str(entry.get("data", "")).splitlines()]
        except (ValueError, AttributeError, TypeError):
            lines += raw.splitlines()
        path.unlink()
    kept, dropped = scrub(lines, grams)
    target = r.folder / "notebook.log"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("\n".join(kept) + ("\n" if kept else ""))
    r.record["log_lines_dropped"] = dropped
    return kept


def _delete(kaggle: Kaggle, r: Remote, say_error) -> bool:
    """Delete what was created. A failed delete counts as done only if the thing is no longer listed."""
    ok = True
    for kind in ("kernels", "datasets"):  # the kernel first: it mounts the dataset
        ident = r.created.get(kind)
        if ident is None:
            continue
        try:
            done = kaggle(kind, "delete", "-y", ident, check=False, timeout=300)
            if done.returncode == 0:
                continue
            listed = kaggle(kind, "list", "--mine", "-s", ident.split("/")[1], check=False, timeout=300)
            if listed.returncode == 0 and ident not in (listed.stdout or ""):
                continue
        except (subprocess.SubprocessError, OSError, RemoteError):
            pass
        ok = False
        say_error(f"error: could not delete the {kind[:-1]} {ident}; delete it yourself: kaggle {kind} delete -y {ident}")
    return ok


def run_remote(bundle: Path, config_path: Path | None = None, smoke: bool = False, allow_upload: bool = False,
               max_minutes: int = 120, rounds: Path | None = None, say=print, say_error=None) -> int:
    say_error = say_error or (lambda message: print(message, file=sys.stderr))
    if not allow_upload:
        say_error("error: refusing to upload personal audio without --allow-upload")
        return REFUSE
    creds = credentials()
    if creds is None:
        say_error(f"error: no Kaggle API token. Make one at {TOKEN_PAGE} (https://www.kaggle.com/settings/api) and "
                  f"save it as ~/.kaggle/access_token (with your username in ~/.kaggle/kaggle.json) or "
                  f"~/.kaggle/kaggle.json, mode 0600. Local training (`healing train --local`) needs none.")
        return NO_TOKEN

    # every local check before anything leaves the Mac
    config_path = Path(config_path or config_module.DEFAULT)
    config, effective = remote_config(config_path, smoke)
    bundle = Path(bundle).resolve()
    bundle_sha = verify_bundle(bundle)
    base_sha = config["base"]["sha256"]  # the notebook checks the download against it
    paths = train_run.default_paths(config, rounds=rounds)
    owner_dir(paths.rounds)
    key = hashlib.sha256(f"{config_module.text_sha256(effective)}\n{bundle_sha}\n{base_sha}\n".encode()).hexdigest()
    folder = paths.rounds / f"{bundle.name}-{key[:12]}"
    for path in (folder, folder / "work", folder / "out"):
        owner_dir(path)
    record_path = folder / "out" / train_run.RECORD
    if record_path.exists() and json.loads(record_path.read_text()).get("status") == "complete":
        say(f"already trained: {folder / 'out' / train_run.DELTA}")
        return 0

    kaggle = Kaggle(creds)
    quota = gpu_quota(kaggle)
    if quota is not None and quota[0] < max_minutes:
        say_error(f"error: Kaggle's weekly GPU quota has {quota[0]:.0f} minutes left, under --max-minutes {max_minutes}; "
                  f"it resets at {quota[1]}. Nothing was uploaded: train this round locally "
                  f"(`healing train --local --bundle {bundle}`) or rerun after the reset.")
        return QUOTA

    run_id = f"healing-{secrets.token_hex(6)}"
    remote_root = healing_home() / "remote"
    owner_dir(remote_root)
    r = Remote(run_id, creds.username, remote_root / run_id, {
        "run_id": run_id, "provider": "kaggle", "dataset": f"{creds.username}/{run_id}",
        "kernel": f"{creds.username}/{run_id}-run", "uploaded_bytes": 0, "gpu_minutes": None,
        "started_at_ms": _now_ms(), "finished_at_ms": None, "deleted_remote": None, "error": None,
        "round": folder.name, "smoke": smoke, "max_minutes": max_minutes,
        "versions": {"kaggle_cli": KAGGLE_CLI_VERSION, "transcribe_cpp_rev": None, "uv": UV_VERSION}})
    owner_dir(r.folder)
    job = {"key": key, "config": config, "bundle_sha256": bundle_sha, "base_sha256": base_sha,
           "transcribe_cpp_repo": TRANSCRIBE_CPP_REPO, "transcribe_cpp_rev": transcribe_cpp_rev(),
           "uv_version": UV_VERSION, "python_version": NEMO_PYTHON}
    r.record["versions"]["transcribe_cpp_rev"] = job["transcribe_cpp_rev"]
    code = FAILED
    try:
        code = _round(kaggle, r, job, bundle, folder, config_path, effective, smoke, max_minutes, say)
    except KeyboardInterrupt:
        r.record["error"] = "interrupted"
        code = INTERRUPTED
    except QuotaUsedUp as error:
        r.record["error"] = str(error)
        say_error(f"error: {error}")
        code = QUOTA
    except (RemoteError, TrainError, OSError, ValueError, subprocess.SubprocessError) as error:
        r.record["error"] = str(error)[:500]
        say_error(f"error: {error}")
        code = FAILED
    finally:
        r.record["deleted_remote"] = _delete(kaggle, r, say_error)
        for name in ("upload", "kernel", "output"):
            shutil.rmtree(r.folder / name, ignore_errors=True)
        r.record["finished_at_ms"] = _now_ms()
        with open(remote_root / "runs.jsonl", "a") as handle:
            handle.write(json.dumps(r.record, sort_keys=True) + "\n")
        os.chmod(remote_root / "runs.jsonl", 0o600)
        if not r.record["deleted_remote"] and code == 0:
            code = FAILED
        say(f"run {run_id}: {'deleted' if r.record['deleted_remote'] else 'NOT deleted'} on Kaggle; "
            f"recorded in {remote_root / 'runs.jsonl'}")
    return code


def _round(kaggle: Kaggle, r: Remote, job: dict, bundle: Path, folder: Path, config_path: Path, effective: str,
           smoke: bool, max_minutes: int, say) -> int:
    upload = _stage_dataset(r, bundle)
    say(f"uploading {r.record['uploaded_bytes'] / 1e6:.1f} MB as the private dataset {r.dataset}")
    r.created["datasets"] = r.dataset
    kaggle("datasets", "create", "-p", str(upload), "-r", "skip", "-q")
    shutil.rmtree(upload, ignore_errors=True)  # the local copy of the audio is no longer needed
    _wait_dataset(kaggle, r)

    kernel = _stage_kernel(r, job)
    r.created["kernels"] = r.kernel
    pushed = kaggle("kernels", "push", "-p", str(kernel), "-t", str(max_minutes * 60), "--accelerator", ACCELERATOR, check=False)
    if pushed.returncode != 0:
        text = f"{pushed.stdout} {pushed.stderr}"
        if "quota" in text.lower():
            raise QuotaUsedUp("Kaggle refused the GPU run: the weekly quota is used up; the round stays queued "
                              "for `healing train --local` or next week")
        raise RemoteError(f"kaggle kernels push failed (exit {pushed.returncode}): {' '.join(text.split())[-300:]}")
    say(f"pushed the private kernel {r.kernel} ({ACCELERATOR}, at most {max_minutes} min)")
    started = time.monotonic()
    status = _poll(kaggle, r, max_minutes, say)
    r.record["gpu_minutes"] = round((time.monotonic() - started) / 60, 1)

    output = r.folder / "output"
    owner_dir(output)
    kaggle("kernels", "output", r.kernel, "-p", str(output), "-o", "-q", check=status == "complete")
    kept = _keep_log(r, output, text_grams(manifest_texts(bundle)))
    result = {}
    if (output / "result.json").is_file():
        result = json.loads((output / "result.json").read_text())
    if isinstance(result.get("seconds"), (int, float)):
        r.record["gpu_minutes"] = round(result["seconds"] / 60, 1)
    env = {}
    if (output / "env.json").is_file():
        env = json.loads((output / "env.json").read_text())
        r.record["versions"].update({k: env.get(k) for k in ("python", "torch", "nemo")})
    r.record["versions"]["gpu"] = result.get("gpu")
    if status != "complete" or result.get("code") != 0:
        for line in kept[-20:]:
            say(f"  | {line}")
        raise RemoteError(f"the kernel ended {status} at stage {result.get('stage', 'unknown')} "
                          f"(trainer exit {result.get('code')}); its log (scrubbed) is {r.folder / 'notebook.log'}")

    out, work = folder / "out", folder / "work"
    for name, target in ((train_run.DELTA, out), (train_run.LOG, out), ("env.json", work), ("targets.json", work)):
        if not (output / name).is_file():
            raise RemoteError(f"the kernel's output lacks {name}")
        os.replace(output / name, target / name)
    trainer_job = {"key": job["key"], "config": job["config"], "bundle": str(bundle), "bundle_sha256": job["bundle_sha256"],
                   "base": f"hf://{job['config']['base']['repo']}@{job['config']['base']['revision']}/{job['config']['base']['file']}",
                   "base_sha256": job["base_sha256"]}
    (work / "trainer.json").write_text(json.dumps(trainer_job, indent=2, sort_keys=True) + "\n")
    (out / train_run.EFFECTIVE).write_text(effective)
    runner = {"runner": "kaggle", "run_id": r.run_id, "kernel": r.kernel, "accelerator": ACCELERATOR, "gpu": result.get("gpu"),
              "transcribe_cpp": job["transcribe_cpp_repo"], "transcribe_cpp_rev": job["transcribe_cpp_rev"]}
    return train_run._finish(folder, trainer_job, config_path, effective, smoke, None,
                             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r.record["started_at_ms"] / 1000)),
                             time.monotonic() - started, say, runner=runner)


def main(args) -> int:
    try:
        return run_remote(args.bundle, args.config, smoke=args.smoke, allow_upload=args.allow_upload,
                          max_minutes=args.max_minutes, rounds=args.rounds_dir)
    except (TrainError, config_module.ConfigError, RemoteError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return FAILED
