"""The model registry (M5, T5.1): the event log, objects, snapshots, retention, restore and `current`.

Every test runs in temp folders with a fake `hex` (never the real Hex data or `hex-healing` folder) and
a fake converter in a throwaway git checkout, standing in for the T6.3 export.
"""

import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from healing.cli import main
from healing.registry import objects
from healing.registry import restore as restore_module
from healing.registry import snapshot as snapshots
from healing.registry import store as registry
from healing.registry.commands import golden_pin_problem, show
from healing.store import Store

CATALOG_ID = "parakeet-unified-en-0.6b"

FAKE_HEX = """#!{python}
import json, os, sys
args = sys.argv[1:]
if os.environ.get("FAKE_HEX_DOWN"):
    sys.exit(3)
selected = os.environ["FAKE_SELECTED"]
if args[:1] == ["status"]:
    fallback = os.environ.get("FAKE_FALLBACK")
    model = {{"id": selected, "name": selected, "language": "en", "installed": not os.environ.get("FAKE_NOT_INSTALLED"),
             "preparing": None, "downloaded_bytes": 0, "error": None,
             "fallback": {{"id": fallback, "name": fallback, "language": "en", "reason": "missing"}} if fallback else None}}
    print(json.dumps({{"version": 4, "status": "status", "app_version": "x", "pid": 1, "model": model}}, indent=2))
    sys.exit(0)
if args[:2] != ["model", "list"]:
    sys.exit(2)
models = [{{"id": i, "name": i, "languages": ["en"], "size": "700 MB", "installed": True, "selected": i == selected,
           "streams": True, "vocabulary": "bias", "supports_recognition_hints": False, "source": "hex"}}
          for i in ["{catalog}", "parakeet-tdt-0.6b-v2", selected] if i]
seen, unique = set(), []
for m in models:
    if m["id"] not in seen:
        seen.add(m["id"]); unique.append(m)
print(json.dumps({{"version": 4, "status": "models", "models": unique}}, indent=2))
"""

# The stand-in for the T6.3 export: base + delta + config, plus random bytes when non-deterministic.
CONVERTER = """import os, sys
base, delta, config, output = sys.argv[1:5]
data = open(base, "rb").read() + open(delta, "rb").read() + open(config, "rb").read()
if os.environ.get("NONDETERMINISTIC"):
    data += os.urandom(8)
open(output, "wb").write(data)
"""


def git(folder, *args):
    return subprocess.run(["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    support = tmp_path / "hex-support"
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(support))  # never the real one
    monkeypatch.setenv("FAKE_SELECTED", CATALOG_ID)
    monkeypatch.delenv("HEALING_CONVERTER_DIR", raising=False)
    binary = tmp_path / "hex"
    binary.write_text(FAKE_HEX.format(python=sys.executable, catalog=CATALOG_ID))
    binary.chmod(0o755)
    (support / "models").mkdir(parents=True)
    catalog = support / "models" / f"{CATALOG_ID}-Q8_0.gguf"
    catalog.write_bytes(b"GGUF base model weights")
    checkout = tmp_path / "converter"
    checkout.mkdir()
    (checkout / "convert.py").write_text(CONVERTER)
    git(checkout, "init", "-q")
    git(checkout, "add", "convert.py")
    git(checkout, "commit", "-q", "-m", "converter")

    class Env:
        hex = str(binary)
        root = tmp_path
        rev = git(checkout, "rev-parse", "HEAD")

        def __init__(self):
            self.catalog = catalog
            self.checkout = checkout

        def write(self, name, data: bytes) -> Path:
            path = tmp_path / name
            path.write_bytes(data)
            return path

        def export(self, name: str, delta: bytes, config: str) -> Path:
            """What the converter makes from the base, `delta` and `config`: the candidate's GGUF."""
            return self.write(name, catalog.read_bytes() + delta + config.encode())

    return Env()


def cli(*argv) -> int:
    return main([str(a) for a in argv])


def base(env, *extra) -> None:
    assert cli("models", "register-base", "--hex", env.hex, *extra) == 0


def candidate(env, n: int, *extra, delta: bytes | None = None, config="lora: {rank: 16}\n") -> str:
    delta = delta if delta is not None else f"delta {n}".encode()
    gguf = env.export(f"v{n}.gguf", delta, config)
    args = ["models", "register", "--gguf", gguf, "--delta", env.write(f"delta-{n}.safetensors", delta),
            "--config", env.write(f"config-{n}.yaml", config.encode())]
    if not registry.load().versions.get("v0") or not registry.load().get("current").record.get("converter"):
        args += ["--converter-repo", "https://github.com/anomalyco/transcribe.cpp", "--converter-rev", env.rev,
                 "--export-command", f"{sys.executable} convert.py {{base}} {{delta}} {{config}} {{output}}",
                 "--converter-dir", env.checkout]
    assert cli(*args, *extra) == 0
    return objects.sha256_file(gguf)


# ---- the log ------------------------------------------------------------------------------------

def test_fold_events_to_state():
    events = [
        {"event": "registered", "version": "v0", "artifacts": {"gguf": "a" * 64}},
        {"event": "promoted", "version": "v0", "reason": "base"},
        {"event": "scored", "version": "v0", "scores": {"general": {"wer": 0.05}}},
        {"event": "registered", "version": "v1", "parent": "v0", "artifacts": {"gguf": "b" * 64}},
        {"event": "registered", "version": "v2", "parent": "v1", "artifacts": {"gguf": "c" * 64}},
        {"event": "scored", "version": "v1", "scores": {"rolling": {"wer": 0.09}}},
        {"event": "scored", "version": "v1", "scores": {"general": {"wer": 0.051}}},
        {"event": "promoted", "version": "v1"},
        {"event": "rejected", "version": "v2", "reason": "rolling CI crosses 0"},
        {"event": "some_future_event", "version": "v1"},
        {"event": "rolled_back", "version": "v1", "reason": "worse after promotion"},
        {"event": "restored", "version": "v2", "method": "wer_parity", "gguf_sha256": "d" * 64},
        {"event": "pruned_artifact", "version": "v2", "kind": "gguf", "sha256": "c" * 64},
    ]
    state = registry.fold(events)
    assert state.current == "v0"  # the rollback falls back to the latest still-promoted version
    assert [v.state for v in state.ordered()] == ["promoted", "rolled_back", "rejected"]
    assert state.get("v2").reason == "rolling CI crosses 0"
    assert state.get("v1").scores == {"rolling": {"wer": 0.09}, "general": {"wer": 0.051}}
    assert state.get("v2").gguf == "d" * 64 and state.get("v2").ggufs() == {"c" * 64, "d" * 64}
    assert state.get("v2").pruned == {"c" * 64: "gguf"}
    assert state.next_name() == "v3"
    assert state.get("current").name == "v0"
    with pytest.raises(registry.RegistryError):
        registry.fold([{"event": "promoted", "version": "v9"}])


def test_torn_last_line_ignored(env):
    with registry.lock():
        registry.append({"event": "registered", "version": "v0", "artifacts": {"gguf": "a" * 64}},
                        {"event": "promoted", "version": "v0"})
    path = registry.versions_path()
    with open(path, "ab") as file:
        file.write(b'{"event":"registered","version":"v1","art')  # a crash mid-append
    assert [e["event"] for e in registry.read_events()] == ["registered", "promoted"]
    assert registry.load().current == "v0"
    with registry.lock():  # the next append cuts the torn tail instead of gluing onto it
        registry.append({"event": "scored", "version": "v0", "scores": {"general": {"wer": 0.05}}})
    lines = path.read_bytes().split(b"\n")
    assert lines[-1] == b"" and all(json.loads(line) for line in lines[:-1]) and len(lines) == 4
    assert registry.load().get("v0").scores["general"]["wer"] == 0.05
    # a bad line that is not the last is damage, never skipped
    path.write_bytes(b'{"event":"promoted","version":"v0"}\nnot json\n{"event":"promoted","version":"v0"}\n')
    with pytest.raises(registry.RegistryError, match="line 2"):
        registry.read_events()


def test_golden_pin_immutable(env, capsys):
    assert registry.pin_golden("1" * 64, 200) is True
    assert registry.pin_golden("1" * 64, 200) is False  # the same pin again is a no-op
    with pytest.raises(registry.GoldenPinRefused, match="G-10"):
        registry.pin_golden("2" * 64, 200)
    assert sum(e["event"] == "golden_frozen" for e in registry.read_events()) == 1
    assert registry.load().golden["sha256"] == "1" * 64
    # `split verify` reads the pin: a frozen golden row whose manifest moved is a violation
    assert golden_pin_problem({"frozen": 1, "manifest_sha256": "1" * 64}) is None
    assert "differs from the registry's pin" in golden_pin_problem({"frozen": 1, "manifest_sha256": "2" * 64})


# ---- snapshots ----------------------------------------------------------------------------------

def test_snapshot_hash_stable_under_row_order(env):
    rows = [(f"clip{i}", f"{i:064x}", snapshots.text_sha256(f"text {i}"), 1.0 if i % 2 else 0.2) for i in range(50)]
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    assert snapshots.digest(rows) == snapshots.digest(shuffled)
    assert snapshots.digest(rows) != snapshots.digest(rows[:-1])
    snap = snapshots.Snapshot(snapshots.digest(shuffled), shuffled)
    path = snapshots.write(snap)
    assert path.name == f"{snap.sha256}.jsonl.gz" and path.stat().st_mode & 0o777 == 0o600
    first = path.read_bytes()
    path.unlink()
    assert snapshots.write(snap).read_bytes() == first  # no time or name in the gzip header
    assert [tuple(r) for r in snapshots.read(snap.sha256)] == sorted(rows)
    assert b"text" not in gzip.decompress(first)  # hashes only, no text
    path.write_bytes(gzip.compress(snapshots.encode(rows[:-1])))
    with pytest.raises(snapshots.SnapshotError, match="no longer matches"):
        snapshots.read(snap.sha256)


def test_snapshot_from_a_train_row_and_reproducibility(env):
    from dataset import add_clip

    store = Store()
    try:
        strong = add_clip(store, 1, raw="send it to kip", label="strong", reference="send it to kit")
        weak = add_clip(store, 2, raw="water the roses", label="weak")
        store.db.execute("UPDATE clips SET weight = CASE label WHEN 'strong' THEN 1.0 ELSE 0.2 END")
        detail = {"members": {strong: ["send it to kit", "send it to kip"], weak: ["water the roses"]}}
        store.db.execute("INSERT INTO splits (split_id, kind, parent_version, created_at_ms, frozen, n_clips, detail)"
                         " VALUES ('train-v0-x', 'train', 'v0', 1, 0, 2, ?)", (json.dumps(detail),))
        store.db.executemany("INSERT INTO split_members VALUES ('train-v0-x', ?)", [(strong,), (weak,)])
        store.db.execute("UPDATE splits SET frozen = 1 WHERE split_id = 'train-v0-x'")
        store.db.commit()
        snap = snapshots.from_train(store.db, "train-v0-x")
        assert (snap.train_n, snap.strong_n, snap.weak_n) == (2, 1, 1)
        by_clip = {r[0]: r for r in snap.rows}
        assert by_clip[strong][2] == snapshots.text_sha256("send it to kit")  # the stored reference
        # a later relabel does not change what the snapshot says the model saw
        store.db.execute("UPDATE clips SET reference_text = 'send it to kat' WHERE clip_id = ?", (strong,))
        assert snapshots.from_train(store.db, "train-v0-x").sha256 == snap.sha256
        assert snapshots.reproducible(store, snap.rows) == "full"
        audio = by_clip[weak][1]
        store.db.execute("UPDATE audio SET evicted_at_ms = 1 WHERE sha256 = ?", (audio,))
        store.audio_path(audio).unlink()
        assert snapshots.reproducible(store, snap.rows) == "partial"
        store.db.commit()
        base(env)
        candidate(env, 1, "--train", "train-v0-x")
        dataset = registry.load().get("v1").record["dataset"]
        assert dataset == {"snapshot_sha256": snap.sha256, "snapshot_path": f"snapshots/{snap.sha256}.jsonl.gz",
                           "train_split": "train-v0-x", "train_n": 2, "strong_n": 1, "weak_n": 1, "weights_frozen": False,
                           "bundle_sha256": None}
        assert snapshots.snapshot_path(snap.sha256).is_file()
        assert show(registry.load(), "v1")["reproducible"] == "partial"  # the weak clip's audio is gone
        candidate(env, 2, "--snapshot", snap.sha256)  # a stored snapshot, by hash
        assert registry.load().get("v2").record["dataset"]["snapshot_sha256"] == snap.sha256
        with pytest.raises(snapshots.SnapshotError, match="not a frozen train"):
            store.db.execute("INSERT INTO splits (split_id, kind, created_at_ms) VALUES ('rolling-x', 'rolling', 1)")
            snapshots.from_train(store.db, "rolling-x")
    finally:
        store.close()


# ---- register, list, retention ------------------------------------------------------------------

def test_register_v0_and_v1_then_list_shows_both_with_scores(env, capsys):
    baseline = env.write("baseline.json", json.dumps({
        "model": CATALOG_ID, "gguf_sha256": objects.sha256_file(env.catalog), "set_sha256": "5" * 64,
        "clips": 500, "reference_words": 7818, "wer": 0.0572}).encode())
    base(env, "--scores", baseline)
    v0 = registry.load().get("v0")
    assert v0.record["base"] == {"catalog_id": CATALOG_ID, "gguf_sha256": objects.sha256_file(env.catalog)}
    assert v0.record["arch"] == "parakeet-unified" and v0.state == "promoted"
    assert objects.has(v0.gguf)  # v0's GGUF is copied into the objects store
    gguf = env.export("v1.gguf", b"delta 1", "lora: {rank: 16}\n")
    eval_doc = env.write("eval.json", json.dumps({
        "models": [{"gguf_path": "/x/v1.gguf", "gguf_sha256": objects.sha256_file(gguf)}],
        "sets": [{"set": "general", "manifest_sha256": "5" * 64}, {"set": "rolling", "manifest_sha256": "6" * 64}],
        "scores": {"/x/v1.gguf": {"general": {"wer": 0.0575, "ref_words": 7818}, "rolling": {"wer": 0.081}}},
    }).encode())
    candidate(env, 1, "--scores", eval_doc)
    capsys.readouterr()
    assert cli("models", "list") == 0
    out = capsys.readouterr().out.splitlines()
    v0_line = next(line for line in out if line.startswith("*v0"))
    v1_line = next(line for line in out if line.lstrip().startswith("v1"))
    assert "promoted" in v0_line and "5.72" in v0_line
    assert "candidate" in v1_line and "5.75" in v1_line and "8.10" in v1_line
    assert cli("models", "list", "--json") == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["current"] == "v0" and [v["state"] for v in listed["versions"]] == ["promoted", "candidate"]
    assert cli("models", "show", "v1") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["parent"] == "v0" and shown["converter"]["rev"] == env.rev
    assert shown["config"]["sha256"] == snapshots.text_sha256("lora: {rank: 16}\n")
    assert shown["hex_model_id"] == "custom-" + objects.sha256_file(gguf)[:16]


def test_register_base_pins_by_hash_and_never_moves(env, capsys):
    base(env)
    assert cli("models", "register-base", "--hex", env.hex) == 0  # same file: already registered
    assert "already" in capsys.readouterr().out
    env.catalog.write_bytes(b"GGUF a catalog update")  # the catalog id now names another file
    assert cli("models", "register-base", "--hex", env.hex) == 1
    assert "never moves" in capsys.readouterr().err
    assert len(registry.load().versions) == 1


def test_retention_keeps_current_prev2_and_v0(env):
    base(env)
    shas = {n: candidate(env, n) for n in range(1, 7)}
    for n in range(1, 5):
        registry.promote(f"v{n}")
    registry.reject("v5", "failed the gate")
    state = registry.load()
    assert state.current == "v4"
    kept = {name for name, v in state.versions.items() if objects.has(v.gguf)}
    # v0, current v4, the previous two promoted (v3, v2), and v6 (a candidate still waiting for its gate)
    assert kept == {"v0", "v2", "v3", "v4", "v6"}
    pruned = [(e["version"], e["kind"]) for e in registry.read_events() if e["event"] == "pruned_artifact"]
    assert sorted(pruned) == [("v1", "gguf"), ("v5", "gguf")]
    assert not objects.has(shas[1]) and not objects.has(shas[5])
    assert all(objects.has(v.artifact("lora_delta")) for v in state.versions.values() if v.name != "v0")  # deltas forever
    registry.roll_back("v4", "worse after promotion")
    assert registry.load().current == "v3"
    registry.retain()  # v4 is still one of the previous two promoted; no duplicate events
    assert len([e for e in registry.read_events() if e["event"] == "pruned_artifact"]) == 2


def test_status_reports_registry_object_bytes(env, tmp_path):
    from healing.status import compute

    base(env)
    store = Store()
    try:
        status = compute(store, tmp_path / "no-hex")
    finally:
        store.close()
    assert status.registry_object_bytes == env.catalog.stat().st_size
    assert status.as_dict()["registry_object_bytes"] == status.registry_object_bytes
    assert f"registry objects {status.registry_object_bytes} bytes" in status.text()


# ---- current ------------------------------------------------------------------------------------

def test_current_mismatch_reported(env, capsys, monkeypatch):
    base(env)
    capsys.readouterr()
    assert cli("models", "current", "--hex", env.hex) == 0
    assert "matches" in capsys.readouterr().out
    sha = candidate(env, 1)
    registry.promote("v1")
    monkeypatch.setenv("FAKE_SELECTED", "custom-" + sha[:16])  # you imported and picked v1 in Hex
    capsys.readouterr()
    assert cli("models", "current", "--hex", env.hex) == 0
    monkeypatch.setenv("FAKE_SELECTED", CATALOG_ID)  # then picked the catalog model again by hand
    before = registry.versions_path().read_bytes()
    assert cli("models", "current", "--hex", env.hex) == 1
    out = capsys.readouterr().out
    assert "mismatch" in out and CATALOG_ID in out and "custom-" + sha[:16] in out
    assert registry.versions_path().read_bytes() == before  # reported, never fixed
    monkeypatch.setenv("FAKE_HEX_DOWN", "1")
    assert cli("models", "current", "--hex", env.hex) == 3


def test_current_reports_a_catalog_update_under_v0(env, capsys):
    base(env)
    env.catalog.write_bytes(b"GGUF a catalog update")
    capsys.readouterr()
    assert cli("models", "current", "--hex", env.hex) == 1
    assert "catalog update" in capsys.readouterr().out


# ---- restore ------------------------------------------------------------------------------------

def delete_every_copy(env, n: int) -> str:
    """The candidate's GGUF from the objects store and from where it was registered from."""
    sha = registry.load().get(f"v{n}").artifact("gguf")
    assert objects.remove(sha)
    (env.root / f"v{n}.gguf").unlink()
    assert not objects.has(sha) and not list(env.root.rglob(f"*{sha}*"))
    return sha


def test_restore_rebuilds_a_deleted_gguf_byte_identical(env, capsys):
    base(env)
    candidate(env, 1)
    sha = delete_every_copy(env, 1)
    capsys.readouterr()
    assert cli("models", "restore", "v1") == 0  # the converter checkout recorded at register
    assert "sha256 matches" in capsys.readouterr().out
    assert objects.has(sha) and objects.sha256_file(objects.path(sha)) == sha
    (event,) = [e for e in registry.read_events() if e["event"] == "restored"]
    assert event["method"] == "sha256" and event["rebuilt_sha256"] == sha and event["gguf_sha256"] == sha


def test_restore_of_a_non_deterministic_converter_records_wer_parity(env, monkeypatch):
    base(env)
    candidate(env, 1)
    registry.record_scores("v1", {"general": {"wer": 0.0575, "set_sha256": "5" * 64, "vocabulary_terms": 0}})
    original = delete_every_copy(env, 1)
    monkeypatch.setenv("NONDETERMINISTIC", "1")
    seen = []

    def parity(version, gguf):
        seen.append(gguf.read_bytes())
        return restore_module.compare(version.scores, {"general": (0.0579, "5" * 64)})  # 0.04 pt off

    event = restore_module.restore("v1", parity=parity)
    assert event["method"] == "wer_parity" and event["parity"]["passed"]
    assert event["parity"]["sets"]["general"]["delta_pt"] == pytest.approx(0.04)
    rebuilt = event["gguf_sha256"]
    assert rebuilt != original and objects.has(rebuilt) and not objects.has(original)
    assert seen and seen[0].startswith(env.catalog.read_bytes())
    v1 = registry.load().get("v1")
    assert v1.gguf == rebuilt and v1.restores[0]["method"] == "wer_parity"  # recorded in the version

    def failing(version, gguf):
        return restore_module.compare(version.scores, {"general": (0.0600, "5" * 64)})  # 0.25 pt off

    objects.remove(rebuilt)
    events = len(registry.read_events())
    with pytest.raises(restore_module.RestoreError, match="fails WER parity"):
        restore_module.restore("v1", parity=failing)
    assert len(registry.read_events()) == events and len(list(objects.objects_dir().glob("*/*"))) == 2  # base + delta


def test_parity_compare_skips_a_changed_set_and_needs_one():
    recorded = {"general": {"wer": 0.05, "set_sha256": "a", "vocabulary_terms": 0},
                "golden": {"wer": 0.04, "set_sha256": "b", "vocabulary_terms": 0}}
    result = restore_module.compare(recorded, {"general": (0.0504, "a"), "golden": (0.09, "changed")})
    assert result["passed"] and result["skipped"] == {"golden": "the set changed since it was scored"}
    assert not restore_module.compare(recorded, {"golden": (0.04, "changed")})["passed"]


def test_restore_refuses_another_converter_rev_or_a_dirty_checkout(env):
    base(env)
    candidate(env, 1)
    delete_every_copy(env, 1)
    (env.checkout / "convert.py").write_text(CONVERTER + "# local change\n")
    with pytest.raises(restore_module.RestoreError, match="local changes"):
        restore_module.restore("v1")
    git(env.checkout, "commit", "-qam", "newer converter")
    with pytest.raises(restore_module.RestoreError, match="not the recorded"):
        restore_module.restore("v1")
    git(env.checkout, "checkout", "-q", env.rev)
    assert restore_module.restore("v1")["method"] == "sha256"


def test_restore_needs_the_base_and_delta_objects(env):
    base(env)
    candidate(env, 1)
    delta = registry.load().get("v1").artifact("lora_delta")
    objects.remove(delta)
    with pytest.raises(restore_module.RestoreError, match="LoRA delta"):
        restore_module.restore("v1")
    with pytest.raises(restore_module.RestoreError, match="base model"):
        restore_module.restore("v0")


# ---- review fixes (Codex on 2f44f91c) -----------------------------------------------------------

def test_register_base_resumes_an_init_interrupted_before_promotion(env, capsys):
    sha = objects.sha256_file(env.catalog)
    with registry.lock():  # a crash between the `registered` and `promoted` lines
        registry.append({"event": "registered", "version": "v0", "base": {"catalog_id": CATALOG_ID, "gguf_sha256": sha},
                         "parent": None, "arch": "parakeet-unified", "artifacts": {"lora_delta": None, "gguf": sha}})
    assert registry.load().current is None
    base(env)
    state = registry.load()
    assert state.current == "v0" and state.get("v0").state == "promoted" and objects.has(sha)
    assert "resumed" in capsys.readouterr().out


def test_parity_eval_scores_golden_with_the_dataset_store(env, monkeypatch, tmp_path):
    from healing.eval import run as eval_run

    gguf = env.write("rebuilt.gguf", b"x")
    seen = {}

    def evaluate(models, sets, **kwargs):
        seen.update(kwargs)
        golden = kwargs.get("store") is not None  # golden is a dataset set: unavailable without the store
        return {"sets": [{"set": "golden", "manifest_sha256": "g" * 64}] if golden else [],
                "scores": {str(gguf): {"golden": {"wer": 0.04}} if golden else {}}}

    monkeypatch.setattr(eval_run, "evaluate", evaluate)
    version = registry.Version("v1", {}, scores={"golden": {"wer": 0.0402, "set_sha256": "g" * 64, "vocabulary_terms": 0}})
    result = restore_module.eval_parity(version, gguf)
    assert seen["store"] is not None and seen["vocabulary"] == "none"
    assert result["passed"] and "golden" in result["sets"]


def test_events_record_the_resolved_version_name(env):
    base(env)
    candidate(env, 1)
    registry.promote("v1")
    registry.record_scores("current", {"general": {"wer": 0.05}})
    registry.roll_back("current", "worse after promotion")
    events = registry.read_events()
    assert "current" not in {e.get("version") for e in events}
    state = registry.load()  # the log still reads
    assert state.get("v1").scores["general"]["wer"] == 0.05 and state.current == "v0"


def test_roll_back_needs_a_distinct_still_promoted_successor(env):
    base(env)
    with pytest.raises(registry.RegistryError, match="no other promoted"):
        registry.roll_back("v0", "the only promoted version")
    candidate(env, 1)
    registry.promote("v1")
    with pytest.raises(registry.RegistryError, match="itself"):
        registry.roll_back("v1", "x", to="v1")
    candidate(env, 2)
    with pytest.raises(registry.RegistryError, match="not promoted"):
        registry.roll_back("v1", "x", to="v2")
    assert registry.load().current == "v1"
    registry.roll_back("v1", "worse", to="v0")
    assert registry.load().current == "v0"


def test_snapshot_hashes_the_weights_the_train_row_froze(env):
    from dataset import add_clip

    store = Store()
    try:
        clip = add_clip(store, 1, raw="send it to kip", label="strong", reference="send it to kit")
        store.db.execute("UPDATE clips SET weight = 1.0")
        detail = {"members": {clip: ["send it to kit", "send it to kip"]}, "weights": {clip: 1.0}}
        store.db.execute("INSERT INTO splits (split_id, kind, created_at_ms, frozen, detail) VALUES ('train-y', 'train', 1, 0, ?)",
                         (json.dumps(detail),))
        store.db.execute("UPDATE splits SET frozen = 1 WHERE split_id = 'train-y'")
        first = snapshots.from_train(store.db, "train-y")
        store.db.execute("UPDATE clips SET weight = 0.2, label = 'weak'")  # relabelled after training
        again = snapshots.from_train(store.db, "train-y")
        assert again.sha256 == first.sha256 and again.rows[0][3] == 1.0 and again.weights_frozen
    finally:
        store.close()


def test_put_replaces_a_corrupt_stored_object(env):
    source = env.write("model.gguf", b"good bytes")
    sha = objects.put(source)
    objects.path(sha).write_bytes(b"rotten bytes")
    assert objects.put(source) == sha
    assert objects.sha256_file(objects.path(sha)) == sha


def test_a_corrupt_gguf_object_is_repaired_by_a_hash_restore(env):
    base(env)
    sha = candidate(env, 1)
    objects.path(sha).write_bytes(b"bit rot")
    assert restore_module.restore("v1")["method"] == "sha256"
    assert objects.sha256_file(objects.path(sha)) == sha


def test_a_later_hash_restore_supersedes_a_parity_restore(env, monkeypatch):
    base(env)
    original = candidate(env, 1)
    registry.record_scores("v1", {"general": {"wer": 0.05, "set_sha256": "5" * 64, "vocabulary_terms": 0}})
    monkeypatch.setenv("NONDETERMINISTIC", "1")
    parity = restore_module.restore(
        "v1", parity=lambda v, g: restore_module.compare(v.scores, {"general": (0.05, "5" * 64)}))
    assert registry.load().get("v1").gguf == parity["gguf_sha256"] != original
    monkeypatch.delenv("NONDETERMINISTIC")
    event = restore_module.restore("v1")
    assert event["method"] == "sha256" and event["gguf_sha256"] == original
    assert registry.load().get("v1").gguf == original


def test_parity_ignores_scores_whose_set_cannot_be_identified():
    unnamed = restore_module.compare({"general": {"wer": 0.05, "vocabulary_terms": 0}}, {"general": (0.05, "a")})
    assert not unnamed["passed"] and "identify" in unnamed["skipped"]["general"]
    unknown_now = restore_module.compare({"general": {"wer": 0.05, "set_sha256": "a", "vocabulary_terms": 0}},
                                         {"general": (0.05, None)})
    assert not unknown_now["passed"] and "identify" in unknown_now["skipped"]["general"]


def test_current_reports_an_unverifiable_v0_file(env, capsys):
    base(env)
    env.catalog.unlink()
    capsys.readouterr()
    code = cli("models", "current", "--hex", env.hex)
    out = capsys.readouterr().out
    assert code == 4 and "unverifiable" in out and "matches" not in out


def test_a_prune_interrupted_by_a_crash_is_finished_and_recorded(env, monkeypatch):
    base(env)
    shas = {n: candidate(env, n) for n in (1, 2)}
    real_append, real_remove = registry.append, objects.remove

    def crash_on_prune(*events):
        if any(e["event"] == "pruned_artifact" for e in events):
            raise KeyboardInterrupt("crash")
        real_append(*events)

    monkeypatch.setattr(registry, "append", crash_on_prune)
    with pytest.raises(KeyboardInterrupt):
        registry.reject("v1", "failed the gate")
    monkeypatch.setattr(registry, "append", real_append)

    def crash_on_remove(sha):
        raise KeyboardInterrupt("crash")

    monkeypatch.setattr(objects, "remove", crash_on_remove)
    with registry.lock(), pytest.raises(KeyboardInterrupt):
        registry.retain()
    monkeypatch.setattr(objects, "remove", real_remove)
    with registry.lock():
        registry.retain()
    pruned = [e for e in registry.read_events() if e["event"] == "pruned_artifact"]
    assert [(e["version"], e["sha256"]) for e in pruned] == [("v1", shas[1])]
    assert not objects.has(shas[1]) and objects.has(shas[2])


def test_new_object_folders_are_fsynced_in_their_parents(env, monkeypatch):
    synced = []
    real = objects.fsync_dir
    monkeypatch.setattr(objects, "fsync_dir", lambda path: (synced.append(Path(path)), real(path)))
    sha = objects.put(env.write("model.gguf", b"bytes"))
    assert objects.registry_dir() in synced  # holds the new `objects` entry
    assert objects.objects_dir() in synced  # holds the new `<aa>` entry
    assert objects.path(sha).parent in synced  # holds the object


def test_hexcli_reads_a_pretty_printed_reply():
    from healing import hexcli

    assert hexcli.reply('note\n{\n  "status": "models",\n  "models": []\n}\n') == {"status": "models", "models": []}


# ---- review fixes, round 2 (Codex on 954ad580) --------------------------------------------------

def converter_args(env) -> list:
    return ["--converter-repo", "r", "--converter-rev", env.rev, "--converter-dir", env.checkout,
            "--export-command", f"{sys.executable} convert.py {{base}} {{delta}} {{config}} {{output}}"]


def test_candidate_objects_are_published_under_the_registry_lock(env):
    import threading

    base(env)
    shared = candidate(env, 1)  # v1's GGUF; v2 registers the same file with another delta
    real_put = objects.put
    racer = {}

    def put(source, sha256=None):
        digest = real_put(source, sha256)
        if "thread" not in racer:  # a concurrent `reject v1` (retention) right after the first put
            racer["thread"] = threading.Thread(target=registry.reject, args=("v1", "failed the gate"))
            racer["thread"].start()
            racer["thread"].join(0.5)
        return digest

    import healing.registry.objects as objects_module

    objects_module.put = put
    try:
        assert cli("models", "register", "--gguf", env.root / "v1.gguf", "--delta", env.write("d2", b"other delta"),
                   "--config", env.root / "config-1.yaml", *converter_args(env)) == 0
    finally:
        objects_module.put = real_put
        racer["thread"].join()
    assert registry.load().get("v2").artifact("gguf") == shared
    assert objects.has(shared)  # the registered candidate still has its GGUF


def test_owner_dir_leaves_existing_ancestors_alone(env, tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    monkeypatch.setenv("HEALING_HOME", str(shared / "new" / "home"))
    synced = []
    real = objects.fsync_dir
    monkeypatch.setattr(objects, "fsync_dir", lambda path: (synced.append(Path(path)), real(path)))
    objects.put(env.write("model.gguf", b"bytes"))
    assert shared.stat().st_mode & 0o777 == 0o755  # an ancestor it did not create keeps its mode
    assert (shared / "new").stat().st_mode & 0o777 == 0o700
    assert tmp_path not in synced  # nothing was created in it


def test_current_does_not_match_a_saved_import_hex_cannot_run(env, capsys, monkeypatch):
    base(env)
    sha = candidate(env, 1)
    registry.promote("v1")
    monkeypatch.setenv("FAKE_SELECTED", "custom-" + sha[:16])
    monkeypatch.setenv("FAKE_FALLBACK", "parakeet-tdt-0.6b-v2")  # the import is missing; Hex runs its default
    capsys.readouterr()
    assert cli("models", "current", "--hex", env.hex) == 1
    out = capsys.readouterr().out
    assert "mismatch" in out and "parakeet-tdt-0.6b-v2" in out and "matches" not in out
    monkeypatch.delenv("FAKE_FALLBACK")
    monkeypatch.setenv("FAKE_NOT_INSTALLED", "1")
    assert cli("models", "current", "--hex", env.hex) == 4
    assert "unverifiable" in capsys.readouterr().out


def test_parity_counts_only_vocabulary_free_scores(env):
    from healing.registry.commands import scores_from

    gguf_sha = "a" * 64
    document = {"models": [{"gguf_path": "/m", "gguf_sha256": gguf_sha}], "vocabulary": {"mode": "current", "terms": 12, "boost": "normal"},
                "sets": [{"set": "general", "manifest_sha256": "5" * 64}], "scores": {"/m": {"general": {"wer": 0.05}}}}
    biased = scores_from(document, gguf_sha, "eval.json")
    assert biased["general"]["vocabulary_terms"] == 12
    result = restore_module.compare(biased, {"general": (0.05, "5" * 64)})
    assert not result["passed"] and "vocabulary" in result["skipped"]["general"]
    document["vocabulary"] = {"mode": "none", "terms": 0, "boost": None}
    assert restore_module.compare(scores_from(document, gguf_sha, "eval.json"), {"general": (0.05, "5" * 64)})["passed"]
    baseline = scores_from({"wer": 0.05, "set_sha256": "5" * 64, "gguf_sha256": gguf_sha}, gguf_sha, "baseline.json")
    assert baseline["general"]["vocabulary_terms"] == 0  # `eval baseline` is `--vocabulary none`


def test_every_deletion_of_a_recreated_object_is_recorded(env):
    base(env)
    shared = candidate(env, 1)
    registry.reject("v1", "failed")
    assert not objects.has(shared)
    assert cli("models", "register", "--gguf", env.export("again.gguf", b"delta 1", "lora: {rank: 16}\n"),
               "--delta", env.write("d2", b"other delta"), "--config", env.root / "config-1.yaml", *converter_args(env)) == 0
    assert objects.has(shared)  # recreated for v2
    registry.reject("v2", "failed again")
    assert not objects.has(shared)
    pruned = [(e["version"], e["sha256"]) for e in registry.read_events() if e["event"] == "pruned_artifact"]
    assert pruned.count(("v1", shared)) == 2 and pruned.count(("v2", shared)) == 1


def test_register_base_retry_repairs_a_corrupt_v0_object(env):
    base(env)
    sha = objects.sha256_file(env.catalog)
    objects.path(sha).write_bytes(b"bit rot")
    base(env)
    assert objects.sha256_file(objects.path(sha)) == sha


def test_restore_resolves_a_relative_converter_checkout(env, monkeypatch):
    base(env)
    gguf = env.export("v1.gguf", b"delta 1", "c\n")
    assert cli("models", "register", "--gguf", gguf, "--delta", env.write("d1", b"delta 1"), "--config", env.write("c1", b"c\n"),
               "--converter-repo", "r", "--converter-rev", env.rev, "--converter-dir", env.checkout,
               "--export-command", f"{sys.executable} {{checkout}}/convert.py {{base}} {{delta}} {{config}} {{output}}") == 0
    monkeypatch.chdir(env.root)
    assert restore_module.restore("v1", checkout=Path("converter"))["method"] == "sha256"
    monkeypatch.setenv("HEALING_CONVERTER_DIR", "converter")
    assert restore_module.restore("v1")["method"] == "sha256"


def test_a_bad_scores_file_registers_nothing(env, capsys):
    base(env)
    bad = env.write("bad.json", b"{not json")
    gguf = env.export("v1.gguf", b"delta 1", "c\n")
    code = cli("models", "register", "--gguf", gguf, "--delta", env.write("d1", b"delta 1"), "--config", env.write("c1", b"c\n"),
               "--converter-repo", "r", "--converter-rev", env.rev, "--converter-dir", env.checkout,
               "--export-command", "x", "--scores", bad)
    assert code == 1 and set(registry.load().versions) == {"v0"}
    assert not objects.has(objects.sha256_file(gguf))
