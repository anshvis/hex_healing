"""T7.2: `healing promote` and `healing rollback` (AC-13).

A real registry in a temp `HEALING_HOME` and a fake `hex` that keeps its state (what is imported, what
is selected, whether Labs › Stream imported models is on) in a JSON file, so a promote's import, switch,
confirmation and cleanup can be followed call by call. No real Hex runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from healing import ship
from healing.cli import main
from healing.registry import objects
from healing.registry import store as registry
from healing.train import auto

CATALOG = "parakeet-unified-en-0.6b"
FOREIGN = "custom-0123456789abcdef"

FAKE_HEX = r'''#!{python}
import hashlib, json, os, sys
path = os.environ["FAKE_HEX_STATE"]
state = json.load(open(path))
args = sys.argv[1:]
state["calls"].append(args)

def save():
    json.dump(state, open(path, "w"))

def error(code, message):
    save()
    print(json.dumps({{"version": 4, "status": "error", "code": code, "message": message}}))
    sys.exit(1)

def need_running():
    if not state["running"]:
        save()
        sys.exit(3)

def streams(model_id):
    if not model_id.startswith("custom-"):
        return True
    return state["lab"] and model_id not in state["no_stream"]

if args[:2] == ["status", "--json"]:
    need_running()
    save()
    fallback = state["fallback"] if state["fallback"] and state["selected"] == state["fallback"]["for"] else None
    preparing = state["selected"] if state["selected"] in state["stuck"] else None
    print(json.dumps({{"version": 4, "status": "status", "labs": ["stream-imported-models"] if state["lab"] else [],
                      "model": {{"id": state["selected"], "installed": True, "preparing": preparing, "error": None,
                                 "fallback": fallback}}}}, indent=2))
elif args[:3] == ["model", "list", "--json"]:
    need_running()
    save()
    models = [{{"id": i, "selected": i == state["selected"], "installed": True, "streams": streams(i),
                "source": "hex" if i == state["catalog"] else state["sources"].get(i, "imported")}}
              for i in [state["catalog"], *state["imports"]]]
    print(json.dumps({{"version": 4, "status": "models", "models": models}}, indent=2))
elif args[:2] == ["model", "import"]:
    digest = hashlib.sha256(open(args[2], "rb").read()).hexdigest()
    model_id = "custom-" + digest[:16]
    state["imports"][model_id] = args[args.index("--name") + 1] if "--name" in args else "unnamed"
    save()
    print(f"imported\t{{model_id}}\t{{state['imports'][model_id]}}")
elif args[:2] == ["model", "use"]:
    need_running()
    model_id = args[2]
    if model_id != state["catalog"] and model_id not in state["imports"]:
        error("unknown-model", f"no model {{model_id}}")
    if not state["ignore_use"]:
        state["selected"] = model_id
    save()
    print(json.dumps({{"version": 4, "status": "done", "message": "ok", "changed": True}}))
elif args[:2] == ["model", "remove"]:
    model_id = args[2]
    if model_id == state["selected"]:
        error("selected", "the selected model")
    state["imports"].pop(model_id, None)
    save()
    print(f"removed\t{{model_id}}")
else:
    save()
    sys.exit(f"unexpected {{args}}")
'''


class FakeHex:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.binary = tmp_path / "hex"
        self.binary.write_text(FAKE_HEX.format(python=sys.executable))
        self.binary.chmod(0o755)
        self.path = tmp_path / "fake-hex.json"
        monkeypatch.setenv("FAKE_HEX_STATE", str(self.path))
        self.write({"running": True, "lab": True, "catalog": CATALOG, "selected": CATALOG, "imports": {},
                    "no_stream": [], "ignore_use": False, "stuck": [], "fallback": None, "sources": {}, "calls": []})

    def read(self) -> dict:
        return json.loads(self.path.read_text())

    def write(self, state: dict) -> None:
        self.path.write_text(json.dumps(state))

    def set(self, **changes) -> None:
        self.write({**self.read(), **changes})

    @property
    def selected(self) -> str:
        return self.read()["selected"]

    @property
    def imports(self) -> dict:
        return self.read()["imports"]

    def calls(self, verb: str | None = None) -> list[list[str]]:
        return [c for c in self.read()["calls"] if verb is None or c[:2] == ["model", verb]]


@pytest.fixture
def hexbin(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))  # never the real one
    registry.append(
        {"event": "registered", "version": "v0", "parent": None, "base": {"catalog_id": CATALOG},
         "dataset": None, "artifacts": {"gguf": _put(tmp_path, "v0")}},
        {"event": "promoted", "version": "v0", "reason": "base"},
    )
    return FakeHex(tmp_path, monkeypatch)


def _put(tmp_path: Path, name: str) -> str:
    source = tmp_path / f"{name}.gguf"
    source.write_bytes(b"GGUF" + name.encode() * 100)
    return objects.put(source)


def candidate(tmp_path: Path, *, parent: str = "v0") -> str:
    return registry.register({"parent": parent, "dataset": None, "artifacts": {"gguf": _put(tmp_path, f"c{registry.load().next_name()}")}})


def passing_gate(name: str, *, against: str | None = None, candidate_sha: str | None = None, **decision) -> None:
    state = registry.load()
    version, target = state.get(name), state.get(against or "current")
    models = {"candidate": {"version": version.name, "gguf_sha256": candidate_sha or version.gguf},
              "against": {"version": target.name, "gguf_sha256": target.gguf},
              "v0": {"version": "v0", "gguf_sha256": state.get("v0").gguf}}
    registry.record_gate(name, {"against": target.name, "models": models, "pass": True, "refused": False,
                                "checks": [{"name": "streaming", "status": "pass"}], **decision}, None)


def promote(hexbin: FakeHex, name: str, **kw) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = ship.promote(name, binary=hexbin.binary, say=lines.append, confirm_s=kw.pop("confirm_s", 2.0),
                        poll_s=0.01, **kw)
    return code, lines


def rollback(hexbin: FakeHex, **kw) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = ship.rollback(binary=hexbin.binary, say=lines.append, confirm_s=2.0, poll_s=0.01, **kw)
    return code, lines


def hex_id(name: str) -> str:
    return objects.hex_model_id(registry.load().get(name).gguf)


def events(kind: str) -> list[dict]:
    return [e for e in registry.read_events() if e["event"] == kind]


# ---- promote ---------------------------------------------------------------------------------------

def test_promote_switches_hex_and_records(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    code, lines = promote(hexbin, "v1")
    assert code == 0, lines
    assert hexbin.selected == hex_id("v1")
    imported = hexbin.calls("import")
    assert len(imported) == 1 and imported[0][3:] == ["--name", "Hex healing v1"]
    assert not any("--language" in c for c in hexbin.calls())  # Parakeet customs take no language hint
    state = registry.load()
    assert state.current == "v1" and state.versions["v1"].state == "promoted"
    [event] = events("promoted")[-1:]
    assert event["version"] == "v1" and event["hex_id"] == hex_id("v1") and event["previous"] == "v0"


@pytest.mark.parametrize("gate", [None, {"pass": False, "refused": False}, {"pass": False, "refused": True}])
def test_refuses_without_passing_gate(hexbin, tmp_path, gate):
    candidate(tmp_path)
    if gate is not None:
        passing_gate("v1", **gate)
    before = registry.read_events()
    code, lines = promote(hexbin, "v1")
    assert code == ship.EXIT_NO_GATE == 8
    assert hexbin.calls("import") == [] and hexbin.calls("use") == [] and hexbin.selected == CATALOG
    assert registry.read_events() == before


def test_gate_against_another_current_does_not_count(hexbin, tmp_path):
    candidate(tmp_path)
    candidate(tmp_path)
    passing_gate("v1")
    passing_gate("v2")  # judged against v0
    assert promote(hexbin, "v1")[0] == 0
    code, lines = promote(hexbin, "v2")  # current is v1 now
    assert code == ship.EXIT_NO_GATE
    assert any("v0" in line and "v1" in line for line in lines)
    assert registry.load().current == "v1"


def test_gate_on_a_swapped_candidate_gguf_does_not_count(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    swapped = _put(tmp_path, "rebuilt")
    with registry.lock():
        registry.append({"event": "restored", "version": "v1", "method": "parity", "rebuilt_sha256": swapped})
    code, _ = promote(hexbin, "v1")
    assert code == ship.EXIT_NO_GATE
    assert hexbin.calls("use") == []


def test_hex_not_running_changes_nothing(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(running=False)
    before = registry.read_events()
    code, _ = promote(hexbin, "v1")
    assert code == ship.EXIT_HEX_DOWN == 2
    assert hexbin.calls("import") == [] and registry.read_events() == before


def test_lab_off_refuses_a_streaming_candidate(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(lab=False)
    before = registry.read_events()
    code, lines = promote(hexbin, "v1")
    assert code == ship.EXIT_LAB_OFF == 10
    assert any("turn on Labs › Stream imported models" in line for line in lines)
    assert hexbin.calls("import") == [] and hexbin.selected == CATALOG
    assert registry.read_events() == before


def test_status_mismatch_records_failure(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(ignore_use=True)  # Hex answers, but keeps the old model
    code, lines = promote(hexbin, "v1", confirm_s=0.3)
    assert code == ship.EXIT_MISMATCH == 9
    [failed] = events("promote_failed")
    assert failed["version"] == "v1" and failed["hex_id"] == hex_id("v1") and "status" in failed["reason"]
    state = registry.load()
    assert state.current == "v0" and state.versions["v1"].state == "candidate"
    assert hexbin.calls("use")[-1][2] == CATALOG  # switched back to what Hex ran before


def test_not_streaming_after_use_switches_back(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(no_stream=[hex_id("v1")])  # its smoke stream failed as it loaded
    code, lines = promote(hexbin, "v1", confirm_s=0.3)
    assert code == ship.EXIT_MISMATCH
    assert hexbin.selected == CATALOG
    [failed] = events("promote_failed")
    assert "stream" in failed["reason"]
    assert registry.load().current == "v0"


def test_promote_failed_candidate_can_be_promoted_again(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(ignore_use=True)
    assert promote(hexbin, "v1", confirm_s=0.2)[0] == ship.EXIT_MISMATCH
    hexbin.set(ignore_use=False)
    assert promote(hexbin, "v1")[0] == 0
    assert registry.load().current == "v1"


def test_cleanup_keeps_current_prev2_v0_and_foreign_imports(hexbin, tmp_path):
    hexbin.set(imports={FOREIGN: "my own model"})
    for n in range(1, 5):
        candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        assert promote(hexbin, f"v{n}")[0] == 0
    imports = hexbin.imports
    assert FOREIGN in imports  # never removed: healing didn't create it
    assert {hex_id("v4"), hex_id("v3"), hex_id("v2")} <= set(imports)
    assert hex_id("v1") not in imports
    removed = [c[2] for c in hexbin.calls("remove")]
    assert removed == [hex_id("v1")]
    assert hexbin.selected == hex_id("v4")


def test_registry_refuses_a_second_promotion_from_a_stale_current(hexbin, tmp_path):
    """Two promotes started from the same current: the second's registry write re-checks and loses."""
    candidate(tmp_path)
    candidate(tmp_path)
    registry.promote("v1", "first", expect_current="v0")
    with pytest.raises(registry.RegistryError, match="current"):
        registry.promote("v2", "second", expect_current="v0")


def test_a_second_promote_is_refused_while_one_runs(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    with ship.ship_lock():
        code, lines = promote(hexbin, "v1")
    assert code == 1 and any("another" in line for line in lines)
    assert hexbin.calls("use") == []


# ---- rollback --------------------------------------------------------------------------------------

def test_rollback_returns_to_the_previous_version(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    assert promote(hexbin, "v1")[0] == 0
    code, lines = rollback(hexbin, reason="worse on my voice")
    assert code == 0, lines
    assert hexbin.selected == CATALOG
    [event] = events("rolled_back")
    assert event == {**event, "version": "v1", "reason": "worse on my voice", "to": "v0"}
    state = registry.load()
    assert state.current == "v0" and state.versions["v1"].state == "rolled_back"


def test_rollback_reimports_missing_gguf(hexbin, tmp_path):
    for n in (1, 2):
        candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        assert promote(hexbin, f"v{n}")[0] == 0
    state = hexbin.read()
    state["imports"].pop(hex_id("v1"))  # removed from Hex by hand
    hexbin.write(state)
    imports_before = len(hexbin.calls("import"))
    code, lines = rollback(hexbin)
    assert code == 0, lines
    assert hexbin.selected == hex_id("v1")
    assert len(hexbin.calls("import")) == imports_before + 1
    assert hexbin.calls("import")[-1][3:] == ["--name", "Hex healing v1"]
    assert registry.load().current == "v1"
    assert events("rolled_back")[-1]["reason"] == "manual"


def test_rollback_to_a_named_version(hexbin, tmp_path):
    for n in (1, 2):
        candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        assert promote(hexbin, f"v{n}")[0] == 0
    code, _ = rollback(hexbin, to="v0")
    assert code == 0 and hexbin.selected == CATALOG and registry.load().current == "v0"


def test_rollback_with_hex_down_changes_nothing(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    assert promote(hexbin, "v1")[0] == 0
    hexbin.set(running=False)
    before = registry.read_events()
    code, _ = rollback(hexbin)
    assert code == ship.EXIT_HEX_DOWN and registry.read_events() == before


def test_rollback_hex_mismatch_records_nothing(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    assert promote(hexbin, "v1")[0] == 0
    hexbin.set(ignore_use=True)
    before = registry.read_events()
    lines: list[str] = []
    code = ship.rollback(binary=hexbin.binary, say=lines.append, confirm_s=0.2, poll_s=0.01)
    assert code == ship.EXIT_MISMATCH and registry.read_events() == before


# ---- the command line and train auto -----------------------------------------------------------

def test_cli_promote_and_rollback(hexbin, tmp_path):
    candidate(tmp_path)
    assert main(["promote", "v1", "--hex", str(hexbin.binary)]) == ship.EXIT_NO_GATE
    passing_gate("v1")
    assert main(["promote", "v1", "--hex", str(hexbin.binary)]) == 0
    assert main(["rollback", "--reason", "regressed", "--hex", str(hexbin.binary)]) == 0
    assert registry.load().current == "v0" and events("rolled_back")[-1]["reason"] == "regressed"


def test_promoted_and_rolled_back_candidates_count_as_decided(hexbin, tmp_path):
    from healing.store import Store

    candidate(tmp_path)
    store = Store()
    try:
        assert auto.progress(store).pending == "v1"
        passing_gate("v1")
        assert promote(hexbin, "v1")[0] == 0
        assert auto.progress(store).pending is None
        assert rollback(hexbin)[0] == 0
        assert auto.progress(store).pending is None
    finally:
        store.close()


def test_auto_says_a_promote_failed(hexbin, tmp_path):
    from healing.store import Store

    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(ignore_use=True)
    assert promote(hexbin, "v1", confirm_s=0.2)[0] == ship.EXIT_MISMATCH
    store = Store()
    try:
        progress = auto.progress(store)
    finally:
        store.close()
    assert progress.pending == "v1" and "promote failed" in progress.waiting()


# ---- review fixes ---------------------------------------------------------------------------------

def test_restore_refused_while_a_promote_runs(hexbin, tmp_path):
    from healing.registry import restore as restore_module
    from healing.store import StoreError

    candidate(tmp_path)
    with registry.ship_lock():
        with pytest.raises(StoreError, match="rollback or restore"):
            restore_module.restore("v1")


def _swap_during_wait(monkeypatch, tmp_path, name: str):
    """A restore that lands between Hex's switch and the registry write (as if it took no lock)."""
    real = ship._wait
    done = []

    def wait(*args, **kwargs):
        if not done:
            done.append(True)
            with registry.lock():
                registry.append({"event": "restored", "version": name, "method": "wer_parity",
                                 "rebuilt_sha256": _put(tmp_path, f"swap-{name}"), "gguf_sha256": None})
        return real(*args, **kwargs)

    monkeypatch.setattr(ship, "_wait", wait)


def test_promote_refuses_a_gguf_swapped_after_selection(hexbin, tmp_path, monkeypatch):
    candidate(tmp_path)
    passing_gate("v1")
    selected = hex_id("v1")
    _swap_during_wait(monkeypatch, tmp_path, "v1")
    code, lines = promote(hexbin, "v1")
    assert code != 0 and events("promoted")[1:] == []
    assert hexbin.selected == CATALOG and hexbin.calls("use")[0][2] == selected


def test_rollback_refuses_a_gguf_swapped_after_selection(hexbin, tmp_path, monkeypatch):
    for n in (1, 2):
        candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        assert promote(hexbin, f"v{n}")[0] == 0
    v2 = hex_id("v2")
    _swap_during_wait(monkeypatch, tmp_path, "v1")
    code, lines = rollback(hexbin)
    assert code != 0 and events("rolled_back") == []
    assert registry.load().current == "v2" and hexbin.selected == v2


def test_promoted_event_records_the_exact_gguf(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    assert promote(hexbin, "v1")[0] == 0
    assert events("promoted")[-1]["gguf_sha256"] == registry.load().get("v1").gguf


def test_cleanup_never_removes_an_import_healing_did_not_create(hexbin, tmp_path):
    candidate(tmp_path)
    hexbin.set(imports={hex_id("v1"): "imported by hand"})  # the same file, imported by you first
    for n in range(1, 5):
        if n > 1:
            candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        assert promote(hexbin, f"v{n}")[0] == 0
    assert hex_id("v1") in hexbin.imports
    assert hexbin.calls("remove") == []
    assert [c[2] for c in hexbin.calls("import")] and len(hexbin.calls("import")) == 3


def test_cleanup_never_removes_a_found_model(hexbin, tmp_path):
    for n in range(1, 5):
        candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        if n == 4:
            hexbin.set(sources={hex_id("v1"): "found: someone/model@abcdef1"})
        assert promote(hexbin, f"v{n}")[0] == 0
    assert hex_id("v1") in hexbin.imports and hexbin.calls("remove") == []


def test_cleanup_forgets_what_it_removed(hexbin, tmp_path):
    """Once removed, a later import of the same file by hand is yours: cleanup leaves it."""
    for n in range(1, 5):
        candidate(tmp_path, parent=f"v{n - 1}")
        passing_gate(f"v{n}")
        assert promote(hexbin, f"v{n}")[0] == 0
    assert hex_id("v1") not in hexbin.imports
    hexbin.set(imports={**hexbin.imports, hex_id("v1"): "by hand"})
    candidate(tmp_path, parent="v4")
    passing_gate("v5")
    assert promote(hexbin, "v5")[0] == 0
    assert hex_id("v1") in hexbin.imports
    assert [c[2] for c in hexbin.calls("remove")] == [hex_id("v1"), hex_id("v2")]


def test_unconfirmed_rollback_switches_back(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    assert promote(hexbin, "v1")[0] == 0
    hexbin.set(stuck=[CATALOG])  # Hex takes the switch but never finishes preparing it
    before = registry.read_events()
    lines: list[str] = []
    code = ship.rollback(binary=hexbin.binary, say=lines.append, confirm_s=0.2, poll_s=0.01)
    assert code == ship.EXIT_MISMATCH and registry.read_events() == before
    assert hexbin.selected == hex_id("v1") and hexbin.calls("use")[-1][2] == hex_id("v1")
    assert any("switched Hex back" in line for line in lines)


def test_unconfirmed_rollback_says_when_switching_back_is_unconfirmed(hexbin, tmp_path):
    candidate(tmp_path)
    passing_gate("v1")
    assert promote(hexbin, "v1")[0] == 0
    hexbin.set(stuck=[CATALOG, hex_id("v1")])
    lines: list[str] = []
    code = ship.rollback(binary=hexbin.binary, say=lines.append, confirm_s=0.2, poll_s=0.01)
    assert code == ship.EXIT_MISMATCH
    assert any("could not confirm" in line and "hex status" in line for line in lines)


def test_recovery_target_is_the_model_hex_runs(hexbin, tmp_path):
    """Hex saved an import that's missing and runs the catalog model in its place: a failed promote
    returns Hex to what it ran, not to the saved id."""
    candidate(tmp_path)
    passing_gate("v1")
    hexbin.set(selected="custom-ffffffffffffffff", imports={"custom-ffffffffffffffff": "gone"},
               fallback={"for": "custom-ffffffffffffffff", "id": CATALOG, "reason": "missing"}, ignore_use=True)
    code, lines = promote(hexbin, "v1", confirm_s=0.2)
    assert code == ship.EXIT_MISMATCH
    assert hexbin.calls("use")[-1][2] == CATALOG
    assert events("promote_failed")[-1]["previous_hex_id"] == CATALOG
