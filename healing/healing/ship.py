"""`healing promote <v>` and `healing rollback [--to <v>] [--reason <text>]` (T7.2, AC-13).

`promote` is the only command that changes which model Hex runs. It takes a candidate whose latest
gate passed against the version that is current now, on the GGUFs the registry holds now (D-55), then:

1. `hex model import <its registry object> --name "Hex healing v<n>"` (skipped when Hex already has it;
   no `--language`: Parakeet imports take no language hint; an import healing creates is recorded as
   `hex_import`),
2. `hex model use custom-<16 hex>`,
3. waits up to 60 s for `hex status --json` to report that model, installed, with nothing preparing and
   no fallback, and for `hex model list --json` to say it `streams` when it should (D-56),
4. appends `promoted` and makes it current, then removes from Hex every import healing created
   (`hex_import`, never one you or another app made, whatever its hash) except current's, the previous
   two's and v0's (D-57).

When Hex doesn't confirm, `promote` switches Hex back to the model it ran before, appends
`promote_failed` (the version stays a candidate) and exits 9. `rollback` switches Hex to the previous
version (or `--to`), importing it again from the registry when Hex no longer has it, and appends
`rolled_back` with the reason only once `hex status` confirms; an unconfirmed rollback switches Hex back
to what it ran and records nothing.

Exit codes: 2 Hex isn't running (nothing changes), 8 no passing gate, 9 Hex didn't confirm the switch,
10 the candidate must stream but Labs › Stream imported models is off, 1 anything else.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from healing import hexcli
from healing.registry import objects
from healing.registry import store as registry
from healing.registry.store import RegistryError, State, Version
from healing.store import StoreError

EXIT_HEX_DOWN = 2
EXIT_NO_GATE = 8
EXIT_MISMATCH = 9
EXIT_LAB_OFF = 10
EXIT_SWITCHED = 11  # `healing watch`'s rollback: Hex no longer runs the version it would roll back
CONFIRM_S = 60.0
POLL_S = 1.0
IMPORT_TIMEOUT_S = 900  # `hex model import` copies and hashes ~700 MB
STREAM_LAB = "stream-imported-models"
USE_REFUSALS = frozenset({"unknown-model", "model-unavailable", "not-installed", "busy", "not-ready", "invalid-value"})


class ShipError(RuntimeError):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def ship_lock():
    """One promote, rollback or restore at a time, held from the first `hex` call to the registry write."""
    return registry.ship_lock()


def hex_id(version: Version) -> str:
    """The model id Hex knows `version` by: v0's catalog id, else `custom-` + its GGUF's first 16 hex."""
    catalog_id = (version.record.get("base") or {}).get("catalog_id")
    if version.name == "v0" and catalog_id:
        return catalog_id
    return objects.hex_model_id(version.gguf)


def gate_problem(state: State, version: Version) -> str | None:
    """Why `version` has no passing gate now, or None. Only its latest gate counts, and only when it
    passed, was not refused, was judged against the version current now, and judged the GGUFs the
    candidate and current hold now (a restore that swapped either one voids it)."""
    name, gate = version.name, version.last_gate
    if version.state == "rolled_back":
        return (f"{name} was rolled back, and a rolled-back version is never promoted again; "
                "a new training round makes a new version (D-60)")
    if version.state != "candidate":
        return f"{name} is {version.state}; only a candidate that passed the gate is promoted"
    if gate is None:
        return f"{name} was never gated; run `healing gate {name}`"
    if gate.get("refused"):
        return f"the gate refused {name} for thin evidence; review more clips and gate it again"
    if gate.get("pass") is not True:
        return f"{name}'s latest gate did not pass"
    current = state.get("current")
    if gate.get("against") != current.name:
        return (f"{name} was gated against {gate.get('against')}, but the current version is {current.name}; "
                f"gate it again (`healing gate {name}`)")
    models = gate.get("models") or {}
    for role, now in (("candidate", version), ("against", current)):
        judged = (models.get(role) or {}).get("gguf_sha256")
        if judged != now.gguf:
            return (f"the gate judged {now.name}'s GGUF {str(judged)[:12]}, but it is {str(now.gguf)[:12]} now; "
                    f"gate {name} again")
    return None


# ---- talking to Hex --------------------------------------------------------------------------------

def _reply(binary: Path, args: list[str], status: str) -> dict:
    try:
        done = hexcli.run(binary, args)
    except hexcli.HexNotRunning as error:
        raise ShipError(f"{error}; start it and run this again", EXIT_HEX_DOWN) from error
    except hexcli.HexError as error:
        raise ShipError(str(error)) from error
    reply = hexcli.reply(done.stdout)
    if not reply or reply.get("status") != status:
        raise ShipError(f"hex {' '.join(args[:2])} printed something this version cannot read")
    return reply


def _status(binary: Path) -> dict:
    reply = _reply(binary, ["status", "--json"], "status")
    if not isinstance(reply.get("model"), dict) or not isinstance(reply["model"].get("id"), str):
        raise ShipError("hex status --json printed no model this version can read")
    return reply


def _models(binary: Path) -> dict[str, dict]:
    reply = _reply(binary, ["model", "list", "--json"], "models")
    return {m.get("id"): m for m in reply.get("models") or [] if isinstance(m, dict)}


def _running(status: dict) -> str:
    """The model Hex runs now: `model.id` is only the saved selection, and while a saved import is
    missing, changed or switched off Hex runs `model.fallback.id` in its place. It is what a failed
    promote or rollback returns Hex to."""
    fallback = status["model"].get("fallback")
    if isinstance(fallback, dict) and isinstance(fallback.get("id"), str):
        return fallback["id"]
    return status["model"]["id"]


def _lab_on(status: dict) -> bool:
    """Labs › Stream imported models, as the running Hex reports it (`labs`), else its settings file."""
    if isinstance(status.get("labs"), list):
        return STREAM_LAB in status["labs"]
    from healing.gate import _lab_on as saved_lab_on

    return saved_lab_on()


def _confirmed(model: dict, target: str) -> str | None:
    """Why `hex status`'s `model` isn't `target` running, or None when it is."""
    if model.get("id") != target:
        return f"hex status shows model {model.get('id')}, not {target}"
    fallback = model.get("fallback")
    if isinstance(fallback, dict) and fallback.get("id"):
        return f"hex status shows {target} saved but {fallback['id']} running in its place ({fallback.get('reason')})"
    if model.get("installed") is not True:
        return f"hex status shows {target} not installed"
    if model.get("preparing"):
        return f"hex status shows {model['preparing']} still preparing"
    if model.get("error"):
        return f"hex status shows {target} failed: {model['error']}"
    return None


def _wait(binary: Path, target: str, *, streams: bool, confirm_s: float, poll_s: float) -> str | None:
    """Wait for Hex to run `target` (and stream it, when `streams`): None once it does, else the last
    thing Hex showed. Hex switches only when idle, so a dictation in progress delays it."""
    deadline = time.monotonic() + confirm_s
    while True:
        try:
            problem = _confirmed(_status(binary)["model"], target)
            if problem is None and streams and _models(binary).get(target, {}).get("streams") is not True:
                problem = f"Hex runs {target} but doesn't stream it (its smoke stream failed, or it isn't streaming yet)"
        except ShipError as error:
            problem = f"hex status could not be read: {error}"
        if problem is None:
            return None
        if time.monotonic() >= deadline:
            return f"{problem} after {confirm_s:g} s"
        time.sleep(poll_s)


def _use(binary: Path, model_id: str) -> str | None:
    """`hex model use` (never `--language`): None when sent (the status poll decides), else why Hex refused."""
    try:
        hexcli.run(binary, ["model", "use", model_id, "--json"], write=True, refusal_codes=USE_REFUSALS)
    except hexcli.HexNotRunning as error:
        raise ShipError(f"{error}; start it and run this again", EXIT_HEX_DOWN) from error
    except hexcli.HexOutcomeUnknown:
        return None
    except hexcli.HexError as error:
        return f"hex model use refused {model_id}: {error}"
    return None


def _ensure_imported(binary: Path, version: Version, listed: dict[str, dict], say: Callable[[str], None]) -> str:
    """Hex has `version` installed: import its registry GGUF when Hex lacks it (or its copy is missing or
    changed, which a re-import repairs). Returns its Hex id."""
    target = hex_id(version)
    if (listed.get(target) or {}).get("installed") is True:
        return target
    created = target not in listed  # a listed copy that is missing or changed is repaired, not created
    if not target.startswith(objects.CUSTOM_PREFIX):
        raise ShipError(f"{version.name}'s catalog model {target} isn't installed in Hex; `hex model install {target}`")
    if not objects.has(version.gguf):
        raise ShipError(f"{version.name}'s GGUF {str(version.gguf)[:12]} is not kept; run `healing models restore {version.name}`")
    say(f"importing {version.name} into Hex as {target}")
    try:
        done = hexcli.run(binary, ["model", "import", str(objects.path(version.gguf)), "--name", f"Hex healing {version.name}"],
                          timeout=IMPORT_TIMEOUT_S)
    except hexcli.HexError as error:
        raise ShipError(f"hex model import failed: {error}") from error
    fields = next((line.split("\t") for line in done.stdout.splitlines() if line.startswith("imported\t")), None)
    if not fields or len(fields) < 2 or fields[1] != target:
        raise ShipError(f"hex model import printed {done.stdout.strip()!r}, not the id {target}")
    if created:
        registry.record_hex_import(version.name, target, version.gguf)
    return target


def _should_stream(version: Version, current: Version, listed: dict[str, dict]) -> bool:
    """The candidate must stream once Hex runs it: its gate's streaming check passed (so it streams like the
    model it replaces), or Hex streams the current version now."""
    checks = (version.last_gate or {}).get("checks") or []
    gated = any(c.get("name") == "streaming" and c.get("status") == "pass" for c in checks)
    return gated or (listed.get(hex_id(current)) or {}).get("streams") is True


# ---- promote ---------------------------------------------------------------------------------------

def _promote(name: str, binary: Path, say: Callable[[str], None], confirm_s: float, poll_s: float) -> int:
    state = registry.load()
    version = state.get(name)
    problem = gate_problem(state, version)
    if problem:
        raise ShipError(problem, EXIT_NO_GATE)
    current = state.get("current")
    status = _status(binary)
    previous_id = _running(status)
    listed = _models(binary)
    streams = _should_stream(version, current, listed)
    if streams and not _lab_on(status):
        raise ShipError(f"{version.name} must stream like {current.name}, and an import streams only with the lab: "
                        f"turn on Labs › Stream imported models (`hex labs on {STREAM_LAB}`, D-26)", EXIT_LAB_OFF)
    selected = version.gguf  # the exact file Hex is switched to; the registry write checks it again
    target = _ensure_imported(binary, version, listed, say)
    say(f"switching Hex from {previous_id} to {target}")
    problem = _use(binary, target) or _wait(binary, target, streams=streams, confirm_s=confirm_s, poll_s=poll_s)
    if problem is None:
        try:
            registry.promote(version.name, "gate passed", expect_current=current.name, check=gate_problem,
                             expect_gguf=selected, hex_id=target, previous=current.name, gate_eval_id=version.last_gate.get("eval_id"))
        except RegistryError as error:
            problem = f"the registry refused the promotion: {error}"
            _switch_back(binary, previous_id, target, say, confirm_s, poll_s)
            raise ShipError(problem) from error
        say(f"{version.name} is current; Hex runs {target}")
        _cleanup(binary, say)
        return 0
    say(f"promote failed: {problem}")
    _switch_back(binary, previous_id, target, say, confirm_s, poll_s)
    registry.record_promote_failure(version.name, problem, hex_id=target, previous_hex_id=previous_id)
    say(f"{version.name} stays a candidate; `healing promote {version.name}` tries again")
    return EXIT_MISMATCH


def _switch_back(binary: Path, previous_id: str, target: str, say, confirm_s: float, poll_s: float) -> None:
    if previous_id == target:
        return
    try:
        problem = _use(binary, previous_id) or _wait(binary, previous_id, streams=False, confirm_s=confirm_s, poll_s=poll_s)
    except ShipError as error:
        problem = str(error)
    say(f"switched Hex back to {previous_id}" if problem is None
        else f"could not confirm Hex is back on {previous_id} ({problem}); check `hex status`")


def _cleanup(binary: Path, say: Callable[[str], None]) -> None:
    """Remove every import healing created (`hex_import`, not since removed) except current's, the
    previous two's and v0's. Anything else (your own import of the same file, a found Hugging Face
    model) is never touched, and neither is the selected model."""
    state = registry.load()
    ours = state.owned_imports
    keep = {objects.hex_model_id(sha) for n in registry.promoted_keepers(state) for sha in state.versions[n].ggufs()}
    try:
        listed = _models(binary)
    except ShipError as error:
        say(f"cleanup skipped: {error}")
        return
    for model_id, model in listed.items():
        if model_id not in ours or model_id in keep or model.get("selected") or model.get("source") != "imported":
            continue
        try:
            hexcli.run(binary, ["model", "remove", model_id])
        except hexcli.HexError as error:
            say(f"could not remove {model_id}: {error}")
            continue
        sha = next(s for s in state.versions[ours[model_id]].ggufs() if objects.hex_model_id(s) == model_id)
        registry.record_hex_import(ours[model_id], model_id, sha, removed=True)
        say(f"removed {ours[model_id]}'s import {model_id} from Hex")


# ---- rollback --------------------------------------------------------------------------------------

def _rollback(to: str | None, reason: str, binary: Path, say: Callable[[str], None], confirm_s: float,
              poll_s: float, expect_leaving: str | None = None, expect_running: set[str] | None = None,
              expect_ggufs: dict[str, str] | None = None, fields: dict | None = None) -> int:
    state = registry.load()
    leaving = state.get("current")
    if expect_leaving is not None and leaving.name != expect_leaving:
        raise ShipError(f"the current version is {leaving.name} now, not {expect_leaving}; nothing rolled back")
    for name, sha in (expect_ggufs or {}).items():  # under the ship lock: no restore can swap one from here on
        if state.get(name).gguf != sha:
            raise ShipError(f"{name}'s GGUF is {str(state.get(name).gguf)[:12]} now, not the {sha[:12]} that was "
                            "scored (restored meanwhile?); nothing rolled back")
    target_name = state.get(to).name if to is not None else registry._fallback(state, leaving.name)
    if target_name is None:
        raise ShipError(f"{leaving.name} is current and no other promoted version could take over")
    target = state.versions[target_name]
    if target.name == leaving.name:
        raise ShipError(f"{leaving.name} cannot roll back to itself")
    if target.state != "promoted":
        raise ShipError(f"{target.name} is {target.state}; only a promoted version becomes current again")
    original = _running(_status(binary))
    if expect_running is not None and original not in expect_running:
        raise ShipError(f"Hex runs {original}, not {leaving.name}: you switched models by hand, so nothing is "
                        "rolled back and your choice stays", EXIT_SWITCHED)
    selected = target.gguf  # the exact file Hex is switched to; the registry write checks it again
    model_id = _ensure_imported(binary, target, _models(binary), say)
    say(f"switching Hex from {leaving.name} to {target.name} ({model_id})")
    problem = _use(binary, model_id) or _wait(binary, model_id, streams=False, confirm_s=confirm_s, poll_s=poll_s)
    if problem is not None:
        say(f"rollback failed: {problem}; the registry is unchanged")
        _switch_back(binary, original, model_id, say, confirm_s, poll_s)  # or the queued switch lands later
        return EXIT_MISMATCH
    try:
        registry.roll_back(leaving.name, reason, to=target.name, expect_current=leaving.name, expect_gguf=selected,
                           hex_id=model_id, **(fields or {}))
    except RegistryError as error:
        _switch_back(binary, original, model_id, say, confirm_s, poll_s)
        raise ShipError(f"the registry refused the rollback: {error}") from error
    say(f"{leaving.name} rolled back ({reason}); {target.name} is current and Hex runs {model_id}")
    return 0


def _guarded(work: Callable[[], int], say: Callable[[str], None]) -> int:
    try:
        with ship_lock():
            return work()
    except ShipError as error:
        say(f"error: {error}")
        return error.code
    except (RegistryError, StoreError, OSError) as error:
        say(f"error: {error}")
        return 1


def promote(name: str, *, binary: Path, say: Callable[[str], None] = print, confirm_s: float = CONFIRM_S,
            poll_s: float = POLL_S) -> int:
    return _guarded(lambda: _promote(name, Path(binary), say, confirm_s, poll_s), say)


def rollback(*, binary: Path, to: str | None = None, reason: str | None = None, say: Callable[[str], None] = print,
             confirm_s: float = CONFIRM_S, poll_s: float = POLL_S, expect_leaving: str | None = None,
             expect_running: set[str] | None = None, expect_ggufs: dict[str, str] | None = None,
             fields: dict | None = None) -> int:
    """`healing watch`'s guards, all checked under the ship lock right before Hex is switched: roll back
    only while `expect_leaving` is current, Hex runs one of `expect_running` (else exit 11: you switched
    by hand), and each version in `expect_ggufs` still has that GGUF. `fields` ride on the `rolled_back`
    event (the watch decision, in the same line)."""
    return _guarded(lambda: _rollback(to, reason or "manual", Path(binary), say, confirm_s, poll_s, expect_leaving,
                                      expect_running, expect_ggufs, fields), say)


def main(args) -> int:
    from healing.eval.bench import history_hex

    binary = Path(args.hex or history_hex())
    if args.group == "promote":
        return promote(args.version, binary=binary)
    return rollback(binary=binary, to=args.to, reason=args.reason)
