"""`$HEALING_HOME/registry/versions.jsonl`: the append-only model registry.

Each line is one event, `{event, version, at_ms, ...}`: `registered` (the whole version record),
`scored`, `gate` (one `healing gate` decision, T7.1), `promoted`, `promote_failed` (Hex didn't confirm the
switch, T7.2; the version stays a candidate), `hex_import` and `hex_import_removed` (an import of the
version's GGUF that healing itself created in Hex, and its removal: the only imports cleanup may remove),
`rejected`, `rolled_back`, `restored`,
`pruned_artifact`, and the one `golden_frozen` pin. The current state is a fold of the events; nothing is ever rewritten.

Every append is one whole line written with a single `write` and fsynced, under `registry/.lock`. A
crash mid-append can only leave a torn last line: reads ignore it, and the next append cuts it off
before writing, so it never becomes a bad line in the middle. A bad line anywhere else is corruption
and is reported, never skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from healing.registry import objects
from healing.registry.objects import fsync_dir, owner_dir, registry_dir
from healing.store import locked, now_ms

EVENTS = ("registered", "scored", "gate", "promoted", "promote_failed", "hex_import", "hex_import_removed", "rejected", "rolled_back", "restored", "pruned_artifact", "golden_frozen")
ARTIFACT_KINDS = ("gguf", "lora_delta", "merged_nemo")
KEEP_PREVIOUS_PROMOTED = 2  # besides current and v0, the GGUFs of the last two promoted versions stay


class RegistryError(RuntimeError):
    pass


class GoldenPinRefused(RegistryError):
    """G-10: the golden set is pinned once; a different hash is refused."""


def versions_path() -> Path:
    return registry_dir() / "versions.jsonl"


def lock():
    owner_dir(registry_dir())
    return locked(registry_dir() / ".lock", wait=True)


def ship_lock():
    """One promote, rollback or restore at a time: a restore swaps a version's GGUF, which must not
    happen while a promote or rollback selects that GGUF in Hex (T7.2). Never waits; a held lock
    raises `StoreError`."""
    owner_dir(registry_dir())
    return locked(registry_dir() / ".ship.lock", wait=False, busy="another promote, rollback or restore is running")


def _check_gguf(version: "Version", expect_gguf: str | None) -> None:
    if expect_gguf is not None and version.gguf != expect_gguf:
        raise RegistryError(f"{version.name}'s GGUF is {str(version.gguf)[:12]} now, not the {expect_gguf[:12]} "
                            "selected in Hex (restored meanwhile?)")


# ---- the log ------------------------------------------------------------------------------------

def read_events(path: Path | None = None) -> list[dict]:
    """Every whole event in order. A torn last line (whatever follows the last newline) is ignored."""
    path = path or versions_path()
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    lines = data.split(b"\n")
    lines.pop()  # after the last newline: empty, or a line a crash cut short
    events = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as error:
            raise RegistryError(f"{path}: line {number} is not JSON; the registry is damaged") from error
        if not isinstance(event, dict) or "event" not in event:
            raise RegistryError(f"{path}: line {number} is not an event")
        events.append(event)
    return events


def _cut_torn_tail(descriptor: int) -> None:
    """Drop whatever follows the last newline (what reads ignore), so the next line starts on its own."""
    size = os.fstat(descriptor).st_size
    if size == 0:
        return
    os.lseek(descriptor, 0, os.SEEK_SET)
    end = os.read(descriptor, size).rfind(b"\n") + 1
    if end < size:
        os.ftruncate(descriptor, end)
        os.fsync(descriptor)


def append(*events: dict) -> None:
    """Append events, one whole line each, fsynced. The caller holds `lock()`."""
    owner_dir(registry_dir())
    path = versions_path()
    fresh = not path.exists()
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _cut_torn_tail(descriptor)
        for event in events:
            if event.get("event") not in EVENTS:
                raise RegistryError(f"unknown event {event.get('event')!r}")
            line = (json.dumps({"at_ms": now_ms(), **event}, sort_keys=True, separators=(",", ":")) + "\n").encode()
            os.lseek(descriptor, 0, os.SEEK_END)
            if os.write(descriptor, line) != len(line):
                raise RegistryError("a registry line was written short")
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if fresh:
        fsync_dir(path.parent)


# ---- the fold -----------------------------------------------------------------------------------

@dataclass
class Version:
    name: str
    record: dict
    state: str = "candidate"
    reason: str | None = None
    scores: dict = field(default_factory=dict)
    restores: list[dict] = field(default_factory=list)
    pruned: dict[str, str] = field(default_factory=dict)  # sha pruned -> artifact kind
    gates: list[dict] = field(default_factory=list)  # every `healing gate` decision, oldest first
    promote_failures: list[dict] = field(default_factory=list)  # every `promote_failed`, oldest first

    @property
    def number(self) -> int:
        return int(self.name[1:])

    def artifact(self, kind: str) -> str | None:
        return (self.record.get("artifacts") or {}).get(kind)

    @property
    def gguf(self) -> str | None:
        """The GGUF Hex would run: the latest successful restore's (a hash match is the recorded file, a
        parity restore its rebuild), else the recorded one."""
        for restore in reversed(self.restores):
            found = restore.get("gguf_sha256") or restore.get("rebuilt_sha256")
            if found:
                return found
        return self.artifact("gguf")

    def ggufs(self) -> set[str]:
        """Every GGUF hash that is this version: recorded and restored."""
        restored = [r.get("gguf_sha256") or r.get("rebuilt_sha256") for r in self.restores]
        return {s for s in [self.artifact("gguf"), *restored] if s}

    @property
    def last_gate(self) -> dict | None:
        """The latest gate decision (what `healing promote` reads), or None when never gated."""
        return self.gates[-1] if self.gates else None

    def as_dict(self) -> dict:
        return {**self.record, "version": self.name, "state": self.state, "reason": self.reason,
                "scores": self.scores, "restores": self.restores, "pruned": self.pruned, "gates": self.gates,
                "promote_failures": self.promote_failures}


@dataclass
class State:
    versions: dict[str, Version] = field(default_factory=dict)
    current: str | None = None
    promotions: list[str] = field(default_factory=list)  # promotion order, a version again if promoted again
    golden: dict | None = None
    owned_imports: dict[str, str] = field(default_factory=dict)  # Hex id healing imported -> its version

    def get(self, name: str) -> Version:
        if name == "current":
            if self.current is None:
                raise RegistryError("no version is current; run `healing models register-base`")
            name = self.current
        try:
            return self.versions[name]
        except KeyError:
            raise RegistryError(f"no version {name!r}") from None

    def next_name(self) -> str:
        return f"v{max((v.number for v in self.versions.values()), default=-1) + 1}"

    def ordered(self) -> list[Version]:
        return sorted(self.versions.values(), key=lambda v: v.number)


def fold(events: list[dict]) -> State:
    state = State()
    for event in events:
        kind, name = event["event"], event.get("version")
        if kind == "golden_frozen":
            if state.golden is None:
                state.golden = {"sha256": event["sha256"], "n": event.get("n"), "at_ms": event.get("at_ms")}
            continue
        if kind == "registered":
            if name in state.versions:
                raise RegistryError(f"{name} is registered twice")
            state.versions[name] = Version(name, {k: v for k, v in event.items() if k not in ("event", "version")})
            _republished(state, (event.get("artifacts") or {}).values())
            continue
        if kind not in EVENTS:
            continue  # a newer healing's event: leave it to that healing
        version = state.versions.get(name)
        if version is None:
            raise RegistryError(f"a {kind} event names {name!r}, which was never registered")
        if kind == "scored":
            version.scores.update(event.get("scores") or {})
        elif kind == "gate":
            version.gates.append({k: v for k, v in event.items() if k not in ("event", "version")})
        elif kind == "promoted":
            version.state, version.reason = "promoted", event.get("reason")
            state.current = name
            state.promotions.append(name)
        elif kind == "hex_import":
            state.owned_imports[event["hex_id"]] = name
        elif kind == "hex_import_removed":
            state.owned_imports.pop(event["hex_id"], None)
        elif kind == "promote_failed":
            version.promote_failures.append({k: v for k, v in event.items() if k not in ("event", "version")})
        elif kind == "rejected":
            version.state, version.reason = "rejected", event.get("reason")
        elif kind == "rolled_back":
            version.state, version.reason = "rolled_back", event.get("reason")
            if state.current == name:
                state.current = event.get("to") or _fallback(state, name)
        elif kind == "restored":
            version.restores.append({k: v for k, v in event.items() if k not in ("event", "version")})
            _republished(state, (event.get("gguf_sha256"), event.get("rebuilt_sha256")))
        elif kind == "pruned_artifact":
            version.pruned[event["sha256"]] = event["kind"]
    return state


def _republished(state: State, shas) -> None:
    """An event that stores an object (a registration, a restore) recreates it: every earlier prune of it
    is complete, so deleting it again is a new deletion with its own events. A prune marker left
    standing therefore means only a pending unlink (a crash between the event and the unlink)."""
    for sha in filter(None, shas):
        for version in state.versions.values():
            version.pruned.pop(sha, None)


def _fallback(state: State, leaving: str) -> str | None:
    """After a rollback with no `to`: the latest promoted version still promoted."""
    for name in reversed(state.promotions):
        if name != leaving and state.versions[name].state == "promoted":
            return name
    return None


def load() -> State:
    return fold(read_events())


# ---- writes -------------------------------------------------------------------------------------

def register(record: dict, publish: list[tuple[Path, str]] = (), scores: dict | None = None,
             expect: str | None = None) -> str:
    """Store `publish` (each `(file, sha256)`) as objects and append the `registered` event (and `scored`,
    when given) for the next version number, all under one hold of the lock, so retention can never
    delete an object between its store and the event that names it. Returns the name. With `expect`,
    the next number must be that name (an export writes it into the GGUF before registering)."""
    with lock():
        state = load()
        name = state.next_name()
        if expect is not None and name != expect:
            raise RegistryError(f"another version took {expect} while this one was built; the next is {name}, so export again")
        if record.get("parent") and record["parent"] not in state.versions:
            raise RegistryError(f"parent {record['parent']} is not registered")
        for source, sha in publish:
            objects.put(source, sha)
        append({"event": "registered", "version": name, **record},
               *([{"event": "scored", "version": name, "scores": scores}] if scores else []))
        retain()
        return name


# Every event names the resolved version (`v<n>`), never the alias `current`, which fold can't read.

def record_scores(name: str, scores: dict) -> None:
    with lock():
        version = load().get(name)
        append({"event": "scored", "version": version.name, "scores": scores})


def promote(name: str, reason: str | None = None, *, expect_current: str | None = None,
            check: Callable[[State, Version], str | None] | None = None, expect_gguf: str | None = None,
            **fields) -> None:
    """Make a candidate current. With `expect_current`, the current version must still be that one, and
    `check(state, version)` (a problem, or None) is asked again, both under the lock, so of two promotes
    started from the same current only the first is recorded. With `expect_gguf`, the version's GGUF
    must still be the one Hex was switched to, and the event records it."""
    with lock():
        state = load()
        version = state.get(name)
        if version.state != "candidate":
            raise RegistryError(f"{version.name} is {version.state}; only a candidate is promoted")
        if expect_current is not None and state.current != expect_current:
            raise RegistryError(f"the current version is {state.current} now, not {expect_current}; "
                                "another promote or rollback ran meanwhile")
        problem = check(state, version) if check else None
        if problem:
            raise RegistryError(problem)
        _check_gguf(version, expect_gguf)
        append({"event": "promoted", "version": version.name, "reason": reason,
                **({"gguf_sha256": expect_gguf} if expect_gguf else {}), **fields})
        retain()


def record_hex_import(name: str, hex_id: str, gguf_sha256: str, removed: bool = False) -> None:
    """Healing created (or, `removed`, removed) the Hex import `hex_id` of `name`'s GGUF."""
    with lock():
        version = load().get(name)
        append({"event": "hex_import_removed" if removed else "hex_import", "version": version.name,
                "hex_id": hex_id, "gguf_sha256": gguf_sha256})


def record_promote_failure(name: str, reason: str, **fields) -> None:
    """Hex didn't confirm a promote's switch: the version stays a candidate (promote may run again)."""
    with lock():
        version = load().get(name)
        if version.state != "candidate":
            raise RegistryError(f"{version.name} is {version.state}; only a candidate's promote fails")
        append({"event": "promote_failed", "version": version.name, "reason": reason, **fields})


def reject(name: str, reason: str) -> None:
    with lock():
        version = load().get(name)
        if version.state != "candidate":
            raise RegistryError(f"{version.name} is {version.state}; only a candidate is rejected")
        append({"event": "rejected", "version": version.name, "reason": reason})
        retain()


def record_gate(name: str, decision: dict, reject_reason: str | None) -> None:
    """Append a `gate` decision for a candidate and, when it rejects, the `rejected` event after it,
    under one hold of the lock so no other decision lands between them. The decision's `models`
    (`{role: {version, gguf_sha256}}`) are the GGUFs the eval scored; each must still be its version's
    GGUF now, so a restore that swapped one during the eval can't carry a decision about other bytes."""
    with lock():
        state = load()
        version = state.get(name)
        if version.state != "candidate":
            raise RegistryError(f"{version.name} is {version.state}; only a candidate is gated")
        for role, model in (decision.get("models") or {}).items():
            now = state.get(model["version"]).gguf
            if now != model["gguf_sha256"]:
                raise RegistryError(f"{model['version']} ({role}) is GGUF {str(now)[:12]} now, not the "
                                    f"{model['gguf_sha256'][:12]} the eval scored (restored meanwhile?); gate it again")
        events = [{"event": "gate", "version": version.name, **decision}]
        if reject_reason is not None:
            events.append({"event": "rejected", "version": version.name, "reason": reject_reason})
        append(*events)
        if reject_reason is not None:
            retain()


def roll_back(name: str, reason: str, to: str | None = None, *, expect_current: str | None = None,
              expect_gguf: str | None = None, **fields) -> None:
    """Roll back a promoted version. When it is current, a distinct, still-promoted successor (`to`, else
    the latest other promoted version) becomes current, so exactly one version stays current. With
    `expect_current`, the current version must still be that one; with `expect_gguf`, the successor's GGUF
    must still be the one Hex was switched to (recorded in the event)."""
    with lock():
        state = load()
        if expect_current is not None and state.current != expect_current:
            raise RegistryError(f"the current version is {state.current} now, not {expect_current}; "
                                "another promote or rollback ran meanwhile")
        version = state.get(name)
        if version.state != "promoted":
            raise RegistryError(f"{version.name} is {version.state}; only a promoted version rolls back")
        successor = state.get(to).name if to is not None else None
        if successor == version.name:
            raise RegistryError(f"{version.name} cannot roll back to itself")
        if successor is not None and state.versions[successor].state != "promoted":
            raise RegistryError(f"{successor} is not promoted, so it cannot become current")
        if version.name == state.current and successor is None:
            successor = _fallback(state, version.name)
            if successor is None:
                raise RegistryError(f"{version.name} is current and no other promoted version could take over")
        if successor is not None:
            _check_gguf(state.versions[successor], expect_gguf)
        append({"event": "rolled_back", "version": version.name, "reason": reason,
                **({"to": successor} if successor else {}),
                **({"gguf_sha256": expect_gguf} if expect_gguf else {}), **fields})
        retain()


def pin_golden(sha256: str, n: int) -> bool:
    """Pin the frozen golden set's manifest hash (G-10). True when written now, False when this hash is
    already pinned; a different hash is refused."""
    with lock():
        pinned = load().golden
        if pinned is not None:
            if pinned["sha256"] == sha256:
                return False
            raise GoldenPinRefused(
                f"the golden set is pinned at {pinned['sha256'][:12]}; refusing {sha256[:12]} (G-10: it never changes)"
            )
        append({"event": "golden_frozen", "sha256": sha256, "n": n})
        return True


# ---- retention ----------------------------------------------------------------------------------

def promoted_keepers(state: State) -> set[str]:
    """v0, current and the previous two promoted versions: the GGUFs retention keeps for a rollback, and
    the imports `healing promote` leaves in Hex (T7.2)."""
    keep = {"v0"} if "v0" in state.versions else set()
    if state.current:
        keep.add(state.current)
    previous = []
    for name in reversed(state.promotions):
        if name != state.current and name not in previous:
            previous.append(name)
        if len(previous) == KEEP_PREVIOUS_PROMOTED:
            break
    keep.update(previous)
    return keep


def gguf_keepers(state: State) -> set[str]:
    """Versions whose GGUF stays: `promoted_keepers` and undecided candidates (a candidate still waiting
    for its gate needs its file; once rejected it goes)."""
    return promoted_keepers(state) | {v.name for v in state.versions.values() if v.state == "candidate"}


def retain() -> list[tuple[str, str, str]]:
    """Delete the GGUFs (and merged NeMo checkpoints, dropped once exported) retention no longer keeps,
    appending `pruned_artifact` (before the unlink) for each file removed. Metadata, snapshots and LoRA deltas stay forever.
    An object a kept version still names is not deleted, and a version restored on demand keeps its
    file only until the next state change. The caller holds `lock()`."""
    state = load()
    keepers = gguf_keepers(state)
    needed = {sha for name in keepers for sha in state.versions[name].ggufs()}
    needed |= {v.artifact("lora_delta") for v in state.versions.values()} - {None}
    holders: dict[str, list[tuple[Version, str]]] = {}  # sha -> every (version, kind) that lets it go
    for version in state.ordered():
        if version.name not in keepers:
            for sha in sorted(version.ggufs()):
                holders.setdefault(sha, []).append((version, "gguf"))
        if version.artifact("merged_nemo"):
            holders.setdefault(version.artifact("merged_nemo"), []).append((version, "merged_nemo"))
    pruned = []
    for sha, who in holders.items():
        if sha in needed or not objects.path(sha).is_file():
            continue
        # The events are the intent and go first, one per version naming the file: a crash before the
        # unlink leaves markers whose file still exists, which the next retain removes with no new events.
        for version, kind in who:
            if version.pruned.get(sha) != kind:
                append({"event": "pruned_artifact", "version": version.name, "kind": kind, "sha256": sha})
                version.pruned[sha] = kind
                pruned.append((version.name, kind, sha))
        objects.remove(sha)
    return pruned
