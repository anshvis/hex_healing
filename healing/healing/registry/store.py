"""`$HEALING_HOME/registry/versions.jsonl`: the append-only model registry.

Each line is one event, `{event, version, at_ms, ...}`: `registered` (the whole version record),
`scored`, `promoted`, `rejected`, `rolled_back`, `restored`, `pruned_artifact`, and the one
`golden_frozen` pin. The current state is a fold of the events; nothing is ever rewritten.

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

from healing.registry import objects
from healing.registry.objects import fsync_dir, owner_dir, registry_dir
from healing.store import locked, now_ms

EVENTS = ("registered", "scored", "promoted", "rejected", "rolled_back", "restored", "pruned_artifact", "golden_frozen")
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

    @property
    def number(self) -> int:
        return int(self.name[1:])

    def artifact(self, kind: str) -> str | None:
        return (self.record.get("artifacts") or {}).get(kind)

    @property
    def gguf(self) -> str | None:
        """The GGUF Hex would run: the recorded one, or a parity-checked rebuild's after a restore."""
        for restore in reversed(self.restores):
            if restore.get("gguf_sha256"):
                return restore["gguf_sha256"]
        return self.artifact("gguf")

    def ggufs(self) -> set[str]:
        """Every GGUF hash that is this version: recorded and restored."""
        return {s for s in [self.artifact("gguf"), *(r.get("gguf_sha256") for r in self.restores)] if s}

    def as_dict(self) -> dict:
        return {**self.record, "version": self.name, "state": self.state, "reason": self.reason,
                "scores": self.scores, "restores": self.restores, "pruned": self.pruned}


@dataclass
class State:
    versions: dict[str, Version] = field(default_factory=dict)
    current: str | None = None
    promotions: list[str] = field(default_factory=list)  # promotion order, a version again if promoted again
    golden: dict | None = None

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
            continue
        if kind not in EVENTS:
            continue  # a newer healing's event: leave it to that healing
        version = state.versions.get(name)
        if version is None:
            raise RegistryError(f"a {kind} event names {name!r}, which was never registered")
        if kind == "scored":
            version.scores.update(event.get("scores") or {})
        elif kind == "promoted":
            version.state, version.reason = "promoted", event.get("reason")
            state.current = name
            state.promotions.append(name)
        elif kind == "rejected":
            version.state, version.reason = "rejected", event.get("reason")
        elif kind == "rolled_back":
            version.state, version.reason = "rolled_back", event.get("reason")
            if state.current == name:
                state.current = event.get("to") or _fallback(state, name)
        elif kind == "restored":
            version.restores.append({k: v for k, v in event.items() if k not in ("event", "version")})
        elif kind == "pruned_artifact":
            version.pruned[event["sha256"]] = event["kind"]
    return state


def _fallback(state: State, leaving: str) -> str | None:
    """After a rollback with no `to`: the latest promoted version still promoted."""
    for name in reversed(state.promotions):
        if name != leaving and state.versions[name].state == "promoted":
            return name
    return None


def load() -> State:
    return fold(read_events())


# ---- writes -------------------------------------------------------------------------------------

def register(record: dict) -> str:
    """Append a `registered` event for the next version number; returns the name. Holds the lock."""
    with lock():
        state = load()
        name = state.next_name()
        if record.get("parent") and record["parent"] not in state.versions:
            raise RegistryError(f"parent {record['parent']} is not registered")
        append({"event": "registered", "version": name, **record})
        retain()
        return name


def record_scores(name: str, scores: dict) -> None:
    with lock():
        load().get(name)
        append({"event": "scored", "version": name, "scores": scores})


def promote(name: str, reason: str | None = None) -> None:
    with lock():
        version = load().get(name)
        if version.state != "candidate":
            raise RegistryError(f"{name} is {version.state}; only a candidate is promoted")
        append({"event": "promoted", "version": name, "reason": reason})
        retain()


def reject(name: str, reason: str) -> None:
    with lock():
        version = load().get(name)
        if version.state != "candidate":
            raise RegistryError(f"{name} is {version.state}; only a candidate is rejected")
        append({"event": "rejected", "version": name, "reason": reason})
        retain()


def roll_back(name: str, reason: str, to: str | None = None) -> None:
    with lock():
        state = load()
        version = state.get(name)
        if version.state != "promoted":
            raise RegistryError(f"{name} is {version.state}; only a promoted version rolls back")
        if to is not None and state.get(to).state != "promoted":
            raise RegistryError(f"{to} is not promoted, so it cannot become current")
        append({"event": "rolled_back", "version": name, "reason": reason, **({"to": to} if to else {})})
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

def gguf_keepers(state: State) -> set[str]:
    """Versions whose GGUF stays: v0, current, the previous two promoted, and undecided candidates
    (a candidate still waiting for its gate needs its file; once rejected it goes)."""
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
    keep.update(v.name for v in state.versions.values() if v.state == "candidate")
    return keep


def retain() -> list[tuple[str, str, str]]:
    """Delete the GGUFs (and merged NeMo checkpoints, dropped once exported) retention no longer keeps,
    appending `pruned_artifact` for each file removed. Metadata, snapshots and LoRA deltas stay forever.
    An object a kept version still names is not deleted, and a version restored on demand keeps its
    file only until the next state change. The caller holds `lock()`."""
    state = load()
    keepers = gguf_keepers(state)
    needed = {sha for name in keepers for sha in state.versions[name].ggufs()}
    needed |= {v.artifact("lora_delta") for v in state.versions.values()} - {None}
    pruned = []
    for version in state.ordered():
        doomed = []
        if version.name not in keepers:
            doomed += [("gguf", sha) for sha in sorted(version.ggufs())]
        if version.artifact("merged_nemo"):
            doomed.append(("merged_nemo", version.artifact("merged_nemo")))
        for kind, sha in doomed:
            if sha in needed or not objects.remove(sha):
                continue
            append({"event": "pruned_artifact", "version": version.name, "kind": kind, "sha256": sha})
            pruned.append((version.name, kind, sha))
    return pruned
