"""`healing vocab apply` and `revert`: the mined terms in Hex's vocabulary, behind your own.

Hex biases only the first 64 entries in stored order and a mode's own terms come before the global
list, so the written order is the whole design (D-20): the terms you wrote first, in the order you
have them, then the mined ones in rank order, up to Hex's 500. Mined terms below the cut stay stored
and move up as they are used more. The list is read with `hex vocabulary list --json` and written
with one `hex vocabulary set --file` (T4.4), never by editing `settings.json` (G-6).

**Ownership is by content.** `$HEALING_HOME/vocab/applied.json` holds what healing last wrote. An entry
is healing's only while its term and its spellings both still equal that record; one you edited is
yours from then on and is never removed or reordered. A term healing wrote that is no longer in the
list was removed by you: it goes to `rejected` and is never added again. `pending` is the list a
`set` was about to write, kept until the `set` answers, so a crash in between loses no ownership
(that run cannot tell a removal from a drop and records none).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from healing.paths import healing_home, hex_support_dir
from healing.vocab import mine as mine_module

MAX_ENTRIES = 500  # Hex's cap on the saved list (`src/dictation/vocabulary.rs`)
BIAS_SLOTS = 64  # Hex's cap on the entries it biases, taken in stored order
EXIT_NOT_RUNNING = 3  # docs/cli.md: control commands never launch Hex


class ApplyError(RuntimeError):
    pass


class HexNotRunning(ApplyError):
    pass


def applied_path() -> Path:
    return healing_home() / "vocab" / "applied.json"


def entry(term: str, sounds_like=()) -> dict:
    return {"term": term, "sounds_like": [str(s) for s in sounds_like or []]}


def normalize(item) -> dict:
    """A list entry as `{term, sounds_like}`, whether Hex printed a bare string or an object."""
    if isinstance(item, str):
        return entry(item)
    return entry(item.get("term", ""), item.get("sounds_like"))


# ---- the hex calls -------------------------------------------------------------------------------

def _run(binary: Path, args: list[str]) -> subprocess.CompletedProcess:
    # Not the isolated environment the benchmark uses: this talks to the one running Hex, found through
    # its own data folder's discovery file.
    env = {**os.environ, "HEX_DISABLE_UPDATES": "1"}
    try:
        done = subprocess.run([str(binary), *args], capture_output=True, text=True, env=env, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ApplyError(f"could not run {binary}: {error}") from error
    if done.returncode == EXIT_NOT_RUNNING:
        raise HexNotRunning("Hex isn't running")
    if done.returncode != 0:
        raise ApplyError(f"hex {' '.join(args[:2])} exited {done.returncode}: "
                         + (done.stderr.strip() or done.stdout.strip())[-300:])
    return done


def hex_list(binary: Path) -> list[dict]:
    done = _run(binary, ["vocabulary", "list", "--json"])
    try:
        reply = json.loads(done.stdout)
        return [normalize(item) for item in reply["entries"]]
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ApplyError("hex vocabulary list --json printed something this version cannot read") from error


def hex_set(binary: Path, entries: list[dict]) -> bool:
    """The one write. True when Hex changed its list, False when it already held exactly this."""
    folder = applied_path().parent
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="set-", dir=folder) as scratch:
        path = Path(scratch) / "vocabulary.json"
        path.write_text(json.dumps([e if e["sounds_like"] else e["term"] for e in entries], indent=1))
        done = _run(binary, ["vocabulary", "set", "--file", str(path), "--json"])
    try:
        return bool(json.loads(done.stdout).get("changed", True))
    except ValueError:
        return True


def hex_modes() -> list[tuple[str, list[dict]]]:
    """Each mode's own vocabulary, Global first, read-only from Hex's `settings.json` (there is no
    `hex` command for modes). Best effort: a missing or unreadable file or an unknown shape is no modes."""
    try:
        settings = json.loads((hex_support_dir() / "settings.json").read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(settings, dict):
        return []
    holder = settings.get("dictation_processing") if isinstance(settings.get("dictation_processing"), dict) else settings
    rows = []
    default = holder.get("default_mode")
    if isinstance(default, dict):
        rows.append(("Global", default))
    modes = holder.get("modes")
    rows += [(m.get("name", "?"), m) for m in modes if isinstance(m, dict)] if isinstance(modes, list) else []
    return [(name, [normalize(v) for v in mode.get("vocabulary", []) if isinstance(v, (str, dict))]) for name, mode in rows]


# ---- the record ---------------------------------------------------------------------------------

@dataclass
class Applied:
    applied_at_ms: int = 0
    terms: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    pending: list[dict] | None = None
    user_terms: int = 0  # how many entries were yours at the last apply (`healing status` warns on 64)

    def owned(self) -> list[dict]:
        return self.terms + (self.pending or [])


def load_applied() -> Applied:
    try:
        raw = json.loads(applied_path().read_text())
        return Applied(
            int(raw.get("applied_at_ms", 0)), [normalize(t) for t in raw.get("terms", [])],
            [str(t) for t in raw.get("rejected", [])],
            None if raw.get("pending") is None else [normalize(t) for t in raw["pending"]],
            int(raw.get("user_terms", 0)),
        )
    except FileNotFoundError:
        return Applied()
    except (OSError, ValueError, AttributeError, TypeError) as error:
        raise ApplyError(f"cannot read {applied_path()}: {error}") from error


def save_applied(applied: Applied) -> None:
    path = applied_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = {"applied_at_ms": applied.applied_at_ms, "terms": applied.terms, "rejected": applied.rejected,
            "user_terms": applied.user_terms}
    if applied.pending is not None:
        body["pending"] = applied.pending
    scratch = path.with_suffix(".json.tmp")
    descriptor = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(json.dumps(body, indent=1) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(scratch, path)


# ---- the plan -----------------------------------------------------------------------------------

@dataclass
class Plan:
    new_list: list[dict]
    yours: list[dict]
    mined: list[dict]
    rejected: list[str]  # all rejected terms, old and new
    newly_rejected: list[str]
    changed: bool  # the list differs from what Hex holds


def split_owned(current: list[dict], applied: Applied) -> tuple[list[dict], list[dict]]:
    """(yours, healing's) in list order. An entry is healing's only while term and spellings both match."""
    owned = {(e["term"], tuple(e["sounds_like"])) for e in applied.owned()}
    yours, ours = [], []
    for item in current:
        (ours if (item["term"], tuple(item["sounds_like"])) in owned else yours).append(item)
    return yours, ours


def removed_by_you(current: list[dict], applied: Applied) -> list[str]:
    """Terms healing wrote that you took out of the list. Unknowable while a `set` is unconfirmed."""
    if applied.pending is not None:
        return []
    present = {item["term"].lower() for item in current}
    return [t["term"] for t in applied.terms if t["term"].lower() not in present]


def plan(current: list[dict], applied: Applied, candidates: list[mine_module.Candidate]) -> Plan:
    gone = removed_by_you(current, applied)
    rejected = list(dict.fromkeys(applied.rejected + [t for t in gone if t not in applied.rejected]))
    yours, _ = split_owned(current, applied)
    taken = {item["term"].lower() for item in yours} | {t.lower() for t in rejected}
    mined: list[dict] = []
    for candidate in candidates:
        if len(yours) + len(mined) >= MAX_ENTRIES:
            break
        if candidate.term.lower() in taken:
            continue
        taken.add(candidate.term.lower())
        mined.append(entry(candidate.term, candidate.sounds_like))
    new_list = yours + mined
    return Plan(new_list, yours, mined, rejected, [t for t in gone if t not in applied.rejected], new_list != current)


def biased_by_mode(new_list: list[dict], mined: list[dict], modes: list[tuple[str, list[dict]]],
                   slots: int = BIAS_SLOTS) -> list[tuple[str, int, int]]:
    """`(name, mode terms, mined terms biased)` per mode, Global first. Hex biases a mode's own terms,
    then the shared list with duplicates dropped, and keeps the first `slots`."""
    mined_terms = {m["term"].lower() for m in mined}
    rows = [("Global (no mode)", 0, new_list)]
    rows += [(name, len(own), own + new_list) for name, own in modes]
    out = []
    for name, own_count, merged in rows:
        seen, kept = set(), []
        for item in merged:
            if item["term"].lower() not in seen:
                seen.add(item["term"].lower())
                kept.append(item)
        out.append((name, own_count, sum(1 for item in kept[:slots] if item["term"].lower() in mined_terms)))
    return out


def report(p: Plan, modes, slots: int) -> list[str]:
    lines = [f"{len(p.yours)} of yours first, then {len(p.mined)} mined (list of {len(p.new_list)}; Hex biases the first {BIAS_SLOTS})"]
    for index, item in enumerate(p.new_list, 1):
        kind = "mined" if index > len(p.yours) else "yours"
        spelled = f" ← {', '.join(item['sounds_like'])}" if item["sounds_like"] else ""
        lines.append(f"  {index:>3} {item['term']}{spelled}  [{kind}{'' if index <= slots else ', below the cut'}]")
    for name, own, biased in biased_by_mode(p.new_list, p.mined, modes, slots):
        suffix = f" ({own} of its own terms)" if own else ""
        lines.append(f"  {name}{suffix}: {biased} of {len(p.mined)} mined terms biased")
    if len(p.yours) >= slots:
        lines.append(f"warning: your terms alone take {len(p.yours)} of {slots} slots, so no mined term is biased")
    return lines


# ---- commands -----------------------------------------------------------------------------------

def apply(binary: Path, candidates: list[mine_module.Candidate], *, dry_run: bool, slots: int, now: int) -> list[str]:
    current = hex_list(binary)
    applied = load_applied()
    p = plan(current, applied, candidates)
    lines = report(p, hex_modes(), slots)
    if p.newly_rejected:
        lines.append("removed by you, never added again: " + ", ".join(p.newly_rejected))
    if dry_run:
        return ["dry run, nothing written"] + lines
    if not p.changed:
        # Nothing to write to Hex; keep the record true (new rejections, the count of your terms).
        _, ours = split_owned(current, applied)
        record = Applied(applied.applied_at_ms, ours, p.rejected, None, len(p.yours))
        if (record.terms, record.rejected, record.user_terms, applied.pending) != (
                applied.terms, applied.rejected, applied.user_terms, None):
            save_applied(record)
        return ["no change, Hex already holds this list"] + lines
    # the intent is on disk before Hex sees it, so a crash between the two leaves ownership known
    save_applied(Applied(applied.applied_at_ms, applied.terms, p.rejected, p.mined, len(p.yours)))
    hex_set(binary, p.new_list)
    save_applied(Applied(now, p.mined, p.rejected, None, len(p.yours)))
    return ["applied"] + lines


def revert(binary: Path, *, now: int) -> list[str]:
    current = hex_list(binary)
    applied = load_applied()
    gone = removed_by_you(current, applied)
    rejected = list(dict.fromkeys(applied.rejected + gone))
    yours, ours = split_owned(current, applied)
    if ours:
        hex_set(binary, yours)
    save_applied(Applied(now, [], rejected, None, len(yours)))
    if not ours:
        return ["nothing to revert: no term healing wrote is still in the list as written"]
    return [f"removed {len(ours)} mined term(s): " + ", ".join(e["term"] for e in ours), f"left {len(yours)} of yours"]


def main(args) -> int:
    from healing.eval.bench import BenchError, history_hex
    from healing.store import Store, StoreError, now_ms

    if args.command == "apply" and args.max_biased < 1:
        print("error: --max-biased is at least 1", file=sys.stderr)
        return 2
    try:
        binary = Path(args.hex or history_hex())
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        if args.command == "revert":
            lines = revert(binary, now=now_ms())
        else:
            store = Store(sweep=False)
            try:
                candidates = mine_module.load_candidates(store)
            finally:
                store.close()
            if not candidates:
                print("note: no saved candidates; run `healing vocab mine` first", file=sys.stderr)
            lines = apply(binary, candidates, dry_run=args.dry_run, slots=min(args.max_biased, BIAS_SLOTS), now=now_ms())
    except HexNotRunning as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except (ApplyError, StoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0
