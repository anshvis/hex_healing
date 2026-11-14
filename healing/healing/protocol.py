"""`healing protocol corrections`: the seeded edit-recall protocol (X-3).

Observing a paste is not the same as capturing its edit. This measures the second: in each
app of the matrix it dictates a fixed clip 10 times, replaces one word in half of them
(some followed by Return or send within 5 s), then asks Hex's History what it recorded.

  plan   writes plan.json and RUNBOOK.md (offline, deterministic from --seed)
  run    drives Hex and System Events through the plan (macOS, needs a person at the Mac)
  score  matches History to the runs; edit recall and false-edit rate, per app and overall

Edit recall = seeded edits stored as `edited` with the seeded word in the span / seeded edits.
False-edit rate = unedited dictations stored as `edited` / unedited dictations.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from healing import corrections
from healing.paths import healing_home

# (name for `tell application`, notes on the target field the person must focus)
APPS = {
    "TextEdit": "a new plain-text or rich-text document",
    "Notes": "a new note",
    "Slack": "the message box of any channel or DM to yourself",
    "Google Chrome": "a <textarea> (for example a local page: data:text/html,<textarea autofocus>)",
    "Safari": "a <textarea> (same page as Chrome)",
    "Visual Studio Code": "a new untitled editor",
}
RUNS_PER_APP = 10
SEEDED_PER_APP = 5
SEND_PER_APP = 2  # of the seeded runs: Return within SEND_WITHIN_S of the edit
SEND_WITHIN_S = 3.0
REPLACEMENTS = ["Kit", "Rune", "Friday", "Hex", "Sarah", "Monday", "banana"]
MIN_RECALL = 0.5
MAX_FALSE_EDIT = 0.05
PHRASE = "send it to kip tomorrow"


def protocol_dir(given: Path | None) -> Path:
    return given or healing_home() / "protocol"


def build_plan(seed: int, apps: dict[str, str] | None = None) -> dict:
    rng = random.Random(seed)
    runs = []
    for app in apps or APPS:
        order = list(range(RUNS_PER_APP))
        rng.shuffle(order)
        seeded = set(order[:SEEDED_PER_APP])
        sends = set(order[:SEND_PER_APP])
        for index in range(RUNS_PER_APP):
            runs.append(
                {
                    "app": app,
                    "index": index,
                    "seeded": index in seeded,
                    "send": index in sends,
                    "replacement": rng.choice(REPLACEMENTS) if index in seeded else None,
                }
            )
    return {"version": 1, "seed": seed, "phrase": PHRASE, "runs": runs}


RUNBOOK = """# Seeded edit-recall protocol (X-3)

A person has to do this at a Mac; it cannot run unattended. About {minutes} minutes.

## Once
1. `hex quit`. Make a fixture clip and start Hex on it, so no microphone is used:
   `say -o /tmp/fixture.aiff "{phrase}" && afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/fixture.aiff /tmp/fixture.wav`
   `HEX_AUDIO_FIXTURE=/tmp/fixture.wav hex start`
2. `hex labs on learn-from-corrections`; confirm the lab shows on in Settings > Labs.
3. Grant the terminal you run `healing` from Accessibility (System Settings > Privacy & Security),
   so System Events may type. Dictate once by hand and check it pastes and appears in `hex history --raw`.
4. Do not touch the keyboard or mouse while a batch runs: the driver types into the frontmost app.

## Per app
For each app below, open the target field and click into it, then run
`healing protocol corrections run --apps "<app>"`. The driver does 10 dictations
({seeded} get a one-word edit, {sends} of those followed by Return within {send_within:g} s), and waits
{settle:g} s after each for Hex's 30 s watch to end. Targets:

{targets}

## Score
After all six apps: `healing protocol corrections score`. It reads `hex history --json --raw`
and prints recall and false-edit rate per app and overall. Pass: recall >= {min_recall:g} and
false-edit rate <= {max_false:g}. Below that, X-3 fires: report the per-app table and
replan M2 toward an explicit "Correct this" action in History.
Paste the table into the PR body.
"""


def write_plan(directory: Path, seed: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plan.json").write_text(json.dumps(build_plan(seed), indent=2))
    (directory / "RUNBOOK.md").write_text(
        RUNBOOK.format(
            minutes=len(APPS) * 8, phrase=PHRASE, seeded=SEEDED_PER_APP, sends=SEND_PER_APP,
            send_within=SEND_WITHIN_S, settle=33, min_recall=MIN_RECALL, max_false=MAX_FALSE_EDIT,
            targets="\n".join(f"- {app}: {about}" for app, about in APPS.items()),
        )
    )
    return directory


# ---------------------------------------------------------------- run


def _osascript(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=True, capture_output=True)


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class Driver:
    """Everything that touches the outside world, so tests can substitute it."""

    hex: Path
    settle: float = 33.0
    dictate_seconds: float = 4.0
    paste_seconds: float = 3.0
    sleep = staticmethod(time.sleep)
    clock = staticmethod(lambda: int(time.time() * 1000))

    def hex_dictation(self, verb: str) -> None:
        subprocess.run([str(self.hex), "dictation", verb], check=True, capture_output=True)

    def activate(self, app: str) -> None:
        _osascript(f'tell application "{_quote(app)}" to activate')

    def select_last_word(self) -> None:
        _osascript('tell application "System Events" to key code 123 using {option down, shift down}')

    def type_text(self, text: str) -> None:
        _osascript(f'tell application "System Events" to keystroke "{_quote(text)}"')

    def press_return(self) -> None:
        _osascript('tell application "System Events" to key code 36')


def run_plan(plan: dict, results_path: Path, driver: Driver, only: set[str] | None = None) -> int:
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["app"], r["index"]) for r in results}
    count = 0
    for run in plan["runs"]:
        if (only and run["app"] not in only) or (run["app"], run["index"]) in done:
            continue
        driver.activate(run["app"])
        driver.sleep(1.0)
        started = driver.clock()
        driver.hex_dictation("start")
        driver.sleep(driver.dictate_seconds)
        driver.hex_dictation("stop")
        driver.sleep(driver.paste_seconds)
        if run["seeded"]:
            driver.select_last_word()
            driver.type_text(run["replacement"])
            if run["send"]:
                driver.sleep(SEND_WITHIN_S - 1.0)
                driver.press_return()
        driver.sleep(driver.settle)
        results.append({"app": run["app"], "index": run["index"], "start_ms": started})
        results_path.write_text(json.dumps(results, indent=2))  # resumable
        count += 1
    return count


# ---------------------------------------------------------------- score


def _norm(text: str) -> list[str]:
    return [t.norm for t in corrections.tokens(text)]


def match_entries(results: list[dict], entries: list[dict]) -> dict[tuple[str, int], dict | None]:
    """Each run owns the first History entry from its start until the next run's start."""
    ordered = sorted(results, key=lambda r: r["start_ms"])
    stamped = sorted(entries, key=lambda e: e["timestamp_ms"])
    matched: dict[tuple[str, int], dict | None] = {}
    for position, run in enumerate(ordered):
        end = ordered[position + 1]["start_ms"] if position + 1 < len(ordered) else float("inf")
        matched[(run["app"], run["index"])] = next(
            (e for e in stamped if run["start_ms"] <= e["timestamp_ms"] < end), None
        )
    return matched


def seeded_edit_captured(entry: dict | None, replacement: str) -> bool:
    correction = (entry or {}).get("correction") or {}
    if correction.get("outcome") != "edited" or "edited_text" not in correction:
        return False
    edited = _norm(correction["edited_text"])
    return edited != _norm(entry["final_text"]) and _norm(replacement)[0] in edited


def score(plan: dict, results: list[dict], entries: list[dict]) -> dict:
    by_run = {(r["app"], r["index"]): r for r in plan["runs"]}
    matched = match_entries(results, entries)
    apps: dict[str, dict] = {}
    for key, entry in matched.items():
        run = by_run[key]
        row = apps.setdefault(
            run["app"],
            {"seeded": 0, "captured": 0, "unedited": 0, "false_edits": 0, "no_entry": 0, "observed": 0},
        )
        correction = (entry or {}).get("correction") or {}
        row["no_entry"] += entry is None
        row["observed"] += bool(correction) and correction.get("outcome") != "lost"
        if run["seeded"]:
            row["seeded"] += 1
            row["captured"] += seeded_edit_captured(entry, run["replacement"])
        else:
            row["unedited"] += 1
            row["false_edits"] += correction.get("outcome") == "edited"
    total = {k: sum(r[k] for r in apps.values()) for k in ("seeded", "captured", "unedited", "false_edits")}
    recall = total["captured"] / total["seeded"] if total["seeded"] else 0.0
    false_rate = total["false_edits"] / total["unedited"] if total["unedited"] else 0.0
    for row in apps.values():
        row["recall"] = row["captured"] / row["seeded"] if row["seeded"] else 0.0
        row["false_edit_rate"] = row["false_edits"] / row["unedited"] if row["unedited"] else 0.0
    return {
        "apps": apps, "recall": recall, "false_edit_rate": false_rate,
        "pass": bool(total["seeded"]) and recall >= MIN_RECALL and false_rate <= MAX_FALSE_EDIT,
        "x3": recall < MIN_RECALL,
    }


def format_score(result: dict) -> str:
    lines = [f"{'app':<22}{'seeded':>7}{'captured':>9}{'recall':>8}{'unedited':>9}{'false':>6}{'no entry':>9}"]
    for app, r in result["apps"].items():
        lines.append(
            f"{app:<22}{r['seeded']:>7}{r['captured']:>9}{r['recall']:>8.0%}{r['unedited']:>9}"
            f"{r['false_edits']:>6}{r['no_entry']:>9}"
        )
    lines.append(f"edit recall {result['recall']:.0%} (need >= {MIN_RECALL:.0%}), "
                 f"false-edit rate {result['false_edit_rate']:.0%} (need <= {MAX_FALSE_EDIT:.0%})")
    lines.append("PASS" if result["pass"] else "FAIL" + (": X-3 triggers, replan M2 toward an explicit Correct this" if result["x3"] else ""))
    return "\n".join(lines)


def main(args) -> int:
    from healing.collect import CollectError, rfc3339, run_history
    from healing.eval.bench import BenchError, hex_binary

    directory = protocol_dir(args.dir)
    if args.step == "plan":
        print(f"wrote {write_plan(directory, args.seed)}/RUNBOOK.md")
        return 0
    try:
        plan = json.loads((directory / "plan.json").read_text())
    except OSError:
        print("error: no plan: run `healing protocol corrections plan` first", file=sys.stderr)
        return 2
    results_path = directory / "results.json"
    try:
        binary = Path(args.hex or hex_binary())
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.step == "run":
        only = {a.strip() for a in args.apps.split(",")} if args.apps else None
        if only and not only <= set(APPS):
            print(f"error: unknown app; choose from {', '.join(APPS)}", file=sys.stderr)
            return 2
        count = run_plan(plan, results_path, Driver(binary, settle=args.settle), only)
        print(f"ran {count} dictations; results in {results_path}")
        return 0
    if not results_path.exists():
        print("error: no results.json: run the protocol first", file=sys.stderr)
        return 2
    results = json.loads(results_path.read_text())
    try:
        document = run_history(binary, rfc3339(min(r["start_ms"] for r in results) - 60_000))
    except CollectError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    result = score(plan, results, document.get("entries", []))
    (directory / "score.json").write_text(json.dumps(result, indent=2))
    print(format_score(result))
    return 0 if result["pass"] else 1
