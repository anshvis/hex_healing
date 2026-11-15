"""`healing review`: confirm a reference by ear, one clip at a time.

Plays each clip, shows the text, and takes your answer: Enter accepts it, typing replaces it, `r`
rejects the clip (noise, a cough, someone else), `p` plays it again, `q` stops. Each answer is a
`reviews` row at once, so stopping loses nothing, and refreshes that clip's label (`verified`).

`random` is the only strategy that feeds gating sets (T3.2): it samples clips whatever the model
got wrong. `uncertain` picks weak clips and feeds training only.

The audio player and the input function are parameters, so tests need no speaker and no terminal.
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from healing import labels
from healing.store import Store, now_ms

Player = Callable[[Path], None]
Ask = Callable[[str], str]
STRATEGIES = ("random", "uncertain")
PROMPT = "[Enter] accept  [type] replace  [r] reject  [p] replay  [q] quit > "


class ReviewError(RuntimeError):
    pass


def afplay(path: Path) -> None:
    try:
        subprocess.run(["afplay", str(path)], check=False)
    except FileNotFoundError as error:
        raise ReviewError("afplay not found: healing review plays audio with macOS's afplay") from error


@dataclass
class ReviewResult:
    shown: int = 0
    accepted: int = 0
    fixed: int = 0
    rejected: int = 0
    available: int = 0
    stopped: bool = False

    def text(self) -> str:
        done = self.accepted + self.fixed + self.rejected
        tail = ", stopped early" if self.stopped else ""
        return (f"reviewed {done} of {self.available} available: accepted {self.accepted}, "
                f"fixed {self.fixed}, rejected {self.rejected}{tail}")


def candidates(store: Store, strategy: str, only: set[str] | None = None) -> list:
    """Unreviewed clips with audio that `strategy` may show; `only` limits them to those ids."""
    base = (
        f"SELECT c.* FROM clips c WHERE {labels.HAS_AUDIO} AND c.kind = 'dictation' AND c.golden = 0"
        " AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.clip_id = c.clip_id)"
    )
    if strategy == "uncertain":
        # The benchmark exposes no model confidence, so: the newest weak clips.
        rows = store.db.execute(base + " AND c.label = 'weak' ORDER BY c.timestamp_ms DESC, c.clip_id").fetchall()
    else:
        rows = store.db.execute(base + " ORDER BY c.clip_id").fetchall()
    return [r for r in rows if only is None or r["clip_id"] in only]


def pick(rows: list, n: int, strategy: str, rng: random.Random) -> list:
    if strategy == "uncertain":
        return rows[:n]
    return rng.sample(rows, min(n, len(rows)))


def review_clips(
    store: Store,
    rows: list,
    strategy: str,
    *,
    player: Player,
    ask: Ask,
    out: Callable[[str], None] = print,
    after: Callable[[object, str], bool] | None = None,
    now: Callable[[], int] = now_ms,
) -> ReviewResult:
    """Show `rows` in order. `after(row, action)` runs once each answer is saved and returns True to stop."""
    result = ReviewResult(available=len(rows))
    store.db.execute("PRAGMA busy_timeout = 60000")  # a scheduled collect may hold the file for a moment
    for index, row in enumerate(rows, 1):
        shown = row["reference_text"] or row["raw_text"]
        audio = store.audio_path(row["audio_sha256"])
        out(f"\n[{index}/{len(rows)}] {row['application'] or '(unknown app)'}  {row['audio_ms'] or '?'} ms")
        out(f"  {shown}")
        player(audio)
        while True:
            try:
                answer = ask(PROMPT)
            except (EOFError, KeyboardInterrupt):
                answer = "q"
            answer = answer.strip()
            if answer == "p":
                player(audio)
                continue
            break
        if answer == "q":
            result.stopped = True
            break
        result.shown += 1
        if answer == "r":
            action, text = "reject", None
            result.rejected += 1
        elif answer == "" or answer == shown:
            action, text = "accept", shown
            result.accepted += 1
        else:
            action, text = "fix", answer
            result.fixed += 1
        store.db.execute(
            "INSERT INTO reviews (clip_id, reviewed_at_ms, action, text, strategy) VALUES (?, ?, ?, ?, ?)",
            (row["clip_id"], now(), action, text, strategy),
        )
        store.db.commit()
        labels.refresh(store, [row["clip_id"]])
        if after is not None and after(row, action):
            break
    return result


def run(
    store: Store,
    n: int = 20,
    strategy: str = "random",
    since_cutoff: bool = False,
    cutoff_ms: int | None = None,
    parent: str = "current",
    *,
    player: Player = afplay,
    ask: Ask = input,
    out: Callable[[str], None] = print,
    seed: int | None = None,
) -> ReviewResult:
    from healing import splits

    if strategy not in STRATEGIES:
        raise ReviewError(f"strategy is one of {', '.join(STRATEGIES)}")
    only = None
    if since_cutoff:
        if strategy != "random":
            raise ReviewError("--since-cutoff fills the rolling set, which takes random reviews only")
        cutoff = splits.ensure_cutoff(store, parent, cutoff_ms)
        only = splits.rolling_eligible_ids(store, cutoff)
    pool = candidates(store, strategy, only)
    if not pool:
        out("nothing to review: no unreviewed clips with audio match")
        return ReviewResult()
    rows = pick(pool, n, strategy, random.Random(seed))
    return review_clips(store, rows, strategy, player=player, ask=ask, out=out)


def main(args) -> int:
    from healing.store import Store, StoreError

    try:
        store = Store(sweep=False)
        try:
            result = run(store, args.n, args.strategy, args.since_cutoff)
        finally:
            store.close()
    except (ReviewError, StoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result.text())
    return 0
