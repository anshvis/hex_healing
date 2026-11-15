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

from healing import labels, splits
from healing.store import Store, StoreError, now_ms

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


def candidates(
    store: Store, strategy: str, cutoff_ms: int | None = None, *, zones: splits.Zones | None = None,
    apart_from_registered: bool = False,
) -> list:
    """Unreviewed clips with audio that `strategy` may show.

    A random review shows only clips `Zones.may_test` allows: never one in a cluster golden, the
    general set or a recorded train snapshot holds, so a clip a model trained on cannot later become
    a test clip (D-41). With `cutoff_ms` it also takes only the window after it (the rolling frame);
    without, any age (golden's frame, which passes `apart_from_registered` to avoid rolling and
    edited clusters too).
    """
    # A random review draws from every clip not yet sampled at random (an uncertain review is not a
    # sample, so it doesn't shrink the frame) and never a clip you rejected.
    seen = "r.strategy = 'random' OR r.action = 'reject'" if strategy == "random" else "1 = 1"
    base = (
        f"SELECT c.* FROM clips c WHERE {labels.USABLE}"
        f" AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.clip_id = c.clip_id AND ({seen}))"
    )
    if strategy == "uncertain":
        # The benchmark exposes no model confidence, so: the newest weak clips.
        return store.db.execute(base + " AND c.label = 'weak' ORDER BY c.timestamp_ms DESC, c.clip_id").fetchall()
    zones = zones or splits.Zones.load(store.db)
    return [
        r for r in store.db.execute(base + " ORDER BY c.clip_id").fetchall()
        if zones.may_test(r["clip_id"], windowed=cutoff_ms is not None, cutoff_ms=cutoff_ms,
                          apart_from_registered=apart_from_registered)
    ]


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
    *,
    player: Player = afplay,
    ask: Ask = input,
    out: Callable[[str], None] = print,
    seed: int | None = None,
) -> ReviewResult:
    if strategy not in STRATEGIES:
        raise ReviewError(f"strategy is one of {', '.join(STRATEGIES)}")
    cutoff = None
    if cutoff_ms is not None and not since_cutoff:
        raise ReviewError("--cutoff only means something with --since-cutoff")
    if since_cutoff:
        if strategy != "random":
            raise ReviewError("--since-cutoff fills the rolling set, which takes random reviews only")
        cutoff = splits.cutoff_for(cutoff_ms)  # the window `split build` uses for rolling and edited
    pool = candidates(store, strategy, cutoff)
    if not pool:
        out("nothing to review: no unreviewed clips with audio match")
        return ReviewResult()
    rows = pick(pool, n, strategy, random.Random(seed))
    return review_clips(store, rows, strategy, player=player, ask=ask, out=out)


def main(args) -> int:
    from healing.collect import since_to_ms

    try:
        # No `process_lock`: a review is minutes of waiting on you, and holding the lock would stall the
        # scheduled collect. Each answer is one short transaction; the store's busy timeout covers contention.
        cutoff = since_to_ms(args.cutoff, now_ms()) if args.cutoff else None
        if args.cutoff and cutoff is None:
            raise ReviewError(f"cannot read --cutoff {args.cutoff!r}")
        store = Store(sweep=False)
        try:
            result = run(store, args.n, args.strategy, args.since_cutoff, cutoff)
        finally:
            store.close()
    except (ReviewError, StoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result.text())
    return 0
