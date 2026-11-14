"""`healing`: the self-healing pipeline's command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from healing.paths import healing_home


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="healing")
    commands = parser.add_subparsers(dest="group", required=True)

    sets = commands.add_parser("sets").add_subparsers(dest="command", required=True)
    general = sets.add_parser("build-general", help="build the fixed general regression set")
    general.add_argument("--librispeech", type=int, default=300)
    general.add_argument("--ami", type=int, default=200)
    general.add_argument("--seed", type=int, default=13)

    sub = sets.add_parser("subset", help="write a sub-manifest of the general set by id prefix")
    sub.add_argument("--prefix", required=True)
    sub.add_argument("--name", required=True)
    sub.add_argument("--expected", type=int, required=True, help="clip count to insist on")

    evals = commands.add_parser("eval").add_subparsers(dest="command", required=True)
    base = evals.add_parser("baseline", help="score the shipped model on the general set")
    base.add_argument("--model", type=Path, help="GGUF to score (default: the catalog Unified)")
    base.add_argument("--model-id", default=None)
    base.add_argument("--output", type=Path, default=None)

    stats = commands.add_parser("stats").add_subparsers(dest="command", required=True)
    paired = stats.add_parser("paired", help="paired-bootstrap Δ WER (b − a) over clips")
    paired.add_argument("a", type=Path)
    paired.add_argument("b", type=Path)
    paired.add_argument("--resamples", type=int, default=10_000)
    paired.add_argument("--seed", type=int, default=13)

    collect = commands.add_parser("collect", help="copy Hex's History into the dataset")
    collect.add_argument("--since", help="3d, a date or RFC 3339 (default: last run minus 2 days)")
    collect.add_argument("--hex", type=Path, default=None, help="path to the hex binary")
    collect.add_argument("--min-free", default="10GiB", help="copy no audio below this much free space")

    prune = commands.add_parser("prune", help="cluster clips, keep k audio each, hold the budget")
    prune.add_argument("--budget", default="5GiB")
    prune.add_argument("--k", type=int, default=3)
    prune.add_argument("--dry-run", action="store_true")

    clean = commands.add_parser("clean", help="keep the audio index true and the disk from filling")
    clean.add_argument("--dry-run", action="store_true")
    clean.add_argument("--min-free", default="10GiB")
    clean.add_argument("--deep", action="store_true", help="re-hash every audio file")

    schedule = commands.add_parser("schedule", help="the launchd job that collects every 6 hours")
    schedule.add_argument("command", choices=["install", "remove", "show", "run"])
    schedule.add_argument("--hex", type=Path, default=None, help="path to the hex binary (install)")

    status = commands.add_parser("status", help="what the dataset holds and what is at risk")
    status.add_argument("--json", action="store_true")
    status.add_argument("--list", action="store_true", help="list every warning code")
    status.add_argument("--budget", default="5GiB")
    status.add_argument("--min-free", default="10GiB")
    status.add_argument("--hex", type=Path, default=None, help="path to the hex binary")

    args = parser.parse_args(argv)
    if args.group == "schedule":
        return _schedule(args)
    if args.group == "status":
        return _status(args)
    if args.group == "prune":
        return _prune(args)
    if args.group == "clean":
        return _clean(args)
    if args.group == "collect":
        from healing.budget import parse_size
        from healing.collect import main_collect

        try:
            floor = parse_size(args.min_free)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        return main_collect(args.since, args.hex, floor)
    if args.group == "sets":
        return _subset(args) if args.command == "subset" else _build_general(args)
    if args.group == "eval":
        return _baseline(args)
    return _paired(args)


def _schedule(args) -> int:
    from healing import schedule

    if args.command == "run":
        return schedule.run_once()
    try:
        if args.command == "install":
            print(f"installed {schedule.install(args.hex)}")
        elif args.command == "remove":
            schedule.remove()
            print("removed")
        else:
            print(schedule.show())
    except schedule.ScheduleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _status(args) -> int:
    from healing import status as status_module
    from healing.budget import parse_size
    from healing.eval.bench import BenchError, hex_binary
    from healing.store import Store, StoreError

    if args.list:
        for code, about in status_module.WARNINGS.items():
            print(f"{code}\t{about}")
        return 0
    try:
        budget = parse_size(args.budget)
        min_free = parse_size(args.min_free)
        binary = Path(args.hex or hex_binary())
    except (ValueError, BenchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        store = Store(sweep=False)  # status is a read: never delete a running collect's temp files
        try:
            result = status_module.compute(store, binary, budget, min_free=min_free)
        finally:
            store.close()
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.as_dict(), indent=2) if args.json else result.text())
    return 1 if result.warnings else 0


def _clean(args) -> int:
    from healing.budget import parse_size
    from healing.clean import main_clean

    try:
        floor = parse_size(args.min_free)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return main_clean(floor, args.deep, args.dry_run)


def _prune(args) -> int:
    from healing.budget import PruneError, parse_size, prune
    from healing.store import Store, StoreError, process_lock

    try:
        budget = parse_size(args.budget)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        with process_lock():
            store = Store()
            try:
                result = prune(store, budget, args.k, args.dry_run)
            finally:
                store.close()
    except PruneError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.dry_run:
        for eviction in result.evictions:
            print(eviction)
    print(result.summary())
    return 0


def _build_general(args) -> int:
    from healing.sets.general import AmiIhmTest, BuildError, LibriSpeechTestOther, build

    cache = healing_home() / "cache"
    try:
        result = build(
            healing_home() / "sets" / "general",
            [(LibriSpeechTestOther(cache), args.librispeech), (AmiIhmTest(cache), args.ami)],
            seed=args.seed,
        )
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    state = "unchanged" if result.unchanged else "written"
    print(f"{state}: {result.clips} clips, sha256 {result.sha256}\n{result.manifest}")
    return 0


def _subset(args) -> int:
    from healing.sets.general import BuildError, subset

    try:
        print(subset(healing_home() / "sets" / "general", args.prefix, args.name, args.expected))
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _baseline(args) -> int:
    from healing.eval import baseline
    from healing.eval.bench import BenchError

    model = args.model or baseline.DEFAULT_GGUF
    model_id = args.model_id or (baseline.DEFAULT_MODEL_ID if args.model is None else model.stem)
    try:
        result = baseline.baseline(model, model_id, args.output or baseline.default_output())
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _paired(args) -> int:
    from healing.stats import paired_bootstrap, read_clip_scores

    result = paired_bootstrap(
        read_clip_scores(args.a),
        read_clip_scores(args.b),
        resamples=args.resamples,
        seed=args.seed,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
