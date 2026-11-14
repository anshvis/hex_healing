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

    prune = commands.add_parser("prune", help="cluster clips, keep k audio each, hold the budget")
    prune.add_argument("--budget", default="5GiB")
    prune.add_argument("--k", type=int, default=3)
    prune.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.group == "prune":
        return _prune(args)
    if args.group == "collect":
        from healing.collect import main_collect

        return main_collect(args.since, args.hex)
    if args.group == "sets":
        return _subset(args) if args.command == "subset" else _build_general(args)
    if args.group == "eval":
        return _baseline(args)
    return _paired(args)


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
