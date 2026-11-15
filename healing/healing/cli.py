"""`healing`: the self-healing pipeline's command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from healing.paths import healing_home
from healing.protocol import SETTLE_S


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

    label = commands.add_parser("label").add_subparsers(dest="command", required=True)
    corrected = label.add_parser("corrections", help="classify recorded edits against raw_text")
    corrected.add_argument("--since", help="3d, a date or RFC 3339")
    corrected.add_argument("--recompute", action="store_true", help="classify every edited clip again")

    label_all = label.add_parser("all", help="recompute every clip's label and weight from corrections and reviews")
    label_all.add_argument("--json", action="store_true")

    report = commands.add_parser("report").add_subparsers(dest="command", required=True)
    observation = report.add_parser("observation", help="per app: pastes observed and edited")
    observation.add_argument("--json", action="store_true")

    protocol = commands.add_parser("protocol").add_subparsers(dest="command", required=True)
    seeded = protocol.add_parser("corrections", help="the seeded edit-recall protocol (X-3)")
    seeded.add_argument("step", choices=["plan", "run", "score"])
    seeded.add_argument("--dir", type=Path, default=None, help="protocol folder (default: $HEALING_HOME/protocol)")
    seeded.add_argument("--hex", type=Path, default=None, help="path to the hex binary (run, score)")
    seeded.add_argument("--seed", type=int, default=13)
    seeded.add_argument("--settle", type=float, default=SETTLE_S, help="seconds to wait for Hex's 30 s watch (run)")
    seeded.add_argument("--apps", help="comma list to run a subset (run)")

    args = parser.parse_args(argv)
    if args.group == "label":
        return _label_all(args) if args.command == "all" else _label(args)
    if args.group == "report":
        return _report(args)
    if args.group == "protocol":
        return _protocol(args)
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


def _label(args) -> int:
    from healing.collect import CollectError, check_since, since_to_ms
    from healing.corrections import label_corrections
    from healing.store import Store, StoreError, now_ms, process_lock

    try:
        since_ms = None
        if args.since:
            check_since(args.since)
            since_ms = since_to_ms(args.since, now_ms())
            if since_ms is None:
                raise CollectError(f"cannot read --since {args.since!r}", 2)
        with process_lock():
            store = Store()
            try:
                counts = label_corrections(store, since_ms, args.recompute)
            finally:
                store.close()
    except CollectError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.code
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    kinds = " ".join(f"{k} {counts[k]}" for k in ("acoustic", "rewrite", "unmapped"))
    print(f"clips {counts['clips']}, {kinds}, strong {counts['strong']}, "
          f"strong_without_audio {counts['strong_without_audio']}")
    return 0


def _label_all(args) -> int:
    from healing.labels import LABELS, label_all
    from healing.store import Store, StoreError, process_lock

    try:
        with process_lock():
            store = Store()
            try:
                counts = label_all(store)
            finally:
                store.close()
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({label: counts[label] for label in LABELS}))
    else:
        print(", ".join(f"{label} {counts[label]}" for label in LABELS))
    return 0


def _report(args) -> int:
    from healing.corrections import observation_report
    from healing.store import Store, StoreError

    try:
        store = Store(sweep=False)
        try:
            rows = observation_report(store)
        finally:
            store.close()
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'application':<24}{'pastes':>8}{'observed':>10}{'span_not_found':>16}{'edited':>9}")
    for r in rows:
        print(f"{r['application']:<24}{r['pastes']:>8}{r['observed']:>10}"
              f"{r['span_not_found_rate']:>15.0%} {r['edited_rate']:>8.0%}")
    return 0


def _protocol(args) -> int:
    from healing import protocol

    return protocol.main(args)


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
    from healing.eval.bench import BenchError, history_hex
    from healing.store import Store, StoreError

    if args.list:
        for code, about in status_module.WARNINGS.items():
            print(f"{code}\t{about}")
        return 0
    try:
        budget = parse_size(args.budget)
        min_free = parse_size(args.min_free)
        binary = Path(args.hex or history_hex())
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
