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

    fetch = sets.add_parser("fetch-librispeech", help="download LibriSpeech train-clean-100 (replay source)")
    fetch.add_argument("--dest", type=Path, help="write the shards here and symlink the cache path to it")

    sub = sets.add_parser("subset", help="write a sub-manifest of the general set by id prefix")
    sub.add_argument("--prefix", required=True)
    sub.add_argument("--name", required=True)
    sub.add_argument("--expected", type=int, required=True, help="clip count to insist on")

    evaluate = commands.add_parser("eval", help="score GGUFs on rolling, golden, general and edited-span")
    evaluate.add_argument("--model", dest="models", action="append", type=Path, metavar="GGUF",
                          help="a model to score; repeat to compare (interleaved per set)")
    evaluate.add_argument("--sets", default="rolling,golden,general,edited")
    evaluate.add_argument("--parent", default="current",
                          help="whose rolling and edited sets to score (each model version has its own holdout)")
    evaluate.add_argument("--runs", type=int, default=1, help="rounds per set (extra rounds only measure speed)")
    evaluate.add_argument("--vocabulary", choices=["current", "none"], default="current",
                          help="bias with Hex's saved vocabulary, as dictation does (needs the T3.4 hex)")
    evals = evaluate.add_subparsers(dest="command")
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
    schedule.add_argument("--job", choices=["collect", "train-auto"], default="collect",
                          help="train-auto: `healing train auto` daily at 03:00 (T6.4; not installed by default)")

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

    review = commands.add_parser("review", help="confirm references by ear (plays with afplay)")
    review.add_argument("--n", type=int, default=20)
    review.add_argument("--strategy", choices=["random", "uncertain"], default="random")
    review.add_argument("--since-cutoff", action="store_true",
                        help="only clips whose session began after the cut-off (fills the rolling set)")
    review.add_argument("--cutoff", help="with --since-cutoff: 14d, a date or RFC 3339 (default: the last 14 days, sliding; D-40)")

    split = commands.add_parser("split").add_subparsers(dest="command", required=True)
    build = split.add_parser("build", help="rolling and edited-span sets; registers general and golden (builds are not training)")
    build.add_argument("--parent", default="current", help="the version the rolling set is fresh for")
    build.add_argument("--cutoff", help="the window's start: 14d, a date or RFC 3339 (default: the last 14 days, sliding; D-40)")
    split.add_parser("verify", help="exit 1 on any leak, stale pin or non-verified gating reference")

    golden = commands.add_parser("golden").add_subparsers(dest="command", required=True)
    golden_review = golden.add_parser("review", help="confirm golden clips; freezes at 200")
    golden_review.add_argument("--n", type=int, default=20)

    report = commands.add_parser("report").add_subparsers(dest="command", required=True)
    observation = report.add_parser("observation", help="per app: pastes observed and edited")
    observation.add_argument("--json", action="store_true")

    vocab = commands.add_parser("vocab").add_subparsers(dest="command", required=True)
    mine = vocab.add_parser("mine", help="corrections you made twice become candidate vocabulary terms")
    mine.add_argument("--min-pairs", type=int, default=2)
    mine.add_argument("--window", default="30d", help="how far back a term's recent use is counted: 30d, a date or RFC 3339")
    mine.add_argument("--train-only", action="store_true", help="only clips outside the test clusters (the A/B needs this)")
    mine.add_argument("--json", action="store_true")
    apply = vocab.add_parser("apply", help="write your terms, then the mined ones by rank, into Hex's vocabulary")
    apply.add_argument("--dry-run", action="store_true")
    apply.add_argument("--max-biased", type=int, default=64, help="slots Hex biases, for the report (Hex's own cap is 64)")
    apply.add_argument("--force", action="store_true", help="write even without a passing A/B at Hex's boost (logged)")
    apply.add_argument("--hex", type=Path, default=None, help="path to the hex binary")
    ab = vocab.add_parser("ab", help="score the current GGUF with and without the mined terms (needs the T3.4 hex)")
    ab.add_argument("--boost", choices=["light", "normal", "strong"], default="normal")
    ab.add_argument("--model", type=Path, default=None, help="GGUF to score (default: the catalog Unified)")
    ab.add_argument("--parent", default="current", help="whose rolling set to score")
    ab.add_argument("--hex", type=Path, default=None, help="path to the hex binary that reads your vocabulary")
    revert = vocab.add_parser("revert", help="remove the terms healing still owns by content, in one write")
    revert.add_argument("--hex", type=Path, default=None, help="path to the hex binary")

    models = commands.add_parser("models", help="the model registry (M5)").add_subparsers(dest="command", required=True)
    base_model = models.add_parser("register-base", help="record Hex's catalog model as v0, pinned by its GGUF's SHA-256")
    base_model.add_argument("--hex", type=Path, default=None, help="path to the hex binary")
    base_model.add_argument("--gguf", type=Path, default=None, help="the GGUF (default: the selected catalog model's file)")
    base_model.add_argument("--model-id", default=None, help="its catalog id (default: Hex's selected model)")
    base_model.add_argument("--arch", choices=["parakeet-unified", "tdt-v2"], default=None)
    base_model.add_argument("--scores", type=Path, default=None, help="baseline.json, an eval document or {set: {wer}}")
    candidate = models.add_parser("register", help="record a fine-tuned GGUF as a candidate")
    candidate.add_argument("--gguf", type=Path, required=True)
    candidate.add_argument("--delta", type=Path, required=True, help="the LoRA delta it was merged from")
    candidate.add_argument("--config", type=Path, required=True, help="the training config (YAML)")
    candidate.add_argument("--parent", default="current")
    candidate.add_argument("--train", default=None, help="the frozen train split it trained on (`splits.snapshot_train`)")
    candidate.add_argument("--snapshot", default=None, help="or an already stored snapshot's sha256")
    candidate.add_argument("--arch", choices=["parakeet-unified", "tdt-v2"], default=None, help="default: the parent's")
    candidate.add_argument("--quantization", default="q8_0")
    candidate.add_argument("--converter-repo", default=None, help="default: the parent's")
    candidate.add_argument("--converter-rev", default=None, help="default: the parent's")
    candidate.add_argument("--export-command", default=None,
                           help="the export argv, with {base} {delta} {config} {output} {checkout} (default: the parent's)")
    candidate.add_argument("--converter-dir", default=None, help="the converter checkout restore runs in")
    candidate.add_argument("--hex-sha", default=None, help="the Hex commit the GGUF was checked with")
    candidate.add_argument("--scores", type=Path, default=None, help="an eval document or {set: {wer}}")
    models_list = models.add_parser("list", help="every version, its state and scores")
    models_list.add_argument("--json", action="store_true")
    models.add_parser("show", help="one version's whole record").add_argument("version")
    restore = models.add_parser("restore", help="rebuild a version's GGUF from base + delta + the pinned converter")
    restore.add_argument("version")
    restore.add_argument("--converter-dir", type=Path, default=None, help="a checkout of the converter at the recorded rev")
    model_current = models.add_parser("current", help="the current version, cross-checked with Hex's selected model")
    model_current.add_argument("--hex", type=Path, default=None, help="path to the hex binary")
    models.add_parser("pin-golden", help="pin the frozen golden set's hash in the registry (G-10)")

    gate = commands.add_parser("gate", help="the promotion gate: candidate against current on D-12's paired checks (T7.1)")
    gate.add_argument("version", help="the candidate, e.g. v3")
    gate.add_argument("--against", default="current", help="the version it would replace (default: current)")
    gate.add_argument("--eval", type=Path, default=None, help="judge a saved `healing eval` holding both models instead of scoring again")
    gate.add_argument("--hex", type=Path, default=None, help="the hex binary to ask whether the candidate streams")
    gate.add_argument("--allow-thin-evidence", action="store_true", help=argparse.SUPPRESS)  # always refused: fixtures call run(allow_thin=True)

    promote = commands.add_parser("promote", help="import a candidate that passed the gate into Hex, switch to it, confirm and record (T7.2)")
    promote.add_argument("version", help="the candidate, e.g. v3")
    promote.add_argument("--hex", type=Path, default=None, help="path to the hex binary")
    rollback = commands.add_parser("rollback", help="switch Hex back to the previous version (or --to) and record why (T7.2)")
    rollback.add_argument("--to", default=None, help="the promoted version to return to (default: the previous one)")
    rollback.add_argument("--reason", default=None, help="why (default: manual)")
    rollback.add_argument("--hex", type=Path, default=None, help="path to the hex binary")

    watch = commands.add_parser("watch", help="after a promotion, roll back only if it is significantly worse on new reviewed clips (T7.3)")
    watch.add_argument("--dry-run", action="store_true", help="say what it would do; changes nothing")
    watch.add_argument("--hex", type=Path, default=None, help="path to the hex binary")

    train_group = commands.add_parser("train", help="Tier 2 training (M6): `prepare`, `auto`, or a round with --local")
    train_group.add_argument("--local", action="store_true", help="train a LoRA round on this Mac's CPU (T6.5)")
    train_group.add_argument("--smoke", action="store_true", help="with --local or --remote: 20 steps over 50 clips")
    train_group.add_argument("--bundle", type=Path, default=None, help="with --local: the bundle `train prepare` wrote")
    train_group.add_argument("--config", type=Path, default=None, help="with --local: the training config (default: healing/train/configs/lora.yaml)")
    train_group.add_argument("--base", type=Path, default=None, help="with --local: the base .nemo (default: $HEALING_BASE_NEMO, else M0's model cache)")
    train_group.add_argument("--python", type=Path, default=None, help="with --local: NeMo's Python (default: $HEALING_NEMO_PYTHON, else M0's env)")
    train_group.add_argument("--rounds-dir", type=Path, default=None, help="with --local: where rounds work (default: $HEALING_ROUNDS_DIR, else $HEALING_HOME/rounds)")
    train_group.add_argument("--remote", action="store_true", help="train the round on Kaggle's free GPU (T6.2): uploads the bundle")
    train_group.add_argument("--allow-upload", action="store_true", help="with --remote: allow uploading the bundle (your voice and words) to a private Kaggle dataset")
    train_group.add_argument("--max-minutes", type=int, default=120, help="with --remote: the GPU run's time limit (default 120)")
    train = train_group.add_subparsers(dest="command")
    prepare = train.add_parser("prepare", help="snapshot the train clips and write the weighted training bundle")
    prepare.add_argument("--include-weak", action="store_true", help="also train on weak clips (weight 0.2)")
    prepare.add_argument("--replay", type=float, default=0.2, help="share of the sampler's weight given to replay")
    prepare.add_argument("--parent", default="current", help="the version this round trains from")
    prepare.add_argument("--seed", type=int, default=13)
    prepare.add_argument("--replay-source", type=Path, default=None,
                         help="LibriSpeech train-clean-100 parquet shards (default: $HEALING_HOME/cache/librispeech-train-clean-100)")

    auto = train.add_parser("auto", help="start a round once 200 minutes of new strong speech are in (T6.4)")
    auto.add_argument("--dry-run", action="store_true", help="print what it would do; read-only")

    export = commands.add_parser("export", help="merge a LoRA delta, convert to Q8_0 GGUF and register a candidate (T6.3)")
    export.add_argument("delta", type=Path, help="the round's out/lora_delta.safetensors")
    export.add_argument("--config", type=Path, required=True, help="the round's out/lora.yaml")
    export.add_argument("--snapshot", required=True, help="the registry snapshot's sha256, or the frozen train split it trained on")
    export.add_argument("--parent", default="current", help="the version it was trained from")
    export.add_argument("--converter-dir", type=Path, default=None,
                        help="a transcribe.cpp checkout at the pinned rev (default: $HEALING_CONVERTER_DIR, else $HEALING_HOME/tools)")
    export.add_argument("--hex", type=Path, default=None, help="the hex binary for the isolated import check")
    export.add_argument("--tmp-dir", type=Path, default=None, help="where the ~3 GB of temp files go (default: $HEALING_HOME/registry/tmp)")
    gguf_build = commands.add_parser("export-gguf", help="the recorded build command export and restore run (merge, convert, Q8_0, name)")
    for flag in ("--delta", "--config", "--output", "--checkout", "--base-gguf"):
        gguf_build.add_argument(flag, type=Path, required=True)
    gguf_build.add_argument("--name", required=True, help="the GGUF's general.name")
    protocol = commands.add_parser("protocol").add_subparsers(dest="command", required=True)
    seeded = protocol.add_parser("corrections", help="the seeded edit-recall protocol (X-3)")
    seeded.add_argument("step", choices=["plan", "run", "score"])
    seeded.add_argument("--dir", type=Path, default=None, help="protocol folder (default: $HEALING_HOME/protocol)")
    seeded.add_argument("--hex", type=Path, default=None, help="path to the hex binary (run, score)")
    seeded.add_argument("--seed", type=int, default=13)
    seeded.add_argument("--settle", type=float, default=SETTLE_S, help="seconds to wait for Hex's 30 s watch (run)")
    seeded.add_argument("--apps", help="comma list to run a subset (run)")

    tidy_group = commands.add_parser("tidy").add_subparsers(dest="command", required=True)
    habits = tidy_group.add_parser("habits", help="which fillers you accept removing; writes tidy/habits.json for Hex (T8.2)")
    habits.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.group == "label":
        return _label_all(args) if args.command == "all" else _label(args)
    if args.group == "review":
        from healing import review as review_module

        return review_module.main(args)
    if args.group == "split":
        return _split(args)
    if args.group == "golden":
        from healing import golden as golden_module

        return golden_module.main(args)
    if args.group == "vocab":
        from healing.vocab import apply as apply_module
        from healing.vocab import mine as mine_module

        if args.command == "ab":
            from healing.vocab import ab as ab_module

            return ab_module.main(args)
        return (mine_module if args.command == "mine" else apply_module).main(args)
    if args.group == "models":
        from healing.registry import commands as registry_commands

        return registry_commands.main(args)
    if args.group == "report":
        return _report(args)
    if args.group == "gate":
        from healing import gate as gate_module

        return gate_module.main(args)
    if args.group in ("promote", "rollback"):
        from healing import ship

        return ship.main(args)
    if args.group == "watch":
        from healing import watch as watch_module

        return watch_module.main(args)
    if args.group == "train":
        if args.command == "prepare":
            return _train_prepare(args)
        if args.command == "auto":
            from healing.train import auto as train_auto

            return train_auto.main(args)
        if args.local and args.remote:
            train_group.error("give --local or --remote, not both")
        if args.remote:
            from healing.remote.kaggle_runner import main as remote_main, run_remote  # G-3: only remote/ uploads

            if not args.allow_upload:  # before anything else: no file read, no subprocess, no network
                return run_remote(args.bundle, allow_upload=False)
            if args.bundle is None:
                train_group.error("--remote needs --bundle")
            if args.max_minutes < 1:
                train_group.error("--max-minutes must be at least 1")
            return remote_main(args)
        if not args.local:
            train_group.error("give a subcommand (prepare, auto), --local or --remote")
        if args.bundle is None:
            train_group.error("--local needs --bundle")
        from healing.train import run as train_run

        return train_run.main(args)
    if args.group in ("export", "export-gguf"):
        from healing import export as export_module

        return export_module.main(args)
    if args.group == "tidy":
        from healing import tidy as tidy_module

        return tidy_module.main(args)
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
        if args.command == "fetch-librispeech":
            return _fetch_librispeech(args)
        return _subset(args) if args.command == "subset" else _build_general(args)
    if args.group == "eval":
        if args.command == "baseline":
            return _baseline(args)
        from healing.eval import run as eval_run

        return eval_run.main(args)
    return _paired(args)


def _label(args) -> int:
    from healing.collect import CollectError, check_since, since_to_ms
    from healing import labels
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
                labels.refresh(store)  # the one label writer
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


def _split(args) -> int:
    from healing import manifest, splits
    from healing.collect import since_to_ms
    from healing.registry.store import RegistryError
    from healing.store import Store, StoreError, now_ms, process_lock

    try:
        if args.command == "verify":
            store = Store(sweep=False)
            try:
                report = splits.verify_report(store)
            finally:
                store.close()
            violations = report.leaks + [p for p in [_golden_pin_problem()] if p]
            for problem in violations:
                print(f"violation: {problem}")
            for problem in report.snapshot_overlaps:
                print(f"snapshot overlap: {problem}")
            total = len(violations) + len(report.snapshot_overlaps)
            print("split verify: " + (
                f"{len(violations)} violation(s), {len(report.snapshot_overlaps)} snapshot overlap(s)" if total else "ok"))
            return 1 if total else 0
        cutoff = None
        if args.cutoff:
            cutoff = since_to_ms(args.cutoff, now_ms())
            if cutoff is None:
                print(f"error: cannot read --cutoff {args.cutoff!r}", file=sys.stderr)
                return 2
        with process_lock():
            store = Store()
            try:
                result = splits.build(store, args.parent, cutoff)
            finally:
                store.close()
    except (StoreError, splits.SplitError, manifest.ManifestError, RegistryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result.text())
    return 0


def _train_prepare(args) -> int:
    from healing import splits
    from healing.eval.bench import BenchError
    from healing.registry.store import RegistryError
    from healing.store import Store, StoreError, process_lock
    from healing.train import manifest as train_manifest
    from healing.train.replay import LocalLibriSpeech, ReplayError

    try:
        with process_lock():
            store = Store()
            try:
                result = train_manifest.prepare(
                    store, include_weak=args.include_weak, ratio=args.replay, parent=args.parent, seed=args.seed,
                    source=LocalLibriSpeech(args.replay_source) if args.replay_source else None,
                )
            finally:
                store.close()
    except train_manifest.TestClusterLeak as error:
        print(f"error: test split-cluster in the training data: {error}", file=sys.stderr)
        return 1
    except (StoreError, splits.SplitError, RegistryError, BenchError, ReplayError, train_manifest.TrainError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(result.text())
    return 0


def _golden_pin_problem() -> str | None:
    """G-10: the frozen golden set against the registry's pin."""
    from healing import splits
    from healing.registry.commands import golden_pin_problem
    from healing.store import Store

    store = Store(sweep=False)
    try:
        row = splits.golden_row(store.db)
    finally:
        store.close()
    return golden_pin_problem(row)


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

    job = schedule.JOBS[args.job]
    if args.command == "run":
        if job is not schedule.COLLECT:
            print("error: `run` is the collect job's; run `healing train auto` directly", file=sys.stderr)
            return 2
        return schedule.run_once()
    try:
        if args.command == "install":
            print(f"installed {schedule.install(args.hex, job=job)}")
        elif args.command == "remove":
            schedule.remove(job=job)
            print("removed")
        else:
            print(schedule.show(job=job))
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
    from healing.sets import general
    from healing.sets.general import AmiIhmTest, BuildError, LibriSpeechTestOther, build

    cache = healing_home() / "cache"
    try:
        result = build(
            general.folder(),
            [(LibriSpeechTestOther(cache), args.librispeech), (AmiIhmTest(cache), args.ami)],
            seed=args.seed,
        )
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    state = "unchanged" if result.unchanged else "written"
    print(f"{state}: {result.clips} clips, sha256 {result.sha256}\n{result.manifest}")
    return 0


def _fetch_librispeech(args) -> int:
    from healing.sets import librispeech

    try:
        result = librispeech.fetch(args.dest, progress=librispeech.print_progress)
    except librispeech.FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    state = "unchanged" if result.unchanged else f"fetched {len(result.fetched)} shard(s)"
    print(f"{state}: {len(librispeech.SHARDS)} shards, {result.total_bytes:,} bytes verified\n{result.folder}")
    if result.linked:
        print(f"linked {librispeech.cache_folder()} -> {result.folder}")
    return 0


def _subset(args) -> int:
    from healing.sets import general
    from healing.sets.general import BuildError, subset

    try:
        print(subset(general.folder(), args.prefix, args.name, args.expected))
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _baseline(args) -> int:
    from healing.eval import baseline
    from healing.eval.bench import BenchError
    from healing.eval.run import EvalError

    model = args.model or baseline.DEFAULT_GGUF
    model_id = args.model_id or (baseline.DEFAULT_MODEL_ID if args.model is None else model.stem)
    try:
        result = baseline.baseline(model, model_id, args.output or baseline.default_output())
    except (BenchError, EvalError, baseline.BaselineError) as error:
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
