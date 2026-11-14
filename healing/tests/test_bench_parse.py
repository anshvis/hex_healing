import json

import pytest

from healing.eval.baseline import wer_by_source
from healing.eval.bench import BenchError, parse_jsonl

# Shapes copied from src/transcription/benchmark.rs's `model`, `clip` and
# `summary` records.
RECORDS = [
    {"type": "model", "backend": "transcribe-cpp", "load_ms": 812.0, "clips": 2},
    {"type": "warmup_clip", "id": "ls-1", "inference_ms": 40.0},
    {"type": "measurement", "run": 1, "id": "ls-1", "inference_ms": 30.0},
    {
        "type": "clip", "id": "ls-1", "audio_ms": 3000.0, "inference_ms": 30.0, "rtf": 0.01,
        "word_errors": 1, "reference_words": 10, "expected": "a", "transcript": "a",
        "group": "ordinary", "quality": {},
    },
    {
        "type": "clip", "id": "ami-1", "audio_ms": 2000.0, "inference_ms": 20.0, "rtf": 0.01,
        "word_errors": 3, "reference_words": 5, "expected": "b", "transcript": "c",
        "group": "ordinary", "quality": {},
    },
    {"type": "run", "run": 1, "inference_ms": 50.0},
    {
        "type": "summary", "load_ms": 812.0, "clips": 2, "runs": 1, "median_rtf": 0.01,
        "word_errors": 4, "reference_words": 15, "wer": 4 / 15,
        "quality": {"term_recall": None}, "groups": {"ordinary": {"term_recall": None}},
    },
]


def text(records):
    return "\n".join(json.dumps(record) for record in records) + "\n"


def test_parses_summary_wer_and_per_group_quality():
    run = parse_jsonl(text(RECORDS))
    assert run.summary["wer"] == pytest.approx(4 / 15)
    assert run.summary["groups"] == {"ordinary": {"term_recall": None}}
    assert [clip["id"] for clip in run.clips] == ["ls-1", "ami-1"]
    assert run.model["load_ms"] == 812.0


def test_wer_by_source_sums_errors_over_words():
    run = parse_jsonl(text(RECORDS))
    by_source = wer_by_source(run, {"ls-1": "librispeech", "ami-1": "ami"})
    assert by_source["librispeech"]["wer"] == pytest.approx(0.1)
    assert by_source["ami"]["wer"] == pytest.approx(0.6)


def test_a_changed_output_shape_is_refused():
    summary = dict(RECORDS[-1])
    del summary["groups"]
    with pytest.raises(BenchError, match="output shape changed"):
        parse_jsonl(text(RECORDS[:-1] + [summary]))
    with pytest.raises(BenchError, match="no model or summary"):
        parse_jsonl(text(RECORDS[:-1]))
    with pytest.raises(BenchError, match="clip records"):
        parse_jsonl(text([r for r in RECORDS if r.get("id") != "ami-1" or r["type"] != "clip"]))
