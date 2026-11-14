"""Manifests healing writes must pass the real benchmark's validation.

Runs `hex benchmark-transcription` on a 3-clip manifest built by the general
set's writer. Skipped where no hex binary or model is available.
"""

import json

import numpy as np
import pytest

from healing.eval.baseline import DEFAULT_GGUF
from healing.eval.bench import BenchError, hex_binary, run_benchmark
from healing.sets.general import Utterance, build


class Tones:
    name = "tones"

    def utterances(self):
        for index in range(3):
            tone = (np.sin(np.arange(24_000) * 0.02 * (index + 1)) * 0.2).astype(np.float32)
            yield Utterance(f"tone-{index}", "tones", "hello world",
                            lambda tone=tone: (tone, 16_000))


def available():
    try:
        hex_binary()
    except BenchError:
        return False
    return DEFAULT_GGUF.is_file()


@pytest.mark.skipif(not available(), reason="needs a hex binary and the Unified model")
def test_general_set_manifest_runs_in_the_real_benchmark(tmp_path):
    build(tmp_path, [(Tones(), 3)])
    run, _ = run_benchmark(tmp_path / "manifest.json", model=DEFAULT_GGUF, runs=1, warmups=1)
    assert run.summary["clips"] == 3
    assert run.summary["reference_words"] == 6


@pytest.mark.skipif(not available(), reason="needs a hex binary and the Unified model")
def test_the_benchmark_rejects_an_invalid_group(tmp_path):
    build(tmp_path, [(Tones(), 3)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["clips"][0]["group"] = "general"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BenchError, match="unknown benchmark group"):
        run_benchmark(tmp_path / "manifest.json", model=DEFAULT_GGUF, runs=1, warmups=1)
