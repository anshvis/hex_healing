import json

import numpy as np
import pytest

from healing.sets.general import BuildError, Utterance, ami_reference, build


class FakeSource:
    def __init__(self, name, count, text="the quick brown fox"):
        self.name = name
        self.count = count
        self.text = text

    def utterances(self):
        for index in range(self.count):
            tone = np.sin(np.arange(16_000) * (index + 1) * 0.01).astype(np.float32) * 0.3
            yield Utterance(f"{self.name}-{index:03}", self.name, f"{self.text} {index}",
                            lambda tone=tone: (tone, 16_000))


def test_same_seed_gives_same_manifest_hash(tmp_path):
    first = build(tmp_path / "a", [(FakeSource("ls", 50), 10), (FakeSource("ami", 50), 5)])
    second = build(tmp_path / "b", [(FakeSource("ls", 50), 10), (FakeSource("ami", 50), 5)])
    assert first.sha256 == second.sha256
    assert not first.unchanged
    again = build(tmp_path / "a", [(FakeSource("ls", 50), 10), (FakeSource("ami", 50), 5)])
    assert again.unchanged and again.sha256 == first.sha256


def test_another_seed_or_slice_is_refused_over_an_existing_set(tmp_path):
    build(tmp_path, [(FakeSource("ls", 50), 10)], seed=13)
    with pytest.raises(BuildError, match="never changes"):
        build(tmp_path, [(FakeSource("ls", 50), 10)], seed=14)


def test_changed_audio_is_refused(tmp_path):
    build(tmp_path, [(FakeSource("ls", 50), 3)])
    wav = next((tmp_path / "audio").glob("*.wav"))
    wav.write_bytes(wav.read_bytes()[:-2] + b"\x01\x02")
    with pytest.raises(BuildError, match="missing or changed"):
        build(tmp_path, [(FakeSource("ls", 50), 3)])


def test_manifest_shape_matches_the_benchmark(tmp_path):
    build(tmp_path, [(FakeSource("ls", 20), 4), (FakeSource("ami", 20), 2)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["clips"]) == 6
    for clip in manifest["clips"]:
        assert clip["group"] == "ordinary"
        assert (tmp_path / clip["audio"]).is_file()
        assert clip["expected"].strip()
    sidecar = [json.loads(line) for line in (tmp_path / "clips.jsonl").read_text().splitlines()]
    assert {row["source"] for row in sidecar} == {"ls", "ami"}


def test_too_few_utterances_is_an_error(tmp_path):
    with pytest.raises(BuildError, match="need 10"):
        build(tmp_path, [(FakeSource("ls", 5), 10)])


def test_ami_reference_drops_fillers_and_tags():
    assert ami_reference("UM I MEAN <laugh> UH-HUH WE'RE NOT MM-HMM THERE") == "I MEAN WE'RE NOT THERE"
    assert ami_reference("YEAH OKAY OH") == "YEAH OKAY OH"


def test_an_incomplete_set_is_refused_with_a_clear_reason(tmp_path):
    build(tmp_path, [(FakeSource("ls", 50), 3)])
    (tmp_path / "manifest.sha256").unlink()
    with pytest.raises(BuildError, match="incomplete set"):
        build(tmp_path, [(FakeSource("ls", 50), 3)])


def test_subset_keeps_only_matching_clips(tmp_path):
    from healing.sets.general import subset

    build(tmp_path, [(FakeSource("ls", 20), 4), (FakeSource("ami", 20), 2)])
    path = subset(tmp_path, "ls-", "ls4", 4)
    clips = json.loads(path.read_text())["clips"]
    assert [clip["id"][:3] for clip in clips] == ["ls-"] * 4
    with pytest.raises(BuildError, match="expected 5"):
        subset(tmp_path, "ls-", "ls5", 5)
