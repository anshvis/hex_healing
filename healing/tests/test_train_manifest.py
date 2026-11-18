"""T6.1: the training bundle (weighted, cluster-balanced, 30 s cap, self-distilled replay).

Every test runs under a temp `HEALING_HOME` with a fake base model (a transcriber function) and a fake
LibriSpeech source: no `hex`, no network, never the real data folders. Weighting is checked the way
training sees it: 10,000 draws by the manifest's weights, not by reading the config.
"""

import hashlib
import json

import numpy as np
import pytest
from dataset import BASE_MS, DAY, add_clip, review

from healing import labels, splits
from healing.cli import main
from healing.paths import sets_dir
from healing.registry import commands as registry_commands
from healing.store import Pcm, Store
from healing.train import manifest as tm
from healing.train import pauses, replay
from healing.wer import normalized_words

NOW = BASE_MS + 60 * DAY  # every fixture clip is well outside the 14-day test window
UNIFIED = "parakeet-unified-en-0.6b"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEX_APPLICATION_SUPPORT_DIR", str(tmp_path / "hex-support"))  # never the real one
    return tmp_path / "home"


@pytest.fixture
def store(home):
    s = Store()
    yield s
    s.close()


@pytest.fixture
def base(tmp_path):
    gguf = tmp_path / "base.gguf"
    gguf.write_bytes(b"GGUF base")
    return tm.Base(hashlib.sha256(b"GGUF base").hexdigest(), gguf, UNIFIED)


def noise(seconds: float, seed: int) -> np.ndarray:
    return (np.random.RandomState(seed).randn(int(seconds * 16000)) * 3000).astype(np.int16)


class FakeSource:
    """LibriSpeech-like utterances: uppercase, unpunctuated source texts the replay never uses."""

    name = "fake-librispeech"

    def __init__(self, ids):
        self.ids = list(ids)

    def utterances(self):
        return [replay.Utterance(i, lambda i=i: noise(2, sum(map(ord, i)))) for i in self.ids]


class Base:
    """The fake base model: punctuated, cased replay transcripts; long-clip pieces heard from a script."""

    def __init__(self, pieces=None, replay_text=None):
        self.calls = []
        self.pieces = pieces or (lambda clip, a, b: "")
        self.replay_text = replay_text or (lambda i: f"Hello, Doctor {i.split('-')[-1]}. It's done!")

    def __call__(self, clips):
        self.calls.append([c.id for c in clips])
        out = {}
        for clip in clips:
            if "@" in clip.id:
                name, span = clip.id.split("@")
                a, b = map(int, span.split("-"))
                out[clip.id] = self.pieces(name, a, b)
            else:
                out[clip.id] = self.replay_text(clip.id)
        return out


def source(n=300):
    return FakeSource(f"ls-19-198-{i:04d}" for i in range(n))


def phrase(i: int) -> str:
    words = [f"w{k:03d}" for k in range(400)]
    return " ".join(np.random.RandomState(1000 + i).choice(words, 9))


def verified(store, n, text=None, cluster=None, **kw):
    cid = add_clip(store, n, raw=text or phrase(n), **kw)
    review(store, cid, "accept")
    if cluster:
        store.db.execute("UPDATE clips SET cluster_id = ? WHERE clip_id = ?", (cluster, cid))
        store.db.commit()
    return cid


def prepare(store, base, fake=None, **kw):
    kw.setdefault("source", source())
    return tm.prepare(store, base=base, transcriber=fake or Base(), now=NOW, **kw)


def frequencies(bundle, key):
    rows = tm.read_manifest(bundle)
    drawn = tm.draw([r["weight"] for r in rows], 10_000, seed=7)
    counts = {}
    for index in drawn:
        k = key(rows[index])
        counts[k] = counts.get(k, 0) + 1
    return {k: v / 10_000 for k, v in counts.items()}, rows


def register_test_set(store, split_id, kind, clips):
    store.db.execute("INSERT INTO splits (split_id, kind, parent_version, created_at_ms) VALUES (?, ?, 'current', 1)",
                     (split_id, kind))
    store.db.executemany("INSERT INTO split_members (split_id, clip_id) VALUES (?, ?)", [(split_id, c) for c in clips])
    store.db.commit()


# --- the tests T6.1 lists --------------------------------------------------------------------


def test_cluster_capped_at_three(store, base):
    repeated = [verified(store, i, text="send it to kit by friday", cluster="c-kit") for i in range(50)]
    others = [verified(store, 100 + i) for i in range(20)]
    result = prepare(store, base, ratio=0.0)
    rows = tm.read_manifest(result.path)
    mass = {}
    for r in rows:
        mass[r["cluster"]] = mass.get(r["cluster"], 0.0) + r["weight"]
    assert mass["c-kit"] == pytest.approx(3.0)  # 50 copies count as 3
    assert all(m <= tm.CLUSTER_CAP + 1e-9 for m in mass.values())
    assert {r["id"] for r in rows} == set(repeated + others)
    freq, _ = frequencies(result.path, lambda r: r["cluster"] == "c-kit")
    assert freq[True] == pytest.approx(3 / 23, abs=0.01)  # the sampler sees 3 of 23, not 50 of 70


def test_replay_ratio(store, base):
    for i in range(30):
        verified(store, i)
    fake = Base()
    result = prepare(store, base, fake=fake)
    freq, rows = frequencies(result.path, lambda r: r["source"])
    assert freq["replay"] == pytest.approx(0.20, abs=0.01)
    replay_mass = sum(r["weight"] for r in rows if r["source"] == "replay")
    assert replay_mass / sum(r["weight"] for r in rows) == pytest.approx(0.20, abs=1e-9)
    assert result.replay_share == pytest.approx(0.20)
    # transcripts were made once and cached by the base's sha: the next round transcribes nothing
    assert [len(c) for c in fake.calls] == [200, 100]  # in batches, so memory stays bounded
    assert replay.cache_path(base.sha256).is_file()
    verified(store, 500)
    again = Base()
    prepare(store, base, fake=again)
    assert again.calls == []


def test_no_test_split_cluster(store, base, monkeypatch):
    trainable = [verified(store, i) for i in range(10)]
    golden_twin = verified(store, 50, text=phrase(3))  # same words as trainable clip 3
    register_test_set(store, "golden-v1", "golden", [golden_twin])
    result = prepare(store, base, ratio=0.0)
    ids = {r["id"] for r in tm.read_manifest(result.path)}
    assert trainable[3] not in ids and golden_twin not in ids  # the zone kept them out
    # a clip that slipped past the zones fails the run; it is not silently dropped
    monkeypatch.setattr(splits.Zones, "may_train", lambda self, clip: True)
    verified(store, 60)
    with pytest.raises(tm.TestClusterLeak, match="split-cluster with a test set"):
        prepare(store, base, ratio=0.0)
    assert len(list(tm.train_dir().glob("train-*"))) == 1  # no bundle was written for the failed round


def test_weak_excluded_by_default(store, base):
    strong = [verified(store, i) for i in range(5)]
    weak = [add_clip(store, 100 + i, raw=phrase(100 + i), outcome="unchanged") for i in range(5)]
    labels.refresh(store)
    ids = {r["id"] for r in tm.read_manifest(prepare(store, base, ratio=0.0).path)}
    assert ids == set(strong)
    rows = tm.read_manifest(prepare(store, base, ratio=0.0, include_weak=True).path)
    weights = {r["id"]: r["weight"] for r in rows}
    assert set(weights) == set(strong + weak)
    assert all(weights[c] == pytest.approx(0.2) for c in weak) and all(weights[c] == 1.0 for c in strong)


def speech(parts):
    """`(speech_ms, pause_ms)` pairs: 240 ms bursts at -20 dBFS with 60 ms dips, then silence (as
    `pause_segmenter.rs`'s tests build an utterance)."""
    rng = np.random.RandomState(7)
    out = []
    for speech_ms, pause_ms in parts:
        remaining = speech_ms
        while remaining > 0:
            burst = remaining if remaining <= 360 else 240
            out.append(rng.uniform(-1, 1, burst * 16) * 0.1 * np.sqrt(3) * 32767)
            remaining -= burst
            if remaining > 0:
                out.append(np.zeros(60 * 16))
                remaining -= 60
        out.append(np.zeros(pause_ms * 16))
    return np.concatenate(out).astype(np.int16)


SEGMENTS = ["Alpha one, bravo two.", "Charlie three; delta four.", "Echo five foxtrot six.",
            "Golf seven, hotel eight.", "India nine juliet ten!"]


def long_clip(store, n=900):
    audio = speech([(14_000, 1_000)] * 5)
    assert len(audio) == 75 * 16000
    pcm = Pcm(hashlib.sha256(audio.astype("<i2").tobytes()).hexdigest(), len(audio), audio)
    store.write_audio(pcm)
    cid = verified(store, n, text="alpha one bravo two charlie three delta four echo five foxtrot six golf seven hotel eight india nine juliet ten")
    review(store, cid, "fix", text=" ".join(SEGMENTS), at=BASE_MS + 1)
    store.db.execute("UPDATE clips SET audio_sha256 = ?, audio_ms = 75000 WHERE clip_id = ?", (pcm.sha256, cid))
    store.db.commit()
    return cid


def heard(_clip, a, b):
    """What the base model hears in a piece: the segments starting inside it, lower case, no marks."""
    return " ".join(" ".join(normalized_words(text)) for k, text in enumerate(SEGMENTS) if a <= k * 15 * 16000 < b)


def test_long_clip_cut_at_pauses(store, base):
    cid = long_clip(store)
    result = prepare(store, base, fake=Base(pieces=heard), ratio=0.0)
    rows = [r for r in tm.read_manifest(result.path) if r["id"].startswith(cid)]
    assert len(rows) >= 3 and all(r["duration"] <= 30.0 for r in rows)
    assert " ".join(r["text"] for r in rows) == " ".join(SEGMENTS)  # every word once, marks and case kept
    assert [r["id"] for r in rows] == [f"{cid}-p{k}" for k in range(len(rows))]
    # each piece's text is what was said in it
    assert rows[0]["text"] == "Alpha one, bravo two. Charlie three; delta four."


def test_long_clip_skipped_when_the_boundary_is_unsure(store, base):
    cid = long_clip(store)
    keep = verified(store, 1)
    garbled = Base(pieces=lambda clip, a, b: "something else entirely")
    result = prepare(store, base, fake=garbled, ratio=0.0)
    ids = {r["id"] for r in tm.read_manifest(result.path)}
    assert ids == {keep} and not any(i.startswith(cid) for i in ids)
    assert result.skipped == {"over 30 s and its text could not be split at the pauses": 1}


def test_replay_keeps_punctuation(store, base):
    for i in range(20):
        verified(store, i)
    rows = tm.read_manifest(prepare(store, base).path)
    replayed = [r for r in rows if r["source"] == "replay"]
    assert replayed
    for r in replayed:
        assert r["text"] == f"Hello, Doctor {r['id'].split('-')[-1]}. It's done!"  # verbatim, never lowercased


def test_replay_never_from_test_sets(store, base):
    for i in range(20):
        verified(store, i)
    gold = verified(store, 90, text="a golden sentence that nobody may train on today")
    register_test_set(store, "golden-v1", "golden", [gold])
    (sets_dir() / "general").mkdir(parents=True)
    (sets_dir() / "general" / "manifest.json").write_text(json.dumps(
        {"clips": [{"id": "ls-3-4-0001", "expected": "the general set says this one thing out loud"}]}))
    near = {"ls-19-198-0001": "A golden sentence that nobody may train on today.",
            "ls-19-198-0002": "The general set says this one thing out loud!"}
    fake = Base(replay_text=lambda i: near.get(i, f"Plain words for clip {i[-4:]}, okay."))
    rows = tm.read_manifest(prepare(store, base, fake=fake, source=source(40)).path)
    replay_ids = {r["id"] for r in rows if r["source"] == "replay"}
    assert replay_ids and not replay_ids & set(near)
    # a source that holds a test clip is refused outright
    verified(store, 91)
    with pytest.raises(replay.ReplayError, match="general test set"):
        prepare(store, base, source=FakeSource(["ls-3-4-0001", "ls-19-198-0003"]))


# --- the rest ------------------------------------------------------------------------------


def test_padding_matches_what_hex_fed_the_model(store, base):
    unified = verified(store, 1, model_id=UNIFIED)  # 1 s of audio
    other = verified(store, 2, model_id="parakeet-tdt-0.6b-v2")
    rows = {r["id"]: r for r in tm.read_manifest(prepare(store, base, ratio=0.0).path)}
    assert rows[unified]["duration"] == pytest.approx(1.5 + 0.4)  # up to 1.5 s, then Unified's 400 ms
    assert rows[other]["duration"] == pytest.approx(1.5)


def test_pause_cuts_follow_the_segmenter_rules():
    audio = speech([(2_000, 700), (2_000, 700), (9_000, 400), (1_000, 0)])
    cuts = pauses.candidate_cuts(audio)
    # no cut before 3 s of piece; then a 600 ms pause cuts; after 8 s a 300 ms pause is enough
    assert len(cuts) == 2
    assert 4.7 * 16000 < cuts[0] < 5.4 * 16000
    assert 14.4 * 16000 < cuts[1] < 14.8 * 16000
    assert pauses.pieces(noise(31, 1), 30 * 16000) is None  # nothing to cut at: the clip is skipped


def test_bundle_hash_is_verified_and_recorded_in_the_registry(store, base, capsys):
    for i in range(5):
        verified(store, i)
    result = prepare(store, base, ratio=0.0)
    assert tm.verify_bundle(result.path) == result.sha256
    assert (result.path / "bundle.sha256").read_bytes().count(b"\n") == 6  # manifest + 5 WAVs
    info = json.loads((result.path / "bundle.json").read_text())
    assert info["train_split"] == result.train_split and "text" not in json.dumps(info)
    assert registry_commands._bundle(result.train_split) == result.sha256
    wav = next((result.path / "audio").iterdir())
    wav.write_bytes(wav.read_bytes()[:-2] + b"\x01\x02")
    with pytest.raises(registry_commands.CommandError, match="damaged"):
        registry_commands._bundle(result.train_split)


def test_cli_prepare(store, capsys):
    verified(store, 1)
    assert main(["train", "prepare"]) == 1  # replay needs LibriSpeech on disk: nothing is downloaded
    assert "parquet shards" in capsys.readouterr().err
    assert main(["train", "prepare", "--replay", "1.5"]) == 1
    assert "under 1" in capsys.readouterr().err
    assert main(["train", "prepare", "--replay", "0"]) == 0  # no replay and no long clip: no base model needed
    out = capsys.readouterr().out
    assert "sha256 " in out and "1 clips as 1 entries" in out


# --- review fixes (Codex, 8b0b200f) --------------------------------------------------------------

# A stand-in for `hex benchmark-transcription` that enforces `src/transcription/benchmark.rs`'s argument
# rules (`run`: at least one warmup and one run) and its catalog match (`Backend::transcribe_cpp`: the
# model file's name and size), and logs what it was given.
FAKE_BENCH = """#!{python}
import json, os, sys
args = sys.argv[1:]
assert args[0] == "benchmark-transcription", args
manifest = args[1]
opts = dict(zip(args[2::2], args[3::2]))
log = open(os.environ["FAKE_LOG"], "a")
if int(opts["--warmups"]) == 0:
    print("Error: benchmark requires at least one warmup run", file=sys.stderr); sys.exit(1)
if int(opts["--runs"]) == 0:
    print("Error: benchmark requires at least one measured run", file=sys.stderr); sys.exit(1)
model = opts["--model"]
catalog = os.path.basename(model) == os.environ["FAKE_CATALOG_NAME"] and os.path.getsize(model) == int(os.environ["FAKE_CATALOG_BYTES"])
log.write(json.dumps({{"args": args, "catalog": catalog}}) + "\\n")
clips = json.load(open(manifest))["clips"]
print(json.dumps({{"type": "model", "path": model}}))
for c in clips:
    print(json.dumps({{"type": "clip", "id": c["id"], "transcript": "Heard, " + c["id"] + ".", "word_errors": 0, "reference_words": 1}}))
print(json.dumps({{"type": "summary", "clips": len(clips), "wer": 0, "word_errors": 0, "reference_words": len(clips),
                  "median_rtf": 0, "quality": {{}}, "groups": {{}}}}))
"""


@pytest.fixture
def fake_hex(tmp_path, monkeypatch, base):
    import sys

    binary = tmp_path / "hex"
    binary.write_text(FAKE_BENCH.format(python=sys.executable))
    binary.chmod(0o755)
    log = tmp_path / "bench.log"
    monkeypatch.setenv("HEX_BIN", str(binary))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_CATALOG_NAME", "parakeet-unified-en-0.6b-Q8_0.gguf")
    monkeypatch.setenv("FAKE_CATALOG_BYTES", str(base.gguf.stat().st_size))
    return lambda: [json.loads(line) for line in log.read_text().splitlines()]


def test_fake_bench_mirrors_the_rust_argument_rules():
    from healing.paths import repo_root

    rust = (repo_root() / "src" / "transcription" / "benchmark.rs").read_text()
    assert "if warmups == 0 {" in rust and 'bail!("benchmark requires at least one warmup run")' in rust
    assert "if runs == 0 {" in rust and "catalog_match(None, None, name, meta.len())" in rust


def test_real_transcriber_argv_is_accepted_by_the_benchmark(home, base, fake_hex):
    transcribe = replay.benchmark_transcriber(base.gguf, "parakeet-unified-en-0.6b-Q8_0.gguf")
    out = transcribe([replay.Audio("ls-1-2-0003", noise(2, 3))])
    assert out == {"ls-1-2-0003": "Heard, ls-1-2-0003."}
    (call,) = fake_hex()
    opts = dict(zip(call["args"][2::2], call["args"][3::2]))
    assert int(opts["--warmups"]) >= 1 and int(opts["--runs"]) >= 1


def test_replay_labels_hear_the_catalog_tail(home, base, fake_hex):
    """v0 is a hash-named registry object; the benchmark must still see it as the catalog Unified model,
    or replay transcripts are made without the 400 ms the training audio gets."""
    replay.benchmark_transcriber(base.gguf, "parakeet-unified-en-0.6b-Q8_0.gguf")([replay.Audio("a", noise(2, 1))])
    (call,) = fake_hex()
    assert call["catalog"] is True
    assert tm.catalog_filename({"source_path": "/x/models/parakeet-unified-en-0.6b-Q8_0.gguf"}, UNIFIED) == "parakeet-unified-en-0.6b-Q8_0.gguf"
    assert tm.catalog_filename({}, UNIFIED) == "parakeet-unified-en-0.6b-Q8_0.gguf"


def test_a_piece_matching_a_test_reference_never_trains(store, base):
    gold = verified(store, 950, text=" ".join(SEGMENTS[:2]))  # exactly piece 0's text
    register_test_set(store, "golden-v1", "golden", [gold])
    cid = long_clip(store)
    keep = verified(store, 1)
    result = prepare(store, base, fake=Base(pieces=heard), ratio=0.0)
    ids = {r["id"] for r in tm.read_manifest(result.path)}
    assert keep in ids and not any(i.startswith(cid) for i in ids)
    assert result.skipped["over 30 s and a piece is near a test set"] == 1
    entry = tm.Entry("x-p0", "user", " ".join(SEGMENTS[:2]), 1.0, noise(1, 1), "x", ["unrelated words entirely here now"])
    with pytest.raises(tm.TestClusterLeak):
        tm.assert_no_test_cluster(store.db, [entry])


def test_repeated_words_make_the_boundary_ambiguous():
    reference = "Alpha beta. Alpha beta. Alpha beta."
    assert pauses.split_reference(reference, ["alpha beta", "alpha beta"]) is None
    assert pauses.split_reference(reference, ["alpha beta alpha beta", "alpha beta"]) == ["Alpha beta. Alpha beta.", "Alpha beta."]


def test_pinned_test_references_are_held_out(store, base):
    from healing import manifest as set_manifest

    edited = verified(store, 960, text="quiet morning notes about nothing in particular here")
    review(store, edited, "fix", text="the pinned reference names kip at the harbour today", at=BASE_MS + 1)
    folder = sets_dir() / "rolling-current-x"
    digest, _ = set_manifest.write_set(folder, [{"id": edited, "expected": "the pinned reference names kip at the harbour today", "terms": []}],
                                       lambda _: noise(1, 1))
    store.db.execute("INSERT INTO splits (split_id, kind, parent_version, created_at_ms, manifest_path, manifest_sha256,"
                     " sidecar_sha256) VALUES ('rolling-current-x', 'rolling', 'current', 1, ?, ?, ?)",
                     (str(folder / "manifest.json"), digest, set_manifest.sidecar_digest(folder)))
    store.db.execute("INSERT INTO split_members VALUES ('rolling-current-x', ?)", (edited,))
    store.db.commit()
    review(store, edited, "fix", text="a later correction says something else", at=BASE_MS + 5)  # live text moves on
    verified(store, 961, text="the pinned reference names kip at the harbour today")  # what the set still scores
    with pytest.raises(tm.TestClusterLeak):
        prepare(store, base, ratio=0.0)
    (folder / "manifest.json").write_text("{}")  # a changed pin is refused, not read
    with pytest.raises(tm.TrainError, match="rolling-current-x"):
        tm.held_out_texts(store.db)
