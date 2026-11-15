import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from dataset import BASE_MS, DAY, add_clip, review, row

from healing import dedupe, golden, labels, manifest, review as review_module, splits
from healing.cli import main
from healing.eval.baseline import DEFAULT_GGUF
from healing.eval.bench import BenchError, hex_binary, run_benchmark
from healing.sets import general
from healing.store import Store

CUTOFF = BASE_MS


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def store(home):
    s = Store()
    yield s
    s.close()


def sentences(count, seed=7):
    rng = np.random.RandomState(seed)
    words = [f"w{i:03d}" for i in range(400)]
    return [" ".join(rng.choice(words, 9)) for _ in range(count)]


def world(store):
    """300 clips over 40 sessions (20 before the cut-off, 20 after), with repeats that differ in
    misrecognition and rate, and the labels and reviews a real dataset would have."""
    phrases = sentences(100)
    ids = {"strong": [], "weak": [], "verified_after": [], "verified_before": [], "plain": [], "after": [], "before": []}
    for i in range(294):
        session = i * 40 // 294
        start = CUTOFF + int((session - 19.5) * DAY)
        ts = start + (i % 8) * 60_000 + (i // 8 % 3) * 7  # a minute apart: one session
        phrase = phrases[i % 100]
        kind = i % 6
        if kind == 0:
            cid = add_clip(store, i, raw=f"{phrase} kip", ts=ts, outcome="edited", edited=f"{phrase} kit",
                           final=f"{phrase} kip.", audio_ms=2500 + (i % 5) * 900, app="Notes" if i % 2 else "Slack")
            ids["strong"].append(cid)
        elif kind == 1:
            cid = add_clip(store, i, raw=phrase, ts=ts, outcome="unchanged", audio_ms=2500 + (i % 5) * 900)
            ids["weak"].append(cid)
        else:
            cid = add_clip(store, i, raw=phrase, ts=ts, audio_ms=2500 + (i % 5) * 900, app="Mail")
            ids["plain"].append(cid)
        ids["after" if session >= 20 else "before"].append(cid)
    labels.label_all(store)
    return ids, phrases


def build_at(store, later=0):
    return splits.build(store, cutoff_ms=CUTOFF, now=CUTOFF + 30 * DAY + later)


def verify_random(store, cids):
    for cid in cids:
        review(store, cid)
    labels.refresh(store, list(cids))


def build_world(store):
    ids, phrases = world(store)
    # random verified after the cut-off feed rolling; random verified before it never can
    after_plain = [c for c in ids["plain"] if c in set(ids["after"])][:25]
    before_plain = [c for c in ids["plain"] if c in set(ids["before"])][:10]
    verify_random(store, after_plain + before_plain)
    # a weak clip verified by the uncertain strategy feeds training only
    uncertain = [c for c in ids["weak"] if c in set(ids["after"])][:3]
    for cid in uncertain:
        review(store, cid, strategy="uncertain")
    labels.refresh(store, uncertain)
    ids.update(after_plain=after_plain, before_plain=before_plain, uncertain=uncertain)
    return ids, phrases


# --- clusters -------------------------------------------------------------------------------


def test_union_find_joins_misrecognition_variants():
    clusters = splits.split_clusters({
        "heard_kip": ["send it to kit", "send it to kip"],  # corrected to "kit"
        "heard_kit": ["send it to kit"],
        "other": ["what is the weather in berlin tomorrow"],
    })
    assert clusters["heard_kip"] == clusters["heard_kit"] != clusters["other"]


def test_union_find_is_transitive_where_representatives_are_not():
    base = "please send the quarterly report to kit before the end of the day tomorrow"
    a, b, c = base, base.replace("tomorrow", "today"), base.replace("tomorrow", "today").replace("please", "kindly")
    assert dedupe.jaccard(dedupe.shingles(dedupe.normalize(a)), dedupe.shingles(dedupe.normalize(c))) < 0.8
    clusters = splits.split_clusters({"a": [a], "b": [b], "c": [c]})
    assert clusters["a"] == clusters["b"] == clusters["c"]


def test_no_duration_condition(store):
    slow = add_clip(store, 1, raw="send it to kit please now", audio_ms=9000)
    fast = add_clip(store, 2, raw="send it to kit please now", audio_ms=2500)
    clusters = splits.clip_clusters(store.db)
    assert clusters[slow] == clusters[fast]
    # T1.3's storage clusters would have kept them apart
    stored = dedupe.cluster([dedupe.Clip(slow, 1, "send it to kit please now", 9000),
                             dedupe.Clip(fast, 2, "send it to kit please now", 2500)])
    assert stored[slow] != stored[fast]


def test_sessions_cut_at_ten_minute_gaps(store):
    a = add_clip(store, 1, ts=BASE_MS)
    b = add_clip(store, 2, ts=BASE_MS + 9 * 60_000)
    c = add_clip(store, 3, ts=BASE_MS + 9 * 60_000 + 10 * 60_000 + 1)
    s = splits.sessions(store.db)
    assert s[a] == s[b] and s[c] != s[b]


# --- the AC-8 fixture -------------------------------------------------------------------------


def test_fixture_shape(store):
    ids, _ = world(store)
    assert store.db.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 294
    assert len({s for s, _ in splits.sessions(store.db).values()}) == 40


def test_rolling_only_after_cutoff_and_random_verified(store):
    ids, _ = build_world(store)
    result = build_at(store)
    assert result.cutoff_ms == CUTOFF
    members = {r["clip_id"] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling'")}
    after = {c for c in ids["after"]}
    assert members and members <= after
    for cid in members:
        r = row(store, cid)
        assert r["label"] == "verified" and r["review_strategy"] == "random" and r["timestamp_ms"] > result.cutoff_ms
    assert not members & set(ids["before_plain"])
    assert not members & set(ids["uncertain"])
    assert not members & (set(ids["strong"]) | set(ids["weak"]))


def test_strong_and_weak_never_gate(store):
    ids, _ = build_world(store)
    build_at(store)
    assert splits.verify(store) == []
    rolling = store.db.execute("SELECT split_id FROM splits WHERE kind = 'rolling' AND manifest_path IS NOT NULL").fetchone()[0]
    for bad, label in ((ids["strong"][0], "strong"), (ids["weak"][0], "weak")):
        store.db.execute("UPDATE clips SET label = ? WHERE clip_id = (SELECT clip_id FROM split_members WHERE split_id = ? LIMIT 1)", (label, rolling))
        problems = splits.verify(store)
        assert any(f"comes from a {label} label" in p for p in problems), problems
        store.db.execute("UPDATE clips SET label = 'verified' WHERE label = ?", (label,))
        store.db.execute("UPDATE clips SET label = 'verified' WHERE clip_id IN (SELECT clip_id FROM split_members WHERE split_id = ?)", (rolling,))
        store.db.commit()
        assert bad


def test_no_split_cluster_is_shared_and_the_kip_kit_pair_never_straddles(store):
    ids, phrases = build_world(store)
    # "send it to kip" (slow, corrected to kit, before the cut-off) and "send it to kit" (fast, verified, after)
    kip = add_clip(store, 900, raw="send it to kip", ts=CUTOFF - 5 * DAY, outcome="edited", edited="Send it to Kit.", audio_ms=6000)
    kit = add_clip(store, 901, raw="send it to kit", ts=CUTOFF + 6 * DAY, audio_ms=2500)
    labels.label_all(store)
    verify_random(store, [kit])
    result = build_at(store)
    assert row(store, kip)["split_cluster"] == row(store, kit)["split_cluster"]
    rolling = {r[0] for r in store.db.execute("SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling'")}
    train = {r[0] for r in store.db.execute("SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'train'")}
    assert kit in rolling and kip not in train
    tests = {row(store, c)["split_cluster"] for c in rolling} | {
        row(store, r[0])["split_cluster"] for r in store.db.execute(
            "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'edited'")}
    assert train and not {row(store, c)["split_cluster"] for c in train} & tests
    assert result.excluded_train > 0
    assert splits.verify(store) == []


def test_verify_catches_a_leak(store):
    build_world(store)
    build_at(store)
    rolling_clip = store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling' LIMIT 1").fetchone()[0]
    train_id = store.db.execute("SELECT split_id FROM splits WHERE kind = 'train'").fetchone()[0]
    store.db.execute("INSERT INTO split_members VALUES (?, ?)", (train_id, rolling_clip))
    store.db.commit()
    assert any("shares a split-cluster" in p for p in splits.verify(store))


def test_verify_catches_a_pre_cutoff_rolling_clip(store):
    ids, _ = build_world(store)
    build_at(store)
    rolling = store.db.execute("SELECT split_id FROM splits WHERE kind = 'rolling' AND manifest_path IS NOT NULL").fetchone()[0]
    store.db.execute("INSERT INTO split_members VALUES (?, ?)", (rolling, ids["before_plain"][0]))
    store.db.commit()
    assert any("predates the cut-off" in p for p in splits.verify(store))


def test_edited_span_set_is_the_strong_clips_of_later_sessions(store):
    ids, _ = build_world(store)
    build_at(store)
    edited = store.db.execute("SELECT * FROM splits WHERE kind = 'edited'").fetchone()
    members = {r[0] for r in store.db.execute("SELECT clip_id FROM split_members WHERE split_id = ?", (edited["split_id"],))}
    assert members and members <= set(ids["strong"]) & set(ids["after"])
    sidecar = [json.loads(line) for line in (Path(edited["manifest_path"]).parent / "clips.jsonl").read_text().splitlines()]
    assert all(json.loads(row(store, s["id"])["edited_positions"]) == s["edited_positions"] for s in sidecar)
    assert {s["application"] for s in sidecar} and all(s["session"].startswith("s") for s in sidecar)


def test_rebuild_replaces_the_previous_sets(store):
    build_world(store)
    build_at(store)
    before = store.db.execute("SELECT COUNT(*) FROM splits").fetchone()[0]
    build_at(store, 5000)
    assert store.db.execute("SELECT COUNT(*) FROM splits").fetchone()[0] == before
    kept = {p.name for p in (Path(store.root).parent / "sets").iterdir()}
    assert len([k for k in kept if k.startswith("rolling-")]) == 1


def test_cutoff_is_kept_between_builds(store):
    assert splits.ensure_cutoff(store, "current", now=CUTOFF) == CUTOFF - 14 * DAY
    assert splits.ensure_cutoff(store, "current", now=CUTOFF + 5 * DAY) == CUTOFF - 14 * DAY
    assert splits.ensure_cutoff(store, "current", CUTOFF, now=CUTOFF + 5 * DAY) == CUTOFF
    assert splits.ensure_cutoff(store, "current", now=CUTOFF + 9 * DAY) == CUTOFF


def test_small_rolling_flagged_by_words(store):
    build_world(store)
    result = build_at(store)
    assert result.rolling.n_words < splits.ROLLING_MIN_WORDS and result.rolling_small
    assert any(w.startswith("rolling_small") for w in result.warnings)
    assert "rolling_small" in result.text()


def test_manifest_rules_mirror_the_benchmark(store):
    ids, _ = build_world(store)
    build_at(store)
    for kind in ("rolling", "edited"):
        path = Path(store.db.execute("SELECT manifest_path FROM splits WHERE kind = ?", (kind,)).fetchone()[0])
        document = json.loads(path.read_text())
        manifest.validate(document)
        assert {c["group"] for c in document["clips"]} <= {"target", "ordinary"}
        for c in document["clips"]:
            for term in c.get("expected_terms", []):
                assert manifest.term_key(term) in {manifest.term_key(v) for v in document["vocabulary"]}
                assert manifest.term_count(c["expected"], term) > 0
    edited = json.loads(Path(store.db.execute("SELECT manifest_path FROM splits WHERE kind = 'edited'").fetchone()[0]).read_text())
    assert any(c["group"] == "target" and c["expected_terms"] == ["kit"] for c in edited["clips"])
    assert edited["vocabulary"] == ["kit"]


@pytest.mark.parametrize("change, message", [
    (lambda m: m["clips"][0].update(group="general"), "unknown benchmark group"),
    (lambda m: m["clips"][0].update(group="target"), "no expected terms"),
    (lambda m: m["clips"][0].update(expected_terms=["kit"]), "guard clip"),
    (lambda m: m["clips"][1].update(id=m["clips"][0]["id"]), "duplicated"),
    (lambda m: m["clips"][0].update(expected="..."), "cannot be empty"),
    (lambda m: m.update(vocabulary=["kit", "Kit"]), "unique"),
    (lambda m: m["clips"][0].update(group="target", expected_terms=["zebra"]), "in vocabulary and reference"),
])
def test_validate_rejects_what_the_benchmark_rejects(change, message):
    document = {"vocabulary": ["kit"], "clips": [
        {"id": "a", "audio": "a.wav", "expected": "send it to kit", "group": "ordinary"},
        {"id": "b", "audio": "b.wav", "expected": "hello world", "group": "ordinary"}]}
    change(document)
    with pytest.raises(manifest.ManifestError, match=message):
        manifest.validate(document)


def test_term_count_matches_the_benchmark_rules():
    assert manifest.term_count("use open code today", "OpenCode") == 1
    assert manifest.term_count("hexadecimal", "Hex") == 0
    assert manifest.term_count("a toolkit", "Kit") == 0
    assert manifest.term_count("kit and Kit.", "kit") == 2


# --- golden ----------------------------------------------------------------------------------


@pytest.fixture
def small_golden(monkeypatch):
    monkeypatch.setattr(splits, "GOLDEN_SIZE", 5)


class Accept:
    def __init__(self, *answers):
        self.answers = list(answers)

    def ask(self, prompt):
        return self.answers.pop(0) if self.answers else ""

    def play(self, path):
        pass


def freeze_golden(store):
    ids, _ = build_world(store)
    script = Accept()
    result = golden.run(store, 20, player=script.play, ask=script.ask, out=lambda *_: None, seed=3)
    return ids, result


def test_golden_freeze_pins_hash(store, home, small_golden):
    freeze_golden(store)
    pinned = store.db.execute("SELECT * FROM splits WHERE split_id = 'golden-v1'").fetchone()
    assert pinned["frozen"] == 1 and pinned["n_clips"] == 5
    folder = home / "sets" / "golden-v1"
    digest = (folder / "manifest.sha256").read_text().split()[0]
    assert digest == pinned["manifest_sha256"] == general.verify(folder)
    assert store.db.execute("SELECT COUNT(*) FROM clips WHERE golden = 1").fetchone()[0] == 5
    # one per split-cluster, every one random and verified
    members = [r[0] for r in store.db.execute("SELECT clip_id FROM split_members WHERE split_id = 'golden-v1'")]
    clusters = splits.clip_clusters(store.db)
    assert len({clusters[m] for m in members}) == 5
    assert all(row(store, m)["label"] == "verified" and row(store, m)["review_strategy"] == "random" for m in members)
    # frozen: refuses more, and the schema refuses edits to it
    with pytest.raises(golden.GoldenError, match="frozen"):
        golden.run(store)
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError, match="frozen split never changes"):
        store.db.execute("UPDATE splits SET manifest_sha256 = 'x' WHERE split_id = 'golden-v1'")
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("DELETE FROM split_members WHERE split_id = 'golden-v1'")
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("INSERT INTO splits (split_id, kind, created_at_ms, frozen) VALUES ('golden-v2', 'golden', 1, 1)")
        store.db.execute("INSERT INTO splits (split_id, kind, created_at_ms, frozen) VALUES ('golden-v3', 'golden', 1, 1)")
    store.db.rollback()
    assert splits.verify(store) == []


def test_golden_pending_reports_na_and_does_not_freeze(store, small_golden):
    build_world(store)
    script = Accept()
    golden.run(store, 3, player=script.play, ask=script.ask, out=lambda *_: None, seed=3)
    pending = store.db.execute("SELECT * FROM splits WHERE split_id = 'golden-v1'").fetchone()
    assert pending["frozen"] == 0 and pending["n_clips"] == 3 and pending["manifest_path"] is None
    result = build_at(store)
    assert result.golden is None and result.golden_pending == 3 and "n/a" in result.text()
    assert splits.verify(store) == []
    # more reviews top it up and it freezes at the size
    golden.run(store, 20, player=script.play, ask=script.ask, out=lambda *_: None, seed=4)
    assert store.db.execute("SELECT frozen, n_clips FROM splits WHERE split_id = 'golden-v1'").fetchone()[:] == (1, 5)


def test_golden_rejects_are_not_added(store, small_golden):
    build_world(store)
    script = Accept("r", "r", "r")
    golden.run(store, 3, player=script.play, ask=script.ask, out=lambda *_: None, seed=3)
    assert store.db.execute("SELECT COUNT(*) FROM split_members WHERE split_id = 'golden-v1'").fetchone()[0] == 0


def test_golden_clusters_are_excluded_from_rolling_and_train(store, small_golden):
    freeze_golden(store)
    result = build_at(store)
    gold_clusters = {row(store, r[0])["split_cluster"] for r in store.db.execute("SELECT clip_id FROM split_members WHERE split_id = 'golden-v1'")}
    other = store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind IN ('rolling', 'train')").fetchall()
    assert not {row(store, r[0])["split_cluster"] for r in other} & gold_clusters
    assert splits.verify(store) == [] and result.golden.n_clips == 5


def test_golden_tamper_fails_verify(store, home, small_golden):
    freeze_golden(store)
    build_at(store)
    assert splits.verify(store) == []
    folder = home / "sets" / "golden-v1"
    members = [r[0] for r in store.db.execute("SELECT clip_id FROM split_members WHERE split_id = 'golden-v1'")]

    # 1. the manifest is edited
    original = (folder / "manifest.json").read_bytes()
    document = json.loads(original)
    document["clips"][0]["expected"] += " extra"
    (folder / "manifest.json").write_text(json.dumps(document))
    assert any("no longer matches its pin" in p for p in splits.verify(store))
    (folder / "manifest.json").write_bytes(original)
    assert splits.verify(store) == []

    # 2. a clip's audio is edited
    wav = folder / "audio" / f"{members[0]}.wav"
    saved = wav.read_bytes()
    wav.write_bytes(saved[:-2] + b"\x01\x01")
    assert any("missing or changed" in p for p in splits.verify(store))
    wav.write_bytes(saved)
    assert splits.verify(store) == []

    # 3. a clip's reference is edited in the dataset
    real = row(store, members[0])["reference_text"]
    store.db.execute("UPDATE clips SET reference_text = 'something else entirely' WHERE clip_id = ?", (members[0],))
    store.db.commit()
    assert any("reference changed" in p for p in splits.verify(store))
    store.db.execute("UPDATE clips SET reference_text = ? WHERE clip_id = ?", (real, members[0]))
    store.db.commit()
    assert splits.verify(store) == []

    # 4. a later review rewrites a golden clip
    stored = row(store, members[1])["reference_text"]
    review(store, members[1], "fix", "a different reference", at=BASE_MS * 2)
    assert any("latest review no longer matches" in p for p in splits.verify(store))
    store.db.execute("DELETE FROM reviews WHERE text = 'a different reference'")
    store.db.commit()
    assert splits.verify(store) == [] and row(store, members[1])["reference_text"] == stored


def test_commands(store, home, capsys, small_golden):
    build_world(store)
    store.close()
    assert main(["split", "build", "--cutoff", "14d"]) in (0,)
    out = capsys.readouterr().out
    assert "rolling:" in out and "excluded_train_clips" in out
    assert main(["split", "verify"]) == 0
    assert "split verify: ok" in capsys.readouterr().out


# --- the real benchmark ---------------------------------------------------------------------


def _live():
    try:
        hex_binary()
    except BenchError:
        return False
    return DEFAULT_GGUF.is_file()


@pytest.mark.skipif(not _live(), reason="needs a hex binary and the Unified model")
def test_manifests_accepted_by_real_benchmark(store, home, small_golden):
    ids, _ = build_world(store)
    script = Accept()
    golden.run(store, 20, player=script.play, ask=script.ask, out=lambda *_: None, seed=3)
    build_at(store)
    checked = 0
    for kind in ("rolling", "edited", "golden"):
        path = Path(store.db.execute("SELECT manifest_path FROM splits WHERE kind = ?", (kind,)).fetchone()[0])
        count = len(json.loads(path.read_text())["clips"])
        run, _ = run_benchmark(path, model=DEFAULT_GGUF, runs=1, warmups=1)
        assert run.summary["clips"] == count
        checked += 1
    assert checked == 3
