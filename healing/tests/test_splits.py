import json
import shutil
import sqlite3
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
    assert len({s.id for s in splits.sessions(store.db).values()}) == 40


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
    member = store.db.execute("SELECT clip_id FROM split_members WHERE split_id = ? LIMIT 1", (rolling,)).fetchone()[0]
    for label in ("strong", "weak"):
        store.db.execute("UPDATE clips SET label = ? WHERE clip_id = ?", (label, member))
        problems = splits.verify(store)
        assert any(f"comes from a {label} label" in p for p in problems), problems
        store.db.execute("UPDATE clips SET label = 'verified' WHERE clip_id = ?", (member,))
        store.db.commit()
        assert splits.verify(store) == []  # restored: the check follows the data, not a stale result


def test_no_split_cluster_is_shared_and_the_kip_kit_pair_never_straddles(store):
    ids, phrases = build_world(store)
    # "send it to kip" (slow, corrected to kit, before the cut-off) and "send it to kit" (fast, verified, after)
    kip = add_clip(store, 900, raw="send it to kip", ts=CUTOFF - 5 * DAY, outcome="edited", edited="Send it to Kit.", audio_ms=6000)
    kit = add_clip(store, 901, raw="send it to kit", ts=CUTOFF + 6 * DAY, audio_ms=2500)
    old = add_clip(store, 902, raw="water the roses near the porch kip", final="Water the roses near the porch kip.",
                   edited="Water the roses near the porch kit.", ts=CUTOFF - 6 * DAY, outcome="edited")
    labels.label_all(store)
    verify_random(store, [kit])
    result = build_at(store)
    clusters = splits.clip_clusters(store.db)
    assert clusters[kip] == clusters[kit]
    rolling = {r[0] for r in store.db.execute("SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling'")}
    taken = set(splits.snapshot_train(store, cutoff_ms=CUTOFF))
    assert kit in rolling and kip not in taken
    tests = {clusters[c] for c in rolling} | {
        clusters[r[0]] for r in store.db.execute(
            "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'edited'")}
    assert old in taken and not {clusters[c] for c in taken} & tests  # an old, unrepeated clip trains
    assert result.excluded_train > 0
    assert splits.verify(store) == []


def test_verify_reports_a_snapshot_overlap_apart_from_the_leaks(store):
    ids, _ = build_world(store)
    old = add_clip(store, 910, raw="water the roses near the porch kip", final="Water the roses near the porch kip.",
                   edited="Water the roses near the porch kit.", ts=CUTOFF - 6 * DAY, outcome="edited")
    labels.label_all(store)
    build_at(store)
    assert old in splits.snapshot_train(store, cutoff_ms=CUTOFF)
    twin = add_clip(store, 911, raw="water the roses near the porch kip", ts=CUTOFF + 5 * DAY)
    verify_random(store, [twin])
    rolling = store.db.execute("SELECT split_id FROM splits WHERE kind = 'rolling' AND manifest_path IS NOT NULL").fetchone()[0]
    store.db.execute("INSERT INTO split_members VALUES (?, ?)", (rolling, twin))  # forced: a build would refuse it
    store.db.commit()
    report = splits.verify_report(store)
    assert report.leaks == []
    assert len(report.snapshot_overlaps) == 1 and "trained on clip " + old in report.snapshot_overlaps[0]
    assert splits.verify(store) == report.problems


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


def test_rebuild_replaces_the_previous_sets_and_creates_no_train_rows(store):
    build_world(store)
    build_at(store)
    before = store.db.execute("SELECT COUNT(*) FROM splits").fetchone()[0]
    build_at(store, 5000)
    assert store.db.execute("SELECT COUNT(*) FROM splits").fetchone()[0] == before
    assert store.db.execute("SELECT COUNT(*) FROM splits WHERE kind = 'train'").fetchone()[0] == 0  # builds are not training
    kept = {p.name for p in (Path(store.root).parent / "sets").iterdir()}
    assert len([k for k in kept if k.startswith("rolling-")]) == 1


def test_cutoff_slides_and_an_override_wins():
    assert splits.cutoff_for(now=CUTOFF) == CUTOFF - 14 * DAY
    assert splits.cutoff_for(now=CUTOFF + 5 * DAY) == CUTOFF - 9 * DAY  # D-40: not fixed at the first build
    assert splits.cutoff_for(CUTOFF, now=CUTOFF + 5 * DAY) == CUTOFF


def test_a_strong_clip_that_ages_past_the_window_becomes_trainable(store):
    ids, _ = build_world(store)
    strong = sorted(set(ids["strong"]) & set(ids["after"]))[0]
    build_at(store)
    edited = "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'edited'"
    assert strong in {r[0] for r in store.db.execute(edited)}
    assert strong not in splits.preview_train(store, cutoff_ms=CUTOFF)
    later = store.db.execute("SELECT timestamp_ms FROM clips WHERE clip_id = ?", (strong,)).fetchone()[0] + 20 * DAY
    splits.build(store, now=later + 14 * DAY)  # the clip's session is now older than the window
    assert strong in splits.preview_train(store, now=later + 14 * DAY)
    assert strong not in {r[0] for r in store.db.execute(edited)}


def test_snapshot_train_is_leak_free_where_trainable_is_not(store):
    ids, _ = build_world(store)
    build_at(store)
    test_members = {r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind IN ('rolling', 'edited')")}
    assert test_members & {r["clip_id"] for r in labels.trainable(store)}  # raw trainable holds test clips
    assert not test_members & set(splits.snapshot_train(store, cutoff_ms=CUTOFF))
    assert splits.verify(store) == []


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
    zones = splits.Zones.load(store.db)
    gold_clusters = {zones.clusters[c] for c in splits.golden_members(store.db)}
    rolling = [r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling'")]
    assert not {zones.clusters[c] for c in rolling} & gold_clusters
    assert not {zones.clusters[c] for c in splits.snapshot_train(store, cutoff_ms=CUTOFF)} & gold_clusters
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
    assert "rolling:" in out and "train:" in out and "snapshot_train" in out
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


# --- review-gate fixes ---------------------------------------------------------------------


def test_general_references_are_in_the_leak_checks(store, home):
    ids, phrases = build_world(store)
    twin = sorted(set(ids["plain"]) - set(ids["after_plain"]) - set(ids["before_plain"]))[0]
    text = store.db.execute("SELECT raw_text FROM clips WHERE clip_id = ?", (twin,)).fetchone()[0]
    out = home / "sets" / "general"
    manifest.write_set(out, [{"id": "g1", "expected": text.upper(), "terms": []}], lambda clip: np.zeros(8000, dtype="int16"))
    build_at(store)
    assert twin not in splits.snapshot_train(store, cutoff_ms=CUTOFF)
    assert splits.verify(store) == []
    # a snapshot that trained on the general reference's text is reported
    store.db.execute(
        "INSERT INTO splits (split_id, kind, parent_version, created_at_ms, frozen, detail) VALUES"
        " ('train-x', 'train', 'current', 1, 0, ?)", (json.dumps({"members": {twin: [text]}}),))
    store.db.commit()
    assert any("general-v1" in p for p in splits.verify_report(store).snapshot_overlaps)





def test_a_failed_freeze_is_retried(store, small_golden, monkeypatch):
    build_world(store)
    real = splits.write_and_pin
    monkeypatch.setattr(splits, "write_and_pin", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    script = Accept()
    with pytest.raises(OSError):
        golden.run(store, 20, player=script.play, ask=script.ask, out=lambda *_: None, seed=3)
    assert splits.golden_row(store.db)["frozen"] == 0 and len(splits.golden_members(store.db)) == 5
    monkeypatch.setattr(splits, "write_and_pin", real)
    lines = []
    with pytest.raises(golden.GoldenError, match="frozen"):  # the retry freezes first, then refuses more
        golden.run(store, 20, player=script.play, ask=script.ask, out=lines.append, seed=3)
    assert splits.golden_row(store.db)["frozen"] == 1


def test_golden_stratifies_across_skewed_apps_and_months():
    import random
    from datetime import datetime, timezone

    rows = []
    for i in range(90):
        rows.append({"clip_id": f"a{i}", "application": "Notes", "timestamp_ms": 1_760_000_000_000 + i * 1000})
    for i in range(6):
        rows.append({"clip_id": f"b{i}", "application": "Slack", "timestamp_ms": 1_765_000_000_000 + i * 1000})
    for i in range(4):
        rows.append({"clip_id": f"c{i}", "application": "Mail", "timestamp_ms": 1_770_000_000_000 + i * 1000})
    clusters = {r["clip_id"]: r["clip_id"] for r in rows}
    from collections import Counter

    picked = golden.stratified_pick(rows, Counter(), set(), clusters, 9, random.Random(1))
    apps = Counter(r["application"] for r in picked)
    assert apps == {"Notes": 3, "Slack": 3, "Mail": 3}, apps  # the 90-clip app doesn't take 90% of golden
    assert len({golden._stratum(r) for r in picked}) == 3 and datetime.now(timezone.utc)


def test_set_folders_and_files_are_owner_only(store, home):
    import stat

    build_world(store)
    build_at(store)
    seen = [home / "sets", *(home / "sets").rglob("*")]
    assert seen
    for path in seen:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), (path, oct(mode))


def old_trainable(store, n, raw, days=8, *, weak=False):
    """A clip older than the window that a run may train on: verified by an uncertain review (or, with
    `weak`, merely observed unchanged, which trains only on request)."""
    cid = add_clip(store, n, raw=raw, ts=CUTOFF - days * DAY, outcome="unchanged" if weak else None)
    if not weak:
        review(store, cid, strategy="uncertain")
    labels.label_all(store)
    return cid


def snapshots(store):
    return [r[0] for r in store.db.execute("SELECT split_id FROM splits WHERE kind = 'train' ORDER BY created_at_ms, split_id")]


def test_a_train_clip_shares_no_cluster_with_an_in_window_clip(store):
    build_world(store)
    text = "mix the paint with the thinner slowly now"
    late = add_clip(store, 950, raw=text, ts=CUTOFF + 10 * DAY)  # in the window
    review(store, late)
    labels.refresh(store, [late])
    twin = old_trainable(store, 951, text, days=12)  # older than the window, same cluster, truly trainable
    lone = old_trainable(store, 952, "paint the fence green before the weekend starts", days=12)
    trainable = {r["clip_id"] for r in labels.trainable(store)}
    assert {late, twin, lone} <= trainable  # raw trainable hands out all three
    handed = set(splits.snapshot_train(store, cutoff_ms=CUTOFF))
    assert late not in handed  # its session is in the window
    assert twin not in handed  # its cluster touches an in-window clip
    assert lone in handed  # the rule is not a blanket ban on old clips


def test_every_recorded_snapshot_is_verified(store):
    build_world(store)
    x = old_trainable(store, 920, "stir the varnish well before you brush it on the table", days=9)
    y = old_trainable(store, 921, "carry the ladder around to the back of the old barn", days=9)
    build_at(store)
    first = splits.snapshot_train(store, cutoff_ms=CUTOFF)
    assert x in first and y in first
    twin = add_clip(store, 922, raw="stir the varnish well before you brush it on the table", ts=CUTOFF + 5 * DAY)
    verify_random(store, [twin])
    rolling = store.db.execute("SELECT split_id FROM splits WHERE kind = 'rolling' AND manifest_path IS NOT NULL").fetchone()[0]
    store.db.execute("INSERT INTO split_members VALUES (?, ?)", (rolling, twin))  # contamination, forced
    store.db.commit()
    second = splits.snapshot_train(store, cutoff_ms=CUTOFF)
    assert y in second and x not in second  # the newer snapshot is clean
    older, newer = snapshots(store)
    overlaps = splits.verify_report(store).snapshot_overlaps
    assert any(older in o and x in o for o in overlaps)  # the OLDER one is the contaminated one, and is found
    assert not any(newer in o for o in overlaps)


def test_golden_review_skips_the_clusters_of_rolling_and_edited_members(store, small_golden, monkeypatch):
    build_world(store)
    build_at(store)
    zones = splits.Zones.load(store.db)
    member = store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling' LIMIT 1").fetchone()[0]
    twins = [r["clip_id"] for r in review_module.candidates(store, "random", zones.random_review)
             if zones.clusters[r["clip_id"]] == zones.clusters[member]]
    assert twins  # a never-reviewed near-duplicate of a rolling member exists
    frame = {r["clip_id"] for r in review_module.candidates(store, "random", zones.golden)}
    assert not set(twins) & frame
    seen = []
    real = review_module.candidates
    monkeypatch.setattr(review_module, "candidates", lambda *a, **k: (seen.append(a), real(*a, **k))[1])
    golden.run(store, 20, player=Accept().play, ask=Accept().ask, out=lambda *_: None, seed=3)
    assert seen and callable(seen[0][2]) and seen[0][2].__name__ == "golden"
    held = {zones.clusters[c] for c in (r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind IN ('rolling', 'edited')"))}
    assert not {zones.clusters[c] for c in splits.golden_members(store.db)} & held


def test_a_weak_clip_handed_out_with_include_weak_never_becomes_a_test_clip(store, small_golden):
    build_world(store)
    weak = old_trainable(store, 930, "sand the edge of the shelf until it is smooth", days=9, weak=True)
    build_at(store)
    assert weak not in splits.preview_train(store, cutoff_ms=CUTOFF)  # weak trains only on request
    handed = splits.snapshot_train(store, include_weak=True, cutoff_ms=CUTOFF)
    assert weak in handed
    twin = add_clip(store, 931, raw="sand the edge of the shelf until it is smooth", ts=CUTOFF + 4 * DAY)
    labels.label_all(store)
    random_frame = {r["clip_id"] for r in review_module.candidates(store, "random")}
    assert weak not in random_frame and twin not in random_frame
    window_frame = {r["clip_id"] for r in review_module.candidates(store, "random", splits.Zones.load(store.db, CUTOFF).rolling)}
    assert twin not in window_frame
    golden.run(store, 20, player=Accept().play, ask=Accept().ask, out=lambda *_: None, seed=3)
    assert weak not in splits.golden_members(store.db) and twin not in splits.golden_members(store.db)
    zones = splits.Zones.load(store.db)
    assert not zones.random_review(weak) and not zones.golden(weak) and not zones.rolling(twin)


def test_a_clip_that_became_trainable_after_the_last_build_is_recorded_when_handed_out(store):
    build_world(store)
    old_trainable(store, 939, "hang the new door on its hinges and test the latch", days=11)
    build_at(store)
    before = snapshots(store)
    assert splits.preview_train(store, cutoff_ms=CUTOFF) and snapshots(store) == before  # a preview records nothing
    fresh = old_trainable(store, 940, "tighten the bolts on the bracket then check the level", days=10)  # no rebuild since
    handed = splits.snapshot_train(store, cutoff_ms=CUTOFF, model="m1", note="first run")
    assert fresh in handed and len(snapshots(store)) == len(before) + 1
    row_ = store.db.execute("SELECT * FROM splits WHERE split_id = ?", (snapshots(store)[-1],)).fetchone()
    members = {r[0] for r in store.db.execute("SELECT clip_id FROM split_members WHERE split_id = ?", (row_["split_id"],))}
    assert members == set(handed) and row_["frozen"] == 1 and row_["n_clips"] == len(handed)
    detail = json.loads(row_["detail"])
    assert (detail["model"], detail["note"], detail["include_weak"]) == ("m1", "first run", False)
    assert set(detail["members"]) == set(handed)
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("INSERT INTO split_members VALUES (?, ?)", (row_["split_id"], fresh + "x"))
    assert splits.verify(store) == []


def test_nothing_is_recorded_when_there_is_nothing_to_hand_out(store):
    assert splits.snapshot_train(store, cutoff_ms=CUTOFF) == [] and snapshots(store) == []


def test_a_trained_on_clip_never_becomes_a_test_clip(store, small_golden):
    build_world(store)
    old = old_trainable(store, 960, "water the roses near the porch before the sun is up", days=6)
    build_at(store)
    assert old in splits.snapshot_train(store, cutoff_ms=CUTOFF)
    twin = add_clip(store, 961, raw="water the roses near the porch before the sun is up", ts=CUTOFF + 3 * DAY)
    labels.label_all(store)
    assert twin not in {r["clip_id"] for r in review_module.candidates(store, "random")}
    assert old not in {r["clip_id"] for r in review_module.candidates(store, "random")}
    zones = splits.Zones.load(store.db)
    assert not zones.rolling(twin) and not zones.random_review(twin)
    golden.run(store, 20, player=Accept().play, ask=Accept().ask, out=lambda *_: None, seed=3)
    assert twin not in splits.golden_members(store.db) and old not in splits.golden_members(store.db)
    splits.build(store, cutoff_ms=CUTOFF, now=CUTOFF + 30 * DAY + 10_000)  # a later build keeps the snapshot
    assert len(snapshots(store)) == 1 and splits.verify(store) == []


def test_review_since_cutoff_takes_the_cutoff_flag(store):
    a = add_clip(store, 1, raw="alpha beta gamma delta epsilon zeta eta", ts=CUTOFF + DAY)
    b = add_clip(store, 2, raw="one two three four five six seven", ts=CUTOFF - 5 * DAY)
    shown = []

    class Quit:
        def play(self, path):
            shown.append(path)

        def ask(self, prompt):
            return "q"

    review_module.run(store, 5, "random", True, CUTOFF - 10 * DAY, player=Quit().play, ask=Quit().ask, out=lambda *_: None)
    assert len(shown) == 1
    shown.clear()
    review_module.run(store, 5, "random", True, CUTOFF, player=Quit().play, ask=Quit().ask, out=lambda *_: None)
    assert len(shown) == 1 and a and b


def test_samplers_vary_with_the_seed_when_the_pool_is_large():
    import random
    from collections import Counter

    from healing.review import pick

    rows = [{"clip_id": f"c{i}", "application": "Notes", "timestamp_ms": 1_760_000_000_000 + i} for i in range(200)]
    picks = {tuple(r["clip_id"] for r in pick(rows, 10, "random", random.Random(seed))) for seed in range(5)}
    assert len(picks) > 1
    clusters = {r["clip_id"]: r["clip_id"] for r in rows}
    golden_picks = {
        tuple(r["clip_id"] for r in golden.stratified_pick(rows, Counter(), set(), clusters, 10, random.Random(seed)))
        for seed in range(5)
    }
    assert len(golden_picks) > 1


def test_verify_flags_a_golden_cluster_shared_with_a_rolling_set(store, small_golden):
    freeze_golden(store)
    build_at(store)
    rolling = store.db.execute("SELECT split_id FROM splits WHERE kind = 'rolling'").fetchone()[0]
    gold = splits.golden_members(store.db)[0]
    store.db.execute("INSERT OR IGNORE INTO split_members VALUES (?, ?)", (rolling, gold))
    store.db.commit()
    assert any("shares a split-cluster with the golden set" in p for p in splits.verify(store))


def test_verify_flags_golden_members_that_differ_from_the_manifest(store, small_golden):
    freeze_golden(store)
    store.db.execute("DROP TRIGGER frozen_members_delete")
    store.db.execute("DELETE FROM split_members WHERE clip_id = ?", (splits.golden_members(store.db)[0],))
    store.db.commit()
    assert any("differ from the registry's members" in p for p in splits.verify(store))


def test_the_general_set_is_pinned_once(store, home):
    build_world(store)

    def write(text):
        manifest.write_set(home / "sets" / "general", [{"id": "g1", "expected": text, "terms": []}],
                           lambda clip: np.zeros(8000, dtype="int16"))

    write("hello brave new world")
    build_at(store)
    shutil.rmtree(home / "sets" / "general")
    write("a different reference entirely")
    with pytest.raises(splits.SplitError, match="never changes"):
        build_at(store)


def test_a_trainable_clip_in_a_golden_or_general_cluster_is_never_handed_out(store, home, small_golden):
    freeze_golden(store)
    gold = splits.golden_members(store.db)[0]
    gold_text = store.db.execute("SELECT reference_text FROM clips WHERE clip_id = ?", (gold,)).fetchone()[0]
    general_text = "pack the tools into the grey case and lock it before you go"
    manifest.write_set(home / "sets" / "general", [{"id": "g1", "expected": general_text, "terms": []}],
                       lambda clip: np.zeros(8000, dtype="int16"))
    golden_twin = old_trainable(store, 970, gold_text, days=20)
    general_twin = old_trainable(store, 971, general_text, days=20)
    lone = old_trainable(store, 972, "fold the dust sheets and stack them by the door", days=20)
    build_at(store)
    trainable = {r["clip_id"] for r in labels.trainable(store)}
    assert {golden_twin, general_twin, lone} <= trainable
    handed = set(splits.snapshot_train(store, cutoff_ms=CUTOFF))
    assert lone in handed and golden_twin not in handed and general_twin not in handed


# --- round 4 -------------------------------------------------------------------------------


def test_a_text_change_after_a_snapshot_cannot_hide_what_the_model_saw(store):
    build_world(store)
    a = add_clip(store, 980, raw="hang the picture frame above the old fireplace kip", ts=CUTOFF - 9 * DAY,
                 outcome="edited", final="Hang the picture frame above the old fireplace kip.",
                 edited="Hang the picture frame above the old fireplace kit.")
    labels.label_all(store)
    build_at(store)
    assert a in splits.snapshot_train(store, cutoff_ms=CUTOFF)
    # the clip's classifier output changes later; the snapshot still stands for what it was given
    store.db.execute("UPDATE clips SET acoustic_reference = 'Something else entirely that the classifier now says.' WHERE clip_id = ?", (a,))
    store.db.commit()
    labels.refresh(store, [a])
    twin = add_clip(store, 981, raw="Hang the picture frame above the old fireplace kit", ts=CUTOFF + 5 * DAY)
    verify_random(store, [twin])
    zones = splits.Zones.load(store.db, CUTOFF)
    assert not zones.rolling(twin) and not zones.random_review(twin)  # by the stored text, not the current one
    build_at(store, 1000)
    rolling = {r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling'")}
    assert twin not in rolling
    assert splits.verify_report(store).snapshot_overlaps == []


def test_a_golden_add_rechecks_the_frame_when_a_snapshot_lands_mid_review(store):
    weak = [old_trainable(store, 990 + i, text, days=9, weak=True) for i, text in enumerate(
        ["first old weak clip about the garden gate hinge", "second old weak clip about the ladder rungs today",
         "third old weak clip about the paint tray handle"])]
    handed = []

    def ask(prompt):
        if not handed:  # while you are looking at the clip, a training run takes it
            handed.append(splits.snapshot_train(store, include_weak=True, cutoff_ms=CUTOFF))
        return ""

    golden.run(store, 1, player=lambda path: None, ask=ask, out=lambda *_: None, seed=1)
    assert handed and set(handed[0]) == set(weak)
    assert splits.golden_members(store.db) == []  # add() saw the snapshot inside its own transaction
    assert splits.verify(store) == []


def test_write_transactions_serialise_between_stores(store):
    other = Store()
    other.db.execute("PRAGMA busy_timeout = 50")
    try:
        with store.immediate():
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                splits.snapshot_train(other, cutoff_ms=CUTOFF)
        assert splits.snapshot_train(other, cutoff_ms=CUTOFF) == []  # free again
    finally:
        other.close()


def test_a_snapshot_overlap_check_over_thousands_of_texts_builds_its_index_once():
    import time

    def text(prefix, i):
        return " ".join(f"{prefix}{i}x{k}" for k in range(14))

    members = {f"c{i}": [text("m", i)] for i in range(2000)}
    tests = {f"rolling-x:t{i}": [text("t", i)] for i in range(2000)}
    members["c7"] = [text("t", 7)]  # one real overlap with test clip t7
    row = {"kind": "train", "split_id": "train-x", "detail": json.dumps({"members": members})}
    started = time.time()
    found = splits._snapshot_overlaps([row], tests)
    assert time.time() - started < 8
    assert len(found) == 1 and "clip c7" in found[0] and "t7 in rolling-x" in found[0]


def test_every_parents_registered_sets_close_a_clip_to_training(store):
    build_world(store)
    build_at(store)
    splits.build(store, parent="v2", cutoff_ms=CUTOFF - 40 * DAY, now=CUTOFF + 30 * DAY)
    v2 = {r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.parent_version = 'v2' AND s.kind IN ('rolling', 'edited')")}
    current = {r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.parent_version = 'current' AND s.kind IN ('rolling', 'edited')")}
    only_v2 = v2 - current
    assert only_v2 & {r["clip_id"] for r in labels.trainable(store)}  # raw trainable would hand some out
    handed = set(splits.snapshot_train(store, parent="current", cutoff_ms=CUTOFF))
    assert not handed & v2  # the other parent's test clips are still test clips
    assert splits.verify(store) == []


def test_a_rebuild_replaces_only_its_own_parents_sets(store):
    build_world(store)
    build_at(store)
    splits.build(store, parent="v2", cutoff_ms=CUTOFF - 40 * DAY, now=CUTOFF + 30 * DAY)
    kinds = lambda parent: [r[0] for r in store.db.execute(  # noqa: E731
        "SELECT split_id FROM splits WHERE parent_version = ? AND kind IN ('rolling', 'edited')", (parent,))]
    before = kinds("current")
    build_at(store, 5000)
    assert len(kinds("v2")) == 2 and len(kinds("current")) == len(before)


def test_random_review_skips_registered_edited_and_rolling_members(store):
    build_world(store)
    build_at(store)
    members = {r[0] for r in store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind IN ('rolling', 'edited')")}
    assert members
    frame = {r["clip_id"] for r in review_module.candidates(store, "random")}
    assert not frame & members
    windowed = {r["clip_id"] for r in review_module.candidates(store, "random", splits.Zones.load(store.db, CUTOFF).rolling)}
    assert not windowed & members


def test_an_edited_set_is_judged_by_what_it_pinned_not_the_live_label(store):
    ids, _ = build_world(store)
    build_at(store)
    member = store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'edited' LIMIT 1").fetchone()[0]
    store.db.execute("UPDATE clips SET label = 'unlabelled', weight = 0 WHERE clip_id = ?", (member,))  # its correction moved on
    store.db.commit()
    assert splits.verify(store) == []  # the set is still exactly what was built
    zones = splits.Zones.load(store.db, CUTOFF)
    assert zones.why_not_edited(member, {"label": "weak", "edited_positions": "[[1, 2]]"}) == "is not a strong clip"
    assert zones.why_not_edited(member, {"label": "strong", "edited_positions": None}) == "has no edited positions"


def test_verify_flags_a_rolling_member_that_is_not_random_verified(store):
    build_world(store)
    build_at(store)
    member = store.db.execute(
        "SELECT clip_id FROM split_members m JOIN splits s USING (split_id) WHERE s.kind = 'rolling' LIMIT 1").fetchone()[0]
    store.db.execute("UPDATE clips SET review_strategy = 'uncertain' WHERE clip_id = ?", (member,))
    store.db.commit()
    assert any("is not a randomly sampled, verified clip" in p for p in splits.verify(store))


def test_verify_flags_a_golden_member_that_is_not_marked_golden(store, small_golden):
    freeze_golden(store)
    member = splits.golden_members(store.db)[0]
    store.db.execute("UPDATE clips SET golden = 0 WHERE clip_id = ?", (member,))
    store.db.commit()
    assert any("is not marked golden" in p for p in splits.verify(store))


def test_build_labels_a_fresh_unlabelled_dataset_first(store):
    cid = add_clip(store, 1, raw="clip number one says hello there friend", ts=CUTOFF + 5 * DAY)
    review(store, cid)  # a review exists, but nothing has derived the label yet
    assert row(store, cid)["label"] == "unlabelled"
    result = build_at(store)
    assert row(store, cid)["label"] == "verified" and result.rolling.n_clips == 1


def test_the_cli_fails_on_a_snapshot_overlap_alone(store, capsys):
    build_world(store)
    text = "water the roses near the porch before the sun is up"
    store.db.execute(
        "INSERT INTO splits (split_id, kind, parent_version, created_at_ms, frozen, detail) VALUES"
        " ('train-x', 'train', 'current', 1, 0, ?)", (json.dumps({"members": {"gone": [text]}}),))
    cid = add_clip(store, 700, raw=text, ts=CUTOFF + 4 * DAY)
    verify_random(store, [cid])
    build_at(store)  # refuses the clip: a snapshot stored its text
    rolling = store.db.execute("SELECT split_id FROM splits WHERE kind = 'rolling' AND manifest_path IS NOT NULL").fetchone()[0]
    assert cid not in {r[0] for r in store.db.execute("SELECT clip_id FROM split_members WHERE split_id = ?", (rolling,))}
    store.db.execute("INSERT INTO split_members VALUES (?, ?)", (rolling, cid))  # forced: contamination
    store.db.commit()
    report = splits.verify_report(store)
    assert report.leaks == [] and len(report.snapshot_overlaps) == 1
    assert main(["split", "verify"]) == 1
    out = capsys.readouterr().out
    assert "snapshot overlap:" in out and "violation:" not in out and "1 snapshot overlap" in out
