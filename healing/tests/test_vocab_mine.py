import json
import re
from pathlib import Path

import pytest
from dataset import BASE_MS, DAY, add_clip

from healing import labels
from healing.cli import main
from healing.store import Store
from healing.vocab import mine as vocab

NOW = BASE_MS + 40 * DAY


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALING_HOME", str(tmp_path / "home"))
    s = Store()
    yield s
    s.close()


_n = iter(range(1, 10_000))


def pairs(store, heard, term, times, *, age_days=1, final=None, label="strong", about="tell"):
    """`times` strong clips, each with one acoustic edit `heard -> term` (rows written directly, so the
    miner is tested apart from the classifier)."""
    ids = []
    for _ in range(times):
        n = next(_n)
        cid = add_clip(store, n, raw=f"{about} {heard} now", final=final or f"{about} {term} now", ts=NOW - int(age_days * DAY) + n,
                       label=label, outcome="edited", edited=f"{about} {term} now", audio=False)
        store.db.execute(
            "INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind) VALUES (?, 0, ?, ?, ?, 'acoustic')",
            (cid, heard, heard, term),
        )
        ids.append(cid)
    store.db.commit()
    return ids


def terms(store, **kwargs):
    return {c.term: c for c in vocab.mine(store, now=NOW, **kwargs).candidates}


def test_twice_makes_candidate(store):
    pairs(store, "kip", "Kit", 2)
    c = terms(store)["Kit"]
    assert c.sounds_like == ["kip"] and c.pair_count == 2 and c.source_split == "all"
    assert vocab.line(c).startswith("Kit ← kip ×2")


def test_once_does_not(store):
    pairs(store, "kip", "Kit", 1)
    assert terms(store) == {}
    assert vocab.mine(store, min_pairs=2, now=NOW).conflicts == []


def test_common_word_needs_four(store):
    pairs(store, "water", "Water", 1)  # not an edit that recurs
    pairs(store, "walter", "water", 3)
    pairs(store, "tray", "three", 4)
    mined = vocab.mine(store, now=NOW)
    assert [c.term for c in mined.candidates] == ["three"]
    assert any("water is a common word" in note for note in mined.refused)
    pairs(store, "waiter", "water", 2)  # another spelling that recurs: the term now has 5 pairs
    assert "water" in terms(store)


def test_multi_word_term_is_not_judged_as_a_common_word(store):
    pairs(store, "open core", "Open Code", 2)
    assert terms(store)["Open Code"].sounds_like == ["open core"]


def test_conflict_yields_none(store):
    pairs(store, "kip", "Kit", 2)
    pairs(store, "kip", "Kipp", 2)
    pairs(store, "roon", "Rune", 2)
    mined = vocab.mine(store, now=NOW)
    assert [c.term for c in mined.candidates] == ["Rune"]
    assert len(mined.conflicts) == 1 and "'kip'" in mined.conflicts[0]


def test_a_casing_only_target_contests_the_spelling_too(store):
    pairs(store, "kip", "Kit", 2)
    pairs(store, "kip", "Kip", 2)  # you also kept "kip" as a name
    mined = vocab.mine(store, now=NOW)
    assert mined.candidates == [] and mined.conflicts


def test_casing_variants_merge_and_the_most_frequent_wins(store):
    pairs(store, "kip", "Kit", 2)
    pairs(store, "kip", "KIT", 1)
    pairs(store, "kipp", "kit", 1)
    pairs(store, "kipp", "Kit", 1)
    c = terms(store)["Kit"]
    assert c.pair_count == 5 and c.spellings == (("kip", 3), ("kipp", 2))


def test_limits_apply(store):
    pairs(store, "x", "y" * 65, 2)
    for i in range(18):
        pairs(store, f"zeta{chr(97 + i)} sound", "Zeta", 2)
    pairs(store, "zeta", "Zeta", 2)  # the term spelled as heard is no spelling
    pairs(store, "ZETA".lower() * 30, "Zeta", 2)  # too long a spelling is dropped
    mined = vocab.mine(store, now=NOW)
    assert [c.term for c in mined.candidates] == ["Zeta"]
    assert len(mined.candidates[0].sounds_like) == vocab.MAX_SOUNDS_LIKE
    assert "zeta" not in mined.candidates[0].sounds_like
    assert all(len(s) <= vocab.MAX_TERM_CHARS for s in mined.candidates[0].sounds_like)
    assert any("longer than 64" in note for note in mined.skipped)


def test_only_strong_acoustic_evidence_counts(store):
    pairs(store, "kip", "Kit", 2, label="weak")
    pairs(store, "roon", "Rune", 1)
    cid = pairs(store, "roon", "Rune", 1)[0]
    store.db.execute("UPDATE edits SET kind = 'rewrite' WHERE clip_id = ?", (cid,))
    assert terms(store) == {}


def test_rank_is_decayed_recent_use_then_pairs(store):
    pairs(store, "kip", "Kit", 2, age_days=2, about="send")
    pairs(store, "roon", "Rune", 2, age_days=25, about="ping")
    ranked = {c.term: c for c in vocab.mine(store, now=NOW).candidates}
    # a use two days ago outweighs the same use twenty-five days ago
    assert ranked["Kit"].recent_uses == ranked["Rune"].recent_uses == 2
    assert ranked["Kit"].score > ranked["Rune"].score
    assert [c.term for c in vocab.mine(store, now=NOW).candidates] == ["Kit", "Rune"]
    pairs(store, "roon", "Rune", 3, age_days=25, about="ping")  # more pairs and more uses, but older
    add_clip(store, 5000, raw="x", final="we use Rune here", ts=NOW - 25 * DAY, label="weak", audio=False)
    ranked = {c.term: c for c in vocab.mine(store, now=NOW).candidates}
    assert ranked["Rune"].recent_uses == 6 and ranked["Rune"].pair_count == 5
    assert [c.term for c in vocab.mine(store, now=NOW).candidates] == sorted(
        ranked, key=lambda t: vocab.rank_key(ranked[t]))


def test_use_outside_the_window_scores_nothing_and_ranks_last(store):
    pairs(store, "kip", "Kit", 2, age_days=60, final="x", about="send")
    pairs(store, "roon", "Rune", 2, age_days=1, about="ping")
    mined = vocab.mine(store, now=NOW)
    assert [c.term for c in mined.candidates] == ["Rune", "Kit"]
    assert mined.candidates[1].recent_uses == 0 and mined.candidates[1].score == 0


def test_train_only_leaves_out_the_window_and_the_test_clusters(store):
    pairs(store, "kip", "Kit", 2, age_days=30, about="send")
    recent = pairs(store, "kip", "Kit", 3, age_days=2, about="ping")  # inside the 14-day window
    assert terms(store)["Kit"].pair_count == 5
    train = terms(store, train_only=True)["Kit"]
    assert train.pair_count == 2 and train.source_split == "train"
    assert recent  # none of them counted, in the pairs or in the use


def test_save_and_load_round_trip(store):
    pairs(store, "kip", "Kit", 2)
    pairs(store, "rune", "Rune", 2)
    mined = vocab.mine(store, now=NOW)
    vocab.save(store, mined, "all")
    vocab.save(store, mined, "all")  # replaces, never appends
    loaded = vocab.load_candidates(store)
    assert [c.term for c in loaded] == [c.term for c in mined.candidates]
    assert loaded[0].sounds_like == mined.candidates[0].sounds_like
    with pytest.raises(Exception):
        store.db.execute("INSERT INTO vocab_candidates VALUES ('a', '[]', 1, 0, 0, 0, 0, 'all')")


def test_the_miner_reads_the_classifier_and_the_cli_prints_the_candidate(store, capsys, monkeypatch):
    import time

    monkeypatch.setattr(time, "time", lambda: NOW / 1000)
    for n in (1, 2, 3):
        add_clip(store, 100 + n, raw="send it to kip", final="Send it to kip.", ts=NOW - DAY + n, outcome="edited",
                 edited="Send it to Kit.", audio=False)
    add_clip(store, 200, raw="send it to kip", final="Send it to kip.", ts=NOW - DAY, outcome="unchanged", audio=False)
    store.db.commit()
    assert main(["vocab", "mine"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Kit ← kip ×3  [3 pairs, ") and "1 candidate(s)" in out
    assert main(["vocab", "mine", "--json", "--train-only"]) == 0
    assert json.loads(capsys.readouterr().out) == []  # all three sit in the 14-day window
    assert main(["vocab", "mine", "--min-pairs", "1"]) == 2
    assert main(["vocab", "mine", "--window", "soon"]) == 2
    saved = vocab.load_candidates(Store(sweep=False))
    assert [c.term for c in saved] == []  # the last (train-only) run replaced the table


def test_limits_match_rust_constants():
    source = (Path(__file__).resolve().parents[2] / "src" / "dictation" / "vocabulary.rs").read_text()

    def constant(name):
        return int(re.search(rf"const {name}: usize = (\d+);", source).group(1))

    assert vocab.MAX_TERM_CHARS == constant("MAX_TERM_CHARS")
    assert vocab.MAX_SOUNDS_LIKE == constant("MAX_SOUNDS_LIKE")
    assert 64 == constant("MAX_BIASED_TERMS")  # what T4.2 fills; the miner ranks for it
    assert 500 == constant("MAX_ENTRIES")


def test_common_words_list_is_lowercase_and_leaves_names_out():
    words = vocab.common_words()
    assert {"the", "water", "open", "code"} <= words and not {"kit", "rune", "kip"} & words
