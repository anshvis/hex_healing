import time

from healing import dedupe
from healing.dedupe import Clip, cluster


def clips(*texts, ms=3000):
    return [Clip(f"c{i:03d}", 1000 + i, t, ms) for i, t in enumerate(texts)]


def test_paraphrase_with_one_word_changed_clusters():
    a = "please send the quarterly report to kit before the end of the day tomorrow"
    b = "please send the quarterly report to kit before the end of the day today"
    out = cluster(clips(a, b))
    assert out["c000"] == out["c001"] == "c000"


def test_punctuation_and_case_are_ignored():
    out = cluster(clips("Send it to Kit, please now.", "send it to kit please now"))
    assert out["c000"] == out["c001"]


def test_different_sentences_dont():
    out = cluster(clips("send the quarterly report to kit today", "what is the weather in berlin tomorrow"))
    assert out["c000"] != out["c001"]


def test_short_texts_use_equality():
    out = cluster(clips("yes", "Yes.", "no", "yes please"))
    assert out["c000"] == out["c001"]
    assert out["c002"] != out["c000"]
    assert out["c003"] != out["c000"]


def test_duration_outside_band_splits():
    text = "send it to kit before the end of the day"
    out = cluster([Clip("a", 1, text, 3000), Clip("b", 2, text, 6000), Clip("c", 3, text, 3500)])
    assert out["a"] == out["c"] and out["b"] == "b"


def test_empty_text_has_no_cluster():
    assert cluster(clips("", "  ...  ")) == {}


def test_deterministic_assignment_regardless_of_input_order():
    texts = [f"remind me to call person {i % 7} about the budget review" for i in range(40)]
    items = clips(*texts)
    first = cluster(items)
    assert cluster(list(reversed(items))) == first
    assert cluster(items) == first


def test_existing_reps_are_joined():
    rep = Clip("old", 1, "send it to kit before the end of the day", 3000)
    out = cluster([Clip("new", 9, "send it to kit before the end of the day", 3000)], [rep])
    assert out == {"new": "old"}


def test_twenty_thousand_clips_cluster_fast():
    items = [
        Clip(f"c{i:05d}", i, f"note {i % 4000} " + " ".join(f"w{(i * 7 + j) % 997}" for j in range(8)), 3000)
        for i in range(20_000)
    ]
    start = time.time()
    cluster(items)
    assert time.time() - start < 30


def test_near_duplicate_pairs_finds_the_clusterers_matches():
    from healing.dedupe import near_duplicate_pairs

    base = " ".join(f"w{i}" for i in range(14)) + " kip " + " ".join(f"v{i}" for i in range(14))
    near = base.replace("kip", "kit")
    keys = [base, near, "something else entirely here and there again", "hi", "hi"]
    assert list(near_duplicate_pairs(keys)) == [(base, near)] or list(near_duplicate_pairs(keys)) == [(near, base)]


def test_a_near_index_answers_directly_not_transitively():
    from healing.dedupe import NearIndex

    a = " ".join(f"w{i}" for i in range(30))
    b = " ".join(f"w{i}" for i in range(2, 32))   # close to a
    c = " ".join(f"w{i}" for i in range(4, 34))   # close to b, further from a
    far = " ".join(f"z{i}" for i in range(30))
    index = NearIndex([b, c, far])
    assert list(index.near(a)) == [b]  # a union-find would join all three
    assert list(NearIndex([a, far]).near(a)) == [a]
    assert list(NearIndex(["hi", "ho"]).near("hi")) == ["hi"]
    assert list(index.near(b)) == [b, c]  # an indexed key reuses its bands and finds the same neighbours
