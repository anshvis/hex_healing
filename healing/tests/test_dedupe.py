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
