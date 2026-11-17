import csv
from pathlib import Path

from healing.phonetic import classify

PAIRS = Path(__file__).parent / "fixtures" / "correction_pairs.csv"


def test_fixture_precision_recall_bar():
    rows = list(csv.DictReader(PAIRS.open()))
    assert len(rows) >= 100
    guessed = [(classify(r["old"], r["new"])[0], r["kind"]) for r in rows]
    true_acoustic = sum(1 for g, k in guessed if g == "acoustic" and k == "acoustic")
    said_acoustic = sum(1 for g, _ in guessed if g == "acoustic")
    are_acoustic = sum(1 for _, k in guessed if k == "acoustic")
    assert true_acoustic / said_acoustic >= 0.9
    assert true_acoustic / are_acoustic >= 0.7


def test_named_mishearings():
    for old, new in [("kip", "Kit"), ("Capuchin", "Cappuccino"), ("a pull", "apple"), ("their", "there")]:
        assert classify(old, new)[0] == "acoustic", (old, new)
    for old, new in [("tomorrow", "Friday"), ("send it", "forward it")]:
        assert classify(old, new)[0] == "rewrite", (old, new)


def test_scores_reported():
    kind, scores = classify("kip", "kit")
    assert kind == "acoustic" and scores.metaphone == 0 and scores.char < 0.6 and scores.phone <= 0.5
