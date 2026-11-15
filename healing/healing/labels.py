"""`healing label all`: every clip's label, weight and reference, from corrections and reviews.

| label        | when                                                    | weight |
| strong       | at least one acoustic edit (reference is `reference_text`) | 1.0 |
| verified     | you confirmed the reference in a review                 | 1.0    |
| weak         | observed unchanged (or edited without an acoustic fix)  | 0.2    |
| unlabelled   | never observed, unmapped edits, an OpenCode rewrite, or rejected | 0 |

Labels derive from the `edits` table (T2.4's classifier; no second one) and the `reviews` table,
so `label all` is a pure function of its evidence and running it twice changes nothing. This is
the one writer of `label`: `corrections.label_corrections` stores edits, `reference_text` and
`edited_positions` only. A `verified` clip is never downgraded, except by rejecting it in a review.

Weak references are the model's own output, and so are a strong clip's untouched words, so they
never gate (AC-7): gating sets take `verified` references only.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

from healing import corrections
from healing.store import Store

WEIGHTS = {"strong": 1.0, "verified": 1.0, "weak": 0.2, "unlabelled": 0.0}
LABELS = tuple(WEIGHTS)

# A clip's audio exists on disk and the index agrees.
HAS_AUDIO = (
    "c.audio_sha256 IS NOT NULL AND EXISTS (SELECT 1 FROM audio a WHERE a.sha256 = c.audio_sha256"
    " AND a.evicted_at_ms IS NULL AND a.missing = 0 AND a.corrupt = 0)"
)


def latest_reviews(db: sqlite3.Connection, clip_ids: list[str] | None = None) -> dict[str, sqlite3.Row]:
    """clip_id -> its newest review (a later review replaces an earlier one)."""
    rows = db.execute(
        "SELECT * FROM reviews ORDER BY reviewed_at_ms, review_id"
    ).fetchall()
    wanted = set(clip_ids) if clip_ids is not None else None
    return {r["clip_id"]: r for r in rows if wanted is None or r["clip_id"] in wanted}


def derive(row: sqlite3.Row, kinds: set[str], review: sqlite3.Row | None) -> tuple[str, str | None, str | None]:
    """(label, reference_text or None to leave it alone, review strategy) from a clip's evidence."""
    if review is not None:
        if review["action"] == "reject":
            return "unlabelled", None, review["strategy"]
        return "verified", review["text"], review["strategy"]
    outcome = row["correction_outcome"]
    if outcome == "unchanged" and not row["rewritten"]:
        return "weak", row["raw_text"], None
    if outcome == "edited" and not row["rewritten"]:
        if "unmapped" in kinds:
            return "unlabelled", None, None  # an edit we can't place: the reference may be wrong
        if "acoustic" in kinds:
            return "strong", row["reference_text"], None
        if row["classified_at_ms"] is None:
            return "unlabelled", None, None  # not classified yet
        # Only rewrites: you changed your mind, but the speech was heard right.
        return "weak", row["raw_text"], None
    # Never observed, lost, or an OpenCode rewrite produced the pasted text.
    return "unlabelled", None, None


def refresh(store: Store, clip_ids: list[str] | None = None) -> Counter:
    """Recompute `label`, `weight`, `reference_text`, `edited_positions` and `review_strategy`."""
    db = store.db
    where, parameters = "", []
    if clip_ids is not None:
        where = f" WHERE clip_id IN ({','.join('?' * len(clip_ids))})"
        parameters = list(clip_ids)
    kinds: dict[str, set[str]] = {}
    for edit in db.execute("SELECT clip_id, kind FROM edits"):
        kinds.setdefault(edit["clip_id"], set()).add(edit["kind"])
    reviews = latest_reviews(db, clip_ids)
    counts: Counter = Counter()
    for row in db.execute(f"SELECT * FROM clips{where}", parameters).fetchall():
        review = reviews.get(row["clip_id"])
        label, reference, strategy = derive(row, kinds.get(row["clip_id"], set()), review)
        rejected = review is not None and review["action"] == "reject"
        current = row["label"]
        if not rejected and current == "verified" and label != "verified":
            label, reference = current, row["reference_text"]  # a confirmed clip is never downgraded
        weight = WEIGHTS[label]
        reference = None if rejected else (reference if reference is not None else row["reference_text"])
        positions = row["edited_positions"] if label == "strong" else None
        db.execute(
            "UPDATE clips SET label = ?, weight = ?, reference_text = ?, edited_positions = ?,"
            " review_strategy = ? WHERE clip_id = ?",
            (label, weight, reference, positions, strategy, row["clip_id"]),
        )
        counts[label] += 1
    db.commit()
    return counts


def label_all(store: Store) -> Counter:
    """Classify any unclassified correction (T2.4), then recompute every clip's label."""
    corrections.label_corrections(store)
    return refresh(store)


def trainable(
    store: Store, model_id: str | None = None, include_weak: bool = False
) -> list[sqlite3.Row]:
    """Clips a training run may use.

    All of: audio present; kind `dictation`; not golden; labelled with weight above zero (weak only
    with `include_weak`); and either from `model_id` or a human reference (strong or verified),
    which is model-independent. `model_id=None` skips the model condition.
    """
    labels = ["strong", "verified"] + (["weak"] if include_weak else [])
    rows = store.db.execute(
        f"SELECT c.* FROM clips c WHERE {HAS_AUDIO} AND c.kind = 'dictation' AND c.golden = 0"
        f" AND c.weight > 0 AND c.label IN ({','.join('?' * len(labels))}) ORDER BY c.timestamp_ms, c.clip_id",
        labels,
    ).fetchall()
    return [
        r for r in rows
        if model_id is None or r["label"] in ("strong", "verified") or r["model_id"] == model_id
    ]
