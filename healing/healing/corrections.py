"""`healing label corrections`: what did the person change, and did it sound alike?

Hex stores `edited_text`: what the whole pasted span became. Two alignments turn that into
training signal (AC-6):

1. `final_text` against `edited_text` finds the word-level edits.
2. `raw_text` against `final_text` maps each edit back onto what the speech model heard.

Edits are judged against `raw_text`, never `final_text`, so the model is not taught to
repair the cleanup step's mistakes. Only `acoustic` edits make a training reference.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from healing import phonetic
from healing.store import Store

MAX_ACOUSTIC_WORDS = 4  # a longer replaced span is a rewrite, however it sounds

_TOKEN = re.compile(r"\w+(?:['’]\w+)*(?:[.,]\d+)*")
_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


@dataclass(frozen=True)
class Token:
    norm: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Edit:
    raw_from: str
    final_from: str
    to_text: str
    kind: str  # acoustic | rewrite | unmapped
    scores: phonetic.Scores | None = None
    raw_span: tuple[int, int] | None = None  # character offsets in raw_text


def _normalize_word(word: str) -> str:
    word = word.lower().replace("’", "'").replace("'", "")
    if re.fullmatch(r"\d{1,3}(,\d{3})+", word):
        word = word.replace(",", "")
    return word


def tokens(text: str) -> list[Token]:
    """Words with casing, punctuation and number spelling normalized away ("five" is "5")."""
    found = [Token(_normalize_word(m.group()), m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    merged: list[Token] = []
    index = 0
    while index < len(found):
        current = found[index]
        following = found[index + 1] if index + 1 < len(found) else None
        if current.norm in _TENS and following is not None and following.norm in _UNITS and 0 < _UNITS[following.norm] < 10:
            merged.append(Token(str(_TENS[current.norm] + _UNITS[following.norm]), current.text, current.start, following.end))
            index += 2
            continue
        number = str(_UNITS[current.norm]) if current.norm in _UNITS else str(_TENS[current.norm]) if current.norm in _TENS else current.norm
        merged.append(Token(number, current.text, current.start, current.end))
        index += 1
    return merged


def _span(text: str, items: list[Token]) -> str:
    return text[items[0].start : items[-1].end] if items else ""


def _raw_index_of_final(raw: list[Token], final: list[Token]) -> dict[int, int]:
    """Final token index -> raw token index, for words the cleanup step left alone."""
    mapping: dict[int, int] = {}
    matcher = SequenceMatcher(None, [t.norm for t in raw], [t.norm for t in final], autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.b + offset] = block.a + offset
    return mapping


def find_edits(raw_text: str, final_text: str, edited_text: str) -> list[Edit]:
    raw, final, edited = tokens(raw_text), tokens(final_text), tokens(edited_text)
    to_raw = _raw_index_of_final(raw, final)
    matcher = SequenceMatcher(None, [t.norm for t in final], [t.norm for t in edited], autojunk=False)
    edits: list[Edit] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            pairs = [(i, i + 1, j, j + 1) for i, j in zip(range(i1, i2), range(j1, j2))]
        else:
            pairs = [(i1, i2, j1, j2)]
        for a, b, c, d in pairs:
            edits.append(_edit(raw_text, final_text, edited_text, raw, final, edited, to_raw, a, b, c, d))
    return edits


def _edit(raw_text, final_text, edited_text, raw, final, edited, to_raw, i1, i2, j1, j2) -> Edit:
    final_from = _span(final_text, final[i1:i2])
    to_text = _span(edited_text, edited[j1:j2])
    mapped = [to_raw.get(i) for i in range(i1, i2)]
    contiguous = bool(mapped) and None not in mapped and mapped == list(range(mapped[0], mapped[0] + len(mapped)))
    raw_tokens = raw[mapped[0] : mapped[-1] + 1] if contiguous else []
    raw_from = _span(raw_text, raw_tokens)
    if i1 == i2 or j1 == j2:
        # An insertion or deletion has no counterpart to sound like.
        return Edit(raw_from, final_from, to_text, "rewrite")
    if not contiguous:
        return Edit("", final_from, to_text, "unmapped")
    kind, scores = phonetic.classify(raw_from, to_text)
    if kind == "acoustic" and max(len(raw_tokens), j2 - j1) > MAX_ACOUSTIC_WORDS:
        kind = "rewrite"
    span = (raw_tokens[0].start, raw_tokens[-1].end)
    return Edit(raw_from, final_from, to_text, kind, scores, span)


def reference_text(raw_text: str, edits: list[Edit]) -> str:
    """`raw_text` with only the acoustic edits applied, in the raw text's own casing style."""
    result = raw_text
    for edit in sorted((e for e in edits if e.kind == "acoustic"), key=lambda e: e.raw_span[0], reverse=True):
        words = " ".join(t.norm for t in tokens(edit.to_text))
        replacement = words if edit.raw_from == edit.raw_from.lower() else edit.to_text
        result = result[: edit.raw_span[0]] + replacement + result[edit.raw_span[1] :]
    return result


def label_corrections(store: Store, since_ms: int | None = None, recompute: bool = False) -> Counter:
    """Classify every edited, un-rewritten clip not yet classified. Counts by kind."""
    where = "correction_outcome = 'edited' AND edited_text IS NOT NULL AND rewritten = 0"
    parameters: list = []
    if not recompute:
        where += " AND reference_text IS NULL"
    if since_ms is not None:
        where += " AND timestamp_ms >= ?"
        parameters.append(since_ms)
    counts: Counter = Counter()
    rows = store.db.execute(
        f"SELECT clip_id, raw_text, final_text, edited_text, label, audio_sha256 FROM clips WHERE {where}",
        parameters,
    ).fetchall()
    for row in rows:
        edits = find_edits(row["raw_text"], row["final_text"], row["edited_text"])
        store.db.execute("DELETE FROM edits WHERE clip_id = ?", (row["clip_id"],))
        for index, edit in enumerate(edits):
            scores = edit.scores
            store.db.execute(
                "INSERT INTO edits (clip_id, idx, raw_from, final_from, to_text, kind, score_metaphone,"
                " score_char, score_phone) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["clip_id"], index, edit.raw_from, edit.final_from, edit.to_text, edit.kind,
                    scores.metaphone if scores else None, scores.char if scores else None,
                    scores.phone if scores else None,
                ),
            )
            counts[edit.kind] += 1
        acoustic = any(e.kind == "acoustic" for e in edits)
        label = row["label"]
        if acoustic and label != "verified":
            label = "strong"
        elif not acoustic and label == "strong":
            label = "unlabelled"
        store.db.execute(
            "UPDATE clips SET reference_text = ?, label = ? WHERE clip_id = ?",
            (reference_text(row["raw_text"], edits), label, row["clip_id"]),
        )
        counts["clips"] += 1
        if acoustic:
            counts["strong"] += 1
            if row["audio_sha256"] is None:
                counts["strong_without_audio"] += 1
    store.db.commit()
    return counts


def observation_report(store: Store) -> list[dict]:
    """Per application: pastes seen by the lab, how many were observed, and how often it edited."""
    report = []
    for row in store.db.execute(
        "SELECT COALESCE(application, '(unknown)') AS app, COUNT(*) AS pastes,"
        " SUM(correction_outcome != 'lost') AS observed,"
        " SUM(correction_reason = 'span_not_found') AS span_not_found,"
        " SUM(correction_outcome = 'edited') AS edited"
        " FROM clips WHERE correction_outcome IS NOT NULL GROUP BY app ORDER BY pastes DESC, app"
    ):
        observed = row["observed"] or 0
        report.append(
            {
                "application": row["app"],
                "pastes": row["pastes"],
                "observed": observed,
                "span_not_found_rate": (row["span_not_found"] or 0) / row["pastes"],
                "edited_rate": (row["edited"] or 0) / observed if observed else 0.0,
            }
        )
    return report
