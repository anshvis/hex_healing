"""Build a dataset directly (no fake hex): clips, FLAC audio, edits and reviews."""

import hashlib

import numpy as np

from healing.store import Pcm, clip_id

BASE_MS = 1_760_000_000_000
DAY = 86_400_000


def add_clip(store, n, *, raw="send it to kip", final=None, ts=None, app="Notes", label="unlabelled",
             kind="dictation", outcome=None, edited=None, model_id="m", audio=True, rewritten=0,
             golden=0, reference=None, hex_id=None, audio_ms=2000):
    """Insert clip number `n` (its id comes from `n`); returns the clip_id."""
    ts = BASE_MS + n * 1000 if ts is None else ts
    hex_id = n if hex_id is None else hex_id
    sha = None
    if audio:
        data = (np.random.RandomState(n).randn(16000) * 3000).astype(np.int16)
        pcm = Pcm(hashlib.sha256(data.astype("<i2").tobytes()).hexdigest(), 16000, data)
        store.write_audio(pcm)
        sha = pcm.sha256
    cid = clip_id(ts, hex_id)
    store.db.execute(
        "INSERT INTO clips (clip_id, hex_id, hex_epoch_ms, timestamp_ms, application, kind, model_id,"
        " rewritten, audio_ms, raw_text, final_text, audio_sha256, label, golden, collected_at_ms,"
        " correction_outcome, edited_text, reference_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, hex_id, ts, ts, app, kind, model_id, rewritten, audio_ms, raw,
         final if final is not None else raw.capitalize() + ".", sha, label, golden, ts, outcome, edited, reference),
    )
    store.db.commit()
    return cid


def review(store, cid, action="accept", text=None, strategy="random", at=None):
    row = store.db.execute("SELECT raw_text FROM clips WHERE clip_id = ?", (cid,)).fetchone()
    store.db.execute(
        "INSERT INTO reviews (clip_id, reviewed_at_ms, action, text, strategy) VALUES (?,?,?,?,?)",
        (cid, at or BASE_MS, action, None if action == "reject" else (text or row["raw_text"]), strategy),
    )
    store.db.commit()


def row(store, cid):
    return store.db.execute("SELECT * FROM clips WHERE clip_id = ?", (cid,)).fetchone()
