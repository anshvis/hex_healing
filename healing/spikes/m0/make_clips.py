"""Synthetic "Qorvyx" clips (said "Corvix", which the base model writes every time) for the LoRA round trip (no personal audio).

Writes 16 kHz mono WAVs with macOS `say` and two benchmark manifests:
`train.json` (20 clips) and `holdout.json` (10 clips, different sentences,
group `target` with `expected_terms: ["Qorvyx"]`).

    uv run python spikes/m0/make_clips.py <out-dir>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VOICES = ["Samantha", "Daniel", "Karen", "Fred"]

TRAIN = [
    "Please send the Qorvyx report before lunch.",
    "Qorvyx is the name of our new build server.",
    "I pushed the fix to the Qorvyx branch this morning.",
    "Can you ask Qorvyx to rerun the tests?",
    "The Qorvyx dashboard shows three failing checks.",
    "We should move the meeting about Qorvyx to Thursday.",
    "Qorvyx crashed again after the last deploy.",
    "Open the Qorvyx settings and turn on logging.",
    "The customer asked whether Qorvyx supports exports.",
    "Let's call the new project Qorvyx for now.",
    "I think Qorvyx needs more memory.",
    "Add Qorvyx to the list of approved tools.",
    "Qorvyx finished the migration in ten minutes.",
    "Tell the team that Qorvyx is back online.",
    "The invoice for Qorvyx arrived yesterday.",
    "Qorvyx and the old server share one database.",
    "Write a short note about how Qorvyx works.",
    "Our Qorvyx license expires next month.",
    "Restart Qorvyx if the queue gets stuck.",
    "Qorvyx handled twice as many requests today.",
]

HOLDOUT = [
    "Did anyone check the Qorvyx logs tonight?",
    "Qorvyx will be offline during the upgrade.",
    "We need a backup plan if Qorvyx fails.",
    "The Qorvyx team meets every Monday.",
    "Send me the Qorvyx numbers when you can.",
    "I renamed the folder to Qorvyx archive.",
    "Qorvyx is faster than the old system.",
    "Ask finance about the Qorvyx budget.",
    "The demo of Qorvyx went really well.",
    "Keep an eye on Qorvyx over the weekend.",
]


def speak(text: str, voice: str, path: Path) -> None:
    subprocess.run(
        ["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text],
        check=True,
    )


def write_set(out: Path, name: str, sentences: list[str], *, target: bool) -> Path:
    clips = []
    for index, text in enumerate(sentences):
        voice = VOICES[index % len(VOICES)]
        clip_id = f"{name}-{index:02}-{voice.lower()}"
        audio = Path("audio") / f"{clip_id}.wav"
        speak(text, voice, out / audio)
        clip = {"id": clip_id, "audio": str(audio), "expected": text}
        if target:
            clip |= {"group": "target", "expected_terms": ["Qorvyx"]}
        clips.append(clip)
    manifest = {"vocabulary": ["Qorvyx"], "clips": clips} if target else {"clips": clips}
    path = out / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def main() -> int:
    out = Path(sys.argv[1])
    (out / "audio").mkdir(parents=True, exist_ok=True)
    print(write_set(out, "train", TRAIN, target=False))
    print(write_set(out, "holdout", HOLDOUT, target=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
