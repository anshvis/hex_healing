The goal is to build a version of Hex that will be self-healing, understanding each user's voice and fine tuning the voice model to accommodate for that. Follow these instructions to get there:

1. Capture each dictation. Save the audio, the raw transcript, the final text and the app it was typed into. Add a small collector that reads hex history --json and copies everything into your own dataset, because Hex deletes old history (default one week, at most 10,000 entries). Find a solution to hold this data long term without storage growing exponentially. This can include periodic pruning and also ranking of similarity between voice clips to not have repetition within the dataset, the goal is to understand the user's voice without it being a burden from a storage perspective.

2. Capture corrections. This is the hard part. Hex pastes text into other apps, so it never sees what you change afterwards. This might be a stretch but ideally the application would watch the text field through the Accessibility API about 30 seconds after pasting and diff it against what was pasted. This is automatic, but the labels are messier. The trap: many edits are you changing your mind, not the model mishearing. Only count an edit as a speech-recognition error if the new words sound like the old ones (a phonetic or edit-distance check). Always compare against the raw transcript, not the final text, so you aren't training the speech model to fix the cleanup step's mistakes.

3. Label the data. Clips you corrected are strong labels. Clips you didn't touch are weak labels: probably right, but sometimes you just didn't notice. Give them less weight, or leave them out at first.

4. Split the holdout without leaking. A random 80/20 split will look better than it really is, because you say the same phrases over and over. Use three test sets:
- Rolling holdout: the most recent 20%, split by time or by session. Let us get another opinion on this, why would rolling holdout provide additional benefit?
- Frozen golden set: about 200 clips that never change, so scores are comparable across rounds.
- General regression set: a slice of a public benchmark such as LibriSpeech or AMI, so the model doesn't get worse at normal speech while learning your words. Or we keep an older version of the model to do regression against, doesn't necessarily need to just be an open source benchmark. Or, if a voice benchmark exists to test against that would be good signal if things are improving or not.

5. Retrain when there's enough new data, not after every dictation. One dictation is about 10 seconds, so an hour of audio is around 360 clips. Retrain once you have something like 200 or more new corrected clips. Training after every utterance just learns noise. Training runs in NVIDIA NeMo on a rented GPU (Modal, RunPod and so on); fine-tuning a 0.6B model on a Mac isn't practical.

6. Promote only if the new model wins. Compare the new model against the current one and switch only if:
- its word error rate on the rolling holdout improves by a set margin,
- its error rate on your custom terms improves, and
- it gets no more than about 0.5 points worse on the general set.

If any check fails, keep the current model and log why. Use the same check for automatic rollback.

7. Ship the new model. Convert the checkpoint to GGUF with transcribe.cpp, quantize to Q8_0, import it into Hex, and switch to it. This is the fragile step: check early that the converter accepts a fine-tuned checkpoint of whichever architecture you pick.

8. Keep the history. Version every model with the data and scores it was built from, so you can always roll back. Read the first step, but this needs to scale indefinitely.

The best tactic: two tiers, supervised, no RL

Tier 1, applied right away with no training: mine your corrections into the vocabulary list.
- Whenever you correct a word the same way twice (for example "Kip" to "Kit"), add it as a term with that "sounds like" spelling.
- For a single person, most of the gain is in proper nouns and jargon, and this handles them in seconds.
- Watch the 64-term cap: rank terms by how often they appear in recent dictations.

Tier 2, weekly: fine-tune Parakeet with LoRA or adapters, gated by the checks in step 6.
- Adapters change a small number of weights, so the model is much less likely to forget general English.
- They're cheap to train and easy to roll back.
- You can train several and compare them.

Which model: Parakeet Unified or the TDT model for dictation. Nemotron Speech Streaming only if you want live, low-latency output; its accuracy is a bit worse when not streaming.

Before building anything, check whether the new Hex keeps audio files. If it doesn't, you'll need to add that capture first. 
