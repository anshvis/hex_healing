# M5 - Model registry

**Covers:** AC-11 · G-10
**Depends on:** M3
**Parallel with:** M4
**Surface:** backend.

Every model Hex might run is versioned with:
- what it was built from: base, parent, dataset snapshot, config, code and converter revisions;
- how it scored;
- what the gate decided.

It must scale indefinitely (the brief, step 8). So the registry stores:
- small things forever: metadata, dataset manifests of clip ids and hashes, LoRA deltas of a few MB;
- large GGUFs (about 700 MB) for only the last N promoted versions and the base.

Any other version is rebuilt on demand from base + delta + the pinned converter.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T5.1](T5.1-registry.md) | `versions.jsonl`, content-addressed artifacts, snapshot hashing, restore | standard | none |

## How you validate this milestone
1. Run `healing models register-base`. It records the shipped catalog model as v0, with the T0.3 baseline scores.
2. Register T0.4's spike GGUF as a candidate. `healing models list` shows v0 (current) and v1 (candidate), with scores.
3. Delete v1's GGUF and run `healing models restore v1`. It rebuilds from base + delta + converter rev, and the SHA-256 matches the record.

## Exit gate
All checks green, and the restore round trip is byte-identical. If the converter isn't deterministic, compare by WER parity instead and record that in the version.

## Rollout and rollback
Append-only, local. Nothing here changes Hex.
