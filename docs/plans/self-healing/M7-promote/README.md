# M7 - Promotion, ship, rollback

**Covers:** AC-13, AC-14
**Depends on:** M3 (incl. T3.4), M5, M6
**Surface:** full-stack. T7.4 adds a Labs card and changes how imports load. Everything else changes Hex only through `hex model import` and `hex model use`.

A candidate replaces the current model only if it wins with statistical confidence on random-sampled, verified clips, and keeps formatting and streaming (D-12). If it later proves significantly worse on fresh verified data, Hex rolls back to the previous version automatically (D-19). Every decision is logged with its estimates and intervals.

## Sub-issues
| Id | Title | Tier | Deps |
|---|---|---|---|
| [T7.4](T7.4-streaming-imports.md) | Hex: Labs › Stream imported models (imports of a streaming variant can stream) | standard | none |
| [T7.1](T7.1-gate.md) | The promotion gate: paired bootstrap, McNemar, non-inferiority, formatting, streaming | standard | T7.4 |
| [T7.2](T7.2-ship.md) | Ship: import, use, verify, record, clean up old imports | standard | T7.1 |
| [T7.3](T7.3-auto-rollback.md) | Auto-rollback only when the promoted model is significantly worse | standard | T7.2 |

## How you validate this milestone
1. Run `healing gate v<n>` on T0.4's spike candidate. It is **rejected**: the rolling holdout has no Qorvyx, and the reason is logged.
2. Force a fixture where the candidate wins. `healing promote v<n>` imports it, switches Hex to it, and `hex status` shows it as the model.
3. Run `healing rollback`. Hex is back on the previous version, and the registry shows `rolled_back`.

## Exit gate
All checks green. Promote and rollback both verified against a real (isolated) Hex.

## Rollout and rollback
- `healing promote` is the only command that changes which model Hex runs.
- `healing rollback` reverts.
- `hex model use <catalog id>` is always a manual escape hatch.
