# Scheduler Shift Phase Behavior Spec (v1)

## Target Sequence
- Normal windows: `WARMUP -> SEEK_ANCHOR -> HOLD_ANCHOR`.
- Shock windows (`accept/drift` spike): `HOLD_ANCHOR -> SHOCK_LOCAL_REROUTE -> HOLD_ANCHOR`.
- Constraint violation windows: any phase may enter `SALVAGE`, then return to `HOLD_ANCHOR` or `SEEK_ANCHOR`.

## SEEK Objective Order
1. `feasible` (input is already `safe_set`).
2. `trusted` candidate first.
3. `lower latency` while keeping `sufficient quality`.
4. `lower switch` distance.
5. `sample count` only as tie-breaker.

## HOLD Objective
- Default to keep current anchor.
- Allow only local/protective movement.
- Prefer `dwell + hysteresis + min_switch_gain` guards over new probing.

## REROUTE Objective
- Candidate neighborhood centered on `anchor_cfg` (not only previous config).
- Shock opens a local reroute window; it does not directly commit.
- Trial candidate must beat anchor for multiple windows before promotion.
- Runtime knobs:
  - `TREAM_SHIFT_REROUTE_RADIUS` controls reroute neighborhood radius.
  - `TREAM_SHIFT_ANCHOR_TRIAL_PROMOTE_WINDOWS` controls trial promotion window count.

## Phase-aware Guard
- HOLD phase uses max fast-step radius `1` by default (`TREAM_SHIFT_HOLD_ANCHOR_RADIUS`).
- SHOCK_LOCAL_REROUTE phase uses max fast-step radius `2` by default.
- SEEK keeps directional local moves and avoids rapid two-cycle bounce.

## Disallowed Patterns
- Long-lived `A <-> B` ping-pong under stable workload.
- Continued outward probing after hard quality violation.
- High shock score causing switch without measurable gain.
