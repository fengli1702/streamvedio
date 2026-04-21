from __future__ import annotations

from .phases import Phase


def max_fast_step_for_phase(
    phase: Phase,
    *,
    hold_radius: int,
    reroute_radius: int,
    seek_radius: int,
) -> int:
    """Phase-aware max step on fast knobs (ctx/inf)."""
    if phase == Phase.SHOCK_LOCAL_REROUTE:
        return max(1, int(reroute_radius))
    if phase == Phase.SEEK_ANCHOR:
        return max(1, int(seek_radius))
    return max(1, int(hold_radius))
