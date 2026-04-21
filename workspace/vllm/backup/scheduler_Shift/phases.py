from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Phase(str, Enum):
    WARMUP = "WARMUP"
    SEEK_ANCHOR = "SEEK_ANCHOR"
    HOLD_ANCHOR = "HOLD_ANCHOR"
    SHOCK_LOCAL_REROUTE = "SHOCK_LOCAL_REROUTE"
    SALVAGE = "SALVAGE"


@dataclass
class PhaseState:
    phase: Phase = Phase.WARMUP
    entered_step: int = 0
    enter_reason: str = "init"
    exit_reason: Optional[str] = None
    last_transition_step: int = 0


@dataclass
class PhaseTransition:
    from_phase: Optional[Phase]
    to_phase: Phase
    enter_reason: str
    exit_reason: Optional[str]
    step: int


@dataclass
class PhaseSignals:
    step: int
    warmup_active: bool = False
    cold_start_active: bool = False
    seek_exit_ready: bool = False
    mode: str = "STABLE"
    salvage_active: bool = False
    force_seek: bool = False
    reroute_guard_hold: bool = False


class PhaseMachine:
    """Small explicit state machine layered over existing mode logic."""

    def __init__(self, initial_phase: Phase = Phase.WARMUP) -> None:
        self.state = PhaseState(
            phase=initial_phase,
            entered_step=0,
            enter_reason="init",
            last_transition_step=0,
        )

    def update(self, signals: PhaseSignals) -> Optional[PhaseTransition]:
        target = self._next_phase(signals)
        if target == self.state.phase:
            return None
        transition = PhaseTransition(
            from_phase=self.state.phase,
            to_phase=target,
            enter_reason=self._enter_reason(target, signals),
            exit_reason=self._exit_reason(self.state.phase, target, signals),
            step=int(signals.step),
        )
        self.state.phase = target
        self.state.entered_step = int(signals.step)
        self.state.enter_reason = transition.enter_reason
        self.state.exit_reason = transition.exit_reason
        self.state.last_transition_step = int(signals.step)
        return transition

    def _next_phase(self, signals: PhaseSignals) -> Phase:
        if bool(signals.salvage_active):
            return Phase.SALVAGE
        if bool(signals.warmup_active):
            return Phase.WARMUP
        if bool(signals.force_seek):
            return Phase.SEEK_ANCHOR
        if bool(signals.cold_start_active) and not bool(signals.seek_exit_ready):
            return Phase.SEEK_ANCHOR
        if bool(signals.reroute_guard_hold):
            return Phase.HOLD_ANCHOR
        if str(signals.mode).upper() == "ADAPT":
            return Phase.SHOCK_LOCAL_REROUTE
        return Phase.HOLD_ANCHOR

    @staticmethod
    def _enter_reason(target: Phase, signals: PhaseSignals) -> str:
        if target == Phase.WARMUP:
            return "warmup_hold"
        if target == Phase.SEEK_ANCHOR:
            return "cold_start_seek"
        if target == Phase.SHOCK_LOCAL_REROUTE:
            return "shock_or_force_adapt"
        if target == Phase.SALVAGE:
            return "quality_or_feasibility_salvage"
        if bool(signals.reroute_guard_hold):
            return "reroute_guard_hold"
        return "stable_hold"

    @staticmethod
    def _exit_reason(current: Phase, target: Phase, signals: PhaseSignals) -> str:
        if current == target:
            return "no_transition"
        if current == Phase.WARMUP and target != Phase.WARMUP:
            return "warmup_done"
        if current == Phase.SEEK_ANCHOR and target == Phase.HOLD_ANCHOR:
            return "seek_stabilized"
        if current == Phase.HOLD_ANCHOR and target == Phase.SHOCK_LOCAL_REROUTE:
            return "shock_triggered"
        if current == Phase.SHOCK_LOCAL_REROUTE and target == Phase.HOLD_ANCHOR:
            if bool(signals.reroute_guard_hold):
                return "reroute_guard_hold"
            return "reroute_window_closed"
        if target == Phase.SALVAGE:
            return "constraint_violation"
        if current == Phase.SALVAGE:
            return "salvage_recovered"
        return f"{current.value.lower()}_to_{target.value.lower()}"
