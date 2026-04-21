from __future__ import annotations

from typing import Any, Dict, Optional

from .anchor import AnchorState
from .phases import PhaseState, PhaseTransition
from .types import ConfigX


def build_phase_log_meta(
    *,
    phase_state: PhaseState,
    transition: Optional[PhaseTransition],
    anchor_state: AnchorState,
    candidate_pool_type: str,
    switch_block_reason: Optional[str],
    trial_promoted: bool,
) -> Dict[str, Any]:
    trial_cfg = anchor_state.trial.cfg
    return {
        "phase": phase_state.phase.value,
        "phase_enter_reason": phase_state.enter_reason,
        "phase_exit_reason": phase_state.exit_reason,
        "phase_transition": 0.0 if transition is None else 1.0,
        "phase_from": None if transition is None or transition.from_phase is None else transition.from_phase.value,
        "phase_to": None if transition is None else transition.to_phase.value,
        "phase_transition_step": float(phase_state.last_transition_step),
        "anchor_cfg": _compact_cfg(anchor_state.anchor_cfg),
        "anchor_confidence": float(anchor_state.anchor_confidence),
        "trial_cfg": _compact_cfg(trial_cfg),
        "trial_age": float(anchor_state.trial.age),
        "trial_beats_anchor_count": float(anchor_state.trial.beats_anchor_count),
        "trial_promoted": 1.0 if trial_promoted or anchor_state.trial.promoted else 0.0,
        "candidate_pool_type": str(candidate_pool_type),
        "switch_block_reason": switch_block_reason,
    }


def _compact_cfg(cfg: Optional[ConfigX]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    return {
        "ctx": int(cfg.ctx),
        "inf": int(cfg.inf),
        "ib": None if cfg.ib is None else int(cfg.ib),
        "tb": None if cfg.tb is None else int(cfg.tb),
        "key": cfg.hash_key(),
    }
