from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .phases import Phase
from .types import ConfigX, Decision, DerivedMetrics


@dataclass
class TrialState:
    cfg: Optional[ConfigX] = None
    age: int = 0
    beats_anchor_count: int = 0
    promoted: bool = False


@dataclass
class AnchorState:
    anchor_cfg: Optional[ConfigX] = None
    anchor_regime: Optional[str] = None
    anchor_confidence: float = 0.0
    anchor_quality_margin: Optional[float] = None
    anchor_latency_margin: Optional[float] = None
    trial: TrialState = field(default_factory=TrialState)
    last_promoted_step: Optional[int] = None


class AnchorTracker:
    """Track anchor/trial lifecycle without changing policy semantics."""

    def __init__(self) -> None:
        self.state = AnchorState()

    def update(
        self,
        *,
        phase: Phase,
        regime_id: str,
        decision: Decision,
        derived: DerivedMetrics,
        step: int,
        promote_windows: int = 2,
        init_promote_windows: int = 2,
        init_confidence: float = 0.20,
    ) -> bool:
        promoted = False
        promote_windows = max(1, int(promote_windows))
        init_promote_windows = max(1, int(init_promote_windows))
        init_confidence = min(1.0, max(0.0, float(init_confidence)))
        reason = str(decision.reason)
        bootstrap_promotable_reasons = {
            "seek_anchor_hold",
            "hold_anchor",
            "hold_anchor_score",
            "phase_hold",
            "local_reroute_trial",
            "reroute_fallback_anchor_hold",
            "salvage_local_recovery",
        }

        if self.state.anchor_cfg is None:
            if phase in (Phase.WARMUP, Phase.SALVAGE):
                self.state.trial = TrialState()
                return promoted
            trial = self.state.trial
            if trial.cfg is None or not self._same(trial.cfg, decision.x_next):
                trial.cfg = decision.x_next
                trial.age = 1
                trial.beats_anchor_count = 1
                trial.promoted = False
                return promoted
            trial.age += 1
            trial.beats_anchor_count += 1
            if reason not in bootstrap_promotable_reasons:
                return promoted
            if trial.age < init_promote_windows:
                return promoted
            self.state.anchor_cfg = trial.cfg
            self.state.anchor_regime = regime_id
            self.state.anchor_confidence = init_confidence
            self.state.trial = TrialState()
            self._update_anchor_margins(derived)
            return promoted

        anchor = self.state.anchor_cfg
        if anchor is None:
            return promoted

        if self._same(decision.x_next, anchor):
            self.state.anchor_confidence = min(1.0, float(self.state.anchor_confidence) + 0.08)
            self.state.trial = TrialState()
            self._update_anchor_margins(derived)
            return promoted

        if phase == Phase.SALVAGE:
            return promoted

        trial = self.state.trial
        if trial.cfg is None or not self._same(trial.cfg, decision.x_next):
            trial.cfg = decision.x_next
            trial.age = 1
            trial.beats_anchor_count = 1
            trial.promoted = False
        else:
            trial.age += 1
            trial.beats_anchor_count += 1

        if phase == Phase.HOLD_ANCHOR:
            # HOLD allows only micro deviations; lower confidence if drifted from anchor.
            self.state.anchor_confidence = max(0.0, float(self.state.anchor_confidence) - 0.05)
            return promoted

        if phase == Phase.SHOCK_LOCAL_REROUTE and trial.beats_anchor_count >= promote_windows:
            self.state.anchor_cfg = trial.cfg
            self.state.anchor_regime = regime_id
            self.state.anchor_confidence = 0.50
            self.state.last_promoted_step = int(step)
            self.state.trial.promoted = True
            self._update_anchor_margins(derived)
            promoted = True
        return promoted

    def _update_anchor_margins(self, derived: DerivedMetrics) -> None:
        self.state.anchor_quality_margin = (
            None if derived.quality_margin is None else float(derived.quality_margin)
        )
        self.state.anchor_latency_margin = (
            None if derived.latency_margin is None else float(derived.latency_margin)
        )

    @staticmethod
    def _same(a: ConfigX, b: Optional[ConfigX]) -> bool:
        if b is None:
            return False
        return a.hash_key() == b.hash_key()

    def to_dict(self) -> Dict[str, Any]:
        trial_cfg = self.state.trial.cfg
        anchor_cfg = self.state.anchor_cfg
        return {
            "anchor_cfg": None if anchor_cfg is None else anchor_cfg.to_dict(),
            "anchor_regime": self.state.anchor_regime,
            "anchor_confidence": float(self.state.anchor_confidence),
            "anchor_quality_margin": self.state.anchor_quality_margin,
            "anchor_latency_margin": self.state.anchor_latency_margin,
            "trial_cfg": None if trial_cfg is None else trial_cfg.to_dict(),
            "trial_age": int(self.state.trial.age),
            "trial_beats_anchor_count": int(self.state.trial.beats_anchor_count),
            "trial_promoted": bool(self.state.trial.promoted),
            "last_promoted_step": self.state.last_promoted_step,
        }
