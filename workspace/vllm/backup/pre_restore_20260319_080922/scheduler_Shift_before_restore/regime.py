from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .types import DerivedMetrics

STABLE = "STABLE"
ADAPT = "ADAPT"


def _safe(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


@dataclass
class ShockResult:
    mode: str
    shock_score: float
    components: Dict[str, float]
    adapt_hold_counter: int = 0


class ShockDetector:
    """Shock detector with hysteresis and hold logic."""

    def __init__(
        self,
        *,
        w_drift: float = 1.0,
        w_accept: float = 1.0,
        w_reverify: float = 1.0,
        w_verify_ratio: float = 1.0,
        w_waste_rate: float = 1.0,
        f_drift: float = 0.02,
        f_accept: float = 0.05,
        f_reverify: float = 500.0,
        f_verify_ratio: float = 0.05,
        f_waste_rate: float = 0.05,
        component_cap: float = 1.0,
        t_high: float = 0.35,
        t_low: float = 0.20,
        adapt_hold: int = 2,
    ) -> None:
        self.weights = {
            "drift": float(w_drift),
            "accept": float(w_accept),
            "reverify": float(w_reverify),
            "verify_ratio": float(w_verify_ratio),
            "waste_rate": float(w_waste_rate),
        }
        self.floors = {
            "drift": max(1e-9, float(f_drift)),
            "accept": max(1e-9, float(f_accept)),
            "reverify": max(1e-9, float(f_reverify)),
            "verify_ratio": max(1e-9, float(f_verify_ratio)),
            "waste_rate": max(1e-9, float(f_waste_rate)),
        }
        self.component_cap = max(0.0, float(component_cap))
        self.t_high = float(t_high)
        self.t_low = float(t_low)
        self.adapt_hold = int(max(0, adapt_hold))

        self.mode = STABLE
        self._hold_counter = 0
        self._prev: Optional[DerivedMetrics] = None

    def force_stable(self) -> None:
        self.mode = STABLE
        self._hold_counter = 0

    def _compute_components(self, current: DerivedMetrics) -> Dict[str, float]:
        if self._prev is None:
            return {
                "drift": 0.0,
                "accept": 0.0,
                "reverify": 0.0,
                "verify_ratio": 0.0,
                "waste_rate": 0.0,
            }
        prev = self._prev

        def _normalized_delta(cur: float, prv: float, floor: float) -> float:
            base = max(abs(prv), float(floor))
            if base <= 0.0:
                return 0.0
            val = abs(float(cur) - float(prv)) / base
            if self.component_cap > 0.0:
                val = min(float(self.component_cap), val)
            return float(val)

        return {
            "drift": _normalized_delta(
                _safe(current.shock_drift),
                _safe(prev.shock_drift),
                self.floors["drift"],
            ),
            "accept": _normalized_delta(
                _safe(current.shock_accept),
                _safe(prev.shock_accept),
                self.floors["accept"],
            ),
            "reverify": _normalized_delta(
                _safe(current.shock_reverify),
                _safe(prev.shock_reverify),
                self.floors["reverify"],
            ),
            "verify_ratio": _normalized_delta(
                _safe(current.verify_ratio),
                _safe(prev.verify_ratio),
                self.floors["verify_ratio"],
            ),
            "waste_rate": _normalized_delta(
                _safe(current.waste_rate),
                _safe(prev.waste_rate),
                self.floors["waste_rate"],
            ),
        }

    def update(
        self,
        current: DerivedMetrics,
        *,
        can_exit_adapt: bool = True,
    ) -> ShockResult:
        components = self._compute_components(current)
        weighted = (
            self.weights["drift"] * components["drift"]
            + self.weights["accept"] * components["accept"]
            + self.weights["reverify"] * components["reverify"]
            + self.weights["verify_ratio"] * components["verify_ratio"]
            + self.weights["waste_rate"] * components["waste_rate"]
        )
        weight_total = (
            max(0.0, self.weights["drift"])
            + max(0.0, self.weights["accept"])
            + max(0.0, self.weights["reverify"])
            + max(0.0, self.weights["verify_ratio"])
            + max(0.0, self.weights["waste_rate"])
        )
        shock_score = 0.0 if weight_total <= 0.0 else float(weighted) / float(weight_total)

        if self.mode == STABLE:
            if shock_score >= self.t_high:
                self.mode = ADAPT
                self._hold_counter = self.adapt_hold
        else:
            if shock_score <= self.t_low and bool(can_exit_adapt):
                if self._hold_counter > 0:
                    self._hold_counter -= 1
                if self._hold_counter == 0:
                    self.mode = STABLE
            else:
                self._hold_counter = self.adapt_hold

        self._prev = current
        return ShockResult(
            mode=self.mode,
            shock_score=shock_score,
            components=components,
            adapt_hold_counter=int(self._hold_counter),
        )


class RegimeQuantizer:
    """Quantize continuous runtime state into a discrete regime id."""

    def __init__(
        self,
        *,
        drift_bins: Sequence[float] = (0.02, 0.05, 0.10),
        accept_bins: Sequence[float] = (0.60, 0.80, 0.92),
        verify_ratio_bins: Sequence[float] = (0.30, 0.55, 0.75),
    ) -> None:
        self.drift_bins = tuple(float(v) for v in drift_bins)
        self.accept_bins = tuple(float(v) for v in accept_bins)
        self.verify_ratio_bins = tuple(float(v) for v in verify_ratio_bins)

    @staticmethod
    def _bucket(value: Optional[float], bins: Sequence[float]) -> int:
        if value is None:
            return -1
        x = float(value)
        idx = 0
        for edge in bins:
            if x <= float(edge):
                return idx
            idx += 1
        return idx

    def quantize(
        self,
        *,
        drift: Optional[float],
        accept: Optional[float],
        verify_ratio: Optional[float],
    ) -> str:
        d_bin = self._bucket(drift, self.drift_bins)
        a_bin = self._bucket(accept, self.accept_bins)
        v_bin = self._bucket(verify_ratio, self.verify_ratio_bins)
        return f"d{d_bin}_a{a_bin}_v{v_bin}"
