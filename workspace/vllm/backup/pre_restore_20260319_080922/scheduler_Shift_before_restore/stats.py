from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .types import ConfigX


class EMAStats:
    """Exponential moving mean/variance for multiple metrics."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._state: Dict[str, Dict[str, float]] = {}

    def update(self, metric: str, value: float, alpha: Optional[float] = None) -> None:
        a = self.alpha if alpha is None else float(alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        x = float(value)
        if metric not in self._state:
            self._state[metric] = {"mu": x, "var": 0.0, "count": 1.0}
            return

        s = self._state[metric]
        mu_prev = s["mu"]
        var_prev = s["var"]
        mu_new = (1.0 - a) * mu_prev + a * x
        var_new = (1.0 - a) * var_prev + a * ((x - mu_new) ** 2)
        s["mu"] = mu_new
        s["var"] = max(0.0, var_new)
        s["count"] += 1.0

    def update_many(self, metrics: Mapping[str, float], alpha: Optional[float] = None) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            self.update(k, float(v), alpha=alpha)

    def get_mu_sigma(self, metric: str) -> Tuple[Optional[float], Optional[float]]:
        s = self._state.get(metric)
        if s is None:
            return None, None
        return float(s["mu"]), math.sqrt(max(0.0, float(s["var"])))

    def count(self, metric: str) -> int:
        s = self._state.get(metric)
        if s is None:
            return 0
        return int(s["count"])


class StatsStore:
    """Regime-aware recency stats keyed by (regime_id, config_key)."""

    def __init__(self, default_alpha: float = 0.2) -> None:
        self.default_alpha = float(default_alpha)
        self._tick = 0
        self._table: Dict[Tuple[str, str], Dict[str, object]] = {}

    @staticmethod
    def _cfg_key(x: ConfigX) -> str:
        return x.hash_key()

    def update(
        self,
        regime_id: str,
        x: ConfigX,
        metrics: Mapping[str, Optional[float]],
        alpha: Optional[float] = None,
    ) -> None:
        self._tick += 1
        key = (str(regime_id), self._cfg_key(x))
        if key not in self._table:
            self._table[key] = {
                "ema": EMAStats(alpha=self.default_alpha),
                "last_seen": self._tick,
                "count": 0,
            }
        row = self._table[key]
        ema = row["ema"]
        assert isinstance(ema, EMAStats)
        ema.update_many(
            {k: float(v) for k, v in metrics.items() if v is not None},
            alpha=alpha,
        )
        row["last_seen"] = self._tick
        row["count"] = int(row["count"]) + 1

    def get_mu_sigma(
        self,
        regime_id: str,
        x: ConfigX,
        metric: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        key = (str(regime_id), self._cfg_key(x))
        row = self._table.get(key)
        if row is None:
            return None, None
        ema = row["ema"]
        assert isinstance(ema, EMAStats)
        return ema.get_mu_sigma(metric)

    def get_last_seen_count(self, regime_id: str, x: ConfigX) -> Tuple[Optional[int], int]:
        key = (str(regime_id), self._cfg_key(x))
        row = self._table.get(key)
        if row is None:
            return None, 0
        return int(row["last_seen"]), int(row["count"])

    def current_tick(self) -> int:
        return int(self._tick)


def confidence_bounds(
    mu: Optional[float],
    sigma: Optional[float],
    beta: float,
    direction: str,
) -> Optional[float]:
    if mu is None:
        return None
    s = 0.0 if sigma is None else float(sigma)
    b = max(0.0, float(beta))
    d = direction.lower()
    if d in ("maximize", "max", "larger", "higher"):
        return float(mu) - b * s
    if d in ("minimize", "min", "smaller", "lower"):
        return float(mu) + b * s
    raise ValueError(f"unknown direction: {direction}")
