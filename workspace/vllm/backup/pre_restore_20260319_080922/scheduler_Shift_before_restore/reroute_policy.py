from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .config_space import ConfigSpace
from .stats import StatsStore
from .types import ConfigX, Decision


@dataclass
class ReroutePolicyConfig:
    reroute_radius: int = 2
    quality_floor_delta: float = 0.03
    accept_delta_pressure: float = 0.08


class LocalReroutePolicy:
    """Shock phase local reroute: centered on anchor neighborhood, not global re-search."""

    def __init__(self, cfg: Optional[ReroutePolicyConfig] = None) -> None:
        self.cfg = cfg or ReroutePolicyConfig()

    def select_next(
        self,
        *,
        mode: str,
        regime_id: str,
        x_prev: ConfigX,
        anchor_cfg: Optional[ConfigX],
        safe_set: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        config_space: ConfigSpace,
        stats_store: StatsStore,
        tau: Optional[float],
        force_adapt_reasons: Optional[Sequence[str]] = None,
        accept_delta: Optional[float] = None,
    ) -> Optional[Decision]:
        if not safe_set:
            return None
        current_key = x_prev.hash_key()
        center = anchor_cfg or x_prev

        neighbors = config_space.neighbors(
            center,
            eps_fast=max(1, int(self.cfg.reroute_radius)),
            eps_slow=0,
        )
        neighbor_keys = {cfg.hash_key() for cfg in neighbors}
        pool = [
            cfg
            for cfg in safe_set
            if cfg.hash_key() != current_key and cfg.hash_key() in neighbor_keys
        ]
        if not pool:
            pool = [cfg for cfg in safe_set if cfg.hash_key() != current_key]
        if not pool:
            return None

        reasons = {str(r) for r in (force_adapt_reasons or [])}
        latency_pressure = "latency_margin_low" in reasons
        quality_pressure = "quality_margin_low" in reasons
        accept_pressure = (
            accept_delta is not None
            and float(accept_delta) >= float(max(0.0, self.cfg.accept_delta_pressure))
        )
        report = margin_report or {}

        prefer_descending = str(mode).upper() == "ADAPT" or latency_pressure or accept_pressure
        if prefer_descending:
            descending = [
                cfg
                for cfg in pool
                if int(cfg.inf) <= int(x_prev.inf)
                and int(cfg.ctx) <= int(x_prev.ctx)
                and (int(cfg.inf) < int(x_prev.inf) or int(cfg.ctx) < int(x_prev.ctx))
            ]
            if descending:
                pool = descending
            else:
                # In ADAPT/shock, avoid uphill reroute. Let caller fallback to hold.
                return None

        if quality_pressure and not latency_pressure and not accept_pressure:
            quality_recovery = [
                cfg
                for cfg in pool
                if int(cfg.ctx) >= int(x_prev.ctx) and int(cfg.inf) <= int(x_prev.inf)
            ]
            if quality_recovery:
                pool = quality_recovery

        def _quality_deficit(cfg: ConfigX) -> float:
            row = dict(report.get(cfg.hash_key(), {}))
            if tau is not None:
                if "quality_margin" in row:
                    return max(0.0, -float(row.get("quality_margin", 0.0)))
                q_mu, _ = stats_store.get_mu_sigma(regime_id, cfg, "acc_mean")
                if q_mu is None:
                    return 1e6
                return max(0.0, float(tau) - float(q_mu))
            return 0.0

        def _latency_est(cfg: ConfigX) -> float:
            row = dict(report.get(cfg.hash_key(), {}))
            if "latency_ucb" in row:
                val = float(row.get("latency_ucb", -1.0))
                if val >= 0.0:
                    return val
            lat_mu, _ = stats_store.get_mu_sigma(regime_id, cfg, "lat_mean")
            if lat_mu is not None:
                return float(lat_mu)
            return float("inf")

        def _score(cfg: ConfigX):
            row = dict(report.get(cfg.hash_key(), {}))
            trusted_rank = 0 if float(row.get("is_trusted", 0.0)) > 0.0 else 1
            _, cnt = stats_store.get_last_seen_count(regime_id, cfg)
            return (
                trusted_rank,
                _quality_deficit(cfg),
                _latency_est(cfg),
                self._switch_distance(cfg, x_prev),
                int(float(row.get("count", float(cnt)))),
                int(cfg.ctx),
                int(cfg.inf),
            )

        choice = min(pool, key=_score)
        return Decision(
            x_next=choice,
            reason="local_reroute_trial",
            mode=mode,
            regime_id=regime_id,
            meta={
                "reroute_center": "anchor" if anchor_cfg is not None else "prev",
                "reroute_radius": float(max(1, int(self.cfg.reroute_radius))),
                "reroute_accept_delta": None if accept_delta is None else float(accept_delta),
                "reroute_accept_pressure": 1.0 if accept_pressure else 0.0,
            },
        )

    @staticmethod
    def _switch_distance(a: ConfigX, b: ConfigX) -> float:
        distance = abs(int(a.ctx) - int(b.ctx)) + abs(int(a.inf) - int(b.inf))
        if a.ib != b.ib:
            distance += 2.0
        if a.tb != b.tb:
            distance += 2.0
        return float(distance)
