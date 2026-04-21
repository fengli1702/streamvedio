from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .stats import StatsStore, confidence_bounds
from .types import ConfigX


@dataclass
class SafetyResult:
    safe_set: List[ConfigX]
    margin_report: Dict[str, Dict[str, float]]
    salvaged: bool = False
    salvage_choice: Optional[ConfigX] = None
    salvage_reason: Optional[str] = None


class SafetyFilter:
    """Constraint screening with conservative confidence bounds."""

    def __init__(
        self,
        *,
        tau: Optional[float] = None,
        sla: Optional[float] = None,
        mem_limit: Optional[float] = None,
        train_min: Optional[float] = None,
        beta: float = 1.0,
        sigma_floor: float = 1e-3,
        min_count_for_trust: int = 2,
        max_staleness_ticks: Optional[int] = 64,
        quality_metric: str = "acc_mean",
        latency_metric: str = "lat_mean",
        mem_metric: str = "mem_peak",
        train_tps_metric: str = "train_tps",
    ) -> None:
        self.tau = tau
        self.sla = sla
        self.mem_limit = mem_limit
        self.train_min = train_min
        self.beta = float(beta)
        self.sigma_floor = max(0.0, float(sigma_floor))
        self.min_count_for_trust = max(0, int(min_count_for_trust))
        self.max_staleness_ticks = (
            None
            if max_staleness_ticks is None
            else max(0, int(max_staleness_ticks))
        )
        self.quality_metric = quality_metric
        self.latency_metric = latency_metric
        self.mem_metric = mem_metric
        self.train_tps_metric = train_tps_metric

    def filter_candidates(
        self,
        *,
        regime_id: str,
        candidates: Sequence[ConfigX],
        stats_store: StatsStore,
        x_safe_default: Optional[ConfigX] = None,
        x_prev: Optional[ConfigX] = None,
    ) -> SafetyResult:
        safe_set: List[ConfigX] = []
        report: Dict[str, Dict[str, float]] = {}

        for cfg in candidates:
            entry = self._evaluate_one(regime_id=regime_id, x=cfg, stats_store=stats_store)
            report[cfg.hash_key()] = entry
            if entry["is_safe"] > 0.0:
                safe_set.append(cfg)

        if safe_set:
            return SafetyResult(safe_set=safe_set, margin_report=report, salvaged=False)

        salvage, salvage_reason = self._salvage(
            candidates,
            report,
            x_safe_default=x_safe_default,
            x_prev=x_prev,
        )
        if salvage is None:
            return SafetyResult(safe_set=[], margin_report=report, salvaged=False)
        return SafetyResult(
            safe_set=[salvage],
            margin_report=report,
            salvaged=True,
            salvage_choice=salvage,
            salvage_reason=salvage_reason,
        )

    def _evaluate_one(
        self,
        *,
        regime_id: str,
        x: ConfigX,
        stats_store: StatsStore,
    ) -> Dict[str, float]:
        last_seen, count = stats_store.get_last_seen_count(regime_id, x)
        staleness = None
        if last_seen is not None:
            staleness = max(0, int(stats_store.current_tick()) - int(last_seen))
        trusted_by_count = count >= self.min_count_for_trust
        trusted_by_recency = True
        if self.max_staleness_ticks is not None:
            trusted_by_recency = staleness is not None and staleness <= self.max_staleness_ticks
        is_trusted = bool(trusted_by_count and trusted_by_recency)

        q_mu, q_sigma = stats_store.get_mu_sigma(regime_id, x, self.quality_metric)
        l_mu, l_sigma = stats_store.get_mu_sigma(regime_id, x, self.latency_metric)
        m_mu, m_sigma = stats_store.get_mu_sigma(regime_id, x, self.mem_metric)
        t_mu, t_sigma = stats_store.get_mu_sigma(regime_id, x, self.train_tps_metric)

        q_lcb = confidence_bounds(
            q_mu,
            self._effective_sigma(mu=q_mu, sigma=q_sigma),
            self.beta,
            "maximize",
        )
        l_ucb = confidence_bounds(
            l_mu,
            self._effective_sigma(mu=l_mu, sigma=l_sigma),
            self.beta,
            "minimize",
        )
        m_ucb = confidence_bounds(
            m_mu,
            self._effective_sigma(mu=m_mu, sigma=m_sigma),
            self.beta,
            "minimize",
        )
        t_lcb = confidence_bounds(
            t_mu,
            self._effective_sigma(mu=t_mu, sigma=t_sigma),
            self.beta,
            "maximize",
        )

        q_margin = self._margin_ge(q_lcb, self.tau)
        l_margin = self._margin_le(l_ucb, self.sla)
        m_margin = self._margin_le(m_ucb, self.mem_limit)
        t_margin = self._margin_ge(t_lcb, self.train_min)

        safe = True
        for margin in (q_margin, l_margin, m_margin, t_margin):
            if margin is None:
                continue
            if margin < 0.0:
                safe = False
                break

        constraints_active = (
            self.tau is not None
            or self.sla is not None
            or self.mem_limit is not None
            or self.train_min is not None
        )
        if constraints_active and not is_trusted:
            safe = False

        # Missing bound for an active constraint is treated as unsafe (conservative).
        if self.tau is not None and q_lcb is None:
            safe = False
        if self.sla is not None and l_ucb is None:
            safe = False
        if self.mem_limit is not None and m_ucb is None:
            safe = False
        if self.train_min is not None and t_lcb is None:
            safe = False

        return {
            "quality_lcb": -1.0 if q_lcb is None else float(q_lcb),
            "latency_ucb": -1.0 if l_ucb is None else float(l_ucb),
            "mem_ucb": -1.0 if m_ucb is None else float(m_ucb),
            "train_tps_lcb": -1.0 if t_lcb is None else float(t_lcb),
            "quality_margin": 0.0 if q_margin is None else float(q_margin),
            "latency_margin": 0.0 if l_margin is None else float(l_margin),
            "mem_margin": 0.0 if m_margin is None else float(m_margin),
            "train_tps_margin": 0.0 if t_margin is None else float(t_margin),
            "count": float(count),
            "last_seen": -1.0 if last_seen is None else float(last_seen),
            "staleness": -1.0 if staleness is None else float(staleness),
            "is_trusted": 1.0 if is_trusted else 0.0,
            "is_safe": 1.0 if safe else 0.0,
        }

    def _effective_sigma(
        self,
        *,
        mu: Optional[float],
        sigma: Optional[float],
    ) -> Optional[float]:
        if mu is None:
            return None
        s = 0.0 if sigma is None else abs(float(sigma))
        return max(self.sigma_floor, s)

    @staticmethod
    def _margin_ge(bound: Optional[float], threshold: Optional[float]) -> Optional[float]:
        if threshold is None:
            return None
        if bound is None:
            return -1.0
        return float(bound) - float(threshold)

    @staticmethod
    def _margin_le(bound: Optional[float], threshold: Optional[float]) -> Optional[float]:
        if threshold is None:
            return None
        if bound is None:
            return -1.0
        return float(threshold) - float(bound)

    def _salvage(
        self,
        candidates: Sequence[ConfigX],
        report: Dict[str, Dict[str, float]],
        *,
        x_safe_default: Optional[ConfigX],
        x_prev: Optional[ConfigX],
    ) -> Tuple[Optional[ConfigX], Optional[str]]:
        conservative_safe = self._select_conservative(
            candidates,
            report=report,
            avoid_key=x_safe_default.hash_key() if x_safe_default is not None else None,
        )
        if x_safe_default is not None:
            key = x_safe_default.hash_key()
            row = report.get(key)
            if row is not None and float(row.get("is_safe", 0.0)) > 0.0:
                return x_safe_default, "safe_default"
            if conservative_safe is not None:
                return conservative_safe, "conservative_template_safe"
            least_violation = self._select_least_violation(
                candidates,
                report=report,
                x_prev=x_prev,
            )
            if least_violation is not None:
                return least_violation, "least_violation_near_prev"
            if row is None:
                return x_safe_default, "safe_default_unknown"
            return x_safe_default, "safe_default_violation"

        if conservative_safe is not None:
            return conservative_safe, "conservative_template_safe"
        if not candidates:
            return None, None

        # Fallback: choose least-violating candidate by summed negative margins.
        best = self._select_least_violation(candidates, report=report, x_prev=x_prev)
        return best, "least_violation"

    @staticmethod
    def _violation_penalty(row: Dict[str, float]) -> float:
        penalty = 0.0
        for key in ("quality_margin", "latency_margin", "mem_margin", "train_tps_margin"):
            margin = float(row.get(key, 0.0))
            if margin < 0.0:
                penalty += abs(margin)
        return penalty

    @staticmethod
    def _switch_distance(a: ConfigX, b: ConfigX) -> float:
        distance = abs(int(a.ctx) - int(b.ctx)) + abs(int(a.inf) - int(b.inf))
        if a.ib != b.ib:
            distance += 2.0
        if a.tb != b.tb:
            distance += 2.0
        return float(distance)

    def _select_least_violation(
        self,
        candidates: Sequence[ConfigX],
        *,
        report: Mapping[str, Mapping[str, float]],
        x_prev: Optional[ConfigX],
    ) -> Optional[ConfigX]:
        if not candidates:
            return None

        def _score(cfg: ConfigX) -> Tuple[float, float, int, int]:
            penalty = float(self._violation_penalty(dict(report.get(cfg.hash_key(), {}))))
            switch = 0.0 if x_prev is None else self._switch_distance(cfg, x_prev)
            return (
                penalty,
                switch,
                int(cfg.inf),
                -int(cfg.ctx),
            )

        return min(candidates, key=_score)

    @staticmethod
    def _select_conservative(
        candidates: Sequence[ConfigX],
        *,
        report: Optional[Mapping[str, Mapping[str, float]]] = None,
        avoid_key: Optional[str] = None,
    ) -> Optional[ConfigX]:
        pool = [cfg for cfg in candidates if cfg.hash_key() != avoid_key]
        if report is not None:
            pool = [
                cfg
                for cfg in pool
                if float(report.get(cfg.hash_key(), {}).get("is_safe", 0.0)) > 0.0
            ]
        if not pool:
            return None
        return sorted(
            pool,
            key=lambda cfg: (
                int(cfg.inf),
                -int(cfg.ctx),
                int(cfg.ib) if cfg.ib is not None else -1,
                int(cfg.tb) if cfg.tb is not None else -1,
            ),
        )[0]
