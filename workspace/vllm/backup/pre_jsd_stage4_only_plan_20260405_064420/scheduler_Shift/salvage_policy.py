from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from .config_space import ConfigSpace
from .types import ConfigX, Decision, DerivedMetrics


class SalvagePolicy:
    """Constraint or quality rescue policy with local-first preference."""

    def select_next(
        self,
        *,
        mode: str,
        regime_id: str,
        x_prev: ConfigX,
        x_safe_default: ConfigX,
        safe_set: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        derived: DerivedMetrics,
        config_space: ConfigSpace,
        stable_eps_fast: int,
        stable_eps_slow: int,
    ) -> Decision:
        choice = self._select_choice(
            x_prev=x_prev,
            x_safe_default=x_safe_default,
            safe_set=safe_set,
            margin_report=margin_report,
            derived=derived,
            config_space=config_space,
            stable_eps_fast=stable_eps_fast,
            stable_eps_slow=stable_eps_slow,
        )
        if choice is None:
            return Decision(
                x_next=x_safe_default,
                reason="salvage_safe_default",
                mode=mode,
                regime_id=regime_id,
                meta={"quality_salvage": 1.0},
            )
        return Decision(
            x_next=choice,
            reason="salvage_local_recovery",
            mode=mode,
            regime_id=regime_id,
            meta={"quality_salvage": 1.0},
        )

    @staticmethod
    def _switch_distance(a: ConfigX, b: ConfigX) -> float:
        distance = abs(int(a.ctx) - int(b.ctx)) + abs(int(a.inf) - int(b.inf))
        if a.ib != b.ib:
            distance += 2.0
        if a.tb != b.tb:
            distance += 2.0
        return float(distance)

    def _select_choice(
        self,
        *,
        x_prev: ConfigX,
        x_safe_default: ConfigX,
        safe_set: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        derived: DerivedMetrics,
        config_space: ConfigSpace,
        stable_eps_fast: int,
        stable_eps_slow: int,
    ) -> Optional[ConfigX]:
        pool: Dict[str, ConfigX] = {cfg.hash_key(): cfg for cfg in safe_set}
        if not pool:
            return x_safe_default

        current_key = x_prev.hash_key()
        default_key = x_safe_default.hash_key()
        local_reach = config_space.neighbors(
            x_prev,
            eps_fast=max(1, int(stable_eps_fast)),
            eps_slow=max(0, int(stable_eps_slow)),
        )
        local_keys = {cfg.hash_key() for cfg in local_reach}
        preferred: Dict[str, ConfigX] = {}
        for key, cfg in pool.items():
            if key == current_key or key == default_key or key in local_keys:
                preferred[key] = cfg
        if preferred:
            pool = preferred

        quality_bad = derived.quality_margin is not None and float(derived.quality_margin) < 0.0

        def _score(cfg: ConfigX) -> Tuple[float, ...]:
            row = margin_report.get(cfg.hash_key(), {})
            quality_margin = float(row.get("quality_margin", 0.0))
            latency_margin = float(row.get("latency_margin", 0.0))
            is_safe = float(row.get("is_safe", 1.0))
            is_trusted = float(row.get("is_trusted", 0.0))
            switch_penalty = self._switch_distance(cfg, x_prev)
            d_ctx = abs(int(cfg.ctx) - int(x_prev.ctx))
            d_inf = abs(int(cfg.inf) - int(x_prev.inf))
            move_bonus = 1.0 if cfg.hash_key() != current_key else 0.0
            default_bonus = 1.0 if cfg.hash_key() == default_key else 0.0
            quality_axis_bonus = 1.0 if quality_bad and d_ctx > 0 else 0.0
            inf_move_penalty = float(d_inf) if quality_bad else 0.5 * float(d_inf)
            return (
                is_safe,
                is_trusted,
                quality_margin,
                latency_margin,
                quality_axis_bonus,
                default_bonus,
                move_bonus,
                -inf_move_penalty,
                -switch_penalty,
                float(cfg.ctx),
                -float(cfg.inf),
            )

        return max(pool.values(), key=_score)
