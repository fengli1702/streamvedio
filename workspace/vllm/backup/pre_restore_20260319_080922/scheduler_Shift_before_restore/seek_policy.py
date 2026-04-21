from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .config_space import ConfigSpace
from .stats import StatsStore
from .types import ConfigX, Decision, WindowMetrics


@dataclass
class SeekPolicyConfig:
    probe_eps_fast: int = 2
    probe_eps_slow: int = 0
    avoid_two_cycle: bool = True
    quality_floor_delta: float = 0.03
    whitelist_latency_slack: float = 0.15
    whitelist_quality_slack: float = 0.60
    # Maximum allowed one-step jump for whitelist break (L1 distance on ctx/inf).
    # 1 means only single-axis +/-1 jump; prevents large, unstable breaks.
    whitelist_max_switch_distance: int = 1
    directional_enable: bool = True
    prefer_large_descend: bool = True
    prefer_small_ascend: bool = True
    large_ctx_threshold: int = 6
    large_inf_threshold: int = 6
    small_ctx_threshold: int = 1
    small_inf_threshold: int = 1
    target_ctx: int = 2
    target_inf: int = 2


class SeekAnchorPolicy:
    """Seek phase selector: feasible -> trusted -> lower latency/sufficient quality -> lower switch."""

    def __init__(self, cfg: Optional[SeekPolicyConfig] = None) -> None:
        self.cfg = cfg or SeekPolicyConfig()

    def select_next(
        self,
        *,
        mode: str,
        regime_id: str,
        x_prev: ConfigX,
        x_prev_prev: Optional[ConfigX],
        safe_set: Sequence[ConfigX],
        candidates: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        window: Optional[WindowMetrics],
        config_space: ConfigSpace,
        stats_store: StatsStore,
        tau: Optional[float],
        preferred_axis: str,
        locked_axis: str = "any",
        force_adapt_reasons: Optional[Sequence[str]] = None,
        whitelist_probe_enable: bool = True,
        whitelist_probe_remaining: int = 0,
        cold_start_relax_safety: bool = False,
        allow_whitelist_break: bool = True,
    ) -> Optional[Decision]:
        current_key = x_prev.hash_key()
        report = margin_report or {}

        pool = [cfg for cfg in safe_set if cfg.hash_key() != current_key]
        if (
            not pool
            and bool(allow_whitelist_break)
            and whitelist_probe_enable
            and whitelist_probe_remaining > 0
        ):
            deadlock_choice = self._select_whitelist_break(
                x_prev=x_prev,
                x_prev_prev=x_prev_prev,
                candidates=candidates,
                margin_report=report,
                preferred_axis=preferred_axis,
                config_space=config_space,
                stats_store=stats_store,
                regime_id=regime_id,
                tau=tau,
                locked_axis=locked_axis,
                cold_start_relax_safety=cold_start_relax_safety,
            )
            if deadlock_choice is not None:
                return Decision(
                    x_next=deadlock_choice,
                    reason="seek_anchor_whitelist_break",
                    mode=mode,
                    regime_id=regime_id,
                    meta={
                        "cold_start_active": True,
                        "deadlock_break": 1.0,
                    },
                )
            return None
        if not pool:
            return None

        local = config_space.neighbors(
            x_prev,
            eps_fast=max(1, int(self.cfg.probe_eps_fast)),
            eps_slow=max(0, int(self.cfg.probe_eps_slow)),
        )
        local_keys = {cfg.hash_key() for cfg in local}
        local_pool = [cfg for cfg in pool if cfg.hash_key() in local_keys]
        if local_pool:
            pool = local_pool
        pool = self._apply_axis_lock(pool=pool, x_prev=x_prev, locked_axis=locked_axis)
        if not pool:
            return None

        if bool(self.cfg.avoid_two_cycle) and x_prev_prev is not None and len(pool) > 1:
            prev_prev_key = x_prev_prev.hash_key()
            non_cycle = [cfg for cfg in pool if cfg.hash_key() != prev_prev_key]
            if non_cycle:
                pool = non_cycle

        max_jump = int(max(0, self.cfg.whitelist_max_switch_distance))
        if max_jump > 0:
            bounded = [
                cfg
                for cfg in pool
                if self._switch_distance(cfg, x_prev) <= float(max_jump)
            ]
            if bounded:
                pool = bounded

        zone = self._workload_zone(x_prev)
        pool = self._apply_directional_filter(pool=pool, x_prev=x_prev, zone=zone)
        if not pool:
            return None

        latency_pressure = "latency_margin_low" in {str(x) for x in (force_adapt_reasons or [])}
        if latency_pressure and zone == "large":
            non_up = [
                cfg
                for cfg in pool
                if int(cfg.ctx) <= int(x_prev.ctx) and int(cfg.inf) <= int(x_prev.inf)
            ]
            if non_up:
                pool = non_up

        quality_ref = None if window is None else float(window.acc_mean)
        latency_ref = None if window is None else float(window.lat_mean)

        def _axis_penalty(cfg: ConfigX) -> int:
            d_ctx = abs(int(cfg.ctx) - int(x_prev.ctx))
            d_inf = abs(int(cfg.inf) - int(x_prev.inf))
            if preferred_axis == "inf":
                if d_inf > 0 and d_ctx == 0:
                    return 0
                if d_inf > 0:
                    return 1
                if d_ctx > 0:
                    return 2
                return 3
            if preferred_axis == "ctx":
                if d_ctx > 0 and d_inf == 0:
                    return 0
                if d_ctx > 0:
                    return 1
                if d_inf > 0:
                    return 2
                return 3
            return 0

        def _quality_deficit(cfg: ConfigX) -> float:
            row = dict(report.get(cfg.hash_key(), {}))
            if tau is not None:
                if "quality_margin" in row:
                    return max(0.0, -float(row.get("quality_margin", 0.0)))
                q_mu, _ = stats_store.get_mu_sigma(regime_id, cfg, "acc_mean")
                if q_mu is None:
                    return 1e6
                return max(0.0, float(tau) - float(q_mu))

            if quality_ref is None:
                return 0.0
            q_mu, _ = stats_store.get_mu_sigma(regime_id, cfg, "acc_mean")
            if q_mu is None:
                return 0.0
            floor = float(quality_ref) - float(max(0.0, self.cfg.quality_floor_delta))
            return max(0.0, floor - float(q_mu))

        def _latency_est(cfg: ConfigX) -> float:
            row = dict(report.get(cfg.hash_key(), {}))
            if "latency_ucb" in row:
                val = float(row.get("latency_ucb", -1.0))
                if val >= 0.0:
                    return val
            lat_mu, _ = stats_store.get_mu_sigma(regime_id, cfg, "lat_mean")
            if lat_mu is not None:
                return float(lat_mu)
            if latency_ref is not None:
                return float(latency_ref)
            return float("inf")

        def _score(cfg: ConfigX):
            row = dict(report.get(cfg.hash_key(), {}))
            trusted_rank = 0 if float(row.get("is_trusted", 0.0)) > 0.0 else 1
            _, cnt = stats_store.get_last_seen_count(regime_id, cfg)
            switch = self._switch_distance(cfg, x_prev)
            cnt_tiebreak = int(float(row.get("count", float(cnt))))
            direction_rank = self._direction_rank(cfg=cfg, x_prev=x_prev, zone=zone)
            return (
                *direction_rank,
                trusted_rank,
                _quality_deficit(cfg),
                _latency_est(cfg),
                _axis_penalty(cfg),
                switch,
                cnt_tiebreak,
                int(cfg.ctx),
                int(cfg.inf),
            )

        chosen = min(pool, key=_score)
        return Decision(
            x_next=chosen,
            reason="seek_anchor",
            mode=mode,
            regime_id=regime_id,
            meta={"cold_start_active": True},
        )

    def _select_whitelist_break(
        self,
        *,
        x_prev: ConfigX,
        x_prev_prev: Optional[ConfigX],
        candidates: Sequence[ConfigX],
        margin_report: Mapping[str, Mapping[str, float]],
        preferred_axis: str,
        config_space: ConfigSpace,
        stats_store: StatsStore,
        regime_id: str,
        tau: Optional[float],
        locked_axis: str,
        cold_start_relax_safety: bool,
    ) -> Optional[ConfigX]:
        if cold_start_relax_safety:
            return None
        current_key = x_prev.hash_key()
        neighbors = config_space.neighbors(
            x_prev,
            eps_fast=max(1, int(self.cfg.probe_eps_fast)),
            eps_slow=max(0, int(self.cfg.probe_eps_slow)),
        )
        neighbor_keys = {cfg.hash_key() for cfg in neighbors}
        pool = [
            cfg
            for cfg in candidates
            if cfg.hash_key() != current_key and cfg.hash_key() in neighbor_keys
        ]
        if not pool:
            return None
        pool = self._apply_axis_lock(pool=pool, x_prev=x_prev, locked_axis=locked_axis)
        if not pool:
            return None

        if bool(self.cfg.avoid_two_cycle) and x_prev_prev is not None and len(pool) > 1:
            prev_prev_key = x_prev_prev.hash_key()
            non_cycle = [cfg for cfg in pool if cfg.hash_key() != prev_prev_key]
            if non_cycle:
                pool = non_cycle

        current_row = margin_report.get(current_key, {})
        current_lat_margin = float(current_row.get("latency_margin", 0.0))
        lat_slack = float(max(0.0, self.cfg.whitelist_latency_slack))
        q_slack = float(max(0.0, self.cfg.whitelist_quality_slack))

        qualified = []
        for cfg in pool:
            row = margin_report.get(cfg.hash_key(), {})
            lat_margin = float(row.get("latency_margin", -1e9))
            q_margin = float(row.get("quality_margin", -1e9))
            if lat_margin < (current_lat_margin - lat_slack):
                continue
            if tau is not None and q_margin < -q_slack:
                continue
            qualified.append(cfg)
        if qualified:
            pool = qualified
        zone = self._workload_zone(x_prev)
        pool = self._apply_directional_filter(pool=pool, x_prev=x_prev, zone=zone)
        if not pool:
            return None

        def _axis_bonus(cfg: ConfigX) -> int:
            d_ctx = abs(int(cfg.ctx) - int(x_prev.ctx))
            d_inf = abs(int(cfg.inf) - int(x_prev.inf))
            if preferred_axis == "inf":
                if d_inf > 0 and d_ctx == 0:
                    return 2
                if d_inf > 0:
                    return 1
                return 0
            if preferred_axis == "ctx":
                if d_ctx > 0 and d_inf == 0:
                    return 2
                if d_ctx > 0:
                    return 1
                return 0
            return 0

        def _score(cfg: ConfigX):
            row = margin_report.get(cfg.hash_key(), {})
            lat_margin = float(row.get("latency_margin", -1e9))
            q_margin = float(row.get("quality_margin", -1e9))
            _, cnt = stats_store.get_last_seen_count(regime_id, cfg)
            target_gain = self._target_distance(x_prev) - self._target_distance(cfg)
            direction_bonus = self._direction_bonus(cfg=cfg, x_prev=x_prev, zone=zone)
            return (
                direction_bonus,
                float(target_gain),
                _axis_bonus(cfg),
                -self._switch_distance(cfg, x_prev),
                lat_margin,
                q_margin,
                -float(cnt),
                -float(cfg.ctx),
                -float(cfg.inf),
            )

        return max(pool, key=_score)

    @staticmethod
    def _apply_axis_lock(
        *,
        pool: Sequence[ConfigX],
        x_prev: ConfigX,
        locked_axis: str,
    ) -> Sequence[ConfigX]:
        axis = str(locked_axis).lower()
        if axis not in ("ctx", "inf") or len(pool) <= 1:
            return pool
        if axis == "ctx":
            axis_pool = [
                cfg
                for cfg in pool
                if int(cfg.inf) == int(x_prev.inf) and int(cfg.ctx) != int(x_prev.ctx)
            ]
        else:
            axis_pool = [
                cfg
                for cfg in pool
                if int(cfg.ctx) == int(x_prev.ctx) and int(cfg.inf) != int(x_prev.inf)
            ]
        if axis_pool:
            return axis_pool
        return pool

    @staticmethod
    def _switch_distance(a: ConfigX, b: ConfigX) -> float:
        distance = abs(int(a.ctx) - int(b.ctx)) + abs(int(a.inf) - int(b.inf))
        if a.ib != b.ib:
            distance += 2.0
        if a.tb != b.tb:
            distance += 2.0
        return float(distance)

    def _workload_zone(self, cfg: ConfigX) -> str:
        ctx = int(cfg.ctx)
        inf = int(cfg.inf)
        if (
            ctx >= int(max(1, self.cfg.large_ctx_threshold))
            or inf >= int(max(1, self.cfg.large_inf_threshold))
        ):
            return "large"
        if (
            ctx <= int(max(1, self.cfg.small_ctx_threshold))
            and inf <= int(max(1, self.cfg.small_inf_threshold))
        ):
            return "small"
        return "normal"

    def _target_distance(self, cfg: ConfigX) -> int:
        return abs(int(cfg.ctx) - int(max(1, self.cfg.target_ctx))) + abs(
            int(cfg.inf) - int(max(1, self.cfg.target_inf))
        )

    @staticmethod
    def _is_strict_descend(cfg: ConfigX, x_prev: ConfigX) -> bool:
        c_ctx = int(cfg.ctx)
        c_inf = int(cfg.inf)
        p_ctx = int(x_prev.ctx)
        p_inf = int(x_prev.inf)
        return c_ctx <= p_ctx and c_inf <= p_inf and (c_ctx < p_ctx or c_inf < p_inf)

    @staticmethod
    def _is_strict_ascend(cfg: ConfigX, x_prev: ConfigX) -> bool:
        c_ctx = int(cfg.ctx)
        c_inf = int(cfg.inf)
        p_ctx = int(x_prev.ctx)
        p_inf = int(x_prev.inf)
        return c_ctx >= p_ctx and c_inf >= p_inf and (c_ctx > p_ctx or c_inf > p_inf)

    def _apply_directional_filter(
        self,
        *,
        pool: Sequence[ConfigX],
        x_prev: ConfigX,
        zone: str,
    ) -> Sequence[ConfigX]:
        if not bool(self.cfg.directional_enable) or len(pool) <= 1:
            return pool
        if zone == "large" and bool(self.cfg.prefer_large_descend):
            strict_desc = [cfg for cfg in pool if self._is_strict_descend(cfg, x_prev)]
            if strict_desc:
                return strict_desc
            toward = [
                cfg
                for cfg in pool
                if self._target_distance(cfg) < self._target_distance(x_prev)
            ]
            if toward:
                return toward
            non_up = [
                cfg
                for cfg in pool
                if int(cfg.ctx) <= int(x_prev.ctx) and int(cfg.inf) <= int(x_prev.inf)
            ]
            if non_up:
                return non_up
            return pool
        if zone == "small" and bool(self.cfg.prefer_small_ascend):
            strict_up = [cfg for cfg in pool if self._is_strict_ascend(cfg, x_prev)]
            if strict_up:
                return strict_up
            toward = [
                cfg
                for cfg in pool
                if self._target_distance(cfg) < self._target_distance(x_prev)
            ]
            if toward:
                return toward
            non_down = [
                cfg
                for cfg in pool
                if int(cfg.ctx) >= int(x_prev.ctx) and int(cfg.inf) >= int(x_prev.inf)
            ]
            if non_down:
                return non_down
            return pool
        toward = [
            cfg for cfg in pool if self._target_distance(cfg) < self._target_distance(x_prev)
        ]
        if toward:
            return toward
        return pool

    def _direction_rank(self, *, cfg: ConfigX, x_prev: ConfigX, zone: str) -> tuple:
        target_delta = self._target_distance(x_prev) - self._target_distance(cfg)
        if zone == "large" and bool(self.cfg.prefer_large_descend):
            if self._is_strict_descend(cfg, x_prev):
                movement_rank = 0
            elif int(cfg.ctx) < int(x_prev.ctx) or int(cfg.inf) < int(x_prev.inf):
                movement_rank = 1
            else:
                movement_rank = 2
            return (
                movement_rank,
                0 if target_delta > 0 else 1,
                -float(target_delta),
            )
        if zone == "small" and bool(self.cfg.prefer_small_ascend):
            if self._is_strict_ascend(cfg, x_prev):
                movement_rank = 0
            elif int(cfg.ctx) > int(x_prev.ctx) or int(cfg.inf) > int(x_prev.inf):
                movement_rank = 1
            else:
                movement_rank = 2
            return (
                movement_rank,
                0 if target_delta > 0 else 1,
                -float(target_delta),
            )
        return (
            0 if target_delta > 0 else 1,
            float(self._target_distance(cfg)),
        )

    def _direction_bonus(self, *, cfg: ConfigX, x_prev: ConfigX, zone: str) -> float:
        target_delta = float(self._target_distance(x_prev) - self._target_distance(cfg))
        if zone == "large" and bool(self.cfg.prefer_large_descend):
            if self._is_strict_descend(cfg, x_prev):
                return 3.0 + target_delta
            if int(cfg.ctx) < int(x_prev.ctx) or int(cfg.inf) < int(x_prev.inf):
                return 2.0 + target_delta
            return 0.0 + target_delta
        if zone == "small" and bool(self.cfg.prefer_small_ascend):
            if self._is_strict_ascend(cfg, x_prev):
                return 3.0 + target_delta
            if int(cfg.ctx) > int(x_prev.ctx) or int(cfg.inf) > int(x_prev.inf):
                return 2.0 + target_delta
            return 0.0 + target_delta
        return target_delta
