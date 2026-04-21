from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..types import ConfigX, Decision
from .base import DecisionPolicy, SchedulerContext


def _switch_cost(a: ConfigX, b: ConfigX) -> float:
    cost = abs(int(a.ctx) - int(b.ctx)) + abs(int(a.inf) - int(b.inf))
    if a.ib != b.ib:
        cost += 2.0
    if a.tb != b.tb:
        cost += 2.0
    return float(cost)


class AnchorOnlyPolicy(DecisionPolicy):
    def __init__(
        self,
        *,
        allow_adapt_explore: bool = True,
    ) -> None:
        self.allow_adapt_explore = bool(allow_adapt_explore)

    def select_next(self, ctx: SchedulerContext) -> Decision:
        safe = list(ctx.safe_set) if ctx.safe_set else [ctx.x_prev]
        mode = str(ctx.mode).upper()

        if mode == "STABLE":
            # Dwell gate to avoid thrashing in stable regime.
            if int(ctx.dwell_state.get("remaining", 0)) > 0 and self._contains(safe, ctx.x_prev):
                return Decision(
                    x_next=ctx.x_prev,
                    reason="stable_dwell_hold",
                    mode=mode,
                    regime_id=ctx.regime_id,
                )

            reach = ctx.config_space.neighbors(
                ctx.x_prev,
                eps_fast=ctx.stable_eps_fast,
                eps_slow=ctx.stable_eps_slow,
            )
            allowed = {ctx.x_prev.hash_key()} | {x.hash_key() for x in reach}
            safe = [x for x in safe if x.hash_key() in allowed] or [ctx.x_prev]
            ctx.dwell_state["adapt_probe_used"] = 0

        elif mode == "ADAPT" and self.allow_adapt_explore and ctx.allow_uncertainty_exploration:
            if int(ctx.dwell_state.get("adapt_probe_used", 0)) == 0:
                probe = self._pick_max_uncertainty(ctx, safe)
                if probe is not None and probe.hash_key() != ctx.x_prev.hash_key():
                    ctx.dwell_state["adapt_probe_used"] = 1
                    return Decision(
                        x_next=probe,
                        reason="adapt_uncertainty_probe",
                        mode=mode,
                        regime_id=ctx.regime_id,
                    )

        best = None
        best_score = None
        best_mu: Tuple[float, float] = (0.0, 0.0)
        prev_mu = self._mu_pair(ctx, ctx.x_prev)
        for cand in safe:
            cand_mu = self._mu_pair(ctx, cand)
            pref_score = self._preference_score(ctx.preference, cand_mu)
            score = pref_score + float(ctx.switch_cost_lambda) * _switch_cost(cand, ctx.x_prev)
            if best is None or score < best_score:
                best = cand
                best_score = score
                best_mu = cand_mu

        assert best is not None
        return Decision(
            x_next=best,
            reason="anchor_only_score",
            mode=mode,
            regime_id=ctx.regime_id,
            predicted_gains={
                "latency_gain": prev_mu[0] - best_mu[0],
                "quality_gain": best_mu[1] - prev_mu[1],
            },
        )

    @staticmethod
    def _contains(cands: Sequence[ConfigX], target: ConfigX) -> bool:
        key = target.hash_key()
        return any(c.hash_key() == key for c in cands)

    @staticmethod
    def _preference_score(preference: str, mu_pair: Tuple[float, float]) -> float:
        lat, quality = mu_pair
        pref = str(preference).lower()
        if pref in ("latency", "speed", "throughput"):
            return lat - 0.05 * quality
        if pref in ("knee",):
            return lat - quality
        return -quality + 0.05 * lat

    @staticmethod
    def _mu_pair(ctx: SchedulerContext, cfg: ConfigX) -> Tuple[float, float]:
        lat_mu, _ = ctx.stats_store.get_mu_sigma(ctx.regime_id, cfg, "lat_mean")
        q_mu, _ = ctx.stats_store.get_mu_sigma(ctx.regime_id, cfg, "acc_mean")
        lat = float("inf") if lat_mu is None else float(lat_mu)
        quality = 0.0 if q_mu is None else float(q_mu)
        return lat, quality

    @staticmethod
    def _pick_max_uncertainty(ctx: SchedulerContext, safe: Sequence[ConfigX]) -> Optional[ConfigX]:
        if not safe:
            return None
        prev_key = ctx.x_prev.hash_key()
        pareto_keys = {cfg.hash_key() for cfg in ctx.pareto_set}
        anchor_keys = {cfg.hash_key() for cfg in ctx.anchors}

        # Exploration should move locally first; only fallback to wider points when needed.
        local_reach = ctx.config_space.neighbors(
            ctx.x_prev,
            eps_fast=max(1, int(ctx.adapt_eps_fast)),
            eps_slow=max(0, int(ctx.adapt_eps_slow)),
        )
        local_keys = {cfg.hash_key() for cfg in local_reach}

        local_pool = [
            cand
            for cand in safe
            if cand.hash_key() != prev_key and cand.hash_key() in local_keys
        ]
        frontier_pool = [
            cand
            for cand in safe
            if cand.hash_key() != prev_key
            and (cand.hash_key() in pareto_keys or cand.hash_key() in anchor_keys)
        ]
        global_pool = [cand for cand in safe if cand.hash_key() != prev_key]

        pool = local_pool or frontier_pool or global_pool
        if not pool:
            return None

        best = None
        best_tuple = None
        for cand in pool:
            _, lat_sigma = ctx.stats_store.get_mu_sigma(ctx.regime_id, cand, "lat_mean")
            _, q_sigma = ctx.stats_store.get_mu_sigma(ctx.regime_id, cand, "acc_mean")
            u = (0.0 if lat_sigma is None else float(lat_sigma)) + (
                0.0 if q_sigma is None else float(q_sigma)
            )

            key = cand.hash_key()
            frontier_bonus = 0.0
            if key in pareto_keys:
                frontier_bonus += 0.10
            if key in anchor_keys:
                frontier_bonus += 0.05
            local_bonus = 0.10 if key in local_keys else 0.0
            score = u + frontier_bonus + local_bonus

            tup = (
                score,
                -_switch_cost(cand, ctx.x_prev),  # prefer shorter step in tie
                -int(cand.ctx),
                -int(cand.inf),
            )
            if best is None or tup > best_tuple:
                best = cand
                best_tuple = tup
        return best
