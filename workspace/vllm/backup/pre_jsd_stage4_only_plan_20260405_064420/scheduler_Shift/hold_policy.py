from __future__ import annotations

from typing import Optional, Sequence

from .policies.anchor_only import AnchorOnlyPolicy
from .policies.base import SchedulerContext
from .types import ConfigX, Decision


class HoldAnchorPolicy:
    """HOLD stage policy: protect anchor first, only allow local/protective micro-adjustments."""

    def __init__(self) -> None:
        self._anchor_only = AnchorOnlyPolicy(allow_adapt_explore=False)

    def select_next(
        self,
        *,
        ctx: SchedulerContext,
        anchor_cfg: Optional[ConfigX],
        anchor_protected_pool: Sequence[ConfigX],
    ) -> Decision:
        pool = list(anchor_protected_pool) if anchor_protected_pool else list(ctx.safe_set)
        if not pool:
            pool = [ctx.x_prev]

        if anchor_cfg is not None:
            anchor_key = anchor_cfg.hash_key()
            for cand in pool:
                if cand.hash_key() == anchor_key:
                    return Decision(
                        x_next=cand,
                        reason="hold_anchor",
                        mode=ctx.mode,
                        regime_id=ctx.regime_id,
                    )

        # Fallback: score only inside anchor-protected pool.
        hold_ctx = SchedulerContext(
            mode=ctx.mode,
            regime_id=ctx.regime_id,
            x_prev=ctx.x_prev,
            safe_set=pool,
            pareto_set=ctx.pareto_set,
            anchors=ctx.anchors,
            stats_store=ctx.stats_store,
            config_space=ctx.config_space,
            preference=ctx.preference,
            switch_cost_lambda=max(float(ctx.switch_cost_lambda), 0.3),
            dwell_state=ctx.dwell_state,
            derived=ctx.derived,
            history=ctx.history,
            alarm_severity=ctx.alarm_severity,
            allow_uncertainty_exploration=False,
            stable_eps_fast=ctx.stable_eps_fast,
            stable_eps_slow=ctx.stable_eps_slow,
            adapt_eps_fast=ctx.adapt_eps_fast,
            adapt_eps_slow=ctx.adapt_eps_slow,
        )
        decision = self._anchor_only.select_next(hold_ctx)
        decision.reason = "hold_anchor_score"
        return decision
