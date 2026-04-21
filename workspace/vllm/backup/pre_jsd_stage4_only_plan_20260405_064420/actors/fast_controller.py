from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class HardLimits:
    """Hard constraints (SLO/SLA) for the fast loop to enforce."""

    latency_p95_s: float
    staleness_p95: Optional[float] = None
    mem_peak_gb: Optional[float] = None
    allow_oom: bool = False


@dataclass
class Bounds:
    train_ctx: Tuple[int, int]
    infer_len: Tuple[int, int]
    batch_size: Optional[Tuple[int, int]] = None


@dataclass
class Knobs:
    train_ctx: int
    infer_len: int
    batch_size: Optional[int] = None
    apply_check_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    knobs: Knobs
    mode: str
    reason: str
    override: bool
    virtual_targets: Dict[str, Optional[float]] = field(default_factory=dict)


class FastLoopController:
    """
    Fast loop (safety controller): enforce hard limits with virtual goals, and
    override the slow loop (SCOUT) when in danger.

    Design intent:
    - Use virtual goals (a safety cushion) to preemptively brake to avoid oscillations.
    - In DANGER, make a single conservative move per tick (prefer infer_len, then ctx).
    - In SAFE, do not interfere (override=False).
    """

    def __init__(
        self,
        limits: HardLimits,
        bounds: Bounds,
        *,
        vg_latency: float = 0.90,
        vg_stale: float = 0.90,
        vg_mem: float = 0.95,
        step_i: int = 1,
        step_c: int = 1,
        step_b: int = 1,
        safe_k: int = 3,
        oom_cooldown: int = 5,
    ):
        self.limits = limits
        self.bounds = bounds

        self.vg_latency = vg_latency
        self.vg_stale = vg_stale
        self.vg_mem = vg_mem

        self.step_i = step_i
        self.step_c = step_c
        self.step_b = step_b

        self.safe_k = safe_k
        self._safe_streak = 0
        self._last_safe: Optional[Knobs] = None

        self._oom_cooldown = 0
        self._oom_cooldown_len = oom_cooldown

    def _clip(self, k: Knobs) -> Knobs:
        min_c, max_c = self.bounds.train_ctx
        min_i, max_i = self.bounds.infer_len
        k.train_ctx = max(min_c, min(max_c, int(k.train_ctx)))
        k.infer_len = max(min_i, min(max_i, int(k.infer_len)))

        if k.batch_size is not None and self.bounds.batch_size is not None:
            min_b, max_b = self.bounds.batch_size
            k.batch_size = max(min_b, min(max_b, int(k.batch_size)))

        if k.apply_check_s is not None:
            k.apply_check_s = max(0.02, float(k.apply_check_s))

        return k

    def _virtual_targets(self) -> Dict[str, Optional[float]]:
        return {
            "L": self.vg_latency * self.limits.latency_p95_s,
            "S": None
            if self.limits.staleness_p95 is None
            else self.vg_stale * self.limits.staleness_p95,
            "M": None
            if self.limits.mem_peak_gb is None
            else self.vg_mem * self.limits.mem_peak_gb,
        }

    def step(self, metrics: Dict[str, Any], current: Knobs) -> Decision:
        k = self._clip(Knobs(**current.__dict__))
        vt = self._virtual_targets()

        L = metrics.get("latency_p95", None)
        S = metrics.get("staleness_p95", None)
        M = metrics.get("mem_peak_gb", None)
        oom = bool(metrics.get("oom_flag", False)) or (int(metrics.get("oom_count", 0)) > 0)

        if oom and not self.limits.allow_oom:
            self._oom_cooldown = self._oom_cooldown_len

        danger = False
        if vt["L"] is not None and L is not None and float(L) > float(vt["L"]):
            danger = True
        if vt["S"] is not None and S is not None and float(S) > float(vt["S"]):
            danger = True
        if vt["M"] is not None and M is not None and float(M) > float(vt["M"]):
            danger = True
        if self._oom_cooldown > 0:
            danger = True

        if danger:
            self._safe_streak = 0

            # OOM/memory danger: batch_size -> ctx -> infer_len.
            if (self._oom_cooldown > 0) or (vt["M"] is not None and M is not None and float(M) > float(vt["M"])):
                if k.batch_size is not None and self.bounds.batch_size is not None:
                    if k.batch_size > self.bounds.batch_size[0]:
                        k.batch_size -= self.step_b
                        self._oom_cooldown = max(0, self._oom_cooldown - 1)
                        return Decision(self._clip(k), "DANGER", "mem/oom_high_dec_batch", True, vt)

                if k.train_ctx > self.bounds.train_ctx[0]:
                    k.train_ctx -= self.step_c
                    self._oom_cooldown = max(0, self._oom_cooldown - 1)
                    return Decision(self._clip(k), "DANGER", "mem/oom_high_dec_ctx", True, vt)

                if k.infer_len > self.bounds.infer_len[0]:
                    k.infer_len -= self.step_i
                    self._oom_cooldown = max(0, self._oom_cooldown - 1)
                    return Decision(self._clip(k), "DANGER", "mem/oom_high_dec_infer", True, vt)

            # Latency danger: decrease infer_len first.
            if vt["L"] is not None and L is not None and float(L) > float(vt["L"]):
                if k.infer_len > self.bounds.infer_len[0]:
                    k.infer_len -= self.step_i
                    return Decision(self._clip(k), "DANGER", "latency_high_dec_infer", True, vt)
                if k.train_ctx > self.bounds.train_ctx[0]:
                    k.train_ctx -= self.step_c
                    return Decision(self._clip(k), "DANGER", "latency_high_dec_ctx", True, vt)

            # Staleness danger: prefer apply frequency if available, else ctx.
            if vt["S"] is not None and S is not None and float(S) > float(vt["S"]):
                if k.apply_check_s is not None:
                    k.apply_check_s = max(0.02, k.apply_check_s * 0.5)
                    return Decision(self._clip(k), "DANGER", "stale_high_inc_apply_freq", True, vt)
                if k.train_ctx > self.bounds.train_ctx[0]:
                    k.train_ctx -= self.step_c
                    return Decision(self._clip(k), "DANGER", "stale_high_dec_ctx", True, vt)

            if self._last_safe is not None:
                return Decision(self._last_safe, "DANGER", "fallback_last_safe", True, vt)
            return Decision(self._clip(k), "DANGER", "fallback_hold", True, vt)

        # SAFE
        self._safe_streak += 1
        if self._safe_streak >= self.safe_k:
            self._last_safe = self._clip(k)
        if self._oom_cooldown > 0:
            self._oom_cooldown -= 1
        return Decision(self._clip(k), "SAFE", "hold", False, vt)

