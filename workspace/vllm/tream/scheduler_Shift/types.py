from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

MetricDict = Dict[str, float]
MarginDict = Dict[str, float]
RegimeId = str


@dataclass
class ConfigX:
    """Scheduler configuration point."""

    ctx: int
    inf: int
    ib: Optional[int] = None
    tb: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ctx": int(self.ctx),
            "inf": int(self.inf),
            "ib": None if self.ib is None else int(self.ib),
            "tb": None if self.tb is None else int(self.tb),
        }
        out.update(dict(self.extras))
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConfigX":
        known = {"ctx", "inf", "ib", "tb", "extras"}
        extras = dict(payload.get("extras", {}))
        for key, value in payload.items():
            if key not in known:
                extras[key] = value
        return cls(
            ctx=int(payload["ctx"]),
            inf=int(payload["inf"]),
            ib=None if payload.get("ib") is None else int(payload["ib"]),
            tb=None if payload.get("tb") is None else int(payload["tb"]),
            extras=extras,
        )

    def hash_key(self) -> str:
        blob = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
        return f"cfg:{self.ctx}:{self.inf}:{self.ib}:{self.tb}:{digest}"


@dataclass
class WindowMetrics:
    """Per-window runtime metrics consumed by the scheduler."""

    lat_mean: float
    acc_mean: float
    lat_p95: Optional[float] = None
    train_tps: Optional[float] = None
    mem_peak: Optional[float] = None

    token_drift_mean: Optional[float] = None
    jsd_mean: Optional[float] = None

    spec_accept_mean: Optional[float] = None
    spec_reverify_per_step: Optional[float] = None
    spec_draft_ms_per_step: Optional[float] = None
    spec_verify_ms_per_step: Optional[float] = None
    rejected_tokens_per_step: Optional[float] = None
    accepted_tokens_per_step: Optional[float] = None

    source: Dict[str, Any] = field(default_factory=dict)

    def drift_value(self) -> Optional[float]:
        if self.jsd_mean is not None:
            return float(self.jsd_mean)
        if self.token_drift_mean is not None:
            return float(self.token_drift_mean)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WindowMetrics":
        return cls(**dict(payload))


@dataclass
class DerivedMetrics:
    verify_ratio: Optional[float] = None
    waste_rate: Optional[float] = None

    quality_margin: Optional[float] = None
    latency_margin: Optional[float] = None
    mem_margin: Optional[float] = None
    train_tps_margin: Optional[float] = None

    shock_drift: float = 0.0
    shock_accept: float = 0.0
    shock_reverify: float = 0.0
    shock_verify_ratio: float = 0.0
    shock_waste_rate: float = 0.0

    @classmethod
    def from_window(
        cls,
        metrics: WindowMetrics,
        *,
        tau: Optional[float] = None,
        sla: Optional[float] = None,
        mem_limit: Optional[float] = None,
        train_min: Optional[float] = None,
    ) -> "DerivedMetrics":
        verify_ratio: Optional[float] = None
        if (
            metrics.spec_draft_ms_per_step is not None
            and metrics.spec_verify_ms_per_step is not None
        ):
            total = metrics.spec_draft_ms_per_step + metrics.spec_verify_ms_per_step
            if total > 0:
                verify_ratio = float(metrics.spec_verify_ms_per_step) / float(total)

        waste_rate: Optional[float] = None
        if (
            metrics.accepted_tokens_per_step is not None
            and metrics.rejected_tokens_per_step is not None
        ):
            total = metrics.accepted_tokens_per_step + metrics.rejected_tokens_per_step
            if total > 0:
                waste_rate = float(metrics.rejected_tokens_per_step) / float(total)
        elif metrics.spec_accept_mean is not None:
            waste_rate = max(0.0, 1.0 - float(metrics.spec_accept_mean))

        quality_margin = None if tau is None else float(metrics.acc_mean) - float(tau)
        latency_ref = metrics.lat_p95 if metrics.lat_p95 is not None else metrics.lat_mean
        latency_margin = None if sla is None else float(sla) - float(latency_ref)
        mem_margin = (
            None
            if mem_limit is None or metrics.mem_peak is None
            else float(mem_limit) - float(metrics.mem_peak)
        )
        train_tps_margin = (
            None
            if train_min is None or metrics.train_tps is None
            else float(metrics.train_tps) - float(train_min)
        )

        return cls(
            verify_ratio=verify_ratio,
            waste_rate=waste_rate,
            quality_margin=quality_margin,
            latency_margin=latency_margin,
            mem_margin=mem_margin,
            train_tps_margin=train_tps_margin,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    x_next: ConfigX
    reason: str
    predicted_gains: MetricDict = field(default_factory=dict)
    safety_margins: MarginDict = field(default_factory=dict)
    regime_id: Optional[RegimeId] = None
    mode: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_next": self.x_next.to_dict(),
            "reason": self.reason,
            "predicted_gains": dict(self.predicted_gains),
            "safety_margins": dict(self.safety_margins),
            "regime_id": self.regime_id,
            "mode": self.mode,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        return cls(
            x_next=ConfigX.from_dict(payload["x_next"]),
            reason=str(payload["reason"]),
            predicted_gains=dict(payload.get("predicted_gains", {})),
            safety_margins=dict(payload.get("safety_margins", {})),
            regime_id=payload.get("regime_id"),
            mode=payload.get("mode"),
            meta=dict(payload.get("meta", {})),
        )
