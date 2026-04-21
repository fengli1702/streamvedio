from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

from ..config_space import ConfigSpace
from ..stats import StatsStore
from ..types import ConfigX, Decision, DerivedMetrics


@dataclass
class SchedulerContext:
    mode: str
    regime_id: str
    x_prev: ConfigX
    safe_set: Sequence[ConfigX]
    pareto_set: Sequence[ConfigX]
    anchors: Sequence[ConfigX]
    stats_store: StatsStore
    config_space: ConfigSpace

    preference: str = "quality"
    switch_cost_lambda: float = 0.0
    dwell_state: Dict[str, int] = field(default_factory=dict)

    derived: Optional[DerivedMetrics] = None
    history: Sequence[Mapping[str, object]] = field(default_factory=list)
    alarm_severity: float = 0.0

    allow_uncertainty_exploration: bool = False
    stable_eps_fast: int = 1
    stable_eps_slow: int = 0
    adapt_eps_fast: int = 2
    adapt_eps_slow: int = 1


class DecisionPolicy(ABC):
    @abstractmethod
    def select_next(self, ctx: SchedulerContext) -> Decision:
        raise NotImplementedError

