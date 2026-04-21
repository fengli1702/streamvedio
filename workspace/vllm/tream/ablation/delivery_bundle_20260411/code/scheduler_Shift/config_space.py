from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .types import ConfigX

IntOrPair = Union[int, Tuple[int, int]]


def _sorted_unique(values: Iterable[int]) -> List[int]:
    uniq = sorted({int(v) for v in values})
    if not uniq:
        raise ValueError("Config dimension cannot be empty")
    return uniq


class ConfigSpace:
    """Discrete configuration space for fast knobs (ctx/inf) and optional slow knobs (ib/tb)."""

    def __init__(
        self,
        *,
        ctx_values: Sequence[int],
        inf_values: Sequence[int],
        ib_values: Optional[Sequence[int]] = None,
        tb_values: Optional[Sequence[int]] = None,
        max_all_configs: int = 4096,
    ) -> None:
        self.ctx_values = _sorted_unique(ctx_values)
        self.inf_values = _sorted_unique(inf_values)
        self.ib_values = None if ib_values is None else _sorted_unique(ib_values)
        self.tb_values = None if tb_values is None else _sorted_unique(tb_values)
        self.max_all_configs = int(max_all_configs)

    def all_configs(self) -> List[ConfigX]:
        ib_vals = self.ib_values if self.ib_values is not None else [None]
        tb_vals = self.tb_values if self.tb_values is not None else [None]
        total = len(self.ctx_values) * len(self.inf_values) * len(ib_vals) * len(tb_vals)
        if total > self.max_all_configs:
            raise ValueError(
                f"Config space too large to enumerate: {total} > {self.max_all_configs}"
            )

        out: List[ConfigX] = []
        for ctx, inf, ib, tb in product(self.ctx_values, self.inf_values, ib_vals, tb_vals):
            out.append(ConfigX(ctx=ctx, inf=inf, ib=ib, tb=tb))
        return out

    def neighbors(
        self,
        x: ConfigX,
        eps_fast: IntOrPair = 1,
        eps_slow: IntOrPair = 1,
    ) -> List[ConfigX]:
        fast_ctx, fast_inf = self._parse_pair(eps_fast)
        slow_ib, slow_tb = self._parse_pair(eps_slow)
        base = self._canonicalize(x)

        ctx_pool = [base.ctx] + self._neighbor_values(self.ctx_values, base.ctx, fast_ctx)
        inf_pool = [base.inf] + self._neighbor_values(self.inf_values, base.inf, fast_inf)

        if self.ib_values is None:
            ib_pool: List[Optional[int]] = [None]
        else:
            base_ib = self._optional_base(self.ib_values, base.ib)
            ib_pool = [base_ib] + self._neighbor_values(self.ib_values, base_ib, slow_ib)

        if self.tb_values is None:
            tb_pool: List[Optional[int]] = [None]
        else:
            base_tb = self._optional_base(self.tb_values, base.tb)
            tb_pool = [base_tb] + self._neighbor_values(self.tb_values, base_tb, slow_tb)

        out: List[ConfigX] = []
        base_key = base.hash_key()
        seen = {base_key}
        for ctx, inf, ib, tb in product(ctx_pool, inf_pool, ib_pool, tb_pool):
            cand = ConfigX(ctx=ctx, inf=inf, ib=ib, tb=tb, extras=dict(base.extras))
            key = cand.hash_key()
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        return out

    def orthogonal_probes(
        self,
        x: ConfigX,
        deltas: Optional[Dict[str, int]] = None,
    ) -> List[ConfigX]:
        d = {"ctx": 1, "inf": 1}
        if deltas:
            d.update(deltas)
        cross = bool(d.get("cross", False))
        base = self._canonicalize(x)
        out: List[ConfigX] = []
        seen = {base.hash_key()}

        def _add(cand: ConfigX) -> None:
            key = cand.hash_key()
            if key in seen:
                return
            seen.add(key)
            out.append(cand)

        for sign in (-1, 1):
            ctx_next = self._move_by_delta(self.ctx_values, base.ctx, sign * abs(int(d["ctx"])))
            _add(ConfigX(ctx=ctx_next, inf=base.inf, ib=base.ib, tb=base.tb, extras=dict(base.extras)))
            inf_next = self._move_by_delta(self.inf_values, base.inf, sign * abs(int(d["inf"])))
            _add(ConfigX(ctx=base.ctx, inf=inf_next, ib=base.ib, tb=base.tb, extras=dict(base.extras)))

        if self.ib_values is not None and "ib" in d:
            for sign in (-1, 1):
                ib_next = self._move_by_delta(
                    self.ib_values,
                    self._optional_base(self.ib_values, base.ib),
                    sign * abs(int(d["ib"])),
                )
                _add(ConfigX(ctx=base.ctx, inf=base.inf, ib=ib_next, tb=base.tb, extras=dict(base.extras)))

        if self.tb_values is not None and "tb" in d:
            for sign in (-1, 1):
                tb_next = self._move_by_delta(
                    self.tb_values,
                    self._optional_base(self.tb_values, base.tb),
                    sign * abs(int(d["tb"])),
                )
                _add(ConfigX(ctx=base.ctx, inf=base.inf, ib=base.ib, tb=tb_next, extras=dict(base.extras)))

        if cross:
            ctx_down = self._move_by_delta(self.ctx_values, base.ctx, -abs(int(d["ctx"])))
            ctx_up = self._move_by_delta(self.ctx_values, base.ctx, abs(int(d["ctx"])))
            inf_down = self._move_by_delta(self.inf_values, base.inf, -abs(int(d["inf"])))
            inf_up = self._move_by_delta(self.inf_values, base.inf, abs(int(d["inf"])))
            for c in (ctx_down, ctx_up):
                for i in (inf_down, inf_up):
                    _add(ConfigX(ctx=c, inf=i, ib=base.ib, tb=base.tb, extras=dict(base.extras)))

        return out

    @staticmethod
    def _parse_pair(raw: IntOrPair) -> Tuple[int, int]:
        if isinstance(raw, tuple):
            return max(0, int(raw[0])), max(0, int(raw[1]))
        value = max(0, int(raw))
        return value, value

    @staticmethod
    def _index_of_or_nearest(values: Sequence[int], target: int) -> int:
        target = int(target)
        try:
            return values.index(target)
        except ValueError:
            return min(range(len(values)), key=lambda idx: abs(values[idx] - target))

    def _neighbor_values(self, values: Sequence[int], center: int, radius: int) -> List[int]:
        if radius <= 0:
            return []
        idx = self._index_of_or_nearest(values, center)
        lo = max(0, idx - radius)
        hi = min(len(values) - 1, idx + radius)
        return [values[i] for i in range(lo, hi + 1) if values[i] != center]

    @staticmethod
    def _optional_base(values: Sequence[int], current: Optional[int]) -> int:
        if current is None:
            return int(values[0])
        return int(current)

    def _canonicalize(self, x: ConfigX) -> ConfigX:
        ctx = self.ctx_values[self._index_of_or_nearest(self.ctx_values, x.ctx)]
        inf = self.inf_values[self._index_of_or_nearest(self.inf_values, x.inf)]
        ib = x.ib
        tb = x.tb
        if self.ib_values is not None:
            base_ib = self._optional_base(self.ib_values, ib)
            ib = self.ib_values[self._index_of_or_nearest(self.ib_values, base_ib)]
        else:
            ib = None
        if self.tb_values is not None:
            base_tb = self._optional_base(self.tb_values, tb)
            tb = self.tb_values[self._index_of_or_nearest(self.tb_values, base_tb)]
        else:
            tb = None
        return ConfigX(ctx=ctx, inf=inf, ib=ib, tb=tb, extras=dict(x.extras))

    def _move_by_delta(self, values: Sequence[int], current: int, delta: int) -> int:
        current = int(values[self._index_of_or_nearest(values, current)])
        if delta == 0:
            return current
        target = current + int(delta)
        if delta > 0:
            larger = [v for v in values if v > current]
            if not larger:
                return current
            return min(larger, key=lambda v: (abs(v - target), v))
        smaller = [v for v in values if v < current]
        if not smaller:
            return current
        return min(smaller, key=lambda v: (abs(v - target), -v))

