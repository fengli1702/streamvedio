from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .config_space import ConfigSpace
from .types import ConfigX


class CandidateGenerator:
    """Generate candidate set C_t from anchors, local neighbors and adapt probes."""

    def __init__(
        self,
        config_space: ConfigSpace,
        *,
        cap_size: int = 64,
        default_eps_fast: int = 1,
        default_eps_slow: int = 1,
    ) -> None:
        self.config_space = config_space
        self.cap_size = int(cap_size)
        self.default_eps_fast = int(default_eps_fast)
        self.default_eps_slow = int(default_eps_slow)

    def generate(
        self,
        *,
        x_prev: ConfigX,
        anchors: Sequence[ConfigX],
        mode: str,
        eps_fast: Optional[int] = None,
        eps_slow: Optional[int] = None,
        probe_deltas: Optional[Dict[str, int]] = None,
        safe_default: Optional[ConfigX] = None,
        min_probe_keep: int = 0,
    ) -> List[ConfigX]:
        e_fast = self.default_eps_fast if eps_fast is None else int(eps_fast)
        e_slow = self.default_eps_slow if eps_slow is None else int(eps_slow)
        mode_u = str(mode).upper()

        probes: List[ConfigX] = []
        if mode_u == "ADAPT":
            probes = self.config_space.orthogonal_probes(x_prev, deltas=probe_deltas)

        out: List[ConfigX] = []
        seen = set()

        def _push_many(pool: Iterable[ConfigX]) -> None:
            if self.cap_size > 0 and len(out) >= self.cap_size:
                return
            for cfg in pool:
                if self.cap_size > 0 and len(out) >= self.cap_size:
                    return
                key = cfg.hash_key()
                if key in seen:
                    continue
                seen.add(key)
                out.append(cfg)
                if self.cap_size > 0 and len(out) >= self.cap_size:
                    return

        _push_many([x_prev])
        if mode_u == "ADAPT" and min_probe_keep > 0:
            _push_many(probes[: int(max(0, min_probe_keep))])
        _push_many(anchors)
        _push_many(self.config_space.neighbors(x_prev, eps_fast=e_fast, eps_slow=e_slow))
        if safe_default is not None:
            _push_many([safe_default])
        if mode_u == "ADAPT":
            _push_many(probes)
        return out
