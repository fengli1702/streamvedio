from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .types import ConfigX


def _dominates(a: Sequence[float], b: Sequence[float], minimize_mask: Sequence[bool]) -> bool:
    better_or_equal = True
    strictly_better = False
    for idx, minimize in enumerate(minimize_mask):
        av = float(a[idx])
        bv = float(b[idx])
        if minimize:
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
    return better_or_equal and strictly_better


def pareto_filter(
    points: Sequence[Sequence[float]],
    minimize_mask: Sequence[bool],
) -> List[Tuple[float, ...]]:
    """Return the non-dominated subset of points."""
    out: List[Tuple[float, ...]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if _dominates(q, p, minimize_mask):
                dominated = True
                break
        if not dominated:
            out.append(tuple(float(v) for v in p))
    return out


@dataclass
class AnchorManager:
    anchors: Dict[str, List[ConfigX]]

    def __init__(self) -> None:
        self.anchors = {}

    def update_anchors(
        self,
        candidates: Sequence[ConfigX],
        mu_vectors: Mapping[str, Mapping[str, float]],
        k: int,
        *,
        regime_id: Optional[str] = None,
    ) -> List[ConfigX]:
        if k <= 0:
            return []
        scored = self._collect(candidates, mu_vectors)
        if not scored:
            return []
        pareto_items = self._pareto_items(scored)
        if not pareto_items:
            return []

        selected: List[ConfigX] = []
        seen = set()

        def _pick(cfg: ConfigX) -> None:
            key = cfg.hash_key()
            if key in seen or len(selected) >= k:
                return
            seen.add(key)
            selected.append(cfg)

        # Endpoints: min latency, max quality, max train_tps.
        min_lat = min(pareto_items, key=lambda x: x["lat"])["cfg"]
        max_quality = max(pareto_items, key=lambda x: x["quality"])["cfg"]
        max_tps = max(pareto_items, key=lambda x: x["train_tps"])["cfg"]
        _pick(min_lat)
        _pick(max_quality)
        _pick(max_tps)

        # Knee / diversity points using crowding distance proxy.
        if len(selected) < k:
            by_crowding = sorted(
                pareto_items,
                key=lambda item: self._crowding_score(item, pareto_items),
                reverse=True,
            )
            for item in by_crowding:
                _pick(item["cfg"])
                if len(selected) >= k:
                    break

        # Fill remaining anchors by latency buckets for coverage.
        if len(selected) < k:
            pareto_sorted = sorted(pareto_items, key=lambda x: x["lat"])
            bucket_count = max(1, min(k, len(pareto_sorted)))
            for b in range(bucket_count):
                start = (b * len(pareto_sorted)) // bucket_count
                end = ((b + 1) * len(pareto_sorted)) // bucket_count
                bucket = pareto_sorted[start:end] or [pareto_sorted[min(start, len(pareto_sorted) - 1)]]
                _pick(bucket[0]["cfg"])
                if len(selected) >= k:
                    break

        if regime_id is not None:
            self.anchors[str(regime_id)] = list(selected)
        return selected

    @staticmethod
    def _collect(
        candidates: Sequence[ConfigX],
        mu_vectors: Mapping[str, Mapping[str, float]],
    ) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for cfg in candidates:
            mu = mu_vectors.get(cfg.hash_key())
            if not mu:
                continue
            lat = float(mu.get("lat_mean", mu.get("latency", 0.0)))
            quality = float(mu.get("acc_mean", mu.get("quality", 0.0)))
            train_tps = float(mu.get("train_tps", mu.get("train_tps_mean", 0.0)))
            out.append(
                {
                    "cfg": cfg,
                    "lat": lat,
                    "quality": quality,
                    "train_tps": train_tps,
                }
            )
        return out

    @staticmethod
    def _pareto_items(items: Sequence[Mapping[str, float]]) -> List[Mapping[str, float]]:
        vectors = [(it["lat"], it["quality"], it["train_tps"]) for it in items]
        keep = pareto_filter(vectors, minimize_mask=(True, False, False))
        keep_set = set(keep)
        out = []
        for item in items:
            vec = (float(item["lat"]), float(item["quality"]), float(item["train_tps"]))
            if vec in keep_set:
                out.append(item)
        return out

    @staticmethod
    def _crowding_score(item: Mapping[str, float], items: Sequence[Mapping[str, float]]) -> float:
        lat_vals = [float(x["lat"]) for x in items]
        q_vals = [float(x["quality"]) for x in items]
        t_vals = [float(x["train_tps"]) for x in items]

        def _norm_span(vals: List[float]) -> float:
            lo = min(vals)
            hi = max(vals)
            return max(hi - lo, 1e-9)

        lat_span = _norm_span(lat_vals)
        q_span = _norm_span(q_vals)
        t_span = _norm_span(t_vals)
        return (
            abs(float(item["lat"]) - sum(lat_vals) / len(lat_vals)) / lat_span
            + abs(float(item["quality"]) - sum(q_vals) / len(q_vals)) / q_span
            + abs(float(item["train_tps"]) - sum(t_vals) / len(t_vals)) / t_span
        )

