"""
Constrained qNEHVI co-scheduler (train+infer, 3D Pareto, safe).

This rewrites the previous heuristic scheduler while keeping the public
interfaces intact: report_* metrics, config versioning, and actor updates.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
from scheduler_Shift.config_space import ConfigSpace
from scheduler_Shift.core import SchedulerConfig, SchedulerCore
from scheduler_Shift.types import ConfigX, WindowMetrics

try:  # Optional dependency
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _safe_p95(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.percentile(vals, 95)) if vals else None


def _safe_std(values: List[Optional[float]]) -> Optional[float]:
    vals = _finite_array(values)
    return float(np.std(vals)) if vals.size else None


def _finite_array(vals: List[Optional[float]]) -> np.ndarray:
    out = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return np.array(out, dtype=float)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    strictly_better = False
    for d in range(len(a)):
        if a[d] > b[d]:
            return False
        if a[d] < b[d]:
            strictly_better = True
    return strictly_better


def pareto_filter(points: List[np.ndarray]) -> List[np.ndarray]:
    if not points:
        return []
    filtered: List[np.ndarray] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and dominates(q, p):
                dominated = True
                break
        if not dominated:
            filtered.append(p)
    return filtered


def update_pareto(pareto: List[Dict[str, Any]], x: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    for item in pareto:
        if dominates(item["y"], y):
            return pareto
    new_p: List[Dict[str, Any]] = []
    for item in pareto:
        if not dominates(y, item["y"]):
            new_p.append(item)
    new_p.append({"x": x, "y": y})
    return new_p


def hypervolume_2d(points: List[np.ndarray], ref: np.ndarray) -> float:
    if not points:
        return 0.0
    points = [p for p in points if p[0] <= ref[0] and p[1] <= ref[1]]
    if not points:
        return 0.0
    points = pareto_filter(points)
    points.sort(key=lambda v: v[0])
    area = 0.0
    prev_z = ref[1]
    for y, z in points:
        if z >= prev_z:
            continue
        area += (ref[0] - y) * (prev_z - z)
        prev_z = z
    return max(area, 0.0)


def hypervolume_3d(points: List[np.ndarray], ref: np.ndarray) -> float:
    if not points:
        return 0.0
    points = [p for p in points if np.all(p <= ref)]
    if not points:
        return 0.0
    points = pareto_filter(points)
    points.sort(key=lambda v: v[0])
    hv = 0.0
    for i, p in enumerate(points):
        x_i = p[0]
        next_x = ref[0] if i == len(points) - 1 else points[i + 1][0]
        if next_x <= x_i:
            continue
        slice_points = [pt[1:] for pt in points[: i + 1]]
        area = hypervolume_2d(slice_points, ref[1:])
        hv += (next_x - x_i) * area
    return max(hv, 0.0)


class SimpleGP:
    """Minimal exact GP with RBF kernel and fixed hyperparameters."""

    def __init__(self, noise: float = 1e-4, jitter: float = 1e-6) -> None:
        self.noise = float(noise)
        self.jitter = float(jitter)
        self.lengthscale = 1.0
        self.signal_var = 1.0
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._chol: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _kernel(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        dists = np.sum((xa[:, None, :] - xb[None, :, :]) ** 2, axis=-1)
        return self.signal_var * np.exp(-0.5 * dists / (self.lengthscale ** 2))

    def _estimate_lengthscale(self, x: np.ndarray) -> float:
        if x.shape[0] < 2:
            return 1.0
        diffs = x[:, None, :] - x[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
        median = np.median(dists[dists > 0])
        return float(median) if np.isfinite(median) and median > 0 else 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_train = np.array(x, dtype=float)
        y = np.array(y, dtype=float).reshape(-1)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if np.std(y) > 1e-6 else 1.0
        self.y_train = (y - self.y_mean) / self.y_std
        self.lengthscale = self._estimate_lengthscale(self.x_train)
        self.signal_var = 1.0

        k = self._kernel(self.x_train, self.x_train)
        n = k.shape[0]
        diag = (self.noise ** 2 + self.jitter) * np.eye(n)
        k = k + diag

        for jitter in (0.0, 1e-6, 1e-5, 1e-4, 1e-3):
            try:
                k_j = k + jitter * np.eye(n)
                chol = np.linalg.cholesky(k_j)
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, self.y_train))
                self._chol = chol
                self._alpha = alpha
                return
            except np.linalg.LinAlgError:
                continue

        k_inv = np.linalg.pinv(k)
        self._chol = None
        self._alpha = k_inv @ self.y_train

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.x_train is None or self.y_train is None or self._alpha is None:
            mean = np.zeros((x.shape[0],), dtype=float)
            std = np.ones((x.shape[0],), dtype=float)
            return mean, std

        x = np.array(x, dtype=float)
        k_star = self._kernel(self.x_train, x)
        mean = k_star.T @ self._alpha

        if self._chol is not None:
            v = np.linalg.solve(self._chol, k_star)
            var = self.signal_var - np.sum(v * v, axis=0)
        else:
            k = self._kernel(x, x)
            var = np.diag(k) - np.sum(k_star * (self._alpha[:, None]), axis=0)

        var = np.maximum(var, 1e-9)
        mean = mean * self.y_std + self.y_mean
        std = np.sqrt(var) * self.y_std
        return mean, std


@ray.remote
class SchedulerActor:
    def __init__(
        self,
        initial_context_length: int,
        initial_inference_length: int,
        optimization_priority: str = "quality",
        run_name: str = "",
        wandb_run=None,
        wandb_config: Optional[Dict[str, Any]] = None,
        target_latency: float = 0.7,
        latency_margin: float = 0.1,
        window_size: int = 30,
        context_bounds: Tuple[int, int] = (2, 8),
        inference_bounds: Tuple[int, int] = (2, 8),
        # qNEHVI defaults
        delta: float = 0.95,
        q_batch: int = 1,
        raw_samples: int = 64,
        num_restarts: int = 5,
        local_steps: int = 10,
        mc_samples: int = 16,
        mix_lambda: float = 0.7,
        # Safety / constraints
        memory_max_gb: Optional[float] = None,
        memory_margin_gb: float = 0.5,
        quality_min: Optional[float] = None,
        warning_ratio: float = 0.90,
        critical_ratio: float = 0.98,
        # Stability
        epsilon: Optional[Tuple[int, int]] = (1, 1),
        t_dwell: int = 3,
        # Preference / execution policy
        preference: Optional[str] = None,  # None -> derived from optimization_priority
        risk_beta: float = 0.0,            # risk aversion: penalize predictive std
        explore_every: int = 8,            # periodic exploration in stable mode
        explore_on_adapt: bool = True,     # always allow exploration in adapt mode
        explore_regret_ratio: float = 0.30,  # exploration guardrail vs exploit
        # Regime detection / adaptation
        jsd_adapt_threshold: Optional[float] = None,
        latency_cv_adapt_threshold: float = 0.35,
        quality_cv_adapt_threshold: float = 0.35,
        adapt_hold_steps: int = 3,
        epsilon_adapt: Optional[Tuple[int, int]] = None,  # None -> allow full-range moves in adapt
        use_reachable_enumeration: bool = True,
        seed: Optional[int] = None,
        shift_env_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.shift_env_overrides: Dict[str, str] = {}
        if shift_env_overrides:
            for key, value in dict(shift_env_overrides).items():
                if value is None:
                    continue
                key_s = str(key)
                if not key_s.startswith("TREAM_SHIFT_"):
                    continue
                val_s = str(value)
                os.environ[key_s] = val_s
                self.shift_env_overrides[key_s] = val_s
        self.config_version = 0
        self.current_config_tuple = (int(initial_context_length), int(initial_inference_length))
        self.x_prev = np.array(self.current_config_tuple, dtype=int)
        self.x_safe: Optional[np.ndarray] = None
        self.optimization_priority = optimization_priority

        self.sla_latency = float(target_latency)
        self.latency_margin = float(latency_margin)
        self.window_size = int(window_size)
        self.context_bounds = (int(context_bounds[0]), int(context_bounds[1]))
        self.inference_bounds = (int(inference_bounds[0]), int(inference_bounds[1]))
        self.bounds = np.array([self.context_bounds, self.inference_bounds], dtype=float)

        self.delta = float(delta)
        self.q_batch = int(max(1, q_batch))
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.local_steps = int(local_steps)
        self.mc_samples = int(mc_samples)
        self.mix_lambda = float(np.clip(mix_lambda, 0.0, 1.0))

        self.memory_max_gb = memory_max_gb
        self.memory_margin_gb = float(memory_margin_gb)
        self.quality_min = quality_min
        self.warning_ratio = float(warning_ratio)
        self.critical_ratio = float(critical_ratio)

        self.epsilon = np.array(epsilon if epsilon is not None else (1, 1), dtype=float)
        self.t_dwell = int(max(0, t_dwell))
        self.dwell_counter = 0
        self.high_impact_mask = np.array([True, True], dtype=bool)
        self.preference = preference
        self.risk_beta = float(risk_beta)
        self.explore_every = int(max(1, explore_every))
        self.explore_on_adapt = bool(explore_on_adapt)
        self.explore_regret_ratio = float(explore_regret_ratio)
        self.jsd_adapt_threshold = jsd_adapt_threshold
        self.latency_cv_adapt_threshold = float(latency_cv_adapt_threshold)
        self.quality_cv_adapt_threshold = float(quality_cv_adapt_threshold)
        self.adapt_hold_steps = int(max(0, adapt_hold_steps))
        self.epsilon_adapt = epsilon_adapt
        self.use_reachable_enumeration = bool(use_reachable_enumeration)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.inference_metrics: deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.training_metrics: deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.mode = "stable"
        self.adapt_countdown = 0
        self.obs_history: deque[Dict[str, Any]] = deque(maxlen=window_size)

        self.dataset: List[Dict[str, Any]] = []
        self.pareto_2d: List[Dict[str, Any]] = []
        self.pareto_3d: List[Dict[str, Any]] = []

        self.obj_models = [SimpleGP(), SimpleGP(), SimpleGP()]
        self.con_models: Dict[str, SimpleGP] = {
            "g1": SimpleGP(),
            "g2": SimpleGP(),
            "g3": SimpleGP(),
        }
        self.con_ready = {"g1": False, "g2": False, "g3": False}
        self.obj_ready = {"L": False, "M": False, "Q": False}
        self.models_ready = False

        self.inference_actor_handle = None
        self.training_actor_handle = None

        self.run_name = run_name
        self.wandb_config = dict(wandb_config or {})
        self.wandb_run = wandb_run if wandb_run is not None else self._init_wandb_run()
        self.schedule_step = 0
        log_dir = "./inference_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"scheduler_{run_name}.jsonl" if run_name else f"scheduler_logs_{time.time()}.jsonl"
        self.log_path = os.path.join(log_dir, log_filename)

        ctx_lo = min(self.context_bounds[0], int(initial_context_length))
        ctx_hi = max(self.context_bounds[1], int(initial_context_length))
        inf_lo = min(self.inference_bounds[0], int(initial_inference_length))
        inf_hi = max(self.inference_bounds[1], int(initial_inference_length))
        self.shift_config_space = ConfigSpace(
            ctx_values=list(range(ctx_lo, ctx_hi + 1)),
            inf_values=list(range(inf_lo, inf_hi + 1)),
            max_all_configs=4096,
        )
        pref = (self.preference or self.optimization_priority or "quality").lower()
        if pref in ("speed", "throughput"):
            pref = "latency"
        shift_mem_limit = None
        if self.memory_max_gb is not None:
            shift_mem_limit = float(self.memory_max_gb) - float(self.memory_margin_gb)
        eps_fast = int(max(0, int(self.epsilon[0])))
        eps_fast = max(eps_fast, int(max(0, int(self.epsilon[1]))))
        eps_adapt = self.epsilon_adapt if self.epsilon_adapt is not None else self.epsilon
        eps_adapt_fast = int(max(0, int(eps_adapt[0])))
        eps_adapt_fast = max(eps_adapt_fast, int(max(0, int(eps_adapt[1]))))
        stable_eps_override = os.getenv("TREAM_SHIFT_STABLE_EPS_FAST", "").strip()
        if stable_eps_override:
            try:
                eps_fast = max(0, int(stable_eps_override))
            except Exception:
                pass
        adapt_eps_override = os.getenv("TREAM_SHIFT_ADAPT_EPS_FAST", "").strip()
        if adapt_eps_override:
            try:
                eps_adapt_fast = max(1, int(adapt_eps_override))
            except Exception:
                pass
        warmup_hold_windows = int(os.getenv("TREAM_SHIFT_WARMUP_HOLD_WINDOWS", "2"))
        warmup_explore_enable = os.getenv(
            "TREAM_SHIFT_WARMUP_EXPLORE_ENABLE", "0"
        ).strip().lower() not in ("0", "false", "no")
        warmup_force_descend_large = os.getenv(
            "TREAM_SHIFT_WARMUP_FORCE_DESCEND_LARGE", "1"
        ).strip().lower() not in ("0", "false", "no")
        warmup_large_ctx_threshold = int(
            os.getenv("TREAM_SHIFT_WARMUP_LARGE_CTX_THRESHOLD", "6")
        )
        warmup_large_inf_threshold = int(
            os.getenv("TREAM_SHIFT_WARMUP_LARGE_INF_THRESHOLD", "6")
        )
        warmup_osc_exit_enable = os.getenv(
            "TREAM_SHIFT_WARMUP_OSC_EXIT_ENABLE", "1"
        ).strip().lower() not in ("0", "false", "no")
        warmup_osc_patience = int(os.getenv("TREAM_SHIFT_WARMUP_OSC_PATIENCE", "4"))
        warmup_osc_unique_max = int(os.getenv("TREAM_SHIFT_WARMUP_OSC_UNIQUE_MAX", "2"))
        cold_windows = int(os.getenv("TREAM_SHIFT_COLD_START_WINDOWS", "15"))
        cold_max_windows = int(
            os.getenv(
                "TREAM_SHIFT_COLD_START_MAX_WINDOWS",
                str(max(cold_windows, cold_windows)),
            )
        )
        cold_probe_every = int(os.getenv("TREAM_SHIFT_COLD_PROBE_EVERY", "2"))
        cold_probe_eps_fast = int(
            os.getenv("TREAM_SHIFT_COLD_PROBE_EPS_FAST", str(max(2, eps_fast)))
        )
        cold_probe_eps_slow = int(os.getenv("TREAM_SHIFT_COLD_PROBE_EPS_SLOW", "0"))
        cold_avoid_two_cycle = os.getenv("TREAM_SHIFT_COLD_AVOID_TWO_CYCLE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_directional_enable = os.getenv(
            "TREAM_SHIFT_COLD_DIRECTIONAL_ENABLE", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_prefer_large_descend = os.getenv(
            "TREAM_SHIFT_COLD_PREFER_LARGE_DESCEND", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_prefer_small_ascend = os.getenv(
            "TREAM_SHIFT_COLD_PREFER_SMALL_ASCEND", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_large_ctx_threshold = int(
            os.getenv("TREAM_SHIFT_COLD_LARGE_CTX_THRESHOLD", "6")
        )
        cold_large_inf_threshold = int(
            os.getenv("TREAM_SHIFT_COLD_LARGE_INF_THRESHOLD", "6")
        )
        cold_small_ctx_threshold = int(
            os.getenv("TREAM_SHIFT_COLD_SMALL_CTX_THRESHOLD", "1")
        )
        cold_small_inf_threshold = int(
            os.getenv("TREAM_SHIFT_COLD_SMALL_INF_THRESHOLD", "1")
        )
        cold_target_ctx = int(os.getenv("TREAM_SHIFT_COLD_TARGET_CTX", "2"))
        cold_target_inf = int(os.getenv("TREAM_SHIFT_COLD_TARGET_INF", "2"))
        cold_target_ctx_band = int(os.getenv("TREAM_SHIFT_COLD_TARGET_CTX_BAND", "1"))
        cold_target_inf_band = int(os.getenv("TREAM_SHIFT_COLD_TARGET_INF_BAND", "1"))
        cold_single_axis_lock = os.getenv(
            "TREAM_SHIFT_COLD_SINGLE_AXIS_LOCK", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_early_exit_enable = os.getenv(
            "TREAM_SHIFT_COLD_EARLY_EXIT_ENABLE", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_target_stable_windows = int(
            os.getenv("TREAM_SHIFT_COLD_TARGET_STABLE_WINDOWS", "2")
        )
        cold_patience_dirs = int(os.getenv("TREAM_SHIFT_COLD_PATIENCE_DIRECTIONS", "5"))
        cold_improve_lat_eps = float(os.getenv("TREAM_SHIFT_COLD_IMPROVE_LAT_EPS", "0.0"))
        cold_improve_q_eps = float(os.getenv("TREAM_SHIFT_COLD_IMPROVE_Q_EPS", "0.0"))
        cold_relax_safety = os.getenv("TREAM_SHIFT_COLD_RELAX_SAFETY", "0").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_relax_radius = int(os.getenv("TREAM_SHIFT_COLD_RELAX_RADIUS", "2"))
        cold_relax_q_slack = float(os.getenv("TREAM_SHIFT_COLD_RELAX_Q_SLACK", "0.25"))
        cold_relax_lat_slack = float(os.getenv("TREAM_SHIFT_COLD_RELAX_LAT_SLACK", "0.15"))
        cold_axis_rotation = os.getenv("TREAM_SHIFT_COLD_AXIS_ROTATION", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_i_major_span = int(os.getenv("TREAM_SHIFT_COLD_I_MAJOR_SPAN", "3"))
        cold_whitelist_probe = os.getenv("TREAM_SHIFT_COLD_WHITELIST_PROBE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        cold_whitelist_budget = int(os.getenv("TREAM_SHIFT_COLD_WHITELIST_BUDGET", "4"))
        cold_whitelist_lat_slack = float(os.getenv("TREAM_SHIFT_COLD_WHITELIST_LAT_SLACK", "0.15"))
        cold_whitelist_q_slack = float(os.getenv("TREAM_SHIFT_COLD_WHITELIST_Q_SLACK", "0.6"))
        cold_whitelist_max_switch = int(
            os.getenv("TREAM_SHIFT_COLD_WHITELIST_MAX_SWITCH", "2")
        )
        min_count_for_trust = int(os.getenv("TREAM_SHIFT_MIN_COUNT_FOR_TRUST", "2"))
        max_staleness_ticks = int(os.getenv("TREAM_SHIFT_MAX_STALENESS_TICKS", "64"))
        force_adapt_min_count = int(
            os.getenv(
                "TREAM_SHIFT_FORCE_ADAPT_MIN_COUNT",
                str(max(min_count_for_trust, warmup_hold_windows + 2)),
            )
        )
        sigma_floor = float(os.getenv("TREAM_SHIFT_SIGMA_FLOOR", "0.01"))
        latency_metric = os.getenv("TREAM_SHIFT_SAFETY_LATENCY_METRIC", "lat_mean").strip().lower()
        candidate_min_probe_keep = int(os.getenv("TREAM_SHIFT_CANDIDATE_MIN_PROBE_KEEP", "2"))
        adapt_probe_windows = int(os.getenv("TREAM_SHIFT_ADAPT_PROBE_WINDOWS", "2"))
        adapt_exit_q_margin = float(os.getenv("TREAM_SHIFT_ADAPT_EXIT_Q_MARGIN", "0.03"))
        adapt_exit_l_margin = float(os.getenv("TREAM_SHIFT_ADAPT_EXIT_L_MARGIN", "0.08"))
        salvage_infeasible_streak = int(os.getenv("TREAM_SHIFT_SALVAGE_INFEASIBLE_STREAK", "2"))
        salvage_dwell_steps = int(os.getenv("TREAM_SHIFT_SALVAGE_DWELL_STEPS", str(max(2, self.t_dwell))))
        min_switch_gain_eps = float(os.getenv("TREAM_SHIFT_MIN_SWITCH_GAIN_EPS", "0.01"))
        force_adapt_q_margin = float(os.getenv("TREAM_SHIFT_FORCE_ADAPT_Q_MARGIN", "0.02"))
        force_adapt_l_margin = float(os.getenv("TREAM_SHIFT_FORCE_ADAPT_L_MARGIN", "0.05"))
        hold_anchor_radius = int(os.getenv("TREAM_SHIFT_HOLD_ANCHOR_RADIUS", "1"))
        reroute_radius = int(os.getenv("TREAM_SHIFT_REROUTE_RADIUS", "2"))
        anchor_trial_promote_windows = int(
            os.getenv("TREAM_SHIFT_ANCHOR_TRIAL_PROMOTE_WINDOWS", "2")
        )
        anchor_bootstrap_windows = int(
            os.getenv("TREAM_SHIFT_ANCHOR_BOOTSTRAP_WINDOWS", "3")
        )
        anchor_init_confidence = float(
            os.getenv("TREAM_SHIFT_ANCHOR_INIT_CONFIDENCE", "0.15")
        )
        seek_exit_anchor_confidence = float(
            os.getenv("TREAM_SHIFT_SEEK_EXIT_ANCHOR_CONFIDENCE", "0.60")
        )
        seek_exit_min_probe_obs = int(
            os.getenv("TREAM_SHIFT_SEEK_EXIT_MIN_PROBE_OBS", "3")
        )
        seek_exit_stable_windows = int(
            os.getenv("TREAM_SHIFT_SEEK_EXIT_STABLE_WINDOWS", "3")
        )
        seek_exit_stable_radius = int(
            os.getenv("TREAM_SHIFT_SEEK_EXIT_STABLE_RADIUS", "1")
        )
        stabilizer_enable = os.getenv("TREAM_SHIFT_STAB_ENABLE", "0").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        stabilizer_shadow = os.getenv("TREAM_SHIFT_STAB_SHADOW", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        stabilizer_allow_in_adapt = os.getenv(
            "TREAM_SHIFT_STAB_ALLOW_IN_ADAPT", "0"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        stabilizer_two_cycle = os.getenv("TREAM_SHIFT_STAB_TWO_CYCLE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        stabilizer_seek_monotone = os.getenv(
            "TREAM_SHIFT_STAB_SEEK_MONOTONE", "0"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )
        stabilizer_seek_max_inf_rebound = int(
            os.getenv("TREAM_SHIFT_STAB_SEEK_MAX_INF_REBOUND", "0")
        )
        stabilizer_low_shock_max = float(
            os.getenv("TREAM_SHIFT_STAB_LOW_SHOCK_MAX", "0.20")
        )
        stabilizer_low_shock_gain_eps = float(
            os.getenv("TREAM_SHIFT_STAB_LOW_SHOCK_GAIN_EPS", "0.0")
        )
        shock_use_jsd_only = os.getenv("TREAM_SHIFT_SHOCK_USE_JSD_ONLY", "0").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        local_relax_enable = os.getenv("TREAM_SHIFT_LOCAL_RELAX_ENABLE", "0").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        local_relax_windows = int(os.getenv("TREAM_SHIFT_LOCAL_RELAX_WINDOWS", "3"))
        local_relax_radius = int(os.getenv("TREAM_SHIFT_LOCAL_RELAX_RADIUS", "1"))
        local_relax_drift_threshold = float(
            os.getenv("TREAM_SHIFT_LOCAL_RELAX_DRIFT_THRESHOLD", "0.30")
        )

        shock_w_drift = 1.0
        shock_w_accept = 1.0
        shock_w_reverify = 1.0
        shock_w_verify_ratio = 1.0
        shock_w_waste_rate = 1.0
        if shock_use_jsd_only:
            # Keep non-drift signals in logs, but avoid letting them drive mode.
            shock_w_accept = 0.0
            shock_w_reverify = 0.0
            shock_w_verify_ratio = 0.0
            shock_w_waste_rate = 0.0
        quality_min_default_raw = os.getenv("TREAM_SHIFT_DEFAULT_QUALITY_MIN", "").strip()
        shift_quality_min = self.quality_min
        if shift_quality_min is None and quality_min_default_raw:
            try:
                shift_quality_min = float(quality_min_default_raw)
            except Exception:
                shift_quality_min = self.quality_min
        self.shift_scheduler_config = SchedulerConfig(
            tau=shift_quality_min,
            sla=self.sla_latency,
            mem_limit=shift_mem_limit,
            train_min=None,
            beta=self.risk_beta,
            sigma_floor=max(0.0, sigma_floor),
            shock_w_drift=max(0.0, float(shock_w_drift)),
            shock_w_accept=max(0.0, float(shock_w_accept)),
            shock_w_reverify=max(0.0, float(shock_w_reverify)),
            shock_w_verify_ratio=max(0.0, float(shock_w_verify_ratio)),
            shock_w_waste_rate=max(0.0, float(shock_w_waste_rate)),
            shock_use_jsd_only=bool(shock_use_jsd_only),
            min_count_for_trust=max(0, min_count_for_trust),
            max_staleness_ticks=max(0, max_staleness_ticks),
            safety_latency_metric="lat_p95" if latency_metric == "lat_p95" else "lat_mean",
            adapt_hold=self.adapt_hold_steps,
            candidate_cap=128,
            anchor_k=8,
            stable_eps_fast=eps_fast,
            stable_eps_slow=0,
            adapt_eps_fast=max(1, eps_adapt_fast),
            adapt_eps_slow=0,
            candidate_min_probe_keep=max(0, candidate_min_probe_keep),
            adapt_use_all_configs_if_small=True,
            preference=pref,
            switch_cost_lambda=0.2,
            dwell_steps=self.t_dwell,
            warmup_hold_windows=max(0, warmup_hold_windows),
            warmup_explore_enable=bool(warmup_explore_enable),
            warmup_force_descend_large=bool(warmup_force_descend_large),
            warmup_large_ctx_threshold=max(1, int(warmup_large_ctx_threshold)),
            warmup_large_inf_threshold=max(1, int(warmup_large_inf_threshold)),
            warmup_oscillation_exit_enable=bool(warmup_osc_exit_enable),
            warmup_oscillation_patience=max(3, int(warmup_osc_patience)),
            warmup_oscillation_unique_max=max(2, int(warmup_osc_unique_max)),
            cold_start_windows=max(0, cold_windows),
            cold_start_max_windows=max(0, max(cold_windows, cold_max_windows)),
            cold_start_probe_every=max(1, cold_probe_every),
            cold_start_relax_safety=bool(cold_relax_safety),
            cold_start_axis_rotation=bool(cold_axis_rotation),
            cold_start_i_major_span=max(1, cold_i_major_span),
            cold_start_probe_eps_fast=max(1, cold_probe_eps_fast),
            cold_start_probe_eps_slow=max(0, cold_probe_eps_slow),
            cold_start_avoid_two_cycle=bool(cold_avoid_two_cycle),
            cold_start_directional_enable=bool(cold_directional_enable),
            cold_start_prefer_large_descend=bool(cold_prefer_large_descend),
            cold_start_prefer_small_ascend=bool(cold_prefer_small_ascend),
            cold_start_large_ctx_threshold=max(1, cold_large_ctx_threshold),
            cold_start_large_inf_threshold=max(1, cold_large_inf_threshold),
            cold_start_small_ctx_threshold=max(1, cold_small_ctx_threshold),
            cold_start_small_inf_threshold=max(1, cold_small_inf_threshold),
            cold_start_target_ctx=max(1, cold_target_ctx),
            cold_start_target_inf=max(1, cold_target_inf),
            cold_start_target_ctx_band=max(0, cold_target_ctx_band),
            cold_start_target_inf_band=max(0, cold_target_inf_band),
            cold_start_single_axis_lock=bool(cold_single_axis_lock),
            cold_start_early_exit_enable=bool(cold_early_exit_enable),
            cold_start_target_stable_windows=max(1, int(cold_target_stable_windows)),
            cold_start_relax_radius=max(1, int(cold_relax_radius)),
            cold_start_relax_quality_slack=max(0.0, float(cold_relax_q_slack)),
            cold_start_relax_latency_slack=max(0.0, float(cold_relax_lat_slack)),
            cold_start_patience_directions=max(1, cold_patience_dirs),
            cold_start_improve_latency_eps=max(0.0, cold_improve_lat_eps),
            cold_start_improve_quality_eps=max(0.0, cold_improve_q_eps),
            cold_whitelist_probe_enable=bool(cold_whitelist_probe),
            cold_whitelist_probe_budget=max(0, cold_whitelist_budget),
            cold_whitelist_latency_slack=max(0.0, cold_whitelist_lat_slack),
            cold_whitelist_quality_slack=max(0.0, cold_whitelist_q_slack),
            cold_whitelist_max_switch_distance=max(0, cold_whitelist_max_switch),
            adapt_probe_windows=max(0, adapt_probe_windows),
            adapt_exit_quality_margin=adapt_exit_q_margin,
            adapt_exit_latency_margin=adapt_exit_l_margin,
            salvage_infeasible_streak=max(1, salvage_infeasible_streak),
            salvage_dwell_steps=max(1, salvage_dwell_steps),
            min_switch_gain_eps=max(0.0, min_switch_gain_eps),
            stabilizer_enable=bool(stabilizer_enable),
            stabilizer_shadow=bool(stabilizer_shadow),
            stabilizer_allow_in_adapt=bool(stabilizer_allow_in_adapt),
            stabilizer_two_cycle=bool(stabilizer_two_cycle),
            stabilizer_seek_monotone=bool(stabilizer_seek_monotone),
            stabilizer_seek_max_inf_rebound=max(0, stabilizer_seek_max_inf_rebound),
            stabilizer_low_shock_max=max(0.0, stabilizer_low_shock_max),
            stabilizer_low_shock_gain_eps=max(0.0, stabilizer_low_shock_gain_eps),
            force_adapt_min_count=max(1, force_adapt_min_count),
            force_adapt_quality_margin=force_adapt_q_margin,
            force_adapt_latency_margin=force_adapt_l_margin,
            hold_anchor_radius=max(1, hold_anchor_radius),
            reroute_radius=max(1, reroute_radius),
            anchor_trial_promote_windows=max(1, anchor_trial_promote_windows),
            anchor_bootstrap_windows=max(1, anchor_bootstrap_windows),
            anchor_init_confidence=max(0.0, min(1.0, anchor_init_confidence)),
            seek_exit_anchor_confidence=max(0.0, min(1.0, seek_exit_anchor_confidence)),
            seek_exit_min_probe_observations=max(0, seek_exit_min_probe_obs),
            seek_exit_stable_windows=max(0, seek_exit_stable_windows),
            seek_exit_stable_radius=max(0, seek_exit_stable_radius),
            local_relax_enable=bool(local_relax_enable),
            local_relax_windows=max(0, int(local_relax_windows)),
            local_relax_radius=max(1, int(local_relax_radius)),
            local_relax_drift_threshold=max(0.0, float(local_relax_drift_threshold)),
        )
        self.shift_core = SchedulerCore(
            config_space=self.shift_config_space,
            initial_config=ConfigX(
                ctx=int(initial_context_length),
                inf=int(initial_inference_length),
            ),
            scheduler_config=self.shift_scheduler_config,
        )
        self.use_shift_scheduler = True

        print(
            "[Scheduler] scheduler_Shift core initialized with config "
            f"c={self.current_config_tuple[0]}, i={self.current_config_tuple[1]}, "
            f"sla={self.sla_latency}s, window={self.window_size}, "
            f"warmup={self.shift_scheduler_config.warmup_hold_windows}, "
            f"warmup_explore={'on' if self.shift_scheduler_config.warmup_explore_enable else 'off'}, "
            f"cold_windows={self.shift_scheduler_config.cold_start_windows}, "
            f"hold_radius={self.shift_scheduler_config.hold_anchor_radius}, "
            f"reroute_radius={self.shift_scheduler_config.reroute_radius}, "
            f"trial_promote={self.shift_scheduler_config.anchor_trial_promote_windows}, "
            f"stabilizer={'on' if self.shift_scheduler_config.stabilizer_enable else 'off'}, "
            f"stab_shadow={'on' if self.shift_scheduler_config.stabilizer_shadow else 'off'}"
        )
        print(
            "[Scheduler] scheduler_Shift knobs "
            f"cold_max_windows={self.shift_scheduler_config.cold_start_max_windows}, "
            f"cold_probe_every={self.shift_scheduler_config.cold_start_probe_every}, "
            f"cold_probe_eps_fast={self.shift_scheduler_config.cold_start_probe_eps_fast}, "
            f"cold_probe_eps_slow={self.shift_scheduler_config.cold_start_probe_eps_slow}, "
            f"cold_relax_safety={self.shift_scheduler_config.cold_start_relax_safety}, "
            f"cold_whitelist_enable={self.shift_scheduler_config.cold_whitelist_probe_enable}, "
            f"cold_whitelist_budget={self.shift_scheduler_config.cold_whitelist_probe_budget}, "
            f"min_count_for_trust={self.shift_scheduler_config.min_count_for_trust}, "
            f"max_staleness_ticks={self.shift_scheduler_config.max_staleness_ticks}, "
            f"seek_exit_conf={self.shift_scheduler_config.seek_exit_anchor_confidence}, "
            f"seek_exit_min_probe_obs={self.shift_scheduler_config.seek_exit_min_probe_observations}, "
            f"seek_exit_stable={self.shift_scheduler_config.seek_exit_stable_windows}"
            f"@r{self.shift_scheduler_config.seek_exit_stable_radius}, "
            f"anchor_bootstrap={self.shift_scheduler_config.anchor_bootstrap_windows}, "
            f"anchor_init_conf={self.shift_scheduler_config.anchor_init_confidence}, "
            f"adapt_probe_windows={self.shift_scheduler_config.adapt_probe_windows}, "
            f"shock_jsd_only={self.shift_scheduler_config.shock_use_jsd_only}, "
            f"local_relax_enable={self.shift_scheduler_config.local_relax_enable}, "
            f"local_relax_windows={self.shift_scheduler_config.local_relax_windows}, "
            f"local_relax_radius={self.shift_scheduler_config.local_relax_radius}, "
            f"local_relax_drift_threshold={self.shift_scheduler_config.local_relax_drift_threshold}, "
            f"shift_env_overrides={len(self.shift_env_overrides)}"
        )

    def _init_wandb_run(self):
        if wandb is None:
            return None
        cfg = self.wandb_config if isinstance(self.wandb_config, dict) else {}
        if not bool(cfg.get("enabled", False)):
            return None

        project = str(cfg.get("project") or "test-feng1702")
        entity = cfg.get("entity")
        parent_name = str(cfg.get("parent_run_name") or self.run_name or "scheduler")
        run_group = str(cfg.get("run_group") or parent_name)
        run_mode = str(cfg.get("mode") or "online")
        run_name = f"{parent_name}.scheduler"

        for attempt in range(1, 4):
            try:
                run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    group=run_group,
                    job_type="scheduler_actor",
                    mode=run_mode,
                    reinit=True,
                )
                print(
                    f"[SchedulerActor] WandB initialized (name={run_name}, group={run_group})."
                )
                return run
            except Exception as exc:
                if attempt >= 3:
                    print(f"[SchedulerActor] WARN: wandb init failed: {exc}")
                    return None
                time.sleep(0.5)
        return None

    # --- Registration & state access ---

    def register_actor_handles(self, inference_actor, training_actor) -> None:
        with self._lock:
            self.inference_actor_handle = inference_actor
            self.training_actor_handle = training_actor

    def get_current_config_tuple(self) -> tuple:
        with self._lock:
            return self.current_config_tuple

    # --- Metric ingestion ---

    def report_inference_metrics(self, metrics: Dict[str, Any], reported_config: tuple, reported_version: int) -> None:
        with self._lock:
            if reported_version != self.config_version:
                return
            self.inference_metrics.append(metrics)
            if len(self.inference_metrics) >= self.window_size:
                self._step_control()

    def report_training_metrics(self, metrics: Dict[str, Any], reported_config: tuple, reported_version: int) -> None:
        with self._lock:
            if reported_version != self.config_version:
                return
            self.training_metrics.append(metrics)

    # --- Legacy API (time-only) for backward compatibility ---

    def report_inference_time(self, t_inf: float, reported_config: tuple, reported_version: int) -> None:
        self.report_inference_metrics(
            {"latency": t_inf, "tokens_per_second": None, "perplexity": None, "accuracy": None},
            reported_config,
            reported_version,
        )

    def report_training_time(self, t_train: float, reported_config: tuple, reported_version: int) -> None:
        self.report_training_metrics(
            {"latency": t_train, "tokens_per_second": None},
            reported_config,
            reported_version,
        )

    # --- Core scheduling logic ---

    def _update_mode(self, obs: Dict[str, Any]) -> None:
        """Decide whether we are in a stable regime or an adapt regime."""
        jsd = obs.get("jsd_mean")
        lat_mean = obs.get("latency_mean")
        lat_std = obs.get("latency_std")
        q_mean = obs.get("quality")
        q_std = obs.get("quality_std")

        lat_cv = None
        if lat_mean is not None and lat_std is not None and abs(lat_mean) > 1e-9:
            lat_cv = float(lat_std) / float(abs(lat_mean))
        q_cv = None
        if q_mean is not None and q_std is not None and abs(q_mean) > 1e-9:
            q_cv = float(q_std) / float(abs(q_mean))

        trigger = False
        if self.jsd_adapt_threshold is not None and jsd is not None:
            try:
                trigger = trigger or (float(jsd) >= float(self.jsd_adapt_threshold))
            except Exception:
                pass
        if lat_cv is not None:
            trigger = trigger or (lat_cv >= self.latency_cv_adapt_threshold)
        if q_cv is not None:
            trigger = trigger or (q_cv >= self.quality_cv_adapt_threshold)

        if trigger:
            self.mode = "adapt"
            self.adapt_countdown = self.adapt_hold_steps
        else:
            if self.adapt_countdown > 0:
                self.adapt_countdown -= 1
                self.mode = "adapt"
            else:
                self.mode = "stable"

    def _should_explore(self) -> bool:
        if self.mode == "adapt" and self.explore_on_adapt:
            return True
        return (self.schedule_step % self.explore_every) == 0

    def _effective_epsilon(self) -> np.ndarray:
        if self.mode != "adapt":
            return self.epsilon.copy()
        if self.epsilon_adapt is None:
            return np.array(
                [
                    self.context_bounds[1] - self.context_bounds[0],
                    self.inference_bounds[1] - self.inference_bounds[0],
                ],
                dtype=float,
            )
        return np.array(self.epsilon_adapt, dtype=float)

    def _reachable_candidates(self, x_last: np.ndarray) -> List[np.ndarray]:
        eps = self._effective_epsilon().astype(int)
        c0 = int(x_last[0])
        i0 = int(x_last[1])
        c_lo = max(self.context_bounds[0], c0 - int(eps[0]))
        c_hi = min(self.context_bounds[1], c0 + int(eps[0]))
        i_lo = max(self.inference_bounds[0], i0 - int(eps[1]))
        i_hi = min(self.inference_bounds[1], i0 + int(eps[1]))
        out = []
        for c in range(c_lo, c_hi + 1):
            for i in range(i_lo, i_hi + 1):
                out.append(np.array([c, i], dtype=int))
        return out

    def _preference_policy(self) -> str:
        pref = (self.preference or self.optimization_priority or "quality").lower()
        if pref in ("latency", "speed", "throughput"):
            return "latency"
        return "quality"

    def _select_exploit(self, candidates: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        policy = self._preference_policy()
        best = None
        best_tuple = None
        best_pof = None
        best_pofs = None
        feasible_count = 0

        for x in candidates:
            p_total, p_map = self._pof_total(x)
            if p_total < self.delta:
                continue
            feasible_count += 1
            mu, sigma = self._predict_objectives(x)
            L_mu = float(mu[0])
            L_sig = float(sigma[0])
            Q_mu = float(-mu[2])
            Q_sig = float(sigma[2])

            if policy == "latency":
                primary = L_mu + self.risk_beta * L_sig
                secondary = -Q_mu
            else:
                primary = -(Q_mu - self.risk_beta * Q_sig)
                secondary = L_mu

            tup = (primary, secondary)
            if best is None or tup < best_tuple:
                best = x
                best_tuple = tup
                best_pof = p_total
                best_pofs = p_map

        info = {
            "selection_policy": policy,
            "feasible_count": feasible_count,
            "pof": best_pof,
            "pof_map": best_pofs,
        }
        return best, info

    def _select_explore(
        self,
        candidates: List[np.ndarray],
        dims: List[int],
        ref: np.ndarray,
        pareto_y: List[np.ndarray],
        avoid: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        best = None
        best_acq = -float("inf")
        for x in candidates:
            if avoid is not None and np.array_equal(x, avoid):
                continue
            acq = self._acquisition(x.astype(float), ref, dims, pareto_y)
            if acq > best_acq:
                best_acq = acq
                best = x
        if best is None:
            return None, {}
        return best, {"acq_selected": best_acq}

    def _explore_guard_ok(self, x_explore: np.ndarray, x_exploit: np.ndarray) -> bool:
        if x_explore is None or x_exploit is None:
            return False
        mu_e, _ = self._predict_objectives(x_exploit)
        mu_x, _ = self._predict_objectives(x_explore)
        policy = self._preference_policy()
        L_e = float(mu_e[0])
        L_x = float(mu_x[0])
        Q_e = float(-mu_e[2])
        Q_x = float(-mu_x[2])
        r = float(self.explore_regret_ratio)
        if policy == "latency":
            return L_x <= L_e * (1.0 + r)
        return Q_x >= Q_e * (1.0 - r)

    def _constraint_warnings(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        warns: Dict[str, Any] = {}
        lat_p95 = obs.get("latency_p95")
        if lat_p95 is not None and self.sla_latency > 0:
            ratio = float(lat_p95) / float(self.sla_latency)
            warns["latency_scale_ratio"] = ratio
            if ratio >= 100.0 or ratio <= 0.01:
                warns["latency_scale_warning"] = True

        if self.quality_min is not None:
            src = obs.get("quality_source")
            if src in ("none", None):
                warns["quality_scale_warning"] = True
            elif src == "perplexity_inv" and self.quality_min > 1.0:
                warns["quality_scale_warning"] = True
            elif src == "accuracy" and self.quality_min > 1.0:
                warns["quality_scale_warning"] = True

        if len(self.dataset) < 5 and self.delta >= 0.95:
            warns["delta_high_warning"] = True

        return warns

    def _step_control(self) -> None:
        if getattr(self, "use_shift_scheduler", False):
            self._step_control_shift()
            return

        x_last = self.x_prev.copy()
        agg = self._aggregate_metrics()
        if agg is None:
            return

        obs = self._observe_and_update(x_last, agg)
        warn_info = self._constraint_warnings(obs)
        self._update_mode(obs)
        self.obs_history.append(
            {k: obs.get(k) for k in ("latency_mean", "latency_p95", "quality", "jsd_mean")}
        )
        self._fit_models()

        x_next, decision, select_info = self._propose_next(x_last, obs)

        if obs.get("risk_level") == "critical":
            x_next = self._emergency_degrade(x_last)
            decision = "emergency_degrade"
        elif obs.get("risk_level") == "warning":
            x_next = self._apply_warning_guard(x_next, x_last)
            decision = f"{decision}|warning_guard"

        self._apply_config(x_last, x_next, decision)
        merged_info = {**select_info, **warn_info} if warn_info else select_info
        self._log(obs, decision, x_next, merged_info)

        self.inference_metrics.clear()
        self.training_metrics.clear()
        self.schedule_step += 1

    def _step_control_shift(self) -> None:
        x_last = self.x_prev.copy()
        agg = self._aggregate_metrics()
        if agg is None:
            return

        window = self._to_window_metrics(agg)
        decision = self.shift_core.step(window)
        x_next = np.array([int(decision.x_next.ctx), int(decision.x_next.inf)], dtype=int)
        decision_reason = f"shift|{decision.reason}"

        self.mode = str(decision.mode or "STABLE").lower()
        self._apply_config(x_last, x_next, decision_reason)

        obs = dict(agg)
        obs.update(
            {
                "quality": agg.get("quality"),
                "quality_std": None,
                "quality_source": "accuracy",
                "jsd_mean": agg.get("jsd_mean"),
                "feasible": None,
                "risk_level": "normal",
                "dataset_size": len(self.shift_core.history),
                "pareto_2d_size": 0,
                "pareto_3d_size": 0,
            }
        )
        extra = dict(decision.meta or {})
        for k, v in (decision.predicted_gains or {}).items():
            extra[f"gain_{k}"] = v
        for k, v in (decision.safety_margins or {}).items():
            extra[f"safety_{k}"] = v
        self._log(obs, decision_reason, x_next, extra)

        self.inference_metrics.clear()
        self.training_metrics.clear()
        self.schedule_step += 1

    @staticmethod
    def _mean_from_keys(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Optional[float]:
        vals: List[float] = []
        for row in rows:
            for k in keys:
                if k in row and row.get(k) is not None:
                    try:
                        vals.append(float(row.get(k)))
                    except Exception:
                        pass
                    break
        if not vals:
            return None
        return float(np.mean(vals))

    def _to_window_metrics(self, agg: Dict[str, Any]) -> WindowMetrics:
        lat_mean = agg.get("latency_mean")
        if lat_mean is None:
            lat_mean = agg.get("latency_p95")
        if lat_mean is None:
            lat_mean = 0.0
        acc_mean = agg.get("accuracy_mean")
        if acc_mean is None:
            # Fallback for runs that only report perplexity-derived quality.
            acc_mean = agg.get("quality")
        if acc_mean is None:
            acc_mean = 0.0

        return WindowMetrics(
            lat_mean=float(lat_mean),
            lat_p95=None if agg.get("latency_p95") is None else float(agg.get("latency_p95")),
            acc_mean=float(acc_mean),
            train_tps=None if agg.get("train_tps_mean") is None else float(agg.get("train_tps_mean")),
            mem_peak=None if agg.get("mem_peak_p95") is None else float(agg.get("mem_peak_p95")),
            jsd_mean=None if agg.get("jsd_mean") is None else float(agg.get("jsd_mean")),
            spec_accept_mean=None if agg.get("spec_accept_mean") is None else float(agg.get("spec_accept_mean")),
            spec_reverify_per_step=None if agg.get("spec_reverify_per_step") is None else float(agg.get("spec_reverify_per_step")),
            spec_draft_ms_per_step=None if agg.get("spec_draft_ms_per_step") is None else float(agg.get("spec_draft_ms_per_step")),
            spec_verify_ms_per_step=None if agg.get("spec_verify_ms_per_step") is None else float(agg.get("spec_verify_ms_per_step")),
            rejected_tokens_per_step=None if agg.get("rejected_tokens_per_step") is None else float(agg.get("rejected_tokens_per_step")),
            accepted_tokens_per_step=None if agg.get("accepted_tokens_per_step") is None else float(agg.get("accepted_tokens_per_step")),
        )

    def _aggregate_metrics(self) -> Optional[Dict[str, Any]]:
        inf_latencies = [m.get("latency") for m in self.inference_metrics]
        if not _finite_array(inf_latencies).size:
            return None

        inf_perp = [m.get("perplexity") for m in self.inference_metrics]
        inf_acc = [m.get("accuracy") for m in self.inference_metrics]
        inf_acc_pred = [m.get("accuracy_total_predictions") for m in self.inference_metrics]
        mem_keys = ("mem_peak_gb", "memory_peak_gb", "gpu_mem_peak_gb")
        mem_vals: List[Optional[float]] = []
        for m in self.inference_metrics:
            mem_val = None
            for k in mem_keys:
                if k in m and m.get(k) is not None:
                    mem_val = m.get(k)
                    break
            mem_vals.append(mem_val)

        train_lat = [m.get("latency") for m in self.training_metrics]
        train_tps = [m.get("tokens_per_second") for m in self.training_metrics]
        jsd_vals: List[Optional[float]] = []
        for m in self.inference_metrics:
            jsd_val = None
            for k in ("jsd", "drift_jsd", "jsd_mean"):
                if k in m and m.get(k) is not None:
                    jsd_val = m.get(k)
                    break
            jsd_vals.append(jsd_val)

        spec_accept_mean = self._mean_from_keys(
            list(self.inference_metrics),
            ("spec_accept_mean", "acceptance_rate", "spec_acceptance_rate"),
        )
        spec_reverify = self._mean_from_keys(
            list(self.inference_metrics),
            ("spec_reverify_per_step", "spec_reverify", "reverify"),
        )
        spec_draft_ms = self._mean_from_keys(
            list(self.inference_metrics),
            ("spec_draft_ms_per_step", "spec_draft_ms", "draft_ms_per_step"),
        )
        spec_verify_ms = self._mean_from_keys(
            list(self.inference_metrics),
            ("spec_verify_ms_per_step", "spec_verify_ms", "verify_ms_per_step"),
        )
        rejected_tokens = self._mean_from_keys(
            list(self.inference_metrics),
            ("rejected_tokens_per_step", "spec_rejected_tokens_per_step", "rejected_tokens"),
        )
        accepted_tokens = self._mean_from_keys(
            list(self.inference_metrics),
            ("accepted_tokens_per_step", "spec_accepted_tokens_per_step", "accepted_tokens"),
        )

        L_p95 = _safe_p95(inf_latencies)
        L_std = _safe_std(inf_latencies)
        M_p95 = _safe_p95(mem_vals)
        Q_acc = _safe_mean(inf_acc)
        Q_std = _safe_std(inf_acc)
        perp_mean = _safe_mean(inf_perp)

        if Q_acc is not None:
            Q_val = Q_acc
            Q_src = "accuracy"
        elif perp_mean is not None:
            Q_val = 1.0 / max(perp_mean, 1e-6)
            Q_src = "perplexity_inv"
            Q_std = _safe_std([1.0 / max(p, 1e-6) if p is not None else None for p in inf_perp])
        else:
            Q_val = None
            Q_src = "none"
            Q_std = None

        return {
            "latency_p95": L_p95,
            "latency_mean": _safe_mean(inf_latencies),
            "latency_std": L_std,
            "train_latency_mean": _safe_mean(train_lat),
            "train_tps_mean": _safe_mean(train_tps),
            "perplexity_mean": perp_mean,
            "accuracy_mean": Q_acc,
            "accuracy_total_predictions_mean": _safe_mean(inf_acc_pred),
            "mem_peak_p95": M_p95,
            "quality": Q_val,
            "quality_std": Q_std,
            "quality_source": Q_src,
            "jsd_mean": _safe_mean(jsd_vals),
            "spec_accept_mean": spec_accept_mean,
            "spec_reverify_per_step": spec_reverify,
            "spec_draft_ms_per_step": spec_draft_ms,
            "spec_verify_ms_per_step": spec_verify_ms,
            "rejected_tokens_per_step": rejected_tokens,
            "accepted_tokens_per_step": accepted_tokens,
        }

    def _observe_and_update(self, x_last: np.ndarray, agg: Dict[str, Any]) -> Dict[str, Any]:
        L_obs = agg.get("latency_p95")
        if L_obs is None:
            L_obs = float("inf")
        M_obs = agg.get("mem_peak_p95")
        mem_available = M_obs is not None
        Q_obs = agg.get("quality")
        if Q_obs is None:
            Q_obs = 0.0

        y_last = np.array([L_obs, M_obs if mem_available else np.nan, -Q_obs], dtype=float)
        g1_last = L_obs - self.sla_latency

        g2_last = None
        if self.memory_max_gb is not None and mem_available:
            g2_last = M_obs - (self.memory_max_gb - self.memory_margin_gb)

        g3_last = None
        if self.quality_min is not None:
            g3_last = self.quality_min - Q_obs

        feasible = (g1_last <= 0)
        if g2_last is not None:
            feasible = feasible and (g2_last <= 0)
        if g3_last is not None:
            feasible = feasible and (g3_last <= 0)

        self.dataset.append(
            {
                "x": x_last.copy(),
                "y": y_last.copy(),
                "g1": g1_last,
                "g2": g2_last,
                "g3": g3_last,
                "mem_available": mem_available,
            }
        )

        if feasible:
            self.x_safe = x_last.copy()
            if mem_available:
                y_proj = self._project_y(y_last, [0, 1, 2])
                self.pareto_3d = update_pareto(self.pareto_3d, x_last.copy(), y_proj)
            y_proj = self._project_y(y_last, [0, 2])
            self.pareto_2d = update_pareto(self.pareto_2d, x_last.copy(), y_proj)

        risk_level = self._risk_level(L_obs, M_obs if mem_available else None)
        agg = dict(agg)
        agg.update(
            {
                "x_last": x_last.copy(),
                "y_last": y_last.copy(),
                "g1": g1_last,
                "g2": g2_last,
                "g3": g3_last,
                "feasible": feasible,
                "risk_level": risk_level,
                "mem_available": mem_available,
                "dataset_size": len(self.dataset),
                "pareto_2d_size": len(self.pareto_2d),
                "pareto_3d_size": len(self.pareto_3d),
            }
        )
        return agg

    def _risk_level(self, L_obs: float, M_obs: Optional[float]) -> str:
        if L_obs >= self.critical_ratio * self.sla_latency:
            return "critical"
        if L_obs >= self.warning_ratio * self.sla_latency:
            return "warning"
        if self.memory_max_gb is not None and M_obs is not None:
            mem_limit = self.memory_max_gb - self.memory_margin_gb
            if M_obs >= mem_limit:
                return "critical"
            if M_obs >= mem_limit - 0.5:
                return "warning"
        return "normal"

    def _fit_models(self) -> None:
        if len(self.dataset) < 2:
            self.models_ready = False
            return

        X = np.array([d["x"] for d in self.dataset], dtype=float)
        Xn = self._normalize_x(X)

        y_arr = np.array([d["y"] for d in self.dataset], dtype=float)
        self.obj_models[0].fit(Xn, y_arr[:, 0])
        self.obj_models[2].fit(Xn, y_arr[:, 2])
        self.obj_ready["L"] = True
        self.obj_ready["Q"] = True

        g1 = np.array([d["g1"] for d in self.dataset], dtype=float)
        self.con_models["g1"].fit(Xn, g1)
        self.con_ready["g1"] = True

        mem_pairs = [(i, d["y"][1]) for i, d in enumerate(self.dataset) if d.get("mem_available")]
        if len(mem_pairs) >= 2:
            idxs, vals = zip(*mem_pairs)
            self.obj_models[1].fit(Xn[list(idxs)], np.array(vals, dtype=float))
            self.obj_ready["M"] = True
        else:
            self.obj_ready["M"] = False

        if self.memory_max_gb is not None:
            g2_pairs = [(i, d["g2"]) for i, d in enumerate(self.dataset) if d["g2"] is not None]
            if len(g2_pairs) >= 2:
                idxs, vals = zip(*g2_pairs)
                self.con_models["g2"].fit(Xn[list(idxs)], np.array(vals, dtype=float))
                self.con_ready["g2"] = True
            else:
                self.con_ready["g2"] = False

        if self.quality_min is not None:
            g3_pairs = [(i, d["g3"]) for i, d in enumerate(self.dataset) if d["g3"] is not None]
            if len(g3_pairs) >= 2:
                idxs, vals = zip(*g3_pairs)
                self.con_models["g3"].fit(Xn[list(idxs)], np.array(vals, dtype=float))
                self.con_ready["g3"] = True
            else:
                self.con_ready["g3"] = False

        self.models_ready = True

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        denom = np.where(ub > lb, ub - lb, 1.0)
        return (x - lb) / denom

    def _project_y(self, y: np.ndarray, dims: List[int]) -> np.ndarray:
        return np.array([y[d] for d in dims], dtype=float)

    def _objective_dims(self, mem_available: bool) -> List[int]:
        if mem_available and self.obj_ready.get("M"):
            return [0, 1, 2]
        return [0, 2]

    def _pareto_y(self, dims: List[int]) -> List[np.ndarray]:
        if len(dims) == 3:
            return [np.array(item["y"], dtype=float) for item in self.pareto_3d]
        return [np.array(item["y"], dtype=float) for item in self.pareto_2d]

    def _reference_point(self, dims: List[int], mem_available: bool) -> np.ndarray:
        if not self.dataset:
            ref = []
            for d in dims:
                if d == 0:
                    ref.append(self.sla_latency * 1.5)
                elif d == 1:
                    ref.append(self.memory_max_gb or 1.0)
                else:
                    ref.append(0.0)
            return np.array(ref, dtype=float)

        if mem_available and 1 in dims:
            ys = [self._project_y(d["y"], dims) for d in self.dataset if d.get("mem_available")]
        else:
            ys = [self._project_y(d["y"], dims) for d in self.dataset]
        if not ys:
            return np.ones(len(dims), dtype=float)
        ys_arr = np.array(ys, dtype=float)
        y_max = np.max(ys_arr, axis=0)
        y_min = np.min(ys_arr, axis=0)
        margin = np.maximum(0.1 * np.abs(y_max - y_min), 1e-3)
        return y_max + margin

    def _predict_objectives(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xn = self._normalize_x(x.reshape(1, -1))
        mus = []
        sigs = []
        for model in self.obj_models:
            mu, sigma = model.predict(xn)
            mus.append(mu[0])
            sigs.append(sigma[0])
        return np.array(mus, dtype=float), np.array(sigs, dtype=float)

    def _predict_constraints(self, x: np.ndarray) -> Dict[str, Tuple[float, float]]:
        xn = self._normalize_x(x.reshape(1, -1))
        out: Dict[str, Tuple[float, float]] = {}
        if self.con_ready.get("g1"):
            mu, sigma = self.con_models["g1"].predict(xn)
            out["g1"] = (float(mu[0]), float(sigma[0]))
        if self.memory_max_gb is not None and self.con_ready.get("g2"):
            mu, sigma = self.con_models["g2"].predict(xn)
            out["g2"] = (float(mu[0]), float(sigma[0]))
        if self.quality_min is not None and self.con_ready.get("g3"):
            mu, sigma = self.con_models["g3"].predict(xn)
            out["g3"] = (float(mu[0]), float(sigma[0]))
        return out

    def _pof_total(self, x: np.ndarray) -> Tuple[float, Dict[str, float]]:
        if not self.models_ready:
            return 1.0, {}
        cons = self._predict_constraints(x)
        pofs = {}
        p_total = 1.0
        for key, (mu, sigma) in cons.items():
            sigma = max(sigma, 1e-6)
            z = (0.0 - mu) / sigma
            p = _normal_cdf(z)
            pofs[key] = p
            p_total *= p
        return p_total, pofs

    def _expected_hv_improvement(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        ref: np.ndarray,
        dims: List[int],
        pareto_y: List[np.ndarray],
    ) -> float:
        hv_fn = hypervolume_2d if len(dims) == 2 else hypervolume_3d
        base_hv = hv_fn(pareto_y, ref)
        if np.all(sigma < 1e-6):
            hv = hv_fn(pareto_y + [mu], ref)
            return max(0.0, hv - base_hv)

        improvements = []
        for _ in range(self.mc_samples):
            sample = mu + sigma * np.random.randn(*mu.shape)
            hv = hv_fn(pareto_y + [sample], ref)
            improvements.append(max(0.0, hv - base_hv))
        return float(np.mean(improvements)) if improvements else 0.0

    def _expected_hv_improvement_batch(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        ref: np.ndarray,
        dims: List[int],
        pareto_y: List[np.ndarray],
    ) -> float:
        if mus.ndim == 1:
            mus = mus[None, :]
            sigmas = sigmas[None, :]
        if mus.size == 0:
            return 0.0

        hv_fn = hypervolume_2d if len(dims) == 2 else hypervolume_3d
        base_hv = hv_fn(pareto_y, ref)
        if np.all(sigmas < 1e-6):
            points = pareto_y + [np.array(sample, dtype=float) for sample in mus]
            hv = hv_fn(points, ref)
            return max(0.0, hv - base_hv)

        improvements = []
        for _ in range(self.mc_samples):
            samples = mus + sigmas * np.random.randn(*mus.shape)
            points = pareto_y + [np.array(sample, dtype=float) for sample in samples]
            hv = hv_fn(points, ref)
            improvements.append(max(0.0, hv - base_hv))
        return float(np.mean(improvements)) if improvements else 0.0

    def _acquisition(self, x_relax: np.ndarray, ref: np.ndarray, dims: List[int], pareto_y: List[np.ndarray]) -> float:
        mu, sigma = self._predict_objectives(x_relax)
        mu = self._project_y(mu, dims)
        sigma = self._project_y(sigma, dims)
        ehvi = self._expected_hv_improvement(mu, sigma, ref, dims, pareto_y)
        pof, _ = self._pof_total(x_relax)
        return ehvi * pof

    def _batch_acquisition(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        pofs: List[float],
        ref: np.ndarray,
        dims: List[int],
        pareto_y: List[np.ndarray],
    ) -> float:
        ehvi = self._expected_hv_improvement_batch(mus, sigmas, ref, dims, pareto_y)
        # Numerically stable product of probabilities in log-space.
        if pofs:
            tiny = 1e-12
            p_arr = np.maximum(np.array(pofs, dtype=float), tiny)
            log_p = float(np.sum(np.log(p_arr)))
            # exp(log_p) is safe here because q_batch is typically small (2-3).
            p_total = float(np.exp(log_p))
        else:
            p_total = 1.0
        return ehvi * p_total

    def _greedy_batch(
        self,
        candidates: List[np.ndarray],
        ref: np.ndarray,
        dims: List[int],
        pareto_y: List[np.ndarray],
    ) -> List[np.ndarray]:
        if not candidates:
            return []

        cand_info = []
        for x in candidates:
            mu, sigma = self._predict_objectives(x)
            mu = self._project_y(mu, dims)
            sigma = self._project_y(sigma, dims)
            pof, _ = self._pof_total(x)
            # Enforce feasibility threshold early so greedy selection optimizes
            # joint EHVI inside the feasible region.
            if pof < self.delta:
                continue
            cand_info.append((x, mu, sigma, pof))

        if not cand_info:
            return []

        selected: List[np.ndarray] = []
        selected_mus: List[np.ndarray] = []
        selected_sigmas: List[np.ndarray] = []
        selected_pofs: List[float] = []

        remaining = cand_info
        for _ in range(self.q_batch):
            best_idx = None
            best_val = -float("inf")
            for idx, (x, mu, sigma, pof) in enumerate(remaining):
                mus = np.vstack(selected_mus + [mu]) if selected_mus else np.array([mu])
                sigmas = np.vstack(selected_sigmas + [sigma]) if selected_sigmas else np.array([sigma])
                pofs = selected_pofs + [pof]
                val = self._batch_acquisition(mus, sigmas, pofs, ref, dims, pareto_y)
                if val > best_val:
                    best_val = val
                    best_idx = idx
            if best_idx is None:
                break
            x, mu, sigma, pof = remaining.pop(best_idx)
            selected.append(x)
            selected_mus.append(mu)
            selected_sigmas.append(sigma)
            selected_pofs.append(pof)

        return selected

    def _optimize_acquisition(self, ref: np.ndarray, dims: List[int], pareto_y: List[np.ndarray]) -> List[np.ndarray]:
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        rng = ub - lb

        candidates = []
        for _ in range(self.raw_samples):
            x = lb + rng * np.random.rand(len(lb))
            candidates.append(x)

        scored = [(self._acquisition(x, ref, dims, pareto_y), x) for x in candidates]
        scored.sort(key=lambda v: v[0], reverse=True)
        seeds = [x for _, x in scored[: max(1, self.num_restarts)]]

        best = []
        for seed in seeds:
            x = seed.copy()
            best_val = self._acquisition(x, ref, dims, pareto_y)
            step = 0.15 * rng
            for _ in range(self.local_steps):
                noise = np.random.uniform(-1.0, 1.0, size=x.shape) * step
                cand = np.clip(x + noise, lb, ub)
                val = self._acquisition(cand, ref, dims, pareto_y)
                if val > best_val:
                    x, best_val = cand, val
            best.append((best_val, x))

        best.sort(key=lambda v: v[0], reverse=True)
        pool = [x for _, x in best]
        # Enlarge the candidate pool so greedy batch has enough diversity.
        # Otherwise, with small num_restarts (e.g., 5), batch may degenerate.
        K = max(self.q_batch * 4, self.num_restarts * 4, 16)
        pool.extend([x for _, x in scored[:K]])

        uniq: List[np.ndarray] = []
        seen = set()
        for x in pool:
            key = tuple(float(v) for v in x.tolist())
            if key not in seen:
                seen.add(key)
                uniq.append(x)

        return self._greedy_batch(uniq, ref, dims, pareto_y)

    def _round_and_repair(self, x_relax: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        x = np.array(x_relax, dtype=float)
        x = np.round(x)
        x[0] = np.clip(x[0], self.context_bounds[0], self.context_bounds[1])
        x[1] = np.clip(x[1], self.inference_bounds[0], self.inference_bounds[1])

        delta = x - x_ref
        for i in range(len(x)):
            if abs(delta[i]) > self.epsilon[i]:
                x[i] = x_ref[i] + math.copysign(self.epsilon[i], delta[i])

        if self.dwell_counter > 0:
            for i in range(len(x)):
                if self.high_impact_mask[i] and x[i] != x_ref[i]:
                    x[i] = x_ref[i]

        return x.astype(int)

    def _apply_warning_guard(self, x_next: np.ndarray, x_last: np.ndarray) -> np.ndarray:
        if x_next[0] > x_last[0] or x_next[1] > x_last[1]:
            return x_last.copy()
        return x_next

    def _propose_next(self, x_last: np.ndarray, obs: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        if not self.models_ready or len(self.dataset) < 3:
            return x_last.copy(), "hold_no_model", {}

        mem_available = bool(obs.get("mem_available"))
        dims = self._objective_dims(mem_available)
        pareto_y = self._pareto_y(dims)
        ref = self._reference_point(dims, mem_available)
        if self.use_reachable_enumeration:
            candidates = self._reachable_candidates(x_last)
        else:
            relax = self._optimize_acquisition(ref, dims, pareto_y)
            candidates = []
            for x_relax in relax:
                candidates.append(self._round_and_repair(x_relax, x_last))

        candidates.append(x_last.copy())
        if self.x_safe is not None:
            candidates.append(self.x_safe.copy())

        uniq: List[np.ndarray] = []
        seen = set()
        for x in candidates:
            key = tuple(int(v) for v in x.tolist())
            if key not in seen:
                seen.add(key)
                uniq.append(x)

        exploit, exploit_info = self._select_exploit(uniq)
        if exploit is None:
            fallback = self.x_safe.copy() if self.x_safe is not None else x_last.copy()
            return fallback, "fallback_safe", {
                **exploit_info,
                "candidate_count": len(uniq),
                "mode": self.mode,
            }

        decision = "exploit"
        chosen = exploit
        explore_info: Dict[str, Any] = {}

        if self._should_explore():
            explore, explore_info = self._select_explore(uniq, dims, ref, pareto_y, avoid=exploit)
            if explore is not None and self._explore_guard_ok(explore, exploit):
                chosen = explore
                decision = "explore"

        extra = {
            **exploit_info,
            **explore_info,
            "candidate_count": len(uniq),
            "mode": self.mode,
            "explore_enabled": self._should_explore(),
        }
        return chosen, decision, extra

    def _emergency_degrade(self, x_last: np.ndarray) -> np.ndarray:
        if self.x_safe is not None:
            return self.x_safe.copy()
        x = x_last.copy()
        if x[1] > self.inference_bounds[0]:
            x[1] -= 1
        elif x[0] > self.context_bounds[0]:
            x[0] -= 1
        return x

    def _apply_config(self, x_last: np.ndarray, x_next: np.ndarray, decision: str) -> None:
        if np.array_equal(x_next, x_last):
            if self.dwell_counter > 0:
                self.dwell_counter -= 1
            self.x_prev = x_last.copy()
            return

        changed = np.any(x_next != x_last)
        if changed and np.any(self.high_impact_mask & (x_next != x_last)):
            self.dwell_counter = self.t_dwell
        elif self.dwell_counter > 0:
            self.dwell_counter -= 1

        new_config = (int(x_next[0]), int(x_next[1]))
        if new_config != self.current_config_tuple:
            self.config_version += 1
            self.current_config_tuple = new_config
            if self.inference_actor_handle:
                self.inference_actor_handle.update_config.remote(
                    new_config[0], new_config[1], self.config_version
                )
                try:
                    self.inference_actor_handle.start_new_cycle_tracking.remote()
                except Exception:
                    pass
            if self.training_actor_handle:
                self.training_actor_handle.update_config.remote(
                    new_config[0], new_config[1], self.config_version
                )

        self.x_prev = x_next.copy()

    def _log(self, obs: Dict[str, Any], decision: str, new_config: np.ndarray, extra: Dict[str, Any]) -> None:
        log_data = {
            "scheduler/step": self.schedule_step,
            "scheduler/config_version": self.config_version,
            "scheduler/context_length": self.current_config_tuple[0],
            "scheduler/inference_length": self.current_config_tuple[1],
            "scheduler/decision": decision,
            "scheduler/new_context_length": int(new_config[0]),
            "scheduler/new_inference_length": int(new_config[1]),
            "scheduler/latency_p95": obs.get("latency_p95"),
            "scheduler/latency_mean": obs.get("latency_mean"),
            "scheduler/latency_std": obs.get("latency_std"),
            "scheduler/train_latency_mean": obs.get("train_latency_mean"),
            "scheduler/train_tps_mean": obs.get("train_tps_mean"),
            "scheduler/perplexity_mean": obs.get("perplexity_mean"),
            "scheduler/accuracy_mean": obs.get("accuracy_mean"),
            "scheduler/accuracy_total_predictions_mean": obs.get("accuracy_total_predictions_mean"),
            "scheduler/mem_peak_p95": obs.get("mem_peak_p95"),
            "scheduler/quality": obs.get("quality"),
            "scheduler/quality_std": obs.get("quality_std"),
            "scheduler/quality_source": obs.get("quality_source"),
            "scheduler/jsd_mean": obs.get("jsd_mean"),
            "scheduler/spec_accept_mean": obs.get("spec_accept_mean"),
            "scheduler/spec_reverify_per_step": obs.get("spec_reverify_per_step"),
            "scheduler/spec_draft_ms_per_step": obs.get("spec_draft_ms_per_step"),
            "scheduler/spec_verify_ms_per_step": obs.get("spec_verify_ms_per_step"),
            "scheduler/rejected_tokens_per_step": obs.get("rejected_tokens_per_step"),
            "scheduler/accepted_tokens_per_step": obs.get("accepted_tokens_per_step"),
            "scheduler/g1": obs.get("g1"),
            "scheduler/g2": obs.get("g2"),
            "scheduler/g3": obs.get("g3"),
            "scheduler/feasible": obs.get("feasible"),
            "scheduler/risk_level": obs.get("risk_level"),
            "scheduler/dataset_size": obs.get("dataset_size"),
            "scheduler/pareto_2d_size": obs.get("pareto_2d_size"),
            "scheduler/pareto_3d_size": obs.get("pareto_3d_size"),
            "scheduler/dwell_counter": self.dwell_counter,
            "scheduler/mode": getattr(self, "mode", None),
            "scheduler/sla_latency": self.sla_latency,
            "scheduler/delta": self.delta,
            "scheduler/quality_min": self.quality_min,
            "scheduler/warning_ratio": self.warning_ratio,
            "scheduler/critical_ratio": self.critical_ratio,
        }

        if extra:
            if extra.get("latency_scale_warning") is not None:
                log_data["scheduler/latency_scale_warning"] = extra.get("latency_scale_warning")
            if extra.get("latency_scale_ratio") is not None:
                log_data["scheduler/latency_scale_ratio"] = extra.get("latency_scale_ratio")
            if extra.get("quality_scale_warning") is not None:
                log_data["scheduler/quality_scale_warning"] = extra.get("quality_scale_warning")
            if extra.get("delta_high_warning") is not None:
                log_data["scheduler/delta_high_warning"] = extra.get("delta_high_warning")
            if extra.get("pof") is not None:
                log_data["scheduler/pof"] = extra.get("pof")
            pof_map = extra.get("pof_map") or {}
            for k, v in pof_map.items():
                log_data[f"scheduler/{k}_pof"] = v
            if extra.get("acq_selected") is not None:
                log_data["scheduler/acq_selected"] = extra.get("acq_selected")
            if extra.get("acq_best") is not None:
                log_data["scheduler/acq_best"] = extra.get("acq_best")
            if extra.get("utility_selected") is not None:
                log_data["scheduler/utility_selected"] = extra.get("utility_selected")
            if extra.get("score_selected") is not None:
                log_data["scheduler/score_selected"] = extra.get("score_selected")
            if extra.get("mix_lambda") is not None:
                log_data["scheduler/mix_lambda"] = extra.get("mix_lambda")
            if extra.get("selection_policy") is not None:
                log_data["scheduler/selection_policy"] = extra.get("selection_policy")
            if extra.get("candidate_count") is not None:
                log_data["scheduler/candidate_count"] = extra.get("candidate_count")
            if extra.get("feasible_count") is not None:
                log_data["scheduler/feasible_count"] = extra.get("feasible_count")
            # Keep all scheduler_Shift meta fields for post-hoc analysis.
            for key, value in extra.items():
                log_key = key if str(key).startswith("scheduler/") else f"scheduler/{key}"
                if log_key in log_data:
                    continue
                try:
                    json.dumps(value)
                    log_data[log_key] = value
                except Exception:
                    log_data[log_key] = str(value)

        if self.wandb_run and wandb is not None:
            try:
                self.wandb_run.log(log_data)
            except Exception:
                pass

        try:
            log_entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **log_data}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
