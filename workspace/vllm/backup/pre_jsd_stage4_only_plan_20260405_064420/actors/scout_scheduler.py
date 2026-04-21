"""
SCOUT (Shape-Constrained Online Utility Tuning) scheduler.

This is a standalone, pure-Python reference implementation of the spec.
It is not wired into the actors yet; integrate by instantiating ScoutScheduler
and calling run_scheduler_tick(raw_metrics) every CONTROL_INTERVAL with a list
of per-frame/batch metric dicts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore

# ---------------------------
# Hyperparameters (defaults)
# ---------------------------
WINDOW_SIZE = 150  # smaller buffer so history fills sooner
WARMUP_DROP = 3  # shorter warmup to avoid long waits after a switch
BOOTSTRAP_ROUNDS = 10  # fewer bootstrap fits needed to start acting
KAPPA = 2.0
LAMBDA_SW = 0.5
EPSILON_GAIN = 0.05
INIT_JUMP = 2  # cap how far the forced init sweep can jump per dim

# SLA thresholds (set from caller as needed)
A_MIN = 0.0
S_MAX = float("inf")
G_MIN = 0.0
L_MAX = float("inf")

# Bounds for action dimensions (set from caller)
MIN_C, MAX_C = 1, 16
MIN_I, MAX_I = 1, 16
MIN_K, MAX_K = 1, 16


@dataclass(frozen=True)
class Action:
    c: int  # context length
    i: int  # inference length
    k: int  # sync period

    def step(self, dim: str, delta: int) -> "Action":
        if dim == "c":
            new_c = int(np.clip(self.c + delta, MIN_C, MAX_C))
            return Action(new_c, self.i, self.k)
        if dim == "i":
            new_i = int(np.clip(self.i + delta, MIN_I, MAX_I))
            return Action(self.c, new_i, self.k)
        if dim == "k":
            new_k = int(np.clip(self.k + delta, MIN_K, MAX_K))
            return Action(self.c, self.i, new_k)
        raise ValueError(f"Unknown dim {dim}")


@dataclass
class Outcome:
    latency_p95: float
    accuracy_mean: float
    staleness_p95: float
    goodput: float


class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Any] = deque(maxlen=capacity)

    def append(self, item: Any) -> None:
        self.buf.append(item)

    def __len__(self) -> int:
        return len(self.buf)

    def to_list(self) -> List[Any]:
        return list(self.buf)


# ---------------------------
# Helper functions
# ---------------------------
def drop_first_n(metrics: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return metrics[n:] if len(metrics) > n else []


def aggregate(metrics: List[Dict[str, Any]]) -> Outcome:
    def _finite(vals):
        out = []
        for v in vals:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                out.append(fv)
        return np.array(out, dtype=float)

    lat = _finite([m.get("latency") for m in metrics])
    acc = _finite([m.get("accuracy") for m in metrics])
    stl = _finite([m.get("staleness_steps") for m in metrics])
    gpt = _finite([m.get("goodput") for m in metrics])
    return Outcome(
        latency_p95=float(np.percentile(lat, 95)) if lat.size else float("inf"),
        accuracy_mean=float(np.mean(acc)) if acc.size else float("nan"),
        staleness_p95=float(np.percentile(stl, 95)) if stl.size else 0.0,
        goodput=float(np.mean(gpt)) if gpt.size else 0.0,
    )


def resample(history: List[Tuple[Action, Outcome]], n: int) -> List[Tuple[Action, Outcome]]:
    idx = np.random.randint(0, len(history), size=n)
    return [history[i] for i in idx]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

class StitchedUnimodal:
    def __init__(self, left_model, right_model, peak):
        self.left = left_model
        self.right = right_model
        self.peak = peak

    def predict(self, x: List[float]) -> np.ndarray:
        x = np.array(x)
        mask_left = x <= self.peak
        preds = np.zeros_like(x, dtype=float)
        if mask_left.any():
            preds[mask_left] = self.left.predict(x[mask_left])
        if (~mask_left).any():
            preds[~mask_left] = self.right.predict(x[~mask_left])
        return preds


def fit_unimodal_1d(x_train: np.ndarray, y_train: np.ndarray) -> StitchedUnimodal:
    unique_x = np.unique(x_train)
    best_model = None
    min_error = math.inf
    if IsotonicRegression is None:
        # Fallback: simple quadratic fit as a crude unimodal proxy
        coeffs = np.polyfit(x_train, y_train, deg=2)
        model = lambda xs: np.polyval(coeffs, xs)
        class PolyWrapper:
            def predict(self, xs): return model(xs)
        return StitchedUnimodal(PolyWrapper(), PolyWrapper(), peak=float(np.median(unique_x)))

    for p in unique_x:
        mask_left = x_train <= p
        mask_right = x_train >= p
        if mask_left.sum() < 2 or mask_right.sum() < 2:
            continue
        model_left = IsotonicRegression(increasing=True).fit(x_train[mask_left], y_train[mask_left])
        model_right = IsotonicRegression(increasing=False).fit(x_train[mask_right], y_train[mask_right])
        pred_left = model_left.predict(x_train[mask_left])
        pred_right = model_right.predict(x_train[mask_right])
        y_pred = np.concatenate([pred_left, pred_right])
        y_true = np.concatenate([y_train[mask_left], y_train[mask_right]])
        err = mse(y_true, y_pred)
        if err < min_error:
            min_error = err
            best_model = StitchedUnimodal(model_left, model_right, peak=p)
    return best_model


class ShapeConstrainedGAM:
    def __init__(self, constraint_type: str):
        self.type = constraint_type
        self.iso_c = IsotonicRegression(increasing=True) if IsotonicRegression else None
        self.iso_k = IsotonicRegression(increasing=True) if IsotonicRegression else None
        self.iso_i = None

    def fit(self, data: List[Tuple[Action, Outcome]], target_field: str):
        C = np.array([d[0].c for d in data], dtype=float)
        I = np.array([d[0].i for d in data], dtype=float)
        K = np.array([d[0].k for d in data], dtype=float)
        Y = np.array([getattr(d[1], target_field) for d in data], dtype=float)

        if self.iso_c:
            self.iso_c.fit(C, Y)
            resid_1 = Y - self.iso_c.predict(C)
        else:
            resid_1 = Y - np.mean(Y)

        if self.type == "monotonic":
            if IsotonicRegression:
                self.iso_i = IsotonicRegression(increasing=True)
                self.iso_i.fit(I, resid_1)
            else:
                coeff = np.polyfit(I, resid_1, deg=1)
                self.iso_i = lambda xs: np.polyval(coeff, xs)
                class LinWrapper:
                    def __init__(self, fn): self.fn=fn
                    def predict(self, xs): return self.fn(xs)
                self.iso_i = LinWrapper(self.iso_i)
        elif self.type == "unimodal":
            self.iso_i = fit_unimodal_1d(I, resid_1)
        else:
            raise ValueError(f"Unknown constraint type {self.type}")

        resid_2 = resid_1 - self.iso_i.predict(I)
        if self.iso_k:
            self.iso_k.fit(K, resid_2)
        else:
            self.iso_k = lambda xs: np.full_like(xs, np.mean(resid_2))
            class ConstWrapper:
                def __init__(self, fn): self.fn=fn
                def predict(self, xs): return self.fn(xs)
            self.iso_k = ConstWrapper(self.iso_k)

    def predict(self, x: Action) -> float:
        c_term = self.iso_c.predict([x.c])[0] if self.iso_c else 0.0
        i_term = self.iso_i.predict([x.i])[0] if self.iso_i else 0.0
        k_term = self.iso_k.predict([x.k])[0] if self.iso_k else 0.0
        return c_term + i_term + k_term


def fit_bootstrap_ensemble(history: List[Tuple[Action, Outcome]], target: str, constraint: str):
    ensemble = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = resample(history, n=len(history))
        model = ShapeConstrainedGAM(constraint_type=constraint)
        model.fit(sample, target_field=target)
        ensemble.append(model)
    return ensemble


def predict_ensemble(ensemble, x: Action) -> Tuple[float, float]:
    preds = np.array([m.predict(x) for m in ensemble], dtype=float)
    return float(np.mean(preds)), float(np.std(preds))


def generate_candidates(x_curr: Action) -> List[Action]:
    candidates = set()
    for dim in ["c", "i", "k"]:
        candidates.add(x_curr.step(dim, +1))
        candidates.add(x_curr.step(dim, -1))
    candidates.add(Action(MIN_C, MIN_I, MIN_K))
    candidates.add(Action(MAX_C, (MIN_I + MAX_I) // 2, (MIN_K + MAX_K) // 2))
    if random.random() < 0.2:
        candidates.add(
            Action(
                random.randint(MIN_C, MAX_C),
                random.randint(MIN_I, MAX_I),
                random.randint(MIN_K, MAX_K),
            )
        )
    return list(candidates)


def calculate_switch_cost(x_new: Action, x_old: Action) -> float:
    w_c = w_i = w_k = 1.0
    cost = 0.0
    cost += w_c * abs(x_new.c - x_old.c) / max(1, MAX_C - MIN_C)
    cost += w_i * abs(x_new.i - x_old.i) / max(1, MAX_I - MIN_I)
    cost += w_k * abs(x_new.k - x_old.k) / max(1, MAX_K - MIN_K)
    return cost


class ScoutScheduler:
    def __init__(self, x_default: Action, x_fallback: Optional[Action] = None):
        self.history = RingBuffer(capacity=WINDOW_SIZE)
        self.x_current = x_default
        self.x_fallback = x_fallback or Action(MIN_C, MIN_I, MIN_K)
        self.steps_since_last_switch = 0
        self.debug_logs: List[Dict[str, Any]] = []
        self.last_debug: Optional[Dict[str, Any]] = None
        # Forced exploration seed
        self.ticks = 0
        def _clamp_step(curr: int, target: int, lo: int, hi: int) -> int:
            if target > curr:
                return min(curr + INIT_JUMP, target, hi)
            if target < curr:
                return max(curr - INIT_JUMP, target, lo)
            return curr

        # Gentler init sweep: stay local, nudge c and i separately, then both.
        self.init_plan = [
            x_default,
            Action(_clamp_step(x_default.c, MAX_C, MIN_C, MAX_C), x_default.i, x_default.k),
            Action(x_default.c, _clamp_step(x_default.i, MAX_I, MIN_I, MAX_I), x_default.k),
            Action(
                _clamp_step(x_default.c, MAX_C, MIN_C, MAX_C),
                _clamp_step(x_default.i, MAX_I, MIN_I, MAX_I),
                x_default.k,
            ),
        ]
        self.init_idx = 0
        self.explore_every = 8

    def run_warmup_policy(self) -> Action:
        # Simple warmup: stay or try a cheap point
        return self.x_current

    def run_scheduler_tick(self, raw_metrics_buffer: List[Dict[str, Any]]) -> Action:
        self.ticks += 1
        dbg: Dict[str, Any] = {
            "tick": self.ticks,
            "x_current": {"c": self.x_current.c, "i": self.x_current.i, "k": self.x_current.k},
            "history_len": len(self.history),
            "unique_actions": len({(h[0].c, h[0].i, h[0].k) for h in self.history.to_list()}),
        }
        # Force initial sweep to get diverse samples
        if self.init_idx < len(self.init_plan):
            self.x_current = self.init_plan[self.init_idx]
            self.init_idx += 1
            dbg.update(
                {
                    "decision": "init_sweep",
                    "x_next": {"c": self.x_current.c, "i": self.x_current.i, "k": self.x_current.k},
                    "safe_set": 0,
                }
            )
            self.debug_logs.append(dbg)
            self.last_debug = dbg
            return self.x_current

        clean_metrics = drop_first_n(raw_metrics_buffer, n=WARMUP_DROP)
        if not clean_metrics:
            dbg.update({"decision": "warmup_wait_metrics", "safe_set": 0})
            self.debug_logs.append(dbg)
            self.last_debug = dbg
            return self.x_current

        y_obs = aggregate(clean_metrics)
        self.history.append((self.x_current, y_obs))
        dbg["history_len"] = len(self.history)
        dbg["unique_actions"] = len({(h[0].c, h[0].i, h[0].k) for h in self.history.to_list()})

        if len(self.history) < max(20, BOOTSTRAP_ROUNDS):
            x_next = self.run_warmup_policy()
            dbg.update(
                {
                    "decision": "warmup_policy",
                    "x_next": {"c": x_next.c, "i": x_next.i, "k": x_next.k},
                    "safe_set": 0,
                }
            )
            self.debug_logs.append(dbg)
            self.last_debug = dbg
            return x_next

        H = self.history.to_list()
        Models_L = fit_bootstrap_ensemble(H, target="latency_p95", constraint="monotonic")
        Models_A = fit_bootstrap_ensemble(H, target="accuracy_mean", constraint="unimodal")
        Models_S = fit_bootstrap_ensemble(H, target="staleness_p95", constraint="monotonic")
        curr_mu, curr_std = predict_ensemble(Models_L, self.x_current)
        curr_l_ucb = curr_mu + KAPPA * curr_std

        candidates = generate_candidates(self.x_current)
        safe_set = []
        for x in candidates:
            mu_l, std_l = predict_ensemble(Models_L, x)
            mu_a, std_a = predict_ensemble(Models_A, x)
            mu_s, std_s = predict_ensemble(Models_S, x)
            L_ucb = mu_l + KAPPA * std_l
            A_lcb = mu_a - KAPPA * std_a
            S_ucb = mu_s + KAPPA * std_s
            # Safe set (paper-friendly): hard constraints on UCB/LCB.
            #   UCB(latency) <= L_MAX
            #   UCB(staleness) <= S_MAX
            #   LCB(accuracy) >= A_MIN
            if (
                (math.isfinite(L_ucb) and L_ucb <= L_MAX)
                and (math.isfinite(A_lcb) and A_lcb >= A_MIN)
                and (math.isfinite(S_ucb) and S_ucb <= S_MAX)
            ):
                switch_cost = calculate_switch_cost(x, self.x_current)
                objective = L_ucb + LAMBDA_SW * switch_cost
                safe_set.append((x, objective, L_ucb, std_l, std_a, std_s))

        if not safe_set:
            x_next = self.x_fallback
            decision = "fallback"
            best_l_ucb = None
            epsilon_blocked = False
        else:
            # Periodic exploration: pick highest uncertainty
            if self.explore_every and (self.ticks % self.explore_every == 0):
                x_next = max(safe_set, key=lambda t: t[3] + t[4] + t[5])[0]
                decision = "explore_uncertainty"
                best_l_ucb = None
                epsilon_blocked = False
            else:
                x_next, best_obj, best_l_ucb, _, _, _ = min(safe_set, key=lambda item: item[1])
                epsilon_blocked = (curr_l_ucb - best_l_ucb) < EPSILON_GAIN
                if epsilon_blocked:
                    x_next = self.x_current
                    decision = "do_no_harm"
                else:
                    decision = "greedy_best"

        if x_next != self.x_current:
            self.steps_since_last_switch = 0
        else:
            self.steps_since_last_switch += 1
        self.x_current = x_next
        dbg.update(
            {
                "decision": decision,
                "x_next": {"c": x_next.c, "i": x_next.i, "k": x_next.k},
                "safe_set": len(safe_set),
                "curr_l_ucb": curr_l_ucb,
                "best_l_ucb": best_l_ucb,
                "epsilon_blocked": epsilon_blocked,
            }
        )
        if safe_set:
            best_candidate = min(safe_set, key=lambda item: item[1])[0]
            dbg["best_candidate"] = {"c": best_candidate.c, "i": best_candidate.i, "k": best_candidate.k}
        self.debug_logs.append(dbg)
        self.last_debug = dbg
        return x_next
