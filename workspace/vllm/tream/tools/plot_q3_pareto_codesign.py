#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RE = re.compile(
    r"^doh_cod_1\.22_(?P<mode>joint|train_only|infer_only)_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_spec(?P<spec>on|off)$"
)


def parse_run(name: str):
    m = RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["ctx"] = int(d["ctx"])
    d["inf"] = int(d["inf"])
    d["ib"] = int(d["ib"])
    d["tb"] = int(d["tb"])
    return d


def get_col(row, *cols):
    for c in cols:
        if c in row and pd.notna(row[c]):
            return row[c]
    return None


def pareto_front(points: List[Tuple[float, float]]):
    # minimize x (latency), maximize y (accuracy)
    pts = sorted(points, key=lambda v: v[0])
    best = -1e18
    front = []
    for x, y in pts:
        if y > best:
            front.append((x, y))
            best = y
    return front


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)

    for i in range(1, n - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    if n == 2:
        m[0] = delta[0]
        m[1] = delta[0]
        return m
    if n >= 3:
        m0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        if np.sign(m0) != np.sign(delta[0]):
            m0 = 0.0
        elif (np.sign(delta[0]) != np.sign(delta[1])) and (abs(m0) > abs(3 * delta[0])):
            m0 = 3 * delta[0]
        m[0] = m0

        mn = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
        if np.sign(mn) != np.sign(delta[-1]):
            mn = 0.0
        elif (np.sign(delta[-1]) != np.sign(delta[-2])) and (abs(mn) > abs(3 * delta[-1])):
            mn = 3 * delta[-1]
        m[-1] = mn
    return m


def pchip_curve(xs, ys, points_per_seg=30):
    if len(xs) < 2:
        return xs, ys
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)

    uniq = {}
    for xi, yi in zip(x, y):
        uniq[xi] = max(uniq.get(xi, -1e18), yi)
    x = np.array(sorted(uniq.keys()), dtype=float)
    y = np.array([uniq[xi] for xi in x], dtype=float)

    if len(x) < 2:
        return x.tolist(), y.tolist()

    m = _pchip_slopes(x, y)
    xs_new = []
    ys_new = []
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        t = np.linspace(0, 1, points_per_seg, endpoint=False)
        h00 = (2 * t**3 - 3 * t**2 + 1)
        h10 = (t**3 - 2 * t**2 + t)
        h01 = (-2 * t**3 + 3 * t**2)
        h11 = (t**3 - t**2)
        yi = h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]
        xi = x[i] + t * h
        xs_new.extend(xi.tolist())
        ys_new.extend(yi.tolist())

    xs_new.append(float(x[-1]))
    ys_new.append(float(y[-1]))
    return xs_new, ys_new


def plot(df, spec: str, out_path: str, ctx: int, inf: int, title_tag: str = ""):
    colors = {"infer_only": "#1f77b4", "train_only": "#ff7f0e", "joint": "#2ca02c"}
    markers = {"infer_only": "o", "train_only": "s", "joint": "D"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for mode in ["infer_only", "train_only", "joint"]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(sub["lat_ms"], sub["acc"], c=colors[mode], marker=markers[mode],
                   edgecolors="black", linewidths=0.6, label=mode)

        # annotate (ib,tb)
        for _, r in sub.iterrows():
            ax.text(r["lat_ms"], r["acc"], f"({r['ib']},{r['tb']})",
                    fontsize=7, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        # pareto front curve
        pts = list(zip(sub["lat_ms"], sub["acc"]))
        front = pareto_front(pts)
        if front:
            fx, fy = zip(*front)
            sx, sy = pchip_curve(list(fx), list(fy), points_per_seg=30)
            ax.plot(sx, sy, color=colors[mode], linewidth=2.0, alpha=0.9)

    ax.set_xlabel("latency (ms)  (lower is better)")
    ax.set_ylabel("accuracy  (higher is better)")
    ax.set_title(f"Q3: latency vs accuracy + Pareto fronts (spec {spec}, c{ctx}_i{inf}){title_tag}\nlabels = (ib, tb)")

    # axis arrows for direction
    ax.annotate("", xy=(0.02, 0.95), xytext=(0.02, 0.15), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(0.025, 0.96, "higher better", transform=ax.transAxes, fontsize=8, va="bottom")

    ax.annotate("", xy=(0.05, 0.08), xytext=(0.85, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(0.86, 0.065, "lower better", transform=ax.transAxes, fontsize=8, va="top")

    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ctx", type=int, default=4)
    ap.add_argument("--inf", type=int, default=4)
    ap.add_argument("--spec", choices=["off", "on", "both"], default="off")
    ap.add_argument("--title_tag", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    for _, r in df.iterrows():
        name = r.get("run_name")
        if not isinstance(name, str):
            continue
        meta = parse_run(name)
        if not meta:
            continue
        if meta["ctx"] != args.ctx or meta["inf"] != args.inf:
            continue
        lat = get_col(r, "scheduler/latency_mean", "inference/latency", "scheduler/latency_p95")
        acc = get_col(r, "inference/global_accuracy", "scheduler/accuracy_mean")
        if lat is None or acc is None:
            continue
        rows.append({
            "mode": meta["mode"],
            "spec": meta["spec"],
            "ctx": meta["ctx"],
            "inf": meta["inf"],
            "ib": meta["ib"],
            "tb": meta["tb"],
            "lat_ms": float(lat) * 1000.0,
            "acc": float(acc),
        })

    if not rows:
        raise SystemExit("No matching rows")

    all_df = pd.DataFrame(rows)
    specs = ["off", "on"] if args.spec == "both" else [args.spec]

    for spec in specs:
        sub = all_df[all_df["spec"] == spec]
        if sub.empty:
            continue
        out = os.path.join(args.out_dir, f"q3_pareto_spec{spec}_c{args.ctx}_i{args.inf}.png")
        plot(sub, spec, out, args.ctx, args.inf, title_tag=args.title_tag)


if __name__ == "__main__":
    main()
