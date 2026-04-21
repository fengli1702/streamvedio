#!/usr/bin/env python3
import argparse
import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"ctx(?P<ctx>\d+)_inf(?P<inf>\d+)")
ALT_RUN_RE = re.compile(r"_c(?P<ctx>\d+)_i(?P<inf>\d+)")


def parse_run_meta(run_name: str):
    m = RUN_RE.search(run_name)
    if not m:
        m = ALT_RUN_RE.search(run_name)
    spec = None
    if "nospec" in run_name:
        spec = "nospec"
    elif "spec" in run_name:
        spec = "spec"

    if not m:
        return {"run_name": run_name, "ctx": None, "inf": None, "spec": spec}

    return {
        "run_name": run_name,
        "ctx": int(m.group("ctx")),
        "inf": int(m.group("inf")),
        "spec": spec,
    }


def read_log(path):
    rows = []
    fallback_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if "accuracy" not in obj or "latency" not in obj:
                continue

            token_ids = obj.get("token_ids")
            if token_ids is None:
                continue

            fallback_idx += 1
            frame_id = obj.get("frame_index")
            if frame_id is None:
                frame_id = fallback_idx

            try:
                acc = float(obj.get("accuracy"))
                lat = float(obj.get("latency"))
            except Exception:
                continue

            rows.append(
                {
                    "frame_id": int(frame_id),
                    "frame_id0": int(frame_id) - 1 if int(frame_id) >= 1 else int(frame_id),
                    "token_ids": token_ids,
                    "accuracy": acc,
                    "latency": lat,
                }
            )

    rows.sort(key=lambda r: r["frame_id0"])
    return rows


def token_change_rate(prev_ids, cur_ids):
    a = np.asarray(prev_ids)
    b = np.asarray(cur_ids)
    if a.size == 0 or b.size == 0:
        return np.nan
    n = min(a.size, b.size)
    if n == 0:
        return np.nan
    a = a[:n]
    b = b[:n]
    return float(np.mean(a != b))


def add_discontinuity_flags(df: pd.DataFrame, key_cols):
    if df.empty:
        return df
    out = df.sort_values("window_id").copy()
    key = out[key_cols[0]].astype(str)
    for col in key_cols[1:]:
        key = key + "_" + out[col].astype(str)
    out["discontinuity"] = key.ne(key.shift(1))
    if len(out) > 0:
        out.loc[out.index[0], "discontinuity"] = False
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot token drift (top) and best latency under acc>=tau (bottom)")
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--pattern", default="12-20-clean+clean1_base_*_nospec_static.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--split", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(args.logs_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {args.pattern} in {args.logs_dir}")

    # ---- Build per-frame token drift summary ----
    frame_drift_rows = []
    window_metric_rows = []

    for fp in files:
        run_name = os.path.splitext(os.path.basename(fp))[0]
        meta = parse_run_meta(run_name)
        rows = read_log(fp)
        if not rows:
            continue

        # token drift per frame (within-run adjacent-frame token change rate)
        for i in range(1, len(rows)):
            d = token_change_rate(rows[i - 1]["token_ids"], rows[i]["token_ids"])
            frame_drift_rows.append(
                {
                    "run_name": run_name,
                    "frame_idx": rows[i]["frame_id0"],
                    "token_drift": d,
                }
            )

        # window stats for best-point selection
        run_df = pd.DataFrame(
            {
                "frame_id0": [r["frame_id0"] for r in rows],
                "accuracy": [r["accuracy"] for r in rows],
                "latency": [r["latency"] for r in rows],
            }
        )
        run_df["window_id"] = (run_df["frame_id0"] // args.window).astype(int)

        for win_id, sub in run_df.groupby("window_id"):
            if sub.empty:
                continue
            window_metric_rows.append(
                {
                    "run_name": run_name,
                    "ctx": meta["ctx"],
                    "inf": meta["inf"],
                    "spec": meta["spec"],
                    "window_id": int(win_id),
                    "frame_start": int(win_id * args.window),
                    "frame_end": int(win_id * args.window + args.window - 1),
                    "acc_mean": float(sub["accuracy"].mean()),
                    "lat_mean": float(sub["latency"].mean()),
                }
            )

    if not frame_drift_rows:
        raise SystemExit("No token drift rows available (missing token_ids?)")
    if not window_metric_rows:
        raise SystemExit("No window metrics available")

    drift_df = pd.DataFrame(frame_drift_rows)
    drift_summary = (
        drift_df.groupby("frame_idx")["token_drift"]
        .agg(token_drift_mean="mean", token_drift_std="std")
        .reset_index()
        .sort_values("frame_idx")
    )

    win_metrics = pd.DataFrame(window_metric_rows)

    # ---- Best latency per window under acc>=tau ----
    tau_rows = []
    for win_id, sub in win_metrics.groupby("window_id"):
        feasible = sub[sub["acc_mean"] >= args.tau]
        feasible_flag = 1
        if feasible.empty:
            feasible = sub.copy()
            feasible_flag = 0

        min_lat = feasible["lat_mean"].min()
        tied = feasible[np.isclose(feasible["lat_mean"], min_lat)]
        tied = tied.sort_values(["run_name"])  # deterministic tie-break
        pick = tied.iloc[0]

        tau_rows.append(
            {
                "window_id": int(win_id),
                "frame_start": int(pick["frame_start"]),
                "frame_end": int(pick["frame_end"]),
                "time_center": (int(pick["frame_start"]) + int(pick["frame_end"])) / 2,
                "acc_mean": float(pick["acc_mean"]),
                "lat_mean": float(pick["lat_mean"]),
                "run_name": str(pick["run_name"]),
                "ctx": int(pick["ctx"]) if pd.notna(pick["ctx"]) else -1,
                "inf": int(pick["inf"]) if pd.notna(pick["inf"]) else -1,
                "spec": str(pick["spec"]) if pd.notna(pick["spec"]) else "none",
                "feasible": feasible_flag,
                "tau": args.tau,
            }
        )

    tau_df = pd.DataFrame(tau_rows).sort_values("window_id")
    tau_df = add_discontinuity_flags(tau_df, ["ctx", "inf", "spec"])

    # ---- Plot (same style as JS figure, but top is token drift) ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    xs = drift_summary["frame_idx"].to_numpy()
    ys = drift_summary["token_drift_mean"].to_numpy()
    ax_top.plot(xs, ys, color="#e52b2b", linewidth=1.8)
    ax_top.set_ylabel("token drift (↑ drift)")
    ax_top.grid(alpha=0.25)

    # split cue
    ax_top.axvline(args.split, color="#666", linestyle="--", linewidth=1.5, alpha=0.8)

    # switch cues: red vertical lines at config switches
    switch_x = tau_df.loc[tau_df["discontinuity"], "time_center"].to_list()
    for sx in switch_x:
        ax_top.axvline(sx, color="red", linewidth=1.2, alpha=0.85)

    # directional annotation like the reference
    ax_top.annotate(
        "",
        xy=(0.025, 0.88),
        xytext=(0.025, 0.10),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#666", lw=1.4),
    )
    ax_top.text(0.05, 0.90, "high token drift => larger drift", transform=ax_top.transAxes, fontsize=12, va="center")
    ax_top.text(0.05, 0.03, "low token drift => smaller drift", transform=ax_top.transAxes, fontsize=12, va="center")

    # bottom: latency points + labels
    ax_bot.scatter(tau_df["time_center"], tau_df["lat_mean"], color="#ff7f0e", s=28, zorder=3)

    # red square for discontinuities
    sw = tau_df[tau_df["discontinuity"]]
    if not sw.empty:
        ax_bot.scatter(
            sw["time_center"],
            sw["lat_mean"],
            facecolors="none",
            edgecolors="red",
            marker="s",
            s=90,
            linewidths=1.5,
            zorder=4,
        )

    for _, row in tau_df.iterrows():
        if row["spec"] in ("none", "nan"):
            label = f"ctx{int(row['ctx'])}_inf{int(row['inf'])}"
        else:
            label = f"ctx{int(row['ctx'])}_inf{int(row['inf'])}_{row['spec']}"
        ax_bot.text(row["time_center"], row["lat_mean"], label, fontsize=11, rotation=35, ha="left", va="bottom", alpha=0.9)

    for sx in switch_x:
        ax_bot.axvline(sx, color="red", linewidth=1.2, alpha=0.85)

    ax_bot.axvline(args.split, color="#666", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_bot.grid(alpha=0.25)
    ax_bot.set_ylabel("latency (↓ lower better)")
    ax_bot.set_xlabel("frame index")

    # title + footer stats
    title_spec = "(ctx,inf)"
    ax_top.set_title(f"Token drift (top) and best latency under acc>={args.tau} (bottom) {title_spec}", fontsize=18)

    seg1 = drift_summary[drift_summary["frame_idx"] < args.split]["token_drift_mean"].mean()
    seg2 = drift_summary[drift_summary["frame_idx"] >= args.split]["token_drift_mean"].mean()
    x_max = int(drift_summary["frame_idx"].max()) + 1
    fig.text(
        0.01,
        0.01,
        f"token drift mean [0,{args.split}) = {seg1:.4f}  |  token drift mean [{args.split},{x_max}) = {seg2:.4f}",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    tau_tag = str(args.tau).replace(".", "p")
    out_png = os.path.join(args.out_dir, f"token_drift_vs_best_latency_tau{tau_tag}.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # also write key tables
    drift_summary.to_csv(os.path.join(args.out_dir, "token_drift_summary.csv"), index=False)
    tau_df.to_csv(os.path.join(args.out_dir, f"best_point_per_window_tau{tau_tag}.csv"), index=False)

    print("[OK] wrote:", out_png)


if __name__ == "__main__":
    main()
