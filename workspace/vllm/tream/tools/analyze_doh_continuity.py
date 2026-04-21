#!/usr/bin/env python3
import argparse
import json
import os
import re
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"ctx(?P<ctx>\d+)_inf(?P<inf>\d+)")
ALT_RUN_RE = re.compile(r"_c(?P<ctx>\d+)_i(?P<inf>\d+)")
IBTB_RE = re.compile(r"_ib(?P<ib>\d+)_tb(?P<tb>\d+)")


def parse_run_meta(run_name: str):
    m = RUN_RE.search(run_name)
    spec = None
    if "spec" in run_name:
        spec = "spec" if "_spec_" in run_name or "spec" in run_name else None
    if "nospec" in run_name:
        spec = "nospec"
    if not m:
        m = ALT_RUN_RE.search(run_name)
    if not m:
        return {"run_name": run_name, "ctx": None, "inf": None, "ib": None, "tb": None, "spec": spec}
    ibtb = IBTB_RE.search(run_name)
    return {
        "run_name": run_name,
        "ctx": int(m.group("ctx")),
        "inf": int(m.group("inf")),
        "ib": int(ibtb.group("ib")) if ibtb else None,
        "tb": int(ibtb.group("tb")) if ibtb else None,
        "spec": spec,
    }


def read_log(path):
    rows = []
    fallback_idx = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "accuracy" not in obj or "latency" not in obj:
                continue
            acc_val = obj.get("accuracy")
            lat_val = obj.get("latency")
            if acc_val is None or lat_val is None:
                continue
            fallback_idx += 1
            frame_id = obj.get("frame_index")
            if frame_id is None:
                # For logs without frame_index (e.g., nospec_static), use line order
                frame_id = fallback_idx
            rows.append({
                "frame_id": int(frame_id),
                "token_ids": obj.get("token_ids"),
                "accuracy": float(acc_val),
                "latency": float(lat_val),
            })
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["frame_id"])
    return rows


def change_rate(prev_ids, cur_ids):
    # assumes same length
    a = np.array(prev_ids)
    b = np.array(cur_ids)
    if a.size == 0 or b.size == 0:
        return np.nan
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    return float(np.mean(a != b))


def jensen_shannon_divergence(counter_a, counter_b):
    # JS divergence on token histogram; using natural log
    keys = set(counter_a.keys()) | set(counter_b.keys())
    if not keys:
        return np.nan
    a = np.array([counter_a.get(k, 0) for k in keys], dtype=np.float64)
    b = np.array([counter_b.get(k, 0) for k in keys], dtype=np.float64)
    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)
    # avoid log(0)
    def _kl(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return 0.5 * _kl(a, m) + 0.5 * _kl(b, m)


def compute_continuity(token_rows):
    # token_rows: list of dicts with token_ids and frame_id
    n = len(token_rows)
    change = np.full(n, np.nan, dtype=np.float64)
    js = np.full(n, np.nan, dtype=np.float64)
    # precompute counters for JS
    counters = [Counter(r["token_ids"]) for r in token_rows]
    for i in range(1, n):
        change[i] = change_rate(token_rows[i-1]["token_ids"], token_rows[i]["token_ids"])
        js[i] = jensen_shannon_divergence(counters[i-1], counters[i])
    continuity = 1.0 - change
    return continuity, change, js


def window_stats(values, window):
    # values: 1d array length n
    n = len(values)
    win_rows = []
    for w in range(0, n, window):
        seg = values[w:w+window]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            mean = np.nan
            var = np.nan
        else:
            mean = float(np.mean(seg))
            var = float(np.var(seg))
        win_rows.append((w // window, w, min(w+window-1, n-1), mean, var))
    return win_rows


def annotate_segment_means(ax, xs, ys, split_idx):
    seg1 = ys[(xs >= 0) & (xs < split_idx)]
    seg2 = ys[(xs >= split_idx)]
    if len(seg1) > 0:
        m1 = np.nanmean(seg1)
        ax.axhline(m1, color="#666", linestyle="--", linewidth=1)
        ax.text(split_idx * 0.25, m1, f"mean[0,{split_idx})={m1:.3f}", fontsize=9, va="bottom")
    if len(seg2) > 0:
        m2 = np.nanmean(seg2)
        ax.axhline(m2, color="#999", linestyle=":", linewidth=1)
        ax.text(split_idx * 1.05, m2, f"mean[{split_idx},{int(xs.max())+1})={m2:.3f}", fontsize=9, va="bottom")


def add_frame_lines(ax, max_frame: int, step: int = 800) -> None:
    if max_frame is None or max_frame <= 0:
        return
    for x in range(step, int(max_frame) + 1, step):
        ax.axvline(x, color="#bbb", linestyle="--", linewidth=0.8)


def add_discontinuity_flags(df: pd.DataFrame, key_cols=None) -> pd.DataFrame:
    if df.empty:
        return df
    if key_cols is None:
        key_cols = ["ctx", "inf", "spec"]
    out = df.sort_values("window_id").copy()
    key_parts = []
    for col in key_cols:
        if col not in out.columns:
            key_parts.append(pd.Series(["none"] * len(out), index=out.index))
        else:
            key_parts.append(out[col].fillna("none").astype(str))
    key = key_parts[0]
    for part in key_parts[1:]:
        key = key + "_" + part
    out["discontinuity"] = key.ne(key.shift(1))
    out.iloc[0, out.columns.get_loc("discontinuity")] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--pattern", default="12-20-clean+clean1_base_*_nospec_static.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--acc_slack", type=float, default=0.02)
    ap.add_argument("--acc_tau_list", type=str, default="0.5,0.6,0.7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(args.logs_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {args.pattern} in {args.logs_dir}")

    per_frame_rows = []
    per_window_rows = []
    window_metric_rows = []

    rng = np.random.RandomState(args.seed)

    for fp in files:
        run_name = os.path.splitext(os.path.basename(fp))[0]
        meta = parse_run_meta(run_name)
        rows = read_log(fp)
        if not rows:
            continue
        # use frame_id0 for window indexing
        for r in rows:
            r["frame_id0"] = r["frame_id"] - 1 if r["frame_id"] >= 1 else r["frame_id"]

        token_rows = [r for r in rows if r["token_ids"] is not None]
        if token_rows:
            continuity, change, js = compute_continuity(token_rows)
            for i, r in enumerate(token_rows):
                per_frame_rows.append({
                    "run_name": run_name,
                    "ctx": meta["ctx"],
                    "inf": meta["inf"],
                    "spec": meta.get("spec"),
                    "frame_idx": r["frame_id0"],
                    "frame_idx_1based": r["frame_id"],
                    "continuity": continuity[i],
                    "change_rate": change[i],
                    "js_divergence": js[i],
                    "accuracy": r["accuracy"],
                    "latency": r["latency"],
                })

        # window stats for continuity and js
        if token_rows:
            # map continuity arrays to window stats by frame_id0
            token_df = pd.DataFrame({
                "frame_id0": [r["frame_id0"] for r in token_rows],
                "continuity": continuity,
                "js_divergence": js,
            })
            token_df["window_id"] = (token_df["frame_id0"] // args.window).astype(int)
            for metric in ["continuity", "js_divergence"]:
                g = token_df.groupby("window_id")[metric]
                for win_id, series in g:
                    mean = float(series.mean()) if len(series) else np.nan
                    var = float(series.var()) if len(series) else np.nan
                    frame_start = win_id * args.window
                    frame_end = frame_start + args.window - 1
                    per_window_rows.append({
                        "run_name": run_name,
                        "ctx": meta["ctx"],
                        "inf": meta["inf"],
                        "spec": meta.get("spec"),
                        "window_id": int(win_id),
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "metric": metric,
                        "mean": mean,
                        "var": var,
                    })

        # window metrics for best-point selection (accuracy + latency)
        run_df = pd.DataFrame(rows)
        run_df["window_id"] = (run_df["frame_id0"] // args.window).astype(int)
        for win_id, sub in run_df.groupby("window_id"):
            if sub.empty:
                continue
            window_metric_rows.append({
                "run_name": run_name,
                "ctx": meta["ctx"],
                "inf": meta["inf"],
                "ib": meta.get("ib"),
                "tb": meta.get("tb"),
                "spec": meta.get("spec"),
                "window_id": int(win_id),
                "frame_start": int(win_id * args.window),
                "frame_end": int(win_id * args.window + args.window - 1),
                "acc_mean": float(sub["accuracy"].mean()),
                "lat_mean": float(sub["latency"].mean()),
            })

    per_frame = pd.DataFrame(per_frame_rows)
    per_frame.to_csv(os.path.join(args.out_dir, "continuity_per_frame.csv"), index=False)

    per_window = pd.DataFrame(per_window_rows)
    per_window.to_csv(os.path.join(args.out_dir, "continuity_window_stats.csv"), index=False)

    win_metrics = pd.DataFrame(window_metric_rows)
    win_metrics.to_csv(os.path.join(args.out_dir, "window_run_metrics.csv"), index=False)

    if not per_frame.empty:
        # aggregate continuity across runs
        summary = per_frame.groupby("frame_idx").agg(
            cont_mean=("continuity", "mean"),
            cont_std=("continuity", "std"),
            js_mean=("js_divergence", "mean"),
            js_std=("js_divergence", "std"),
        ).reset_index()
        summary.to_csv(os.path.join(args.out_dir, "continuity_summary.csv"), index=False)

        # plot continuity curve
        plt.figure(figsize=(10, 4))
        xs = summary["frame_idx"].values
        ys = summary["cont_mean"].values
        ystd = summary["cont_std"].fillna(0).values
        plt.plot(xs, ys, color="#1f77b4", label="mean continuity")
        plt.fill_between(xs, ys-ystd, ys+ystd, color="#1f77b4", alpha=0.2, label="±1 std")
        plt.axvline(args.split, color="#444", linestyle="--", linewidth=1)
        annotate_segment_means(plt.gca(), xs, ys, args.split)
        plt.xlabel("frame index")
        plt.ylabel("continuity (1 - change rate)")
        plt.title("DOH continuity curve (token-change rate)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "continuity_curve.png"), dpi=160)
        plt.close()

        # plot JS divergence curve
        fig = plt.figure(figsize=(10, 4))
        ys = summary["js_mean"].values
        ystd = summary["js_std"].fillna(0).values
        plt.plot(xs, ys, color="#d62728", label="mean JS divergence")
        plt.fill_between(xs, ys-ystd, ys+ystd, color="#d62728", alpha=0.2, label="±1 std")
        plt.axvline(args.split, color="#444", linestyle="--", linewidth=1)
        plt.xlabel("frame index")
        plt.ylabel("JS divergence (↑ more drift, ↓ more stable)")
        plt.title("DOH token distribution drift (JS divergence)")
        plt.legend()
        # stability marker on y-axis
        ax = plt.gca()
        ax.annotate(
            "",
            xy=(0.02, 0.88),
            xytext=(0.02, 0.12),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )
        ax.text(0.04, 0.90, "low stability", transform=ax.transAxes, fontsize=9, va="bottom")
        ax.text(0.04, 0.06, "high stability", transform=ax.transAxes, fontsize=9, va="top")

        # segment means shown below the plot (not inside axes)
        seg1 = summary[summary["frame_idx"] < args.split]["js_mean"].mean()
        seg2 = summary[summary["frame_idx"] >= args.split]["js_mean"].mean()
        fig.text(
            0.01,
            -0.02,
            f"JS mean [0,{args.split}) = {seg1:.4f}    |    JS mean [{args.split},{int(xs.max())+1}) = {seg2:.4f}",
            ha="left",
            va="top",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "js_divergence_curve.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # best-point per window across runs
    best_rows = []
    for win_id, sub in win_metrics.groupby("window_id"):
        best_acc = sub["acc_mean"].max()
        if np.isnan(best_acc):
            continue
        feasible = sub[sub["acc_mean"] >= best_acc * (1.0 - args.acc_slack)].copy()
        if feasible.empty:
            feasible = sub
        # pick min latency, break ties randomly
        min_lat = feasible["lat_mean"].min()
        tol = 1e-9
        tied = feasible[feasible["lat_mean"] <= min_lat + tol]
        if len(tied) > 1:
            idx = rng.choice(tied.index)
            pick = tied.loc[idx]
        else:
            pick = feasible.loc[feasible["lat_mean"].idxmin()]
        best_rows.append({
            "window_id": win_id,
            "frame_start": int(pick["frame_start"]),
            "frame_end": int(pick["frame_end"]),
            "time_center": (int(pick["frame_start"]) + int(pick["frame_end"])) / 2,
            "frame_start_1based": int(pick["frame_start"]) + 1,
            "frame_end_1based": int(pick["frame_end"]) + 1,
            "time_center_1based": (int(pick["frame_start"]) + int(pick["frame_end"])) / 2 + 1,
            "acc_mean": float(pick["acc_mean"]),
            "lat_mean": float(pick["lat_mean"]),
            "run_name": pick["run_name"],
            "ctx": int(pick["ctx"]),
            "inf": int(pick["inf"]),
            "ib": int(pick["ib"]) if pick.get("ib") is not None else None,
            "tb": int(pick["tb"]) if pick.get("tb") is not None else None,
            "spec": pick.get("spec"),
        })

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(args.out_dir, "best_point_per_window.csv"), index=False)

    # plot best points (accuracy)
    if not best_df.empty:
        plot_df = add_discontinuity_flags(best_df, key_cols=["ctx", "inf", "spec"])
        plt.figure(figsize=(12, 4))
        colors = ["#d62728" if d else "#2ca02c" for d in plot_df["discontinuity"]]
        plt.scatter(plot_df["time_center"], plot_df["acc_mean"], c=colors)
        for _, row in plot_df.iterrows():
            spec_tag = row.get("spec")
            if pd.isna(spec_tag) or spec_tag is None:
                label = f"ctx{row['ctx']}_inf{row['inf']}"
            else:
                label = f"ctx{row['ctx']}_inf{row['inf']}_{spec_tag}"
            plt.text(row["time_center"], row["acc_mean"], label, fontsize=7, rotation=35, ha="left")
        ax = plt.gca()
        add_frame_lines(ax, int(plot_df["frame_end"].max()))
        ax.axvline(args.split, color="#444", linestyle="--", linewidth=1)
        plt.xlabel("time (frame index)")
        plt.ylabel("accuracy (mean per window, ↑ better)")
        plt.title("Best config per window (random tie-break)")
        # visual cue: upward arrow anchored to y-axis in axes coords
        ax.annotate(
            "",
            xy=(0.02, 0.88),
            xytext=(0.02, 0.12),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "best_point_per_window.png"), dpi=160)
        plt.close()

        # plot best points (latency)
        plot_df = add_discontinuity_flags(best_df, key_cols=["ctx", "inf", "spec"])
        plt.figure(figsize=(12, 4))
        colors = ["#d62728" if d else "#ff7f0e" for d in plot_df["discontinuity"]]
        plt.scatter(plot_df["time_center"], plot_df["lat_mean"], c=colors)
        for _, row in plot_df.iterrows():
            spec_tag = row.get("spec")
            if pd.isna(spec_tag) or spec_tag is None:
                label = f"ctx{row['ctx']}_inf{row['inf']}"
            else:
                label = f"ctx{row['ctx']}_inf{row['inf']}_{spec_tag}"
            plt.text(row["time_center"], row["lat_mean"], label, fontsize=7, rotation=35, ha="left")
        ax = plt.gca()
        add_frame_lines(ax, int(plot_df["frame_end"].max()))
        ax.axvline(args.split, color="#444", linestyle="--", linewidth=1)
        plt.xlabel("time (frame index)")
        plt.ylabel("latency (mean per window, ↓ better)")
        plt.title("Best config per window (latency under acc constraint)")
        ax.annotate(
            "",
            xy=(0.02, 0.12),
            xytext=(0.02, 0.88),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "best_point_per_window_latency.png"), dpi=160)
        plt.close()

    # ---- A) JS bucket vs best config distribution (slack-based selection) ----
    js_summary = None
    if not per_window.empty:
        js_rows = per_window[per_window["metric"] == "js_divergence"].copy()
        if not js_rows.empty:
            js_summary = (
                js_rows.groupby("window_id")["mean"]
                .mean()
                .reset_index()
                .rename(columns={"mean": "js_mean"})
            )
            js_summary["js_rank"] = js_summary["js_mean"].rank(method="first")
            js_summary["js_bucket"] = pd.qcut(js_summary["js_rank"], 3, labels=["low", "mid", "high"])
            label_map = {"low": "low (stable)", "mid": "mid", "high": "high (drift)"}
            js_summary["js_bucket_label"] = js_summary["js_bucket"].map(label_map)
            js_summary.to_csv(os.path.join(args.out_dir, "js_window_summary.csv"), index=False)

    if js_summary is not None and not best_df.empty:
        sel = best_df.merge(js_summary[["window_id", "js_bucket_label", "js_mean"]], on="window_id", how="inner")
        # ctx distribution
        dist_ctx = (
            sel.groupby(["js_bucket_label", "ctx"])
            .size()
            .reset_index(name="count")
        )
        dist_ctx["total"] = dist_ctx.groupby("js_bucket_label")["count"].transform("sum")
        dist_ctx["prob"] = dist_ctx["count"] / dist_ctx["total"]
        dist_ctx.to_csv(os.path.join(args.out_dir, "js_bucket_bestctx_dist.csv"), index=False)
        # inf distribution
        dist_inf = (
            sel.groupby(["js_bucket_label", "inf"])
            .size()
            .reset_index(name="count")
        )
        dist_inf["total"] = dist_inf.groupby("js_bucket_label")["count"].transform("sum")
        dist_inf["prob"] = dist_inf["count"] / dist_inf["total"]
        dist_inf.to_csv(os.path.join(args.out_dir, "js_bucket_bestinf_dist.csv"), index=False)

        buckets = ["low (stable)", "mid", "high (drift)"]
        # ctx plot
        ctx_vals = sorted(dist_ctx["ctx"].unique())
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bottom = np.zeros(len(buckets))
        for ctx in ctx_vals:
            vals = []
            for b in buckets:
                row = dist_ctx[(dist_ctx["js_bucket_label"] == b) & (dist_ctx["ctx"] == ctx)]
                vals.append(row["prob"].iloc[0] if not row.empty else 0.0)
            ax.bar(buckets, vals, bottom=bottom, label=f"ctx={ctx}")
            bottom += np.array(vals)
        ax.set_xlabel("JS quantile (low / mid / high)")
        ax.set_ylabel("P(select ctx) ↑")
        ax.set_title("A) Best config distribution vs JS drift (ctx) — JS↑ = more drift")
        ax.legend(loc="best", fontsize=8)
        ax.annotate(
            "",
            xy=(0.02, 0.88),
            xytext=(0.02, 0.12),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "js_bucket_bestctx_dist.png"), dpi=200)
        plt.close(fig)

        # inf plot
        inf_vals = sorted(dist_inf["inf"].unique())
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bottom = np.zeros(len(buckets))
        for inf in inf_vals:
            vals = []
            for b in buckets:
                row = dist_inf[(dist_inf["js_bucket_label"] == b) & (dist_inf["inf"] == inf)]
                vals.append(row["prob"].iloc[0] if not row.empty else 0.0)
            ax.bar(buckets, vals, bottom=bottom, label=f"inf={inf}")
            bottom += np.array(vals)
        ax.set_xlabel("JS quantile (low / mid / high)")
        ax.set_ylabel("P(select inf) ↑")
        ax.set_title("A) Best config distribution vs JS drift (inf) — JS↑ = more drift")
        ax.legend(loc="best", fontsize=8)
        ax.annotate(
            "",
            xy=(0.02, 0.88),
            xytext=(0.02, 0.12),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "js_bucket_bestinf_dist.png"), dpi=200)
        plt.close(fig)

    # ---- B) epsilon-constraint with absolute tau list ----
    tau_list = []
    if args.acc_tau_list:
        for s in args.acc_tau_list.split(","):
            s = s.strip()
            if not s:
                continue
            try:
                tau_list.append(float(s))
            except Exception:
                pass

    if tau_list and not win_metrics.empty:
        for tau in tau_list:
            tau_rows = []
            for win_id, sub in win_metrics.groupby("window_id"):
                feasible = sub[sub["acc_mean"] >= tau]
                feasible_flag = 1
                if feasible.empty:
                    feasible = sub.copy()
                    feasible_flag = 0
                min_lat = feasible["lat_mean"].min()
                tol = 1e-9
                tied = feasible[feasible["lat_mean"] <= min_lat + tol]
                if len(tied) > 1:
                    idx = rng.choice(tied.index)
                    pick = tied.loc[idx]
                else:
                    pick = feasible.loc[feasible["lat_mean"].idxmin()]
                tau_rows.append({
                    "window_id": int(win_id),
                    "frame_start": int(pick["frame_start"]),
                    "frame_end": int(pick["frame_end"]),
                    "time_center": (int(pick["frame_start"]) + int(pick["frame_end"])) / 2,
                    "acc_mean": float(pick["acc_mean"]),
                    "lat_mean": float(pick["lat_mean"]),
                    "run_name": pick["run_name"],
                    "ctx": int(pick["ctx"]),
                    "inf": int(pick["inf"]),
                    "ib": int(pick["ib"]) if pick.get("ib") is not None else None,
                    "tb": int(pick["tb"]) if pick.get("tb") is not None else None,
                    "spec": pick.get("spec"),
                    "feasible": feasible_flag,
                    "tau": tau,
                })
            tau_df = pd.DataFrame(tau_rows)
            tag = f"tau{str(tau).replace('.', 'p')}"
            tau_df.to_csv(os.path.join(args.out_dir, f"best_point_per_window_{tag}.csv"), index=False)

            if not tau_df.empty:
                plot_df = add_discontinuity_flags(tau_df, key_cols=["ctx", "inf", "spec"])
                fig, ax = plt.subplots(figsize=(12, 4))
                colors = ["#d62728" if d else "#ff7f0e" for d in plot_df["discontinuity"]]
                ax.scatter(plot_df["time_center"], plot_df["lat_mean"], c=colors)
                for _, row in plot_df.iterrows():
                    spec_tag = row.get("spec")
                    if pd.isna(spec_tag) or spec_tag is None:
                        label = f"ctx{row['ctx']}_inf{row['inf']}"
                    else:
                        label = f"ctx{row['ctx']}_inf{row['inf']}_{spec_tag}"
                    ax.text(row["time_center"], row["lat_mean"], label, fontsize=7, rotation=35, ha="left")
                add_frame_lines(ax, int(plot_df["frame_end"].max()))
                ax.axvline(args.split, color="#444", linestyle="--", linewidth=1)
                ax.set_xlabel("time (frame index)")
                ax.set_ylabel("latency (mean per window, ↓ better)")
                ax.set_title(f"B) Best config under acc≥{tau} (epsilon-constraint)")
                ax.annotate(
                    "",
                    xy=(0.02, 0.12),
                    xytext=(0.02, 0.88),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                )
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, f"best_point_per_window_{tag}_latency.png"), dpi=200)
                plt.close(fig)

                # batch-size pair plot (ib/tb)
                fig, ax = plt.subplots(figsize=(12, 4))
                plot_df_bt = add_discontinuity_flags(tau_df, key_cols=["ib", "tb"])
                colors_bt = ["#d62728" if d else "#1f77b4" for d in plot_df_bt["discontinuity"]]
                ax.scatter(plot_df_bt["time_center"], plot_df_bt["lat_mean"], c=colors_bt)
                for _, row in plot_df_bt.iterrows():
                    label = f"ib{row['ib']}_tb{row['tb']}"
                    ax.text(row["time_center"], row["lat_mean"], label, fontsize=7, rotation=35, ha="left")
                add_frame_lines(ax, int(plot_df_bt["frame_end"].max()))
                ax.axvline(args.split, color="#444", linestyle="--", linewidth=1)
                ax.set_xlabel("time (frame index)")
                ax.set_ylabel("latency (mean per window, ↓ better)")
                ax.set_title(f"B) Best (ib,tb) under acc≥{tau} (epsilon-constraint)")
                ax.annotate(
                    "",
                    xy=(0.02, 0.12),
                    xytext=(0.02, 0.88),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                )
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, f"best_batchsize_per_window_{tag}_latency.png"), dpi=200)
                plt.close(fig)

            if js_summary is not None and not tau_df.empty:
                sel = tau_df.merge(js_summary[["window_id", "js_bucket_label", "js_mean"]], on="window_id", how="inner")
                dist_ctx = (
                    sel.groupby(["js_bucket_label", "ctx"]).size().reset_index(name="count")
                )
                dist_ctx["total"] = dist_ctx.groupby("js_bucket_label")["count"].transform("sum")
                dist_ctx["prob"] = dist_ctx["count"] / dist_ctx["total"]
                dist_ctx.to_csv(os.path.join(args.out_dir, f"js_bucket_bestctx_dist_{tag}.csv"), index=False)
                dist_inf = (
                    sel.groupby(["js_bucket_label", "inf"]).size().reset_index(name="count")
                )
                dist_inf["total"] = dist_inf.groupby("js_bucket_label")["count"].transform("sum")
                dist_inf["prob"] = dist_inf["count"] / dist_inf["total"]
                dist_inf.to_csv(os.path.join(args.out_dir, f"js_bucket_bestinf_dist_{tag}.csv"), index=False)
                buckets = ["low (stable)", "mid", "high (drift)"]
                # ctx plot
                ctx_vals = sorted(dist_ctx["ctx"].unique())
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                bottom = np.zeros(len(buckets))
                for ctx in ctx_vals:
                    vals = []
                    for b in buckets:
                        row = dist_ctx[(dist_ctx["js_bucket_label"] == b) & (dist_ctx["ctx"] == ctx)]
                        vals.append(row["prob"].iloc[0] if not row.empty else 0.0)
                    ax.bar(buckets, vals, bottom=bottom, label=f"ctx={ctx}")
                    bottom += np.array(vals)
                ax.set_xlabel("JS quantile (low / mid / high)")
                ax.set_ylabel("P(select ctx) ↑")
                ax.set_title(f"B) Best config vs JS drift (acc≥{tau}) — ctx — JS↑ = more drift")
                ax.legend(loc="best", fontsize=8)
                ax.annotate(
                    "",
                    xy=(0.02, 0.88),
                    xytext=(0.02, 0.12),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                )
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, f"js_bucket_bestctx_dist_{tag}.png"), dpi=200)
                plt.close(fig)
                # inf plot
                inf_vals = sorted(dist_inf["inf"].unique())
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                bottom = np.zeros(len(buckets))
                for inf in inf_vals:
                    vals = []
                    for b in buckets:
                        row = dist_inf[(dist_inf["js_bucket_label"] == b) & (dist_inf["inf"] == inf)]
                        vals.append(row["prob"].iloc[0] if not row.empty else 0.0)
                    ax.bar(buckets, vals, bottom=bottom, label=f"inf={inf}")
                    bottom += np.array(vals)
                ax.set_xlabel("JS quantile (low / mid / high)")
                ax.set_ylabel("P(select inf) ↑")
                ax.set_title(f"B) Best config vs JS drift (acc≥{tau}) — inf — JS↑ = more drift")
                ax.legend(loc="best", fontsize=8)
                ax.annotate(
                    "",
                    xy=(0.02, 0.88),
                    xytext=(0.02, 0.12),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                )
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, f"js_bucket_bestinf_dist_{tag}.png"), dpi=200)
                plt.close(fig)


if __name__ == "__main__":
    main()
