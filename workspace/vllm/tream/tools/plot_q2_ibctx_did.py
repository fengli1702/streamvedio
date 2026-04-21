#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RE = re.compile(r"doh_ibctx_did_1\.22_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_rep(?P<rep>\d+)")


def load_wandb_summary(path):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        name = r.get("run_name")
        if not isinstance(name, str):
            continue
        m = RE.match(name)
        if not m:
            continue
        rows.append({
            "run_name": name,
            "ctx": int(m.group("ctx")),
            "inf": int(m.group("inf")),
            "ib": int(m.group("ib")),
            "tb": int(m.group("tb")),
            "lr": m.group("lr"),
            "rep": int(m.group("rep")),
            "acc": float(r.get("inference/global_accuracy", np.nan)),
            "lat_s": float(r.get("inference/latency", np.nan)),
        })
    return pd.DataFrame(rows)


def _style_axes(ax, y_label, y_dir="up"):
    ax.set_xlabel("ib (inference batch size)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    # add y-axis direction arrow
    if y_dir == "up":
        ax.annotate("", xy=(0.03, 0.88), xytext=(0.03, 0.12), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    else:
        ax.annotate("", xy=(0.03, 0.12), xytext=(0.03, 0.88), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))


def plot_latency(df, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ctx in sorted(df["ctx"].unique()):
        sub = df[df["ctx"] == ctx]
        g = sub.groupby("ib")
        xs = []
        means = []
        stds = []
        for ib, gsub in g:
            xs.append(ib)
            lat_ms = gsub["lat_s"].values * 1000.0
            means.append(np.nanmean(lat_ms))
            stds.append(np.nanstd(lat_ms))
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        means = np.array(means)[order]
        stds = np.array(stds)[order]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, label=f"ctx={ctx}")
        # highlight best latency per ctx
        best_idx = np.nanargmin(means)
        ax.scatter([xs[best_idx]], [means[best_idx]], s=120, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
        ax.text(xs[best_idx], means[best_idx], f"best", fontsize=8, ha="left", va="bottom")

    _style_axes(ax, "latency (ms, ↓ better)", y_dir="down")
    ax.set_title("Q2: ib effect on latency depends on ctx\n(mean ± std over reps; circled=best)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_accuracy(df, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ctx in sorted(df["ctx"].unique()):
        sub = df[df["ctx"] == ctx]
        g = sub.groupby("ib")
        xs = []
        means = []
        stds = []
        for ib, gsub in g:
            xs.append(ib)
            vals = gsub["acc"].values
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals))
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        means = np.array(means)[order]
        stds = np.array(stds)[order]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, label=f"ctx={ctx}")
        # highlight best accuracy per ctx
        best_idx = np.nanargmax(means)
        ax.scatter([xs[best_idx]], [means[best_idx]], s=120, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
        ax.text(xs[best_idx], means[best_idx], f"best", fontsize=8, ha="left", va="bottom")

    _style_axes(ax, "global accuracy (↑ better)", y_dir="up")
    ax.set_title("Q2: ib effect on accuracy depends on ctx\n(mean ± std over reps; circled=best)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_wandb_summary(args.wandb_csv)
    if df.empty:
        raise SystemExit("No doh_ibctx_did runs found in wandb summary.")

    plot_latency(df, os.path.join(args.out_dir, "q2_ib_effect_latency.png"))
    plot_accuracy(df, os.path.join(args.out_dir, "q2_ib_effect_accuracy.png"))
