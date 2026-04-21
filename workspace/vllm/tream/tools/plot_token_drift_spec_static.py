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

PARAM_RE = re.compile(
    r"_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib(?P<ib>\d+)_tb(?P<tb>\d+)"
)

# Palette: keep red boxes, but separate line colors for readability.
DRIFT_LINE_COLOR = "#2b6cb0"     # blue
VLINE_GRID_COLOR = "#7aa6c2"     # light steel blue
VLINE_SWITCH_COLOR = "#8e24aa"   # high-contrast purple
BOX_EDGE_COLOR = "#e63946"       # red
ACCEPT_LINE_COLOR = "#f4a261"    # warm orange

# Candidate field aliases across different spec-metrics dumps.
ALIAS_ACCEPTED = ["accepted", "num_accepted", "accepted_tokens", "accepted_token_count"]
ALIAS_PROPOSED = ["proposed", "num_proposed", "proposed_tokens", "proposed_token_count"]
ALIAS_ACCEPT_RATE = ["acceptance_rate", "accept_rate", "accepted_ratio"]
ALIAS_REJECT = ["reject_tokens", "rejected", "rejected_tokens", "num_rejected_tokens"]
ALIAS_REVERIFY = [
    "reverify_count",
    "reverify_times",
    "num_reverify",
    "reverify_steps",
]
ALIAS_EARLY_STOP = [
    "early_stop_count",
    "early_stops",
    "num_early_stops",
]
ALIAS_VERIFY_TOKENS = [
    "verify_tokens",
    "verify_token_count",
    "verify_processed_tokens",
]
ALIAS_DRAFT_TIME_MS = ["draft_time_ms", "draft_ms"]
ALIAS_VERIFY_TIME_MS = ["verify_time_ms", "verify_ms"]


def _coalesce_numeric_columns(df: pd.DataFrame, names, default=np.nan):
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(default, index=df.index, dtype=float)


def _sum_min_count_1(s: pd.Series):
    return s.sum(min_count=1)


def _finite_count(s: pd.Series):
    return int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())


def _safe_quantile(s: pd.Series, q: float):
    z = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan
    return float(np.quantile(z, q))


def _safe_corr(a: pd.Series, b: pd.Series):
    ax = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bx = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(ax) & np.isfinite(bx)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(ax[m], bx[m])[0, 1])


def parse_params(run_name: str):
    m = PARAM_RE.search(run_name)
    if not m:
        return None
    return (
        int(m.group("ctx")),
        int(m.group("inf")),
        int(m.group("ib")),
        int(m.group("tb")),
    )


def js_divergence_from_token_ids(prev_ids, cur_ids):
    a = np.asarray(prev_ids, dtype=np.int64).ravel()
    b = np.asarray(cur_ids, dtype=np.int64).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan

    keys = np.union1d(np.unique(a), np.unique(b))
    if keys.size == 0:
        return np.nan

    pa = np.zeros(keys.size, dtype=np.float64)
    pb = np.zeros(keys.size, dtype=np.float64)

    ua, ca = np.unique(a, return_counts=True)
    ub, cb = np.unique(b, return_counts=True)

    ia = np.searchsorted(keys, ua)
    ib = np.searchsorted(keys, ub)
    pa[ia] = ca.astype(np.float64)
    pb[ib] = cb.astype(np.float64)

    pa_sum = float(pa.sum())
    pb_sum = float(pb.sum())
    if pa_sum <= 0.0 or pb_sum <= 0.0:
        return np.nan
    pa /= pa_sum
    pb /= pb_sum

    m = 0.5 * (pa + pb)
    mask_a = pa > 0.0
    mask_b = pb > 0.0
    kl_am = float(np.sum(pa[mask_a] * np.log2(pa[mask_a] / m[mask_a])))
    kl_bm = float(np.sum(pb[mask_b] * np.log2(pb[mask_b] / m[mask_b])))
    jsd = 0.5 * (kl_am + kl_bm)
    if not np.isfinite(jsd):
        return np.nan
    return float(max(0.0, min(1.0, jsd)))


def parse_frame(obj):
    fi = obj.get("frame_index")
    if fi is not None:
        return int(fi)
    fp = obj.get("frame_path", "")
    bn = os.path.basename(fp)
    if bn.endswith(".jpg") and bn[:-4].isdigit():
        return int(bn[:-4])
    return None


def load_spec_metrics(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "step",
                "acceptance_rate",
                "accepted",
                "proposed",
                "reject_tokens",
                "reverify_count",
                "early_stop_count",
                "verify_tokens",
                "draft_time_ms",
                "verify_time_ms",
            ]
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            row = {}
            for k, v in obj.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v):
                    row[k] = float(v)
            if not row:
                continue
            if "step" not in row:
                row["step"] = float(len(rows))
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "step",
                "acceptance_rate",
                "accepted",
                "proposed",
                "reject_tokens",
                "reverify_count",
                "early_stop_count",
                "verify_tokens",
                "draft_time_ms",
                "verify_time_ms",
            ]
        )

    df = pd.DataFrame(rows)
    if "step" not in df.columns:
        df["step"] = np.arange(len(df), dtype=float)
    step_series = pd.to_numeric(df["step"], errors="coerce")
    if step_series.isna().any():
        fallback = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        step_series = step_series.where(step_series.notna(), fallback)
    df["step"] = step_series.astype(int)

    df["accepted"] = _coalesce_numeric_columns(df, ALIAS_ACCEPTED)
    df["proposed"] = _coalesce_numeric_columns(df, ALIAS_PROPOSED)
    df["acceptance_rate"] = _coalesce_numeric_columns(df, ALIAS_ACCEPT_RATE)

    # Fill missing acceptance_rate from accepted/proposed.
    ratio = df["accepted"] / df["proposed"].replace(0, np.nan)
    df["acceptance_rate"] = df["acceptance_rate"].where(
        df["acceptance_rate"].notna(), ratio
    )

    df["reject_tokens"] = _coalesce_numeric_columns(df, ALIAS_REJECT)
    reject_fallback = df["proposed"] - df["accepted"]
    df["reject_tokens"] = df["reject_tokens"].where(df["reject_tokens"].notna(), reject_fallback)

    df["reverify_count"] = _coalesce_numeric_columns(df, ALIAS_REVERIFY)
    df["early_stop_count"] = _coalesce_numeric_columns(df, ALIAS_EARLY_STOP)
    df["verify_tokens"] = _coalesce_numeric_columns(df, ALIAS_VERIFY_TOKENS)
    df["verify_tokens"] = df["verify_tokens"].where(df["verify_tokens"].notna(), df["accepted"])
    df["draft_time_ms"] = _coalesce_numeric_columns(df, ALIAS_DRAFT_TIME_MS)
    df["verify_time_ms"] = _coalesce_numeric_columns(df, ALIAS_VERIFY_TIME_MS)

    keep_cols = [
        "step",
        "acceptance_rate",
        "accepted",
        "proposed",
        "reject_tokens",
        "reverify_count",
        "early_stop_count",
        "verify_tokens",
        "draft_time_ms",
        "verify_time_ms",
    ]
    return df[keep_cols].sort_values("step")


def load_run(path):
    run_name = os.path.splitext(os.path.basename(path))[0]
    parsed = parse_params(run_name)
    if parsed is None:
        return None
    ctx, inf, ib, tb = parsed

    rows = []
    prev_tokens = None
    prev_tps = None
    token_mode = False
    train_cycle_frames = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            ev = obj.get("event")
            if ev == "training_cycle":
                fi_train = parse_frame(obj)
                if fi_train is not None:
                    train_cycle_frames.append(int(fi_train))
                continue
            if ev is not None and ev != "inference_cycle":
                continue
            if "accuracy" not in obj or "latency" not in obj:
                continue

            frame = parse_frame(obj)
            if frame is None:
                continue

            try:
                acc = float(obj.get("accuracy"))
                lat = float(obj.get("latency"))
            except Exception:
                continue

            tps_raw = obj.get("tokens_per_second")
            tps = np.nan
            try:
                if tps_raw is not None:
                    tps = float(tps_raw)
            except Exception:
                tps = np.nan

            token_ids = obj.get("token_ids")
            drift = np.nan
            drift_src = "none"

            jsd_val = obj.get("jsd_mean")
            if jsd_val is None:
                jsd_val = obj.get("token_drift_mean")
            if jsd_val is None:
                jsd_val = obj.get("drift_jsd")
            if jsd_val is None:
                jsd_val = obj.get("jsd")
            if jsd_val is not None:
                try:
                    drift = float(jsd_val)
                    drift_src = "jsd_field"
                except Exception:
                    drift = np.nan
                    drift_src = "none"
            elif token_ids is not None and prev_tokens is not None:
                drift = js_divergence_from_token_ids(prev_tokens, token_ids)
                drift_src = "token_ids_jsd"
                token_mode = True

            rows.append(
                {
                    "run_name": run_name,
                    "ctx": ctx,
                    "inf": inf,
                    "ib": ib,
                    "tb": tb,
                    "frame": int(frame),
                    "accuracy": acc,
                    "latency": lat,
                    "tps": tps,
                    "token_drift": drift,
                    "drift_source": drift_src,
                }
            )

            if token_ids is not None:
                prev_tokens = token_ids
            prev_tps = tps if np.isfinite(tps) else prev_tps

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("frame")

    # Load speculative decoding acceptance stats and map them to frame axis.
    spec_path = os.path.join(os.path.dirname(path), f"spec_metrics_{run_name}.jsonl")
    spec_df = load_spec_metrics(spec_path)
    if spec_df.empty:
        acc_df = pd.DataFrame(
            columns=[
                "run_name",
                "ctx",
                "inf",
                "ib",
                "tb",
                "frame",
                "acceptance_rate",
                "accepted",
                "proposed",
                "reject_tokens",
                "reverify_count",
                "early_stop_count",
                "verify_tokens",
                "draft_time_ms",
                "verify_time_ms",
            ]
        )
    else:
        if train_cycle_frames:
            k = min(len(train_cycle_frames), len(spec_df))
            mapped_frame = np.asarray(train_cycle_frames[:k], dtype=float)
            mapped = spec_df.iloc[:k].copy()
        else:
            # Fallback when no training_cycle frame exists in inference log.
            mapped = spec_df.copy()
            steps = mapped["step"].to_numpy(dtype=float)
            smax = float(np.max(steps)) if len(steps) else 0.0
            fmin = float(df["frame"].min())
            fmax = float(df["frame"].max())
            if smax <= 1e-8:
                mapped_frame = np.full_like(steps, fmin, dtype=float)
            else:
                mapped_frame = fmin + (steps / smax) * (fmax - fmin)
        mapped["frame"] = mapped_frame
        mapped["run_name"] = run_name
        mapped["ctx"] = ctx
        mapped["inf"] = inf
        mapped["ib"] = ib
        mapped["tb"] = tb

        acc_df = mapped[
            [
                "run_name",
                "ctx",
                "inf",
                "ib",
                "tb",
                "frame",
                "acceptance_rate",
                "accepted",
                "proposed",
                "reject_tokens",
                "reverify_count",
                "early_stop_count",
                "verify_tokens",
                "draft_time_ms",
                "verify_time_ms",
            ]
        ].copy()

    return {"run_name": run_name, "df": df, "token_mode": token_mode, "acc_df": acc_df}


def add_discontinuity_flags(df: pd.DataFrame, key_cols):
    if df.empty:
        return df
    out = df.sort_values("window_id").copy()
    key = out[key_cols[0]].astype(str)
    for col in key_cols[1:]:
        key = key + "_" + out[col].astype(str)
    out["discontinuity"] = key.ne(key.shift(1))
    out.iloc[0, out.columns.get_loc("discontinuity")] = False
    return out


def add_switch_chain_features(df: pd.DataFrame, key_cols):
    out = add_discontinuity_flags(df, key_cols=key_cols)
    if out.empty:
        return out
    if key_cols == ["ctx", "inf"]:
        out["selected_key"] = (
            "ctx"
            + out["ctx"].round().astype(int).astype(str)
            + "_inf"
            + out["inf"].round().astype(int).astype(str)
        )
    elif key_cols == ["ib", "tb"]:
        out["selected_key"] = (
            "ib"
            + out["ib"].round().astype(int).astype(str)
            + "_tb"
            + out["tb"].round().astype(int).astype(str)
        )
    else:
        key = out[key_cols[0]].astype(str)
        for col in key_cols[1:]:
            key = key + "_" + out[col].astype(str)
        out["selected_key"] = key
    out["prev_selected_key"] = out["selected_key"].shift(1)
    out["switch_flag"] = out["discontinuity"].astype(int)
    out["switch_next"] = out["switch_flag"].shift(-1).fillna(0).astype(int)
    out["switch_rolling3"] = out["switch_flag"].rolling(3, min_periods=1).mean()
    if "token_drift_mean" in out.columns and "spec_reject_ratio" in out.columns:
        out["drift_x_reject"] = out["token_drift_mean"] * out["spec_reject_ratio"]
    return out


def make_chain_sync_view(best_df: pd.DataFrame):
    if best_df.empty:
        return best_df
    cols = [
        "window_id",
        "frame_start",
        "frame_end",
        "time_center",
        "run_name",
        "selected_key",
        "prev_selected_key",
        "switch_flag",
        "switch_next",
        "switch_rolling3",
        "token_drift_mean",
        "token_drift_std",
        "token_drift_p90",
        "token_drift_max",
        "token_drift_samples",
        "spec_acceptance_rate_mean",
        "spec_reject_tokens_per_step",
        "spec_reject_ratio",
        "lat_mean",
        "acc_mean",
        "ctx",
        "inf",
        "ib",
        "tb",
        "feasible",
        "tau",
    ]
    keep = [c for c in cols if c in best_df.columns]
    return best_df[keep].copy()


def build_window_metrics(run_df: pd.DataFrame, window_frames: int):
    x = run_df.copy()
    x["window_id"] = (x["frame"] // window_frames).astype(int)
    g = x.groupby("window_id")
    out = (
        g.agg(
            frame_start=("frame", "min"),
            frame_end=("frame", "max"),
            acc_mean=("accuracy", "mean"),
            lat_mean=("latency", "mean"),
            token_drift_mean=("token_drift", "mean"),
            token_drift_std=("token_drift", "std"),
            token_drift_p90=("token_drift", lambda s: _safe_quantile(s, 0.9)),
            token_drift_max=("token_drift", "max"),
            token_drift_samples=("token_drift", _finite_count),
        )
        .reset_index()
        .copy()
    )
    meta_cols = ["run_name", "ctx", "inf", "ib", "tb"]
    for c in meta_cols:
        out[c] = x[c].iloc[0]
    out["time_center"] = (out["frame_start"] + out["frame_end"]) / 2.0
    return out


def build_spec_window_metrics(acc_run_df: pd.DataFrame, window_frames: int):
    if acc_run_df.empty:
        return pd.DataFrame(columns=["window_id"])

    x = acc_run_df.copy()
    x["window_id"] = (x["frame"] // window_frames).astype(int)
    g = x.groupby("window_id")
    out = (
        g.agg(
            spec_samples=("frame", "size"),
            spec_frame_start=("frame", "min"),
            spec_frame_end=("frame", "max"),
            spec_acceptance_rate_mean=("acceptance_rate", "mean"),
            spec_acceptance_rate_std=("acceptance_rate", "std"),
            spec_accepted_tokens_per_step=("accepted", "mean"),
            spec_proposed_tokens_per_step=("proposed", "mean"),
            spec_reject_tokens_per_step=("reject_tokens", "mean"),
            spec_accepted_tokens_sum=("accepted", _sum_min_count_1),
            spec_proposed_tokens_sum=("proposed", _sum_min_count_1),
            spec_reject_tokens_sum=("reject_tokens", _sum_min_count_1),
            spec_reverify_count_per_step=("reverify_count", "mean"),
            spec_reverify_count_sum=("reverify_count", _sum_min_count_1),
            spec_early_stop_count_per_step=("early_stop_count", "mean"),
            spec_early_stop_count_sum=("early_stop_count", _sum_min_count_1),
            spec_verify_tokens_per_step=("verify_tokens", "mean"),
            spec_draft_time_ms_per_step=("draft_time_ms", "mean"),
            spec_verify_time_ms_per_step=("verify_time_ms", "mean"),
        )
        .reset_index()
        .copy()
    )

    out["spec_acceptance_rate_weighted"] = (
        out["spec_accepted_tokens_sum"] / out["spec_proposed_tokens_sum"].replace(0, np.nan)
    )
    out["spec_reject_ratio"] = (
        out["spec_reject_tokens_sum"] / out["spec_proposed_tokens_sum"].replace(0, np.nan)
    )
    out["spec_draft_verify_time_ratio"] = (
        out["spec_draft_time_ms_per_step"] / out["spec_verify_time_ms_per_step"].replace(0, np.nan)
    )

    meta_cols = ["run_name", "ctx", "inf", "ib", "tb"]
    for c in meta_cols:
        out[c] = x[c].iloc[0]
    return out


def choose_best_per_window(win_df: pd.DataFrame, tau: float):
    picks = []
    for window_id, sub in win_df.groupby("window_id"):
        feasible = sub[sub["acc_mean"] >= tau]
        feasible_flag = 1
        if feasible.empty:
            feasible = sub.copy()
            feasible_flag = 0

        min_lat = feasible["lat_mean"].min()
        tied = feasible[np.isclose(feasible["lat_mean"], min_lat)].sort_values("run_name")
        pick = tied.iloc[0]
        row = pick.to_dict()
        row["feasible"] = feasible_flag
        row["tau"] = tau
        picks.append(row)
    if not picks:
        return pd.DataFrame()
    return pd.DataFrame(picks).sort_values("window_id")


def aggregate_drift(run_df: pd.DataFrame):
    z = run_df[np.isfinite(run_df["token_drift"])].copy()
    if z.empty:
        return pd.DataFrame(columns=["frame", "drift_mean", "drift_std"])
    g = z.groupby("frame")["token_drift"]
    out = (
        g.agg(drift_mean="mean", drift_std="std")
        .reset_index()
        .sort_values("frame")
    )
    return out


def aggregate_acceptance(acc_df: pd.DataFrame):
    if acc_df.empty:
        return pd.DataFrame(columns=["frame", "acc_mean", "acc_std"])
    z = acc_df.copy()
    z = z[np.isfinite(z["acceptance_rate"])].copy()
    if z.empty:
        return pd.DataFrame(columns=["frame", "acc_mean", "acc_std"])
    z["frame"] = z["frame"].round().astype(int)
    g = z.groupby("frame")["acceptance_rate"]
    out = (
        g.agg(acc_mean="mean", acc_std="std")
        .reset_index()
        .sort_values("frame")
    )
    return out


def add_vlines(ax, max_frame: int, step: int):
    x = step
    while x <= max_frame:
        ax.axvline(
            x,
            color=VLINE_GRID_COLOR,
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )
        x += step


def panel_plot(
    drift_df: pd.DataFrame,
    best_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    out_png: str,
    title: str,
    label_mode: str,
    max_frame: int,
    vline_step: int,
):
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.8]},
    )

    if not drift_df.empty:
        ax_top.plot(
            drift_df["frame"],
            drift_df["drift_mean"],
            color=DRIFT_LINE_COLOR,
            linewidth=1.8,
        )
    if not best_df.empty and "token_drift_mean" in best_df.columns:
        win_drift = best_df[np.isfinite(best_df["token_drift_mean"])]
        if not win_drift.empty:
            ax_top.plot(
                win_drift["time_center"],
                win_drift["token_drift_mean"],
                color="#264653",
                linewidth=1.6,
                marker="o",
                markersize=2.7,
                alpha=0.9,
                label="window token JSD (best)",
                zorder=4,
            )
            ax_top.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_top.set_ylabel("token JSD")
    ax_top.grid(alpha=0.25)
    ax_top.set_xlim(0, max_frame)
    add_vlines(ax_top, max_frame=max_frame, step=vline_step)

    if not best_df.empty:
        ax_mid.scatter(best_df["time_center"], best_df["lat_mean"], color="#ff7f0e", s=28, zorder=3)
        sw = best_df[best_df["discontinuity"]]
        if not sw.empty:
            ax_mid.scatter(
                sw["time_center"],
                sw["lat_mean"],
                facecolors="none",
                edgecolors=BOX_EDGE_COLOR,
                marker="s",
                s=95,
                linewidths=1.4,
                zorder=4,
            )
            for sx in sw["time_center"].tolist():
                ax_top.axvline(
                    sx, color=VLINE_SWITCH_COLOR, linewidth=1.2, alpha=0.8
                )
                ax_mid.axvline(
                    sx, color=VLINE_SWITCH_COLOR, linewidth=1.2, alpha=0.8
                )
                ax_bot.axvline(
                    sx, color=VLINE_SWITCH_COLOR, linewidth=1.2, alpha=0.8
                )

        for _, row in best_df.iterrows():
            if label_mode == "ctx_inf":
                txt = f"ctx{int(row['ctx'])}_inf{int(row['inf'])}"
            else:
                txt = f"ib{int(row['ib'])}_tb{int(row['tb'])}"
            ax_mid.text(
                row["time_center"],
                row["lat_mean"],
                txt,
                fontsize=10,
                rotation=35,
                ha="left",
                va="bottom",
                alpha=0.88,
            )

    ax_mid.set_ylabel("latency (lower better)")
    ax_mid.grid(alpha=0.25)
    ax_mid.set_xlim(0, max_frame)
    add_vlines(ax_mid, max_frame=max_frame, step=vline_step)

    if not acc_df.empty:
        ax_bot.plot(
            acc_df["frame"],
            acc_df["acc_mean"],
            color=ACCEPT_LINE_COLOR,
            linewidth=1.8,
            label="draft acceptance rate",
        )
    ax_bot.set_ylabel("acceptance rate")
    ax_bot.set_xlabel("frame index")
    ax_bot.grid(alpha=0.25)
    ax_bot.set_xlim(0, max_frame)
    ax_bot.set_ylim(0.0, 1.02)
    add_vlines(ax_bot, max_frame=max_frame, step=vline_step)

    seg_msgs_drift = []
    for a in range(0, max_frame, vline_step):
        b = min(max_frame, a + vline_step)
        seg = drift_df[(drift_df["frame"] >= a) & (drift_df["frame"] < b)]
        if len(seg) == 0:
            continue
        seg_msgs_drift.append(f"[{a},{b})={seg['drift_mean'].mean():.4f}")
    drift_footer = " | ".join(seg_msgs_drift) if seg_msgs_drift else "no drift values"

    seg_msgs_acc = []
    for a in range(0, max_frame, vline_step):
        b = min(max_frame, a + vline_step)
        seg = acc_df[(acc_df["frame"] >= a) & (acc_df["frame"] < b)]
        if len(seg) == 0:
            continue
        seg_msgs_acc.append(f"[{a},{b})={seg['acc_mean'].mean():.4f}")
    acc_footer = " | ".join(seg_msgs_acc) if seg_msgs_acc else "no acceptance values"

    chain_footer = "chain corr unavailable"
    if (
        not best_df.empty
        and "token_drift_mean" in best_df.columns
        and "switch_flag" in best_df.columns
    ):
        drift_switch = _safe_corr(best_df["token_drift_mean"], best_df["switch_flag"])
        if "spec_reject_ratio" in best_df.columns:
            drift_reject = _safe_corr(best_df["token_drift_mean"], best_df["spec_reject_ratio"])
            reject_switch = _safe_corr(best_df["spec_reject_ratio"], best_df["switch_flag"])
            chain_footer = (
                f"window corr: drift~reject={drift_reject:.3f}, "
                f"reject~switch={reject_switch:.3f}, drift~switch={drift_switch:.3f}; "
                f"switch_rate={best_df['switch_flag'].mean():.3f}"
            )
        else:
            chain_footer = (
                f"window corr: drift~switch={drift_switch:.3f}; "
                f"switch_rate={best_df['switch_flag'].mean():.3f}"
            )

    ax_top.set_title(title)
    fig.text(
        0.01,
        0.01,
        (
            f"token JSD mean by {vline_step}-frame segments: {drift_footer}\n"
            f"draft acceptance mean by {vline_step}-frame segments: {acc_footer}\n"
            f"{chain_footer}"
        ),
        ha="left",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Token JSD + best latency plots for spec static logs."
    )
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--pattern", default="doh_spec_static_*.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-frames", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--max-frame", type=int, default=4000)
    ap.add_argument("--vline-step", type=int, default=800)
    ap.add_argument("--min-final-frame", type=int, default=3999)
    ap.add_argument("--fixed-ib", type=int, default=32)
    ap.add_argument("--fixed-tb", type=int, default=16)
    ap.add_argument("--fixed-ctx", type=int, default=4)
    ap.add_argument("--fixed-inf", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(args.logs_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {args.pattern}")

    runs = []
    for fp in files:
        run = load_run(fp)
        if run is None:
            continue
        maxf = int(run["df"]["frame"].max())
        if maxf < args.min_final_frame:
            continue
        runs.append(run)

    if not runs:
        raise SystemExit("No valid full runs after filtering.")

    all_df = pd.concat([r["df"] for r in runs], ignore_index=True)
    acc_frames = [r["acc_df"] for r in runs if r["acc_df"] is not None and not r["acc_df"].empty]
    if acc_frames:
        all_acc_df = pd.concat(acc_frames, ignore_index=True)
    else:
        all_acc_df = pd.DataFrame(
            columns=[
                "run_name",
                "ctx",
                "inf",
                "ib",
                "tb",
                "frame",
                "acceptance_rate",
                "accepted",
                "proposed",
                "reject_tokens",
                "reverify_count",
                "early_stop_count",
                "verify_tokens",
                "draft_time_ms",
                "verify_time_ms",
            ]
        )

    # Panel A: fixed (ib,tb), vary (ctx,inf)
    panel_a = all_df[
        (all_df["ib"] == args.fixed_ib) & (all_df["tb"] == args.fixed_tb)
    ].copy()
    if panel_a.empty:
        raise SystemExit(
            f"No runs found for fixed ib,tb=({args.fixed_ib},{args.fixed_tb})"
        )
    drift_a = aggregate_drift(panel_a)
    acc_a_runs = all_acc_df[
        (all_acc_df["ib"] == args.fixed_ib) & (all_acc_df["tb"] == args.fixed_tb)
    ].copy()
    acc_a = aggregate_acceptance(acc_a_runs)
    win_rows_a = []
    for _, sub in panel_a.groupby("run_name"):
        win_rows_a.append(build_window_metrics(sub, window_frames=args.window_frames))
    win_a = pd.concat(win_rows_a, ignore_index=True)
    spec_rows_a = []
    for _, sub in acc_a_runs.groupby("run_name"):
        spec_rows_a.append(build_spec_window_metrics(sub, window_frames=args.window_frames))
    if spec_rows_a:
        spec_win_a = pd.concat(spec_rows_a, ignore_index=True)
        win_a = win_a.merge(
            spec_win_a,
            on=["run_name", "ctx", "inf", "ib", "tb", "window_id"],
            how="left",
        )
    best_a = choose_best_per_window(win_a, tau=args.tau)
    best_a = add_switch_chain_features(best_a, key_cols=["ctx", "inf"])
    best_a.to_csv(
        os.path.join(
            args.out_dir,
            f"best_fix_ib{args.fixed_ib}_tb{args.fixed_tb}_tau{str(args.tau).replace('.', 'p')}.csv",
        ),
        index=False,
    )
    make_chain_sync_view(best_a).to_csv(
        os.path.join(
            args.out_dir,
            f"chain_fix_ib{args.fixed_ib}_tb{args.fixed_tb}_tau{str(args.tau).replace('.', 'p')}.csv",
        ),
        index=False,
    )
    drift_a.to_csv(
        os.path.join(
            args.out_dir, f"drift_fix_ib{args.fixed_ib}_tb{args.fixed_tb}.csv"
        ),
        index=False,
    )
    acc_a.to_csv(
        os.path.join(
            args.out_dir, f"acceptance_fix_ib{args.fixed_ib}_tb{args.fixed_tb}.csv"
        ),
        index=False,
    )
    out_a = os.path.join(
        args.out_dir,
        f"token_drift_fix_ib{args.fixed_ib}_tb{args.fixed_tb}_vary_ctxinf_tau{str(args.tau).replace('.', 'p')}.png",
    )
    panel_plot(
        drift_df=drift_a,
        best_df=best_a,
        acc_df=acc_a,
        out_png=out_a,
        title=(
            f"Token JSD (top) + best latency under acc>={args.tau} (middle) + draft acceptance (bottom) "
            f"| fixed (ib,tb)=({args.fixed_ib},{args.fixed_tb}), vary (ctx,inf)"
        ),
        label_mode="ctx_inf",
        max_frame=args.max_frame,
        vline_step=args.vline_step,
    )

    # Panel B: fixed (ctx,inf), vary (ib,tb)
    panel_b = all_df[
        (all_df["ctx"] == args.fixed_ctx) & (all_df["inf"] == args.fixed_inf)
    ].copy()
    if panel_b.empty:
        raise SystemExit(
            f"No runs found for fixed ctx,inf=({args.fixed_ctx},{args.fixed_inf})"
        )
    drift_b = aggregate_drift(panel_b)
    acc_b_runs = all_acc_df[
        (all_acc_df["ctx"] == args.fixed_ctx) & (all_acc_df["inf"] == args.fixed_inf)
    ].copy()
    acc_b = aggregate_acceptance(acc_b_runs)
    win_rows_b = []
    for _, sub in panel_b.groupby("run_name"):
        win_rows_b.append(build_window_metrics(sub, window_frames=args.window_frames))
    win_b = pd.concat(win_rows_b, ignore_index=True)
    spec_rows_b = []
    for _, sub in acc_b_runs.groupby("run_name"):
        spec_rows_b.append(build_spec_window_metrics(sub, window_frames=args.window_frames))
    if spec_rows_b:
        spec_win_b = pd.concat(spec_rows_b, ignore_index=True)
        win_b = win_b.merge(
            spec_win_b,
            on=["run_name", "ctx", "inf", "ib", "tb", "window_id"],
            how="left",
        )
    best_b = choose_best_per_window(win_b, tau=args.tau)
    best_b = add_switch_chain_features(best_b, key_cols=["ib", "tb"])
    best_b.to_csv(
        os.path.join(
            args.out_dir,
            f"best_fix_ctx{args.fixed_ctx}_inf{args.fixed_inf}_tau{str(args.tau).replace('.', 'p')}.csv",
        ),
        index=False,
    )
    make_chain_sync_view(best_b).to_csv(
        os.path.join(
            args.out_dir,
            f"chain_fix_ctx{args.fixed_ctx}_inf{args.fixed_inf}_tau{str(args.tau).replace('.', 'p')}.csv",
        ),
        index=False,
    )
    drift_b.to_csv(
        os.path.join(
            args.out_dir, f"drift_fix_ctx{args.fixed_ctx}_inf{args.fixed_inf}.csv"
        ),
        index=False,
    )
    acc_b.to_csv(
        os.path.join(
            args.out_dir, f"acceptance_fix_ctx{args.fixed_ctx}_inf{args.fixed_inf}.csv"
        ),
        index=False,
    )
    out_b = os.path.join(
        args.out_dir,
        f"token_drift_fix_ctx{args.fixed_ctx}_inf{args.fixed_inf}_vary_ibtb_tau{str(args.tau).replace('.', 'p')}.png",
    )
    panel_plot(
        drift_df=drift_b,
        best_df=best_b,
        acc_df=acc_b,
        out_png=out_b,
        title=(
            f"Token JSD (top) + best latency under acc>={args.tau} (middle) + draft acceptance (bottom) "
            f"| fixed (ctx,inf)=({args.fixed_ctx},{args.fixed_inf}), vary (ib,tb)"
        ),
        label_mode="ib_tb",
        max_frame=args.max_frame,
        vline_step=args.vline_step,
    )

    source_mode = (
        "token_ids"
        if (all_df["drift_source"] == "token_ids").any()
        else "tps_proxy"
    )
    print("[OK] output dir:", args.out_dir)
    print("[OK] panel A:", out_a)
    print("[OK] panel B:", out_b)
    print("[OK] drift source used:", source_mode)


if __name__ == "__main__":
    main()
