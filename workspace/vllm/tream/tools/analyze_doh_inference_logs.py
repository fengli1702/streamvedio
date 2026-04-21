import argparse
import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Run-name parsing
# -----------------------------
RE_DID = re.compile(
    r"^doh_ibctx_did_1\.22_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_rep(?P<rep>\d+)$"
)
RE_COD = re.compile(
    r"^doh_cod_1\.22_(?P<mode>joint|train_only|infer_only)_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_spec(?P<spec>on|off)$"
)


def parse_run_name(run_name: str) -> dict:
    m = RE_DID.match(run_name)
    if m:
        d = m.groupdict()
        d.update({
            "group": "ibctx_did",
            "mode": "did",
            "spec": None,
        })
        return _cast(d)

    m = RE_COD.match(run_name)
    if m:
        d = m.groupdict()
        d.update({
            "group": "codesign",
        })
        return _cast(d)

    return {"group": "unknown", "run_name": run_name}


def _cast(d: dict) -> dict:
    out = dict(d)
    for k in ["ctx", "inf", "ib", "tb", "rep"]:
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except Exception:
                pass
    if "lr" in out and out["lr"] is not None:
        try:
            out["lr_f"] = float(out["lr"])
        except Exception:
            out["lr_f"] = np.nan
    return out


# -----------------------------
# Frame id parsing from frame_path
# -----------------------------

def frame_id_from_path(p: str, fallback: int) -> int:
    base = os.path.basename(p)
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return fallback


# -----------------------------
# JSONL reading
# -----------------------------

def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    fallback_idx = 0
    segment_id = 0
    cur_from = None
    cur_to = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # directory transition event
            if isinstance(obj, dict) and obj.get("event") == "directory_transition":
                segment_id += 1
                cur_from = obj.get("from_dir")
                cur_to = obj.get("to_dir")
                continue

            # final stats line
            if "total_processing_time" in obj and "avg_time_per_frame" in obj:
                continue

            if "frame_path" not in obj or "perplexity" not in obj or "latency" not in obj:
                continue

            fallback_idx += 1
            fid = frame_id_from_path(obj["frame_path"], fallback_idx)
            rows.append({
                "frame_id": fid,
                "frame_path": obj["frame_path"],
                "perplexity": float(obj["perplexity"]),
                "latency_s": float(obj["latency"]),
                "lora_step": int(obj.get("lora_step", -1)),
                "segment_id": segment_id,
                "segment_from": cur_from,
                "segment_to": cur_to,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["segment_id", "frame_id"]).reset_index(drop=True)
    return df


# -----------------------------
# Window aggregation + Pareto
# -----------------------------

def add_windows(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    df["window_id"] = (df["frame_id"] // window).astype(int)
    return df


def summarize_run(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_ms"] = df["latency_s"] * 1000.0
    g = df.groupby(["segment_id", "window_id"], as_index=False)
    out = g.agg(
        ppl_mean=("perplexity", "mean"),
        ppl_p50=("perplexity", "median"),
        lat_mean_ms=("lat_ms", "mean"),
        lat_p50_ms=("lat_ms", "median"),
        lat_p95_ms=("lat_ms", lambda x: np.percentile(x, 95)),
        lora_step_mean=("lora_step", "mean"),
        n=("lat_ms", "size"),
    )
    return out


def pareto_front(points: pd.DataFrame, x="lat_mean_ms", y="ppl_mean") -> pd.DataFrame:
    pts = points.sort_values(x).reset_index(drop=True)
    best_y = np.inf
    keep = []
    for i, row in pts.iterrows():
        if row[y] < best_y:
            keep.append(i)
            best_y = row[y]
    return pts.loc[keep].reset_index(drop=True)


def pick_best(points: pd.DataFrame, ppl_slack: float) -> pd.Series:
    best_ppl = points["ppl_mean"].min()
    feasible = points[points["ppl_mean"] <= best_ppl * (1.0 + ppl_slack)]
    if feasible.empty:
        feasible = points
    return feasible.sort_values("lat_mean_ms").iloc[0]


# -----------------------------
# Plot helpers
# -----------------------------

def savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--ppl_slack", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=0, help="drop frames with frame_id < warmup")
    ap.add_argument("--pattern", default="doh_*.jsonl", help="glob pattern inside logs_dir")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.logs_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No jsonl found in {args.logs_dir}")

    inv_rows = []
    run_windows = {}

    # 1) read + inventory
    for fp in files:
        run_name = os.path.splitext(os.path.basename(fp))[0]
        meta = parse_run_name(run_name)
        meta["run_name"] = run_name
        meta["file"] = fp

        df = read_jsonl(fp)
        if df.empty:
            print(f"[WARN] empty parsed: {run_name}")
            continue

        if args.warmup > 0:
            df = df[df["frame_id"] >= args.warmup].copy()

        if df.empty:
            print(f"[WARN] empty after warmup: {run_name}")
            continue

        df = add_windows(df, args.window)

        inv_rows.append({
            **meta,
            "n_records": len(df),
            "n_segments": int(df["segment_id"].max() if not df["segment_id"].isna().all() else 0) + 1,
            "frame_min": int(df["frame_id"].min()),
            "frame_max": int(df["frame_id"].max()),
            "ppl_mean": float(df["perplexity"].mean()),
            "lat_mean_ms": float((df["latency_s"] * 1000).mean()),
        })

        w = summarize_run(df)
        w["run_name"] = run_name
        run_windows[run_name] = w

    inv = pd.DataFrame(inv_rows).sort_values(["group", "mode", "run_name"])
    os.makedirs(args.out_dir, exist_ok=True)
    inv.to_csv(os.path.join(args.out_dir, "runs_inventory.csv"), index=False)
    print(f"[OK] inventory -> {args.out_dir}/runs_inventory.csv  (n={len(inv)})")

    all_w = pd.concat(run_windows.values(), ignore_index=True)
    all_w.to_csv(os.path.join(args.out_dir, "window_metrics.csv"), index=False)
    print(f"[OK] window metrics -> {args.out_dir}/window_metrics.csv  (rows={len(all_w)})")

    # -----------------------------
    # Q2: DID on ibctx_did (overall)
    # -----------------------------
    did_inv = inv[inv["group"] == "ibctx_did"].copy()
    if not did_inv.empty:
        def _did(metric_col: str) -> pd.DataFrame:
            t = did_inv.pivot_table(
                index=["rep"], columns=["ctx", "ib"], values=metric_col, aggfunc="mean"
            )
            D2 = t[(2, 16)] - t[(2, 2)]
            D8 = t[(8, 16)] - t[(8, 2)]
            DID = D8 - D2
            out = pd.DataFrame({
                "rep": D2.index,
                "D_ctx2": D2.values,
                "D_ctx8": D8.values,
                "DID": DID.values,
            })
            out["metric"] = metric_col
            out["DID_mean"] = out["DID"].mean()
            out["DID_std"] = out["DID"].std(ddof=1) if len(out) > 1 else np.nan
            return out

        did_lat = _did("lat_mean_ms")
        did_ppl = _did("ppl_mean")
        did = pd.concat([did_lat, did_ppl], ignore_index=True)
        did.to_csv(os.path.join(args.out_dir, "q2_did_overall.csv"), index=False)

        plt.figure()
        for metric, sub in did.groupby("metric"):
            plt.scatter([metric] * len(sub), sub["DID"])
        plt.ylabel("DID value")
        plt.title("Q2 DID (per-rep)")
        savefig(os.path.join(args.out_dir, "fig_q2_did_overall.png"))
        print(f"[OK] Q2 DID -> {args.out_dir}/q2_did_overall.csv and fig_q2_did_overall.png")

    # -----------------------------
    # Q1/Q3 on codesign workload c4_i4
    # -----------------------------
    cod = inv[(inv["group"] == "codesign") & (inv["ctx"] == 4) & (inv["inf"] == 4)].copy()
    if cod.empty:
        print("[WARN] no codesign c4_i4 runs parsed; skip Q1/Q3")
        return

    cod_runs = cod["run_name"].tolist()
    cod_w = all_w[all_w["run_name"].isin(cod_runs)].copy()

    # Q1: per-window Pareto + best-id timeline
    best_rows = []
    front_samples = []
    for (seg, win), sub in cod_w.groupby(["segment_id", "window_id"]):
        if len(sub) < 4:
            continue
        sub2 = sub.merge(inv[["run_name", "mode", "spec", "ib", "tb", "lr"]], on="run_name", how="left")
        best = pick_best(sub2, args.ppl_slack)
        best_rows.append({
            "segment_id": seg,
            "window_id": win,
            "best_run": best["run_name"],
            "best_mode": best.get("mode"),
            "best_spec": best.get("spec"),
            "best_ib": best.get("ib"),
            "best_tb": best.get("tb"),
            "best_ppl": best["ppl_mean"],
            "best_lat_ms": best["lat_mean_ms"],
            "n_points": len(sub2),
        })

        if win % 20 == 0:
            front = pareto_front(sub2[["run_name", "lat_mean_ms", "ppl_mean", "mode", "spec", "ib", "tb"]])
            front["segment_id"] = seg
            front["window_id"] = win
            front_samples.append(front)

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(args.out_dir, "q1_best_timeline.csv"), index=False)

    if not best_df.empty:
        plt.figure(figsize=(10, 3))
        plt.plot(best_df["window_id"], best_df["best_ib"], marker="o", linestyle="None")
        plt.xlabel("window_id")
        plt.ylabel("best ib")
        plt.title("Q1: best ib over time (c4_i4)")
        savefig(os.path.join(args.out_dir, "fig_q1_best_ib_timeline.png"))

        plt.figure(figsize=(10, 3))
        plt.plot(best_df["window_id"], best_df["best_tb"], marker="o", linestyle="None")
        plt.xlabel("window_id")
        plt.ylabel("best tb")
        plt.title("Q1: best tb over time (c4_i4)")
        savefig(os.path.join(args.out_dir, "fig_q1_best_tb_timeline.png"))

    cod_overall = inv[inv["run_name"].isin(cod_runs)].copy()
    best_ppl_global = cod_overall["ppl_mean"].min()
    feasible = cod_overall[cod_overall["ppl_mean"] <= best_ppl_global * (1 + args.ppl_slack)]
    if feasible.empty:
        feasible = cod_overall
    global_best_run = feasible.sort_values("lat_mean_ms").iloc[0]["run_name"]

    if not best_df.empty:
        gb = cod_w[cod_w["run_name"] == global_best_run][["segment_id", "window_id", "lat_mean_ms"]].rename(
            columns={"lat_mean_ms": "gb_lat_ms"}
        )
        reg = best_df.merge(gb, on=["segment_id", "window_id"], how="left")
        reg["regret_ms"] = reg["gb_lat_ms"] - reg["best_lat_ms"]
        reg.to_csv(os.path.join(args.out_dir, "q1_regret.csv"), index=False)

        plt.figure(figsize=(10, 3))
        plt.plot(reg["window_id"], reg["regret_ms"])
        plt.xlabel("window_id")
        plt.ylabel("regret (ms)")
        plt.title(f"Q1: regret of global-best ({global_best_run})")
        savefig(os.path.join(args.out_dir, "fig_q1_regret.png"))

    if front_samples:
        fs = pd.concat(front_samples, ignore_index=True)
        fs.to_csv(os.path.join(args.out_dir, "q1_front_samples.csv"), index=False)
        plt.figure()
        for (seg, win), sub in fs.groupby(["segment_id", "window_id"]):
            plt.scatter(sub["lat_mean_ms"], sub["ppl_mean"], label=f"s{seg}w{win}", s=12)
        plt.xlabel("lat_mean_ms")
        plt.ylabel("ppl_mean")
        plt.title("Q1: sampled Pareto fronts (overlay)")
        plt.legend(fontsize=7, ncol=2)
        savefig(os.path.join(args.out_dir, "fig_q1_front_samples_overlay.png"))

    print(f"[OK] Q1 outputs -> q1_best_timeline.csv, q1_regret.csv, figs under {args.out_dir}/")

    # -----------------------------
    # Q3: strategy comparison + spec gain
    # -----------------------------
    strat_rows = []
    for (seg, win), sub in cod_w.groupby(["segment_id", "window_id"]):
        sub2 = sub.merge(inv[["run_name", "mode", "spec", "ib", "tb"]], on="run_name", how="left")
        for mode in ["joint", "train_only", "infer_only"]:
            s = sub2[sub2["mode"] == mode]
            if s.empty:
                continue
            best = pick_best(s, args.ppl_slack)
            strat_rows.append({
                "segment_id": seg,
                "window_id": win,
                "mode": mode,
                "best_run": best["run_name"],
                "best_spec": best.get("spec"),
                "best_ib": best.get("ib"),
                "best_tb": best.get("tb"),
                "best_lat_ms": best["lat_mean_ms"],
                "best_ppl": best["ppl_mean"],
            })
    strat = pd.DataFrame(strat_rows)
    strat.to_csv(os.path.join(args.out_dir, "q3_strategy_best_timeline.csv"), index=False)

    if not strat.empty:
        plt.figure(figsize=(10, 3))
        for mode, sub in strat.groupby("mode"):
            tmp = sub.groupby("window_id", as_index=False)["best_lat_ms"].mean()
            plt.plot(tmp["window_id"], tmp["best_lat_ms"], label=mode)
        plt.xlabel("window_id")
        plt.ylabel("best latency (ms)")
        plt.title("Q3: per-window best latency by strategy (c4_i4)")
        plt.legend()
        savefig(os.path.join(args.out_dir, "fig_q3_strategy_best_latency.png"))

    cod_meta = inv[inv["run_name"].isin(cod_runs)][["run_name", "mode", "spec", "ib", "tb"]]
    overall = inv[inv["run_name"].isin(cod_runs)][["run_name", "ppl_mean", "lat_mean_ms"]]
    paired = cod_meta.merge(overall, on="run_name", how="left")
    paired = paired.dropna(subset=["spec", "ib", "tb", "mode"])
    piv = paired.pivot_table(
        index=["mode", "ib", "tb"], columns="spec", values=["lat_mean_ms", "ppl_mean"], aggfunc="mean"
    )
    if ("lat_mean_ms", "on") in piv.columns and ("lat_mean_ms", "off") in piv.columns:
        piv = piv.reset_index()
        # flatten multi-index columns
        flat_cols = []
        for col in piv.columns:
            if isinstance(col, tuple):
                flat_cols.append("_".join([c for c in col if c]))
            else:
                flat_cols.append(col)
        piv.columns = flat_cols

        piv["delta_lat_ms_off_minus_on"] = piv["lat_mean_ms_off"] - piv["lat_mean_ms_on"]
        piv["delta_ppl_off_minus_on"] = piv["ppl_mean_off"] - piv["ppl_mean_on"]
        piv.to_csv(os.path.join(args.out_dir, "q3_spec_gain_paired.csv"), index=False)

        sub = piv[(piv["mode"] == "joint") & (piv["ib"] == 8)].sort_values("tb")
        if not sub.empty:
            plt.figure()
            plt.plot(sub["tb"], sub["delta_lat_ms_off_minus_on"], marker="o")
            plt.xlabel("tb")
            plt.ylabel("spec gain: lat_off - lat_on (ms)")
            plt.title("spec gain vs tb (joint, ib=8)")
            savefig(os.path.join(args.out_dir, "fig_q3_spec_gain_vs_tb_joint_ib8.png"))

        sub = piv[(piv["mode"] == "infer_only") & (piv["tb"] == 8)].sort_values("ib")
        if not sub.empty:
            plt.figure()
            plt.plot(sub["ib"], sub["delta_lat_ms_off_minus_on"], marker="o")
            plt.xlabel("ib")
            plt.ylabel("spec gain: lat_off - lat_on (ms)")
            plt.title("spec gain vs ib (infer_only, tb=8)")
            savefig(os.path.join(args.out_dir, "fig_q3_spec_gain_vs_ib_infer_tb8.png"))

        print(f"[OK] Q3 spec gain -> q3_spec_gain_paired.csv + figs")

    print("[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
