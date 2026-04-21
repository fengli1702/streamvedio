#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_dynamic_manifest(path: Path, group: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            if group != "ALL" and not case_id.startswith(group):
                continue
            out.append(row)
    return out


def _load_static_chain(path: Path) -> Dict[str, np.ndarray]:
    frames: List[float] = []
    drift: List[float] = []
    ctxs: List[int] = []
    infs: List[int] = []
    switches: List[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["time_center"])
                d = float(row["token_drift_mean"])
                c = int(float(row["ctx"]))
                i = int(float(row["inf"]))
                s = int(float(row.get("switch_flag", "0")))
            except Exception:
                continue
            if not math.isfinite(t):
                continue
            frames.append(t)
            drift.append(d if math.isfinite(d) else np.nan)
            ctxs.append(c)
            infs.append(i)
            switches.append(s)
    if not frames:
        raise RuntimeError(f"no rows parsed from static chain: {path}")
    order = np.argsort(np.asarray(frames))
    x = np.asarray(frames, dtype=float)[order]
    return {
        "frame": x,
        "drift": np.asarray(drift, dtype=float)[order],
        "ctx": np.asarray(ctxs, dtype=float)[order],
        "inf": np.asarray(infs, dtype=float)[order],
        "switch": np.asarray(switches, dtype=float)[order],
    }


def _load_scheduler_seq(log_dir: Path, run_name: str) -> List[Tuple[int, int, int]]:
    path = log_dir / f"scheduler_{run_name}.jsonl"
    if not path.exists():
        return []
    by_step: Dict[int, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            step = obj.get("scheduler/step", idx)
            try:
                step = int(step)
            except Exception:
                step = idx
            ctx = obj.get("scheduler/new_context_length")
            inf = obj.get("scheduler/new_inference_length")
            if ctx is None:
                ctx = obj.get("scheduler/context_length")
            if inf is None:
                inf = obj.get("scheduler/inference_length")
            try:
                c = int(ctx)
                i = int(inf)
            except Exception:
                continue
            by_step[step] = (c, i)
    if not by_step:
        return []
    out: List[Tuple[int, int, int]] = []
    for dense_idx, (_step, (ctx, inf)) in enumerate(sorted(by_step.items())):
        out.append((dense_idx, ctx, inf))
    return out


def _load_infer_frames(log_dir: Path, run_name: str) -> List[int]:
    path = log_dir / f"{run_name}.jsonl"
    if not path.exists():
        return []
    frames: List[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("event") != "inference_cycle":
                continue
            fi = obj.get("frame_index")
            try:
                fi = int(fi)
            except Exception:
                continue
            frames.append(fi)
    return frames


def _step_to_frame_axis(n_steps: int, frame_axis: List[int]) -> List[float]:
    if n_steps <= 0:
        return []
    if not frame_axis:
        return [float(i) for i in range(n_steps)]
    if n_steps == 1:
        return [float(frame_axis[0])]
    m = len(frame_axis)
    xs: List[float] = []
    for k in range(n_steps):
        q = k / float(n_steps - 1)
        idx = int(round(q * (m - 1)))
        idx = max(0, min(m - 1, idx))
        xs.append(float(frame_axis[idx]))
    return xs


def _sample_hold_last(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    # xs must be sorted ascending
    out = np.empty_like(xq, dtype=float)
    for j, q in enumerate(xq):
        idx = int(np.searchsorted(xs, q, side="right") - 1)
        if idx < 0:
            idx = 0
        if idx >= len(ys):
            idx = len(ys) - 1
        out[j] = float(ys[idx])
    return out


def _estimate_drift_switch_threshold(
    drift: np.ndarray,
    switch: np.ndarray,
    prob_target: float = 0.60,
    min_support: int = 5,
) -> Tuple[float, float, int]:
    m = np.isfinite(drift) & np.isfinite(switch)
    d = drift[m].astype(float)
    s = (switch[m] > 0.5).astype(float)
    if d.size == 0:
        return float("nan"), float("nan"), 0
    uniq = np.unique(d)
    # Try low->high threshold: choose the smallest threshold that already reaches target,
    # so it still covers enough windows.
    chosen: Optional[Tuple[float, float, int]] = None
    for th in uniq:
        mask = d >= th
        supp = int(mask.sum())
        if supp < min_support:
            continue
        prob = float(s[mask].mean()) if supp > 0 else float("nan")
        if math.isfinite(prob) and prob >= prob_target:
            chosen = (float(th), prob, supp)
            break
    if chosen is not None:
        return chosen
    # Fallback: best precision under min support.
    best: Optional[Tuple[float, float, int]] = None
    for th in uniq:
        mask = d >= th
        supp = int(mask.sum())
        if supp < min_support:
            continue
        prob = float(s[mask].mean()) if supp > 0 else float("nan")
        cand = (float(th), prob, supp)
        if best is None or cand[1] > best[1] or (cand[1] == best[1] and cand[2] > best[2]):
            best = cand
    if best is not None:
        return best
    # Last-resort single-point support.
    th = float(np.nanmax(d))
    mask = d >= th
    supp = int(mask.sum())
    prob = float(s[mask].mean()) if supp > 0 else float("nan")
    return th, prob, supp


def build_plot_data(
    dynamic_rows: List[Dict[str, str]],
    static_chain: Dict[str, np.ndarray],
    log_dir: Path,
) -> Dict[str, np.ndarray]:
    x = static_chain["frame"]
    offline_ctx = static_chain["ctx"]
    offline_inf = static_chain["inf"]
    offline_workload = offline_ctx * offline_inf
    offline_switch = static_chain["switch"]
    drift = static_chain["drift"]

    online_ctxs: List[np.ndarray] = []
    online_infs: List[np.ndarray] = []
    online_workloads: List[np.ndarray] = []
    online_switches: List[np.ndarray] = []
    used_cases: List[str] = []

    for row in dynamic_rows:
        run_name = (row.get("run_name") or "").strip()
        case_id = (row.get("case_id") or "").strip()
        if not run_name:
            continue
        seq = _load_scheduler_seq(log_dir, run_name)
        if not seq:
            continue
        frames = _load_infer_frames(log_dir, run_name)
        sx = np.asarray(_step_to_frame_axis(len(seq), frames), dtype=float)
        sctx = np.asarray([p[1] for p in seq], dtype=float)
        sinf = np.asarray([p[2] for p in seq], dtype=float)

        # project scheduler path to static-chain frame axis via hold-last.
        ctx_q = _sample_hold_last(sx, sctx, x)
        inf_q = _sample_hold_last(sx, sinf, x)
        w_q = ctx_q * inf_q
        sw = np.zeros_like(w_q)
        sw[1:] = (ctx_q[1:] != ctx_q[:-1]) | (inf_q[1:] != inf_q[:-1])

        online_ctxs.append(ctx_q)
        online_infs.append(inf_q)
        online_workloads.append(w_q)
        online_switches.append(sw.astype(float))
        used_cases.append(case_id)

    if not online_workloads:
        raise RuntimeError("no dynamic scheduler traces loaded")

    C = np.vstack(online_ctxs)
    I = np.vstack(online_infs)
    W = np.vstack(online_workloads)
    S = np.vstack(online_switches)
    th, prob, supp = _estimate_drift_switch_threshold(drift=drift, switch=offline_switch)

    return {
        "frame": x,
        "drift": drift,
        "offline_ctx": offline_ctx,
        "offline_inf": offline_inf,
        "offline_workload": offline_workload,
        "offline_switch": offline_switch,
        "online_ctx_mean": np.nanmean(C, axis=0),
        "online_ctx_p25": np.nanpercentile(C, 25, axis=0),
        "online_ctx_p75": np.nanpercentile(C, 75, axis=0),
        "online_inf_mean": np.nanmean(I, axis=0),
        "online_inf_p25": np.nanpercentile(I, 25, axis=0),
        "online_inf_p75": np.nanpercentile(I, 75, axis=0),
        "online_workload_mean": np.nanmean(W, axis=0),
        "online_workload_p25": np.nanpercentile(W, 25, axis=0),
        "online_workload_p75": np.nanpercentile(W, 75, axis=0),
        "online_switch_rate": np.nanmean(S, axis=0),
        "drift_threshold": np.asarray([th], dtype=float),
        "drift_threshold_prob": np.asarray([prob], dtype=float),
        "drift_threshold_support": np.asarray([supp], dtype=float),
        "n_runs": np.asarray([W.shape[0]], dtype=float),
    }


def plot_triple(data: Dict[str, np.ndarray], out_path: Path, title: str) -> None:
    x = data["frame"]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(14, 12), sharex=True, constrained_layout=True
    )

    # Panel 1: drift
    ax1.plot(x, data["drift"], color="#2b6cb0", lw=1.8, label="token drift (static chain)")
    th = float(data["drift_threshold"][0])
    prob = float(data["drift_threshold_prob"][0])
    supp = int(data["drift_threshold_support"][0])
    if math.isfinite(th):
        ax1.axhline(
            th,
            color="#ff7f0e",
            lw=2.0,
            ls="--",
            label=f"drift threshold={th:.3f} | P(switch|drift>=th)={prob:.2f} (n={supp})",
        )
    ax1.set_ylabel("token drift")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: ctx only
    ax2.plot(x, data["offline_ctx"], color="#d62728", lw=2.2, label="offline oracle ctx")
    ax2.plot(x, data["online_ctx_mean"], color="#1f77b4", lw=1.8, label="online scheduler mean ctx")
    ax2.fill_between(
        x,
        data["online_ctx_p25"],
        data["online_ctx_p75"],
        color="#1f77b4",
        alpha=0.18,
        label="online ctx IQR",
    )
    ax2.set_ylabel("ctx")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: inf only
    ax3.plot(
        x,
        data["offline_inf"],
        color="#d62728",
        lw=2.2,
        label="offline oracle inf",
    )
    ax3.plot(
        x,
        data["online_inf_mean"],
        color="#2ca02c",
        lw=1.8,
        label="online scheduler mean inf",
    )
    ax3.fill_between(
        x,
        data["online_inf_p25"],
        data["online_inf_p75"],
        color="#2ca02c",
        alpha=0.16,
        label="online inf IQR",
    )
    ax3.set_ylabel("inf")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper right", fontsize=9)

    # Panel 4: switch dynamics (red line + blue line)
    ax4.step(
        x,
        data["offline_switch"],
        where="mid",
        color="#d62728",
        lw=1.8,
        label="offline switch flag",
    )
    ax4.plot(
        x,
        data["online_switch_rate"],
        color="#1f77b4",
        lw=2.2,
        label="online switch rate",
    )
    ax4.set_ylabel("switch / rate")
    ax4.set_xlabel("frame index (0-4000)")
    ax4.set_ylim(-0.03, 1.03)
    ax4.grid(alpha=0.25)
    ax4.legend(loc="upper right", fontsize=9)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(0, 4000)

    n_runs = int(data["n_runs"][0])
    fig.suptitle(f"{title} | n_dynamic_runs={n_runs}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _find_default_dynamic_manifest(log_dir: Path) -> Optional[Path]:
    patched = sorted(log_dir.glob("abc24x3_trigpu_shift_taskpool_*_a06rerun_*.tsv"))
    if patched:
        return patched[-1]
    cands = sorted(log_dir.glob("abc24x3_trigpu_shift_taskpool_*.tsv"))
    cands = [p for p in cands if "static_balanced" not in p.name]
    return cands[-1] if cands else None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_log_dir = root / "inference_logs"
    default_dynamic = _find_default_dynamic_manifest(default_log_dir)
    default_chain = (
        root
        / "analysis_out_doh"
        / "continuity_analysis_token_abc24_static_20260316_235206_w50"
        / "chain_fix_ib32_tb16_tau0p6.csv"
    )

    ap = argparse.ArgumentParser(
        description="Triple plot: token drift vs offline oracle vs online scheduler."
    )
    ap.add_argument("--inference-logs-dir", type=Path, default=default_log_dir)
    ap.add_argument("--dynamic-manifest", type=Path, default=default_dynamic)
    ap.add_argument("--static-chain-csv", type=Path, default=default_chain)
    ap.add_argument(
        "--group",
        type=str,
        default="ALL",
        choices=["ALL", "A", "B", "C"],
        help="Filter dynamic runs by workload group",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
    )
    args = ap.parse_args()

    if args.dynamic_manifest is None or not args.dynamic_manifest.exists():
        raise FileNotFoundError(f"missing dynamic manifest: {args.dynamic_manifest}")
    if not args.static_chain_csv.exists():
        raise FileNotFoundError(f"missing static chain csv: {args.static_chain_csv}")

    dynamic_rows = _load_dynamic_manifest(args.dynamic_manifest, args.group)
    if not dynamic_rows:
        raise RuntimeError(f"no dynamic rows for group={args.group}")
    static_chain = _load_static_chain(args.static_chain_csv)
    data = build_plot_data(
        dynamic_rows=dynamic_rows,
        static_chain=static_chain,
        log_dir=args.inference_logs_dir.resolve(),
    )

    tag = args.dynamic_manifest.stem
    out_path = args.out_dir / f"triple_drift_oracle_scheduler_{args.group.lower()}_{tag}.png"
    title = "Token Drift + Offline Oracle + Online Scheduler"
    plot_triple(data=data, out_path=out_path, title=title)
    print(out_path)


if __name__ == "__main__":
    main()
