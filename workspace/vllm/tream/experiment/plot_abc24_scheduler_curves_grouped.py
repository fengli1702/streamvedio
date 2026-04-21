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


CaseRow = Dict[str, str]


def _load_manifest(path: Path) -> Dict[str, CaseRow]:
    out: Dict[str, CaseRow] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                continue
            out[case_id] = row
    return out


def _mean_latency_accuracy(jsonl_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not jsonl_path.exists():
        return None, None
    lat_vals: List[float] = []
    acc_vals: List[float] = []
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
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
            lat = obj.get("latency")
            acc = obj.get("accuracy")
            if isinstance(lat, (int, float)) and math.isfinite(float(lat)):
                lat_vals.append(float(lat))
            if isinstance(acc, (int, float)) and math.isfinite(float(acc)):
                acc_vals.append(float(acc))
    if not lat_vals or not acc_vals:
        return None, None
    return sum(lat_vals) / len(lat_vals), sum(acc_vals) / len(acc_vals)


def _pick_group_optimum(
    static_rows: Dict[str, CaseRow],
    log_dir: Path,
    group_prefix: str,
    acc_threshold: float,
) -> Tuple[str, int, int, float, float]:
    records: List[Tuple[str, int, int, float, float]] = []
    for k in range(1, 9):
        case_id = f"{group_prefix}{k:02d}"
        row = static_rows.get(case_id)
        if not row:
            continue
        run_name = (row.get("run_name") or "").strip()
        if not run_name:
            continue
        lat, acc = _mean_latency_accuracy(log_dir / f"{run_name}.jsonl")
        if lat is None or acc is None:
            continue
        ctx = int(row["ctx"])
        inf = int(row["inf"])
        records.append((case_id, ctx, inf, lat, acc))
    if not records:
        raise RuntimeError(f"no static records for group {group_prefix}")

    good = [r for r in records if r[4] >= acc_threshold]
    if good:
        best = min(good, key=lambda r: (r[3], -r[4], r[1] * r[2]))
    else:
        best = min(records, key=lambda r: (-r[4], r[3], r[1] * r[2]))
    return best


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
    seq: List[Tuple[int, int, int]] = []
    for dense_idx, (_step, (ctx, inf)) in enumerate(sorted(by_step.items())):
        seq.append((dense_idx, ctx, inf))
    return seq


def _load_frame_axis(log_dir: Path, run_name: str) -> List[int]:
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
    if len(frame_axis) == 1:
        return [float(frame_axis[0])] * n_steps
    m = len(frame_axis)
    xs: List[float] = []
    for k in range(n_steps):
        q = k / float(n_steps - 1)
        idx = int(round(q * (m - 1)))
        idx = max(0, min(m - 1, idx))
        xs.append(float(frame_axis[idx]))
    return xs


def _load_static_chain(chain_csv: Path) -> List[Tuple[float, int, int]]:
    if not chain_csv.exists():
        return []
    out: List[Tuple[float, int, int]] = []
    with chain_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row.get("time_center", "nan"))
                ctx = int(float(row["ctx"]))
                inf = int(float(row["inf"]))
            except Exception:
                continue
            if not math.isfinite(t):
                continue
            out.append((t, ctx, inf))
    out.sort(key=lambda x: x[0])
    return out


def _plot_group(
    dynamic_rows: Dict[str, CaseRow],
    static_rows: Dict[str, CaseRow],
    log_dir: Path,
    group_prefix: str,
    group_label: str,
    acc_threshold: float,
    static_chain: List[Tuple[float, int, int]],
    out_path: Path,
) -> Tuple[str, int, int]:
    best_case, best_ctx, best_inf, best_lat, best_acc = _pick_group_optimum(
        static_rows=static_rows,
        log_dir=log_dir,
        group_prefix=group_prefix,
        acc_threshold=acc_threshold,
    )

    fig, (ax_inf, ax_ctx) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
    )
    cases = [f"{group_prefix}{k:02d}" for k in range(1, 9)]
    max_frame = 4000

    for case_id in cases:
        row = dynamic_rows.get(case_id)
        if not row:
            continue
        run_name = (row.get("run_name") or "").strip()
        if not run_name:
            continue
        seq = _load_scheduler_seq(log_dir=log_dir, run_name=run_name)
        if not seq:
            continue
        frame_axis = _load_frame_axis(log_dir=log_dir, run_name=run_name)
        n = len(seq)
        if n <= 0:
            continue
        xs = _step_to_frame_axis(n_steps=n, frame_axis=frame_axis)
        ys_ctx = [t[1] for t in seq[:n]]
        ys_inf = [t[2] for t in seq[:n]]
        is_best_case = case_id == best_case
        color = "#1b3a57" if is_best_case else "#5fa8d3"
        alpha = 0.95 if is_best_case else 0.32
        lw = 2.2 if is_best_case else 1.3
        label = f"{case_id} traj" if is_best_case else None
        ax_inf.plot(xs, ys_inf, color=color, alpha=alpha, lw=lw, label=label)
        ax_ctx.plot(xs, ys_ctx, color=color, alpha=alpha, lw=lw)

    opt_label = "oracle optimal curve (ctx, inf) from static chain"
    if static_chain:
        xs_opt = [t[0] for t in static_chain]
        opt_ctx = [float(t[1]) for t in static_chain]
        opt_inf = [float(t[2]) for t in static_chain]
        ax_inf.plot(xs_opt, opt_inf, color="#d62728", lw=2.8, label=opt_label, zorder=6)
        ax_ctx.plot(xs_opt, opt_ctx, color="#d62728", lw=2.8, label=opt_label, zorder=6)
    else:
        fallback = (
            f"static fallback ({best_case}) = ({best_ctx},{best_inf}) "
            f"(ctx, inf), lat={best_lat:.3f}, acc={best_acc:.3f}"
        )
        ax_inf.axhline(best_inf, color="#d62728", lw=2.8, label=fallback)
        ax_ctx.axhline(best_ctx, color="#d62728", lw=2.8, label=fallback)

    ax_inf.set_ylabel("inference length")
    ax_ctx.set_ylabel("context length")
    ax_ctx.set_xlabel("frame index (0-4000)")
    ax_inf.grid(alpha=0.25)
    ax_ctx.grid(alpha=0.25)
    ax_inf.set_title(f"{group_label} Workload Group: Inference Trajectory over Frames")
    ax_ctx.set_title("Context Trajectory over Frames")
    ax_inf.set_xlim(0, max_frame)
    ax_ctx.set_xlim(0, max_frame)

    handles, labels = ax_inf.get_legend_handles_labels()
    if handles:
        ax_inf.legend(handles, labels, loc="upper right", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return best_case, best_ctx, best_inf


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
    default_static = default_log_dir / "abc24x3_trigpu_shift_taskpool_static_balanced_20260316_235206.tsv"
    default_static_chain = (
        root
        / "analysis_out_doh"
        / "continuity_analysis_token_abc24_static_20260316_235206_w50"
        / "chain_fix_ib32_tb16_tau0p6.csv"
    )
    default_dynamic = _find_default_dynamic_manifest(default_log_dir)

    ap = argparse.ArgumentParser(
        description="Plot grouped ABC24 scheduler curves (small/medium/large)."
    )
    ap.add_argument(
        "--inference-logs-dir",
        type=Path,
        default=default_log_dir,
    )
    ap.add_argument(
        "--dynamic-manifest",
        type=Path,
        default=default_dynamic,
        help="dynamic abc24 manifest tsv (prefer patched manifest with A06 rerun).",
    )
    ap.add_argument(
        "--static-manifest",
        type=Path,
        default=default_static,
    )
    ap.add_argument(
        "--static-chain-csv",
        type=Path,
        default=default_static_chain,
        help="frame-varying oracle path from static continuity analysis (columns include ctx,inf).",
    )
    ap.add_argument("--acc-threshold", type=float, default=0.60)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
    )
    args = ap.parse_args()

    if args.dynamic_manifest is None:
        raise FileNotFoundError("cannot auto-find dynamic manifest")
    if not args.dynamic_manifest.exists():
        raise FileNotFoundError(f"missing dynamic manifest: {args.dynamic_manifest}")
    if not args.static_manifest.exists():
        raise FileNotFoundError(f"missing static manifest: {args.static_manifest}")

    log_dir = args.inference_logs_dir.resolve()
    dynamic_rows = _load_manifest(args.dynamic_manifest.resolve())
    static_rows = _load_manifest(args.static_manifest.resolve())
    static_chain = _load_static_chain(args.static_chain_csv.resolve())
    run_tag = args.dynamic_manifest.stem

    outputs = []
    for prefix, label in [
        ("A", "Small"),
        ("B", "Medium"),
        ("C", "Large"),
    ]:
        out_path = args.out_dir / f"scheduler_curves_{label.lower()}_{run_tag}.png"
        best_case, best_ctx, best_inf = _plot_group(
            dynamic_rows=dynamic_rows,
            static_rows=static_rows,
            log_dir=log_dir,
            group_prefix=prefix,
            group_label=label,
            acc_threshold=args.acc_threshold,
            static_chain=static_chain,
            out_path=out_path,
        )
        outputs.append((label, out_path, best_case, best_ctx, best_inf))

    for label, path, best_case, best_ctx, best_inf in outputs:
        mode = "drifting static-chain curve" if static_chain else "fixed static fallback"
        print(
            f"{label}: {path} | mode={mode} | fallback_optimal={best_case} ({best_ctx},{best_inf}) "
            f"from static(acc>={args.acc_threshold:.2f})"
        )


if __name__ == "__main__":
    main()
