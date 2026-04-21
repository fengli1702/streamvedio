#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


STEP_PATTERNS = (
    "stepA_bs_*.jsonl",
    "stepA_more_*.jsonl",
    "stepA_a*.jsonl",
    "stepA_window_*.jsonl",
)


def _safe_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _parse_run_id(filename: str) -> Dict[str, Optional[str]]:
    base = os.path.basename(filename)
    run_id = base[:-6] if base.endswith(".jsonl") else base
    meta: Dict[str, Optional[str]] = {
        "run_id": run_id,
        "group": None,
        "ctx": None,
        "inf": None,
        "ib": None,
        "tb": None,
        "lr": None,
        "spec": None,
        "sched": None,
        "dataset": None,
    }

    patterns = [
        (
            r"^stepA_bs_ctx(?P<ctx>\d+)_inf(?P<inf>\d+)_bs(?P<ib>\d+)_"
            r"(?P<spec>spec|nospec)_(?P<sched>dyn|static)(?:_data-(?P<dataset>[^_]+))?$",
            "A1",
        ),
        (
            r"^stepA_(?:more_)?a(?P<phase>[234])_ctx(?P<ctx>\d+)_inf(?P<inf>\d+)_"
            r"ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_"
            r"(?P<spec>spec|nospec)_(?P<sched>dyn|static)(?:_data-(?P<dataset>[^_]+))?$",
            None,
        ),
        (
            r"^stepA_window_(?P<phase>win|repl)_ctx(?P<ctx>\d+)_inf(?P<inf>\d+)_"
            r"ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_"
            r"(?P<spec>spec|nospec)_(?P<sched>dyn|static)(?P<rep>_rep\d+)?"
            r"(?:_data-(?P<dataset>[^_]+))?$",
            "window",
        ),
    ]

    for pattern, group in patterns:
        match = re.match(pattern, run_id)
        if not match:
            continue
        meta.update({k: v for k, v in match.groupdict().items() if k in meta})
        if group:
            meta["group"] = group
        else:
            phase = match.groupdict().get("phase")
            if phase == "2":
                meta["group"] = "A2"
            elif phase == "3":
                meta["group"] = "A3"
            elif phase == "4":
                meta["group"] = "A4"
        return meta
    return meta


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.array(values, dtype=float)))


def _pareto_mask(points: List[Tuple[float, float]]) -> List[bool]:
    mask = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_runs(logs_dir: str) -> List[str]:
    files = []
    for pattern in STEP_PATTERNS:
        files.extend(
            [
                os.path.join(logs_dir, name)
                for name in os.listdir(logs_dir)
                if re.fullmatch(pattern.replace("*", ".*"), name)
            ]
        )
    return sorted(set(files))


def _extract_run_name_from_config(config_path: str) -> Optional[str]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None
    for idx, line in enumerate(lines):
        if line.strip() in ("- --wandb_run_name", "- \"--wandb_run_name\""):
            for j in range(idx + 1, min(idx + 4, len(lines))):
                val = lines[j].strip()
                if val.startswith("- "):
                    name = val[2:].strip().strip("\"'")
                    return name or None
    return None


def _find_wandb_summary(run_name: str, wandb_dir: str) -> Optional[Dict[str, object]]:
    if not run_name:
        return None
    for run_dir in os.listdir(wandb_dir):
        if not run_dir.startswith("run-"):
            continue
        cfg = os.path.join(wandb_dir, run_dir, "files", "config.yaml")
        if not os.path.exists(cfg):
            continue
        cfg_run = _extract_run_name_from_config(cfg)
        if cfg_run != run_name:
            continue
        summary_path = os.path.join(wandb_dir, run_dir, "files", "wandb-summary.json")
        if not os.path.exists(summary_path):
            return None
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze StepA logs.")
    parser.add_argument(
        "--logs-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "inference_logs"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "analysis_stepA"),
    )
    parser.add_argument(
        "--wandb-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "wandb"),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--acc-bar", type=float, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    out_dir = os.path.abspath(args.out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    wandb_dir = os.path.abspath(args.wandb_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    runs = _load_runs(logs_dir)
    if not runs:
        raise SystemExit(f"No StepA logs found in {logs_dir}")
    if args.dataset:
        runs = [
            path
            for path in runs
            if _parse_run_id(os.path.basename(path)).get("dataset") == args.dataset
        ]
        if not runs:
            raise SystemExit(f"No StepA logs found for dataset={args.dataset}")

    tag = args.tag or args.dataset
    tag_suffix = f"_{tag}" if tag else ""
    tag_label = f" ({tag})" if tag else ""

    run_meta_rows: List[Dict[str, object]] = []
    cycle_rows: List[Dict[str, object]] = []
    train_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for path in runs:
        meta = _parse_run_id(os.path.basename(path))
        run_meta_rows.append(meta.copy())

        latencies = []
        cycle_times = []
        tps_values = []
        acc_values = []
        acc_valid = 0
        cycle_count = 0
        last_global_acc = None
        train_latencies = []
        train_tps_values = []
        train_cycle_count = 0
        train_time_est_values = []
        weight_update_est_values = []
        last_train_step_end = None
        wandb_summary = _find_wandb_summary(meta["run_id"], wandb_dir)
        wandb_acc = None
        if wandb_summary:
            wandb_acc = wandb_summary.get("inference/global_accuracy")

        last_stats = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                event = obj.get("event", "inference_cycle")
                if event == "final_stats":
                    last_stats = obj
                if event == "training_cycle":
                    train_cycle_count += 1
                    train_lat = obj.get("avg_step_latency")
                    train_tps = obj.get("train_tokens_per_second")
                    train_step_end = obj.get("train_step_end")
                    weight_share_freq = obj.get("weight_sharing_freq")
                    if train_lat is not None:
                        train_latencies.append(float(train_lat))
                    if train_tps is not None:
                        train_tps_values.append(float(train_tps))
                    train_step_delta = None
                    train_time_est = None
                    if train_step_end is not None:
                        try:
                            step_end = int(train_step_end)
                        except Exception:
                            step_end = None
                        if step_end is not None and last_train_step_end is not None:
                            delta = step_end - last_train_step_end
                            if delta > 0 and train_lat is not None:
                                train_step_delta = delta
                                train_time_est = float(train_lat) * delta
                                train_time_est_values.append(train_time_est)
                        if step_end is not None:
                            last_train_step_end = step_end
                    if weight_share_freq is not None and train_lat is not None:
                        try:
                            wsf = int(weight_share_freq)
                            weight_update_est = float(train_lat) * wsf
                            weight_update_est_values.append(weight_update_est)
                        except Exception:
                            pass
                    train_rows.append(
                        {
                            "run_id": meta["run_id"],
                            "group": meta["group"],
                            "ctx": meta["ctx"],
                            "inf": meta["inf"],
                            "ib": meta["ib"],
                            "tb": meta["tb"],
                            "lr": meta["lr"],
                            "spec": meta["spec"],
                            "sched": meta["sched"],
                            "dataset": meta["dataset"],
                            "training_cycle_id": obj.get("training_cycle_id"),
                            "train_step_end": train_step_end,
                            "avg_step_latency": train_lat,
                            "train_tokens_per_second": train_tps,
                            "train_loss": obj.get("train_loss"),
                            "training_batch_size": obj.get("training_batch_size"),
                            "learning_rate": obj.get("learning_rate"),
                            "frame_index": obj.get("frame_index"),
                            "current_directory": obj.get("current_directory"),
                            "train_step_delta": train_step_delta,
                            "train_time_est": train_time_est,
                        }
                    )
                    continue
                if event != "inference_cycle":
                    continue

                cycle_count += 1
                latency = obj.get("latency")
                cycle_time = obj.get("t_inference_cycle")
                tps = obj.get("tokens_per_second")
                acc = obj.get("accuracy")
                global_acc = obj.get("global_accuracy")
                if global_acc is not None:
                    last_global_acc = float(global_acc)
                if latency is not None:
                    latencies.append(float(latency))
                if cycle_time is not None:
                    cycle_times.append(float(cycle_time))
                if tps is not None:
                    tps_values.append(float(tps))
                if acc is not None:
                    acc_values.append(float(acc))
                    acc_valid += 1

                cycle_rows.append(
                    {
                        "run_id": meta["run_id"],
                        "group": meta["group"],
                        "ctx": meta["ctx"],
                        "inf": meta["inf"],
                        "ib": meta["ib"],
                        "tb": meta["tb"],
                        "lr": meta["lr"],
                        "spec": meta["spec"],
                        "sched": meta["sched"],
                        "dataset": meta["dataset"],
                        "frame_index": obj.get("frame_index"),
                        "current_directory": obj.get("current_directory"),
                        "latency": latency,
                        "t_inference_cycle": cycle_time,
                        "tokens_per_second": tps,
                        "accuracy": acc,
                        "global_accuracy": global_acc,
                        "perplexity": obj.get("perplexity"),
                    }
                )

        acc_valid_ratio = acc_valid / cycle_count if cycle_count else 0.0
        lat_mean = float(np.mean(np.array(latencies, dtype=float))) if latencies else None
        cycle_time_mean = float(np.mean(np.array(cycle_times, dtype=float))) if cycle_times else None
        cycle_time_p95 = _percentile(cycle_times, 95)
        train_lat_mean = float(np.mean(np.array(train_latencies, dtype=float))) if train_latencies else None
        train_lat_p95 = _percentile(train_latencies, 95)
        train_tps_median = _median(train_tps_values)
        train_time_total_est = float(np.sum(np.array(train_time_est_values, dtype=float))) if train_time_est_values else None
        weight_update_interval_mean = (
            float(np.mean(np.array(weight_update_est_values, dtype=float))) if weight_update_est_values else None
        )
        weight_update_interval_p95 = _percentile(weight_update_est_values, 95)
        acc_cycle_mean = float(np.mean(np.array(acc_values, dtype=float))) if acc_values else None
        acc_cycle_median = _median(acc_values)
        acc_global = wandb_acc if wandb_acc is not None else last_global_acc
        acc_source = "wandb_summary"
        if acc_global is None:
            acc_global = acc_cycle_mean
            acc_source = "cycle_mean"

        summary_rows.append(
            {
                "run_id": meta["run_id"],
                "group": meta["group"],
                "ctx": meta["ctx"],
                "inf": meta["inf"],
                "ib": meta["ib"],
                "tb": meta["tb"],
                "lr": meta["lr"],
                "spec": meta["spec"],
                "sched": meta["sched"],
                "dataset": meta["dataset"],
                "cycle_count": cycle_count,
                "acc_valid_count": acc_valid,
                "lat_p50": _percentile(latencies, 50),
                "lat_p95": _percentile(latencies, 95),
                "lat_mean": lat_mean,
                "cycle_time_mean": cycle_time_mean,
                "cycle_time_p95": cycle_time_p95,
                "tps_median": _median(tps_values),
                "acc_global": acc_global,
                "acc_source": acc_source,
                "acc_cycle_mean": acc_cycle_mean,
                "acc_cycle_median": acc_cycle_median,
                "acc_valid_ratio": acc_valid_ratio,
                "frames_processed": last_stats.get("frames_processed"),
                "wandb_global_accuracy": wandb_acc,
                "train_lat_mean": train_lat_mean,
                "train_lat_p95": train_lat_p95,
                "train_tps_median": train_tps_median,
                "train_cycle_count": train_cycle_count,
                "train_time_total_est": train_time_total_est,
                "weight_update_interval_mean": weight_update_interval_mean,
                "weight_update_interval_p95": weight_update_interval_p95,
            }
        )

    _write_csv(os.path.join(out_dir, "run_meta.csv"), run_meta_rows)
    _write_csv(os.path.join(out_dir, "cycle_metrics.csv"), cycle_rows)
    _write_csv(os.path.join(out_dir, "train_cycle_metrics.csv"), train_rows)
    _write_csv(os.path.join(out_dir, "run_summary.csv"), summary_rows)

    if args.skip_plots or plt is None:
        return

    # Convert summary to indexable dicts
    summary = summary_rows
    min_cycles = 50

    def to_float(v):
        return float(v) if v is not None else None

    # Figure 1: Pareto overview for ctx4/inf4 A1-A4 (latency vs accuracy)
    fig1 = plt.figure(figsize=(7, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    markers = {"A1": "o", "A2": "s", "A3": "D", "A4": "^"}
    points = []
    labels = []
    for row in summary:
        if row["group"] not in ("A1", "A2", "A3", "A4"):
            continue
        if str(row.get("ctx")) != "4" or str(row.get("inf")) != "4":
            continue
        x = to_float(row.get("lat_p95"))
        y = to_float(row.get("acc_global"))
        c = to_float(row.get("tps_median"))
        if x is None or y is None:
            continue
        points.append((x, -y))
        labels.append(row)
        ax1.scatter(x, y, c=c, cmap="viridis", marker=markers.get(row["group"], "o"), edgecolors="black")
        label = row.get("ib") or row.get("tb") or row.get("lr")
        if label:
            ax1.text(x, y, str(label), fontsize=7)

    if points:
        pareto = _pareto_mask(points)
        pareto_pts = []
        for is_front, row in zip(pareto, labels):
            if not is_front:
                continue
            x = to_float(row.get("lat_p95"))
            y = to_float(row.get("acc_global"))
            if x is None or y is None:
                continue
            ax1.scatter(x, y, facecolors="none", edgecolors="red", s=120, linewidths=1.5)
            pareto_pts.append((x, y))
        if pareto_pts:
            pareto_pts.sort(key=lambda v: v[0])
            ax1.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts], color="red", linewidth=1.5)

    ax1.set_xlabel("lat_p95 (lower is better)")
    ax1.set_ylabel("acc_global (higher is better)")
    ax1.set_title(f"Pareto overview (ctx4/inf4, A1-A4){tag_label}")
    fig1.tight_layout()
    fig1.savefig(os.path.join(plots_dir, f"fig1_pareto_overview{tag_suffix}.png"), dpi=200)
    plt.close(fig1)

    # Figure 2: A1 ib curve (config on x, latency + accuracy)
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    a1_rows = [r for r in summary if r["group"] == "A1" and str(r.get("ctx")) == "4" and str(r.get("inf")) == "4"]
    a1_rows.sort(key=lambda r: int(r["ib"]) if r.get("ib") else 0)
    xs = [int(r.get("ib")) for r in a1_rows if r.get("ib")]
    lat = [to_float(r.get("lat_p95")) for r in a1_rows]
    acc = [to_float(r.get("acc_global")) for r in a1_rows]
    lat_line = ax2.plot(xs, lat, marker="o", label="lat_p95")[0]
    ax2.set_xlabel("ib (config)")
    ax2.set_ylabel("lat_p95")
    ax2b = ax2.twinx()
    acc_line = ax2b.plot(xs, acc, marker="s", color="tab:orange", label="acc_global")[0]
    ax2b.set_ylabel("acc_global")
    ax2.legend(handles=[lat_line, acc_line], loc="best")
    ax2.set_title(f"A1: ib vs latency/accuracy (ib=inference_batch_size){tag_label}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, f"fig2_a1_ib_curve{tag_suffix}.png"), dpi=200)
    plt.close(fig2)

    # Figure 3: A2+A3 (latency + accuracy)
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
    a2_rows = [
        r
        for r in summary
        if r["group"] == "A2" and (r.get("cycle_count") or 0) >= min_cycles
    ]
    for r in a2_rows:
        x = to_float(r.get("lat_p95"))
        y = to_float(r.get("acc_global"))
        c = to_float(r.get("tb"))
        if x is None or y is None or c is None:
            continue
        axes[0].scatter(x, y, c=c, cmap="plasma", edgecolors="black")
        axes[0].text(x, y, f"tb{int(c)}", fontsize=7)
    axes[0].set_xlabel("lat_p95")
    axes[0].set_ylabel("acc_global")
    axes[0].set_title(f"A2: training_bs effect{tag_label}")

    a3_rows = [r for r in summary if r["group"] == "A3"]
    for r in a3_rows:
        lr = _safe_float(r.get("lr"))
        acc = to_float(r.get("acc_global"))
        lat = to_float(r.get("lat_p95"))
        if lr is None or acc is None:
            continue
        axes[1].scatter(lr, acc, c=lat, cmap="magma", edgecolors="black")
        axes[1].text(lr, acc, f"{r.get('lr')}", fontsize=8)
    axes[1].set_xlabel("learning_rate")
    axes[1].set_ylabel("acc_global")
    axes[1].set_title(f"A3: lr vs accuracy (color=lat_p95){tag_label}")
    fig3.tight_layout()
    fig3.savefig(os.path.join(plots_dir, f"fig3_a2_a3{tag_suffix}.png"), dpi=200)
    plt.close(fig3)

    # Figure 4: A4 heatmaps (latency + accuracy)
    a4_rows = [r for r in summary if r["group"] == "A4"]
    ib_vals = sorted({int(r["ib"]) for r in a4_rows if r.get("ib")})
    tb_vals = sorted({int(r["tb"]) for r in a4_rows if r.get("tb")})
    lat_grid = np.full((len(tb_vals), len(ib_vals)), np.nan)
    acc_grid = np.full((len(tb_vals), len(ib_vals)), np.nan)
    for r in a4_rows:
        ib = int(r["ib"])
        tb = int(r["tb"])
        i = tb_vals.index(tb)
        j = ib_vals.index(ib)
        lat_grid[i, j] = to_float(r.get("lat_p95"))
        acc_grid[i, j] = to_float(r.get("acc_global"))

    fig4, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(lat_grid, cmap="magma", aspect="auto")
    axes[0].set_xticks(range(len(ib_vals)), ib_vals)
    axes[0].set_yticks(range(len(tb_vals)), tb_vals)
    axes[0].set_xlabel("ib")
    axes[0].set_ylabel("tb")
    axes[0].set_title(f"A4 lat_p95 heatmap{tag_label}")
    fig4.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(acc_grid, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(ib_vals)), ib_vals)
    axes[1].set_yticks(range(len(tb_vals)), tb_vals)
    axes[1].set_xlabel("ib")
    axes[1].set_ylabel("tb")
    axes[1].set_title(f"A4 acc_global heatmap{tag_label}")
    fig4.colorbar(im1, ax=axes[1])

    fig4.tight_layout()
    fig4.savefig(os.path.join(plots_dir, f"fig4_a4_heatmaps{tag_suffix}.png"), dpi=200)
    plt.close(fig4)

    # Figure 5: StepA-window 3x3 grid (latency vs accuracy)
    window_rows = [r for r in summary if r["group"] == "window"]
    combos = [(1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2), (8, 4), (8, 8)]
    fig5, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig5.suptitle(f"StepA-window (ib=inference_batch_size){tag_label}")
    for idx, (ctx, inf) in enumerate(combos):
        ax = axes[idx // 3][idx % 3]
        rows = [r for r in window_rows if str(r.get("ctx")) == str(ctx) and str(r.get("inf")) == str(inf)]
        rows.sort(key=lambda r: int(r["ib"]) if r.get("ib") else 0)
        xs = [to_float(r.get("lat_p95")) for r in rows]
        ys = [to_float(r.get("acc_global")) for r in rows]
        ax.plot(xs, ys, marker="o")
        ax.set_title(f"ctx{ctx}/inf{inf}")
        ax.set_xlabel("lat_p95")
        ax.set_ylabel("acc_global")
        for r, x, y in zip(rows, xs, ys):
            if x is None or y is None:
                continue
            ax.text(x, y, f"ib{r.get('ib')}", fontsize=7)
        if rows:
            pareto = _pareto_mask([(x, -y) for x, y in zip(xs, ys) if x is not None and y is not None])
            pareto_pts = []
            for is_front, r, x, y in zip(pareto, rows, xs, ys):
                if not is_front or x is None or y is None:
                    continue
                ax.scatter(x, y, facecolors="none", edgecolors="red", s=80, linewidths=1.2)
                pareto_pts.append((x, y))
            if pareto_pts:
                pareto_pts.sort(key=lambda v: v[0])
                ax.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts], color="red", linewidth=1.0)
    fig5.tight_layout()
    fig5.savefig(os.path.join(plots_dir, f"fig5_window_grid{tag_suffix}.png"), dpi=200)
    plt.close(fig5)

    # Figure 7: Pareto overview in latency-tps plane (ctx4/inf4)
    fig7 = plt.figure(figsize=(7, 5))
    ax7 = fig7.add_subplot(1, 1, 1)
    points = []
    labels = []
    for row in summary:
        if row["group"] not in ("A1", "A2", "A3", "A4"):
            continue
        if str(row.get("ctx")) != "4" or str(row.get("inf")) != "4":
            continue
        x = to_float(row.get("lat_p95"))
        y = to_float(row.get("tps_median"))
        c = to_float(row.get("acc_global"))
        if x is None or y is None:
            continue
        points.append((x, -y))
        labels.append(row)
        ax7.scatter(x, y, c=c, cmap="viridis", marker=markers.get(row["group"], "o"), edgecolors="black")
    if points:
        pareto = _pareto_mask(points)
        pareto_pts = []
        for is_front, row in zip(pareto, labels):
            if not is_front:
                continue
            x = to_float(row.get("lat_p95"))
            y = to_float(row.get("tps_median"))
            if x is None or y is None:
                continue
            ax7.scatter(x, y, facecolors="none", edgecolors="red", s=120, linewidths=1.5)
            pareto_pts.append((x, y))
        if pareto_pts:
            pareto_pts.sort(key=lambda v: v[0])
            ax7.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts], color="red", linewidth=1.5)
    ax7.set_xlabel("lat_p95 (lower is better)")
    ax7.set_ylabel("tps_median (higher is better)")
    ax7.set_title(f"Latency–TPS Pareto (ctx4/inf4, color=acc){tag_label}")
    fig7.tight_layout()
    fig7.savefig(os.path.join(plots_dir, f"fig7_latency_tps_pareto{tag_suffix}.png"), dpi=200)
    plt.close(fig7)

    # Figure 8: StepA-window grid (latency vs tps, color=acc)
    fig8, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig8.suptitle(f"StepA-window latency–TPS (color=acc, ib=inference_batch_size){tag_label}")
    for idx, (ctx, inf) in enumerate(combos):
        ax = axes[idx // 3][idx % 3]
        rows = [r for r in window_rows if str(r.get("ctx")) == str(ctx) and str(r.get("inf")) == str(inf)]
        rows.sort(key=lambda r: int(r["ib"]) if r.get("ib") else 0)
        xs = [to_float(r.get("lat_p95")) for r in rows]
        ys = [to_float(r.get("tps_median")) for r in rows]
        cs = [to_float(r.get("acc_global")) for r in rows]
        ax.scatter(xs, ys, c=cs, cmap="viridis", edgecolors="black")
        if xs and ys:
            ax.plot(xs, ys, color="gray", linewidth=1.0)
        ax.set_title(f"ctx{ctx}/inf{inf}")
        ax.set_xlabel("lat_p95")
        ax.set_ylabel("tps_median")
        for r, x, y in zip(rows, xs, ys):
            if x is None or y is None:
                continue
            ax.text(x, y, f"ib{r.get('ib')}", fontsize=7)
    fig8.tight_layout()
    fig8.savefig(os.path.join(plots_dir, f"fig8_window_latency_tps{tag_suffix}.png"), dpi=200)
    plt.close(fig8)

    # Figure 6: window best feasible (latency + accuracy)
    fig6 = plt.figure(figsize=(8, 4))
    ax6 = fig6.add_subplot(1, 1, 1)
    best_rows = []
    for ctx, inf in combos:
        rows = [r for r in window_rows if str(r.get("ctx")) == str(ctx) and str(r.get("inf")) == str(inf)]
        if args.acc_bar is not None:
            rows = [r for r in rows if r.get("acc_global") is not None and r["acc_global"] >= args.acc_bar]
        if not rows:
            continue
        rows = [r for r in rows if r.get("lat_p95") is not None and r.get("tps_median") is not None]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["lat_p95"])
        best_rows.append(best)

    for i, row in enumerate(best_rows):
        x = i
        y = row.get("lat_p95")
        size = (row.get("tps_median") or 0) / 10 + 20
        color = row.get("acc_global")
        label = f"{row.get('ctx')}/{row.get('inf')}"
        ax6.scatter(x, y, s=size, c=color, cmap="viridis", edgecolors="black")
        ax6.text(x, y, label, fontsize=8, ha="center", va="bottom")

    ax6.set_xticks([])
    ax6.set_ylabel("best_feasible_lat_p95")
    ax6.set_title(f"Window best feasible latency (color=acc_global){tag_label}")
    fig6.tight_layout()
    fig6.savefig(os.path.join(plots_dir, f"fig6_window_best{tag_suffix}.png"), dpi=200)
    plt.close(fig6)

    # Figure 9: A1 latency vs throughput / accuracy (mean latency)
    a1_rows = [r for r in summary if r["group"] == "A1" and str(r.get("ctx")) == "4" and str(r.get("inf")) == "4"]
    a1_rows.sort(key=lambda r: int(r["ib"]) if r.get("ib") else 0)
    fig9, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [to_float(r.get("lat_mean")) for r in a1_rows]
    ys_tps = [to_float(r.get("tps_median")) for r in a1_rows]
    ys_acc = [to_float(r.get("acc_global")) for r in a1_rows]
    axes[0].plot(xs, ys_tps, marker="o")
    axes[1].plot(xs, ys_acc, marker="o")
    for r, x, y in zip(a1_rows, xs, ys_tps):
        if x is None or y is None:
            continue
        axes[0].text(x, y, f"ib={r.get('ib')}", fontsize=8)
    for r, x, y in zip(a1_rows, xs, ys_acc):
        if x is None or y is None:
            continue
        axes[1].text(x, y, f"ib={r.get('ib')}", fontsize=8)
    axes[0].set_title(f"A1: latency vs throughput (ib sweep){tag_label}")
    axes[0].set_xlabel("mean latency_amortized (s) ↓")
    axes[0].set_ylabel("median tokens_per_second ↑")
    axes[1].set_title(f"A1: latency vs accuracy (ib sweep){tag_label}")
    axes[1].set_xlabel("mean latency_amortized (s) ↓")
    axes[1].set_ylabel("acc_global ↑")
    fig9.tight_layout()
    fig9.savefig(os.path.join(plots_dir, f"fig9_a1_latency_mean{tag_suffix}.png"), dpi=200)
    plt.close(fig9)

    # Figure 10: A2 latency vs throughput / accuracy (mean latency)
    a2_rows = [
        r
        for r in summary
        if r["group"] == "A2" and (r.get("cycle_count") or 0) >= min_cycles
    ]
    a2_rows.sort(key=lambda r: int(r["tb"]) if r.get("tb") else 0)
    fig10, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [to_float(r.get("lat_mean")) for r in a2_rows]
    ys_tps = [to_float(r.get("tps_median")) for r in a2_rows]
    ys_acc = [to_float(r.get("acc_global")) for r in a2_rows]
    axes[0].plot(xs, ys_tps, marker="o")
    axes[1].plot(xs, ys_acc, marker="o")
    for r, x, y in zip(a2_rows, xs, ys_tps):
        if x is None or y is None:
            continue
        axes[0].text(x, y, f"tb={r.get('tb')}", fontsize=8)
    for r, x, y in zip(a2_rows, xs, ys_acc):
        if x is None or y is None:
            continue
        axes[1].text(x, y, f"tb={r.get('tb')}", fontsize=8)
    axes[0].set_title(f"A2: latency vs throughput (tb sweep){tag_label}")
    axes[0].set_xlabel("mean latency_amortized (s) ↓")
    axes[0].set_ylabel("median tokens_per_second ↑")
    axes[1].set_title(f"A2: latency vs accuracy (tb sweep){tag_label}")
    axes[1].set_xlabel("mean latency_amortized (s) ↓")
    axes[1].set_ylabel("acc_global ↑")
    fig10.tight_layout()
    fig10.savefig(os.path.join(plots_dir, f"fig10_a2_latency_mean{tag_suffix}.png"), dpi=200)
    plt.close(fig10)

    # Figure 11: A3 latency vs throughput / accuracy (mean latency)
    a3_rows = [r for r in summary if r["group"] == "A3"]
    a3_rows.sort(key=lambda r: str(r["lr"]))
    fig11, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [to_float(r.get("lat_mean")) for r in a3_rows]
    ys_tps = [to_float(r.get("tps_median")) for r in a3_rows]
    ys_acc = [to_float(r.get("acc_global")) for r in a3_rows]
    axes[0].plot(xs, ys_tps, marker="o")
    axes[1].plot(xs, ys_acc, marker="o")
    for r, x, y in zip(a3_rows, xs, ys_tps):
        if x is None or y is None:
            continue
        axes[0].text(x, y, f"lr={r.get('lr')}", fontsize=8)
    for r, x, y in zip(a3_rows, xs, ys_acc):
        if x is None or y is None:
            continue
        axes[1].text(x, y, f"lr={r.get('lr')}", fontsize=8)
    axes[0].set_title(f"A3: latency vs throughput (lr sweep){tag_label}")
    axes[0].set_xlabel("mean latency_amortized (s) ↓")
    axes[0].set_ylabel("median tokens_per_second ↑")
    axes[1].set_title(f"A3: latency vs accuracy (lr sweep){tag_label}")
    axes[1].set_xlabel("mean latency_amortized (s) ↓")
    axes[1].set_ylabel("acc_global ↑")
    fig11.tight_layout()
    fig11.savefig(os.path.join(plots_dir, f"fig11_a3_latency_mean{tag_suffix}.png"), dpi=200)
    plt.close(fig11)

    # Figure 12: A2 training latency vs batch size
    a2_rows = [
        r
        for r in summary
        if r["group"] == "A2" and (r.get("cycle_count") or 0) >= min_cycles
    ]
    a2_rows.sort(key=lambda r: int(r["tb"]) if r.get("tb") else 0)
    fig12 = plt.figure(figsize=(6, 4))
    ax12 = fig12.add_subplot(1, 1, 1)
    xs = [int(r.get("tb")) for r in a2_rows if r.get("tb")]
    ys = [to_float(r.get("train_lat_mean")) for r in a2_rows]
    ax12.plot(xs, ys, marker="o")
    for r, x, y in zip(a2_rows, xs, ys):
        if y is None:
            continue
        ax12.text(x, y, f"tb={r.get('tb')}", fontsize=8)
    ax12.set_xlabel("training_batch_size (tb)")
    ax12.set_ylabel("train avg_step_latency mean (s)")
    ax12.set_title(f"A2: training latency vs tb{tag_label}")
    fig12.tight_layout()
    fig12.savefig(os.path.join(plots_dir, f"fig12_a2_train_latency{tag_suffix}.png"), dpi=200)
    plt.close(fig12)

    # Figure 13: A2 training total time vs tb
    fig13 = plt.figure(figsize=(6, 4))
    ax13 = fig13.add_subplot(1, 1, 1)
    xs = [int(r.get("tb")) for r in a2_rows if r.get("tb")]
    ys = [to_float(r.get("train_time_total_est")) for r in a2_rows]
    ax13.plot(xs, ys, marker="o")
    for r, x, y in zip(a2_rows, xs, ys):
        if y is None:
            continue
        ax13.text(x, y, f"tb={r.get('tb')}", fontsize=8)
    ax13.set_xlabel("training_batch_size (tb)")
    ax13.set_ylabel("train_time_total_est (s)")
    ax13.set_title(f"A2: training total time vs tb{tag_label}")
    fig13.tight_layout()
    fig13.savefig(os.path.join(plots_dir, f"fig13_a2_train_total_time{tag_suffix}.png"), dpi=200)
    plt.close(fig13)

    # Figure 14: A2 weight update interval vs tb
    fig14 = plt.figure(figsize=(6, 4))
    ax14 = fig14.add_subplot(1, 1, 1)
    ys = [to_float(r.get("weight_update_interval_mean")) for r in a2_rows]
    ax14.plot(xs, ys, marker="o")
    for r, x, y in zip(a2_rows, xs, ys):
        if y is None:
            continue
        ax14.text(x, y, f"tb={r.get('tb')}", fontsize=8)
    ax14.set_xlabel("training_batch_size (tb)")
    ax14.set_ylabel("weight_update_interval_mean (s)")
    ax14.set_title(f"A2: weight update interval vs tb{tag_label}")
    fig14.tight_layout()
    fig14.savefig(
        os.path.join(plots_dir, f"fig14_a2_weight_update_interval{tag_suffix}.png"),
        dpi=200,
    )
    plt.close(fig14)

    # Extra plots: cycle time (t_inference_cycle) versions to match prior figures.
    a1_rows = [r for r in summary if r["group"] == "A1" and str(r.get("ctx")) == "4" and str(r.get("inf")) == "4"]
    a1_rows.sort(key=lambda r: int(r["ib"]) if r.get("ib") else 0)
    figA1, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [to_float(r.get("cycle_time_mean")) for r in a1_rows]
    ys_tps = [to_float(r.get("tps_median")) for r in a1_rows]
    ys_acc = [to_float(r.get("acc_global")) for r in a1_rows]
    axes[0].plot(xs, ys_tps, marker="o")
    axes[1].plot(xs, ys_acc, marker="o")
    for r, x, y in zip(a1_rows, xs, ys_tps):
        if x is None or y is None:
            continue
        axes[0].text(x, y, f"ib={r.get('ib')}", fontsize=8)
    for r, x, y in zip(a1_rows, xs, ys_acc):
        if x is None or y is None:
            continue
        axes[1].text(x, y, f"ib={r.get('ib')}", fontsize=8)
    axes[0].set_title(f"A1 (ctx=4, inf=4) ib sweep: cycle time vs throughput{tag_label}")
    axes[0].set_xlabel("mean inference cycle time (s) ↑")
    axes[0].set_ylabel("median tokens_per_second ↑")
    axes[1].set_title(f"A1 (ctx=4, inf=4) ib sweep: cycle time vs accuracy{tag_label}")
    axes[1].set_xlabel("mean inference cycle time (s) ↑")
    axes[1].set_ylabel("global_accuracy ↑")
    figA1.tight_layout()
    figA1.savefig(os.path.join(plots_dir, f"figA1_cycle_time{tag_suffix}.png"), dpi=200)
    plt.close(figA1)

    a2_rows = [
        r
        for r in summary
        if r["group"] == "A2" and (r.get("cycle_count") or 0) >= min_cycles
    ]
    a2_rows.sort(key=lambda r: int(r["tb"]) if r.get("tb") else 0)
    figA2, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [to_float(r.get("cycle_time_mean")) for r in a2_rows]
    ys_tps = [to_float(r.get("tps_median")) for r in a2_rows]
    ys_acc = [to_float(r.get("acc_global")) for r in a2_rows]
    axes[0].plot(xs, ys_tps, marker="o")
    axes[1].plot(xs, ys_acc, marker="o")
    for r, x, y in zip(a2_rows, xs, ys_tps):
        if x is None or y is None:
            continue
        axes[0].text(x, y, f"tb={r.get('tb')}", fontsize=8)
    for r, x, y in zip(a2_rows, xs, ys_acc):
        if x is None or y is None:
            continue
        axes[1].text(x, y, f"tb={r.get('tb')}", fontsize=8)
    axes[0].set_title(f"A2 tb sweep: cycle time vs throughput{tag_label}")
    axes[0].set_xlabel("mean inference cycle time (s) ↑")
    axes[0].set_ylabel("median tokens_per_second ↑")
    axes[1].set_title(f"A2 tb sweep: cycle time vs accuracy{tag_label}")
    axes[1].set_xlabel("mean inference cycle time (s) ↑")
    axes[1].set_ylabel("global_accuracy ↑")
    figA2.tight_layout()
    figA2.savefig(os.path.join(plots_dir, f"figA2_cycle_time{tag_suffix}.png"), dpi=200)
    plt.close(figA2)

    figA2_train = plt.figure(figsize=(6, 4))
    axA2_train = figA2_train.add_subplot(1, 1, 1)
    xs = [int(r.get("tb")) for r in a2_rows if r.get("tb")]
    ys = [to_float(r.get("train_time_total_est")) for r in a2_rows]
    axA2_train.plot(xs, ys, marker="o")
    for r, x, y in zip(a2_rows, xs, ys):
        if y is None:
            continue
        axA2_train.text(x, y, f"tb={r.get('tb')}", fontsize=8)
    axA2_train.set_xlabel("training_batch_size (tb)")
    axA2_train.set_ylabel("train_time_total_est (s)")
    axA2_train.set_title(f"A2: training total time vs tb{tag_label}")
    figA2_train.tight_layout()
    figA2_train.savefig(
        os.path.join(plots_dir, f"figA2_train_total_time{tag_suffix}.png"), dpi=200
    )
    plt.close(figA2_train)

    figA2_update = plt.figure(figsize=(6, 4))
    axA2_update = figA2_update.add_subplot(1, 1, 1)
    ys = [to_float(r.get("weight_update_interval_mean")) for r in a2_rows]
    axA2_update.plot(xs, ys, marker="o")
    for r, x, y in zip(a2_rows, xs, ys):
        if y is None:
            continue
        axA2_update.text(x, y, f"tb={r.get('tb')}", fontsize=8)
    axA2_update.set_xlabel("training_batch_size (tb)")
    axA2_update.set_ylabel("weight_update_interval_mean (s)")
    axA2_update.set_title(f"A2: weight update interval vs tb{tag_label}")
    figA2_update.tight_layout()
    figA2_update.savefig(
        os.path.join(plots_dir, f"figA2_weight_update_interval{tag_suffix}.png"), dpi=200
    )
    plt.close(figA2_update)

    a3_rows = [r for r in summary if r["group"] == "A3"]
    a3_rows.sort(key=lambda r: str(r["lr"]))
    figA3, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [to_float(r.get("cycle_time_mean")) for r in a3_rows]
    ys_tps = [to_float(r.get("tps_median")) for r in a3_rows]
    ys_acc = [to_float(r.get("acc_global")) for r in a3_rows]
    axes[0].plot(xs, ys_tps, marker="o")
    axes[1].plot(xs, ys_acc, marker="o")
    for r, x, y in zip(a3_rows, xs, ys_tps):
        if x is None or y is None:
            continue
        axes[0].text(x, y, f"lr={r.get('lr')}", fontsize=8)
    for r, x, y in zip(a3_rows, xs, ys_acc):
        if x is None or y is None:
            continue
        axes[1].text(x, y, f"lr={r.get('lr')}", fontsize=8)
    axes[0].set_title(f"A3 lr sweep: cycle time vs throughput{tag_label}")
    axes[0].set_xlabel("mean inference cycle time (s) ↑")
    axes[0].set_ylabel("median tokens_per_second ↑")
    axes[1].set_title(f"A3 lr sweep: cycle time vs accuracy{tag_label}")
    axes[1].set_xlabel("mean inference cycle time (s) ↑")
    axes[1].set_ylabel("global_accuracy ↑")
    figA3.tight_layout()
    figA3.savefig(os.path.join(plots_dir, f"figA3_cycle_time{tag_suffix}.png"), dpi=200)
    plt.close(figA3)

    # A4 interaction plot: ib/tb slices on latency-accuracy plane.
    a4_rows = [r for r in summary if r["group"] == "A4"]
    if a4_rows:
        ib_vals = sorted({int(r["ib"]) for r in a4_rows if r.get("ib")})
        tb_vals = sorted({int(r["tb"]) for r in a4_rows if r.get("tb")})
        grid = {}
        for r in a4_rows:
            ib = int(r["ib"])
            tb = int(r["tb"])
            lat = to_float(r.get("lat_mean"))
            acc = to_float(r.get("acc_global"))
            if lat is None or acc is None:
                continue
            grid[(ib, tb)] = (lat, acc)

        figA4 = plt.figure(figsize=(8, 6))
        axA4 = figA4.add_subplot(1, 1, 1)
        for tb in tb_vals:
            pts = [(ib, grid[(ib, tb)]) for ib in ib_vals if (ib, tb) in grid]
            if not pts:
                continue
            pts.sort(key=lambda v: v[0])
            xs = [p[1][0] for p in pts]
            ys = [p[1][1] for p in pts]
            axA4.plot(xs, ys, marker="o", label=f"tb={tb}")
            for ib, (x, y) in pts:
                axA4.text(x, y, f"ib{ib},tb{tb}", fontsize=7)

        for ib in ib_vals:
            pts = [(tb, grid[(ib, tb)]) for tb in tb_vals if (ib, tb) in grid]
            if len(pts) < 2:
                continue
            pts.sort(key=lambda v: v[0])
            xs = [p[1][0] for p in pts]
            ys = [p[1][1] for p in pts]
            axA4.plot(xs, ys, linestyle="--", color="gray", alpha=0.6)

        pareto_points = []
        coords = []
        for (ib, tb), (lat, acc) in grid.items():
            coords.append((lat, acc, ib, tb))
        mask = _pareto_mask([(lat, -acc) for lat, acc, _, _ in coords])
        for is_front, (lat, acc, ib, tb) in zip(mask, coords):
            if not is_front:
                continue
            pareto_points.append((lat, acc))
            axA4.scatter(lat, acc, facecolors="none", edgecolors="black", s=120, linewidths=2)
        if pareto_points:
            pareto_points.sort(key=lambda v: v[0])
            axA4.plot(
                [p[0] for p in pareto_points],
                [p[1] for p in pareto_points],
                color="magenta",
                linewidth=3,
            )

        axA4.set_xlabel("mean latency_amortized (s) ↓")
        axA4.set_ylabel("global_accuracy ↑")
        axA4.set_title(
            f"Q2 (A4 grid): ib/tb interaction on latency–accuracy{tag_label}"
        )
        axA4.legend(loc="best", fontsize=8)
        figA4.tight_layout()
        figA4.savefig(os.path.join(plots_dir, f"figA4_interaction{tag_suffix}.png"), dpi=200)
        plt.close(figA4)

    # 2D heatmaps per (ctx, inf): color=accuracy, text=latency, axes=ib/tb.
    rows_ib_tb = [
        r
        for r in summary
        if r.get("ib") is not None
        and r.get("tb") not in (None, "")
        and r.get("lat_mean") is not None
        and r.get("acc_global") is not None
    ]
    if rows_ib_tb:
        combos = sorted(
            {
                (int(r["ctx"]), int(r["inf"]))
                for r in rows_ib_tb
                if r.get("ctx") is not None and r.get("inf") is not None
            }
        )
        ib_vals = sorted({int(r["ib"]) for r in rows_ib_tb if r.get("ib")})
        tb_vals = sorted({int(r["tb"]) for r in rows_ib_tb if r.get("tb")})
        acc_vals = [to_float(r.get("acc_global")) for r in rows_ib_tb if r.get("acc_global") is not None]
        vmin = min(acc_vals) if acc_vals else None
        vmax = max(acc_vals) if acc_vals else None

        ncols = 3
        nrows = (len(combos) + ncols - 1) // ncols
        figH = plt.figure(figsize=(4.2 * ncols + 0.6, 3.2 * nrows), constrained_layout=True)
        gs = figH.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.06])
        axes = []
        for i in range(nrows * ncols):
            axes.append(figH.add_subplot(gs[i // ncols, i % ncols]))
        cax = figH.add_subplot(gs[:, -1])
        last_im = None
        for idx, (ctx, inf) in enumerate(combos):
            ax = axes[idx]
            grid_acc = np.full((len(tb_vals), len(ib_vals)), np.nan)
            grid_lat = np.full((len(tb_vals), len(ib_vals)), np.nan)
            for r in rows_ib_tb:
                if int(r["ctx"]) != ctx or int(r["inf"]) != inf:
                    continue
                ib = int(r["ib"])
                tb = int(r["tb"])
                i = tb_vals.index(tb)
                j = ib_vals.index(ib)
                grid_acc[i, j] = to_float(r.get("acc_global"))
                grid_lat[i, j] = to_float(r.get("lat_mean"))

            masked_acc = np.ma.masked_invalid(grid_acc)
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color="#f0f0f0")
            last_im = ax.imshow(
                masked_acc,
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xticks(range(len(ib_vals)), ib_vals)
            ax.set_yticks(range(len(tb_vals)), tb_vals)
            ax.set_xlabel("ib")
            ax.set_ylabel("tb")
            ax.set_title(f"ctx{ctx}/inf{inf}")
            for i in range(len(tb_vals)):
                for j in range(len(ib_vals)):
                    if np.isnan(grid_acc[i, j]):
                        continue
                    lat = grid_lat[i, j]
                    if np.isnan(lat):
                        continue
                    ax.text(j, i, f"{lat:.2f}s", ha="center", va="center", fontsize=7, color="white")

        for ax in axes[len(combos) :]:
            ax.axis("off")
        if last_im is not None:
            figH.colorbar(last_im, cax=cax, label="acc_global")
        figH.suptitle(
            f"Accuracy (color) + latency (text) vs ib/tb per ctx/inf{tag_label}"
        )
        figH.savefig(os.path.join(plots_dir, f"figA5_ctx_inf_ib_tb{tag_suffix}.png"), dpi=200)
        plt.close(figH)

    # Pareto drift over time: split each run into early/mid/late segments.
    window_runs = {
        r["run_id"]: r
        for r in summary
        if r.get("group") == "window"
        and r.get("ctx") is not None
        and r.get("inf") is not None
        and r.get("ib") is not None
    }
    if window_runs and cycle_rows:
        cycles_by_run = {}
        for row in cycle_rows:
            run_id = row.get("run_id")
            if run_id not in window_runs:
                continue
            acc = row.get("accuracy")
            lat = row.get("latency")
            if acc is None or lat is None:
                continue
            try:
                idx = int(row.get("frame_index", 0))
            except Exception:
                idx = 0
            cycles_by_run.setdefault(run_id, []).append((idx, float(lat), float(acc)))

        segments = [
            ("early", 0.0, 1.0 / 3.0),
            ("mid", 1.0 / 3.0, 2.0 / 3.0),
            ("late", 2.0 / 3.0, 1.0),
        ]
        segment_colors = {"early": "tab:blue", "mid": "tab:orange", "late": "tab:green"}

        combos = sorted(
            {
                (int(r["ctx"]), int(r["inf"]))
                for r in window_runs.values()
            }
        )
        ncols = 3
        nrows = (len(combos) + ncols - 1) // ncols
        figD, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
        axes = np.array(axes).reshape(-1)

        for idx, (ctx, inf) in enumerate(combos):
            ax = axes[idx]
            for seg_name, lo, hi in segments:
                pts = []
                labels = []
                for run_id, meta in window_runs.items():
                    if int(meta["ctx"]) != ctx or int(meta["inf"]) != inf:
                        continue
                    series = cycles_by_run.get(run_id, [])
                    if not series:
                        continue
                    series.sort(key=lambda v: v[0])
                    n = len(series)
                    start = int(n * lo)
                    end = int(n * hi)
                    if end <= start:
                        continue
                    seg = series[start:end]
                    lat_mean = float(np.mean([v[1] for v in seg]))
                    acc_mean = float(np.mean([v[2] for v in seg]))
                    pts.append((lat_mean, acc_mean))
                    labels.append(meta.get("ib"))

                if not pts:
                    continue
                mask = _pareto_mask([(x, -y) for x, y in pts])
                pareto_pts = [(x, y) for (x, y), m in zip(pts, mask) if m]
                pareto_pts.sort(key=lambda v: v[0])
                if pareto_pts:
                    ax.plot(
                        [p[0] for p in pareto_pts],
                        [p[1] for p in pareto_pts],
                        color=segment_colors[seg_name],
                        linewidth=1.6,
                        label=seg_name,
                    )
                for (x, y), ib in zip(pts, labels):
                    ax.scatter(x, y, color=segment_colors[seg_name], s=20)
                    if ib:
                        ax.text(x, y, f"ib{ib}", fontsize=6)

            ax.set_title(f"ctx{ctx}/inf{inf}")
            ax.set_xlabel("mean latency_amortized (s) ↓")
            ax.set_ylabel("mean cycle accuracy ↑")

        for ax in axes[len(combos) :]:
            ax.axis("off")
        handles = [
            plt.Line2D([0], [0], color=segment_colors[k], label=k, linewidth=2)
            for k in ("early", "mid", "late")
        ]
        figD.legend(handles=handles, loc="upper right")
        figD.suptitle(f"Window Pareto drift over time (early/mid/late){tag_label}")
        figD.tight_layout()
        figD.savefig(os.path.join(plots_dir, f"figA6_pareto_drift{tag_suffix}.png"), dpi=200)
        plt.close(figD)

        # Pareto drift projected onto config space: x=inf, y=ctx; each subplot is (ib,tb).
        ib_tb_combos = sorted(
            {
                (int(r["ib"]), int(r["tb"]))
                for r in window_runs.values()
                if r.get("ib") is not None and r.get("tb") is not None
            }
        )
        if ib_tb_combos:
            ncols = 3
            nrows = (len(ib_tb_combos) + ncols - 1) // ncols
            figC, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
            axes = np.array(axes).reshape(-1)

            for idx, (ib_val, tb_val) in enumerate(ib_tb_combos):
                ax = axes[idx]
                for seg_name, lo, hi in segments:
                    pts = []
                    for run_id, meta in window_runs.items():
                        if int(meta["ib"]) != ib_val or int(meta["tb"]) != tb_val:
                            continue
                        series = cycles_by_run.get(run_id, [])
                        if not series:
                            continue
                        series.sort(key=lambda v: v[0])
                        n = len(series)
                        start = int(n * lo)
                        end = int(n * hi)
                        if end <= start:
                            continue
                        seg = series[start:end]
                        lat_mean = float(np.mean([v[1] for v in seg]))
                        acc_mean = float(np.mean([v[2] for v in seg]))
                        pts.append((lat_mean, acc_mean, int(meta["ctx"]), int(meta["inf"])))

                    if not pts:
                        continue
                    ax.scatter(
                        [p[3] for p in pts],
                        [p[2] for p in pts],
                        color=segment_colors[seg_name],
                        s=18,
                        alpha=0.4,
                    )
                    mask = _pareto_mask([(x, -y) for x, y, _, _ in pts])
                    pareto_pts = [(ctx, inf) for (x, y, ctx, inf), m in zip(pts, mask) if m]
                    pareto_pts.sort(key=lambda v: (v[1], v[0]))
                    if pareto_pts:
                        ax.plot(
                            [p[1] for p in pareto_pts],
                            [p[0] for p in pareto_pts],
                            color=segment_colors[seg_name],
                            linewidth=1.6,
                            label=seg_name,
                        )
                        ax.scatter(
                            [p[1] for p in pareto_pts],
                            [p[0] for p in pareto_pts],
                            color=segment_colors[seg_name],
                            s=25,
                        )
                        for ctx, inf in pareto_pts:
                            ax.text(inf, ctx, f"{ctx}/{inf}", fontsize=6, ha="center", va="bottom")

                ax.set_xlabel("inf")
                ax.set_ylabel("ctx")
                ax.set_title(f"ib{ib_val}/tb{tb_val}")
                ax.set_xticks(sorted({int(r['inf']) for r in window_runs.values()}))
                ax.set_yticks(sorted({int(r['ctx']) for r in window_runs.values()}))

            for ax in axes[len(ib_tb_combos) :]:
                ax.axis("off")
            handles = [
                plt.Line2D([0], [0], color=segment_colors[k], label=k, linewidth=2)
                for k in ("early", "mid", "late")
            ]
            figC.legend(handles=handles, loc="upper right")
            figC.suptitle(
                f"Pareto configs over time (x=inf, y=ctx; per ib/tb){tag_label}"
            )
            figC.tight_layout()
            figC.savefig(
                os.path.join(plots_dir, f"figA7_pareto_ctx_inf{tag_suffix}.png"),
                dpi=200,
            )
            plt.close(figC)

    # 3D Pareto views: x=latency, y=accuracy, z=config dimension.
    if summary:
        dims = [
            ("ib", "inference_batch_size"),
            ("tb", "training_batch_size"),
            ("ctx", "context_length"),
            ("inf", "inference_length"),
        ]

        def _pareto_xy(rows):
            pts = []
            for r in rows:
                x = to_float(r.get("lat_mean"))
                y = to_float(r.get("acc_global"))
                if x is None or y is None:
                    continue
                pts.append((x, -y))
            return _pareto_mask(pts) if pts else []

        for dim_key, dim_label in dims:
            rows = [
                r
                for r in summary
                if r.get(dim_key) is not None
                and r.get("lat_mean") is not None
                and r.get("acc_global") is not None
            ]
            if not rows:
                continue
            fig3d = plt.figure(figsize=(8, 6))
            ax3d = fig3d.add_subplot(1, 1, 1, projection="3d")
            xs = [to_float(r.get("lat_mean")) for r in rows]
            ys = [to_float(r.get("acc_global")) for r in rows]
            zs = [to_float(r.get(dim_key)) for r in rows]
            ax3d.scatter(xs, ys, zs, c=ys, cmap="viridis", edgecolors="black", s=40)

            mask = _pareto_xy(rows)
            pareto_pts = []
            for is_front, r in zip(mask, rows):
                if not is_front:
                    continue
                x = to_float(r.get("lat_mean"))
                y = to_float(r.get("acc_global"))
                z = to_float(r.get(dim_key))
                if x is None or y is None or z is None:
                    continue
                pareto_pts.append((x, y, z))
                ax3d.scatter(x, y, z, facecolors="none", edgecolors="red", s=80, linewidths=1.5)
            if pareto_pts:
                pareto_pts.sort(key=lambda v: v[0])
                ax3d.plot(
                    [p[0] for p in pareto_pts],
                    [p[1] for p in pareto_pts],
                    [p[2] for p in pareto_pts],
                    color="red",
                    linewidth=1.2,
                )

            ax3d.set_xlabel("mean latency_amortized (s) ↓")
            ax3d.set_ylabel("global_accuracy ↑")
            ax3d.set_zlabel(dim_label)
            ax3d.set_title(f"3D Pareto (latency/accuracy + {dim_key}){tag_label}")
            fig3d.tight_layout()
            fig3d.savefig(
                os.path.join(plots_dir, f"fig3d_pareto_{dim_key}{tag_suffix}.png"), dpi=200
            )
            plt.close(fig3d)


if __name__ == "__main__":
    main()
