#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _extract_run_suffix(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) < 2:
        return run_id
    return "_".join(parts[-2:])


def _latest_dynamic_run_id(log_dir: Path) -> str:
    candidates = sorted(
        log_dir.glob("abc24x3_trigpu_shift_taskpool_*.status.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        run_id = p.name.replace(".status.log", "")
        if "static_balanced" in run_id:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "FINISH " in text:
            return run_id
    raise RuntimeError("No finished dynamic abc24 run found.")


def _load_manifest(tsv_path: Path) -> List[Tuple[str, str, int, int]]:
    rows: List[Tuple[str, str, int, int]] = []
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 11:
            continue
        run_name = cols[0]
        case_id = cols[2]
        ctx = int(cols[5])
        inf = int(cols[6])
        rows.append((run_name, case_id, ctx, inf))
    return rows


def _mean_metrics(jsonl_path: Path) -> Tuple[Optional[float], Optional[float], int]:
    latencies: List[float] = []
    accuracies: List[float] = []
    if not jsonl_path.exists():
        return None, None, 0
    with jsonl_path.open("r", encoding="utf-8") as f:
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
                latencies.append(float(lat))
            if isinstance(acc, (int, float)) and math.isfinite(float(acc)):
                accuracies.append(float(acc))
    lat_mean = sum(latencies) / len(latencies) if latencies else None
    acc_mean = sum(accuracies) / len(accuracies) if accuracies else None
    return lat_mean, acc_mean, len(latencies)


def _case_sort_key(case_id: str) -> Tuple[str, int]:
    prefix = case_id[0]
    idx = int(case_id[1:])
    return (prefix, idx)


def plot_bars(
    case_labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: Path,
    colors: List[str],
) -> None:
    plt.figure(figsize=(16, 5))
    x = list(range(len(case_labels)))
    plt.bar(x, values, color=colors)
    plt.xticks(x, case_labels, rotation=60, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_combined(
    case_labels: List[str],
    lat_values: List[float],
    acc_values: List[float],
    out_path: Path,
    colors: List[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), constrained_layout=True)
    x = list(range(len(case_labels)))

    axes[0].bar(x, lat_values, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(case_labels, rotation=60, ha="right")
    axes[0].set_title("Latency Mean per Case (Dynamic Scheduler ON)")
    axes[0].set_ylabel("latency (s)")

    axes[1].bar(x, acc_values, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(case_labels, rotation=60, ha="right")
    axes[1].set_title("Accuracy Mean per Case (Dynamic Scheduler ON)")
    axes[1].set_ylabel("accuracy")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot latency/accuracy bar charts for ABC24 dynamic-scheduler runs."
    )
    ap.add_argument(
        "--inference-logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "inference_logs",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default="",
        help="abc24x3_trigpu_shift_taskpool_* run id. If empty, auto-pick latest finished dynamic run.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
    )
    args = ap.parse_args()

    log_dir = args.inference_logs_dir.resolve()
    run_id = args.run_id.strip() or _latest_dynamic_run_id(log_dir)
    tsv_path = log_dir / f"{run_id}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing manifest: {tsv_path}")

    rows = _load_manifest(tsv_path)
    rows = sorted(rows, key=lambda r: _case_sort_key(r[1]))

    suffix = _extract_run_suffix(run_id)
    summary_path = args.out_dir / f"abc24_dynamic_latency_accuracy_{suffix}.csv"
    lat_plot = args.out_dir / f"latency_bar_dynamic_abc24_{suffix}.png"
    acc_plot = args.out_dir / f"accuracy_bar_dynamic_abc24_{suffix}.png"
    combined_plot = args.out_dir / f"latency_accuracy_bar_dynamic_abc24_{suffix}.png"

    case_labels: List[str] = []
    lat_values: List[float] = []
    acc_values: List[float] = []
    summary_lines = ["case_id,ctx,inf,latency_mean,accuracy_mean,infer_points,run_name"]
    colors: List[str] = []

    for run_name, case_id, ctx, inf in rows:
        jsonl_path = log_dir / f"{run_name}.jsonl"
        lat_mean, acc_mean, n_points = _mean_metrics(jsonl_path)
        if lat_mean is None or acc_mean is None:
            continue
        case_labels.append(case_id)
        lat_values.append(lat_mean)
        acc_values.append(acc_mean)
        summary_lines.append(
            f"{case_id},{ctx},{inf},{lat_mean:.6f},{acc_mean:.6f},{n_points},{run_name}"
        )
        if case_id.startswith("A"):
            colors.append("#4C78A8")
        elif case_id.startswith("B"):
            colors.append("#F58518")
        else:
            colors.append("#54A24B")

    if not case_labels:
        raise RuntimeError("No valid case metrics found in jsonl files.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plot_bars(
        case_labels=case_labels,
        values=lat_values,
        title=f"Latency Mean per Case (Dynamic Scheduler ON) - {run_id}",
        ylabel="latency (s)",
        out_path=lat_plot,
        colors=colors,
    )
    plot_bars(
        case_labels=case_labels,
        values=acc_values,
        title=f"Accuracy Mean per Case (Dynamic Scheduler ON) - {run_id}",
        ylabel="accuracy",
        out_path=acc_plot,
        colors=colors,
    )
    plot_combined(
        case_labels=case_labels,
        lat_values=lat_values,
        acc_values=acc_values,
        out_path=combined_plot,
        colors=colors,
    )

    print(f"run_id={run_id}")
    print(f"cases={len(case_labels)}")
    print(f"summary={summary_path}")
    print(f"latency_plot={lat_plot}")
    print(f"accuracy_plot={acc_plot}")
    print(f"combined_plot={combined_plot}")


if __name__ == "__main__":
    main()

