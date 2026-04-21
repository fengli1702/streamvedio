#!/usr/bin/env python3
import argparse
import csv
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


def _latest_finished_dynamic_run_id(log_dir: Path) -> str:
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


def _latest_finished_static_run_id(log_dir: Path) -> str:
    candidates = sorted(
        log_dir.glob("abc24x3_trigpu_shift_taskpool_static_balanced_*.status.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        run_id = p.name.replace(".status.log", "")
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "FINISH " in text:
            return run_id
    raise RuntimeError("No finished static-balanced abc24 run found.")


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


def _load_manifest(tsv_path: Path) -> Dict[str, Tuple[int, int, str]]:
    out: Dict[str, Tuple[int, int, str]] = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            case_id = row["case_id"]
            ctx = int(row["ctx"])
            inf = int(row["inf"])
            run_name = row["run_name"]
            out[case_id] = (ctx, inf, run_name)
    return out


def _pair_records(
    log_dir: Path,
    dynamic_manifest: Dict[str, Tuple[int, int, str]],
    static_manifest: Dict[str, Tuple[int, int, str]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    common_cases = sorted(set(dynamic_manifest) & set(static_manifest))
    if not common_cases:
        raise RuntimeError("No overlapping case_id between dynamic/static manifests.")

    for case_id in common_cases:
        d_ctx, d_inf, d_run = dynamic_manifest[case_id]
        s_ctx, s_inf, s_run = static_manifest[case_id]
        if (d_ctx, d_inf) != (s_ctx, s_inf):
            raise RuntimeError(
                f"Case {case_id} mismatch: dynamic=({d_ctx},{d_inf}) static=({s_ctx},{s_inf})"
            )

        d_lat, d_acc, d_n = _mean_metrics(log_dir / f"{d_run}.jsonl")
        s_lat, s_acc, s_n = _mean_metrics(log_dir / f"{s_run}.jsonl")
        if d_lat is None or d_acc is None or s_lat is None or s_acc is None:
            continue

        records.append(
            {
                "case_id": case_id,
                "ctx": d_ctx,
                "inf": d_inf,
                "dynamic_latency": d_lat,
                "dynamic_accuracy": d_acc,
                "dynamic_points": d_n,
                "dynamic_run": d_run,
                "static_latency": s_lat,
                "static_accuracy": s_acc,
                "static_points": s_n,
                "static_run": s_run,
            }
        )

    records.sort(key=lambda x: (int(x["ctx"]), int(x["inf"]), str(x["case_id"])))
    return records


def _plot_grouped_bar(
    labels: List[str],
    v_dynamic: List[float],
    v_static: List[float],
    title: str,
    xlabel: str,
    out_path: Path,
    compare_text: str = "",
    y_label: str = "↑",
    show_better_up: bool = False,
) -> None:
    n = len(labels)
    x = list(range(n))
    width = 0.38
    x_dyn = [i - width / 2 for i in x]
    x_sta = [i + width / 2 for i in x]

    plt.figure(figsize=(max(16, n * 0.62), 5.8))
    plt.bar(x_dyn, v_dynamic, width=width, color="#1f77b4", label="Dynamic ON")
    plt.bar(x_sta, v_static, width=width, color="#ff7f0e", label="Dynamic OFF")
    plt.xticks(x, labels, rotation=60, ha="right")
    plot_title = title if not compare_text else f"{title}\n{compare_text}"
    plt.title(plot_title)
    plt.ylabel(y_label)
    plt.xlabel(xlabel)
    if show_better_up:
        plt.annotate(
            "better",
            xy=(0.03, 0.93),
            xytext=(0.03, 0.78),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=1.5),
            color="#2ca02c",
            ha="center",
            va="center",
        )
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_grouped_combined(
    labels: List[str],
    lat_dynamic: List[float],
    lat_static: List[float],
    acc_dynamic: List[float],
    acc_static: List[float],
    compare_text: str,
    out_path: Path,
) -> None:
    n = len(labels)
    x = list(range(n))
    width = 0.38
    x_dyn = [i - width / 2 for i in x]
    x_sta = [i + width / 2 for i in x]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(18, n * 0.74), 6.0),
        constrained_layout=True,
    )

    axes[0].bar(x_dyn, lat_dynamic, width=width, color="#1f77b4", label="Dynamic ON")
    axes[0].bar(x_sta, lat_static, width=width, color="#ff7f0e", label="Dynamic OFF")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=60, ha="right")
    axes[0].set_ylabel("Latency (lower is better ↓)")
    axes[0].set_xlabel("(ctx,inf)")
    axes[0].set_title("Average Latency")
    axes[0].annotate(
        "better",
        xy=(0.03, 0.80),
        xytext=(0.03, 0.93),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=1.5),
        color="#2ca02c",
        ha="center",
        va="center",
    )
    axes[0].legend()

    axes[1].bar(x_dyn, acc_dynamic, width=width, color="#1f77b4", label="Dynamic ON")
    axes[1].bar(x_sta, acc_static, width=width, color="#ff7f0e", label="Dynamic OFF")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right")
    axes[1].set_ylabel("Accuracy (higher is better ↑)")
    axes[1].set_xlabel("(ctx,inf)")
    axes[1].set_title("Average Accuracy")
    axes[1].annotate(
        "better",
        xy=(0.03, 0.93),
        xytext=(0.03, 0.80),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=1.5),
        color="#2ca02c",
        ha="center",
        va="center",
    )
    axes[1].legend()

    fig.suptitle("Dynamic vs Static")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot ABC24 grouped bar comparison for dynamic ON vs OFF (24+24)."
    )
    ap.add_argument(
        "--inference-logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "inference_logs",
    )
    ap.add_argument(
        "--dynamic-run-id",
        type=str,
        default="",
        help="abc24x3_trigpu_shift_taskpool_* (dynamic ON). Empty means latest finished.",
    )
    ap.add_argument(
        "--static-run-id",
        type=str,
        default="",
        help="abc24x3_trigpu_shift_taskpool_static_balanced_* (dynamic OFF). Empty means latest finished.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
    )
    args = ap.parse_args()

    log_dir = args.inference_logs_dir.resolve()
    dynamic_run_id = args.dynamic_run_id.strip() or _latest_finished_dynamic_run_id(log_dir)
    static_run_id = args.static_run_id.strip() or _latest_finished_static_run_id(log_dir)

    dynamic_tsv = log_dir / f"{dynamic_run_id}.tsv"
    static_tsv = log_dir / f"{static_run_id}.tsv"
    if not dynamic_tsv.exists():
        raise FileNotFoundError(f"Missing dynamic manifest: {dynamic_tsv}")
    if not static_tsv.exists():
        raise FileNotFoundError(f"Missing static manifest: {static_tsv}")

    dynamic_manifest = _load_manifest(dynamic_tsv)
    static_manifest = _load_manifest(static_tsv)
    records = _pair_records(log_dir, dynamic_manifest, static_manifest)
    if not records:
        raise RuntimeError("No valid paired records with metrics.")

    dynamic_suffix = _extract_run_suffix(dynamic_run_id)
    static_suffix = _extract_run_suffix(static_run_id)
    pair_suffix = f"{dynamic_suffix}_vs_{static_suffix}"

    summary_csv = args.out_dir / f"abc24_latency_accuracy_compare_{pair_suffix}.csv"
    latency_plot = args.out_dir / f"latency_bar_compare_abc24_{pair_suffix}.png"
    accuracy_plot = args.out_dir / f"accuracy_bar_compare_abc24_{pair_suffix}.png"
    combined_plot = args.out_dir / f"latency_accuracy_bar_compare_abc24_{pair_suffix}.png"

    labels = [f"({r['ctx']},{r['inf']})" for r in records]
    lat_dynamic = [float(r["dynamic_latency"]) for r in records]
    lat_static = [float(r["static_latency"]) for r in records]
    acc_dynamic = [float(r["dynamic_accuracy"]) for r in records]
    acc_static = [float(r["static_accuracy"]) for r in records]

    summary_lines = [
        "case_id,ctx,inf,dynamic_latency,static_latency,latency_delta,dynamic_accuracy,static_accuracy,accuracy_delta,dynamic_points,static_points,dynamic_run,static_run"
    ]
    for r in records:
        d_lat = float(r["dynamic_latency"])
        s_lat = float(r["static_latency"])
        d_acc = float(r["dynamic_accuracy"])
        s_acc = float(r["static_accuracy"])
        summary_lines.append(
            f"{r['case_id']},{r['ctx']},{r['inf']},{d_lat:.6f},{s_lat:.6f},{(d_lat - s_lat):.6f},{d_acc:.6f},{s_acc:.6f},{(d_acc - s_acc):.6f},{r['dynamic_points']},{r['static_points']},{r['dynamic_run']},{r['static_run']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    compare_text = f"Dynamic: {dynamic_run_id}  vs  Static: {static_run_id}"

    _plot_grouped_bar(
        labels=labels,
        v_dynamic=lat_dynamic,
        v_static=lat_static,
        title="Average Latency",
        xlabel="(ctx,inf)",
        compare_text=compare_text,
        out_path=latency_plot,
    )
    _plot_grouped_bar(
        labels=labels,
        v_dynamic=acc_dynamic,
        v_static=acc_static,
        title="Dynamic vs Static",
        xlabel="(ctx,inf)",
        y_label="Accuracy (higher is better ↑)",
        show_better_up=True,
        out_path=accuracy_plot,
    )
    _plot_grouped_combined(
        labels=labels,
        lat_dynamic=lat_dynamic,
        lat_static=lat_static,
        acc_dynamic=acc_dynamic,
        acc_static=acc_static,
        compare_text=compare_text,
        out_path=combined_plot,
    )

    print(f"dynamic_run_id={dynamic_run_id}")
    print(f"static_run_id={static_run_id}")
    print(f"cases={len(records)}")
    print(f"summary={summary_csv}")
    print(f"latency_plot={latency_plot}")
    print(f"accuracy_plot={accuracy_plot}")
    print(f"combined_plot={combined_plot}")


if __name__ == "__main__":
    main()
