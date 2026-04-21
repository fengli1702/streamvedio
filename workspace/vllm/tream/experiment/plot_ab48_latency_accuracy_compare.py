#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _case_sort_key(case_id: str) -> Tuple[str, int]:
    prefix = case_id[:1]
    try:
        idx = int(case_id[1:])
    except Exception:
        idx = 0
    return (prefix, idx)


def _mean_metrics(jsonl_path: Path) -> Tuple[Optional[float], Optional[float], int]:
    if not jsonl_path.exists():
        return None, None, 0
    latencies: List[float] = []
    accuracies: List[float] = []
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
                latencies.append(float(lat))
            if isinstance(acc, (int, float)) and math.isfinite(float(acc)):
                accuracies.append(float(acc))
    lat_mean = sum(latencies) / len(latencies) if latencies else None
    acc_mean = sum(accuracies) / len(accuracies) if accuracies else None
    return lat_mean, acc_mean, len(latencies)


def _load_manifest(tsv_path: Path) -> List[dict]:
    out: List[dict] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append(row)
    return out


def _load_dynamic_merged(main_manifest: Path, retry_manifest: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path, source in [(main_manifest, "main48"), (retry_manifest, "retry14")]:
        rows = _load_manifest(path)
        for row in rows:
            out[row["case_id"]] = {
                "case_id": row["case_id"],
                "ctx": int(row["ctx"]),
                "inf": int(row["inf"]),
                "run_name": row["run_name"],
                "source": source,
            }
    return out


def _load_static_case_manifest(static_manifest: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    rows = _load_manifest(static_manifest)
    for row in rows:
        out[row["case_id"]] = {
            "ctx": int(row["ctx"]),
            "inf": int(row["inf"]),
            "run_name": row["run_name"],
            "source": "abc24_static_balanced_20260316_235206",
        }
    return out


def _collect_static_pool(log_dir: Path) -> Dict[Tuple[int, int], List[str]]:
    # "original non-dynamic" pool: inference jsonl without scheduler twin.
    pool: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    pat = re.compile(r".+_c(\d+)_i(\d+)_ib32_tb16\.jsonl$")
    for p in log_dir.glob("*_c*_i*_ib32_tb16.jsonl"):
        name = p.name
        if name.startswith("scheduler_") or name.startswith("spec_metrics_"):
            continue
        m = pat.match(name)
        if not m:
            continue
        run_base = p.stem
        ctx_s, inf_s = m.group(1), m.group(2)
        # Non-dynamic marker for this repo layout.
        if (log_dir / f"scheduler_{name}").exists():
            continue
        ctx = int(ctx_s)
        inf = int(inf_s)
        pool[(ctx, inf)].append(run_base)

    # Stable preference order for static source selection.
    def _pri(run_base: str) -> Tuple[int, str]:
        if run_base.startswith("doh_spec_static_"):
            return (0, run_base)
        if run_base.startswith("doh_spec_focus_"):
            return (1, run_base)
        if run_base.startswith("doh_shift_abc24x3_"):
            return (2, run_base)
        return (3, run_base)

    for k in list(pool.keys()):
        pool[k] = sorted(pool[k], key=_pri)
    return pool


def _plot_grouped_combined(
    labels: List[str],
    lat_dynamic: List[float],
    lat_static: List[float],
    acc_dynamic: List[float],
    acc_static: List[float],
    title: str,
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
    axes[0].legend()

    axes[1].bar(x_dyn, acc_dynamic, width=width, color="#1f77b4", label="Dynamic ON")
    axes[1].bar(x_sta, acc_static, width=width, color="#ff7f0e", label="Dynamic OFF")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right")
    axes[1].set_ylabel("Accuracy (higher is better ↑)")
    axes[1].set_xlabel("(ctx,inf)")
    axes[1].set_title("Average Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_single_bar(
    labels: List[str],
    v_dynamic: List[float],
    v_static: List[float],
    title: str,
    y_label: str,
    out_path: Path,
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
    plt.title(title)
    plt.xlabel("(ctx,inf)")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_scatter_delta(records: List[dict], out_path: Path) -> None:
    # x = dynamic - static latency (left is better), y = dynamic - static accuracy (up is better)
    xs = [float(r["dynamic_latency"]) - float(r["static_latency"]) for r in records]
    ys = [float(r["dynamic_accuracy"]) - float(r["static_accuracy"]) for r in records]
    labels = [f"({r['ctx']},{r['inf']})" for r in records]

    plt.figure(figsize=(11, 8))
    plt.scatter(xs, ys, c="#2ca02c", s=48, alpha=0.9)
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab, fontsize=8, alpha=0.9)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("Δ latency (dynamic - static)  [lower is better ←]")
    plt.ylabel("Δ accuracy (dynamic - static)  [higher is better ↑]")
    plt.title("Dynamic vs Static: Δlatency vs Δaccuracy")
    plt.grid(True, linestyle="--", alpha=0.25)

    # mark preferred direction (top-left)
    plt.annotate(
        "better",
        xy=(0.12, 0.88),
        xytext=(0.24, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.5),
        color="#1f77b4",
    )
    plt.annotate(
        "",
        xy=(0.12, 0.88),
        xytext=(0.22, 0.88),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.5),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare latest dynamic 48 vs original static pool on overlap points."
    )
    ap.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parents[1] / "inference_logs")
    ap.add_argument(
        "--dynamic-main-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "inference_logs/ab48x2_dualgpu_shift_taskpool_jsdonly_20260320_090523.tsv",
    )
    ap.add_argument(
        "--dynamic-retry-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "inference_logs/ab14x2_retry_from_ab48_jsdonly_20260320_214650.tsv",
    )
    ap.add_argument(
        "--static-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "inference_logs/abc24x3_trigpu_shift_taskpool_static_balanced_20260316_235206.tsv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots" / f"compare_ab48_vs_static_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    ap.add_argument(
        "--target-pairs",
        type=int,
        default=36,
        help="Expected pair count target. Script does not force-fill missing points.",
    )
    args = ap.parse_args()

    log_dir = args.log_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dyn = _load_dynamic_merged(args.dynamic_main_manifest.resolve(), args.dynamic_retry_manifest.resolve())
    sta_case = _load_static_case_manifest(args.static_manifest.resolve())
    sta_pool = _collect_static_pool(log_dir)

    records: List[dict] = []
    missing: List[dict] = []
    for case_id in sorted(dyn.keys(), key=_case_sort_key):
        d = dyn[case_id]
        ctx = int(d["ctx"])
        inf = int(d["inf"])

        d_lat, d_acc, d_n = _mean_metrics(log_dir / f"{d['run_name']}.jsonl")
        if d_lat is None or d_acc is None:
            missing.append(
                {
                    "case_id": case_id,
                    "ctx": ctx,
                    "inf": inf,
                    "reason": "dynamic_metrics_missing",
                }
            )
            continue

        static_run = None
        static_source = None
        if case_id in sta_case:
            s = sta_case[case_id]
            if int(s["ctx"]) == ctx and int(s["inf"]) == inf:
                static_run = s["run_name"]
                static_source = s["source"]
        if static_run is None:
            cands = sta_pool.get((ctx, inf), [])
            if cands:
                static_run = cands[0]
                static_source = "static_pool_nondynamic"

        if static_run is None:
            missing.append(
                {
                    "case_id": case_id,
                    "ctx": ctx,
                    "inf": inf,
                    "reason": "static_not_found",
                }
            )
            continue

        s_lat, s_acc, s_n = _mean_metrics(log_dir / f"{static_run}.jsonl")
        if s_lat is None or s_acc is None:
            missing.append(
                {
                    "case_id": case_id,
                    "ctx": ctx,
                    "inf": inf,
                    "reason": "static_metrics_missing",
                    "static_run": static_run,
                }
            )
            continue

        records.append(
            {
                "case_id": case_id,
                "ctx": ctx,
                "inf": inf,
                "dynamic_run": d["run_name"],
                "dynamic_source": d["source"],
                "dynamic_latency": d_lat,
                "dynamic_accuracy": d_acc,
                "dynamic_points": d_n,
                "static_run": static_run,
                "static_source": static_source,
                "static_latency": s_lat,
                "static_accuracy": s_acc,
                "static_points": s_n,
            }
        )

    records.sort(key=lambda r: (int(r["ctx"]), int(r["inf"]), _case_sort_key(str(r["case_id"]))))

    labels = [f"({r['ctx']},{r['inf']})" for r in records]
    lat_dynamic = [float(r["dynamic_latency"]) for r in records]
    lat_static = [float(r["static_latency"]) for r in records]
    acc_dynamic = [float(r["dynamic_accuracy"]) for r in records]
    acc_static = [float(r["static_accuracy"]) for r in records]

    summary_csv = out_dir / "latency_accuracy_compare_dynamic48_vs_static_pool.csv"
    missing_csv = out_dir / "missing_pairs_dynamic48_vs_static_pool.csv"
    combined_png = out_dir / "latency_accuracy_bar_compare_dynamic48_vs_static_pool.png"
    latency_png = out_dir / "latency_bar_compare_dynamic48_vs_static_pool.png"
    accuracy_png = out_dir / "accuracy_bar_compare_dynamic48_vs_static_pool.png"
    scatter_png = out_dir / "scatter_delta_latency_accuracy_dynamic48_vs_static_pool.png"
    meta_md = out_dir / "README.md"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "ctx",
                "inf",
                "dynamic_latency",
                "static_latency",
                "latency_delta",
                "dynamic_accuracy",
                "static_accuracy",
                "accuracy_delta",
                "dynamic_points",
                "static_points",
                "dynamic_run",
                "static_run",
                "dynamic_source",
                "static_source",
            ]
        )
        for r in records:
            dl = float(r["dynamic_latency"])
            sl = float(r["static_latency"])
            da = float(r["dynamic_accuracy"])
            sa = float(r["static_accuracy"])
            w.writerow(
                [
                    r["case_id"],
                    r["ctx"],
                    r["inf"],
                    f"{dl:.6f}",
                    f"{sl:.6f}",
                    f"{(dl - sl):.6f}",
                    f"{da:.6f}",
                    f"{sa:.6f}",
                    f"{(da - sa):.6f}",
                    r["dynamic_points"],
                    r["static_points"],
                    r["dynamic_run"],
                    r["static_run"],
                    r["dynamic_source"],
                    r["static_source"],
                ]
            )

    with missing_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "ctx", "inf", "reason", "static_run"])
        w.writeheader()
        for row in missing:
            w.writerow(row)

    _plot_single_bar(
        labels=labels,
        v_dynamic=lat_dynamic,
        v_static=lat_static,
        title="Average Latency: Dynamic ON vs Dynamic OFF",
        y_label="Latency (lower is better ↓)",
        out_path=latency_png,
    )
    _plot_single_bar(
        labels=labels,
        v_dynamic=acc_dynamic,
        v_static=acc_static,
        title="Average Accuracy: Dynamic ON vs Dynamic OFF",
        y_label="Accuracy (higher is better ↑)",
        out_path=accuracy_png,
    )
    _plot_grouped_combined(
        labels=labels,
        lat_dynamic=lat_dynamic,
        lat_static=lat_static,
        acc_dynamic=acc_dynamic,
        acc_static=acc_static,
        title="Dynamic ON vs Original Dynamic OFF Baseline",
        out_path=combined_png,
    )
    _plot_scatter_delta(records=records, out_path=scatter_png)

    meta_md.write_text(
        "\n".join(
            [
                "# Dynamic48 vs Static Pool",
                "",
                f"- generated_utc: {datetime.utcnow().isoformat()}Z",
                f"- dynamic_main_manifest: `{args.dynamic_main_manifest.resolve()}`",
                f"- dynamic_retry_manifest: `{args.dynamic_retry_manifest.resolve()}`",
                f"- static_manifest: `{args.static_manifest.resolve()}`",
                f"- target_pairs: {args.target_pairs}",
                f"- paired_records: {len(records)}",
                f"- missing_records: {len(missing)}",
                "",
                "Outputs:",
                f"- `{summary_csv.name}`",
                f"- `{missing_csv.name}`",
                f"- `{latency_png.name}`",
                f"- `{accuracy_png.name}`",
                f"- `{combined_png.name}`",
                f"- `{scatter_png.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"paired={len(records)}")
    print(f"missing={len(missing)}")
    print(f"target_pairs={args.target_pairs}")
    print(f"out_dir={out_dir}")
    print(f"summary_csv={summary_csv}")
    print(f"combined_png={combined_png}")
    print(f"scatter_png={scatter_png}")


if __name__ == "__main__":
    main()
