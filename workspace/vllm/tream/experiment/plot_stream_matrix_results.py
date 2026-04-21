#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 run_streaming_mini_spec_nsys.sh 产生的 4 组日志，输出多种统计与对比图。

Usage:
  python3 experiments/plot_stream_matrix_results.py \
      --input-root /app/mini_logs_stream_matrix \
      --output-dir /app/mini_logs_stream_matrix/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_spec_metrics(spec_path: Path) -> Dict[str, Any]:
    if not spec_path.exists():
        return {}
    accepted = 0
    proposed = 0
    series: list[float] = []
    with spec_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            accepted += int(rec.get("accepted") or 0)
            proposed += int(rec.get("proposed") or 0)
            acc_rate = rec.get("acceptance_rate")
            if acc_rate is None and rec.get("proposed"):
                try:
                    acc_rate = rec.get("accepted", 0) / rec.get("proposed", 1)
                except Exception:
                    acc_rate = None
            series.append(acc_rate if acc_rate is not None else 0.0)
    result: Dict[str, float] = {
        "accepted_tokens_calc": accepted,
        "proposed_tokens_calc": proposed,
    }
    if proposed:
        result["acceptance_rate_calc"] = accepted / proposed
    if series:
        result["acceptance_series"] = series
    return result


def load_chunk_stats(jsonl_path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if not jsonl_path.exists():
        return stats
    chunks: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    if not chunks:
        return stats
    add_times = np.array([c.get("add_time", 0.0) for c in chunks])
    decode_times = np.array([c.get("decode_time", 0.0) for c in chunks])
    chunk_times = np.array([c.get("chunk_time", 0.0) for c in chunks])
    steps = np.arange(1, len(chunks) + 1)
    stats.update({
        "chunk_count": len(chunks),
        "avg_add_time_per_chunk": float(add_times.mean()),
        "avg_decode_time_per_chunk": float(decode_times.mean()),
        "avg_chunk_time": float(chunk_times.mean()),
        "p95_chunk_time": float(np.percentile(chunk_times, 95)),
        "max_chunk_time": float(chunk_times.max()),
        "sum_chunk_time": float(chunk_times.sum()),
        "chunk_series": chunk_times.tolist(),
        "chunk_steps": steps.tolist(),
        "chunk_add_times": add_times.tolist(),
        "chunk_decode_times": decode_times.tolist(),
    })
    return stats


def downsample_series(steps: List[int], values: List[float],
                      stride: int) -> Tuple[List[int], List[float]]:
    if stride <= 1 or len(values) <= stride:
        return steps, values
    ds_steps: List[int] = []
    ds_vals: List[float] = []
    for i in range(0, len(values), stride):
        chunk_vals = values[i:i + stride]
        chunk_steps = steps[i:i + stride]
        if not chunk_vals:
            continue
        ds_vals.append(float(np.mean(chunk_vals)))
        ds_steps.append(int(np.mean(chunk_steps)))
    return ds_steps, ds_vals


def extract_metrics(case_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"case": case_dir.name}
    logs_dir = case_dir / "mini_spec_logs"
    baseline_path = logs_dir / "baseline_profile.json"
    spec_path = logs_dir / "spec_profile.json"

    if baseline_path.exists():
        data = load_json(baseline_path)
        profiling = data.get("profiling") or {}
        metrics.update({
            "run_mode": "baseline",
            "throughput": data["result"].get("throughput"),
            "avg_latency": data["result"].get("avg_latency"),
            "prompt_tokens": data["result"].get("prompt_tokens"),
            "generated_tokens": data["result"].get("generated_tokens"),
            "total_decode_time": profiling.get("total_decode_time") or profiling.get("total_chunk_time"),
            "engine_init_time": profiling.get("engine_init_time"),
        })
        baseline_chunks = logs_dir / "baseline_chunks.jsonl"
        metrics.update(load_chunk_stats(baseline_chunks))
    if spec_path.exists():
        data = load_json(spec_path)
        profiling = data.get("profiling") or {}
        accepted = data["result"].get("accepted")
        proposed = data["result"].get("proposed")
        acceptance_rate = data["result"].get("acceptance_rate")
        if acceptance_rate is None and accepted is not None and proposed:
            try:
                acceptance_rate = accepted / proposed
            except ZeroDivisionError:
                acceptance_rate = None
        metrics.update({
            "run_mode": "spec",
            "throughput": data["result"].get("throughput"),
            "avg_latency": data["result"].get("avg_latency"),
            "prompt_tokens": data["result"].get("prompt_tokens"),
            "generated_tokens": data["result"].get("generated_tokens"),
            "acceptance_rate": acceptance_rate,
            "accepted_tokens": accepted,
            "proposed_tokens": proposed,
            "total_decode_time": profiling.get("total_decode_time"),
            "engine_init_time": profiling.get("engine_init_time"),
            "accuracy": profiling.get("acc_accuracy"),
        })
        spec_chunks = logs_dir / "spec_chunks.jsonl"
        metrics.update(load_chunk_stats(spec_chunks))
        spec_metrics_path = case_dir / "spec_metrics.jsonl"
        calc = load_spec_metrics(spec_metrics_path)
        if calc:
            if metrics.get("acceptance_rate") is None:
                metrics["acceptance_rate"] = calc.get("acceptance_rate_calc")
            if metrics.get("accepted_tokens") is None:
                metrics["accepted_tokens"] = calc.get("accepted_tokens_calc")
            if metrics.get("proposed_tokens") is None:
                metrics["proposed_tokens"] = calc.get("proposed_tokens_calc")
            if calc.get("acceptance_series"):
                metrics["acceptance_series"] = calc["acceptance_series"]
    return metrics


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str,
             output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(df[x_col], df[y_col], color="#4C72B0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multi_bars(df: pd.DataFrame, columns: List[str], title: str,
                    output_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.15
    plotted = False
    for idx, col in enumerate(columns):
        if col not in df:
            continue
        plt.bar(x + idx * width, df[col], width, label=col)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xticks(x + width * (len(columns) - 1) / 2, df["case"], rotation=20, ha="right")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multi_lines(df: pd.DataFrame, columns: List[str], title: str,
                     ylabel: str, output_path: Path) -> None:
    valid_cols = [col for col in columns if col in df and df[col].notna().any()]
    if not valid_cols:
        return
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    for col in valid_cols:
        plt.plot(x, df[col], marker="o", label=col)
    plt.xticks(x, df["case"], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_chunk_series_multi(df: pd.DataFrame,
                            columns: List[str],
                            title: str,
                            ylabel: str,
                            output_path: Path,
                            stride: int = 1) -> None:
    plt.figure(figsize=(10, 5))
    has_curve = False
    for _, row in df.iterrows():
        for col in columns:
            series = row.get(col)
            steps = row.get("chunk_steps")
            if isinstance(series, list) and series and isinstance(steps, list):
                has_curve = True
                ds_steps, ds_vals = downsample_series(steps, series, stride)
                plt.plot(ds_steps, ds_vals, marker="o",
                         label=f"{row['case']}:{col}")
    if not has_curve:
        plt.close()
        return
    plt.xlabel("Chunk Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_acceptance_series(df: pd.DataFrame,
                           output_path: Path,
                           stride: int = 1) -> None:
    plt.figure(figsize=(10, 5))
    has_curve = False
    for _, row in df.iterrows():
        series = row.get("acceptance_series")
        if isinstance(series, list) and series:
            has_curve = True
            steps = list(range(1, len(series) + 1))
            ds_steps, ds_vals = downsample_series(steps, series, stride)
            plt.plot(ds_steps, ds_vals, marker="o", label=row["case"])
    if not has_curve:
        plt.close()
        return
    plt.xlabel("Chunk Index")
    plt.ylabel("Acceptance Rate")
    plt.title("Spec Acceptance Rate Series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot streaming mini spec matrix results.")
    parser.add_argument("--input-root", type=Path, required=True,
                        help="run_streaming_mini_spec_nsys.sh 生成的根目录。")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="图表与汇总输出路径。")
    parser.add_argument("--downsample-stride", type=int, default=1,
                        help="对 chunk/acceptance 序列做步长平均，缓解图像过密。")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for case_dir in sorted(p for p in args.input_root.iterdir() if p.is_dir()):
        logs_dir = case_dir / "mini_spec_logs"
        if logs_dir.exists():
            rows.append(extract_metrics(case_dir))

    if not rows:
        raise SystemExit(f"No mini_spec logs found under {args.input_root}")

    df = pd.DataFrame(rows)
    summary_json = output_dir / "summary_metrics.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Throughput comparison
    plot_bar(df, "case", "throughput", "Throughput (tokens/s)", "tokens/s",
             output_dir / "throughput.png")
    # Average latency
    plot_bar(df, "case", "avg_latency", "Average Latency (s/request)",
             "seconds", output_dir / "avg_latency.png")
    # Total decode time
    plot_bar(df, "case", "total_decode_time", "Total Decode Time (s)",
             "seconds", output_dir / "total_decode_time.png")

    # Spec-only metrics: acceptance rate, avg chunk/decode time
    spec_df = df[df["run_mode"] == "spec"].copy()
    if not spec_df.empty:
        if spec_df["acceptance_rate"].notna().any():
            plot_bar(spec_df, "case", "acceptance_rate",
                     "Spec Acceptance Rate", "acceptance rate",
                     output_dir / "spec_acceptance_rate.png")
        multi_cols = [c for c in ["avg_chunk_time", "avg_decode_time_per_chunk"] if c in spec_df]
        if multi_cols:
            plot_multi_bars(spec_df, multi_cols,
                            "Spec Chunk vs Decode Time (avg per chunk)",
                            output_dir / "spec_chunk_decode_compare.png")

    # Prompt vs generated tokens comparison
    token_cols = [col for col in ["prompt_tokens", "generated_tokens"] if col in df]
    if len(token_cols) == 2:
        plot_multi_bars(df, token_cols, "Prompt vs Generated Tokens",
                        output_dir / "tokens_prompt_vs_generated.png")
        plot_multi_lines(df, token_cols, "Prompt vs Generated Tokens (Lines)",
                         "Tokens", output_dir / "tokens_prompt_vs_generated_lines.png")

    # Spec-only: accepted vs proposed
    if not spec_df.empty and spec_df[["accepted_tokens", "proposed_tokens"]].notna().any().any():
        plot_multi_bars(spec_df, ["accepted_tokens", "proposed_tokens"],
                        "Spec Accepted vs Proposed Tokens",
                        output_dir / "spec_accept_proposed.png")
        plot_multi_lines(spec_df, ["accepted_tokens", "proposed_tokens"],
                         "Spec Accepted vs Proposed Tokens (Lines)",
                         "Tokens", output_dir / "spec_accept_proposed_lines.png")

    # Spec chunk timing comparisons
    spec_chunk_cols = [c for c in ["avg_add_time_per_chunk",
                                   "avg_decode_time_per_chunk",
                                   "avg_chunk_time"] if c in spec_df]
    if spec_chunk_cols:
        plot_multi_bars(spec_df, spec_chunk_cols,
                        "Spec Avg Chunk/Add/Decode Time",
                        output_dir / "spec_chunk_breakdown.png")

    stride = max(1, args.downsample_stride)

    # Chunk-level multi-line figures
    if not spec_df.empty:
        plot_chunk_series_multi(
            spec_df,
            columns=["chunk_series", "chunk_add_times", "chunk_decode_times"],
            title="Spec Chunk/Add/Decode Time Series",
            ylabel="Time (s)",
            output_path=output_dir / "spec_chunk_series_all.png",
            stride=stride,
        )
        plot_acceptance_series(spec_df,
                               output_path=output_dir / "spec_accept_series.png",
                               stride=stride)
        if spec_df[["accepted_tokens", "proposed_tokens"]].notna().any().any():
            plot_chunk_series_multi(
                spec_df,
                columns=["chunk_series"],
                title="Spec Chunk Time Series (Per Case)",
                ylabel="Chunk Time (s)",
                output_path=output_dir / "spec_chunk_time_lines.png",
                stride=stride,
            )
        if spec_df["acceptance_rate"].notna().any():
            plot_multi_lines(
                spec_df,
                ["acceptance_rate"],
                "Spec Acceptance Rate (per case)",
                "Acceptance Rate",
                output_dir / "spec_acceptance_rate_lines.png",
            )
    baseline_df = df[df["run_mode"] == "baseline"].copy()
    if not baseline_df.empty:
        plot_chunk_series_multi(
            baseline_df,
            columns=["chunk_series"],
            title="Baseline Chunk Time Series",
            ylabel="Chunk Time (s)",
            output_path=output_dir / "baseline_chunk_series_all.png",
            stride=stride,
        )

    # Mixed metrics vs chunk index
    plot_multi_lines(
        df,
        ["throughput", "avg_latency", "generated_tokens"],
        "Throughput / Latency / Output Tokens",
        "Value",
        output_dir / "throughput_latency_output_lines.png",
    )

    print(f"[INFO] Summary saved to {summary_json}")
    print(f"[INFO] Plots saved under {output_dir}")


if __name__ == "__main__":
    main()
