#!/usr/bin/env python3
"""
Build 4 plots from local W&B logs:
1) latency_mean bar (nospec/spec x static/dyn)
2) accuracy (global_acc) bar
3) scatter: spec_dyn vs nospec_dyn (delta latency vs delta acc)
4) scatter: spec_dyn vs spec_static (delta latency vs delta acc)

Static runs are reused from 12-20 clean; dynamic from 1-6.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_config_value(val):
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


def load_config_values(path: Path) -> Dict[str, object]:
    """Minimal parser for wandb config.yaml (key -> value)."""
    values: Dict[str, object] = {}
    current_key: Optional[str] = None
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.endswith(":") and not line.startswith(" "):
            current_key = line[:-1].strip()
            continue
        if current_key and line.strip().startswith("value:"):
            val = line.split(":", 1)[1].strip()
            # basic type coercion
            if val.lower() in ("true", "false"):
                parsed: object = val.lower() == "true"
            else:
                try:
                    if "." in val:
                        parsed = float(val)
                    else:
                        parsed = int(val)
                except Exception:
                    parsed = val.strip("'\"")
            values[current_key] = parsed
            current_key = None
    return values


def load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8").strip())
    except Exception:
        return {}


def load_history_latency(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    keys = ["inference/latency", "inference/latency_mean", "scheduler/latency_mean"]
    values: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for key in keys:
                if key in rec and rec[key] is not None:
                    try:
                        values.append(float(rec[key]))
                        break
                    except Exception:
                        continue
    if values:
        return float(np.mean(values))
    return None


def pick_acc(summary: Dict[str, object], history_path: Path) -> Optional[float]:
    acc_keys = [
        "inference/global_accuracy",
        "inference/global_acc",
        "inference/globalAccuracy",
        "global_acc",
        "global_accuracy",
    ]
    for key in acc_keys:
        if key in summary and summary[key] is not None:
            try:
                return float(summary[key])
            except Exception:
                continue
    # fallback: last in history
    if history_path.exists():
        last_val: Optional[float] = None
        with history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for key in acc_keys:
                    if key in rec and rec[key] is not None:
                        try:
                            last_val = float(rec[key])
                        except Exception:
                            pass
        return last_val
    return None


def bool_from(val: object) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
    return None


def classify_method(run_name: str, cfg: Dict[str, object]) -> Tuple[bool, bool]:
    # returns (spec, dynamic)
    spec = bool_from(cfg.get("use_speculative_decoding"))
    if spec is None:
        spec = "_spec_" in run_name
    dyn_flag = bool_from(cfg.get("disable_dynamic_scheduling"))
    if dyn_flag is None:
        dynamic = "_dyn" in run_name
    else:
        dynamic = not dyn_flag
    return bool(spec), bool(dynamic)


def scan_runs(wandb_dir: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for run_dir in sorted(wandb_dir.glob("run-*")):
        files_dir = run_dir / "files"
        cfg_path = files_dir / "config.yaml"
        summary_path = files_dir / "wandb-summary.json"
        history_dir = files_dir / "wandb-history.jsonl"
        cfg = load_config_values(cfg_path)
        run_name = cfg.get("wandb_run_name")
        if not run_name:
            continue
        summary = load_summary(summary_path)
        latency_mean = load_history_latency(history_dir)
        if latency_mean is None:
            for key in ("inference/latency", "scheduler/latency_mean"):
                if key in summary and summary[key] is not None:
                    try:
                        latency_mean = float(summary[key])
                        break
                    except Exception:
                        pass
        acc = pick_acc(summary, history_dir)
        ctx = cfg.get("context_length")
        inf = cfg.get("inference_length")
        if ctx is None or inf is None:
            continue
        spec, dynamic = classify_method(str(run_name), cfg)
        runs.append({
            "run_name": str(run_name),
            "context_length": int(ctx),
            "inference_length": int(inf),
            "spec": spec,
            "dynamic": dynamic,
            "latency_mean": latency_mean,
            "global_acc": acc,
        })
    return runs


def filter_runs(runs: List[Dict[str, object]], prefixes: Iterable[str]) -> List[Dict[str, object]]:
    pref = tuple(prefixes)
    if not pref:
        return runs
    out = []
    for r in runs:
        name = r["run_name"]
        if any(name.startswith(p) for p in pref):
            out.append(r)
    return out


def build_table(runs: List[Dict[str, object]]) -> Dict[Tuple[int, int], Dict[str, Dict[str, float]]]:
    table: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
    for r in runs:
        ctx = r["context_length"]
        inf = r["inference_length"]
        method = ("spec" if r["spec"] else "nospec") + ("_dyn" if r["dynamic"] else "_static")
        key = (ctx, inf)
        table.setdefault(key, {})[method] = {
            "latency_mean": r.get("latency_mean"),
            "global_acc": r.get("global_acc"),
        }
    return table


def plot_bars(table: Dict[Tuple[int, int], Dict[str, Dict[str, float]]],
              metric: str,
              title: str,
              out_path: Path) -> None:
    methods = ["nospec_static", "nospec_dyn", "spec_static", "spec_dyn"]
    labels = [f"{c}/{i}" for c, i in sorted(table.keys())]
    values = {m: [] for m in methods}
    for key in sorted(table.keys()):
        data = table[key]
        for m in methods:
            val = data.get(m, {}).get(metric)
            values[m].append(val if val is not None else np.nan)

    x = np.arange(len(labels))
    width = 0.18
    colors = {
        "nospec_static": "#4C72B0",
        "nospec_dyn": "#2C5D9B",
        "spec_static": "#DD8452",
        "spec_dyn": "#E15759",
    }

    plt.figure(figsize=(12, 5))
    for idx, m in enumerate(methods):
        plt.bar(x + idx * width, values[m], width, label=m, color=colors[m])
    plt.xticks(x + width * 1.5, labels, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(table: Dict[Tuple[int, int], Dict[str, Dict[str, float]]],
                 method_a: str,
                 method_b: str,
                 title: str,
                 out_path: Path) -> None:
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for (ctx, inf), data in sorted(table.items()):
        a = data.get(method_a)
        b = data.get(method_b)
        if not a or not b:
            continue
        lat_a = a.get("latency_mean")
        lat_b = b.get("latency_mean")
        acc_a = a.get("global_acc")
        acc_b = b.get("global_acc")
        if None in (lat_a, lat_b, acc_a, acc_b):
            continue
        xs.append(float(lat_a) - float(lat_b))
        ys.append(float(acc_a) - float(acc_b))
        labels.append(f"{ctx}/{inf}")

    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, color="#E15759" if "spec" in method_a else "#4C72B0")
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab, fontsize=8)
    plt.axhline(0.0, color="#666", linestyle="--", linewidth=1)
    plt.axvline(0.0, color="#666", linestyle="--", linewidth=1)
    plt.xlabel(f"Delta latency ({method_a} - {method_b})")
    plt.ylabel(f"Delta accuracy ({method_a} - {method_b})")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stream results from local W&B logs")
    parser.add_argument("--wandb-dir", type=str, default=None, help="Path to local wandb runs")
    parser.add_argument("--dyn-prefix", type=str, default="1-6", help="Run name prefix for dynamic runs")
    parser.add_argument("--static-prefix", type=str, default="12-20-clean+clean1_base", help="Run name prefix for static runs")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    tream_dir = script_dir.parent
    wandb_dir = Path(args.wandb_dir) if args.wandb_dir else (tream_dir / "wandb")
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(wandb_dir)
    dyn_runs = filter_runs([r for r in runs if r["dynamic"]], [args.dyn_prefix])
    static_runs = filter_runs([r for r in runs if not r["dynamic"]], [args.static_prefix])
    all_runs = dyn_runs + static_runs
    table = build_table(all_runs)

    plot_bars(
        table,
        "latency_mean",
        "latency_mean: static vs dynamic, spec vs nospec",
        out_dir / "latency_mean_bar.png",
    )
    plot_bars(
        table,
        "global_acc",
        "accuracy (global_acc): static vs dynamic, spec vs nospec",
        out_dir / "accuracy_bar.png",
    )
    plot_scatter(
        table,
        "spec_dyn",
        "nospec_dyn",
        "Spec Dyn vs Nospec Dyn: Delta latency vs Delta accuracy",
        out_dir / "scatter_specdyn_vs_nospecdyn.png",
    )
    plot_scatter(
        table,
        "spec_dyn",
        "spec_static",
        "Spec Dyn vs Spec Static: Delta latency vs Delta accuracy",
        out_dir / "scatter_specdyn_vs_specstatic.png",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
