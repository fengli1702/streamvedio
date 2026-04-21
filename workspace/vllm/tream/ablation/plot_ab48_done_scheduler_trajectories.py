#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DONE_RE = re.compile(r"\] DONE (\S+) rc=0")


@dataclass
class CaseSeries:
    run_name: str
    case_id: str
    steps: List[int]
    ctx: List[int]
    inf: List[int]
    jsd: List[float]
    drift_th: List[float]
    warmup: List[bool]
    cold: List[bool]
    local_relax: List[bool]
    phase: List[str]


def _to_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "on"}
    return False


def _extract_done_runs(status_log: Path) -> List[str]:
    runs: List[str] = []
    with status_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = DONE_RE.search(line)
            if m:
                runs.append(m.group(1))
    # preserve order, de-duplicate
    seen = set()
    uniq: List[str] = []
    for r in runs:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _load_case_series(inference_logs: Path, run_name: str) -> Optional[CaseSeries]:
    sched = inference_logs / f"scheduler_{run_name}.jsonl"
    if not sched.exists():
        return None
    rows: List[Dict] = []
    with sched.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None

    rows.sort(key=lambda x: _to_int(x.get("scheduler/step")) or 0)

    m = re.search(r"_([A-Z]\d{2})_\d{8}_\d{6}_w[A-Z]_c\d+_i\d+_", run_name)
    case_id = m.group(1) if m else run_name

    steps: List[int] = []
    ctx: List[int] = []
    inf: List[int] = []
    jsd: List[float] = []
    drift_th: List[float] = []
    warmup: List[bool] = []
    cold: List[bool] = []
    local_relax: List[bool] = []
    phase: List[str] = []

    for r in rows:
        s = _to_int(r.get("scheduler/step"))
        if s is None:
            continue
        c = _to_int(r.get("scheduler/new_context_length"))
        i = _to_int(r.get("scheduler/new_inference_length"))
        if c is None:
            c = _to_int(r.get("scheduler/context_length")) or 0
        if i is None:
            i = _to_int(r.get("scheduler/inference_length")) or 0

        steps.append(s)
        ctx.append(c)
        inf.append(i)
        jsd.append(_to_float(r.get("scheduler/jsd_mean")))
        drift_th.append(_to_float(r.get("scheduler/local_relax_drift_threshold")))
        warmup.append(_to_bool(r.get("scheduler/warmup_active")))
        cold.append(_to_bool(r.get("scheduler/cold_start_active")))
        local_relax.append(_to_bool(r.get("scheduler/local_relax_active")))
        phase.append(str(r.get("scheduler/phase", "")))

    if not steps:
        return None
    return CaseSeries(
        run_name=run_name,
        case_id=case_id,
        steps=steps,
        ctx=ctx,
        inf=inf,
        jsd=jsd,
        drift_th=drift_th,
        warmup=warmup,
        cold=cold,
        local_relax=local_relax,
        phase=phase,
    )


def _first_fall_true_to_false(steps: List[int], flags: List[bool]) -> Optional[int]:
    for i in range(1, len(flags)):
        if flags[i - 1] and not flags[i]:
            return steps[i]
    return None


def _first_true(steps: List[int], flags: List[bool]) -> Optional[int]:
    for s, f in zip(steps, flags):
        if f:
            return s
    return None


def _first_phase(steps: List[int], phases: List[str], target: str) -> Optional[int]:
    for s, p in zip(steps, phases):
        if p == target:
            return s
    return None


def _safe_median(vals: List[float], default: float = 0.2) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return default
    return float(np.median(arr))


def _plot_one(case: CaseSeries, out_png: Path) -> Dict[str, str]:
    x = np.asarray(case.steps, dtype=int)
    y_ctx = np.asarray(case.ctx, dtype=float)
    y_inf = np.asarray(case.inf, dtype=float)
    y_jsd = np.asarray(case.jsd, dtype=float)
    th = _safe_median(case.drift_th, default=0.2)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.2]},
    )

    ax0, ax1 = axes
    ax0.plot(x, y_ctx, color="#1f77b4", marker="o", linewidth=2.1, markersize=3.5, label="ctx")
    ax0.plot(x, y_inf, color="#d62728", marker="o", linewidth=2.1, markersize=3.5, label="inf")
    ax0.set_ylabel("ctx / inf")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")
    ax0.set_title(f"{case.case_id}: scheduler trajectory (ctx/inf) + JSD")

    ax1.plot(x, y_jsd, color="#2ca02c", marker="o", linewidth=1.9, markersize=3.2, label="JSD")
    ax1.axhline(th, color="#9467bd", linestyle="--", linewidth=1.6, label=f"JSD threshold ~ {th:.3f}")
    ax1.set_xlabel("scheduler step")
    ax1.set_ylabel("JSD")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    marks: List[Tuple[str, Optional[int], str]] = [
        ("warmup_end", _first_fall_true_to_false(case.steps, case.warmup), "#7f7f7f"),
        ("cold_end", _first_fall_true_to_false(case.steps, case.cold), "#ff7f0e"),
        ("local_relax_on", _first_true(case.steps, case.local_relax), "#8c564b"),
        ("reroute_phase", _first_phase(case.steps, case.phase, "SHOCK_LOCAL_REROUTE"), "#17becf"),
    ]
    for name, s, color in marks:
        if s is None:
            continue
        for ax in (ax0, ax1):
            ax.axvline(s, color=color, linestyle=":", linewidth=1.25, alpha=0.95)
        ax1.text(
            s,
            ax1.get_ylim()[1] * 0.96,
            name,
            color=color,
            fontsize=8.5,
            rotation=90,
            ha="right",
            va="top",
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "case_id": case.case_id,
        "run_name": case.run_name,
        "steps": str(len(case.steps)),
        "start_cfg": f"({case.ctx[0]},{case.inf[0]})",
        "end_cfg": f"({case.ctx[-1]},{case.inf[-1]})",
        "warmup_end_step": str(_first_fall_true_to_false(case.steps, case.warmup) or ""),
        "cold_end_step": str(_first_fall_true_to_false(case.steps, case.cold) or ""),
        "local_relax_on_step": str(_first_true(case.steps, case.local_relax) or ""),
        "reroute_phase_step": str(_first_phase(case.steps, case.phase, "SHOCK_LOCAL_REROUTE") or ""),
        "jsd_threshold": f"{th:.6f}",
        "png": str(out_png),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot C06-style trajectories for all DONE cases in one AB48 status log.")
    ap.add_argument("--status-log", required=True, type=Path)
    ap.add_argument("--inference-logs-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    done_runs = _extract_done_runs(args.status_log)
    if not done_runs:
        raise SystemExit(f"No DONE runs found in {args.status_log}")

    summaries: List[Dict[str, str]] = []
    plotted = 0
    for run_name in done_runs:
        case = _load_case_series(args.inference_logs_dir, run_name)
        if case is None:
            continue
        out_png = args.out_dir / f"{case.case_id}_{run_name}_ctx_inf_jsd_annotated.png"
        summaries.append(_plot_one(case, out_png))
        plotted += 1

    if not summaries:
        raise SystemExit("No plottable scheduler jsonl found for DONE runs.")

    csv_path = args.out_dir / "done_cases_ctx_inf_jsd_annotated_summary.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(summaries[0].keys())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in summaries:
            f.write(",".join(row.get(k, "") for k in keys) + "\n")

    print(f"done_runs={len(done_runs)} plotted={plotted}")
    print(f"summary_csv={csv_path}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
