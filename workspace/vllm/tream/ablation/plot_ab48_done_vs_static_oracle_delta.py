#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DONE_RE = re.compile(r"\] DONE (\S+) rc=0")
CASE_RE = re.compile(r"_([A-F]\d{2})_")
CTX_INF_RE = re.compile(r"_c(\d+)_i(\d+)_ib")
TS_RE = re.compile(r"_(20\d{12})_")


@dataclass
class DynSeries:
    run_name: str
    case_id: str
    steps: List[int]
    ctx: List[int]
    inf: List[int]


def _to_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _extract_done_runs(status_log: Path) -> List[str]:
    out: List[str] = []
    with status_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = DONE_RE.search(line)
            if m:
                out.append(m.group(1))
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _parse_case_ctx_inf(name: str) -> Optional[Tuple[str, int, int]]:
    m1 = CASE_RE.search(name)
    m2 = CTX_INF_RE.search(name)
    if not (m1 and m2):
        return None
    return m1.group(1), int(m2.group(1)), int(m2.group(2))


def _parse_ts(name: str) -> str:
    m = TS_RE.search(name)
    return m.group(1) if m else ""


def _collect_latest_static_runs(patterns: Sequence[str]) -> Dict[str, Dict[str, object]]:
    chosen: Dict[str, Path] = {}
    ts_map: Dict[str, str] = {}
    for pat in patterns:
        for p in sorted(Path("/").glob(pat.lstrip("/"))):
            parsed = _parse_case_ctx_inf(p.name)
            if not parsed:
                continue
            case_id, _, _ = parsed
            ts = _parse_ts(p.name)
            if case_id not in chosen or ts >= ts_map[case_id]:
                chosen[case_id] = p
                ts_map[case_id] = ts

    out: Dict[str, Dict[str, object]] = {}
    for case_id, p in chosen.items():
        parsed = _parse_case_ctx_inf(p.name)
        if not parsed:
            continue
        _, ctx, inf = parsed
        lat: List[float] = []
        acc: List[float] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("event") != "inference_cycle":
                    continue
                l = obj.get("latency")
                a = obj.get("accuracy")
                if isinstance(l, (int, float)) and isinstance(a, (int, float)):
                    lf = float(l)
                    af = float(a)
                    if math.isfinite(lf) and math.isfinite(af):
                        lat.append(lf)
                        acc.append(af)
        if not lat:
            continue
        out[case_id] = {
            "ctx": ctx,
            "inf": inf,
            "lat": np.asarray(lat, dtype=float),
            "acc": np.asarray(acc, dtype=float),
            "file": str(p),
            "ts": ts_map[case_id],
        }
    return out


def _oracle_from_static(
    static_runs: Dict[str, Dict[str, object]],
    num_steps: int,
    tau: float,
) -> Tuple[List[int], List[int]]:
    lengths = [int(v["lat"].shape[0]) for v in static_runs.values()]
    n = min(lengths) if lengths else 0
    if n <= 0:
        return [1] * num_steps, [1] * num_steps

    edges = np.linspace(0, n, num_steps + 1, dtype=int)
    items = list(static_runs.values())
    out_ctx: List[int] = []
    out_inf: List[int] = []

    for s in range(num_steps):
        a, b = int(edges[s]), int(edges[s + 1])
        if b <= a:
            b = min(n, a + 1)
        cand = []
        for item in items:
            lat = float(np.mean(item["lat"][a:b]))
            acc = float(np.mean(item["acc"][a:b]))
            cand.append((lat, acc, int(item["ctx"]), int(item["inf"])))
        feas = [x for x in cand if x[1] >= tau]
        pool = feas if feas else cand
        best = min(pool, key=lambda x: (x[0], -x[1]))
        out_ctx.append(best[2])
        out_inf.append(best[3])
    return out_ctx, out_inf


def _load_dyn_series(inference_logs_dir: Path, run_name: str) -> Optional[DynSeries]:
    p = inference_logs_dir / f"scheduler_{run_name}.jsonl"
    if not p.exists():
        return None
    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
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

    case_m = CASE_RE.search(run_name)
    case_id = case_m.group(1) if case_m else run_name

    steps: List[int] = []
    ctx: List[int] = []
    inf: List[int] = []
    for r in rows:
        s = _to_int(r.get("scheduler/step"))
        if s is None:
            continue
        c = _to_int(r.get("scheduler/new_context_length"))
        i = _to_int(r.get("scheduler/new_inference_length"))
        if c is None:
            c = _to_int(r.get("scheduler/context_length")) or -1
        if i is None:
            i = _to_int(r.get("scheduler/inference_length")) or -1
        steps.append(s)
        ctx.append(c)
        inf.append(i)
    if not steps:
        return None
    return DynSeries(run_name=run_name, case_id=case_id, steps=steps, ctx=ctx, inf=inf)


def _resample_dyn(series: DynSeries, num_steps: int) -> Tuple[List[int], List[int]]:
    by_step = {s: (c, i) for s, c, i in zip(series.steps, series.ctx, series.inf)}
    max_step = max(by_step.keys())
    last = by_step[min(0, max_step)]
    out_c: List[int] = []
    out_i: List[int] = []
    for s in range(num_steps):
        q = min(s, max_step)
        if q in by_step:
            last = by_step[q]
        out_c.append(int(last[0]))
        out_i.append(int(last[1]))
    return out_c, out_i


def _plot_delta(
    case_id: str,
    run_name: str,
    dyn_ctx: List[int],
    dyn_inf: List[int],
    orc_ctx: List[int],
    orc_inf: List[int],
    out_png: Path,
) -> None:
    x = np.arange(len(dyn_ctx))
    dctx = np.asarray(dyn_ctx, dtype=float) - np.asarray(orc_ctx, dtype=float)
    dinf = np.asarray(dyn_inf, dtype=float) - np.asarray(orc_inf, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.2), sharex=True, constrained_layout=True)

    for ax, y, title, color in [
        (axes[0], dctx, "Δctx = our_ctx - oracle_ctx", "#1f77b4"),
        (axes[1], dinf, "Δinf = our_inf - oracle_inf", "#d62728"),
    ]:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
        ax.axhspan(-1.0, 1.0, color="#2ca02c", alpha=0.08)
        ax.plot(x, y, marker="o", markersize=3.3, linewidth=1.8, color=color)
        ax.grid(alpha=0.25)
        ax.set_ylabel(title)

    axes[1].set_xlabel("scheduler step")
    fig.suptitle(f"{case_id}: our schedule vs static oracle (delta view)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot done-case delta curves: (our ctx/inf) - (static oracle ctx/inf).")
    ap.add_argument("--status-log", required=True, type=Path)
    ap.add_argument("--inference-logs-dir", required=True, type=Path)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=41)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    static_patterns = [
        str(args.inference_logs_dir / "doh_shift_ab48x3_static_nodyn_ib32_tb16_*.jsonl"),
        str(args.inference_logs_dir / "20260325_ab48_3groups_static_dynamic_notokenshift_内容总结" / "doh_shift_ab48x3_static_nodyn_ib32_tb16_*.jsonl"),
    ]
    static_runs = _collect_latest_static_runs(static_patterns)
    if len(static_runs) < 8:
        raise SystemExit(f"Too few static runs found for oracle: {len(static_runs)}")

    done_runs = _extract_done_runs(args.status_log)
    if not done_runs:
        raise SystemExit(f"No DONE run in {args.status_log}")

    oracle_ctx, oracle_inf = _oracle_from_static(static_runs, args.steps, args.tau)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[str] = [
        "case,run,step,our_ctx,oracle_ctx,delta_ctx,our_inf,oracle_inf,delta_inf"
    ]
    case_rows: List[str] = [
        "case,run,mean_abs_delta_ctx,mean_abs_delta_inf,max_abs_delta_ctx,max_abs_delta_inf,out_png"
    ]

    plotted = 0
    for run_name in done_runs:
        dyn = _load_dyn_series(args.inference_logs_dir, run_name)
        if dyn is None:
            continue
        dctx, dinf = _resample_dyn(dyn, args.steps)
        out_png = args.out_dir / f"{dyn.case_id}_{run_name}_delta_vs_oracle_tau{str(args.tau).replace('.', 'p')}.png"
        _plot_delta(dyn.case_id, run_name, dctx, dinf, oracle_ctx, oracle_inf, out_png)
        plotted += 1

        arr_dctx = np.asarray(dctx, dtype=float) - np.asarray(oracle_ctx, dtype=float)
        arr_dinf = np.asarray(dinf, dtype=float) - np.asarray(oracle_inf, dtype=float)
        case_rows.append(
            ",".join(
                [
                    dyn.case_id,
                    run_name,
                    f"{float(np.mean(np.abs(arr_dctx))):.6f}",
                    f"{float(np.mean(np.abs(arr_dinf))):.6f}",
                    f"{float(np.max(np.abs(arr_dctx))):.6f}",
                    f"{float(np.max(np.abs(arr_dinf))):.6f}",
                    str(out_png),
                ]
            )
        )

        for s in range(args.steps):
            summary_rows.append(
                ",".join(
                    [
                        dyn.case_id,
                        run_name,
                        str(s),
                        str(int(dctx[s])),
                        str(int(oracle_ctx[s])),
                        str(int(dctx[s] - oracle_ctx[s])),
                        str(int(dinf[s])),
                        str(int(oracle_inf[s])),
                        str(int(dinf[s] - oracle_inf[s])),
                    ]
                )
            )

    summary_csv = args.out_dir / f"done_cases_delta_vs_oracle_tau{str(args.tau).replace('.', 'p')}.csv"
    summary_csv.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    case_csv = args.out_dir / f"done_cases_delta_metrics_tau{str(args.tau).replace('.', 'p')}.csv"
    case_csv.write_text("\n".join(case_rows) + "\n", encoding="utf-8")

    oracle_csv = args.out_dir / f"static_oracle_windowed_tau{str(args.tau).replace('.', 'p')}.csv"
    with oracle_csv.open("w", encoding="utf-8") as f:
        f.write("step,oracle_ctx,oracle_inf\n")
        for s in range(args.steps):
            f.write(f"{s},{oracle_ctx[s]},{oracle_inf[s]}\n")

    print(f"done_runs={len(done_runs)} plotted={plotted}")
    print(f"oracle_pool_cases={len(static_runs)} tau={args.tau}")
    print(f"summary_csv={summary_csv}")
    print(f"case_csv={case_csv}")
    print(f"oracle_csv={oracle_csv}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
