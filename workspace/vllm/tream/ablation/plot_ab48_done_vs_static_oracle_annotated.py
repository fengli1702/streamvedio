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
TS_RE = re.compile(r"_(20\d{6})_(\d{6})_")


@dataclass
class DynSeries:
    run_name: str
    case_id: str
    steps: List[int]
    ctx: List[int]
    inf: List[int]
    jsd: List[float]
    shock: List[float]
    decision: List[str]
    cold_active: List[bool]
    local_relax_active: List[bool]


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
    return (m.group(1) + m.group(2)) if m else ""


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
    jsd: List[float] = []
    shock: List[float] = []
    decision: List[str] = []
    cold_active: List[bool] = []
    local_relax_active: List[bool] = []

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
        jsd.append(_to_float(r.get("scheduler/jsd_mean")))
        shock.append(_to_float(r.get("scheduler/shock_score")))
        decision.append(str(r.get("scheduler/decision", "")))
        cold_active.append(_to_bool(r.get("scheduler/cold_start_active")))
        local_relax_active.append(_to_bool(r.get("scheduler/local_relax_active")))

    if not steps:
        return None
    return DynSeries(
        run_name=run_name,
        case_id=case_id,
        steps=steps,
        ctx=ctx,
        inf=inf,
        jsd=jsd,
        shock=shock,
        decision=decision,
        cold_active=cold_active,
        local_relax_active=local_relax_active,
    )


def _resample_dyn(series: DynSeries, num_steps: int) -> DynSeries:
    by_step = {
        s: (c, i, j, h, d, ca, la)
        for s, c, i, j, h, d, ca, la in zip(
            series.steps,
            series.ctx,
            series.inf,
            series.jsd,
            series.shock,
            series.decision,
            series.cold_active,
            series.local_relax_active,
        )
    }
    max_step = max(by_step.keys())
    init = by_step[min(0, max_step)]
    out_steps: List[int] = []
    out_ctx: List[int] = []
    out_inf: List[int] = []
    out_jsd: List[float] = []
    out_shock: List[float] = []
    out_dec: List[str] = []
    out_cold: List[bool] = []
    out_relax: List[bool] = []

    last = init
    for s in range(num_steps):
        q = min(s, max_step)
        if q in by_step:
            last = by_step[q]
        c, i, j, h, d, ca, la = last
        out_steps.append(s)
        out_ctx.append(int(c))
        out_inf.append(int(i))
        out_jsd.append(float(j))
        out_shock.append(float(h))
        out_dec.append(str(d))
        out_cold.append(bool(ca))
        out_relax.append(bool(la))

    return DynSeries(
        run_name=series.run_name,
        case_id=series.case_id,
        steps=out_steps,
        ctx=out_ctx,
        inf=out_inf,
        jsd=out_jsd,
        shock=out_shock,
        decision=out_dec,
        cold_active=out_cold,
        local_relax_active=out_relax,
    )


def _first_target_zone_step(ctx: List[int], inf: List[int], threshold: int = 3) -> Optional[int]:
    for s, (c, i) in enumerate(zip(ctx, inf)):
        if max(c, i) <= threshold:
            return s
    return None


def _first_cold_end_step(cold_active: List[bool]) -> Optional[int]:
    for s in range(1, len(cold_active)):
        if cold_active[s - 1] and not cold_active[s]:
            return s
    return None


def _first_jsd_probe_step(decisions: List[str], local_relax: List[bool]) -> Optional[int]:
    for s, d in enumerate(decisions):
        if "jsd_probe" in d:
            return s
    for s, lr in enumerate(local_relax):
        if lr:
            return s
    return None


def _plot_one(
    dyn: DynSeries,
    oracle_ctx: List[int],
    oracle_inf: List[int],
    out_png: Path,
    shock_low: float,
    shock_high: float,
) -> Dict[str, str]:
    x = np.asarray(dyn.steps, dtype=int)
    y_jsd = np.asarray(dyn.jsd, dtype=float)
    y_shock = np.asarray(dyn.shock, dtype=float)
    y_ctx = np.asarray(dyn.ctx, dtype=float)
    y_inf = np.asarray(dyn.inf, dtype=float)
    y_oc = np.asarray(oracle_ctx, dtype=float)
    y_oi = np.asarray(oracle_inf, dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
        constrained_layout=True,
    )
    ax0, ax1 = axes

    # top: jsd + shock
    ax0.plot(x, y_jsd, color="#2aa198", linewidth=2.0, marker="o", markersize=3.0, label="JSD mean")
    ax0.plot(x, y_shock, color="#9ecae1", linewidth=1.5, alpha=0.95, label="shock score")
    ax0.axhline(shock_low, color="#7f7f7f", linestyle="--", linewidth=1.1, label=f"shock_t_low={shock_low}")
    ax0.axhline(shock_high, color="#333333", linestyle="--", linewidth=1.1, label=f"shock_t_high={shock_high}")
    ax0.set_ylabel("JSD / shock")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper right", ncol=2, fontsize=9)
    ax0.set_title(f"{dyn.case_id}: JSD-driven scheduler (annotated, vs static oracle)")

    # bottom: dynamic ctx/inf + oracle ctx/inf
    ax1.axhspan(1, 3, color="#2ca02c", alpha=0.08, label="target zone <=3")
    ax1.plot(x, y_ctx, color="#1f77b4", linewidth=1.8, marker="o", markersize=2.5, label="dynamic ctx")
    ax1.plot(x, y_inf, color="#d62728", linewidth=1.8, marker="o", markersize=2.5, label="dynamic inf")
    ax1.plot(x, y_oc, color="#5fa2dd", linewidth=1.5, linestyle="--", label="oracle ctx")
    ax1.plot(x, y_oi, color="#ff9896", linewidth=1.5, linestyle="--", label="oracle inf")
    ax1.set_ylabel("ctx / inf")
    ax1.set_xlabel("scheduler step (window)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", ncol=2, fontsize=9)

    # annotations
    s_target = _first_target_zone_step(dyn.ctx, dyn.inf, threshold=3)
    s_cold_end = _first_cold_end_step(dyn.cold_active)
    s_probe = _first_jsd_probe_step(dyn.decision, dyn.local_relax_active)

    markers = [
        ("first_target_zone", s_target, "#2aa198"),
        ("cold_end", s_cold_end, "#ff7f0e"),
        ("first_jsd_probe", s_probe, "#66c2a5"),
    ]
    for name, step, color in markers:
        if step is None:
            continue
        for ax in (ax0, ax1):
            ax.axvline(step, color=color, linestyle=":", linewidth=1.1)
        ax1.annotate(
            name,
            xy=(step, 8.0),
            xytext=(step + 0.2, 8.35),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            color=color,
            fontsize=8.5,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "case_id": dyn.case_id,
        "run_name": dyn.run_name,
        "steps": str(len(dyn.steps)),
        "start_cfg": f"({dyn.ctx[0]},{dyn.inf[0]})",
        "end_cfg": f"({dyn.ctx[-1]},{dyn.inf[-1]})",
        "first_target_zone_step": str(s_target or ""),
        "cold_end_step": str(s_cold_end or ""),
        "first_jsd_probe_step": str(s_probe or ""),
        "shock_low": str(shock_low),
        "shock_high": str(shock_high),
        "out_png": str(out_png),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch plot C06-style annotated figures for DONE cases, with static oracle ctx/inf.")
    ap.add_argument("--status-log", required=True, type=Path)
    ap.add_argument("--inference-logs-dir", required=True, type=Path)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=41)
    ap.add_argument("--shock-low", type=float, default=0.2)
    ap.add_argument("--shock-high", type=float, default=0.35)
    ap.add_argument(
        "--oracle-fixed-csv",
        type=Path,
        default=None,
        help="Optional fixed oracle csv/txt with columns step,oracle_ctx,oracle_inf. If set, skip recomputation.",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    infer_logs_dir = args.inference_logs_dir.resolve()
    done_runs = _extract_done_runs(args.status_log.resolve())
    if not done_runs:
        raise SystemExit(f"No DONE run in {args.status_log}")

    static_runs: Dict[str, Dict[str, object]] = {}
    if args.oracle_fixed_csv is not None:
        oracle_ctx: List[int] = []
        oracle_inf: List[int] = []
        with args.oracle_fixed_csv.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
            h = {name: idx for idx, name in enumerate(header)}
            if "oracle_ctx" not in h or "oracle_inf" not in h:
                raise SystemExit(f"oracle_fixed_csv missing columns oracle_ctx/oracle_inf: {args.oracle_fixed_csv}")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                try:
                    oracle_ctx.append(int(parts[h["oracle_ctx"]]))
                    oracle_inf.append(int(parts[h["oracle_inf"]]))
                except Exception:
                    continue
        if len(oracle_ctx) < args.steps:
            raise SystemExit(
                f"oracle_fixed_csv rows too short: {len(oracle_ctx)} < steps={args.steps}, file={args.oracle_fixed_csv}"
            )
        oracle_ctx = oracle_ctx[: args.steps]
        oracle_inf = oracle_inf[: args.steps]
    else:
        static_patterns = [
            str(infer_logs_dir / "doh_shift_ab48x3_static_nodyn_ib32_tb16_*.jsonl"),
            str(infer_logs_dir / "20260325_ab48_3groups_static_dynamic_notokenshift_内容总结" / "doh_shift_ab48x3_static_nodyn_ib32_tb16_*.jsonl"),
        ]
        static_runs = _collect_latest_static_runs(static_patterns)
        if len(static_runs) < 8:
            raise SystemExit(f"Too few static runs for oracle: {len(static_runs)}")
        oracle_ctx, oracle_inf = _oracle_from_static(static_runs, args.steps, args.tau)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, str]] = []
    detail_lines = [
        "case,run,step,dyn_ctx,dyn_inf,oracle_ctx,oracle_inf,delta_ctx,delta_inf,jsd,shock,decision"
    ]
    plotted = 0
    for run_name in done_runs:
        dyn = _load_dyn_series(infer_logs_dir, run_name)
        if dyn is None:
            continue
        dyn = _resample_dyn(dyn, args.steps)
        out_png = args.out_dir / f"{dyn.case_id}_{run_name}_annotated_vs_oracle_tau{str(args.tau).replace('.', 'p')}.png"
        summary.append(_plot_one(dyn, oracle_ctx, oracle_inf, out_png, args.shock_low, args.shock_high))
        plotted += 1

        for s in range(args.steps):
            detail_lines.append(
                ",".join(
                    [
                        dyn.case_id,
                        dyn.run_name,
                        str(s),
                        str(dyn.ctx[s]),
                        str(dyn.inf[s]),
                        str(oracle_ctx[s]),
                        str(oracle_inf[s]),
                        str(dyn.ctx[s] - oracle_ctx[s]),
                        str(dyn.inf[s] - oracle_inf[s]),
                        f"{dyn.jsd[s]:.6f}" if math.isfinite(dyn.jsd[s]) else "",
                        f"{dyn.shock[s]:.6f}" if math.isfinite(dyn.shock[s]) else "",
                        dyn.decision[s].replace(",", ";"),
                    ]
                )
            )

    summary_csv = args.out_dir / "done_cases_annotated_vs_oracle_summary.csv"
    if summary:
        keys = list(summary[0].keys())
        with summary_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in summary:
                f.write(",".join(row.get(k, "") for k in keys) + "\n")

    detail_csv = args.out_dir / "done_cases_vs_oracle_step_detail.csv"
    detail_csv.write_text("\n".join(detail_lines) + "\n", encoding="utf-8")

    oracle_csv = args.out_dir / f"static_oracle_windowed_tau{str(args.tau).replace('.', 'p')}.csv"
    with oracle_csv.open("w", encoding="utf-8") as f:
        f.write("step,oracle_ctx,oracle_inf\n")
        for s in range(args.steps):
            f.write(f"{s},{oracle_ctx[s]},{oracle_inf[s]}\n")

    print(f"done_runs={len(done_runs)} plotted={plotted}")
    if args.oracle_fixed_csv is not None:
        print(f"oracle_source=fixed_csv file={args.oracle_fixed_csv}")
    else:
        print(f"oracle_pool_cases={len(static_runs)} tau={args.tau}")
    print(f"out_dir={args.out_dir}")
    print(f"summary_csv={summary_csv}")
    print(f"detail_csv={detail_csv}")
    print(f"oracle_csv={oracle_csv}")


if __name__ == "__main__":
    main()
