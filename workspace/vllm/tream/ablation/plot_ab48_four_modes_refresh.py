#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INFER_LOGS = Path("/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs")
ARCHIVE_20260325 = INFER_LOGS / "20260325_ab48_3groups_static_dynamic_notokenshift_内容总结"

CASE_RE = re.compile(r"_(?P<case>[A-F]\d{2})_")
CTX_INF_RE = re.compile(r"_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib")
TS_RE = re.compile(r"_(20\d{12})_")


def case_order() -> List[str]:
    return [f"{g}{i:02d}" for g in "ABCDEF" for i in range(1, 9)]


def parse_case_ctx_inf(name: str) -> Optional[Tuple[str, int, int]]:
    m1 = CASE_RE.search(name)
    m2 = CTX_INF_RE.search(name)
    if not (m1 and m2):
        return None
    return m1.group("case"), int(m2.group("ctx")), int(m2.group("inf"))


def parse_ts(name: str) -> str:
    m = TS_RE.search(name)
    return m.group(1) if m else ""


def mean_lat_acc(path: Path) -> Optional[Tuple[float, float, int]]:
    lat: List[float] = []
    acc: List[float] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                l = obj.get("latency")
                a = obj.get("accuracy")
                if isinstance(l, (int, float)) and isinstance(a, (int, float)):
                    lat.append(float(l))
                    acc.append(float(a))
    except FileNotFoundError:
        return None
    if not lat or not acc:
        return None
    return float(np.mean(lat)), float(np.mean(acc)), len(lat)


def collect(glob_patterns: List[str]) -> Dict[str, Dict[str, object]]:
    chosen: Dict[str, Path] = {}
    ts_map: Dict[str, str] = {}
    for pat in glob_patterns:
        for p in sorted(Path("/").glob(pat.lstrip("/"))):
            parsed = parse_case_ctx_inf(p.name)
            if not parsed:
                continue
            case, _, _ = parsed
            ts = parse_ts(p.name)
            if case not in chosen or ts >= ts_map[case]:
                chosen[case] = p
                ts_map[case] = ts

    out: Dict[str, Dict[str, object]] = {}
    for case, p in chosen.items():
        parsed = parse_case_ctx_inf(p.name)
        if not parsed:
            continue
        _, ctx, inf = parsed
        ma = mean_lat_acc(p)
        if ma is None:
            continue
        lat, acc, n = ma
        out[case] = {
            "ctx": ctx,
            "inf": inf,
            "lat": lat,
            "acc": acc,
            "n": n,
            "file": str(p),
            "ts": ts_map[case],
        }
    return out


def fmt(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


def quadrant_stats(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Tuple[int, float]]:
    n = len(xs)
    if n == 0:
        return {"Q1": (0, 0.0), "Q2": (0, 0.0), "Q3": (0, 0.0), "Q4": (0, 0.0)}
    q1 = int(np.sum((xs > 0) & (ys > 0)))
    q2 = int(np.sum((xs < 0) & (ys > 0)))
    q3 = int(np.sum((xs < 0) & (ys < 0)))
    q4 = int(np.sum((xs > 0) & (ys < 0)))
    return {
        "Q1": (q1, 100.0 * q1 / n),
        "Q2": (q2, 100.0 * q2 / n),
        "Q3": (q3, 100.0 * q3 / n),
        "Q4": (q4, 100.0 * q4 / n),
    }


def plot_scatter_with_quadrants(
    out_png: Path,
    out_md: Optional[Path],
    title: str,
    xlabel: str,
    ylabel: str,
    xs: np.ndarray,
    ys: np.ndarray,
    color: str = "#66c2a5",
) -> Dict[str, Tuple[int, float]]:
    q = quadrant_stats(xs, ys)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.scatter(xs, ys, s=70, color=color, alpha=0.9)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.5)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    xr = max(1e-6, xmax - xmin)
    yr = max(1e-6, ymax - ymin)
    n = len(xs)
    ax.text(xmin + 0.03 * xr, ymax - 0.08 * yr, f"Q2: {q['Q2'][0]}/{n} ({q['Q2'][1]:.1f}%)")
    ax.text(xmax - 0.25 * xr, ymax - 0.08 * yr, f"Q1: {q['Q1'][0]}/{n} ({q['Q1'][1]:.1f}%)")
    ax.text(xmin + 0.03 * xr, ymin + 0.05 * yr, f"Q3: {q['Q3'][0]}/{n} ({q['Q3'][1]:.1f}%)")
    ax.text(xmax - 0.25 * xr, ymin + 0.05 * yr, f"Q4: {q['Q4'][0]}/{n} ({q['Q4'][1]:.1f}%)")

    fig.savefig(out_png, dpi=240)
    plt.close(fig)

    if out_md is not None:
        with out_md.open("w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"cases={n}\n\n")
            f.write(f"Q1: {q['Q1'][0]}/{n} ({q['Q1'][1]:.1f}%)\n")
            f.write(f"Q2: {q['Q2'][0]}/{n} ({q['Q2'][1]:.1f}%)\n")
            f.write(f"Q3: {q['Q3'][0]}/{n} ({q['Q3'][1]:.1f}%)\n")
            f.write(f"Q4: {q['Q4'][0]}/{n} ({q['Q4'][1]:.1f}%)\n\n")
            f.write(f"mean_x={float(np.mean(xs)):.6f}\n")
            f.write(f"mean_y={float(np.mean(ys)):.6f}\n")

    return q


def main() -> None:
    no_feature = collect([
        str(INFER_LOGS / "doh_shift_ab48x3_static_nodyn_nospec_ib32_tb16_*.jsonl"),
    ])
    spec_only = collect([
        str(INFER_LOGS / "doh_shift_ab48x3_static_nodyn_ib32_tb16_*.jsonl"),
    ])
    spec_dyn_nojsd = collect([
        str(INFER_LOGS / "doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_*.jsonl"),
        str(ARCHIVE_20260325 / "doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_*.jsonl"),
    ])
    spec_dyn_jsd = collect([
        str(INFER_LOGS / "doh_shift_ab48x3_dyn_wb1b12j2_jsd1td0_cw15_*.jsonl"),
    ])

    groups = {
        "no_feature": no_feature,
        "spec_only": spec_only,
        "spec_dyn_nojsd": spec_dyn_nojsd,
        "spec_dyn_jsd": spec_dyn_jsd,
    }

    ordered = case_order()
    rows: List[Dict[str, object]] = []
    for c in ordered:
        base = spec_only.get(c) or spec_dyn_jsd.get(c) or spec_dyn_nojsd.get(c) or no_feature.get(c)
        if not base:
            continue
        row: Dict[str, object] = {"case": c, "ctx": int(base["ctx"]), "inf": int(base["inf"])}
        for k, g in groups.items():
            d = g.get(c)
            row[f"lat_{k}"] = None if d is None else float(d["lat"])
            row[f"acc_{k}"] = None if d is None else float(d["acc"])
            row[f"n_{k}"] = 0 if d is None else int(d["n"])
        rows.append(row)

    csv = OUT_DIR / "ab48_four_modes_latency_accuracy_20260404.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write(
            "case,ctx,inf,"
            "lat_no_feature,acc_no_feature,n_no_feature,"
            "lat_spec_only,acc_spec_only,n_spec_only,"
            "lat_spec_dyn_nojsd,acc_spec_dyn_nojsd,n_spec_dyn_nojsd,"
            "lat_spec_dyn_jsd,acc_spec_dyn_jsd,n_spec_dyn_jsd\n"
        )
        for r in rows:
            f.write(
                f"{r['case']},{r['ctx']},{r['inf']},"
                f"{fmt(r['lat_no_feature'])},{fmt(r['acc_no_feature'])},{r['n_no_feature']},"
                f"{fmt(r['lat_spec_only'])},{fmt(r['acc_spec_only'])},{r['n_spec_only']},"
                f"{fmt(r['lat_spec_dyn_nojsd'])},{fmt(r['acc_spec_dyn_nojsd'])},{r['n_spec_dyn_nojsd']},"
                f"{fmt(r['lat_spec_dyn_jsd'])},{fmt(r['acc_spec_dyn_jsd'])},{r['n_spec_dyn_jsd']}\n"
            )

    cov_txt = OUT_DIR / "coverage_summary_20260404.txt"
    with cov_txt.open("w", encoding="utf-8") as f:
        for k, g in groups.items():
            f.write(f"{k}: {len(g)} cases\n")
        four_inter = [c for c in ordered if all(c in g for g in groups.values())]
        f.write(f"intersection_all_4: {len(four_inter)} cases\n")

    names = ["NoFeature", "Spec", "Spec+Dyn(NoJSD)", "Spec+Dyn+JSD"]
    color_map = ["#bdbdbd", "#8da0cb", "#fc8d62", "#66c2a5"]
    lat_means = [
        float(np.mean([v["lat"] for v in no_feature.values()])) if no_feature else np.nan,
        float(np.mean([v["lat"] for v in spec_only.values()])) if spec_only else np.nan,
        float(np.mean([v["lat"] for v in spec_dyn_nojsd.values()])) if spec_dyn_nojsd else np.nan,
        float(np.mean([v["lat"] for v in spec_dyn_jsd.values()])) if spec_dyn_jsd else np.nan,
    ]
    acc_means = [
        float(np.mean([v["acc"] for v in no_feature.values()])) if no_feature else np.nan,
        float(np.mean([v["acc"] for v in spec_only.values()])) if spec_only else np.nan,
        float(np.mean([v["acc"] for v in spec_dyn_nojsd.values()])) if spec_dyn_nojsd else np.nan,
        float(np.mean([v["acc"] for v in spec_dyn_jsd.values()])) if spec_dyn_jsd else np.nan,
    ]
    counts = [len(no_feature), len(spec_only), len(spec_dyn_nojsd), len(spec_dyn_jsd)]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(x, lat_means, color=color_map)
    axes[0].set_title("Latency Mean")
    axes[0].set_ylabel("latency ↓")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)], rotation=0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, acc_means, color=color_map)
    axes[1].set_title("Accuracy Mean")
    axes[1].set_ylabel("accuracy ↑")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)], rotation=0)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "ab48_four_modes_mean_bar_20260404.png", dpi=220)
    plt.close(fig)

    inter = [r for r in rows if all(r[f"lat_{k}"] is not None for k in groups.keys())]
    inter.sort(key=lambda z: z["case"])

    if inter:
        labels = [f"{r['case']} ({r['ctx']},{r['inf']})" for r in inter]
        xx = np.arange(len(inter))
        w = 0.2
        lat_nf = np.array([r["lat_no_feature"] for r in inter], dtype=float)
        lat_sp = np.array([r["lat_spec_only"] for r in inter], dtype=float)
        lat_nd = np.array([r["lat_spec_dyn_nojsd"] for r in inter], dtype=float)
        lat_js = np.array([r["lat_spec_dyn_jsd"] for r in inter], dtype=float)
        acc_nf = np.array([r["acc_no_feature"] for r in inter], dtype=float)
        acc_sp = np.array([r["acc_spec_only"] for r in inter], dtype=float)
        acc_nd = np.array([r["acc_spec_dyn_nojsd"] for r in inter], dtype=float)
        acc_js = np.array([r["acc_spec_dyn_jsd"] for r in inter], dtype=float)

        fig2, ax = plt.subplots(2, 1, figsize=(max(10, len(inter) * 1.2), 8.5), constrained_layout=True)
        ax[0].bar(xx - 1.5*w, lat_nf, width=w, label="NoFeature", color="#bdbdbd")
        ax[0].bar(xx - 0.5*w, lat_sp, width=w, label="Spec", color="#8da0cb")
        ax[0].bar(xx + 0.5*w, lat_nd, width=w, label="Spec+Dyn(NoJSD)", color="#fc8d62")
        ax[0].bar(xx + 1.5*w, lat_js, width=w, label="Spec+Dyn+JSD", color="#66c2a5")
        ax[0].set_title("AB48 Four Modes (common cases): Latency")
        ax[0].set_ylabel("latency ↓")
        ax[0].set_xticks(xx)
        ax[0].set_xticklabels(labels, rotation=30, ha="right")
        ax[0].grid(axis="y", alpha=0.25)
        ax[0].legend(ncol=2, loc="upper right")

        ax[1].bar(xx - 1.5*w, acc_nf, width=w, label="NoFeature", color="#bdbdbd")
        ax[1].bar(xx - 0.5*w, acc_sp, width=w, label="Spec", color="#8da0cb")
        ax[1].bar(xx + 0.5*w, acc_nd, width=w, label="Spec+Dyn(NoJSD)", color="#fc8d62")
        ax[1].bar(xx + 1.5*w, acc_js, width=w, label="Spec+Dyn+JSD", color="#66c2a5")
        ax[1].set_title("AB48 Four Modes (common cases): Accuracy")
        ax[1].set_ylabel("accuracy ↑")
        ax[1].set_xticks(xx)
        ax[1].set_xticklabels(labels, rotation=30, ha="right")
        ax[1].grid(axis="y", alpha=0.25)
        fig2.savefig(OUT_DIR / "ab48_four_modes_common_cases_bar_20260404.png", dpi=220)
        plt.close(fig2)

        # Green scatter: JSD vs NoJSD
        dx = lat_js - lat_nd
        dy = acc_js - acc_nd
        q = plot_scatter_with_quadrants(
            out_png=OUT_DIR / "ab48_scatter_jsd_vs_nojsd_20260404.png",
            out_md=None,
            title="AB48 Scatter: JSD vs NoJSD",
            xlabel="Δ latency = JSD - NoJSD (x<0 better)",
            ylabel="Δ accuracy = JSD - NoJSD (y>0 better)",
            xs=dx,
            ys=dy,
            color="#66c2a5",
        )

        # Extra scatters: Static vs JSD, Static vs NoJSD
        dx_sj = lat_sp - lat_js
        dy_sj = acc_sp - acc_js
        plot_scatter_with_quadrants(
            out_png=OUT_DIR / "ab48_scatter_static_vs_jsd_20260404.png",
            out_md=None,
            title="AB48 Scatter: Static(Spec) vs JSD",
            xlabel="Δ latency = Static - JSD (x<0: Static better, x>0: JSD better)",
            ylabel="Δ accuracy = Static - JSD (y>0: Static better, y<0: JSD better)",
            xs=dx_sj,
            ys=dy_sj,
            color="#5ab4ac",
        )

        dx_sn = lat_sp - lat_nd
        dy_sn = acc_sp - acc_nd
        plot_scatter_with_quadrants(
            out_png=OUT_DIR / "ab48_scatter_static_vs_nojsd_20260404.png",
            out_md=None,
            title="AB48 Scatter: Static(Spec) vs NoJSD",
            xlabel="Δ latency = Static - NoJSD (x<0: Static better, x>0: NoJSD better)",
            ylabel="Δ accuracy = Static - NoJSD (y>0: Static better, y<0: NoJSD better)",
            xs=dx_sn,
            ys=dy_sn,
            color="#80b1d3",
        )

        # analysis markdown
        md = OUT_DIR / "green_scatter_analysis_20260404.md"
        with md.open("w", encoding="utf-8") as f:
            f.write("# AB48 Green Scatter Analysis (JSD vs NoJSD)\n\n")
            f.write(f"cases={len(dx)}\n\n")
            f.write(f"Q1 (NoJSD acc+lat better): {q['Q1'][0]}/{len(dx)} ({q['Q1'][1]:.1f}%)\n")
            f.write(f"Q2 (JSD acc+lat better): {q['Q2'][0]}/{len(dx)} ({q['Q2'][1]:.1f}%)\n")
            f.write(f"Q3 (JSD latency better, NoJSD acc better): {q['Q3'][0]}/{len(dx)} ({q['Q3'][1]:.1f}%)\n")
            f.write(f"Q4 (NoJSD latency better, JSD acc better): {q['Q4'][0]}/{len(dx)} ({q['Q4'][1]:.1f}%)\n\n")
            f.write(f"mean(Δlat=JSD-NoJSD)={float(np.mean(dx)):.6f} (negative is better)\n")
            f.write(f"mean(Δacc=JSD-NoJSD)={float(np.mean(dy)):.6f} (positive is better)\n")

    print(f"WROTE {csv}")
    print(f"WROTE {cov_txt}")


if __name__ == "__main__":
    main()
