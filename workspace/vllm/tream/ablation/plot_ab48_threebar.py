#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCHIVE_DIR = Path("/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/20260325_ab48_3groups_static_dynamic_notokenshift_内容总结")
OUT_DIR = Path("/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_RE = re.compile(r"_(?P<case>[A-F]\d{2})_")
CTX_INF_RE = re.compile(r"_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib")


def case_order() -> List[str]:
    out: List[str] = []
    for g in "ABCDEF":
        for i in range(1, 9):
            out.append(f"{g}{i:02d}")
    return out


def mean_lat_acc(path: Path) -> Optional[Tuple[float, float]]:
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
                if isinstance(l, (int, float)):
                    lat.append(float(l))
                if isinstance(a, (int, float)):
                    acc.append(float(a))
    except FileNotFoundError:
        return None
    if not lat or not acc:
        return None
    return float(np.mean(lat)), float(np.mean(acc))


def parse_case_ctx_inf(name: str) -> Optional[Tuple[str, int, int]]:
    m1 = CASE_RE.search(name)
    m2 = CTX_INF_RE.search(name)
    if not (m1 and m2):
        return None
    return m1.group("case"), int(m2.group("ctx")), int(m2.group("inf"))


def collect(patterns: List[Tuple[str, bool]]) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    for pat, only_if_missing in patterns:
        for p in sorted(ARCHIVE_DIR.glob(pat)):
            parsed = parse_case_ctx_inf(p.name)
            if not parsed:
                continue
            case, ctx, inf = parsed
            if only_if_missing and case in data:
                continue
            ma = mean_lat_acc(p)
            if ma is None:
                continue
            lat, acc = ma
            data[case] = {
                "ctx": ctx,
                "inf": inf,
                "lat": lat,
                "acc": acc,
                "file": str(p),
            }
    return data


def main() -> None:
    # group patterns: (glob_pattern, only_if_missing)
    static = collect([
        ("doh_shift_ab48x3_static_nodyn_ib32_tb16_*_20260321_211450_*.jsonl", False),
    ])
    dyn_jsd = collect([
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_*_20260322_112345_*.jsonl", False),
        # C04补跑
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_C04_20260322_233952_*.jsonl", True),
        # 20260324 精准补跑（覆盖旧值）
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_B03_20260324_213754_*.jsonl", False),
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_E02_20260324_213754_*.jsonl", False),
    ])
    dyn_td1 = collect([
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_*_20260322_235343_*.jsonl", False),
        # A03补跑
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_A03_20260324_035203_*.jsonl", True),
        # 20260324 精准补跑（覆盖旧值）
        ("doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_E07_20260324_214331_*.jsonl", False),
    ])

    rows: List[Dict[str, object]] = []
    for c in case_order():
        if c not in static or c not in dyn_jsd or c not in dyn_td1:
            continue
        ctx = int(static[c]["ctx"])
        inf = int(static[c]["inf"])
        rows.append({
            "case": c,
            "ctx": ctx,
            "inf": inf,
            "lat_static": float(static[c]["lat"]),
            "acc_static": float(static[c]["acc"]),
            "lat_dyn_jsd": float(dyn_jsd[c]["lat"]),
            "acc_dyn_jsd": float(dyn_jsd[c]["acc"]),
            "lat_dyn_td1": float(dyn_td1[c]["lat"]),
            "acc_dyn_td1": float(dyn_td1[c]["acc"]),
        })

    if not rows:
        raise RuntimeError("No overlap rows found across three groups.")

    # csv
    csv_path = OUT_DIR / "ab48_threebar_latency_accuracy_20260325.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("case,ctx,inf,lat_static,acc_static,lat_dyn_jsd,acc_dyn_jsd,lat_dyn_td1,acc_dyn_td1\n")
        for r in rows:
            f.write(
                f"{r['case']},{r['ctx']},{r['inf']},"
                f"{r['lat_static']:.6f},{r['acc_static']:.6f},"
                f"{r['lat_dyn_jsd']:.6f},{r['acc_dyn_jsd']:.6f},"
                f"{r['lat_dyn_td1']:.6f},{r['acc_dyn_td1']:.6f}\n"
            )

    labels = [f"({r['ctx']},{r['inf']})" for r in rows]
    x = np.arange(len(rows))
    w = 0.26

    lat_static = np.array([r["lat_static"] for r in rows], dtype=float)
    lat_dyn_jsd = np.array([r["lat_dyn_jsd"] for r in rows], dtype=float)
    lat_dyn_td1 = np.array([r["lat_dyn_td1"] for r in rows], dtype=float)

    acc_static = np.array([r["acc_static"] for r in rows], dtype=float)
    acc_dyn_jsd = np.array([r["acc_dyn_jsd"] for r in rows], dtype=float)
    acc_dyn_td1 = np.array([r["acc_dyn_td1"] for r in rows], dtype=float)

    # combined figure
    fig, axes = plt.subplots(2, 1, figsize=(max(24, len(rows) * 0.55), 12), constrained_layout=True)

    ax = axes[0]
    ax.bar(x - w, lat_static, width=w, label="Static", color="#8da0cb")
    ax.bar(x, lat_dyn_td1, width=w, label="NoJSD (No TokenShift)", color="#fc8d62")
    ax.bar(x + w, lat_dyn_jsd, width=w, label="Dynamic (JSD)", color="#66c2a5")
    ax.set_title("AB48 Three-Group Latency Comparison")
    ax.set_ylabel("Latency (s)  ↓")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper right")

    ax2 = axes[1]
    ax2.bar(x - w, acc_static, width=w, label="Static", color="#8da0cb")
    ax2.bar(x, acc_dyn_td1, width=w, label="NoJSD (No TokenShift)", color="#fc8d62")
    ax2.bar(x + w, acc_dyn_jsd, width=w, label="Dynamic (JSD)", color="#66c2a5")
    ax2.set_title("AB48 Three-Group Accuracy Comparison")
    ax2.set_ylabel("Accuracy  ↑")
    ax2.set_xlabel("(ctx, inf)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    fig_path = OUT_DIR / "ab48_threebar_latency_accuracy_20260325.png"
    fig.savefig(fig_path, dpi=180)

    # separate figures
    fig_l, ax_l = plt.subplots(figsize=(max(24, len(rows) * 0.55), 5.5), constrained_layout=True)
    ax_l.bar(x - w, lat_static, width=w, label="Static", color="#8da0cb")
    ax_l.bar(x, lat_dyn_td1, width=w, label="NoJSD (No TokenShift)", color="#fc8d62")
    ax_l.bar(x + w, lat_dyn_jsd, width=w, label="Dynamic (JSD)", color="#66c2a5")
    ax_l.set_title("AB48 Latency Three-Bar")
    ax_l.set_ylabel("Latency (s)  ↓")
    ax_l.set_xlabel("(ctx, inf)")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, rotation=90, fontsize=8)
    ax_l.grid(axis="y", alpha=0.25)
    ax_l.legend(ncol=3, loc="upper right")
    lat_fig = OUT_DIR / "ab48_threebar_latency_20260325.png"
    fig_l.savefig(lat_fig, dpi=180)

    fig_a, ax_a = plt.subplots(figsize=(max(24, len(rows) * 0.55), 5.5), constrained_layout=True)
    ax_a.bar(x - w, acc_static, width=w, label="Static", color="#8da0cb")
    ax_a.bar(x, acc_dyn_td1, width=w, label="NoJSD (No TokenShift)", color="#fc8d62")
    ax_a.bar(x + w, acc_dyn_jsd, width=w, label="Dynamic (JSD)", color="#66c2a5")
    ax_a.set_title("AB48 Accuracy Three-Bar")
    ax_a.set_ylabel("Accuracy  ↑")
    ax_a.set_xlabel("(ctx, inf)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=90, fontsize=8)
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(ncol=3, loc="upper right")
    acc_fig = OUT_DIR / "ab48_threebar_accuracy_20260325.png"
    fig_a.savefig(acc_fig, dpi=180)

    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"combined={fig_path}")
    print(f"lat={lat_fig}")
    print(f"acc={acc_fig}")


if __name__ == "__main__":
    main()
