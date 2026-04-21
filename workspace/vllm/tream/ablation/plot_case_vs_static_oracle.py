#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_scheduler_cfg_seq(path: Path) -> List[Tuple[int, int, int, str]]:
    seq = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            step = int(obj.get("scheduler/step", len(seq)))
            ctx = int(obj.get("scheduler/context_length", -1))
            inf = int(obj.get("scheduler/inference_length", -1))
            reason = str(obj.get("scheduler/decision", ""))
            seq.append((step, ctx, inf, reason))
    return seq


def _load_static_runs(static_dir: Path) -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for p in sorted(static_dir.glob("doh_shift_ab48x3_static_nodyn_ib32_tb16_*_c*_i*_ib32_tb16.jsonl")):
        name = p.stem
        # parse ctx/inf from suffix
        # ..._c3_i5_ib32_tb16
        try:
            ci = name.split("_c")[-1]
            ctx = int(ci.split("_i")[0])
            inf = int(ci.split("_i")[1].split("_ib")[0])
        except Exception:
            continue

        lats: List[float] = []
        accs: List[float] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("event") != "inference_cycle":
                    continue
                lat = obj.get("latency")
                acc = obj.get("accuracy")
                if isinstance(lat, (int, float)) and math.isfinite(float(lat)) and isinstance(acc, (int, float)) and math.isfinite(float(acc)):
                    lats.append(float(lat))
                    accs.append(float(acc))

        if not lats:
            continue
        runs[name] = {"ctx": ctx, "inf": inf, "lat": np.array(lats), "acc": np.array(accs)}
    return runs


def _oracle_from_static(
    static_runs: Dict[str, Dict[str, object]],
    num_steps: int,
    tau: float,
) -> Tuple[List[int], List[int], List[int]]:
    lengths = [int(v["lat"].shape[0]) for v in static_runs.values()]
    n = min(lengths)
    edges = np.linspace(0, n, num_steps + 1, dtype=int)

    out_ctx: List[int] = []
    out_inf: List[int] = []
    out_run_idx: List[int] = []
    run_items = list(static_runs.items())

    for s in range(num_steps):
        a, b = int(edges[s]), int(edges[s + 1])
        if b <= a:
            b = min(n, a + 1)
        cand = []
        for idx, (_name, item) in enumerate(run_items):
            lat = float(np.mean(item["lat"][a:b]))
            acc = float(np.mean(item["acc"][a:b]))
            cand.append((idx, lat, acc, int(item["ctx"]), int(item["inf"])))

        feas = [x for x in cand if x[2] >= tau]
        pool = feas if feas else cand
        best = min(pool, key=lambda x: (x[1], -x[2]))
        out_run_idx.append(best[0])
        out_ctx.append(best[3])
        out_inf.append(best[4])

    return out_ctx, out_inf, out_run_idx


def _resample_dyn_cfg(seq: List[Tuple[int, int, int, str]], num_steps: int) -> Tuple[List[int], List[int], List[str]]:
    by_step = {s: (c, i, r) for s, c, i, r in seq}
    max_step = max(by_step.keys()) if by_step else -1
    last = (None, None, "")
    out_c: List[int] = []
    out_i: List[int] = []
    out_r: List[str] = []
    for s in range(num_steps):
        q = min(s, max_step)
        if q in by_step:
            last = by_step[q]
        c, i, r = last
        out_c.append(int(c) if c is not None else -1)
        out_i.append(int(i) if i is not None else -1)
        out_r.append(str(r))
    return out_c, out_i, out_r


def _plot_case(
    case: str,
    seq_jsd: List[Tuple[int, int, int, str]],
    seq_nojsd: List[Tuple[int, int, int, str]],
    oracle_ctx: List[int],
    oracle_inf: List[int],
    out_png: Path,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    steps = len(oracle_ctx)
    x = np.arange(steps)

    jsd_ctx, jsd_inf, _ = _resample_dyn_cfg(seq_jsd, steps)
    no_ctx, no_inf, _ = _resample_dyn_cfg(seq_nojsd, steps)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)

    axes[0].plot(x, oracle_ctx, color="#222222", linewidth=2.2, linestyle="--", label="Static Oracle (acc>=tau, min latency)")
    axes[0].plot(x, jsd_ctx, color="#1f77b4", linewidth=1.8, label="Dynamic JSD")
    axes[0].plot(x, no_ctx, color="#ff7f0e", linewidth=1.8, label="Dynamic NoJSD")
    axes[0].set_ylabel("ctx")
    axes[0].set_title(f"{case}: scheduler sequence vs static oracle")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(x, oracle_inf, color="#222222", linewidth=2.2, linestyle="--", label="Static Oracle")
    axes[1].plot(x, jsd_inf, color="#1f77b4", linewidth=1.8, label="Dynamic JSD")
    axes[1].plot(x, no_inf, color="#ff7f0e", linewidth=1.8, label="Dynamic NoJSD")
    axes[1].set_ylabel("inf")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return jsd_ctx, jsd_inf, no_ctx, no_inf


def _find_file(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No file found in candidates: {candidates}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot case scheduler sequence vs static-oracle sequence")
    ap.add_argument("--base", type=Path, default=Path("testvllm/vllm/tream/inference_logs/20260325_ab48_3groups_static_dynamic_notokenshift_内容总结"))
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=41)
    ap.add_argument("--out-dir", type=Path, default=Path("testvllm/vllm/tream/ablation/final_outputs"))
    args = ap.parse_args()

    static_runs = _load_static_runs(args.base)
    oracle_ctx, oracle_inf, _ = _oracle_from_static(static_runs, args.steps, args.tau)

    cases = {
        "B04": {
            "jsd": [
                args.base / "scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_B04_20260322_112345_wC_c3_i5_ib32_tb16.jsonl",
                Path("testvllm/vllm/tream/inference_logs/scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_B04_20260322_112345_wC_c3_i5_ib32_tb16.jsonl"),
            ],
            "nojsd": [
                args.base / "scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_B04_20260322_235343_wA_c3_i5_ib32_tb16.jsonl",
                Path("testvllm/vllm/tream/inference_logs/scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_B04_20260322_235343_wA_c3_i5_ib32_tb16.jsonl"),
            ],
        },
        "C04": {
            "jsd": [
                args.base / "scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_C04_20260322_233952_wA_c8_i2_ib32_tb16.jsonl",
                Path("testvllm/vllm/tream/inference_logs/scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1_cw15_C04_20260322_233952_wA_c8_i2_ib32_tb16.jsonl"),
            ],
            "nojsd": [
                args.base / "scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_C04_20260322_235343_wB_c8_i2_ib32_tb16.jsonl",
                Path("testvllm/vllm/tream/inference_logs/scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1td1_cw15_C04_20260322_235343_wB_c8_i2_ib32_tb16.jsonl"),
            ],
        },
    }

    summary_lines = ["case,step,oracle_ctx,oracle_inf,jsd_ctx,jsd_inf,nojsd_ctx,nojsd_inf"]

    for case, paths in cases.items():
        p_jsd = _find_file(paths["jsd"])
        p_no = _find_file(paths["nojsd"])

        seq_jsd = _load_scheduler_cfg_seq(p_jsd)
        seq_no = _load_scheduler_cfg_seq(p_no)

        out_png = args.out_dir / f"ab48_case_{case}_sequence_vs_static_oracle_tau{str(args.tau).replace('.', 'p')}.png"
        jsd_ctx, jsd_inf, no_ctx, no_inf = _plot_case(case, seq_jsd, seq_no, oracle_ctx, oracle_inf, out_png)

        for s in range(args.steps):
            summary_lines.append(
                f"{case},{s},{oracle_ctx[s]},{oracle_inf[s]},{jsd_ctx[s]},{jsd_inf[s]},{no_ctx[s]},{no_inf[s]}"
            )

    out_csv = args.out_dir / f"ab48_case_sequence_vs_static_oracle_tau{str(args.tau).replace('.', 'p')}.csv"
    out_csv.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"WROTE {out_csv}")


if __name__ == "__main__":
    main()
