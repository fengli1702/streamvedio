#!/usr/bin/env python3
import json
import os
import re
from glob import glob
import pandas as pd

RE = re.compile(
    r"^doh_cod_1\.22_(?P<mode>joint|train_only|infer_only)_c(?P<ctx>\d+)_i(?P<inf>\d+)_ib(?P<ib>\d+)_tb(?P<tb>\d+)_lr(?P<lr>[^_]+)_spec(?P<spec>on|off)$"
)


def read_infer_mean(path: str):
    lats = []
    accs = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, dict) and "latency" in obj and "accuracy" in obj:
                if obj["latency"] is None or obj["accuracy"] is None:
                    continue
                lats.append(float(obj["latency"]))
                accs.append(float(obj["accuracy"]))
    if not lats or not accs:
        return None, None, 0
    return sum(lats) / len(lats), sum(accs) / len(accs), len(lats)


def main():
    logs_dir = "/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs"
    wandb_csv = "/m-coriander/coriander/daifeng/testvllm/vllm/tream/analysis_out_doh/wandb_summary_doh_full.csv"
    out_csv = "/m-coriander/coriander/daifeng/testvllm/vllm/tream/analysis_out_doh/q3_inferlog_vs_wandb_diff_c4_i4.csv"

    files = glob(os.path.join(logs_dir, "doh_cod_1.22_*_c4_i4_*_spec*.jsonl"))
    rows = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        if not RE.match(name):
            continue
        lat, acc, n = read_infer_mean(fp)
        if lat is None or acc is None:
            continue
        rows.append({
            "run_name": name,
            "lat_mean_s_inferlog": lat,
            "acc_mean_inferlog": acc,
            "n_frames": n,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No inference log rows")

    wd = pd.read_csv(wandb_csv)
    wd = wd[["run_name", "inference/latency", "inference/global_accuracy"]]

    m = df.merge(wd, on="run_name", how="left")
    m["lat_diff_s"] = m["lat_mean_s_inferlog"] - m["inference/latency"]
    m["acc_diff"] = m["acc_mean_inferlog"] - m["inference/global_accuracy"]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    m.to_csv(out_csv, index=False)

    # print stats
    print("rows", len(m))
    print("lat abs mean", m["lat_diff_s"].abs().mean(), "lat abs max", m["lat_diff_s"].abs().max())
    print("acc abs mean", m["acc_diff"].abs().mean(), "acc abs max", m["acc_diff"].abs().max())


if __name__ == "__main__":
    main()
