#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_oracle(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        return _load_jsonl(path)

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _scheduler_cfg_key(row: Dict[str, Any]) -> str:
    ctx = row.get("scheduler/new_context_length", row.get("scheduler/context_length"))
    inf = row.get("scheduler/new_inference_length", row.get("scheduler/inference_length"))
    ib = row.get("scheduler/new_ib", row.get("scheduler/ib"))
    tb = row.get("scheduler/new_tb", row.get("scheduler/tb"))

    key = f"ctx{ctx}_inf{inf}"
    if ib is not None or tb is not None:
        key = f"{key}_ib{ib}_tb{tb}"
    return key


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(mean(vals))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay scheduler logs against per-window oracle.")
    p.add_argument("--scheduler-log", required=True, help="Path to scheduler JSONL log.")
    p.add_argument("--oracle", required=True, help="Path to oracle CSV/JSONL.")
    p.add_argument("--out", default=None, help="Optional output JSON path.")

    p.add_argument("--scheduler-lat-col", default="scheduler/latency_mean")
    p.add_argument("--scheduler-acc-col", default="scheduler/quality")
    p.add_argument("--oracle-lat-col", default="lat_mean")
    p.add_argument("--oracle-acc-col", default="acc_mean")
    p.add_argument("--oracle-key-col", default="selected_key")
    p.add_argument("--window-col", default="window_id")

    p.add_argument("--tau", type=float, default=None, help="Quality lower bound.")
    p.add_argument("--sla", type=float, default=None, help="Latency SLA upper bound.")
    return p


def _adapt_delays(
    sched_keys: Sequence[str],
    oracle_keys: Sequence[str],
) -> Tuple[Optional[float], int]:
    if len(sched_keys) < 2 or len(oracle_keys) < 2:
        return None, 0

    event_idxs: List[int] = []
    for i in range(1, len(oracle_keys)):
        if oracle_keys[i] and oracle_keys[i - 1] and oracle_keys[i] != oracle_keys[i - 1]:
            event_idxs.append(i)

    if not event_idxs:
        return None, 0

    delays: List[int] = []
    for idx in event_idxs:
        prev_sched = sched_keys[idx - 1]
        delay = len(sched_keys) - idx
        for j in range(idx, len(sched_keys)):
            if sched_keys[j] != prev_sched:
                delay = j - idx
                break
        delays.append(delay)

    return float(mean(delays)), len(event_idxs)


def main() -> int:
    args = _build_parser().parse_args()
    sched_path = Path(args.scheduler_log)
    oracle_path = Path(args.oracle)

    sched_rows = _load_jsonl(sched_path)
    oracle_rows = _load_oracle(oracle_path)

    if not sched_rows:
        raise SystemExit("scheduler log is empty or unreadable")
    if not oracle_rows:
        raise SystemExit("oracle file is empty or unreadable")

    sched_rows = sorted(
        sched_rows,
        key=lambda r: int(r.get("scheduler/step", 0)),
    )
    if args.window_col in oracle_rows[0]:
        oracle_rows = sorted(
            oracle_rows,
            key=lambda r: int(float(r.get(args.window_col, 0))),
        )

    n = min(len(sched_rows), len(oracle_rows))
    sched_rows = sched_rows[:n]
    oracle_rows = oracle_rows[:n]

    sla = args.sla
    if sla is None:
        sla = _to_float(sched_rows[0].get("scheduler/sla_latency"))

    lat_regrets: List[float] = []
    violation_flags: List[int] = []
    sched_keys: List[str] = []
    oracle_keys: List[str] = []

    for s_row, o_row in zip(sched_rows, oracle_rows):
        s_lat = _to_float(s_row.get(args.scheduler_lat_col))
        if s_lat is None:
            s_lat = _to_float(s_row.get("scheduler/latency_p95"))

        s_acc = _to_float(s_row.get(args.scheduler_acc_col))
        if s_acc is None:
            s_acc = _to_float(s_row.get("scheduler/accuracy_mean"))

        o_lat = _to_float(o_row.get(args.oracle_lat_col))
        o_acc = _to_float(o_row.get(args.oracle_acc_col))

        if s_lat is not None and o_lat is not None:
            lat_regrets.append(max(0.0, float(s_lat) - float(o_lat)))

        violated = 0
        if sla is not None and s_lat is not None and s_lat > float(sla):
            violated = 1
        if args.tau is not None and s_acc is not None and s_acc < float(args.tau):
            violated = 1
        violation_flags.append(violated)

        sched_keys.append(_scheduler_cfg_key(s_row))
        oracle_keys.append(str(o_row.get(args.oracle_key_col, "")))

    switches = 0
    for i in range(1, len(sched_keys)):
        if sched_keys[i] != sched_keys[i - 1]:
            switches += 1

    adapt_delay, adapt_events = _adapt_delays(sched_keys, oracle_keys)

    result = {
        "n_windows_compared": n,
        "lat_regret_mean": _safe_mean(lat_regrets),
        "lat_regret_p95": None if not lat_regrets else sorted(lat_regrets)[int(0.95 * (len(lat_regrets) - 1))],
        "violation_rate": _safe_mean([float(v) for v in violation_flags]),
        "switch_rate": None if n <= 1 else float(switches) / float(n - 1),
        "adapt_delay": adapt_delay,
        "adapt_events": adapt_events,
        "sla": sla,
        "tau": args.tau,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
