#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _parse_int_list(raw: str) -> List[int]:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _guess_prefix(log_dir: Path) -> str | None:
    if not log_dir.exists():
        return None
    for item in sorted(log_dir.iterdir()):
        name = item.name
        if not name.startswith("12-20-"):
            continue
        if "_ctx" in name:
            return name.split("_ctx", 1)[0]
    return None


def _iter_pairs(ctx_vals: Iterable[int], inf_vals: Iterable[int], max_total: int) -> List[Tuple[int, int]]:
    pairs = []
    for ctx in ctx_vals:
        for inf in inf_vals:
            if ctx + inf > max_total:
                continue
            pairs.append((ctx, inf))
    return pairs


def _build_cmd(
    python_bin: str,
    tream_script: Path,
    *,
    input_frames_path: str,
    model_name: str,
    ctx: int,
    inf: int,
    inference_batch_size: int,
    wandb_run_name: str,
    max_frames: int,
    num_workers: int,
    max_loras: int,
    optimization_priority: str,
    enable_training: bool,
    gradient_checkpointing: bool,
    enable_scheduler: bool,
    enable_spec: bool,
    spec_method: str,
    spec_draft_model: str,
    spec_vocab_mapping_path: str,
    num_spec_tokens: int,
    spec_disable_mqa_scorer: bool,
) -> List[str]:
    cmd = [
        python_bin,
        str(tream_script),
        "--input_frames_path",
        input_frames_path,
        "--context_length",
        str(ctx),
        "--model_name",
        model_name,
        "--inference_length",
        str(inf),
        "--inference_batch_size",
        str(inference_batch_size),
        "--wandb_run_name",
        wandb_run_name,
        "--max_frames",
        str(max_frames),
        "--num_workers",
        str(num_workers),
        "--max_loras",
        str(max_loras),
        "--optimization_priority",
        optimization_priority,
    ]

    if enable_training:
        cmd.append("--use_training_actor")
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if not enable_scheduler:
        cmd.append("--disable_dynamic_scheduling")

    if enable_spec:
        cmd.extend(
            [
                "--use_speculative_decoding",
                "--spec_method",
                spec_method,
                "--spec_draft_model",
                spec_draft_model,
                "--spec_vocab_mapping_path",
                spec_vocab_mapping_path,
                "--num_spec_tokens",
                str(num_spec_tokens),
            ]
        )
        if spec_disable_mqa_scorer:
            cmd.append("--spec_disable_mqa_scorer")

    return cmd


def main() -> int:
    here = Path(__file__).resolve()
    tream_dir = here.parents[1]
    default_log_dir = tream_dir / "inference_logs"

    parser = argparse.ArgumentParser(description="Run tream.py over a ctx/inf grid.")
    parser.add_argument("--tream_dir", type=str, default=str(tream_dir))
    parser.add_argument("--tream_script", type=str, default=str(tream_dir / "tream.py"))
    parser.add_argument(
        "--input_frames_path",
        type=str,
        default=str(tream_dir / "data/streaming-lvm-dataset/DOH"),
    )
    parser.add_argument("--model_name", type=str, default=str(tream_dir / "lvm-llama2-7b"))
    parser.add_argument(
        "--spec_draft_model",
        type=str,
        default=str(tream_dir.parent / "SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475"),
    )
    parser.add_argument(
        "--spec_vocab_mapping_path",
        type=str,
        default=str(tream_dir / "vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt"),
    )
    parser.add_argument("--spec_method", type=str, default="eagle3")
    parser.add_argument("--num_spec_tokens", type=int, default=3)
    parser.add_argument("--spec_disable_mqa_scorer", action="store_true")
    parser.add_argument("--inference_batch_size", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=1600)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_loras", type=int, default=1)
    parser.add_argument("--optimization_priority", type=str, default="scout")
    parser.add_argument("--ctx_values", type=str, default="1,2,4,6,8")
    parser.add_argument("--inf_values", type=str, default="1,2,4,6,8")
    parser.add_argument("--max_total_len", type=int, default=12)
    parser.add_argument("--run_prefix", type=str, default="")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tream_dir = Path(args.tream_dir).resolve()
    tream_script = Path(args.tream_script).resolve()

    if not args.run_prefix:
        guess = _guess_prefix(default_log_dir)
        if guess:
            args.run_prefix = guess
        else:
            args.run_prefix = "12-20-clean+clean1_base"

    ctx_vals = _parse_int_list(args.ctx_values)
    inf_vals = _parse_int_list(args.inf_values)
    pairs = _iter_pairs(ctx_vals, inf_vals, args.max_total_len)
    if not pairs:
        raise ValueError("No ctx/inf pairs left after max_total_len filter.")

    combos = [
        (False, True, "nospec", "dyn"),
        (False, False, "nospec", "static"),
        (True, True, "spec", "dyn"),
        (True, False, "spec", "static"),
    ]

    for ctx, inf in pairs:
        for enable_spec, enable_sched, spec_tag, sched_tag in combos:
            run_name = f"{args.run_prefix}_ctx{ctx}_inf{inf}_{spec_tag}_{sched_tag}"
            log_path = default_log_dir / f"{run_name}.jsonl"
            if log_path.exists() and not args.overwrite:
                print(f"[SKIP] {run_name} (log exists: {log_path})")
                continue

            cmd = _build_cmd(
                sys.executable,
                tream_script,
                input_frames_path=args.input_frames_path,
                model_name=args.model_name,
                ctx=ctx,
                inf=inf,
                inference_batch_size=args.inference_batch_size,
                wandb_run_name=run_name,
                max_frames=args.max_frames,
                num_workers=args.num_workers,
                max_loras=args.max_loras,
                optimization_priority=args.optimization_priority,
                enable_training=not args.no_train,
                gradient_checkpointing=args.gradient_checkpointing,
                enable_scheduler=enable_sched,
                enable_spec=enable_spec,
                spec_method=args.spec_method,
                spec_draft_model=args.spec_draft_model,
                spec_vocab_mapping_path=args.spec_vocab_mapping_path,
                num_spec_tokens=args.num_spec_tokens,
                spec_disable_mqa_scorer=args.spec_disable_mqa_scorer,
            )

            print("[RUN]", " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=str(tream_dir), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
