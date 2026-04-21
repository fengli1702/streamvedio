# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Debug utilities for Speculative Decoding + LoRA analysis.

This module provides JSONL-based logging for tracking:
1. Per-step acceptance rates (eagle_accept)
2. EAGLE draft proposals and outputs (eagle_input, eagle_output)
3. LoRA state and active adapters (lora_state)
4. Runtime performance metrics (runtime_step)
"""

import json
import os
import time
from typing import Any, Dict, Optional

# Debug configuration from environment
SPEC_LORA_DEBUG = os.getenv("VLLM_SPEC_LORA_DEBUG", "0") == "1"
SPEC_LORA_DEBUG_PATH = os.getenv("VLLM_SPEC_LORA_DEBUG_PATH", None)
PHASE_NAME = os.getenv("VLLM_SPEC_LORA_PHASE", "unknown")

# Global step counters
_global_step_id = 0
_lock = None

# Acceptance tracking
_proposed_tokens_total = 0
_accepted_tokens_total = 0
_step_proposed_tokens = []
_step_accepted_tokens = []


def _init_lock():
    """Initialize threading lock for thread-safe logging."""
    global _lock
    if _lock is None:
        import threading
        _lock = threading.Lock()


def spec_lora_log(record: Dict[str, Any]) -> None:
    """
    Log a debug record to JSONL file.
    
    Args:
        record: Dictionary to log. Will include timestamp and phase automatically.
    """
    if not SPEC_LORA_DEBUG or SPEC_LORA_DEBUG_PATH is None:
        return
    
    _init_lock()
    
    try:
        record = dict(record)  # Copy to avoid side effects
        record.setdefault("ts", time.time())
        record.setdefault("phase", PHASE_NAME)
        
        with _lock:
            with open(SPEC_LORA_DEBUG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Silently fail to avoid disrupting main inference
        import sys
        print(f"[VLLM_SPEC_LORA_DEBUG] Error logging: {e}", file=sys.stderr)


def get_next_step_id() -> int:
    """Get next global step ID."""
    global _global_step_id
    _init_lock()
    with _lock:
        step_id = _global_step_id
        _global_step_id += 1
        return step_id


def reset_step_id() -> None:
    """Reset global step counter."""
    global _global_step_id
    _init_lock()
    with _lock:
        _global_step_id = 0


def record_eagle_accept(num_proposed: int, num_accepted: int, step_id: Optional[int] = None) -> None:
    """
    Record acceptance statistics for EAGLE tokens.
    
    Args:
        num_proposed: Number of tokens proposed by drafter
        num_accepted: Number of tokens accepted/verified by target model
        step_id: Optional step ID (if None, auto-increment)
    """
    if not SPEC_LORA_DEBUG:
        return
    
    global _proposed_tokens_total, _accepted_tokens_total
    _init_lock()
    
    if step_id is None:
        step_id = get_next_step_id()
    
    with _lock:
        _proposed_tokens_total += num_proposed
        _accepted_tokens_total += num_accepted
        _step_proposed_tokens.append((step_id, num_proposed))
        _step_accepted_tokens.append((step_id, num_accepted))
    
    accept_ratio = (
        float(num_accepted) / float(num_proposed)
        if num_proposed > 0 else 0.0
    )
    
    spec_lora_log({
        "type": LogType.EAGLE_ACCEPT,
        "step": step_id,
        "num_proposed_tokens": int(num_proposed),
        "num_accepted_tokens": int(num_accepted),
        "accept_ratio": accept_ratio,
        "total_proposed": int(_proposed_tokens_total),
        "total_accepted": int(_accepted_tokens_total),
    })


def get_acceptance_stats() -> dict:
    """Get cumulative acceptance statistics."""
    _init_lock()
    with _lock:
        return {
            "total_proposed": _proposed_tokens_total,
            "total_accepted": _accepted_tokens_total,
            "global_accept_ratio": (
                float(_accepted_tokens_total) / float(_proposed_tokens_total)
                if _proposed_tokens_total > 0 else 0.0
            ),
            "num_steps": len(_step_proposed_tokens),
        }


def record_runtime_step(
    dt: float,
    num_tokens: int,
    throughput: float,
    phase: str,
    step_id: Optional[int] = None,
) -> None:
    """
    Record runtime performance metrics for a step.
    
    Args:
        dt: Time taken for this step in seconds
        num_tokens: Number of tokens processed
        throughput: Tokens per second for this step
        phase: Phase name (e.g., 'base', 'lora_infra')
        step_id: Optional step ID (if None, auto-increment)
    """
    if not SPEC_LORA_DEBUG:
        return
    
    if step_id is None:
        step_id = get_next_step_id()
    
    spec_lora_log({
        "type": LogType.RUNTIME_STEP,
        "step": step_id,
        "dt": float(dt),
        "num_tokens": int(num_tokens),
        "throughput": float(throughput),
        "phase": str(phase),
    })


# Log record type constants
class LogType:
    """Log record type constants."""
    EAGLE_INPUT = "eagle_input"           # EAGLE proposer input state
    EAGLE_OUTPUT = "eagle_output"         # EAGLE proposer output (draft tokens)
    EAGLE_ACCEPT = "eagle_accept"         # Acceptance statistics
    LORA_STATE = "lora_state"             # LoRA active state
    RUNTIME_STEP = "runtime_step"         # Runtime performance per step

def record_lora_state(
    active_lora_names: list,
    num_active_loras: int,
    phase: str,
    step_id: Optional[int] = None,
) -> None:
    """
    Record LoRA state and active adapters.
    
    Args:
        active_lora_names: List of active LoRA adapter names
        num_active_loras: Number of active LoRA adapters
        phase: Phase name (e.g., 'base', 'lora_infra')
        step_id: Optional step ID (if None, auto-increment)
    """
    if not SPEC_LORA_DEBUG:
        return
    
    if step_id is None:
        step_id = get_next_step_id()
    
    spec_lora_log({
        "type": LogType.LORA_STATE,
        "step": step_id,
        "active_lora_names": list(active_lora_names),
        "num_active_loras": int(num_active_loras),
        "phase": str(phase),
    })
