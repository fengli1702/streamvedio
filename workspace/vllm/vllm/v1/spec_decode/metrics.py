# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import json
import os
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_rejected_tokens: int = 0
    num_reverify: int = 0
    num_early_stops: int = 0
    num_verify_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    timing_samples: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(
        self,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_rejected_tokens: int = 0,
        num_reverify: int = 0,
        num_early_stops: int = 0,
        num_verify_tokens: int = 0,
    ):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        self.num_rejected_tokens += num_rejected_tokens
        self.num_reverify += num_reverify
        self.num_early_stops += num_early_stops
        self.num_verify_tokens += (
            num_verify_tokens if num_verify_tokens > 0 else num_accepted_tokens
        )
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1

    def observe_timing(self, draft_time_ms: float | None, verify_time_ms: float | None):
        draft_ms = float(draft_time_ms) if draft_time_ms is not None else float("nan")
        verify_ms = float(verify_time_ms) if verify_time_ms is not None else float("nan")
        if not np.isfinite(draft_ms) and not np.isfinite(verify_ms):
            return
        if np.isfinite(draft_ms):
            self.draft_time_ms += draft_ms
        if np.isfinite(verify_ms):
            self.verify_time_ms += verify_ms
        self.timing_samples += 1


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()
        # Simple monotonic step index for JSONL logging.
        self._step_idx = 0

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.num_rejected_tokens: list[int] = []
        self.num_reverify: list[int] = []
        self.num_early_stops: list[int] = []
        self.num_verify_tokens: list[int] = []
        self.draft_time_ms: list[float] = []
        self.verify_time_ms: list[float] = []
        self.timing_samples: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.num_rejected_tokens.append(spec_decoding_stats.num_rejected_tokens)
        self.num_reverify.append(spec_decoding_stats.num_reverify)
        self.num_early_stops.append(spec_decoding_stats.num_early_stops)
        self.num_verify_tokens.append(spec_decoding_stats.num_verify_tokens)
        self.draft_time_ms.append(spec_decoding_stats.draft_time_ms)
        self.verify_time_ms.append(spec_decoding_stats.verify_time_ms)
        self.timing_samples.append(spec_decoding_stats.timing_samples)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = int(np.sum(self.num_drafts))
        num_draft_tokens = int(np.sum(self.num_draft_tokens))
        num_accepted_tokens = int(np.sum(self.num_accepted_tokens))
        num_rejected_tokens = int(np.sum(self.num_rejected_tokens))
        num_reverify = int(np.sum(self.num_reverify))
        num_early_stops = int(np.sum(self.num_early_stops))
        num_verify_tokens = int(np.sum(self.num_verify_tokens))
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )
        reject_ratio = (
            num_rejected_tokens / num_draft_tokens
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = (
            1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else float("nan")
        )

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        if pos_matrix.size == 0 or num_drafts <= 0:
            acceptance_rates = np.array([])
            rates_str = ""
        else:
            acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
            rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        timing_weight = int(np.sum(self.timing_samples))
        draft_time_ms_per_step = (
            float(np.sum(self.draft_time_ms)) / timing_weight
            if timing_weight > 0
            else float("nan")
        )
        verify_time_ms_per_step = (
            float(np.sum(self.verify_time_ms)) / timing_weight
            if timing_weight > 0
            else float("nan")
        )

        # Optionally mirror a lightweight JSONL record for offline analysis.
        # If VLLM_SPEC_METRICS_PATH is set, append one line per log() call.
        spec_path = os.environ.get("VLLM_SPEC_METRICS_PATH")
        if spec_path:
            try:
                os.makedirs(os.path.dirname(spec_path), exist_ok=True)
                frac = (
                    num_accepted_tokens / num_draft_tokens
                    if num_draft_tokens > 0
                    else 0.0
                )
                rec = {
                    "step": int(self._step_idx),
                    "accepted": int(num_accepted_tokens),
                    "proposed": int(num_draft_tokens),
                    "acceptance_rate": float(frac),
                    "reject_tokens": int(num_rejected_tokens),
                    "reverify_count": int(num_reverify),
                    "early_stop_count": int(num_early_stops),
                    "verify_tokens": int(
                        num_verify_tokens if num_verify_tokens > 0 else num_accepted_tokens
                    ),
                    "draft_time_ms": (
                        float(draft_time_ms_per_step)
                        if np.isfinite(draft_time_ms_per_step)
                        else None
                    ),
                    "verify_time_ms": (
                        float(verify_time_ms_per_step)
                        if np.isfinite(verify_time_ms_per_step)
                        else None
                    ),
                    "reject_ratio": (
                        float(reject_ratio) if np.isfinite(reject_ratio) else None
                    ),
                }
                with open(spec_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                # Logging must never break the main decoding path.
                pass
        self._step_idx += 1

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Rejected: %d tokens, "
            "Reverify: %d, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%, "
            "Draft time/step: %.3f ms, "
            "Verify time/step: %.3f ms",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            num_rejected_tokens,
            num_reverify,
            rates_str,
            draft_acceptance_rate,
            draft_time_ms_per_step,
            verify_time_ms_per_step,
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = make_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = make_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = make_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])


def make_per_engine(
    counter: prometheus_client.Counter, per_engine_labelvalues: dict[int, list[str]]
):
    """Create a counter for each label value."""
    return {
        idx: counter.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
