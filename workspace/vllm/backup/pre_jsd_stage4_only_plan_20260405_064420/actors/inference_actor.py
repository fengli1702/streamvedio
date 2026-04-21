import cupy
import ray
import numpy as np
import os
import json
import hashlib
import time
import math
import torch 
import shutil
from torchvision import transforms
from PIL import Image

# 关闭 DeepGEMM 相关功能，避免 deep_gemm_cpp ABI 问题导致 engine 初始化失败。
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt
from vllm.v1.engine.exceptions import EngineDeadError
from lvm_tokenizer.utils import RAW_VQGAN_PATH, COMPILED_VQGAN_PATH, ENCODING_SIZE
from lvm_tokenizer.muse import VQGANModel
from actors.weights_actor import SharedWeightsActor
from actors.training_actor import TrainingActor
from actors.scheduler_actor import SchedulerActor
from collections import deque
import threading
import torch.cuda.nvtx as nvtx
import json as _json
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(
        self,
        config: dict,
        weights_actor: SharedWeightsActor,
        training_actor: TrainingActor,
        scheduler_actor=None,
    ):
        with nvtx.range("InferenceActor.__init__"):
            self.config = config
            self.weights_actor = weights_actor
            self.training_actor = training_actor
            self._load_config()
            self.scheduler_actor = scheduler_actor # 保存句柄
            self.config_version = 0

            self.call_count = 0
            self.current_lora_path = None
            self.current_lora_id = "base_model"
            self.last_weight_update_step = -1
            self._has_applied_lora_update = False
            self.lora_update_lock = threading.Lock()
            self.run_name = config["run_name"]
            self.uuid = 29500 + int(hashlib.md5(self.run_name.encode()).hexdigest(), 16) % 10000
            self.lora_save_dir = f"./tmp/inference_lora_adapters_{self.uuid}"
            os.makedirs(self.lora_save_dir, exist_ok=True)

            self.config_update_lock = threading.Lock() 
           
            self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
            self._init_image_encoder()
            # Track spec config / lora mode for possible engine rebuilds
            self._spec_config = None
            self._engine_enable_lora = None
            self._init_llm_engine()
            self._init_buffers()
            self._init_global_metrics()
            self._init_spec_metrics_reader()
            self._prev_cycle_token_hist = None
            self.macro_cycle_length = self.batch_size 
            print(f"[InferenceActor] Macro-cycle for scheduler reporting set to {self.macro_cycle_length} frames (based on inference_batch_size).")

            self.cycle_metrics_snapshot = {
                'total_log_prob_sum': 0.0,
                'total_tokens': 0,
                'correct_predictions': 0,
                'total_predictions': 0,
            }

            print(
                f"[InferenceActor] Ready on {self.device} | "
                f"Model: {self.model_name} | "
                f"Frame context: {self.context_length} | "
                f"Batch size: {self.batch_size}"
            )

    def _init_global_metrics(self):
        """初始化全局指标跟踪"""
        with nvtx.range("InferenceActor._init_global_metrics"):
            self.global_metrics = {
                # 困惑度相关（基于生成token的logprobs，按原始逻辑）
                'total_log_prob_sum': 0.0,  # 所有logprobs的总和
                'total_tokens': 0,          # 总token数
                'global_perplexity': 0.0,   # 全局平均困惑度
                
                # 准确率相关
                'correct_predictions': 0,
                'total_predictions': 0,
                'global_accuracy': 0.0,
                
                # 批次历史
                'batch_perplexities': [],
                'batch_accuracies': [],
                'batch_count': 0,
                
                # 窗口移动平均（最近N个批次）
                'window_size': 50,
                'recent_perplexities': deque(maxlen=50),
                'recent_accuracies': deque(maxlen=50),
            }
            
            # 线程安全的锁
            self.metrics_lock = threading.Lock()
            
            # === 跨批次准确率计算的缓存 ===
            # 缓存预测结果，等待与真实tokens比较
            self.prediction_cache = []

    def _load_config(self) -> None:
        with nvtx.range("InferenceActor._load_config"):
            c = self.config
            self.model_name = c["model_name"]
            self.lora_rank = c["lora_rank"]
            self.lora_alpha = c["lora_alpha"]
            self.lora_dropout = c["lora_dropout"]
            self.target_modules = c["target_modules"]
            self.max_loras = c["max_loras"]
            self.gpu_memory_utilisation = c["gpu_memory_utilization"]
            self.tensor_parallel_size = c["tensor_parallel_size"]
            self.temperature = c["temperature"]
            self.top_p = c["top_p"]
            self.context_length = c["context_length"]
            self.inference_length = c["inference_length"]
            self.training_batch_size = c.get("training_batch_size", 8) 
            self.batch_size = c["inference_batch_size"]
            self.wandb_run = c.get("wandb_run")
            self.wandb_config = c.get("wandb_config", {})
            self._wandb_logging_disabled = False
            self.use_speculative_decoding = c.get("use_speculative_decoding", False)
            self.spec_draft_model = c.get("spec_draft_model")
            self.num_spec_tokens = c.get("num_spec_tokens", 5)
            # Optional: provide vocab mapping file for EAGLE-3 to align draft/target vocab
            self.spec_vocab_mapping_path = c.get("spec_vocab_mapping_path")
            # Spec method selection: 'eagle3' (draft model) or 'ngram' (prompt-lookup)
            # Default to 'eagle3' when a draft model is provided; else use 'ngram'.
            self.spec_method = c.get(
                "spec_method",
                "eagle3" if (self.use_speculative_decoding and self.spec_draft_model) else "ngram",
            )
            # When using MQA scorer, allow disabling it to fallback to batch expansion.
            self.spec_disable_mqa_scorer = c.get("spec_disable_mqa_scorer", False)
            # N-gram parameters
            self.prompt_lookup_min = c.get("prompt_lookup_min", 2)
            self.prompt_lookup_max = c.get("prompt_lookup_max", 5)
            self.spec_metrics_path = c.get("spec_metrics_path")
            if self.spec_metrics_path:
                os.makedirs(os.path.dirname(self.spec_metrics_path),
                            exist_ok=True)
                os.environ["VLLM_SPEC_METRICS_PATH"] = self.spec_metrics_path
            self.continuous_lora_update = c.get("continuous_lora_update", True)
            # LoRA/runtime scheduler flags are handled at engine build time.
            # Debug dump of prediction vs truth (enable with EAGLE_DUMP=1)
            self.debug_dump_enabled = c.get("debug_dump_pairs", False) or os.getenv("EAGLE_DUMP") == "1"
            self.debug_dump_path = c.get("debug_dump_path", "/app/inference_logs/pairs.jsonl")
            self.debug_dump_max = int(c.get("debug_dump_max", os.getenv("EAGLE_DUMP_MAX", "50")))
            self._debug_dump_count = 0
            # VQGAN encoding controls
            self.vqgan_batch_size = int(c.get("vqgan_batch_size", 8))
            self.vqgan_precision = str(c.get("vqgan_precision", "fp16")).lower()
            self.prefer_trt_vqgan = bool(c.get("prefer_trt_vqgan", True))
        
            self.disable_dynamic_scheduling = c.get("disable_dynamic_scheduling", False)
            if self.disable_dynamic_scheduling:
                print("[InferenceActor] Dynamic scheduling is DISABLED.")
            self._init_wandb_run()

    def _init_wandb_run(self) -> None:
        if self.wandb_run is not None or wandb is None:
            return
        cfg = self.wandb_config if isinstance(self.wandb_config, dict) else {}
        if not bool(cfg.get("enabled", False)):
            return

        project = str(cfg.get("project") or "test-feng1702")
        entity = cfg.get("entity")
        parent_name = str(cfg.get("parent_run_name") or self.run_name)
        run_group = str(cfg.get("run_group") or parent_name)
        run_mode = str(cfg.get("mode") or "online")
        actor_run_name = f"{parent_name}.infer"

        for attempt in range(1, 4):
            try:
                self.wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=actor_run_name,
                    group=run_group,
                    job_type="inference_actor",
                    mode=run_mode,
                    reinit=True,
                )
                print(
                    f"[InferenceActor] WandB initialized (name={actor_run_name}, group={run_group})."
                )
                return
            except Exception as exc:
                self.wandb_run = None
                if attempt >= 3:
                    print(f"[InferenceActor] WARN: wandb init failed: {exc}")
                    return
                time.sleep(0.5)

    def update_config(self, new_context_length: int, new_inference_length: int, new_version: int):
        with self.config_update_lock:
            config_changed = (self.context_length != new_context_length or self.inference_length != new_inference_length)
            if not config_changed and self.config_version == new_version:
                return

            print(f"[InferenceActor] Updating config to v{new_version}. Config: ({new_context_length}, {new_inference_length})")
            self.context_length = new_context_length
            self.inference_length = new_inference_length
            self.config_version = new_version
            
            if config_changed:
                new_maxlen = (self.context_length + self.batch_size - 1) * ENCODING_SIZE
                current_tokens = list(self.token_buffer)
                self.token_buffer = deque(current_tokens, maxlen=new_maxlen)
                print(f"[InferenceActor] Token buffer resized.")

    def _init_spec_metrics_reader(self) -> None:
        with nvtx.range("InferenceActor._init_spec_metrics_reader"):
            self.spec_metrics_lock = threading.Lock()
            self._spec_metrics_cursor = 0
            self._spec_metrics_last_step = -1
            self._spec_metrics_cache = {}
            if not self.spec_metrics_path:
                return
            try:
                if os.path.exists(self.spec_metrics_path):
                    # Skip stale lines if a path is unexpectedly reused.
                    self._spec_metrics_cursor = os.path.getsize(self.spec_metrics_path)
            except Exception:
                self._spec_metrics_cursor = 0

    @staticmethod
    def _safe_mean(values):
        vals = []
        for value in values:
            if value is None:
                continue
            try:
                vals.append(float(value))
            except Exception:
                continue
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    @staticmethod
    def _token_histogram_from_results(results):
        counts = {}
        total = 0
        for result in results:
            if not isinstance(result, dict):
                continue
            token_ids = result.get("token_ids")
            if token_ids is None:
                continue
            try:
                arr = np.asarray(token_ids, dtype=np.int64).ravel()
            except Exception:
                continue
            if arr.size <= 0:
                continue
            uniq, cnt = np.unique(arr, return_counts=True)
            for tid, c in zip(uniq.tolist(), cnt.tolist()):
                key = int(tid)
                counts[key] = counts.get(key, 0) + int(c)
            total += int(arr.size)
        if total <= 0:
            return None
        return counts, total

    @staticmethod
    def _js_divergence_from_hist(prev_hist, cur_hist):
        if prev_hist is None or cur_hist is None:
            return None
        prev_counts, prev_total = prev_hist
        cur_counts, cur_total = cur_hist
        if prev_total <= 0 or cur_total <= 0:
            return None

        keys = sorted(set(prev_counts.keys()) | set(cur_counts.keys()))
        if not keys:
            return None

        p = np.asarray(
            [float(prev_counts.get(k, 0)) / float(prev_total) for k in keys],
            dtype=np.float64,
        )
        q = np.asarray(
            [float(cur_counts.get(k, 0)) / float(cur_total) for k in keys],
            dtype=np.float64,
        )
        m = 0.5 * (p + q)

        mask_p = p > 0.0
        mask_q = q > 0.0
        kl_pm = float(np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])))
        kl_qm = float(np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])))
        jsd = 0.5 * (kl_pm + kl_qm)
        if not np.isfinite(jsd):
            return None
        # log2-based JSD is bounded in [0, 1] for two distributions.
        return float(max(0.0, min(1.0, jsd)))

    def _compute_cycle_jsd(self, results):
        cur_hist = self._token_histogram_from_results(results)
        jsd = self._js_divergence_from_hist(self._prev_cycle_token_hist, cur_hist)
        if cur_hist is not None:
            self._prev_cycle_token_hist = cur_hist
        return jsd

    def _collect_spec_metrics_delta(self) -> dict:
        if not self.spec_metrics_path:
            return dict(self._spec_metrics_cache)

        with self.spec_metrics_lock:
            path = self.spec_metrics_path
            if not os.path.exists(path):
                return dict(self._spec_metrics_cache)

            try:
                file_size = os.path.getsize(path)
                if file_size < self._spec_metrics_cursor:
                    # File was truncated/rotated.
                    self._spec_metrics_cursor = 0
                    self._spec_metrics_last_step = -1
            except Exception:
                pass

            try:
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(self._spec_metrics_cursor)
                    lines = f.readlines()
                    self._spec_metrics_cursor = f.tell()
            except Exception:
                return dict(self._spec_metrics_cache)

            records = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if "spec_final_stats" in rec:
                    continue
                step_val = rec.get("step")
                if step_val is not None:
                    try:
                        step_i = int(step_val)
                    except Exception:
                        step_i = None
                    if step_i is not None:
                        if step_i <= self._spec_metrics_last_step:
                            continue
                        self._spec_metrics_last_step = step_i
                if (
                    "acceptance_rate" in rec
                    or "accepted" in rec
                    or "proposed" in rec
                ):
                    records.append(rec)

            if not records:
                return dict(self._spec_metrics_cache)

            accept_vals = []
            for rec in records:
                acc = rec.get("acceptance_rate")
                if acc is not None:
                    accept_vals.append(acc)
                    continue
                accepted = rec.get("accepted")
                proposed = rec.get("proposed")
                try:
                    if accepted is not None and proposed is not None and float(proposed) > 0:
                        accept_vals.append(float(accepted) / float(proposed))
                except Exception:
                    pass

            accept_mean = self._safe_mean(accept_vals)
            reject_ratio_mean = self._safe_mean([r.get("reject_ratio") for r in records])
            if reject_ratio_mean is None and accept_mean is not None:
                reject_ratio_mean = max(0.0, 1.0 - float(accept_mean))

            summary = {
                "spec_accept_mean": accept_mean,
                "acceptance_rate": accept_mean,
                "spec_acceptance_rate": accept_mean,
                "spec_reverify_per_step": self._safe_mean([r.get("reverify_count") for r in records]),
                "spec_reverify": self._safe_mean([r.get("reverify_count") for r in records]),
                "reverify": self._safe_mean([r.get("reverify_count") for r in records]),
                "spec_draft_ms_per_step": self._safe_mean([r.get("draft_time_ms") for r in records]),
                "draft_ms_per_step": self._safe_mean([r.get("draft_time_ms") for r in records]),
                "spec_verify_ms_per_step": self._safe_mean([r.get("verify_time_ms") for r in records]),
                "verify_ms_per_step": self._safe_mean([r.get("verify_time_ms") for r in records]),
                "accepted_tokens_per_step": self._safe_mean([r.get("accepted") for r in records]),
                "accepted_tokens": self._safe_mean([r.get("accepted") for r in records]),
                "rejected_tokens_per_step": self._safe_mean([r.get("reject_tokens") for r in records]),
                "rejected_tokens": self._safe_mean([r.get("reject_tokens") for r in records]),
                "reject_ratio": reject_ratio_mean,
            }
            summary = {k: v for k, v in summary.items() if v is not None}
            if not summary:
                return dict(self._spec_metrics_cache)

            self._spec_metrics_cache = summary
            return dict(self._spec_metrics_cache)


    def _calculate_token_accuracy(self, generated_tokens, expected_tokens):
        """计算token级别的准确率"""
        with nvtx.range("InferenceActor._calculate_token_accuracy"):
            if not expected_tokens or len(generated_tokens) == 0:
                return 0.0, 0
            
            min_length = min(len(generated_tokens), len(expected_tokens))
            matches = sum(1 for i in range(min_length) 
                         if generated_tokens[i] == expected_tokens[i])
            
            accuracy = matches / min_length if min_length > 0 else 0.0
            return accuracy, matches

    def _evaluate_cached_predictions(self):
        """评估之前缓存的预测结果，现在有了对应的真实tokens"""
        with nvtx.range("InferenceActor._evaluate_cached_predictions"):
            if not self.prediction_cache:
                return
            
            total_matches = 0
            total_predictions = 0
            
            # 当前的真实tokens
            current_buffer = list(self.token_buffer)
            buffer_length = len(current_buffer)
            
            # 评估缓存的预测
            for cached_pred in self.prediction_cache[:]:  # 创建副本以避免修改时出错
                pred_tokens = cached_pred['predicted_tokens']
                required_start_idx = cached_pred['ground_truth_start_idx']
                required_frame_count = cached_pred['ground_truth_frame_count']
                
                # 检查是否现在有足够的真实tokens
                if required_start_idx + required_frame_count <= buffer_length:
                    # 构建对应的真实token序列
                    true_tokens = []
                    for frame_idx in range(required_start_idx, 
                                         required_start_idx + required_frame_count):
                        if frame_idx < buffer_length:
                            true_tokens.extend(current_buffer[frame_idx])
                    
                    # 计算准确率
                    if true_tokens and pred_tokens:
                        min_length = min(len(pred_tokens), len(true_tokens))
                        matches = sum(1 for j in range(min_length) 
                                    if pred_tokens[j] == true_tokens[j])
                        
                        total_matches += matches
                        total_predictions += min_length

                        # 延迟对齐路径也写出配对，便于核查
                        if self.debug_dump_enabled and self._debug_dump_count < self.debug_dump_max:
                            try:
                                os.makedirs(os.path.dirname(self.debug_dump_path), exist_ok=True)
                                with open(self.debug_dump_path, "a") as f:
                                    f.write(_json.dumps({
                                        "req_idx": int(cached_pred.get('batch_idx', -1)),
                                        "pred_first64": pred_tokens[:64],
                                        "truth_first64": true_tokens[:64],
                                        "pred_len": len(pred_tokens),
                                        "truth_len": len(true_tokens),
                                        "matches": int(matches),
                                        "compare_len": int(min_length),
                                        "delayed": True,
                                    }) + "\n")
                                self._debug_dump_count += 1
                            except Exception:
                                pass
                    
                    # 从缓存中移除已评估的预测
                    self.prediction_cache.remove(cached_pred)

            if total_predictions > 0:
                with self.metrics_lock:
                    self.global_metrics['correct_predictions'] += total_matches
                    self.global_metrics['total_predictions'] += total_predictions
                    self.global_metrics['global_accuracy'] = (
                        self.global_metrics['correct_predictions'] /
                        self.global_metrics['total_predictions']
                    )
                    delayed_acc = total_matches / total_predictions
                    self.global_metrics['recent_accuracies'].append(delayed_acc)
                    self.global_metrics['batch_accuracies'].append(delayed_acc)

    def _update_global_metrics(self, batch_results, expected_tokens_batch=None):
        """更新全局指标，包含跨批次准确率计算"""
        with nvtx.range("InferenceActor._update_global_metrics"):
            with self.metrics_lock:
                if not batch_results:
                    return
                
                batch_log_prob_sum = 0.0
                batch_token_count = 0
                batch_eval_matches = 0
                batch_eval_total = 0
                
                for i, result in enumerate(batch_results):
                    if 'logprobs' not in result or not result['logprobs']:
                        continue
                    
                    logprobs = result['logprobs']
                    token_ids = result.get('token_ids', [])
                    
                    # 累计困惑度相关数据（按原始逻辑）
                    batch_log_prob_sum += sum(logprobs)  # 原始逻辑：sum(logprobs)
                    batch_token_count += len(logprobs)
                    
                    # === 跨批次准确率计算逻辑 ===
                    # 计算预测对应的真实frame范围
                    buffer_length = len(list(self.token_buffer))
                    input_end_idx = buffer_length - self.batch_size + 1 + i
                    prediction_frame_start = input_end_idx  # 预测从这个frame开始
                    prediction_frame_count = self.inference_length
                    
                    # 检查是否当前有足够的真实tokens进行比较
                    if prediction_frame_start + prediction_frame_count <= buffer_length:
                        # 有足够的真实tokens，立即计算准确率
                        true_tokens = []
                        current_buffer = list(self.token_buffer)
                        for frame_idx in range(prediction_frame_start, 
                                             prediction_frame_start + prediction_frame_count):
                            if frame_idx < len(current_buffer):
                                true_tokens.extend(current_buffer[frame_idx])
                        
                        # 计算准确率
                        if true_tokens and token_ids:
                            min_length = min(len(token_ids), len(true_tokens))
                            matches = sum(1 for j in range(min_length) 
                                        if token_ids[j] == true_tokens[j])
                            
                            self.global_metrics['correct_predictions'] += matches
                            self.global_metrics['total_predictions'] += min_length
                            batch_eval_matches += matches
                            batch_eval_total += min_length
                            # Debug dump a few pairs for inspection
                            if self.debug_dump_enabled and self._debug_dump_count < self.debug_dump_max:
                                try:
                                    os.makedirs(os.path.dirname(self.debug_dump_path), exist_ok=True)
                                    with open(self.debug_dump_path, "a") as f:
                                        f.write(_json.dumps({
                                            "req_idx": int(i),
                                            "pred_first64": token_ids[:64],
                                            "truth_first64": true_tokens[:64],
                                            "pred_len": len(token_ids),
                                            "truth_len": len(true_tokens),
                                            "matches": int(matches),
                                            "compare_len": int(min_length),
                                        }) + "\n")
                                    self._debug_dump_count += 1
                                except Exception:
                                    pass
                    else:
                        # 需要等待未来的frames，缓存这个预测
                        self.prediction_cache.append({
                            'predicted_tokens': token_ids,
                            'ground_truth_start_idx': prediction_frame_start,
                            'ground_truth_frame_count': prediction_frame_count,
                            'batch_idx': i
                        })
                
                if batch_token_count > 0:
                    # 更新全局困惑度（按原始逻辑）
                    self.global_metrics['total_log_prob_sum'] += batch_log_prob_sum
                    self.global_metrics['total_tokens'] += batch_token_count
                    
                    # 全局平均困惑度：exp(-总logprobs和 / 总token数)
                    global_avg_log_prob = self.global_metrics['total_log_prob_sum'] / self.global_metrics['total_tokens']
                    self.global_metrics['global_perplexity'] = math.exp(-global_avg_log_prob)
                    
                    # 当前批次困惑度（按原始逻辑）
                    batch_avg_log_prob = batch_log_prob_sum / batch_token_count
                    batch_perplexity = math.exp(-batch_avg_log_prob)
                    self.global_metrics['batch_perplexities'].append(batch_perplexity)
                    self.global_metrics['recent_perplexities'].append(batch_perplexity)
                    self.global_metrics['batch_count'] += 1
                
                # 更新全局准确率
                if self.global_metrics['total_predictions'] > 0:
                    self.global_metrics['global_accuracy'] = (
                        self.global_metrics['correct_predictions'] /
                        self.global_metrics['total_predictions']
                    )
                if batch_eval_total > 0:
                    batch_acc = batch_eval_matches / batch_eval_total
                    self.global_metrics['recent_accuracies'].append(batch_acc)
                    self.global_metrics['batch_accuracies'].append(batch_acc)

    def _get_windowed_averages(self):
        """获取窗口移动平均"""
        with nvtx.range("InferenceActor._get_windowed_averages"):
            with self.metrics_lock:
                recent_perp = list(self.global_metrics['recent_perplexities'])
                recent_acc = list(self.global_metrics['recent_accuracies'])
                
                window_avg_perplexity = (
                    sum(recent_perp) / len(recent_perp) if recent_perp else None
                )
                window_avg_accuracy = (
                    sum(recent_acc) / len(recent_acc) if recent_acc else None
                )
                
                return window_avg_perplexity, window_avg_accuracy

    def _init_image_encoder(self) -> None:
        with nvtx.range("InferenceActor._init_image_encoder"):
            print("[InferenceActor] Loading VQGAN encoder …")
            # Simple, direct HF VQGAN on the actor device (no TRT path)
            self.encoder_is_trt = False
            self.encoder_device = self.device
            self.encoder = (
                VQGANModel.from_pretrained(RAW_VQGAN_PATH)
                .to(self.encoder_device)
                .eval()
            )
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ]
            )

    def _init_llm_engine(self) -> None:
        with nvtx.range("InferenceActor._init_llm_engine"):
            # Instrument which vLLM we are actually importing
            try:
                import vllm as _v
                vllm_loc = getattr(_v, "__file__", "<unknown>")
                vllm_ver = getattr(_v, "__version__", getattr(getattr(_v, "version", None), "__version__", "<unknown>"))
                print(f"[InferenceActor] vLLM import: version={vllm_ver}, file={vllm_loc}")
            except Exception as _e:
                print(f"[InferenceActor] vLLM import probe failed: {_e}")
            print(f"[InferenceActor] Initialising vLLM engine for {self.model_name} …")
    
            # --- Build speculative_config if enabled ---
            spec_config = None
            if self.use_speculative_decoding:
                print(f"[InferenceActor] Speculative Decoding ENABLED (method={self.spec_method}).")
                print(f"  -> Num Speculative Tokens (γ): {self.num_spec_tokens}")
                if self.spec_method == "ngram":
                    spec_config = {
                        "method": "ngram",
                        "num_speculative_tokens": self.num_spec_tokens,
                        "prompt_lookup_min": self.prompt_lookup_min,
                        "prompt_lookup_max": self.prompt_lookup_max,
                    }
                    print(f"  -> N-gram params: lookup_min={self.prompt_lookup_min}, lookup_max={self.prompt_lookup_max}")
                else:
                    # Default to EAGLE-3 style draft model proposals
                    method = "eagle3" if self.spec_method in ("eagle3", "mqa", "draft") else self.spec_method
                    print(f"  -> Draft Model: {self.spec_draft_model} (method={method})")
                    spec_config = {
                        "method": method,
                        "model": self.spec_draft_model,
                        "num_speculative_tokens": self.num_spec_tokens,
                        "draft_tensor_parallel_size": 1,
                        # Allow fallback to batch expansion by disabling MQA scorer when requested
                        "disable_mqa_scorer": bool(self.spec_disable_mqa_scorer),
                    }
                    if self.spec_vocab_mapping_path:
                        spec_config["vocab_mapping_path"] = self.spec_vocab_mapping_path
                        print(f"  -> Using vocab_mapping_path: {self.spec_vocab_mapping_path}")
                    if self.spec_disable_mqa_scorer:
                        print("  -> MQA scorer DISABLED: using batch expansion for proposal scoring.")
 
            # Prefer keeping CUDA Graphs even with MQA scorer on vLLM>=0.11.
            # Allow override via env: EAGLE_FORCE_EAGER=1 to disable graphs.
            enforce_eager_env = os.getenv("EAGLE_FORCE_EAGER", "0").lower()
            enforce_eager_flag = (enforce_eager_env in ("1", "true", "yes"))

            # Build single engine; enable LoRA path based on configuration only
            # (do not rebuild engine dynamically here).
            self._spec_config = spec_config
            enable_lora_flag = bool(self.max_loras) and self.max_loras > 0
            engine_args = EngineArgs(
                model=self.model_name,
                tokenizer=self.model_name,
                enable_lora=enable_lora_flag,
                max_lora_rank=self.lora_rank if enable_lora_flag else 0,
                max_loras=self.max_loras if enable_lora_flag else 0,
                gpu_memory_utilization=self.gpu_memory_utilisation,
                tensor_parallel_size=self.tensor_parallel_size,
                enforce_eager=enforce_eager_flag,
                speculative_config=spec_config,
            )
            self.engine = LLMEngine.from_engine_args(engine_args)
            self._engine_enable_lora = enable_lora_flag

            if enforce_eager_flag:
                print("[InferenceActor] enforce_eager=True (MQA scorer path). CUDA Graph disabled for this engine.")
            else:
                print("[InferenceActor] enforce_eager=False. CUDA Graph/async output processor may be enabled.")
            
            try:
                mc = getattr(self.engine, "model_config", None)
                if mc and getattr(mc, "hf_config", None):
                    print("[vLLM] target.hf_config.vocab_size =", mc.hf_config.vocab_size)
                    print("[vLLM] target.hf_config.num_hidden_layers =", mc.hf_config.num_hidden_layers)
            except Exception as e:
                print("[vLLM] model_config not accessible:", e)

            try:
                tok = getattr(self.engine, "tokenizer", None)
                if tok and hasattr(tok, "get_vocab_size"):
                    print("[vLLM] tokenizer.get_vocab_size() =", tok.get_vocab_size())
            except Exception as e:
                print("[vLLM] tokenizer info not accessible:", e)

            mode_str = "ENABLED" if self._engine_enable_lora else "DISABLED"
            print(f"[InferenceActor] vLLM engine initialised (LoRA {mode_str}) on {self.device}.")
            if self.use_speculative_decoding:
                print("[InferenceActor] Speculative decoding is active in the vLLM engine.")



    def _init_buffers(self) -> None:
        with nvtx.range("InferenceActor._init_buffers"):
            self.token_buffer = deque(maxlen=(self.context_length + self.batch_size - 1) * ENCODING_SIZE)
            self.path_buffer = deque(maxlen=self.batch_size)

    def _save_lora_weights(self, state_dict, path) -> None:
        with nvtx.range("InferenceActor._save_lora_weights"):
            os.makedirs(path, exist_ok=True)
            weights_path = os.path.join(path, "adapter_model.bin")
            torch.save(state_dict, weights_path)
            adapter_config = {
                "base_model_name_or_path": self.model_name, "peft_type": "LORA",
                "task_type": "CAUSAL_LM", "r": self.lora_rank, "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout, "target_modules": self.target_modules, "bias": "none",
            }
            with open(os.path.join(path, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f, indent=2)
            print(f"[InferenceActor] LoRA adapter saved to {weights_path}")

    def tokenize_images_batch(self, image_paths):
        with nvtx.range("InferenceActor.tokenize_images_batch"):
            all_tokens = []
            images = []
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    # Ensure the input is on the same device as the encoder
                    transformed_image = self.image_transform(image).to(self.encoder_device)
                    images.append(transformed_image)
                except Exception as e:
                    print(f"[InferenceActor] Failed to process image at path {image_path}: {e}")
                    all_tokens.append([0] * ENCODING_SIZE)
                    continue
            if images:
                with torch.no_grad():
                    with nvtx.range("vqgan_encoder.encode"):
                        # Single-shot batch encode via HF VQGAN (no TRT, no dtype juggling)
                        batch_tensor = torch.stack(images, dim=0)
                        _, tokens_batch = self.encoder.encode(batch_tensor)
                        for tokens in tokens_batch:
                            token_list = tokens.detach().to("cpu").numpy().astype(np.int32).tolist()
                            all_tokens.append(token_list)
            return all_tokens

    def tokenize_image(self, image_path: str) -> dict:
        with nvtx.range("InferenceActor.tokenize_image"):
            return self.tokenize_images_batch([image_path])[0]

    def check_for_updates(self):
        with nvtx.range("InferenceActor.check_for_updates"):
            if (not self.continuous_lora_update) and self._has_applied_lora_update:
                return
            with nvtx.range("ray.get(check_update_status)"):
                update_available = ray.get(self.weights_actor.check_update_status.remote())
            
            if not update_available:
                return
            
            with nvtx.range("ray.get(get_weights)"):
                weights_data = ray.get(self.weights_actor.get_weights.remote())

            with self.lora_update_lock:
                new_step = weights_data["step"]
                if new_step == 0 or new_step <= self.last_weight_update_step:
                    return
                print(f"[InferenceActor] New weights found (Step {new_step}). Updating LoRA adapter...")
                lora_state_dict = weights_data["weights"]
                new_lora_id = f"step_{new_step}"
                new_lora_path = os.path.join(self.lora_save_dir, new_lora_id)
                try:
                    os.makedirs(os.path.dirname(new_lora_path), exist_ok=True)
                    self._save_lora_weights(lora_state_dict, new_lora_path)
                    print(f"[InferenceActor] Saved LoRA weights to {new_lora_path}")
                except TypeError as e:
                    print(f"[InferenceActor] TypeError saving LoRA weights for step {new_step}: {e}")
                    return
                self.current_lora_id = new_lora_id
                self.current_lora_path = new_lora_path
                self.last_weight_update_step = new_step
                if not self._has_applied_lora_update:
                    self._has_applied_lora_update = True
                # Engine remains unchanged; LoRA will be applied via LoRARequest per request

    def infer_batch(self, input_ids_batch: list[list[np.int32]]):
        with nvtx.range("InferenceActor.infer_batch"):
            if not self.engine:
                return [] 
            
            lora_request = None
            if self.current_lora_id != "base_model" and self.last_weight_update_step >= 0:
                lora_request = LoRARequest(
                    lora_name=self.current_lora_id, lora_int_id=self.last_weight_update_step,
                    lora_local_path=self.current_lora_path
                )
            # Stabilize EAGLE acceptance: more deterministic and block chat header tokens.
            if self.use_speculative_decoding:
                sp_temperature = 0.0
                sp_top_p = 1.0
                # Avoid extra logits processors in vLLM>=0.11 which can
                # interfere with speculative acceptance.
                sp_bad_words = None
            else:
                sp_temperature = self.config["temperature"]
                sp_top_p = self.config["top_p"]
                sp_bad_words = None

            sampling_params = SamplingParams(
                temperature=sp_temperature,
                top_p=sp_top_p,
                max_tokens=self.inference_length * ENCODING_SIZE,
                logprobs=1,
                bad_words=sp_bad_words,
                # Critical for identity tokenizer: id=2 is a valid VQGAN code but is EOS in base config.
                # Avoid premature stop by ignoring EOS and clearing stop ids.
                ignore_eos=True,
                stop_token_ids=[],
            )
            request_ids, results = [], {}
            try:
                with nvtx.range("vllm_add_and_process_requests"):
                    for i, input_ids in enumerate(input_ids_batch):
                        request_id = f"inf-{os.urandom(4).hex()}-{i}"
                        request_ids.append(request_id)
                        self.engine.add_request(
                            request_id=request_id, prompt=TokensPrompt(prompt_token_ids=input_ids),
                            params=sampling_params, lora_request=lora_request
                        )
                    with nvtx.range("vllm_engine_step_loop"):
                        while self.engine.has_unfinished_requests():
                            request_outputs = self.engine.step()
                            for request_output in request_outputs:
                                if request_output.finished:
                                    request_id = request_output.request_id
                                    out = request_output.outputs[0]
                                    # Use the chosen token ids from vLLM, not an arbitrary top-k entry
                                    token_ids = list(out.token_ids)
                                    # Find the logprob of the chosen token at each step
                                    chosen_logprobs = []
                                    for tid, cand_dict in zip(token_ids, out.logprobs):
                                        lp_obj = cand_dict.get(tid)
                                        chosen_logprobs.append(lp_obj.logprob if lp_obj is not None else 0.0)
                                    # Perplexity over chosen tokens (avoid div by zero)
                                    denom = max(1, len(chosen_logprobs))
                                    perplexity = math.exp(-sum(chosen_logprobs) / denom)
                                    results[request_id] = (token_ids, chosen_logprobs, perplexity)
                
                ordered_results = []
                for i, request_id in enumerate(request_ids):
                    token_ids, logprobs, perplexity = results[request_id]
                    ordered_results.append({
                        "frame_path": self.path_buffer[i], "token_ids": token_ids, "logprobs": logprobs,
                        "perplexity": perplexity, "lora_step": self.last_weight_update_step, "lora_id": self.current_lora_id,
                    })
                return ordered_results
            except Exception as e:
                # 打印出致命错误，这样我们就能在日志中看到它！
                print(f"\n\n[InferenceActor] ########## CRITICAL ERROR in infer_batch ##########")
                print(f"[InferenceActor] Exception Type: {type(e).__name__}")
                print(f"[InferenceActor] Exception Details: {e}")
                
                # 打印完整的错误堆栈，这对于调试至关重要
                import traceback
                traceback.print_exc()
                
                print(f"[InferenceActor] #################################################\n\n")
                # 若 EngineCore 已死亡，尝试重启引擎，避免后续一直失败
                try:
                    need_restart = isinstance(e, EngineDeadError) or "EngineDeadError" in str(e)
                    if need_restart:
                        print("[InferenceActor] Detected EngineDeadError, restarting vLLM engine…")
                        try:
                            if hasattr(self, "engine") and self.engine is not None:
                                try:
                                    self.engine.shutdown()
                                except Exception:
                                    pass
                                self.engine = None
                            self._init_llm_engine()
                            print("[InferenceActor] vLLM engine restarted successfully.")
                        except Exception as re:
                            print(f"[InferenceActor] Engine restart failed: {re}")
                except Exception:
                    pass
                # 仍然返回空列表，以防整个系统崩溃，但现在我们知道了原因
                return []

    def infer(self, input_ids: list[np.int32]):
        with nvtx.range("InferenceActor.infer"):
            return self.infer_batch([input_ids])

    def __call__(self, image_path: str):
        with nvtx.range(f"InferenceActor.__call__ (count:{self.call_count})"):
            # --- 1. 单帧的常规操作 (保持不变) ---
            tokens = self.tokenize_image(image_path)
            self.token_buffer.append(tokens)
            self.path_buffer.append(image_path)
            
            if self.training_actor:
                self.training_actor.add_token.remote(tokens)
                self.check_for_updates()
            
            self.call_count += 1
            
            results = []
            t_inference_cycle = 0.0 # 初始化
            mem_peak_gb = None

            # --- 2. 宏观推理周期与精确计时 ---
            if self.call_count % self.batch_size == 0: # self.batch_size 是 inference_batch_size
                
                # a. 计时开始
                if self.device.type == "cuda":
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                self.start_new_cycle_tracking()
                cycle_start_time = time.time()
                
                with nvtx.range("prepare_and_run_inference_batch"):
                    self._evaluate_cached_predictions()
                    
                    buffer_list, buffer_length = list(self.token_buffer), len(self.token_buffer)
                    batch = [
                        np.concatenate(buffer_list[max(0, i - self.context_length) : i]).tolist()
                        for i in range(buffer_length - self.batch_size + 1, buffer_length + 1)
                    ]
                    results = self.infer_batch(batch)
                    if results:
                        self._update_global_metrics(results)
                
                # b. 计时结束
                cycle_end_time = time.time()
                t_inference_cycle = cycle_end_time - cycle_start_time
                if self.device.type == "cuda":
                    try:
                        mem_peak_bytes = torch.cuda.max_memory_allocated()
                        mem_peak_gb = mem_peak_bytes / (1024 ** 3)
                    except Exception:
                        mem_peak_gb = None

            # --- 3. 准备返回结果和日志 (基于您的原始版本) ---
            final_results = []
            
            # 只有在 infer_batch 实际运行时，才会有 results
            if results:
                # 修正吞吐量计算
                total_output_tokens = self.batch_size * self.inference_length * ENCODING_SIZE
                tokens_per_second = total_output_tokens / t_inference_cycle if t_inference_cycle > 0 else 0
                
                # 摊分延迟
                latency_amortized = t_inference_cycle / self.batch_size if self.batch_size > 0 else 0
            
                current_config_tuple = None
                current_version = None
                # Windowed quality (used for logging and scheduler)
                cycle_perf = self.get_last_cycle_performance()
                with self.metrics_lock:
                    global_accuracy = self.global_metrics.get("global_accuracy", None)
                    total_predictions = self.global_metrics.get("total_predictions", 0)
                    global_perplexity = self.global_metrics.get("global_perplexity", None)
                    total_tokens = self.global_metrics.get("total_tokens", 0)
                cycle_accuracy = cycle_perf.get("accuracy")
                cycle_perplexity = cycle_perf.get("perplexity")
                cycle_total_predictions = cycle_perf.get("total_predictions")
                acc_value = (
                    cycle_accuracy
                    if cycle_accuracy is not None
                    else (global_accuracy if total_predictions > 0 else None)
                )
                perp_value = (
                    cycle_perplexity
                    if cycle_perplexity is not None
                    else (global_perplexity if total_tokens > 0 else None)
                )
                if self.scheduler_actor and not self.disable_dynamic_scheduling:
                    with self.config_update_lock:
                        current_config_tuple = (self.context_length, self.inference_length)
                        current_version = self.config_version

                spec_metrics = self._collect_spec_metrics_delta()
                # Unified runtime drift signal: cycle-level Jensen-Shannon Divergence
                # between consecutive generated token-id distributions.
                drift_mean = self._compute_cycle_jsd(results)
                if drift_mean is None:
                    drift_mean = self._safe_mean(
                        [
                            (r.get("token_drift_mean") if isinstance(r, dict) else None)
                            for r in results
                        ]
                    )
                if drift_mean is None:
                    drift_mean = self._safe_mean(
                        [
                            (r.get("drift_jsd") if isinstance(r, dict) else None)
                            for r in results
                        ]
                    )
                if drift_mean is None:
                    drift_mean = self._safe_mean(
                        [
                            (r.get("jsd") if isinstance(r, dict) else None)
                            for r in results
                        ]
                    )
                if drift_mean is None:
                    drift_mean = self._safe_mean(
                        [
                            (r.get("jsd_mean") if isinstance(r, dict) else None)
                            for r in results
                        ]
                    )
                perp_report = perp_value

                summary = {
                    "latency": latency_amortized,
                    "tokens_per_second": tokens_per_second,
                    "accuracy": acc_value,
                    "global_accuracy": global_accuracy,
                    "perplexity_cycle": perp_report,
                    "global_perplexity": global_perplexity,
                    "t_inference_cycle": t_inference_cycle,
                    "mem_peak_gb": mem_peak_gb,
                    "context_length": self.context_length,
                    "inference_length": self.inference_length,
                    "config_version": self.config_version,
                    "token_drift_mean": drift_mean,
                    "drift_jsd": drift_mean,
                    "jsd": drift_mean,
                    "jsd_mean": drift_mean,
                    "accuracy_total_predictions": cycle_total_predictions,
                }
                for key in (
                    "acceptance_rate",
                    "spec_accept_mean",
                    "spec_reverify_per_step",
                    "spec_draft_ms_per_step",
                    "spec_verify_ms_per_step",
                    "accepted_tokens_per_step",
                    "rejected_tokens_per_step",
                    "reject_ratio",
                ):
                    if key in spec_metrics:
                        summary[key] = spec_metrics[key]

                for result in results:
                    result.update({k: v for k, v in summary.items() if v is not None})
                    final_results.append(result)

                if self.wandb_run and not self._wandb_logging_disabled:
                    # 获取全局指标（这部分与您的版本完全相同）
                    with self.metrics_lock:
                        global_perplexity = self.global_metrics.get('global_perplexity', 0.0)
                        global_accuracy = self.global_metrics.get('global_accuracy', 0.0)
                    # 构造日志字典 (仅每个周期一条)
                    log_data = {
                        "inference/perplexity": perp_report,
                        "inference/latency": latency_amortized,
                        "inference/tokens_per_second": tokens_per_second,
                        "inference/t_inference_cycle": t_inference_cycle,
                        "inference/global_perplexity": global_perplexity,
                        "inference/global_accuracy": global_accuracy,
                    }
                    if mem_peak_gb is not None:
                        log_data["inference/mem_peak_gb"] = mem_peak_gb
                    try:
                        self.wandb_run.log(log_data)
                    except Exception as exc:
                        # In long-running background jobs W&B attach may race with
                        # service lifecycle; keep inference alive instead of failing.
                        self._wandb_logging_disabled = True
                        print(
                            f"[InferenceActor] WARN: disable wandb logging after exception: {exc}"
                        )
                
                # Scheduler reporting (single aggregated sample per batch)
                if self.scheduler_actor and not self.disable_dynamic_scheduling and current_config_tuple is not None:
                    metrics = {
                        "latency": latency_amortized,
                        "tokens_per_second": tokens_per_second,
                        "perplexity": perp_report,
                        "accuracy": acc_value,
                        "accuracy_total_predictions": cycle_total_predictions,
                        "mem_peak_gb": mem_peak_gb,
                        "token_drift_mean": drift_mean,
                        "drift_jsd": drift_mean,
                        "jsd": drift_mean,
                        "jsd_mean": drift_mean,
                        # staleness_steps could be added later if runtime computes it.
                    }
                    if spec_metrics:
                        metrics.update(spec_metrics)
                    self.scheduler_actor.report_inference_metrics.remote(
                        metrics, current_config_tuple, current_version
                    )
            
            return final_results
        
    def get_global_metrics_summary(self):
        """获取全局指标摘要"""
        with nvtx.range("InferenceActor.get_global_metrics_summary"):
            with self.metrics_lock:
                return {
                    "global_perplexity": self.global_metrics['global_perplexity'],
                    "global_accuracy": self.global_metrics['global_accuracy'],
                    "total_tokens": self.global_metrics['total_tokens'],
                    "total_predictions": self.global_metrics['total_predictions'],
                    "total_batches": self.global_metrics['batch_count'],
                    "window_size": self.global_metrics['window_size'],
                }
    def start_new_cycle_tracking(self):
        """
        【新增】在新的调度周期开始时调用。
        记录当前全局指标的快照，作为本周期的计算基线。
        """
        with self.metrics_lock:
            print("[InferenceActor] Taking snapshot of current metrics for new cycle tracking...")
            self.cycle_metrics_snapshot['total_log_prob_sum'] = self.global_metrics['total_log_prob_sum']
            self.cycle_metrics_snapshot['total_tokens'] = self.global_metrics['total_tokens']
            self.cycle_metrics_snapshot['correct_predictions'] = self.global_metrics['correct_predictions']
            self.cycle_metrics_snapshot['total_predictions'] = self.global_metrics['total_predictions']

    def get_last_cycle_performance(self):
        """
        【新增】在调度周期结束时调用。
        通过计算当前全局指标与周期初快照的差值，得出该周期的性能，绝不修改全局指标。
        """
        with self.metrics_lock:
            # 1. 计算自上次快照以来的增量
            cycle_log_prob_sum = self.global_metrics['total_log_prob_sum'] - self.cycle_metrics_snapshot.get('total_log_prob_sum', 0.0)
            cycle_tokens = self.global_metrics['total_tokens'] - self.cycle_metrics_snapshot.get('total_tokens', 0)
            cycle_correct_predictions = self.global_metrics['correct_predictions'] - self.cycle_metrics_snapshot.get('correct_predictions', 0)
            cycle_total_predictions = self.global_metrics['total_predictions'] - self.cycle_metrics_snapshot.get('total_predictions', 0)

            # 2. 根据增量安全地计算周期内的指标
            cycle_perplexity = None
            if cycle_tokens > 0:
                cycle_avg_log_prob = cycle_log_prob_sum / cycle_tokens
                cycle_perplexity = math.exp(-cycle_avg_log_prob)

            cycle_accuracy = None
            if cycle_total_predictions > 0:
                cycle_accuracy = cycle_correct_predictions / cycle_total_predictions

            acc_str = f"{cycle_accuracy:.4f}" if cycle_accuracy is not None else "None"
            perp_str = f"{cycle_perplexity:.4f}" if cycle_perplexity is not None else "None"
            print(f"[InferenceActor] Calculated last cycle performance: Accuracy={acc_str}, Perplexity={perp_str}")
            
            return {
                "accuracy": cycle_accuracy,
                "perplexity": cycle_perplexity,
                "total_predictions": cycle_total_predictions,
            }

    def reset(self):
        with nvtx.range("InferenceActor.reset"):
            print("[InferenceActor] Resetting state...")
            with self.lora_update_lock:
                self.call_count = 0
                self.current_lora_path = None
                self.current_lora_id = "base_model"
                self.last_weight_update_step = -1
                self._has_applied_lora_update = False
                self.token_buffer.clear()
                self.path_buffer.clear()
                # 重置全局指标和缓存
                self._init_global_metrics()
                self._prev_cycle_token_hist = None
                self.config_version = 0
                # 同样重置周期快照，以防万一
                self.cycle_metrics_snapshot = {
                    'total_log_prob_sum': 0.0,
                    'total_tokens': 0,
                    'correct_predictions': 0,
                    'total_predictions': 0,
                }
                # Gracefully restart vLLM engine to avoid using a dead EngineCore after OOM/interrupt
                try:
                    if hasattr(self, "engine") and self.engine is not None:
                        print("[InferenceActor] Shutting down existing vLLM engine…")
                        try:
                            self.engine.shutdown()
                        except Exception:
                            pass
                        self.engine = None
                    print("[InferenceActor] Re-initialising vLLM engine after reset…")
                    self._init_llm_engine()
                except Exception as e:
                    print(f"[InferenceActor] Engine restart failed during reset: {e}")
            print("[InferenceActor] Reset complete.")
        
    def clear_saved_loras(self):
        with nvtx.range("InferenceActor.clear_saved_loras"):
            print(f"[InferenceActor] Cleaning up resources...")
            try:
                if os.path.exists(self.lora_save_dir):
                    shutil.rmtree(self.lora_save_dir)
                    print(f"[InferenceActor] Successfully deleted the directory {self.lora_save_dir}")
            except Exception as e:
                print(f"[InferenceActor] Error while cleaning up resources: {e}")
            if self.wandb_run:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass
