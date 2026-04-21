# actors/training_actor.py (Complete, final, and safe to replace)

import json
import time
import threading
from pathlib import Path
import os
import numpy as np
import ray
import hashlib
import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx
from accelerate import Accelerator, DeepSpeedPlugin
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

# 兼容性兜底：部分环境的 accelerate.utils 不提供 DummyOptim
try:  # pragma: no cover
    from accelerate.utils import DummyOptim  # type: ignore
except Exception:  # pragma: no cover
    from torch.optim import Optimizer

    class DummyOptim(Optimizer):
        """极简占位 Optimizer，用于避免缺失 DummyOptim 时的 ImportError。"""

        def __init__(self):
            super().__init__(params=[], defaults={})

        def step(self, closure=None):
            return None

        def zero_grad(self, set_to_none: bool = False):
            pass
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from lvm_tokenizer.utils import ENCODING_SIZE
from utils.prepare import SAVE_DIR_FT
from collections import deque 
from actors.scheduler_actor import SchedulerActor


def _ensure_single_process_dist_env() -> None:
    # DeepSpeed checks these variables even for single-process usage.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


_ensure_single_process_dist_env()
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

@ray.remote(num_gpus=1)
class TrainingActor:
    def __init__(self, config, shared_weights_actor, scheduler_actor=None):
        with nvtx.range("TrainingActor.__init__"):
            _ensure_single_process_dist_env()
            self.config = config
            self.shared_weights_actor = shared_weights_actor
            self._load_config()
            self.current_step = 0
            self._lock = threading.Lock()
            self.uuid = 29500 + int(hashlib.md5(self.run_name.encode()).hexdigest(), 16) % 10000
            self._init_logs()
            self.training_cycle_id = 0
            self.last_training_cycle_metrics = None
            self.last_train_loss = None
            self._init_accelerator()
            self._init_tokenizer_and_model()
            self._init_optimizer()
            self.init_state_dir = f"./tmp/initial_state_{self.uuid}"
            with nvtx.range("accelerator.save_state (init)"):
                self.accelerator.save_state(output_dir=self.init_state_dir)
            
            # Streaming batch buffer
            self.buffer = []
            self.buffer_threshold = self.context_length + self.batch_size - 1
            # Optional latency tracking (may be used by reset); always initialize
            self.step_latencies = []
            
            # Scheduler handle is optional; dynamic scheduling may be disabled.
            self.scheduler_actor = scheduler_actor

            # Threading primitives for async training
            self._stop_event = threading.Event()
            self._training_thread = threading.Thread(
                target=self._training_loop,
                daemon=True,
            )
            self._training_thread.start()
            self.config_version = 0
            print(
                f"TrainingActor initialized on {self.device} "
                f"(Accelerator state: {self.accelerator.state})"
            )

    def _load_config(self):
        c = self.config
        self.model_name = c["model_name"]
        self.ds_config_path = c["ds_config_path"]
        self.mixed_precision = c["mixed_precision"]
        self.gradient_accumulation_steps = c["gradient_accumulation_steps"]
        self.learning_rate = c["learning_rate"]
        self.weight_decay = c["weight_decay"]
        self.context_length = c["context_length"] + 1
        self.inference_length = c.get("inference_length")
        self.batch_size = c["training_batch_size"]
        self.train_epochs = c["train_epochs"]
        self.max_grad_norm = c["max_grad_norm"]
        self.gradient_checkpointing = c["gradient_checkpointing"]
        self.lora_rank = c["lora_rank"]
        self.lora_alpha = c["lora_alpha"]
        self.target_modules = c["target_modules"]
        self.lora_dropout = c["lora_dropout"]
        self.wandb_run = c.get("wandb_run")
        self.wandb_config = c.get("wandb_config", {})
        self._wandb_logging_disabled = False
        self.console_log_freq = c["console_log_freq"]
        self.wandb_log_freq = c["wandb_log_freq"]
        self.weight_sharing_freq = c["weight_sharing_freq"]
        self.disable_weight_sharing = bool(c.get("disable_weight_sharing", False))
        self.logs_dir = Path(c["training_logs_dir"])
        self.run_name = c["run_name"]
        self.max_step = c.get("max_step", 100000000)
        self.disable_dynamic_scheduling = c.get("disable_dynamic_scheduling", False)
        if self.disable_dynamic_scheduling:
            print("[TrainingActor] Dynamic scheduling reporting is DISABLED.")
        if self.disable_weight_sharing:
            print("[TrainingActor] Runtime LoRA weight sharing is DISABLED (no updates will be published to inference).")
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
        actor_run_name = f"{parent_name}.train"

        for attempt in range(1, 4):
            try:
                self.wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=actor_run_name,
                    group=run_group,
                    job_type="training_actor",
                    mode=run_mode,
                    reinit=True,
                )
                print(
                    f"[TrainingActor] WandB initialized (name={actor_run_name}, group={run_group})."
                )
                return
            except Exception as exc:
                self.wandb_run = None
                if attempt >= 3:
                    print(f"[TrainingActor] WARN: wandb init failed: {exc}")
                    return
                time.sleep(0.5)

            
    def _init_logs(self):
        # Ensure log directory exists and pick a stable filename based on run_name.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / (
            f"{self.run_name}.jsonl" or f"training_logs_{int(time.time())}.jsonl"
        )

    def _init_accelerator(self):
        with nvtx.range("TrainingActor._init_accelerator"):
            if not self.ds_config_path:
                raise ValueError("DeepSpeed config path must be provided.")
            ds_path = Path(self.ds_config_path)
            if not ds_path.is_file():
                raise FileNotFoundError(
                    f"DeepSpeed config file not found at {self.ds_config_path}"
                )
            if dist.is_available() and not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method=f"tcp://localhost:{self.uuid}",
                    rank=0,
                    world_size=1,
                )
                print(f"[TrainingActor] Initialized process group with host: {f'tcp://localhost:{self.uuid}'}")
            ds_plugin = DeepSpeedPlugin(hf_ds_config=self.ds_config_path)
            self.accelerator = Accelerator(
                mixed_precision=self.mixed_precision,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                deepspeed_plugin=ds_plugin,
                # We log to W&B manually to avoid tracker attach races in Ray actors.
                log_with=None,
            )
            self.device = self.accelerator.device
            assert self.device.type == "cuda", "TrainingActor must be run on a GPU"

    def _init_tokenizer_and_model(self):
        with nvtx.range("TrainingActor._init_tokenizer_and_model"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
            lora_cfg = LoraConfig(
                r=self.lora_rank, lora_alpha=self.lora_alpha,
                target_modules=self.target_modules, lora_dropout=self.lora_dropout,
                bias="none", task_type="CAUSAL_LM",
            )
            self.base_model.enable_input_require_grads()
            self.model = get_peft_model(self.base_model, lora_cfg)
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                print("[TrainingActor] Gradient checkpointing enabled.")

    def _init_optimizer(self):
        with nvtx.range("TrainingActor._init_optimizer"):
            _ensure_single_process_dist_env()
            plugin = self.accelerator.state.deepspeed_plugin
            ds_cfg = plugin.deepspeed_config
            OptimCls = DummyOptim if "optimizer" in ds_cfg else torch.optim.AdamW
            self.optimizer = OptimCls(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
            )
            self.lr_scheduler = None # Not using a scheduler in this setup
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            def show_trainables(model):
                names = ["model.embed_tokens.weight", "lm_head.weight"]
                table = {}
                for n, p in model.named_parameters():
                    for key in names:
                        if n.endswith(key):
                            table[key] = (True, p.requires_grad, tuple(p.shape))
                for key in names:
                    print(key, table.get(key, (False, None, None)))
                n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in model.parameters())
                print(f"[PARAM] trainable={n_tr:,} / total={n_all:,}")

            show_trainables(self.model)

    def add_token(self, token_data):
        with nvtx.range("TrainingActor.add_token"):
            with self._lock:
                self.buffer.append(token_data)

    def _training_loop(self):
        with nvtx.range("TrainingActor._training_loop"):
            while not self._stop_event.is_set():
                batch = None
                with self._lock:
                    if len(self.buffer) >= self.buffer_threshold:
                        cycle_start_time = time.time()
                        batch = [
                            np.concatenate(self.buffer[i: i + self.context_length])
                            for i in range(self.batch_size)
                        ]
                        # Trim the buffer
                        self.buffer = self.buffer[self.batch_size:]
                if batch:
                    for _ in range(self.train_epochs):
                        self._train_step(batch)

                    cycle_end_time = time.time()
                    t_train_cycle = cycle_end_time - cycle_start_time

                    total_tokens = (
                        self.train_epochs * self.batch_size * self.context_length * ENCODING_SIZE
                    )
                    avg_step_latency = t_train_cycle / max(self.train_epochs, 1)
                    tokens_per_second = (
                        float(total_tokens) / t_train_cycle if t_train_cycle > 0 else 0.0
                    )

                    if self.scheduler_actor and not self.disable_dynamic_scheduling:
                        with self._lock:
                            current_config_tuple = (self.context_length - 1, self.inference_length)
                            current_version = self.config_version

                        metrics = {
                            "latency": avg_step_latency,
                            "tokens_per_second": tokens_per_second,
                        }
                        self.scheduler_actor.report_training_metrics.remote(
                            metrics, current_config_tuple, current_version
                        )
                        # print(f"[TrainingActor] Reported T_train_cycle: {t_train_cycle:.3f}s for config v{current_version}")

                    lr = None
                    try:
                        if self.optimizer and self.optimizer.param_groups:
                            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                    except Exception:
                        lr = None
                    cycle_metrics = {
                        "event": "training_cycle",
                        "training_cycle_id": self.training_cycle_id,
                        "train_step_end": self.current_step,
                        "avg_step_latency": avg_step_latency,
                        "train_tokens_per_second": tokens_per_second,
                        "train_loss": self.last_train_loss,
                        "training_batch_size": self.batch_size,
                        "learning_rate": lr,
                        "context_length": self.context_length - 1,
                        "context_length_train": self.context_length,
                        "inference_length": self.inference_length,
                        "weight_sharing_freq": self.weight_sharing_freq,
                        "config_version": self.config_version,
                    }
                    with self._lock:
                        self.last_training_cycle_metrics = cycle_metrics
                        self.training_cycle_id += 1
                else:
                    time.sleep(0.01)

    def _prepare_batch(self, batch_data):
        with nvtx.range("_prepare_batch"):
            input_ids = torch.tensor(np.array(batch_data), dtype=torch.long, device=self.device)
            return {"input_ids": input_ids, "labels": input_ids.clone()}

    def _train_step(self, batch_data):
        with nvtx.range(f"train_step_{self.current_step}"):
            # Measure per-step latency.
            step_start_time = time.time()

            self.model.train()
            batch = self._prepare_batch(batch_data)
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.max_grad_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            step_end_time = time.time()
            latency = step_end_time - step_start_time
            if latency < 0:
                latency = 0.0
            self.step_latencies.append(latency)

            # Approximate token workload for this step.
            # Each example concatenates `context_length` frames, each with ENCODING_SIZE tokens.
            tokens_per_example = self.context_length * ENCODING_SIZE
            total_tokens = self.batch_size * tokens_per_example
            tokens_per_second = (
                float(total_tokens) / latency if latency > 0 else 0.0
            )

            avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean().item()
            self.last_train_loss = avg_loss
            log_data = {
                "train/step": self.current_step,
                "train/loss": avg_loss,
                "train/latency": latency,
                "train/tokens_per_second": tokens_per_second,
            }

            if self.accelerator.is_main_process:
                if self.current_step % self.console_log_freq == 0:
                    print(f"Step {self.current_step} | Loss {avg_loss:.4f} | "
                          f"Latency {latency:.3f}s | Tokens/s {tokens_per_second:.1f}")

                if (
                    self.wandb_run
                    and not self._wandb_logging_disabled
                    and self.current_step % self.wandb_log_freq == 0
                ):
                    try:
                        self.wandb_run.log(log_data)
                    except Exception as exc:
                        self._wandb_logging_disabled = True
                        print(
                            "[TrainingActor] WARN: disable wandb logging after exception: "
                            f"{exc}"
                        )

                if (not self.disable_weight_sharing) and (
                    self.current_step % self.weight_sharing_freq == 0
                ):
                    self.accelerator.wait_for_everyone()
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    lora_weights = get_peft_model_state_dict(
                        unwrapped, adapter_name="default"
                    )
                    cpu_weights = {k: v.cpu() for k, v in lora_weights.items()}
                    self.shared_weights_actor.set_weights.remote(
                        {"weights": cpu_weights, "step": self.current_step}
                    )

            self.current_step += 1

            

    def get_current_step(self) -> int:
        """Returns the current training step."""
        return self.current_step

    def get_last_training_cycle_metrics(self):
        with self._lock:
            if not self.last_training_cycle_metrics:
                return None
            return dict(self.last_training_cycle_metrics)

    def update_config(self, new_context_length: int, new_inference_length: int, new_version: int):
        with self._lock:
            new_internal_context_length = new_context_length + 1
            config_changed = (self.context_length != new_internal_context_length)

            if not config_changed and self.config_version == new_version:
                return

            print(f"[TrainingActor] Updating config to v{new_version}. Context_length: {self.context_length} -> {new_internal_context_length}")
            self.context_length = new_internal_context_length
            self.buffer_threshold = self.context_length + self.batch_size - 1
            self.inference_length = new_inference_length
            self.config_version = new_version
            print(f"[TrainingActor] Buffer threshold updated to {self.buffer_threshold}")


    def _save_checkpoint(self, dir_name):
        with nvtx.range("TrainingActor._save_checkpoint"):
            unwrapped_model = self.accelerator.unwrap_model(self.model).merge_and_unload()
            save_dir = os.path.join(SAVE_DIR_FT, dir_name)
            unwrapped_model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            print(f"[TrainingActor] Checkpoint for {dir_name} saved to {save_dir}")

    def stop_training(self):
        with nvtx.range("TrainingActor.stop_training"):
            print("[TrainingActor] Stopping...")
            self._stop_event.set()
            if hasattr(self, "_training_thread"):
                self._training_thread.join()
            print("[TrainingActor] Stopped.")
            if "tream_base" in self.run_name:
                self._save_checkpoint(self.run_name)
            if self.wandb_run:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass

    def reset(self, data_dir):
        with nvtx.range("TrainingActor.reset"):
            print("[TrainingActor] Resetting...")
            with self._lock:
                self.buffer.clear()
                self.step_latencies.clear() # 重置时也清空延迟记录
            with nvtx.range("accelerator.load_state (reset)"):
                self.accelerator.load_state(self.init_state_dir)
            print("[TrainingActor] Reset complete.")

    
