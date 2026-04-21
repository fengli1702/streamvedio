import argparse
import inspect
import ray
from ray.exceptions import RayActorError
import time
import os
import json
import shutil
import wandb
import threading
import atexit
from typing import Any, Callable, Dict, Optional, Tuple
from actors.weights_actor import SharedWeightsActor
from actors.scheduler_actor import SchedulerActor
from actors.training_actor import TrainingActor
from actors.inference_actor import InferenceActor
from lvm_tokenizer.compile_fast_vqgan import compile_fast_vqgan
from lvm_tokenizer.utils import COMPILED_VQGAN_PATH
from utils.utils import get_available_gpus, clear_log_file, get_image_files
# vLLM 0.8.x exposes internal SpecMetrics; 0.10.x removed/relocated it.
# Provide a backward-compatible shim: try import, else define a no-op stub.
try:
    from vllm.spec_decode._spec_metrics import SpecMetrics  # type: ignore
except Exception:
    class SpecMetrics:  # type: ignore
        _path: str | None = None

        @staticmethod
        def flush() -> None:
            return
from tqdm import tqdm

WANDB_PROJECT = "tream"
WANDB_ENTITY = "tream-team"


def tail_and_log_spec_metrics(spec_path: str,
                              run: "wandb.wandb_run.Run",
                              stop_event: threading.Event,
                              flush_every: float = 0.5) -> threading.Thread:
    """Tail JSONL written by vLLM speculative decoding and forward to W&B."""

    def _worker() -> None:
        while not os.path.exists(spec_path):
            time.sleep(0.2)
        with open(spec_path, "r") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(flush_every)
                    if stop_event.is_set():
                        break
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                run.log({
                    "spec/step": rec.get("step", 0),
                    "spec/acceptance_rate": rec.get("acceptance_rate", 0.0),
                    "spec/accepted": rec.get("accepted", 0),
                    "spec/proposed": rec.get("proposed", 0),
                    "spec/gamma": rec.get("gamma", 0),
                    "spec/batch_size": rec.get("batch_size", 0),
                })

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th


def setup_spec_metrics_logging(
        wandb_run_name: str,
        spec_path: str) -> Tuple[threading.Thread, str, Callable[[], None]]:
    run = wandb.run or wandb.init(name=wandb_run_name or "spec-metrics")
    run.define_metric("spec/step")
    run.define_metric("spec/*", step_metric="spec/step")
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    os.environ["VLLM_SPEC_METRICS_PATH"] = spec_path
    stop_event = threading.Event()
    th = tail_and_log_spec_metrics(spec_path, run, stop_event)
    finalized = {"done": False}

    def _finalize() -> None:
        if finalized["done"]:
            return
        finalized["done"] = True
        stop_event.set()
        SpecMetrics.flush()
        try:
            if os.path.exists(spec_path):
                artifact = wandb.Artifact(
                    name=f"{wandb_run_name or run.name}-spec-metrics",
                    type="metrics",
                    metadata={"source": "vllm-spec-decoding"},
                )
                artifact.add_file(spec_path)
                run.log_artifact(artifact)
                wandb.save(spec_path)
        except Exception as e:  # noqa: BLE001
            print(f"[SpecMetrics] finalize failed: {e}")

    atexit.register(_finalize)
    return th, spec_path, _finalize


def drain_scheduler_log_to_wandb(
    scheduler_log_path: str,
    run: Optional["wandb.wandb_run.Run"],
    state: Dict[str, int],
) -> None:
    """Forward new scheduler JSONL rows into the main W&B run."""
    if run is None or not scheduler_log_path:
        return

    if not os.path.exists(scheduler_log_path):
        return

    cursor = int(state.get("cursor", 0))
    last_step = int(state.get("last_step", -1))
    try:
        file_size = os.path.getsize(scheduler_log_path)
    except Exception:
        return
    if file_size < cursor:
        cursor = 0

    try:
        with open(scheduler_log_path, "r", encoding="utf-8") as f:
            f.seek(cursor)
            while True:
                line = f.readline()
                if not line:
                    break
                cursor = f.tell()
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                step = rec.get("scheduler/step")
                try:
                    step_i = int(step) if step is not None else None
                except Exception:
                    step_i = None
                if step_i is not None and step_i <= last_step:
                    continue
                payload: Dict[str, Any] = {}
                for key, value in rec.items():
                    if key == "timestamp":
                        continue
                    payload[key] = value
                if payload:
                    run.log(payload)
                    if step_i is not None:
                        last_step = max(last_step, step_i)
    except Exception as e:
        print(f"[Driver] WARN: drain scheduler log failed: {e}")
    finally:
        state["cursor"] = cursor
        state["last_step"] = last_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TREAM"
    )
    parser.add_argument(
        "--optimization_priority",
        type=str,
        choices=["speed", "quality", "latency", "qnehvi"],
        default="latency",
        help="Scheduler's optimization priority: 'speed' for throughput/latency, 'quality' for accuracy/perplexity.",
    )
    parser.add_argument(
        "--scheduling_cycle_frames",
        type=int,
        default=20,
        help="Number of frames after which the scheduler re-evaluates the configuration.",
    )
    parser.add_argument(
        "--scheduler_window_size",
        type=int,
        default=3,
        help="Scheduler window size for metric aggregation.",
    )
    parser.add_argument(
        "--scheduler_q_batch",
        type=int,
        default=2,
        help="Batch size (q) for qNEHVI acquisition.",
    )
    parser.add_argument(
        "--scheduler_mix_lambda",
        type=float,
        default=0.7,
        help="Mixing weight for utility vs acquisition (0..1).",
    )
    parser.add_argument(
        "--scheduler_target_latency",
        type=float,
        default=None,
        help="SLA latency target in seconds (None keeps default).",
    )
    parser.add_argument(
        "--scheduler_latency_margin",
        type=float,
        default=0.1,
        help="Latency guard band margin in seconds.",
    )
    parser.add_argument(
        "--scheduler_quality_min",
        type=float,
        default=None,
        help="Minimum quality threshold; omit to disable quality constraint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="./saved_models/lvm-llama2-7b",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--input_frames_path",
        type=str,
        default="./data/room_1D",
        help="Directory containing image frames to stream",
    )
    parser.add_argument(
        "--training_logs_dir",
        type=str,
        default="./training_logs",
        help="Directory for training logs (defaults to timestamp-based name)",
    )
    parser.add_argument(
        "--inference_logs_dir",
        type=str,
        default="./inference_logs",
        help="Directory for inference logs",
    )
    parser.add_argument(
        "--disk_min_free_gb",
        type=float,
        default=5.0,
        help="Minimum free disk space (GB). Set <=0 to disable.",
    )
    parser.add_argument(
        "--disk_check_interval",
        type=int,
        default=50,
        help="Check free disk space every N frames.",
    )
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--inference_length",
        type=int,
        default=1,
        help="Number of frames to generate per token",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of finetuning epochs per batch",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA attention rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
        help="Target modules for LoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout probability",
    )
    parser.add_argument( 
        "-gc", "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=4,
        help="Number of images per training sample",
    )
    parser.add_argument(
        "--console_log_freq",
        type=int,
        default=10,
        help="Frequency of console logging",
    )
    parser.add_argument(
        "--wandb_log_freq",
        type=int,
        default=1,
        help="Frequency of wandb logging",
    )
    parser.add_argument(
        "--training_log_stride",
        type=int,
        default=50,
        help="Frames between training cycle log pulls.",
    )
    parser.add_argument(
        "--weight_sharing_freq",
        type=int,
        default=3,
        help="Frequency of weight sharing (in batches)",
    )
    parser.add_argument(
        "--disable_weight_sharing",
        action="store_true",
        help="Do not publish runtime LoRA weights from TrainingActor (training continues, inference stays on last weights).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0, # TODO: idk what this should be
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--ds_config_path",
        type=str,
        default="./utils/ds_config.json",
        help="Path to DeepSpeed config file",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision training mode",
    )
    parser.add_argument( # TODO: lol this is weird code
        "--use_wandb",
        action="store_true",
        default=True,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="WandB run name",
    )
    
    parser.add_argument(
        "--max_loras",
        type=int,
        default=4,
        help="Max concurrent LoRA adapters in vLLM",
    )
    parser.add_argument(
        "--disable_continuous_lora_update",
        action="store_true",
        help="If set, only the first LoRA weight update is applied; later updates are ignored so inference keeps the initial adapter.",
    )
    parser.add_argument(
        "--disable_dynamic_scheduling",
        action="store_true", # 这会创建一个布尔标志，如果命令行中出现 --disable_dynamic_scheduling，则 args.disable_dynamic_scheduling 为 True
        help="If set, disable the dynamic scheduler and run with fixed initial context/inference lengths.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory for vLLM",
    )
    # parser.add_argument(
    #     "--tensor_parallel_size",
    #     type=int,
    #     default=1,
    #     help="Tensor parallelism size for vLLM",
    # )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for inference",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Sampling top-p for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of Ray workers (defaults to number of GPUs)",
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default="",
        help="Ray cluster address (host:port). If set, connect instead of starting a local cluster.",
    )
    parser.add_argument(
        "--ray_port",
        type=int,
        default=0,
        help="Ray head port for local init. Use different ports per process to avoid collisions.",
    )
    parser.add_argument(
        "--ray_temp_dir",
        type=str,
        default="",
        help="Custom Ray temp dir (useful when running multiple local clusters).",
    )
    parser.add_argument(
        "--ray_num_cpus",
        type=float,
        default=None,
        help="Override local Ray CPU resources to reduce worker prestart pressure.",
    )
    parser.add_argument(
        "--ray_namespace",
        type=str,
        default="",
        help="Ray namespace for this run.",
    )
    parser.add_argument(
        "--reset_on_dir_change",
        action="store_true",
        help="Reset model state (destroy/recreate vLLM engine) when switching input subdirectories.",
    )
    parser.add_argument(
        "-train", "--use_training_actor",
        action="store_true",
        help="Enable training actor",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=20000,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=100000000, # 默认值为0，表示不限制
        help="Maximum number of training steps. The program will terminate after this many steps. 0 means no limit.",
    )
    parser.add_argument(
        "--use_speculative_decoding",
        action="store_true",
        help="Enable speculative decoding for inference.",
    )
    # Restrict to EAGLE-3 explicitly for clarity and consistency
    parser.add_argument(
        "--spec_method",
        type=str,
        choices=["eagle3"],
        default="eagle3",
        help="Speculative method (fixed to 'eagle3').",
    )
    parser.add_argument(
        "--spec_draft_model",
        type=str,
        default=None,
        help="Path to the draft model for speculative decoding. Required if --use_speculative_decoding is set.",
    )
    parser.add_argument(
        "--num_spec_tokens",
        type=int,
        default=3,
        help="Number of tokens to speculate (gamma) per decoding step.",
    )
    parser.add_argument(
        "--prompt_lookup_min",
        type=int,
        default=2,
        help="N-gram: minimum prompt lookup length.",
    )
    parser.add_argument(
        "--prompt_lookup_max",
        type=int,
        default=5,
        help="N-gram: maximum prompt lookup length.",
    )
    parser.add_argument(
        "--spec_disable_mqa_scorer",
        action="store_true",
        help="Disable MQA scorer to preserve CUDA Graphs (batch-expansion scoring).",
    )
    parser.add_argument(
        "--disable_spec_metrics_log",
        action="store_true",
        help="Disable writing speculative decoding metrics to disk.",
    )
    parser.add_argument(
        "--spec_vocab_mapping_path",
        type=str,
        default=None,
        help="Path to vocab_mapping .pt file to align draft/target vocab for EAGLE-3.",
    )


    args = parser.parse_args()
    
    # Clear unified inference log file if it exists
    clear_log_file(args.inference_logs_dir, f"{args.wandb_run_name}.jsonl")

    # Initialize wandb for the main process
    if args.use_wandb:
        run = wandb.init(
            project="test-feng1702", # 
            entity="fengli1702-ustc", # 
            name=args.wandb_run_name or f"tream-main-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
            mode="online"
        )
        run.define_metric("scheduler/step")
        run.define_metric("scheduler/*", step_metric="scheduler/step")
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("inference/step")
        run.define_metric("inference/*", step_metric="inference/step")
        print(f"WandB initialized for main process")
        wandb_config = {
            "enabled": True,
            "project": run.project or "test-feng1702",
            "entity": run.entity or "fengli1702-ustc",
            "run_group": args.wandb_run_name or run.name,
            "parent_run_name": run.name,
            "mode": "online",
        }
    else:
        run = None
        wandb_config = {"enabled": False}

    # Default: keep exactly one W&B run per experiment (driver run).
    # Set TREAM_WANDB_ACTOR_RUNS=1 only when you explicitly want extra actor runs.
    actor_wandb_enabled = os.environ.get("TREAM_WANDB_ACTOR_RUNS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    actor_wandb_config = wandb_config if actor_wandb_enabled else {"enabled": False}

    spec_metrics_run_name = args.wandb_run_name or (run.name if run else "")
    spec_metrics_path = None
    if not args.disable_spec_metrics_log:
        spec_metrics_filename = (f"spec_metrics_{spec_metrics_run_name}.jsonl"
                                 if spec_metrics_run_name else
                                 f"spec_metrics_{int(time.time())}.jsonl")
        spec_metrics_path = os.path.join(args.inference_logs_dir,
                                         spec_metrics_filename)
        os.environ["VLLM_SPEC_METRICS_PATH"] = spec_metrics_path

    # Determine GPU/workers
    free_gpus, num_free_gpus = get_available_gpus()
    print(f"Available GPUs: {free_gpus}")
    if args.num_workers and args.num_workers > num_free_gpus:
        raise ValueError(f"Number of workers ({args.num_workers}) is less than available GPUs ({num_free_gpus}). This configuration is not supported.")
    num_workers = args.num_workers or num_free_gpus or 1
    # print(f"Using GPUs: {free_gpus[:num_workers]}")
    
    print(f"Number of workers: {num_workers}")
    # num_train_gpus = math.ceil(num_workers/2)
    # num_inference_gpus = math.floor(num_workers/2)
    num_inference_gpus = 1
    num_train_gpus = num_workers - num_inference_gpus
    assert num_train_gpus + num_inference_gpus == num_workers, "Number of train and inference GPUs must sum to the total number of workers"
    
    # Build a single config dict for both actors
    # Allow LoRA in pure-inference runs for A/B testing; do not force-disable.

    config = {
        "model_name": args.model_name,
        "ds_config_path": args.ds_config_path,
        "mixed_precision": args.mixed_precision,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # Do not pass W&B run objects across Ray actor processes.
        # Each actor will init its own W&B run using this config.
        "wandb_run": None,
        "wandb_config": actor_wandb_config,
        "wandb_log_freq": args.wandb_log_freq,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "lora_dropout": args.lora_dropout,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "context_length": args.context_length,
        "training_batch_size": args.training_batch_size,
        "inference_batch_size": args.inference_batch_size,
        "console_log_freq": args.console_log_freq,
        "weight_sharing_freq": args.weight_sharing_freq,
        "disable_weight_sharing": args.disable_weight_sharing,
        "training_logs_dir": args.training_logs_dir,
        "train_epochs": args.epochs,
        "max_grad_norm": args.max_grad_norm,
        "max_loras": args.max_loras,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": num_inference_gpus,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "inference_length": args.inference_length,
        "run_name": args.wandb_run_name,
        "max_step": args.max_step,
        "disable_dynamic_scheduling": args.disable_dynamic_scheduling,
        "use_speculative_decoding": args.use_speculative_decoding,
        "spec_method": args.spec_method,
        "spec_draft_model": args.spec_draft_model,
        "num_spec_tokens": args.num_spec_tokens,
        "prompt_lookup_min": args.prompt_lookup_min,
        "prompt_lookup_max": args.prompt_lookup_max,
        "spec_disable_mqa_scorer": args.spec_disable_mqa_scorer,
        "spec_vocab_mapping_path": args.spec_vocab_mapping_path,
        "spec_metrics_path": spec_metrics_path,
        "continuous_lora_update": not args.disable_continuous_lora_update,

    }
    
    # Only compile the VQGAN if TensorRT is available and the compiled model doesn't already exist
    trt_available = True
    try:
        import torch_tensorrt  # type: ignore
    except Exception:
        trt_available = False
    if trt_available:
        if not os.path.exists(COMPILED_VQGAN_PATH):
            print(f"Compiled VQGAN not found at {COMPILED_VQGAN_PATH}, compiling now...")
            compile_fast_vqgan()
            print(f"VQGAN compilation complete.")
        else:
            print(f"Using existing compiled VQGAN from {COMPILED_VQGAN_PATH}")
    else:
        print("TensorRT not available; skipping VQGAN compilation and using HF VQGAN path.")

    if args.use_speculative_decoding and not args.spec_draft_model:
        raise ValueError("--spec_draft_model is required when --use_speculative_decoding with --spec_method=eagle3.")
        
    print(f"Initializing Ray with {num_workers} workers on {num_workers} GPUs")
    # Avoid packaging the (very large) working directory to speed up local starts.
    # Set env RAY_USE_WORKING_DIR=1 to re-enable packaging if needed.
    use_wd = os.environ.get("RAY_USE_WORKING_DIR", "") not in ("", "0", "false", "False")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_paths = [project_root]
    cur_pythonpath = os.environ.get("PYTHONPATH", "")
    if cur_pythonpath:
        py_paths.append(cur_pythonpath)
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": ":".join([p for p in py_paths if p]),
        }
    }
    if not use_wd:
        # Keep behavior of not shipping working_dir while still forwarding env vars.
        runtime_env["working_dir"] = None

    ray_address = args.ray_address or os.environ.get("RAY_ADDRESS", "")
    ray_num_cpus = args.ray_num_cpus
    if ray_num_cpus is None:
        ray_num_cpus_env = os.environ.get("TREAM_RAY_NUM_CPUS", "").strip()
        if ray_num_cpus_env:
            try:
                ray_num_cpus = float(ray_num_cpus_env)
            except ValueError:
                print(
                    f"[Driver] Ignore invalid TREAM_RAY_NUM_CPUS={ray_num_cpus_env!r}; "
                    "use auto-detected CPU resources."
                )

    ray_init_sig = inspect.signature(ray.init)
    ray_init_kwargs = {
        "include_dashboard": False,
        "runtime_env": runtime_env,
    }

    if args.ray_namespace:
        ray_init_kwargs["namespace"] = args.ray_namespace
    if args.ray_temp_dir:
        ray_init_kwargs["_temp_dir"] = args.ray_temp_dir

    if ray_address:
        ray_init_kwargs["address"] = ray_address
        print(f"[Driver] Connecting to Ray at {ray_address}")
    else:
        ray_init_kwargs["num_gpus"] = num_workers
        if ray_num_cpus is not None and ray_num_cpus > 0:
            ray_init_kwargs["num_cpus"] = ray_num_cpus
        if args.ray_port:
            port_param = None
            for candidate in (
                "_gcs_server_port",
                "gcs_server_port",
                "_redis_port",
                "redis_port",
                "_port",
                "port",
            ):
                if candidate in ray_init_sig.parameters:
                    port_param = candidate
                    ray_init_kwargs[candidate] = args.ray_port
                    break
            if port_param is None:
                os.environ.setdefault("RAY_GCS_SERVER_PORT", str(args.ray_port))
                os.environ.setdefault("RAY_PORT", str(args.ray_port))
                print(
                    "[Driver] Ray port requested, but ray.init() does not expose a "
                    "port parameter in this Ray version. Set RAY_GCS_SERVER_PORT/RAY_PORT. "
                    "If you still see conflicts, start a Ray head manually and pass "
                    "--ray_address."
                )

    filtered_kwargs = {
        key: value
        for key, value in ray_init_kwargs.items()
        if key in ray_init_sig.parameters
    }
    ray.init(**filtered_kwargs)

    # Instantiate the shared weights actor, training actor, and inference actor
    weights_actor = SharedWeightsActor.remote()
    # Determine if we enable dynamic scheduling.
    # Requirement: if multiple workers, disable dynamic scheduling.
    enable_scheduler = (not args.disable_dynamic_scheduling) and (num_workers == 1)
    scheduler_actor = None
    if enable_scheduler:
        shift_env_overrides = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TREAM_SHIFT_")
        }
        target_latency = (
            args.scheduler_target_latency
            if args.scheduler_target_latency is not None
            else 0.7
        )
        quality_min = (
            args.scheduler_quality_min
            if args.scheduler_quality_min is not None
            else None
        )
        scheduler_actor = SchedulerActor.remote(
            initial_context_length=args.context_length,
            initial_inference_length=args.inference_length,
            optimization_priority=args.optimization_priority,
            wandb_config=actor_wandb_config,
            run_name=args.wandb_run_name,
            target_latency=target_latency,
            latency_margin=args.scheduler_latency_margin,
            window_size=args.scheduler_window_size,
            q_batch=args.scheduler_q_batch,
            mix_lambda=args.scheduler_mix_lambda,
            quality_min=quality_min,
            shift_env_overrides=shift_env_overrides,
        )
        # 从调度器获取初始配置并添加到 config 字典
        initial_config_tuple = ray.get(scheduler_actor.get_current_config_tuple.remote())
        config["context_length"], config["inference_length"] = initial_config_tuple
        print(f"[Driver] Initial config from scheduler: context_length={config['context_length']}, inference_length={config['inference_length']}")
    else:
        # Force-disable dynamic scheduling in multi-worker mode.
        if num_workers >= 2:
            config["disable_dynamic_scheduling"] = True
            print("[Driver] Multi-worker mode detected; dynamic scheduling DISABLED.")

    if num_workers >= 2: # assign more to inference actor if applicable
        # No scheduler handle passed in multi-worker mode (dynamic scheduling disabled)
        training_actor = TrainingActor.options(num_gpus=num_train_gpus).remote(
            config,
            weights_actor,
        ) if args.use_training_actor else None
        inference_actor = InferenceActor.options(num_gpus=num_inference_gpus).remote(
            config,
            weights_actor,
            training_actor,
        )
    elif num_workers == 1: # TODO: this is rlly messy lol
        if args.use_training_actor:
            config["gpu_memory_utilization"] *= 0.7
        # If only one GPU is available, assign both to the same GPU        
        training_actor = TrainingActor.options(num_gpus=0.5).remote(config, weights_actor, scheduler_actor) if args.use_training_actor else None
        inference_actor = InferenceActor.options(num_gpus=0.5 if args.use_training_actor else 1).remote(
            config, 
            weights_actor, 
            training_actor, 
            scheduler_actor
        )
    else:
        raise ValueError("No GPUs available. Running on CPU.")

    if scheduler_actor and inference_actor and training_actor:
        print("[Driver] Registering actor handles with the scheduler...")
        ray.get(scheduler_actor.register_actor_handles.remote(inference_actor, training_actor))

    # We now directly operate on the inference_actor
    print("\n[PROFILER] Actors created. Waiting 5 seconds for full initialization...")
    time.sleep(5)
    print("[PROFILER] Requesting InferenceActor to START profiling.")
    # Call the remote method of inference_actor
    #ray.get(inference_actor.start_profiling.remote())
    print("[PROFILER] Profiling has been started programmatically on the InferenceActor.")

    # Stream image frames for preprocessing & training
    frame_paths = get_image_files(args.input_frames_path)
    fps = 24  
    frame_delay = 1.0 / fps 
    
    frames_processed = 0
    total_processing_time = 0
    current_directory = None
    
    # Create inference logs directory if it doesn't exist
    if not os.path.exists(args.inference_logs_dir):
        print(f"Creating inference logs directory at {args.inference_logs_dir}")
        os.makedirs(args.inference_logs_dir, exist_ok=True)
    
    # Create training logs directory if it doesn't exist
    if not os.path.exists(args.training_logs_dir):
        print(f"Creating training logs directory at {args.training_logs_dir}")
        os.makedirs(args.training_logs_dir, exist_ok=True)
    
    # Use custom filename if provided, otherwise use timestamp 
    inference_log_filename = f"{args.wandb_run_name}.jsonl" or f"inference_results_{time.time()}.jsonl"
    inference_log_path = os.path.join(args.inference_logs_dir, inference_log_filename)
    unified_log_path = inference_log_path
    scheduler_log_filename = f"scheduler_{args.wandb_run_name}.jsonl" if args.wandb_run_name else f"scheduler_logs_{time.time()}.jsonl"
    scheduler_log_path = os.path.join(args.inference_logs_dir, scheduler_log_filename)
    print(f"[Driver] Scheduler decision logs will be saved to: {scheduler_log_path}")
    spec_tail_thread: Optional[threading.Thread] = None
    spec_metrics_finalizer: Optional[Callable[[], None]] = None
    scheduler_wandb_state: Dict[str, int] = {"cursor": 0, "last_step": -1}
    if args.use_wandb and not args.disable_spec_metrics_log and spec_metrics_path:
        spec_tail_thread, _, spec_metrics_finalizer = setup_spec_metrics_logging(
            run.name if run else spec_metrics_run_name, spec_metrics_path)
        print(f"[SpecMetrics] tailing: {spec_metrics_path}")
    
    try:
        frame_paths = get_image_files(args.input_frames_path)
        
        # 统计变量保持不变
        total_processing_time = 0
        frames_processed = 0
        current_directory = None
        
        frame_iterator = tqdm(frame_paths, desc="Processing Frames")
        disk_check_interval = max(1, int(args.disk_check_interval))
        disk_check_path = os.path.abspath(args.inference_logs_dir)
        last_training_cycle_id = -1
        training_log_stride = max(1, args.training_log_stride)
        for i, frame_path in enumerate(frame_iterator):
            
            # --- 终止条件 1: max_frames ---
            if args.max_frames > 0 and i >= args.max_frames:
                print(f"\n[Driver] Reached max_frames ({args.max_frames}). Stopping data stream.")
                break

            # --- 终止条件 2: disk space ---
            if args.disk_min_free_gb > 0 and (i + 1) % disk_check_interval == 0:
                try:
                    free_gb = shutil.disk_usage(disk_check_path).free / (1024 ** 3)
                    if free_gb < args.disk_min_free_gb:
                        print(
                            f"\n[Driver] Low disk space: {free_gb:.2f} GB free "
                            f"(threshold {args.disk_min_free_gb:.2f} GB). Stopping."
                        )
                        disk_log = {
                            "event": "disk_space_low",
                            "frame_index": i,
                            "free_gb": free_gb,
                            "threshold_gb": args.disk_min_free_gb,
                            "check_path": disk_check_path,
                        }
                        with open(unified_log_path, 'a', encoding="utf-8") as f:
                            f.write(json.dumps(disk_log) + "\n")
                        break
                except Exception as e:
                    print(f"\n[Driver] Disk space check failed: {e}")

            # --- 场景切换逻辑 (保持不变) ---
            frame_directory = os.path.dirname(frame_path)
            if frame_directory != current_directory:
                if current_directory is not None:
                    print(f"\n[Driver] Switching from {current_directory} to {frame_directory}")
                    if args.reset_on_dir_change:
                        print("[Driver] Resetting model state...")
                        # Reset both actors
                        if training_actor:
                            ray.get(training_actor.reset.remote(os.path.basename(current_directory)))
                        if inference_actor:
                            ray.get(inference_actor.reset.remote())
                        print("[Driver] Model state reset complete.")

                    # Log directory transition
                    transition_log = {
                        "event": "directory_transition",
                        "from_dir": current_directory,
                        "to_dir": frame_directory,
                        "frame_index": i
                    }
                    with open(unified_log_path, 'a', encoding="utf-8") as f:
                        f.write(json.dumps(transition_log) + "\n")

                current_directory = frame_directory

            
            # --- 核心调用：异步注入数据 ---
            # 我们不再用 ray.get() 阻塞主循环
            start_time = time.time()
            result_ref = inference_actor.__call__.remote(frame_path)

            results = ray.get(result_ref)
            end_time = time.time()
            
            # --- 统一日志记录：每个推理周期只写一行 ---
            if results:
                result0 = results[0]
                inference_log = {
                    "event": "inference_cycle",
                    "frame_index": i,
                    "frame_path": result0.get("frame_path"),
                    "current_directory": current_directory,
                    "context_length": result0.get("context_length", args.context_length),
                    "inference_length": result0.get("inference_length", args.inference_length),
                    "inference_batch_size": args.inference_batch_size,
                    "training_batch_size": args.training_batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_sharing_freq": args.weight_sharing_freq,
                    "config_version": result0.get("config_version"),
                    "latency": result0.get("latency"),
                    "tokens_per_second": result0.get("tokens_per_second"),
                    "accuracy": result0.get("accuracy"),
                    "perplexity": result0.get("perplexity_cycle", result0.get("perplexity")),
                    "t_inference_cycle": result0.get("t_inference_cycle"),
                    "mem_peak_gb": result0.get("mem_peak_gb"),
                    "lora_step": result0.get("lora_step"),
                    "lora_id": result0.get("lora_id"),
                }
                with open(unified_log_path, 'a', encoding="utf-8") as f:
                    f.write(json.dumps(inference_log) + "\n")
                if args.use_wandb and run:
                    try:
                        infer_payload: Dict[str, Any] = {
                            "inference/step": i,
                            "inference/perplexity": result0.get("perplexity_cycle", result0.get("perplexity")),
                            "inference/latency": result0.get("latency"),
                            "inference/tokens_per_second": result0.get("tokens_per_second"),
                            "inference/t_inference_cycle": result0.get("t_inference_cycle"),
                            "inference/mem_peak_gb": result0.get("mem_peak_gb"),
                            "inference/global_perplexity": result0.get("global_perplexity"),
                            "inference/global_accuracy": result0.get("global_accuracy"),
                        }
                        infer_payload = {
                            key: value for key, value in infer_payload.items() if value is not None
                        }
                        if infer_payload:
                            run.log(infer_payload)
                    except Exception as e:
                        print(f"[Driver] WARN: wandb inference log failed: {e}")
            
            # total_processing_time += sum(result["latency"] for result in results) # 您的原始逻辑
            # 一个更准确的 total_processing_time 是外部测量的时间
            total_processing_time += (end_time - start_time)
            if results: # 仅当有推理结果时才计数
                frames_processed += 1 # 每次 __call__ 调用算处理一帧

            if training_actor and (i + 1) % training_log_stride == 0:
                try:
                    train_metrics = ray.get(training_actor.get_last_training_cycle_metrics.remote())
                    if train_metrics:
                        cycle_id = train_metrics.get("training_cycle_id", -1)
                        if cycle_id is None:
                            cycle_id = -1
                        if cycle_id > last_training_cycle_id:
                            last_training_cycle_id = cycle_id
                            train_metrics = dict(train_metrics)
                            train_metrics["frame_index"] = i
                            train_metrics["current_directory"] = current_directory
                            with open(unified_log_path, 'a', encoding="utf-8") as f:
                                f.write(json.dumps(train_metrics) + "\n")
                            if args.use_wandb and run:
                                try:
                                    train_payload: Dict[str, Any] = {
                                        "train/step": train_metrics.get("train_step_end"),
                                        "train/loss": train_metrics.get("train_loss"),
                                        "train/latency": train_metrics.get("avg_step_latency"),
                                        "train/tokens_per_second": train_metrics.get("train_tokens_per_second"),
                                    }
                                    train_payload = {
                                        key: value for key, value in train_payload.items() if value is not None
                                    }
                                    if train_payload:
                                        run.log(train_payload)
                                except Exception as e:
                                    print(f"[Driver] WARN: wandb train log failed: {e}")
                except RayActorError:
                    pass

            if args.use_wandb and run:
                drain_scheduler_log_to_wandb(
                    scheduler_log_path=scheduler_log_path,
                    run=run,
                    state=scheduler_wandb_state,
                )

            # --- 终止条件 3: max_step (定期检查) ---
            CHECK_STEP_INTERVAL = 50 # 每50帧检查一次
            if training_actor and args.max_step < float('inf') and (i + 1) % CHECK_STEP_INTERVAL == 0:
                try:
                    current_step = ray.get(training_actor.get_current_step.remote())
                    frame_iterator.set_description(f"Processing Frames (Step: {current_step}/{int(args.max_step)})")

                    if current_step >= args.max_step:
                        print(f"\n[Driver] Reached max_step ({args.max_step}). Stopping data stream.")
                        if training_actor:
                            ray.get(training_actor._save_checkpoint.remote(f"final_step_{current_step}"))
                        break

                except RayActorError:
                    print("\n[Driver] TrainingActor seems to have terminated unexpectedly.")
                    break
                

        
    finally:
        # --- 清理逻辑 ---
        print("\n[Driver] Data stream finished. Finalizing...")

        # --- 最终性能统计 (保持不变) ---
        if frames_processed > 0:
            avg_time_per_frame = total_processing_time / frames_processed
            effective_fps = 1 / avg_time_per_frame if avg_time_per_frame > 0 else 0
            
            print(f"[Driver] Total processing time: {total_processing_time:.2f} seconds")
            print(f"[Driver] Frames processed: {frames_processed}")
            print(f"[Driver] Average processing time per frame: {avg_time_per_frame:.4f} seconds")
            print(f"[Driver] Effective frame rate: {effective_fps:.2f} FPS")
            
            stats = {
                "total_processing_time": total_processing_time,
                "avg_time_per_frame": avg_time_per_frame,
                "effective_fps": effective_fps,
                "frames_processed": frames_processed
            }
            with open(unified_log_path, 'a', encoding="utf-8") as f:
                f.write(json.dumps({"event": "final_stats", **stats}) + "\n")
            print(f"[Driver] Inference logs saved to {unified_log_path}")

        # --- 停止 Actors 和 Ray ---
        print("[Driver] Stopping actors and shutting down Ray...")
        stop_tasks = []
        if training_actor:
            stop_tasks.append(training_actor.stop_training.remote())
        if inference_actor: 
            stop_tasks.append(inference_actor.clear_saved_loras.remote())
        
        if stop_tasks:
            try:
                ray.get(stop_tasks)
            except RayActorError as e:
                print(f"[Driver] An actor may have already terminated: {e}")

        if spec_metrics_finalizer:
            spec_metrics_finalizer()

        if args.use_wandb and run:
            drain_scheduler_log_to_wandb(
                scheduler_log_path=scheduler_log_path,
                run=run,
                state=scheduler_wandb_state,
            )

        if args.use_wandb and run: 
            run.finish()
            
        ray.shutdown()
        print("[Driver] Shutdown complete.")
