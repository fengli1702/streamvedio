import cupy
import ray
import numpy as np
import os
import json
import hashlib
import time
import torch 
import shutil
from torchvision import transforms
from PIL import Image
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt
from lvm_tokenizer.utils import RAW_VQGAN_PATH, COMPILED_VQGAN_PATH, ENCODING_SIZE
from lvm_tokenizer.muse import VQGANModel
from actors.weights_actor import SharedWeightsActor
from actors.training_actor import TrainingActor
from collections import deque
import threading
import torch.cuda.nvtx as nvtx

os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(
        self,
        config: dict,
        weights_actor: SharedWeightsActor,
        training_actor: TrainingActor,
    ):
        """
        Modified __init__ method.
        """
        with nvtx.range("InferenceActor.__init__"):
            self.config = config
            self.weights_actor = weights_actor
            self.training_actor = training_actor
            self._load_config()

            self.call_count = 0
            self.current_lora_path = None
            self.current_lora_id = "base_model"
            self.last_weight_update_step = -1
            self.lora_update_lock = threading.Lock()
            self.run_name = config["run_name"]
            self.uuid = 29500 + int(hashlib.md5(self.run_name.encode()).hexdigest(), 16) % 10000
            self.lora_save_dir = f"./tmp/inference_lora_adapters_{self.uuid}"
            os.makedirs(self.lora_save_dir, exist_ok=True)

            self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
            self._init_image_encoder()
            self._init_llm_engine()
            self._init_buffers()

            # --- MODIFICATION START ---
            # Add a new buffer for tokens that are tokenized but not yet sent to the training actor.
            self.pending_training_tokens = []
            # --- MODIFICATION END ---
            
            print(
                f"[InferenceActor] Ready on {self.device} | "
                f"Model: {self.model_name} | "
                f"Frame context: {self.context_length} | "
                f"Batch size: {self.batch_size} | "
                f"Using Interleaved Micro-Inference."
            )

    def _load_config(self) -> None:
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
        self.batch_size = c["inference_batch_size"]
        self.wandb_run = c["wandb_run"]

    def _init_image_encoder(self) -> None:
        with nvtx.range("InferenceActor._init_image_encoder"):
            print("[InferenceActor] Loading VQGAN encoder …")
            self.encoder = (
                VQGANModel.from_pretrained(RAW_VQGAN_PATH).to(self.device).eval()
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
            print(f"[InferenceActor] Initialising vLLM engine for {self.model_name} …")
            engine_args = EngineArgs(
                model=self.model_name,
                tokenizer=self.model_name,
                enable_lora=True,
                max_lora_rank=self.lora_rank,
                max_loras=self.max_loras,
                gpu_memory_utilization=self.gpu_memory_utilisation,
                tensor_parallel_size=self.tensor_parallel_size,
            )
            self.engine = LLMEngine.from_engine_args(engine_args)
            print(f"[InferenceActor] vLLM engine initialised with LoRA support on {self.device}.")

    def _init_buffers(self) -> None:
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
                    transformed_image = self.image_transform(image).to(self.device)
                    images.append(transformed_image)
                except Exception as e:
                    print(f"[InferenceActor] Failed to process image at path {image_path}: {e}")
                    all_tokens.append([0] * ENCODING_SIZE)
                    continue
            if images:
                with torch.no_grad():
                    with nvtx.range("vqgan_encoder.encode"):
                        _, tokens_batch = self.encoder.encode(torch.stack(images, dim=0))
                for tokens in tokens_batch:
                    token_list = tokens.cpu().numpy().astype(np.int32).tolist()
                    all_tokens.append(token_list)
            return all_tokens

    def tokenize_image(self, image_path: str) -> dict:
        return self.tokenize_images_batch([image_path])[0]

    def check_for_updates(self):
        with nvtx.range("InferenceActor.check_for_updates"):
            update_available = ray.get(self.weights_actor.check_update_status.remote())
            if not update_available:
                return
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

    def infer_batch(self, input_ids_batch: list[list[np.int32]]):
        with nvtx.range("InferenceActor.infer_batch"):
            if not self.engine or not input_ids_batch:
                return [] 
            
            lora_request = None
            if self.current_lora_id != "base_model" and self.last_weight_update_step >= 0:
                lora_request = LoRARequest(
                    lora_name=self.current_lora_id, lora_int_id=self.last_weight_update_step,
                    lora_local_path=self.current_lora_path
                )
            sampling_params = SamplingParams(
                temperature=self.config["temperature"], top_p=self.config["top_p"],
                max_tokens=self.inference_length * ENCODING_SIZE, logprobs=1
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
                                    token_ids, logprobs = [], []
                                    for token_dict in request_output.outputs[0].logprobs:
                                        token_id, logprob_obj = list(token_dict.items())[0]
                                        token_ids.append(token_id)
                                        logprobs.append(logprob_obj.logprob)
                                    perplexity = 2 ** (-sum(logprobs) / len(logprobs)) if logprobs else float('inf')
                                    results[request_id] = (token_ids, logprobs, perplexity)
                
                ordered_results = []
                for i, request_id in enumerate(request_ids):
                    token_ids, logprobs, perplexity = results[request_id]
                    # This relies on the path_buffer having the correct corresponding paths
                    # when infer_batch is called. The new logic ensures this.
                    frame_path_index = self.path_buffer.index(self.path_buffer[-len(input_ids_batch)]) + i
                    ordered_results.append({
                        "frame_path": self.path_buffer[frame_path_index], "token_ids": token_ids, "logprobs": logprobs,
                        "perplexity": perplexity, "lora_step": self.last_weight_update_step, "lora_id": self.current_lora_id,
                    })
                return ordered_results
            except Exception as e:
                print(f"[InferenceActor] Error during vLLM inference: {e}")
                return []

    # --- NEW METHOD START ---
    def _execute_interleaved_inference(self, full_batch: list):
        """
        New core method to execute the interleaved "send-infer" pipeline.
        """
        with nvtx.range("execute_interleaved_inference"):
            # --- 1. Split the workload into two halves ---
            batch_size = len(full_batch)
            mid_point = batch_size // 2

            batch_chunk_1 = full_batch[:mid_point]
            batch_chunk_2 = full_batch[mid_point:]
            
            # Also split the corresponding tokens to be sent
            tokens_to_send_1 = self.pending_training_tokens[:mid_point]
            tokens_to_send_2 = self.pending_training_tokens[mid_point:]
            
            all_results = []

            # --- 2. Execute the first leg of the pipeline ---
            with nvtx.range("interleaved_pipeline_leg_1"):
                # Send the first half of the tokens
                if self.training_actor:
                    for token_data in tokens_to_send_1:
                        self.training_actor.add_token.remote(token_data)
                
                # Perform the first micro-inference batch
                results_1 = self.infer_batch(batch_chunk_1)
                all_results.extend(results_1)

            # --- 3. Execute the second leg of the pipeline ---
            with nvtx.range("interleaved_pipeline_leg_2"):
                # Send the second half of the tokens
                if self.training_actor:
                    for token_data in tokens_to_send_2:
                        self.training_actor.add_token.remote(token_data)
                
                # Perform the second micro-inference batch
                results_2 = self.infer_batch(batch_chunk_2)
                all_results.extend(results_2)

            return all_results
    # --- NEW METHOD END ---

    def __call__(self, image_path: str):
        """
        Modified __call__ method to implement the interleaved pipeline.
        """
        with nvtx.range(f"InferenceActor.__call__ (count:{self.call_count})"):
            start = time.time()
            
            with nvtx.range("tokenize_image"):
                tokens = self.tokenize_image(image_path)
            
            self.token_buffer.append(tokens)
            self.path_buffer.append(image_path)
            
            if self.training_actor:
                # --- MODIFICATION START ---
                # Instead of sending immediately, buffer the token.
                with nvtx.range("buffer_pending_training_token"):
                    self.pending_training_tokens.append(tokens)
                
                # The weight update check can still happen here.
                self.check_for_updates()
                # --- MODIFICATION END ---
            
            self.call_count += 1
            
            # When a full macro-batch is ready, trigger the interleaved pipeline
            if self.call_count % self.batch_size == 0:
                with nvtx.range("prepare_and_run_INTERLEAVED_inference_batch"):
                    buffer_list, buffer_length = list(self.token_buffer), len(self.token_buffer)
                    # This logic correctly prepares the prompts for the full batch
                    full_batch = [
                        np.concatenate(buffer_list[max(0, i - self.context_length) : i]).tolist()
                        for i in range(buffer_length - self.batch_size + 1, buffer_length + 1)
                    ]
                    
                    # --- MODIFICATION START ---
                    # Call the new core method to execute the pipeline
                    results = self._execute_interleaved_inference(full_batch)
                    
                    # After the macro-task is complete, clear the pending buffer
                    if self.training_actor:
                        self.pending_training_tokens.clear()
                    # --- MODIFICATION END ---
            else:
                # On non-triggering calls, we just buffer data.
                with nvtx.range("buffering_only_no_inference"):
                    results = []
            
            end = time.time()
            # Calculate metrics based on the macro-batch, as this reflects the true throughput
            latency = (end - start)
            
            for result in results:
                result["latency"] = latency / len(results) if results else 0
                    
            return results

    def reset(self):
        print("[InferenceActor] Resetting state...")
        with self.lora_update_lock:
            self.call_count = 0
            self.current_lora_path = None
            self.current_lora_id = "base_model"
            self.last_weight_update_step = -1
            self.token_buffer.clear()
            self.path_buffer.clear()
            # Also clear the new buffer on reset
            if hasattr(self, 'pending_training_tokens'):
                self.pending_training_tokens.clear()
        print("[InferenceActor] Reset complete.")
        
    def clear_saved_loras(self):
        print(f"[InferenceActor] Cleaning up resources...")
        try:
            if os.path.exists(self.lora_save_dir):
                shutil.rmtree(self.lora_save_dir)
                print(f"[InferenceActor] Successfully deleted the directory {self.lora_save_dir}")
        except Exception as e:
            print(f"[InferenceActor] Error while cleaning up resources: {e}")