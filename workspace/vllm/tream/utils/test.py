import ray
ray.init(num_gpus=4)  # Adjust based on your setup

@ray.remote(num_gpus=1)
def train_on_gpu(gpu_id):
    import torch
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

ray.get([train_on_gpu.remote(i) for i in range(4)])
