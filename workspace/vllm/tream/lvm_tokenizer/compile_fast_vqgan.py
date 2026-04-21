import torch
from lvm_tokenizer.muse import VQGANModel
from lvm_tokenizer.utils import RAW_VQGAN_PATH, COMPILED_VQGAN_PATH

class EncodeAndFlatten(torch.nn.Module):
    def __init__(self, vqgan):
        super().__init__()
        self.vqgan = vqgan

    def forward(self, x):
        _, codes = self.vqgan.encode(x)  
        return codes.flatten(1)        

def compile_fast_vqgan():
    try:
        import torch_tensorrt  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Torch-TensorRT is required to compile the fast VQGAN: {type(e).__name__}: {e}"
        )
    vqgan = VQGANModel.from_pretrained(RAW_VQGAN_PATH).eval().cuda()
    wrapper = EncodeAndFlatten(vqgan)

    trt_wrap = torch_tensorrt.compile(
        wrapper,
        inputs=[torch_tensorrt.Input(
            min_shape=(1,3,256,256),
            opt_shape=(32,3,256,256),
            max_shape=(128,3,256,256),
            dtype=torch.float
        )],
        enabled_precisions={torch.float},
        torch_executed_ops={"aten::scatter_", "aten::argmin"}
    )

    example = torch.randn(1,3,256,256, device="cuda", dtype=torch.float)
    traced = torch.jit.trace(trt_wrap, example)
    torch.jit.save(traced, COMPILED_VQGAN_PATH)
    print("=== SAVED VQGAN ===")
    torch.jit.load(COMPILED_VQGAN_PATH, map_location="cuda").eval()
    print("=== LOADED VQGAN ===")
    
if __name__ == "__main__":
    compile_fast_vqgan()
