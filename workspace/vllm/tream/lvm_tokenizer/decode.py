import os
import json
import torch
import numpy as np
import base64
from PIL import Image
from muse import VQGANModel
from tqdm import tqdm
import imageio


def decode_tokens_from_jsonl(file_path):
    tokens_list = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                b64_data = json.loads(line.strip())["tokens"]
                decoded = base64.b64decode(b64_data)
                tokens = np.frombuffer(decoded, dtype=np.int32)
                assert tokens.shape[0] == 256, f"Expected 256 tokens, got {tokens.shape}"
                tokens_list.append(tokens)
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue

    if not tokens_list:
        raise ValueError(f"No valid tokens in file {file_path}")

    stacked = np.stack(tokens_list)  # much faster and cleaner
    return torch.from_numpy(stacked).long().cuda()  # [17, 256]


def save_full_animation(recons, gif_path, duration=3):
    """
    Save a GIF animation showing all 17 reconstructed frames in sequence.
    
    Args:
        recons (torch.Tensor): shape [17, 3, H, W]
        gif_path (str): path to save the GIF
        duration (float): duration per frame in seconds
    """
    frames = []
    for i in range(recons.shape[0]):
        img = recons[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        frames.append(Image.fromarray(img))

    imageio.mimsave(gif_path, frames, duration=duration)


def decode_and_save_image(tokens, net, save_path):
    print(tokens.shape) 
    with torch.no_grad():
        recons = net.decode_code(tokens)  # [17, 3, H, W]

    imgs = (recons.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # [17, 3, H, W]
    imgs = imgs.transpose(0, 2, 3, 1)  # [17, H, W, C]
    concat_img = np.hstack(imgs)  # [H, W * 17, C]

    img = Image.fromarray(concat_img)
    img.save(save_path)

    return recons  # <-- Return for further use


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    print("🔄 Loading VQ model...")
    net = VQGANModel.from_pretrained("./lvm_tokenizer/ckpt/laion").cuda().eval()

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jsonl")])

    for file in tqdm(files, desc="🖼️ Decoding"):
        file_path = os.path.join(input_folder, file)
        tokens = decode_tokens_from_jsonl(file_path)  # [17, 256]

        if tokens.shape[0] != 16:
            print(f"⚠️ Skipping {file}, incomplete sequence ({tokens.shape[0]} blocks)")
            continue

        filename_base = os.path.splitext(file)[0]
        save_path = os.path.join(output_folder, f"{filename_base}.png")
        gif_path = os.path.join(output_folder, f"{filename_base}_anim.gif")

        recons = decode_and_save_image(tokens, net, save_path)
        save_full_animation(recons, gif_path)


    print(f"✅ Done! Saved images to: {output_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .jsonl files")
    parser.add_argument("--output_dir", type=str, default="./decoded_images", help="Where to save output images")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)