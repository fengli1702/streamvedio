from base64 import b64encode
from tqdm import tqdm, trange
import numpy as np
import cv2
np.float = np.float64
np.int = np.int_
from lvm_tokenizer.utils import read_all_frames_from_video, is_video
import time
import einops
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from muse import VQGANModel
from base64 import b64encode
import json
import os
import mlxu

# TODO: clean this up

def is_video(path):
    return (
        path.endswith('.mp4')
        or path.endswith('.avi')
        or path.endswith('.MP4')
        or path.endswith('.AVI')
        or path.endswith('.webm')
        or path.endswith('.WEBM')
        or path.endswith('.mkv')
        or path.endswith('.MVK')
        or path.endswith('.gif')
    )



FLAGS, _ = mlxu.define_flags_with_default(
    input_dir="./data/example/example.gif",
    output_file="./data/example/example.jsonl",
    batch_size=32,
    window_size=4,
    # n_frames=16,
    n_workers=8,
    # strides='1',
    # n_epochs=1,
    dtype='fp32',
)


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, video_path, window_size):
        self.frames = read_all_frames_from_video(video_path)
        self.window_size = window_size

    def __getitem__(self, index):
        window = self.frames[index:index+self.window_size]
        return window

    def __len__(self):
        if self.frames is None:
            return 0
        return max(0, len(self.frames) - self.window_size + 1)


def main(argv):
    assert FLAGS.input_dir != ''
    assert FLAGS.output_file != ''

    # Load the pre-trained vq model from the hub
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = VQGANModel.from_pretrained("./lvm_tokenizer/ckpt/laion").to(device)
    net.eval()

    # videos = []
    # for root, _, files in os.walk(FLAGS.input_dir):
    #     for file in files:
    #         if is_video(file):
    #             videos.append(os.path.join(root, file))
    # print(videos)
    
    with open(FLAGS.output_file, 'w') as fout:
        with torch.no_grad():
            dataset = VideoDataset(video_path=FLAGS.input_dir, window_size=FLAGS.window_size)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=FLAGS.batch_size,
                shuffle=False,
                num_workers=FLAGS.n_workers,
                # prefetch_factor=4,
                # drop_last=True,
            )
            for batch in tqdm(dataloader, ncols=0):
                batch_size = batch.shape[0]
                batch = einops.rearrange(
                    batch.numpy(), 'b t h w c -> (b t) c h w'
                )
                batch = torch.tensor(batch).to(device)
                
                # Time the inference for a single frame
                start_time = time.time()
                _, tokens = net.encode(batch)
                end_time = time.time()
                inference_time = (end_time - start_time) / batch_size  # Time per frame
                print(f"Inference time per frame: {inference_time:.4f} seconds")
                
                tokens = einops.rearrange(
                    tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=batch_size
                )
                for i in range(batch_size):
                    data = {'tokens': b64encode(tokens[i].tobytes()).decode('utf-8')}
                    fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    mlxu.run(main)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Tokenize video dataset')
#     parser.add_argument('--input_dir', type=str, default="/data/jonnypei/lvm_proj/data", help='Directory containing video files')
#     parser.add_argument('--output_file', type=str, default="/data/jonnypei/lvm_proj/data/test_example.jsonl", help='Output JSONL file path')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
#     parser.add_argument('--n_frames', type=int, default=16, help='Number of frames to process')
#     parser.add_argument('--n_workers', type=int, default=32, help='Number of worker processes')
#     parser.add_argument('--strides', type=str, default='8', help='Comma-separated list of frame strides')
#     # parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs')
#     parser.add_argument('--dtype', type=str, default='fp32', help='Data type for processing')
#     parser.add_argument('--ckpt_path', type=str, default="./ckpt/laion", help='Path to the VQGAN checkpoint')
#     args = parser.parse_args()
#     main(args)