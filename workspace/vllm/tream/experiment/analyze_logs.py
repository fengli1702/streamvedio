import json
import torch
import numpy as np
import einops
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lvm_tokenizer.utils import RAW_VQGAN_PATH, ENCODING_SIZE
from lvm_tokenizer.muse import VQGANModel
from utils.utils import get_image_files

EXPERIMENT_NAME = "exp4_8"
TRAIN_LOG_PATH = f"./training_logs/{EXPERIMENT_NAME}.jsonl"
INFERENCE_LOG_PATH = f"./inference_logs/{EXPERIMENT_NAME}.jsonl"

DATA_DIR = f"./data/room_1D"
DATA_TOKENIZED_PATH = DATA_DIR + "_tokenized.jsonl"
BATCH_SIZE = 16

NUM_INFERERENCE_FRAMES = 1

def perplexity(pred_tokens, true_tokens):
    """
    Compute perplexity between predicted tokens and ground truth tokens.
    Perplexity is calculated as 2^(binary cross entropy loss).
    
    Args:
        pred_tokens (list): Predicted token IDs
        true_tokens (list or dict): Ground truth token IDs or dict containing 'tokens' key
    
    Returns:
        float: Perplexity score (lower is better)
    """
    # Handle case where true_tokens is a dictionary with 'tokens' key
    if isinstance(true_tokens, dict) and 'tokens' in true_tokens:
        true_tokens = true_tokens['tokens']
    
    # Convert to numpy arrays for calculation
    pred_array = np.array(pred_tokens)
    true_array = np.array(true_tokens)
    
    # Calculate binary cross entropy
    # For token prediction, we can use a simple approach:
    # 1 for correct predictions, 0 for incorrect
    correct_predictions = (pred_array == true_array).astype(np.float32)
    
    # Calculate binary cross entropy loss
    # BCE = -1/N * sum(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    # Since we're using 1/0 for correct/incorrect, this simplifies to:
    # BCE = -1/N * sum(log(correct_predictions))
    epsilon = 1e-10  # To avoid log(0)
    bce_loss = -np.mean(np.log2(correct_predictions + epsilon))
    
    # Perplexity = 2^(binary cross entropy)
    return np.power(2, bce_loss)

def read_jsonl(log_file):
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
        
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    return logs

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def tokenize_images(data_path, net, batch_size, num_frames):
    """
    Tokenizes a batch of images with the VQGAN encoder.

    Args:
        image_paths: List of paths to image files.

    Returns:
        List of tokenized images on CPU
    """    
    image_paths = get_image_files(data_path)[:num_frames]
    dataset = CustomImageDataset(image_paths, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    tokens_list = []
    with open(DATA_TOKENIZED_PATH, 'w') as fout:
        for batch in tqdm(dataloader):
            # Process your batch of images here
            x = batch
            x = x.to(net.device)
            with torch.no_grad():
                _, tokens = net.encode(x)

                tokens_save = einops.rearrange(
                    tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=tokens.size(0)
                )
                for i in range(x.size(0)):
                    data = {'tokens': tokens_save[i].tolist()}
                    fout.write(json.dumps(data) + '\n')
    
    return tokens_list

if __name__ == "__main__":
    train_logs = read_jsonl(TRAIN_LOG_PATH)
    inference_logs = read_jsonl(INFERENCE_LOG_PATH)
    num_frames = len(inference_logs) if inference_logs else 0
    
    if num_frames == 0:
        print("No inference logs found, exiting analysis")
        exit()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder = VQGANModel.from_pretrained(RAW_VQGAN_PATH).to(device).eval()    
    image_transform = transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),]
    )
    # Check if tokenized data already exists
    if os.path.exists(DATA_TOKENIZED_PATH):
        print(f"Loading tokenized data from {DATA_TOKENIZED_PATH}")
        true_tokens = read_jsonl(DATA_TOKENIZED_PATH)
    else:
        print(f"Tokenizing images and saving to {DATA_TOKENIZED_PATH}")
        true_tokens = tokenize_images(
            data_path=DATA_DIR,
            net=encoder,
            batch_size=BATCH_SIZE,
            num_frames=num_frames
        )
    
    # total_perplexity = 0.0
    # count = 0
    
    # for i in range(min(len(inference_logs), len(true_tokens))):
    #     pred_tokens = inference_logs[i]["output"]
    #     true_token_list = true_tokens[i]["tokens"]
        
    #     if len(pred_tokens) == len(true_token_list):
    #         total_perplexity += perplexity(pred_tokens, true_token_list)        
    #         count += 1
    
    # # Calculate and print average perplexity
    # if count > 0:
    #     avg_perplexity = total_perplexity / count
    #     print(f"Average perplexity: {avg_perplexity:.4f}")
    # else:
    #     print("Could not calculate perplexity - no matching token pairs found")
    
    # Calculate average latency from inference logs
    total_latency = 0.0
    latency_count = 0
    
    for log_entry in inference_logs:
        if "latency" in log_entry:
            total_latency += log_entry["latency"]
            latency_count += 1
    
    if latency_count > 0:
        avg_latency = total_latency / latency_count
        print(f"Average inference latency: {avg_latency:.4f} seconds")
    else:
        print("No latency information found in inference logs")
            
    if os.path.exists(TRAIN_LOG_PATH):
        train_logs = read_jsonl(TRAIN_LOG_PATH)
        
        # Filter logs for epoch 3 entries
        epoch_3_logs = [log for log in train_logs if log.get("epoch") == 3]
        
        # Calculate average loss for epoch 3
        if epoch_3_logs:
            total_epoch_3_loss = sum(log.get("loss", 0) for log in epoch_3_logs)
            avg_epoch_3_loss = total_epoch_3_loss / len(epoch_3_logs)
            print(f"Average epoch 3 loss: {avg_epoch_3_loss:.4f}")
        else:
            print("No epoch 3 data found in training logs")
    else:
        print(f"Training logs file not found: {TRAIN_LOG_PATH}")
