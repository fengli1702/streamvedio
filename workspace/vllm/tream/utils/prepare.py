import os
import json
import base64
import yaml
import sys
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

SAVE_DIR = "./saved_models/lvm-llama2-7b"
SAVE_DIR_FT = "./saved_models"

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists. If it does not exist, create it.

    Args:
        directory_path (str): Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def main():
    ensure_directory_exists(SAVE_DIR)

    # Login to Hugging Face using token from environment variable
    login(os.environ.get("HF_TOKEN"))

    # Load the model (detailed progress bar is automatically enabled)
    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Emma02/LVM_ckpts",
    )
    
    # Create a simple tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")  # Use a random tokenizer
    tokenizer.save_pretrained(SAVE_DIR)

    # Save the model explicitly to the specified path
    print("Saving the model...")
    model.save_pretrained(SAVE_DIR)
    print("Model saved to", SAVE_DIR)
    
if __name__ == "__main__":
    main()
