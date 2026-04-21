import os
import subprocess

def clear_log_file(log_dir, log_filename):
    """
    Clears the contents of a log file if it exists.
    
    Args:
        log_dir (str): Directory containing the log file
        log_filename (str): Name of the log file
    """
    if log_dir and log_filename:
        log_path = os.path.join(log_dir, log_filename)
        if os.path.exists(log_path):
            print(f"Clearing existing log file: {log_path}")
            open(log_path, 'w').close()  # Makes the file empty

def get_available_gpus():
    try:
        # Get both used and total memory for each GPU
        used_cmd = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        total_cmd = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
        
        memory_used = [int(x) for x in used_cmd.decode('utf-8').strip().split('\n')]
        memory_total = [int(x) for x in total_cmd.decode('utf-8').strip().split('\n')]
        
        # Calculate memory usage percentage and return GPUs with less than 10% usage
        available_gpus = [i for i, (used, total) in enumerate(zip(memory_used, memory_total)) 
                          if (used / total) < 0.1]
        
        if not available_gpus:
            # If no GPU has less than 10% usage, return the one with minimum usage
            available_gpus = [memory_used.index(min(memory_used))]

        return available_gpus, len(available_gpus)
    except:
        raise ValueError("No GPU Available")
    
def get_image_files(folder_path: str):
    """
    Return a sorted list of image file paths in the given folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        list: Sorted list of image file paths
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    image_files = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)
