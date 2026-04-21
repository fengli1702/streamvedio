import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_dir, max_fps=30):
    """
    Extract frames from a video file at a specified maximum FPS.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str, optional): Directory to save extracted frames. If None, frames are not saved.
        max_fps (int): Maximum frames per second to extract.
        
    Returns:
        list: List of extracted frames as numpy arrays if output_dir is None,
              otherwise returns the number of frames extracted.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame extraction interval
    if original_fps > max_fps:
        interval = int(original_fps / max_fps)
    else:
        interval = 1  # Extract all frames if original FPS is already <= max_fps
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    frames = []
    frame_idx = 0
    extracted_count = 0
    
    # Use tqdm for progress tracking
    with tqdm(total=frame_count, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract frame if it falls on the interval
            if frame_idx % interval == 0:
                if output_dir is not None:
                    # Save frame to disk
                    output_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                    cv2.imwrite(output_path, frame)
                else:
                    # Store frame in memory
                    frames.append(frame)
                extracted_count += 1
                
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    
    if output_dir is not None:
        print(f"Extracted {extracted_count} frames to {output_dir}")
        return extracted_count
    else:
        return frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--max_fps", type=int, default=30, 
                        help="Maximum frames per second to extract (default: 30)")
    
    args = parser.parse_args()
    
    output_dir = os.path.splitext(args.video_path)[0]
    extract_frames(args.video_path, output_dir, args.max_fps)

if __name__ == "__main__":
    main()
