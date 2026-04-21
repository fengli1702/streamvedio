import argparse
import os
import re
import gdown

def extract_file_id_from_url(url):
    """
    Extracts the file ID from a Google Drive URL.
    
    Args:
        url (str): The Google Drive URL.
    
    Returns:
        str: The file ID or None if not found.
    """
    # Pattern for Google Drive links
    patterns = [
        r'https://drive\.google\.com/file/d/([^/]+)',  # /file/d/ format
        r'https://drive\.google\.com/open\?id=([^&]+)',  # open?id= format
        r'https://docs\.google\.com/file/d/([^/]+)',  # docs format
        r'id=([^&]+)'  # generic id= parameter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def download_gdrive_video(url, destination):
    """
    Downloads a video from Google Drive using gdown.
    
    Args:
        url (str): The Google Drive URL or file ID.
        destination (str, optional): The path where the file should be saved.
                                    If None, saves to current directory with original filename.
    
    Returns:
        str: Path to the downloaded file or None if download failed.
    """
    # Create directory if destination is provided and doesn't exist
    if destination:
        os.makedirs(os.path.dirname(destination) if os.path.dirname(destination) else '.', exist_ok=True)
    
    file_id = extract_file_id_from_url(url)
    drive_url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading file from Google Drive: {drive_url}")
    
    try:
        # Use gdown to download the file
        output = gdown.download(drive_url, destination, quiet=False)
        
        if output:
            print(f"Download complete: {output}")
            return output
        else:
            print("Download failed.")
            return None
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Download a file from Google Drive')
    parser.add_argument('url', help='Google Drive URL or file ID')
    parser.add_argument('destination', help='Destination path for the downloaded file')
    
    args = parser.parse_args()
    
    url = args.url
    destination = args.destination
    download_gdrive_video(url, destination)