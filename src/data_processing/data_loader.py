"""
## Dataset Loading from Hugging Face Hub
This step establishes a secure connection to the Hugging Face platform to retrieve the private audio dataset.
By authenticating via your access token, the system validates your ownership of the repository and enables the 
download of the dataset into the local runtime memory. Once loaded, the data is structured as an optimized 
Dataset object, ready for further processing, splitting, and training.
"""

import os
import glob
import librosa
import numpy as np
from datasets import Dataset, load_from_disk

def load_hf_dataset(
    repo_id: str, 
    split: str = "train", 
    token: str = None, 
    cache_dir: str = "./data/hf_cache",
    use_local: bool = False,
    local_dir: str = "./data/mp3_data"
):
    """
    Loads dataset with automatic caching to disk.
    If processed dataset exists, loads it immediately.
    """
    cache_path = "./data/processed_dataset"
    
    # Check if we already have a fully processed dataset saved
    # Check if directory exists AND is not empty
    if use_local and os.path.exists(cache_path) and os.listdir(cache_path):
        print(f"[INFO] Loading existing processed dataset from {cache_path}...")
        return load_from_disk(cache_path)
    else:
        print("[INFO] No valid cache found. Loading from source...")
    
    if use_local:
        print(f"[INFO] No cache found. Processing files from '{local_dir}'...")
        audio_files = glob.glob(os.path.join(local_dir, "**/*.mp3"), recursive=True)
        dataset = Dataset.from_dict({"audio": audio_files})
        
        # Process and save to disk for future runs
        dataset = dataset.map(process_audio_and_label)
        dataset.save_to_disk(cache_path)
        
        print(f"[SUCCESS] Dataset processed and saved to '{cache_path}'.")
        return dataset
        
    from datasets import load_dataset
    return load_dataset(repo_id, split=split, cache_dir=cache_dir)

def process_audio_and_label(row):
    """
    Processes audio and adds metadata (label & song_id).
    """
    audio_path = row["audio"]
    
    # 1. Extract label (folder name)
    label = os.path.basename(os.path.dirname(audio_path))
    
    # 2. Extract song_id (for grouped split)
    file_name = os.path.basename(audio_path)
    song_id = file_name.split("_segment")[0] 
    
    # 3. Load and resample audio
    array, _ = librosa.load(audio_path, sr=16000)
    
    return {
        "audio": {"array": array, "sampling_rate": 16000},
        "label": label,
        "song_id": song_id
    }