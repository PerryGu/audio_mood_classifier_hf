"""
## Dataset Loading from Hugging Face Hub
This step establishes a secure connection to the Hugging Face platform to retrieve the private audio dataset.
By authenticating via your access token, the system validates your ownership of the repository and enables the 
download of the dataset into the local runtime memory. Once loaded, the data is structured as an optimized 
Dataset object, ready for further processing, splitting, and training.
"""

import os
import glob
import zipfile
import librosa
import numpy as np
from pathlib import Path
from datasets import Dataset, load_from_disk


def _ensure_mp3_data(local_dir: str) -> None:
    """
    Colab only: if the mp3 data folder is missing or empty, extracts it from
    the zip file whose path is stored in the COLAB_MP3_ZIP_PATH environment
    variable (set at the top of main.py).

    Handles two common zip structures automatically:
      • Zip contains one wrapper folder  → mp3_data.zip/mp3_data/calm_melancholic/...
      • Zip contains category folders directly → mp3_data.zip/calm_melancholic/...

    On a local machine this function returns immediately without doing anything.
    """
    # Only run inside Colab
    try:
        import google.colab  # type: ignore
    except ImportError:
        return

    mp3_dir = Path(local_dir)

    # Skip extraction if data is already in place
    if mp3_dir.exists() and any(mp3_dir.rglob("*.mp3")):
        mp3_count = len(list(mp3_dir.rglob("*.mp3")))
        print(f"[INFO] Colab: MP3 data already present ({mp3_count} files) — skipping extraction.")
        return

    zip_path_str = os.getenv("COLAB_MP3_ZIP_PATH", "")
    if not zip_path_str:
        print("[WARN] Colab: COLAB_MP3_ZIP_PATH is not set — cannot extract MP3 data.")
        print("       Set it at the top of main.py and re-run.")
        return

    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        print(f"[ERROR] Colab: zip not found at '{zip_path_str}'.")
        print("        Make sure Google Drive is mounted and the path is correct.")
        print("        Mount Drive in a notebook cell before running this script:")
        print("            from google.colab import drive")
        print("            drive.mount('/content/drive')")
        return

    print(f"[INFO] Colab: extracting '{zip_path.name}' → '{local_dir}'...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Inspect the top-level names inside the zip
        top_level = {Path(n).parts[0] for n in zf.namelist() if n.strip("/")}

        if len(top_level) == 1:
            # Single wrapper folder (e.g. mp3_data/) — extract to the parent
            # so the result lands at local_dir directly.
            parent = mp3_dir.parent
            parent.mkdir(parents=True, exist_ok=True)
            zf.extractall(parent)
            extracted = parent / next(iter(top_level))
            if extracted.resolve() != mp3_dir.resolve() and extracted.exists():
                if mp3_dir.exists():
                    import shutil
                    shutil.rmtree(mp3_dir)
                extracted.rename(mp3_dir)
        else:
            # Category folders at the root — extract directly into local_dir
            mp3_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(mp3_dir)

    mp3_count = len(list(mp3_dir.rglob("*.mp3")))
    print(f"[INFO] Colab: extraction complete — {mp3_count} MP3 files ready in '{local_dir}'.")


def load_hf_dataset(
    repo_id: str,
    split: str = "train",
    token: str = None,
    cache_dir: str = "./data/hf_cache",
    use_local: bool = False,
    local_dir: str = "./data/mp3_data",
):
    """
    Loads dataset with automatic caching to disk.
    In Colab, extracts mp3_data from Drive if it is not already present.
    If the processed dataset cache already exists, loads it immediately.
    """
    cache_path = "./data/processed_dataset"

    # In Colab, unzip the data from Drive before trying to load anything.
    if use_local:
        _ensure_mp3_data(local_dir)

    # Check if we already have a fully processed dataset saved
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