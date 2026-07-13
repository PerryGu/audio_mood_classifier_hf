"""
## Dataset Uploader Module

This module provides a clean interface to upload local audio datasets 
to the Hugging Face Hub, enabling seamless integration with cloud 
environments like Google Colab.
"""

from huggingface_hub import HfApi

def upload_dataset_to_hub(local_path: str, repo_id: str, commit_msg: str = "Update dataset"):
    """
    Uploads a local directory of audio files to a Hugging Face dataset repository.
    
    Args:
        local_path: Path to the local folder containing the audio files.
        repo_id: The ID of your HF dataset repo (e.g., 'username/dataset-name').
        commit_msg: A brief message for the Git commit history on the Hub.
    """
    api = HfApi()
    
    print(f"[INFO] Uploading {local_path} to {repo_id}...")
    
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_msg
    )
    
    print(f"[SUCCESS] Upload complete.")

if __name__ == "__main__":
    local_path=r"F:\Work_stuff\VisualStudio_cursor\audio_mood_classifier\data\mp3_data"
    repo_id="guyPerry/audio-mood-dataset"
    commit_msg="Uploading dataset first time"
    upload_dataset_to_hub(local_path=local_path, repo_id=repo_id, commit_msg=commit_msg)