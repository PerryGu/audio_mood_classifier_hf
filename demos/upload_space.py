"""
upload_space.py — Upload the audio_mood_classifier demo to Hugging Face Spaces.

Run once from the project root:
    python demos/upload_space.py

Requires:
    - huggingface_hub  (pip install huggingface_hub)
    - HF_TOKEN set in your .env file  OR  as an environment variable
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import create_repo, get_full_repo_name, upload_folder

# ── Configuration ─────────────────────────────────────────────────────────────
# Path to the demo folder to upload (relative to project root).
LOCAL_DEMO_FOLDER = "./demos/audio_mood_classifier"

# Name of the Space on Hugging Face (will become <your-username>/<HF_SPACE_NAME>).
HF_SPACE_NAME = "audio-mood-classifier"

HF_REPO_TYPE = "space"
HF_SPACE_SDK = "gradio"
# ─────────────────────────────────────────────────────────────────────────────

# Load HF_TOKEN from .env if present.
load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "ERROR: HF_TOKEN not found.\n"
        "Add it to your .env file or set it as an environment variable."
    )

# 1. Create the Space repo (exist_ok=True means it's safe to re-run).
print(f"[INFO] Creating Space '{HF_SPACE_NAME}' on Hugging Face Hub ...")
create_repo(
    repo_id=HF_SPACE_NAME,
    repo_type=HF_REPO_TYPE,
    space_sdk=HF_SPACE_SDK,
    private=False,
    exist_ok=True,
    token=token,
)

# 2. Resolve the full repo name (username/repo-name).
hf_full_repo_name = get_full_repo_name(model_id=HF_SPACE_NAME, token=token)
print(f"[INFO] Full repo name: {hf_full_repo_name}")

# 3. Upload the demo folder contents to the root of the Space.
print(f"[INFO] Uploading '{LOCAL_DEMO_FOLDER}' → {hf_full_repo_name} ...")
commit_url = upload_folder(
    repo_id=hf_full_repo_name,
    folder_path=LOCAL_DEMO_FOLDER,
    path_in_repo=".",
    repo_type=HF_REPO_TYPE,
    commit_message="Upload audio mood classifier demo",
    token=token,
)
print(f"[INFO] Upload complete!")
print(f"[INFO] Space URL: https://huggingface.co/spaces/{hf_full_repo_name}")
print(f"[INFO] Commit:    {commit_url}")
