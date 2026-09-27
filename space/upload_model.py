"""
upload_model.py — Upload a trained checkpoint to a Hugging Face model repository.

Run from the project root:
    python space/upload_model.py

Works both locally and in Google Colab (reads HF_TOKEN from environment or .env).
After uploading, set MODEL_ID in space/audio_mood_classifier/app.py to point to
the new repo so the Gradio Space can load it.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import create_repo, get_full_repo_name, upload_folder

# ── Configuration ─────────────────────────────────────────────────────────────
# Checkpoint folder to upload (relative to project root).
CHECKPOINT_PATH = "models/mood_classifier_2026-07-16_09-27/checkpoint-372"

# Name of the model repository on Hugging Face Hub.
# Will be created as <your-username>/HF_MODEL_NAME.
HF_MODEL_NAME = "audio-mood-classifier"

# Commit message shown in the HF repo history.
COMMIT_MESSAGE = "Upload fine-tuned audio mood classifier (AST)"
# ─────────────────────────────────────────────────────────────────────────────

# Load HF_TOKEN from .env if present (local). In Colab it comes from os.environ.
load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "ERROR: HF_TOKEN not found.\n"
        "Add it to your .env file or set it as an environment variable."
    )

checkpoint_path = Path(CHECKPOINT_PATH)
if not checkpoint_path.exists():
    raise SystemExit(
        f"ERROR: Checkpoint not found at '{checkpoint_path}'.\n"
        f"Update CHECKPOINT_PATH at the top of this file."
    )

# 1. Create the model repo on HF Hub (exist_ok=True — safe to re-run).
print(f"[INFO] Creating model repo '{HF_MODEL_NAME}' on Hugging Face Hub ...")
create_repo(
    repo_id=HF_MODEL_NAME,
    repo_type="model",
    private=False,
    exist_ok=True,
    token=token,
)

# 2. Resolve the full repo name (username/repo-name).
full_repo_name = get_full_repo_name(model_id=HF_MODEL_NAME, token=token)
print(f"[INFO] Full repo name: {full_repo_name}")

# 3. Upload the checkpoint folder contents directly — no Python model loading needed.
print(f"[INFO] Uploading '{checkpoint_path}' → {full_repo_name} ...")
commit_url = upload_folder(
    repo_id=full_repo_name,
    folder_path=str(checkpoint_path),
    path_in_repo=".",
    repo_type="model",
    commit_message=COMMIT_MESSAGE,
    token=token,
    ignore_patterns=["optimizer.pt", "*.pth"],   # skip optimizer state — not needed for inference
)

print(f"\n[INFO] Upload complete!")
print(f"[INFO] View it at: https://huggingface.co/{full_repo_name}")
print()
print(f"Next step: update MODEL_ID in space/audio_mood_classifier/app.py:")
print(f'  MODEL_ID = "{full_repo_name}"')
print(f"Then run: python space/upload_space.py")
