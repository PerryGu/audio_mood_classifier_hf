"""
Main Entry Point
This file serves as the primary interface for executing the end-to-end ML pipeline.
It orchestrates the initialization, data processing, and model loading phases 
by utilizing the PipelineManager. This modular approach ensures a clean, 
reproducible workflow for training and inference.
"""
import os
import sys

# ── Colab Configuration ───────────────────────────────────────────────────────
# These variables are only used when running in Google Colab.
# They are ignored automatically on a local machine.

# Path to the project folder in Colab.
COLAB_PROJECT_PATH = "/content/audio_mood_classifier_hf"

# Path to the mp3_data.zip file in Google Drive.
# The zip should contain the three category folders (calm_melancholic,
# energetic_upbeat, moderate_neutral) either at the root or inside one
# wrapper folder — both structures are handled automatically.
# Mount Drive in a notebook cell before running this script:
#     from google.colab import drive; drive.mount('/content/drive')
COLAB_MP3_ZIP_PATH = "/content/drive/MyDrive/audio_mood_classifier_hf/mp3_data.zip"

# Path to the base pre-trained AST model folder on Google Drive.
# If this folder exists and contains model.safetensors, it is copied into
# models/ast_pretrained/ at the start of model loading — no HF Hub download needed.
# Set to "" to always load from HF Hub.
COLAB_DRIVE_MODEL_PATH = "/content/drive/MyDrive/audio_mood_classifier_hf/models/ast_pretrained"

# Base Drive path to the models folder.
# The pipeline auto-derives the checkpoint path from this base combined with
# config.parent_run_folder and config.resume_checkpoint_name — so you never
# need to update this variable again. Just keep config.py up to date.
# Set to "" to disable Drive checkpoint copying entirely (fresh run or local only).
COLAB_DRIVE_MODELS_BASE = "/content/drive/MyDrive/audio_mood_classifier_hf/models"

# ─────────────────────────────────────────────────────────────────────────────

# Bootstrap for Colab: sets the working directory so that all relative paths
# (data/, models/, runs/) and local imports (src.*) resolve correctly.
# NOTE: drive.mount() is intentionally NOT called here — it requires an IPython
# kernel and must be run in a notebook cell, not from a script.
try:
    import google.colab  # type: ignore  — just checks we are in Colab
    if COLAB_PROJECT_PATH and os.path.isdir(COLAB_PROJECT_PATH):
        os.chdir(COLAB_PROJECT_PATH)
        if COLAB_PROJECT_PATH not in sys.path:
            sys.path.insert(0, COLAB_PROJECT_PATH)
        print(f"[INFO] Colab: working directory set to '{COLAB_PROJECT_PATH}'.")
    else:
        print(
            f"[WARN] Colab: path '{COLAB_PROJECT_PATH}' not found.\n"
            f"       Update COLAB_PROJECT_PATH at the top of main.py and re-run.\n"
            f"       If the project is in Drive, mount Drive in a notebook cell first."
        )
    # Expose paths as environment variables so pipeline_manager.py and
    # data_loader.py can read them without changing their signatures.
    os.environ["COLAB_MP3_ZIP_PATH"]        = COLAB_MP3_ZIP_PATH
    os.environ["COLAB_DRIVE_MODEL_PATH"]    = COLAB_DRIVE_MODEL_PATH
    os.environ["COLAB_DRIVE_MODELS_BASE"]   = COLAB_DRIVE_MODELS_BASE
except ImportError:
    pass  # Local execution — no Colab bootstrap needed.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
from src.pipeline_manager import PipelineManager
from src.utils import tests

# Initialize the PipelineManager object.
# This creates a persistent state container to hold the dataset and model
# throughout the session, preventing redundant initialization.
mgr = PipelineManager()

def main():
    print("Starting the main function")
    # 0. SETUP & CONFIGURATION
    # =========================================================================
    # All hyperparameters and run-behaviour flags live in src/config.py.
    # Edit them there.  The CLI args below are optional one-shot overrides that
    # take precedence over the config values without requiring a file edit.
    parser = argparse.ArgumentParser(description="Run the ML Pipeline")
    parser.add_argument(
        "--debug", action="store_true",
        help="Run integrity checks (overrides config.debug).",
    )
    parser.add_argument(
        "--mode", choices=["train", "test", "both", "test_detail"], default=None,
        help="Execution mode — overrides config.mode when provided.",
    )
    args = parser.parse_args()

    # CLI overrides: if a flag was explicitly passed on the command line it wins;
    # otherwise the value from config.py is used.
    debug = args.debug or mgr.config.debug
    mode  = args.mode if args.mode is not None else mgr.config.mode

    os.environ["WANDB_PROJECT"]  = mgr.config.wandb_project
    os.environ["WANDB_RUN_NAME"] = mgr.config.session_name

    print(f"[INFO] Execution Mode : {mode.upper()}")
    print(f"[INFO] Output folder  : {mgr.config.output_dir}")
    print(f"[INFO] Session name   : {mgr.config.session_name}")
    print(f"[INFO] TensorBoard    : {mgr.config.logging_dir}")
    print(f"[INFO] Continuous log : {mgr.config.continuous_logging_dir}")
    if mgr.config.parent_run_folder:
        print(f"[INFO] Resuming weights from: {mgr.config.checkpoint_to_load}")
    else:
        print("[INFO] Starting a new training run from scratch.")
    print(f"[INFO] LR: {mgr.config.learning_rate}  |  Batch: {mgr.config.batch_size}  |  Epochs: {mgr.config.num_train_epochs}")
    print()



    mgr.setup_environment()

    # 1. LOADING DATASET & MODEL
    # =========================================================================
    # Load the dataset into the manager's memory.
    # If the data is already loaded, this step is skipped to save time.
    mgr.run_data_loading()

    # Initialize the model and feature extractor.
    # This step detects the available hardware (GPU/CPU) and prepares the 
    # model for inference or training.
    mgr.run_model_loading()

    # 2. DATA PROCESSING (Feature Extraction & Mapping)
    # =========================================================================
    # Prepare and process the dataset for the model
    # This transforms raw audio into spectrograms
    mgr.prepare_dataset()
    
    # Split the dataset into train and validation sets
    mgr.split_dataset(test_size=0.3, use_eval=True)

    # Create the mapping
    # Map labels to IDs BEFORE training
    mgr.map_labels_to_ids()

    # 3a. Optional Segment Inspection Table
    # =========================================================================
    if mgr.config.inspect_segments:
        _id2label = (
            {v: k for k, v in mgr.label_to_id.items()}
            if hasattr(mgr, "label_to_id") and mgr.label_to_id
            else None
        )
        tests.inspect_segment_table(
            mgr.dataset,
            ids=mgr.config.inspect_segment_ids,
            id2label=_id2label,
            split=mgr.config.inspect_segments_split,
        )

    # 3b. Optional Integrity Tests (Run only in DEBUG_MODE)
    # =========================================================================
    if debug:
        print("[DEBUG_MODE] Running integrity checks...")
        # Perform an integrity test on the loaded dataset.
        mgr.inspect_dataset_samples(num_samples=6)
        # This validates that the audio segments are correctly linked to their
        # respective song IDs and labels.
        #mgr.test_dataset_integrity()
        # Run the new spectrogram integrity test
        #mgr.test_spectrogram_integrity()
        # Validate the dataset split
        #mgr.validate_dataset_split()
    
    # 4. EVALUATE ON TRAINING
    # =========================================================================
    # Pass the HuggingFace dataset splits directly to the Trainer.
    # remove_unused_columns=True (set inside get_trainer) automatically drops
    # 'audio' and 'song_id', keeping only 'input_values' and 'labels'.
    # Get the checkpoint path from the command line

    if mode in ['train', 'both']:
        # Load only model weights via from_pretrained (safetensors, no torch.load).
        # The optimizer always starts fresh — safe on PyTorch < 2.6.
        # In Colab, _ensure_resume_checkpoint() copies the checkpoint from Drive
        # to the local models/ folder if it isn't already there.
        if mgr.config.checkpoint_to_load:
            mgr._ensure_resume_checkpoint()
            mgr.load_model_from_checkpoint(mgr.config.checkpoint_to_load)

        mgr.save_training_info()
        trainer = mgr.get_trainer(train_ds=mgr.dataset['train'], eval_ds=mgr.dataset['eval'])
        mgr.run_training(trainer)
        mgr.save_session_steps(trainer)


    # 5. EVALUATE ON TEST
    # =========================================================================
    if mode in ['test', 'both']:
        print("[INFO] Running evaluation on test set...")

        # After training: evaluate the best in-memory model (load_best_model_at_end).
        # Test-only mode: load from the specified checkpoint if provided.
        eval_checkpoint = (
            mgr.config.checkpoint_to_load
            if mode == 'test' and mgr.config.checkpoint_to_load
            else None
        )
        test_results = mgr.evaluate_on_test(checkpoint_folder=eval_checkpoint)

        file_path = os.path.join(mgr.config.output_dir, "test_performance.txt")
        
        with open(file_path, "w") as f:
            f.write(f"Results for this run: {test_results}\n")
        print(f"[SUCCESS] Results saved to {file_path}")

    # 6. EVALUATE ON TEST (DETAIL)
    # =========================================================================
    if mode == 'test_detail':
        print("[INFO] Running detailed per-segment evaluation...")
        detail_checkpoint = mgr.config.checkpoint_to_load or None
        mgr.evaluate_on_test_detail(
            checkpoint_folder=detail_checkpoint,
            split=mgr.config.detail_split_test,
            filter_by=mgr.config.detail_filter_test,
            aggregate_songs=mgr.config.detail_aggregate_songs,
        )

    # 7. BACKUP TO DRIVE (Colab only)
    # =========================================================================
    # Copies the timestamped checkpoint folder and runs/ to Google Drive so
    # they survive Colab session termination. No-op on a local machine.
    mgr.backup_to_drive()

    # Signal that the full pipeline execution has completed successfully.
    print("[SUCCESS] Main pipeline execution finished.")

if __name__ == "__main__":
    main()