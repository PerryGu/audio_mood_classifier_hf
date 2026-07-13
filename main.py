"""
Main Entry Point
This file serves as the primary interface for executing the end-to-end ML pipeline.
It orchestrates the initialization, data processing, and model loading phases 
by utilizing the PipelineManager. This modular approach ensures a clean, 
reproducible workflow for training and inference.
"""
import os
import argparse
from src.pipeline_manager import PipelineManager

# Initialize the PipelineManager object.
# This creates a persistent state container to hold the dataset and model
# throughout the session, preventing redundant initialization.
mgr = PipelineManager()

def main():
    print("Starting the main function")
    # 0. SETUP & CONFIGURATION
    # =========================================================================
    # Set this to True if you want to run integrity checks, 
    # or False for a clean, fast training run.
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run the ML Pipeline")
    parser.add_argument("--debug", action="store_true", help="Run with integrity checks")

    # Selecting execution mode: 'train', 'test', or 'both' (default is 'both')
    parser.add_argument("--mode", choices=['train', 'test', 'both'], default='both', 
                        help="Choose execution mode: 'train' only, 'test' only, or 'both'.")

    args = parser.parse_args()

    # Configure the environment variables (like HF_TOKEN).
    # This step ensures the manager has the necessary credentials to access
    # private or gated datasets from Hugging Face.

    # === TRAINING HYPERPARAMETERS ===
    BASE_LEARNING_RATE = 1e-5
    BATCH_SIZE = 32
    NUM_TRAIN_EPOCHS = 8
    OPTIMIZER_NAME = "adamw_torch_fused"
    LOGGING_STEPS = 50
    REPORT_TO = "tensorboard"           # "tensorboard" | "wandb" | "all"
    WANDB_PROJECT = "audio-mood-classifier"

    # --- Checkpoint Resumption ---
    # Leave both empty for a clean new run.
    # Fill both to resume weights from a previous checkpoint.
    # A fresh timestamped output folder is ALWAYS created regardless.
    RESUME_RUN_FOLDER = "mood_classifier_2026-07-13_13-22"  # e.g. "mood_classifier_2026-07-12_20-20"
    RESUME_CKPT_NAME  = "checkpoint-62"   # e.g. "checkpoint-248"

    # Update the config object
    mgr.config.learning_rate = BASE_LEARNING_RATE
    mgr.config.batch_size = BATCH_SIZE
    mgr.config.num_train_epochs = NUM_TRAIN_EPOCHS
    mgr.config.optim = OPTIMIZER_NAME
    mgr.config.report_to = REPORT_TO
    mgr.config.logging_steps = LOGGING_STEPS
    mgr.config.parent_run_folder = RESUME_RUN_FOLDER
    mgr.config.resume_checkpoint_name = RESUME_CKPT_NAME

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_RUN_NAME"] = mgr.config.session_name

    print(f"[INFO] Execution Mode : {args.mode.upper()}")
    print(f"[INFO] Output folder  : {mgr.config.output_dir}")
    print(f"[INFO] Session name   : {mgr.config.session_name}")
    print(f"[INFO] TensorBoard    : {mgr.config.logging_dir}")
    print(f"[INFO] Continuous log : {mgr.config.continuous_logging_dir}")
    if RESUME_RUN_FOLDER:
        print(f"[INFO] Resuming weights from: {mgr.config.checkpoint_to_load}")
    else:
        print("[INFO] Starting a new training run from scratch.")
    print(f"[INFO] LR: {BASE_LEARNING_RATE}  |  Batch: {BATCH_SIZE}  |  Epochs: {NUM_TRAIN_EPOCHS}")
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

    # 3. Optional Integrity Tests (Run only in DEBUG_MODE)
    # =========================================================================
     # Use the argument from the terminal, default to False
    if args.debug:
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

    if args.mode in ['train', 'both']:
        # Load only model weights via from_pretrained (safetensors, no torch.load).
        # The optimizer always starts fresh — safe on PyTorch < 2.6.
        if mgr.config.checkpoint_to_load:
            mgr.load_model_from_checkpoint(mgr.config.checkpoint_to_load)

        mgr.save_training_info()
        trainer = mgr.get_trainer(train_ds=mgr.dataset['train'], eval_ds=mgr.dataset['eval'])
        trainer.train()
        mgr.save_session_steps(trainer)


    # 5. EVALUATE ON TEST
    # =========================================================================
    if args.mode in ['test', 'both']:
        print("[INFO] Running evaluation on test set...")

        # After training: evaluate the best in-memory model (load_best_model_at_end).
        # Test-only mode: load from the specified checkpoint if provided.
        eval_checkpoint = (
            mgr.config.checkpoint_to_load
            if args.mode == 'test' and mgr.config.checkpoint_to_load
            else None
        )
        test_results = mgr.evaluate_on_test(checkpoint_folder=eval_checkpoint)

        file_path = os.path.join(mgr.config.output_dir, "test_performance.txt")
        
        with open(file_path, "w") as f:
            f.write(f"Results for this run: {test_results}\n")
        print(f"[SUCCESS] Results saved to {file_path}")

    # Signal that the full pipeline execution has completed successfully.
    print("[SUCCESS] Main pipeline execution finished.")

if __name__ == "__main__":
    main()