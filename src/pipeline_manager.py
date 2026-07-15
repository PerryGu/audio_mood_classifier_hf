"""
Pipeline Manager
This module implements the PipelineManager class, which acts as the central 
state manager for the entire project. It maintains the lifecycle of the 
dataset, model, and environment variables in memory, enabling interactive 
development, persistent state handling, and efficient module reloading 
without the need for full session restarts.
"""

import os
import sys
import torch

from src.utils import tests
from src.utils import load_model
from datasets import load_from_disk
from src.data_processing import data_loader
from src.data_processing import data_processor
from src.training.config import TrainingConfig
from src.data_processing.dataset import AudioDataset
from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator, AutoModelForAudioClassification


class _ContinuousLoggingCallback(TrainerCallback):
    """
    Writes training metrics to a dedicated 'continuous' TensorBoard directory
    using global steps (previous_sessions_total + current_step) so that
    multiple training sessions appear as one unbroken curve.
    """

    def __init__(self, offset: int, log_dir: str):
        self.offset = offset
        self.log_dir = log_dir
        self._writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._writer is None or not logs:
            return
        global_step = state.global_step + self.offset
        for tag, value in logs.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(tag, value, global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class PipelineManager:
    """
    Manages the end-to-end ML pipeline state.
    This class maintains the lifecycle of the dataset and model, 
    allowing for interactive development without re-running initialization steps.
    """

     # =========================================================================
    # PIPELINE WORKFLOW ORDER
    # =========================================================================
    def __init__(self):
        # Persistent storage for pipeline components
        self.dataset = None
        self.feature_extractor = None
        self.model = None
        self.tokenizer = None
        self.hf_token = None
        self.device = None
        self.config = TrainingConfig()
        print("[INFO] Pipeline Manager initialized.")

    # =========================================================================
    # CONFIGURATION: Configure the environment variables (like HF_TOKEN).
    # =========================================================================
    def setup_environment(self):
        """
        Prepares the runtime environment for both local and Google Colab execution.

        Colab:
          - Installs packages that are not bundled with the default Colab runtime
            (torch, transformers, datasets, etc. are already present).
          - Reads HF_TOKEN from Colab Secrets (Add via the 🔑 panel on the left).

        Local:
          - Loads HF_TOKEN from the .env file in the project root.
        """
        try:
            import google.colab  # type: ignore  — just checks we are in Colab

            # Packages missing from the default Colab runtime.
            # torch, transformers, datasets, numpy, pandas, sklearn, tqdm are pre-installed.
            _colab_packages = [
                "librosa==0.11.0",
                "pyloudnorm==0.2.0",
                "pydub==0.25.1",
                "mutagen==1.48.1",
                "python-dotenv==1.2.2",
            ]
            print("[INFO] Colab: installing missing runtime packages...")
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + _colab_packages,
                check=False,
            )
            print("[INFO] Colab: packages ready.")

            # Colab Secrets with "Notebook access" ON are injected into the process
            # environment, so os.getenv() works from a script (!python main.py).
            self.hf_token = os.getenv("HF_TOKEN")
            print("[INFO] Google Colab environment detected.")

        except ImportError:
            from dotenv import load_dotenv
            load_dotenv()
            self.hf_token = os.getenv("HF_TOKEN")
            print("[INFO] Local environment detected.")

        # Authenticate the Hugging Face Hub client globally so that every
        # from_pretrained() / load_dataset() call in this session uses the token,
        # even if the individual call site does not pass it explicitly.
        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token, add_to_git_credential=False)
            print("[INFO] Hugging Face Hub: authenticated successfully.")
        else:
            print("[WARN] HF_TOKEN not found — Hub access will be unauthenticated.")
            print("       Local: add it to your .env file.")
            print("       Colab: add it to Secrets (🔑) and enable 'Notebook access'.")

        return self.hf_token

    # =========================================================================
    # 1. DATA LOADING: Load raw dataset from local cache source.
    # =========================================================================
    def run_data_loading(self):
        """
        Loads the dataset into memory. 
        If the dataset is already loaded, it returns the cached instance.
        """
        if self.dataset is None:
            print("[INFO] Loading dataset from source...")
            self.dataset = data_loader.load_hf_dataset(
                repo_id="guyPerry/audio-mood-dataset", 
                split="train", 
                token=self.hf_token,
                use_local=True
            )
        else:
            print("[INFO] Dataset already exists in memory.")
       
        print("------------------------------------\n")

        print(f"[DEBUG] Cache loaded. Dataset object: {self.dataset}")
        print(f"[DEBUG] Features: {self.dataset.features}") 
        print(f"[DEBUG] Number of samples: {len(self.dataset)}") 
        
        print(f"[DEBUG] Local data loaded from path: {self.dataset.cache_files}")
        print(f"[DEBUG] Total samples found in local cache: {len(self.dataset)}")

        return self.dataset


        
    def run_model_loading(self):
        """
        Initializes the model and feature extractor.
        Automatically detects if CUDA (GPU) is available for optimal performance.
        """
        print("[INFO] Loading model components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Calling the function from load_model.py
        # Passing num_labels from the config to ensure correct model architecture
        self.feature_extractor, self.model = load_model.load_ast_model(
            model_ckpt=self.config.model_ckpt,
            device=self.device,
            num_labels=self.config.num_labels
        )

        # Apply layer freezing
        self.setup_model_for_training()
        
        return self.feature_extractor, self.model
    


    # =========================================================================
    # SETUP MODEL FOR TRAINING: Freezes the base model and ensures the classification head 
    # is set to the correct number of labels.
    # =========================================================================
    def setup_model_for_training(self):
        """
        Freezes the base model and ensures the classification head 
        is set to the correct number of labels.
        """
        # 1. Freeze base model except for the classifier
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        
        # 2. Ensure the correct number of labels is set
        self.model.config.num_labels = self.config.num_labels
        
        print(f"[INFO] Model base frozen. Training only classifier head with {self.config.num_labels} labels.")


    # =========================================================================
    # 2. FEATURE EXTRACTION: Process audio files into spectrograms.
    # =========================================================================
    def map_labels_to_ids(self):
        """
        Maps string labels to numeric IDs after dataset splitting.
        """
        # Ensure we have the mapping dictionary
        if not hasattr(self, 'label_to_id') or self.label_to_id is None:
            self.create_label_to_id()

        for split in self.dataset:
            # We perform the mapping directly on the split dataset
            current_labels = self.dataset[split]['label']
            new_labels = [self.label_to_id[label] for label in current_labels]
            
            # Efficiently replace the 'label' column with numeric IDs under the
            # name 'labels', which is what the HuggingFace Trainer expects.
            self.dataset[split] = self.dataset[split].remove_columns(['label'])
            self.dataset[split] = self.dataset[split].add_column('labels', new_labels)
            
        print("[INFO] Dataset labels mapped to numeric IDs successfully.")


    def create_label_to_id(self):
        """
        Scans the training dataset to create a mapping from label names to numeric IDs.
        """
        print("...create_label_to_id...")
        print(f"[DEBUG] Dataset len: {len(self.dataset['train'])}")
        if self.dataset is None or 'train' not in self.dataset:
            print("[ERROR] Dataset not loaded or split. Run run_data_loading() and split_dataset() first.")
            return None

        unique_labels = sorted(list(set(self.dataset['train']['label'])))
        print(f"[DEBUG] unique_labels: {unique_labels}")
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}

        print(f"[INFO] Label mapping created: {self.label_to_id}")
        return self.label_to_id


    def prepare_dataset(self):
        """
        Processes the dataset to extract features (spectrograms) 
        and prepares it for model training. This method maps the 
        preprocessing logic across all segments in the dataset.
        """

        # Path where we want to save/load our processed data
        cache_path = "./data/processed_dataset"

        # Check if the processed dataset already exists
        if os.path.exists(cache_path):
            # Validate the cache by reading dataset_info.json only — no Arrow files
            # are opened, so there are no memory-mapped file locks (important on Windows).
            import json
            info_path = os.path.join(cache_path, "dataset_info.json")
            cache_is_valid = False
            if os.path.exists(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                cache_is_valid = "input_values" in info.get("features", {})

            if cache_is_valid:
                print(f"[INFO] Loading existing processed dataset from {cache_path}...")
                self.dataset = load_from_disk(cache_path)
                print("[INFO] Dataset loaded successfully.")
                return
            else:
                import shutil
                print("[WARN] Cached dataset is missing 'input_values'. Deleting stale cache and re-running feature extraction...")
                shutil.rmtree(cache_path)

        # If not, proceed with feature extraction
        # Colab doesn't support multiprocessing in dataset.map() — use 1 process there.
        try:
            import google.colab  # type: ignore
            _num_proc = 1
        except ImportError:
            _num_proc = 2

        # Colab: small batches to avoid deserializing hundreds of large audio
        # arrays at once (~160K floats each) which stalls progress at 0%.
        _batch_size = 16 if _num_proc == 1 else 100

        print(f"[INFO] Preparing dataset (feature extraction, {_num_proc} process(es), batch {_batch_size})...")

        self.dataset = self.dataset.map(
            lambda x: data_processor.preprocess_audio(x, self.feature_extractor),
            batched=True,
            batch_size=_batch_size,
            num_proc=_num_proc
        )
        
        # Save to disk so we can load it next time
        self.dataset.save_to_disk(cache_path)
        
        print("[INFO] Dataset preparation complete and saved to disk. Ready for training.")
        

    
    # =========================================================================;
    # 3. DATASET SPLITTING: Divide data into training and validation sets.
    # =========================================================================
    def split_dataset(self, test_size=0.2, use_eval=True):
        """
        Orchestrates the dataset splitting process.
        Updates the internal dataset attribute with the split version.
        """
        from src.data_processing import data_processor
        
        print("[INFO] Starting dataset split process...")
        
        # Calling the updated split_data which now returns 3 values
        train_ds, test_ds, eval_ds = data_processor.split_data(self.dataset, test_size=test_size, use_eval=use_eval)
        
        # Updating the internal attribute based on whether eval is used
        if use_eval:
            self.dataset = {"train": train_ds, "test": test_ds, "eval": eval_ds}
        else:
            self.dataset = {"train": train_ds, "test": test_ds}
            
        print("[INFO] Dataset split successfully completed.")


    # =========================================================================
    # 4. TRAINING PREPARATION: Initialize the AudioDataset for the training process.
    # =========================================================================
    def prepare_dataset_for_training(self):
        """
        Initializes the AudioDataset for the training process.
        
        This method creates the AudioDataset instance directly, ensuring 
        separation between data preparation and validation logic.
        
        Returns:
            train_ds: An initialized instance of AudioDataset ready for loading.
        """
        # Create the AudioDataset instance using the processed training data
        # and the previously initialized feature extractor and label mapping
        self.train_ds = AudioDataset(
            dataset=self.dataset['train'],
            feature_extractor=self.feature_extractor,
            label_to_id=self.label_to_id
        )
        
        print("[INFO] AudioDataset initialized for training.")
        return self.train_ds


    # =========================================================================
    # 5. LOAD MODEL WEIGHTS FROM CHECKPOINT (safe, no torch.load)
    # =========================================================================
    def load_model_from_checkpoint(self, checkpoint_path):
        """
        Loads model weights from a checkpoint using from_pretrained (safetensors).
        Avoids torch.load entirely, so it works on PyTorch < 2.6.
        The optimizer state is NOT restored — training continues with a fresh optimizer.
        """
        from pathlib import Path
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = Path(os.getcwd()) / checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] Checkpoint not found: {path}")

        print(f"[INFO] Loading model weights from checkpoint: {path}")
        self.model = AutoModelForAudioClassification.from_pretrained(
            path, local_files_only=True
        )
        self.model.to(self.device)
        self.setup_model_for_training()
        print("[INFO] Model weights loaded. Optimizer will start fresh.")


    # =========================================================================
    # 6. TRAINING EXECUTION
    # =========================================================================
    # -------------------------------------------------------------------------
    # Session step tracking helpers (for continuous TensorBoard logging)
    # -------------------------------------------------------------------------
    def _load_step_offset(self) -> int:
        """
        Return the total steps completed in all previous sessions of this lineage.
        session_log.json lives alongside the continuous TensorBoard logs so that
        the step count follows the lineage (parent run) rather than the output dir.
        """
        import json
        log_path = os.path.join(self.config.continuous_logging_dir, "session_log.json")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f).get("total_steps", 0)
        return 0

    def save_training_info(self):
        """
        Writes training_info.json to the output folder so you can always tell
        where each session came from — fresh start or which checkpoint it resumed.
        """
        import json
        from datetime import datetime

        resumed_from = self.config.checkpoint_to_load or "fresh start"
        info = {
            "run_name":      self.config.run_name,
            "started_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resumed_from":  resumed_from,
            "num_epochs":    self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size":    self.config.batch_size,
        }
        os.makedirs(self.config.output_dir, exist_ok=True)
        info_path = os.path.join(self.config.output_dir, "training_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"[INFO] training_info.json saved → {info_path}")

    def save_session_steps(self, trainer):
        """
        Call this after trainer.train() to persist the cumulative step count.
        Stored in continuous_logging_dir so resumed sessions always extend the
        correct lineage regardless of which output folder they save checkpoints to.
        """
        import json
        session_steps = trainer.state.global_step
        total = self.config.step_offset + session_steps
        log_path = os.path.join(self.config.continuous_logging_dir, "session_log.json")
        os.makedirs(self.config.continuous_logging_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"total_steps": total}, f, indent=2)
        print(f"[INFO] session_log.json updated → total_steps={total}")

    @staticmethod
    def _compute_metrics(eval_pred):
        """
        Computes accuracy and shows how far above random chance the model is.
        Passed to both the training Trainer (shown after every eval epoch) and
        the test Trainer (shown in the final test metrics block).
        """
        import numpy as np
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        num_classes = logits.shape[-1]
        random_baseline = 1.0 / num_classes                          # e.g. 0.333 for 3 classes
        gain_pp = (accuracy - random_baseline) * 100                  # percentage points above random
        return {
            "accuracy":          round(accuracy * 100, 2),           # e.g. 45.30  (%)
            "random_baseline":   round(random_baseline * 100, 2),    # e.g. 33.33  (%)
            "gain_over_random":  round(gain_pp, 2),                  # e.g. +11.97 (pp above random)
        }

    def get_trainer(self, train_ds, eval_ds=None):
        """
        Initializes and returns the Hugging Face Trainer with the specified
        training arguments, a custom data collator to handle tensor stacking,
        and the model configured for training.
        """
        print(f"[INFO] Configuring Trainer: output_dir={self.config.output_dir}")
        print(f"[INFO] Configuring Trainer: epochs={self.config.num_train_epochs}")

        # Ensure the model is in training mode and gradients are enabled for the classifier
        self.model.train()

        # logging_dir was deprecated in transformers v5.2 — use the environment
        # variable TENSORBOARD_LOGGING_DIR instead.
        os.environ["TENSORBOARD_LOGGING_DIR"] = self.config.logging_dir

        # Determine step offset from previous sessions and attach continuous logger.
        self.config.step_offset = self._load_step_offset()
        continuous_callback = _ContinuousLoggingCallback(
            offset=self.config.step_offset,
            log_dir=self.config.continuous_logging_dir,
        )
        if self.config.step_offset:
            print(f"[INFO] Continuous logging offset: {self.config.step_offset} steps "
                  f"(continuing from previous sessions)")
        print(f"[INFO] Continuous log dir: {self.config.continuous_logging_dir}")

        # Define training arguments based on the config object
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            save_strategy=self.config.save_strategy,
            eval_strategy=self.config.eval_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=self.config.report_to,
            remove_unused_columns=True,
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            label_names=["labels"],
        )

        # Initialize and return the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[continuous_callback],
        )
        
        return trainer


    # =========================================================================
    # EVALUATE ON TEST: Evaluate the model on the test set.
    # =========================================================================
    def evaluate_on_test(self, checkpoint_folder=None):
        """
        Evaluating on the test set.
        If checkpoint_folder is None, it uses the best model already in the trainer.
        If a path is provided it loads that specific checkpoint. The path can be:
          - An absolute path: 'F:\\...\\mood_classifier_2026-07-10_19-52\\checkpoint-288'
          - A path relative to the project root: 'mood_classifier_2026-07-10_19-52/checkpoint-288'
        """
        # If a specific checkpoint folder is requested
        if checkpoint_folder:
            from pathlib import Path
            checkpoint_path = Path(checkpoint_folder)
            if not checkpoint_path.is_absolute():
                # Resolve relative to the project root, NOT to the current session's
                # output_dir — the checkpoint may come from a previous training run
                # which has a different timestamp in its folder name.
                checkpoint_path = Path(os.getcwd()) / checkpoint_folder

            # HuggingFace from_pretrained calls os.path.isdir() first — if the directory
            # does not exist it falls through to Hub repo-ID validation, which rejects
            # Windows absolute paths with a confusing error. Catch this early.
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"[ERROR] Checkpoint directory not found: {checkpoint_path}\n"
                    f"Pass the path relative to the project root, e.g.:\n"
                    f"  evaluate_on_test(checkpoint_folder='mood_classifier_2026-07-10_19-52/checkpoint-288')"
                )

            print(f"[INFO] Loading model from: {checkpoint_path}")
            model_to_test = AutoModelForAudioClassification.from_pretrained(
                checkpoint_path,
                local_files_only=True,
            )
        else:
            # No checkpoint specified — use the model already in memory.
            # After trainer.train() with load_best_model_at_end=True, self.model
            # already holds the best checkpoint weights.
            print("[INFO] No checkpoint specified — evaluating with the in-memory model.")
            model_to_test = self.model

        # Create a new Trainer instance for evaluation.
        # output_dir must be set explicitly — without it transformers 5.x
        # falls back to the default "tmp_trainer" folder in the project root.
        test_args = TrainingArguments(
            output_dir=self.config.output_dir,
            report_to="none",
        )
        test_trainer = Trainer(
            model=model_to_test,
            args=test_args,
            eval_dataset=self.dataset['test'],
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics,
        )

        metrics = test_trainer.evaluate()

        # Log and save metrics via the evaluation trainer
        if test_trainer.state.is_world_process_zero:
            test_trainer.log_metrics("test", metrics)
            test_trainer.save_metrics("test", metrics)

        accuracy        = metrics.get("eval_accuracy", 0)
        random_baseline = metrics.get("eval_random_baseline", 0)
        gain            = metrics.get("eval_gain_over_random", 0)
        loss            = metrics.get("eval_loss", 0)
        print("\n[RESULTS] ─────────────────────────────────")
        print(f"  Accuracy:         {accuracy:.2f}%")
        print(f"  Random baseline:  {random_baseline:.2f}%")
        print(f"  Gain over random: +{gain:.2f} pp")
        print(f"  Loss:             {loss:.4f}")
        print("[RESULTS] ─────────────────────────────────\n")
        return metrics


    # =========================================================================
    # DRIVE BACKUP: Copy outputs to Google Drive (Colab only).
    # =========================================================================
    def backup_to_drive(self, drive_root: str = "/content/drive/MyDrive/audio_mood_classifier_hf"):
        """
        Colab only: copies the timestamped model checkpoint folder and the runs
        folder to Google Drive so they survive session termination.

        drive_root should point to the project folder on Drive, e.g.:
            /content/drive/MyDrive/audio_mood_classifier_hf

        On a local machine this method returns immediately without doing anything.
        """
        try:
            import google.colab  # type: ignore
        except ImportError:
            return  # Not in Colab — nothing to do.

        import shutil
        from pathlib import Path

        drive_root = Path(drive_root)
        if not drive_root.exists():
            print(f"[WARN] Drive backup: root folder not found at '{drive_root}'.")
            print("       Make sure Drive is mounted and the path is correct.")
            return

        errors = []

        # 1. Backup the timestamped model/checkpoint folder
        src_model = Path(self.config.output_dir)
        if src_model.exists():
            dst_model = drive_root / "models" / src_model.name
            print(f"[INFO] Backing up checkpoints → {dst_model} ...")
            if dst_model.exists():
                shutil.rmtree(dst_model)
            try:
                shutil.copytree(src_model, dst_model)
                size_mb = sum(f.stat().st_size for f in dst_model.rglob("*") if f.is_file()) / 1024 / 1024
                print(f"[INFO] Checkpoints backed up ({size_mb:.0f} MB) ✓")
            except Exception as e:
                errors.append(f"Checkpoints: {e}")
        else:
            print(f"[WARN] Drive backup: model folder '{src_model}' not found — skipping.")

        # 2. Backup the full runs folder (TensorBoard logs)
        src_runs = Path(os.getcwd()) / "runs"
        if src_runs.exists():
            dst_runs = drive_root / "runs"
            print(f"[INFO] Backing up runs → {dst_runs} ...")
            if dst_runs.exists():
                shutil.rmtree(dst_runs)
            try:
                shutil.copytree(src_runs, dst_runs)
                print("[INFO] Runs backed up ✓")
            except Exception as e:
                errors.append(f"Runs: {e}")
        else:
            print("[WARN] Drive backup: runs folder not found — skipping.")

        if errors:
            print("[WARN] Some items failed to back up:")
            for err in errors:
                print(f"       {err}")
        else:
            print("[INFO] Drive backup complete.")

    #**************************************************************************
    # RELOAD MODULES: Refresh the underlying utility modules.
    #**************************************************************************
    def reload_modules(self):
        """
        Refreshes underlying utility modules.
        Useful when changes are made to the source files without restarting the session.
        """
        import importlib
        import src.pipeline_manager  
        from src.training import engine
        from src.utils import tests, load_model
        from src.data_processing import data_loader, data_processor
       
        importlib.reload(src.pipeline_manager) # Add this
        importlib.reload(data_loader)
        importlib.reload(data_processor)
        importlib.reload(tests)
        importlib.reload(load_model)
        importlib.reload(engine)
        
        print("[INFO] Pipeline utilities have been reloaded.")


    #**************************************************************************
    # =========================================================================
    # UTILITIES & DEBUGGING:
    # =========================================================================
    # INSPECT DATASET SAMPLES: Inspect the dataset samples.
    def inspect_dataset_samples(self, num_samples=3):
        """
        Calls the external inspection function using the internal dataset.
        """
        if self.dataset is not None:
            # We import the inspection function from the tests module
            from src.utils import tests
            tests.inspect_dataset(self.dataset, num_samples=num_samples)
        else:
            print("[ERROR] Dataset not loaded. Run 'run_data_loading' first.")


    # DATA INTEGRITY CHECK: Validate the integrity of the dataset.
    def test_dataset_integrity(self):
        """
        Executes a data integrity check to ensure segments are correctly grouped.
        Requires the dataset to be loaded first.
        """
        from src.utils import tests
        if self.dataset is not None:
            tests.test_dataset_integrity(self.dataset['train'])
        else:
            print("[ERROR] Dataset is not loaded. Run 'run_data_loading' first.")

    
    # TESTING AND VALIDATION: Validate the integrity of the spectrograms.
    def test_spectrogram_integrity(self):
        """Executes a validation check for processed spectrograms."""
        from src.utils import tests
        if self.dataset is not None:
            tests.test_spectrogram_integrity(self.dataset['train'])
        else:
            print("[ERROR] Dataset not prepared. Run 'prepare_dataset' first.")


    # DATASET VALIDATION: Validate the integrity of the split dataset.
    def validate_dataset_split(self, ids=None):
        """Dedicated manual step to validate the integrity of the split."""
        from src.utils import tests
        print("[INFO] Running manual split validation...", ids)
        # Checking for song leakage
        tests.check_for_leakage(self.dataset)
        # Generating the distribution table
        tests.generate_split_summary(self.dataset, ids=ids)
