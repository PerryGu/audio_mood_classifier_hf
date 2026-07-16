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
from src.config import TrainingConfig
from src.data_processing.dataset import AudioDataset
from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator, AutoModelForAudioClassification
from src.data_processing.augmentation import SpecAugmentCollator


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
            try:
                from huggingface_hub import login
                login(token=self.hf_token, add_to_git_credential=False)
                print("[INFO] Hugging Face Hub: authenticated successfully.")
            except Exception as e:
                # A network error (e.g. 504 Gateway Timeout) during token validation
                # must not abort the run — local checkpoints and cached models work
                # without Hub connectivity.
                print(f"[WARN] Hugging Face Hub login failed ({type(e).__name__}: {e})")
                print("[WARN] Continuing without Hub authentication — local files only.")
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


        
    def _ensure_pretrained_model(self) -> None:
        """
        Resolves the source for the base pre-trained AST model and updates
        self.config.model_ckpt accordingly. Priority order:

          1. models/ast_pretrained/ already exists with model weights
             → use it directly (fastest, works locally and in Colab).
          2. In Colab and COLAB_DRIVE_MODEL_PATH points to a valid Drive folder
             → copy files into models/ast_pretrained/ then use that path.
          3. Fall through to HF Hub (self.config.model_ckpt unchanged).
        """
        import shutil
        from pathlib import Path

        local_dir = Path(os.getcwd()) / "models" / "ast_pretrained"

        def _has_weights(p: Path) -> bool:
            return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()

        # Case 1 — already present locally
        if local_dir.is_dir() and _has_weights(local_dir):
            self.config.model_ckpt = str(local_dir)
            print(f"[INFO] Pre-trained model found in project folder — loading from '{local_dir}'.")
            return

        # Case 2 — Colab: copy from Drive
        try:
            import google.colab  # type: ignore
            drive_path_str = os.environ.get("COLAB_DRIVE_MODEL_PATH", "")
            drive_path = Path(drive_path_str) if drive_path_str else None

            if drive_path and drive_path.is_dir() and _has_weights(drive_path):
                print(f"[INFO] Copying pre-trained model from Drive → '{local_dir}' ...")
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                shutil.copytree(drive_path, local_dir)
                size_mb = (local_dir / "model.safetensors").stat().st_size / 1024 / 1024
                print(f"[INFO] Model copied from Drive ({size_mb:.0f} MB) ✓")
                self.config.model_ckpt = str(local_dir)
                return
            else:
                print("[INFO] Drive model path not found or incomplete — will download from HF Hub.")
        except ImportError:
            pass  # Not in Colab

        # Case 3 — download from HF Hub (model_ckpt already set to HF repo ID)
        # _hf_hub_download is set so run_model_loading() knows to cache it afterwards.
        self._hf_hub_download = str(local_dir)
        print(f"[INFO] Loading pre-trained model from HuggingFace Hub ({self.config.model_ckpt}).")

    def _ensure_resume_checkpoint(self) -> None:
        """
        Colab only: if the checkpoint configured in config.py does not exist in
        the local models/ folder but COLAB_DRIVE_CHECKPOINT_PATH is set and
        points to a valid Drive folder, copies it to the expected local path so
        that load_model_from_checkpoint() can proceed normally.

        On a local machine, or when no resumption is configured, this is a no-op.
        The caller (main.py) is responsible for checking config.checkpoint_to_load
        before calling this method.
        """
        import shutil
        from pathlib import Path

        checkpoint_path = Path(self.config.checkpoint_to_load)

        if checkpoint_path.exists():
            print(f"[INFO] Resume checkpoint found locally — '{checkpoint_path}'.")
            return

        # Not found locally — try copying from Drive (Colab only).
        try:
            import google.colab  # type: ignore
        except ImportError:
            return  # Local machine — let load_model_from_checkpoint raise the error.

        drive_path_str = os.environ.get("COLAB_DRIVE_CHECKPOINT_PATH", "").strip()
        if not drive_path_str:
            print(
                "[WARN] Resume checkpoint not found locally and "
                "COLAB_DRIVE_CHECKPOINT_PATH is not set in main.py.\n"
                "       Set it to the full Drive path of the checkpoint folder."
            )
            return

        drive_path = Path(drive_path_str)
        if not drive_path.exists():
            print(f"[WARN] Drive checkpoint path not found: '{drive_path}'.")
            print("       Make sure Drive is mounted and the path is correct.")
            return

        print(f"[INFO] Copying checkpoint from Drive → '{checkpoint_path}' ...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive_path, checkpoint_path)
        size_mb = sum(
            f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()
        ) / 1024 / 1024
        print(f"[INFO] Checkpoint copied from Drive ({size_mb:.0f} MB) ✓")

    def run_model_loading(self):
        """
        Initializes the model and feature extractor.
        Resolves the model source (local folder, Drive copy, or HF Hub),
        then loads the AST model and feature extractor.
        Automatically detects if CUDA (GPU) is available for optimal performance.

        When the model is downloaded from HF Hub for the first time it is saved
        to models/ast_pretrained/ so subsequent runs load it locally (Case 1).
        """
        self._hf_hub_download = None  # reset flag
        self._ensure_pretrained_model()

        print("[INFO] Loading model components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Calling the function from load_model.py
        # Passing num_labels from the config to ensure correct model architecture
        self.feature_extractor, self.model = load_model.load_ast_model(
            model_ckpt=self.config.model_ckpt,
            device=self.device,
            num_labels=self.config.num_labels
        )

        # If the model was just downloaded from HF Hub, persist it to
        # models/ast_pretrained/ so future runs skip the download entirely.
        if self._hf_hub_download:
            save_dir = self._hf_hub_download
            print(f"[INFO] Saving pre-trained model to '{save_dir}' for future runs...")
            import os as _os
            _os.makedirs(save_dir, exist_ok=True)
            self.feature_extractor.save_pretrained(save_dir)
            self.model.save_pretrained(save_dir)
            print(f"[INFO] Pre-trained model cached locally ✓")
            self._hf_hub_download = None

        # Apply layer freezing
        self.setup_model_for_training()
        
        return self.feature_extractor, self.model
    


    # =========================================================================
    # SETUP MODEL FOR TRAINING: Freezes the base model and ensures the classification head 
    # is set to the correct number of labels.
    # =========================================================================
    def setup_model_for_training(self):
        """
        Selectively freezes the model based on config.num_unfrozen_layers.

          0  → only the classifier head is trainable (default, fastest)
          N  → the last N encoder layers + final layernorm + classifier head
          12 → all encoder layers unfrozen (full fine-tuning)

        Called automatically after every model load (both initial load and
        checkpoint resume), so the freeze state is always consistent.
        """
        n = self.config.num_unfrozen_layers
        total_layers = self.model.config.num_hidden_layers  # 12 for AST base

        # Step 1: freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Step 2: selectively unfreeze
        for name, param in self.model.named_parameters():
            # The classifier head is always trainable
            if "classifier" in name:
                param.requires_grad = True
                continue

            if n > 0:
                # Unfreeze the last n encoder layers.
                # Handles both transformers naming conventions:
                #   new (v5): audio_spectrogram_transformer.layers.{i}.*
                #   old:      audio_spectrogram_transformer.encoder.layer.{i}.*
                for idx in range(total_layers - n, total_layers):
                    if f"layers.{idx}." in name or f"layer.{idx}." in name:
                        param.requires_grad = True
                        break
                # Also unfreeze the final transformer layernorm
                if "audio_spectrogram_transformer.layernorm" in name:
                    param.requires_grad = True

        # Step 3: report what's trainable
        self.model.config.num_labels = self.config.num_labels
        total_params    = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if n == 0:
            desc   = "classifier head only"
            detail = f"(encoder layers 0–{total_layers - 1} all frozen)"
        elif n >= total_layers:
            desc   = f"all {total_layers} encoder layers + layernorm + classifier head"
            detail = "(full fine-tuning)"
        else:
            first = total_layers - n
            desc   = f"encoder layers {first}–{total_layers - 1} + layernorm + classifier head"
            detail = f"(layers 0–{first - 1} frozen)"

        print(f"[INFO] Trainable: {desc} {detail}")
        print(f"[INFO] Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")


    # =========================================================================
    # 2. FEATURE EXTRACTION: Process audio files into spectrograms.
    # =========================================================================
    def map_labels_to_ids(self):
        """
        Maps string labels to numeric IDs after dataset splitting.
        Also writes id2label / label2id into model.config so the mapping
        is saved with every checkpoint and readable at test / inference time.
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

        # Persist the human-readable names into the model config so they are
        # saved with every checkpoint and available at test / inference time.
        if self.model is not None:
            id2label = {v: k for k, v in self.label_to_id.items()}
            self.model.config.id2label  = id2label
            self.model.config.label2id  = self.label_to_id

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

    def run_training(self, trainer) -> None:
        """
        Runs trainer.train() while suppressing the verbose 'missing keys' warning
        that transformers emits when load_best_model_at_end reloads the best
        checkpoint. That message is a known harmless key-naming artefact — the
        classifier weights load correctly — but it floods the terminal with
        hundreds of layer names.

        All other output (loss, eval metrics, progress bars, real warnings) is
        preserved exactly as before.
        """
        import logging

        class _SuppressMissingKeys(logging.Filter):
            def filter(self, record):
                msg = record.getMessage().lower()
                return "missing keys" not in msg and "unexpected keys" not in msg

        hf_logger = logging.getLogger("transformers.modeling_utils")
        _filter = _SuppressMissingKeys()
        hf_logger.addFilter(_filter)
        try:
            trainer.train()
        finally:
            hf_logger.removeFilter(_filter)

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

        # Detect whether a GPU is available and adjust settings accordingly.
        # adamw_torch_fused is a CUDA-only kernel — on CPU it falls back
        # with overhead or errors.  pin_memory is pointless without a GPU.
        _use_gpu = torch.cuda.is_available()
        _optim = self.config.optim if _use_gpu else "adamw_torch"
        # On CPU, drop batch size to 4 so each step is feasible in minutes,
        # not the ~207 s/step that batch_size=32 produces.
        _batch_size = self.config.batch_size if _use_gpu else min(self.config.batch_size, 4)
        if not _use_gpu:
            print(
                f"[WARN] No GPU detected — switching to CPU-safe settings: "
                f"optim={_optim!r}, batch_size={_batch_size} (was {self.config.batch_size}), "
                f"pin_memory=False"
            )

        # Define training arguments based on the config object
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            save_strategy=self.config.save_strategy,
            eval_strategy=self.config.eval_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            save_total_limit=self.config.save_total_limit,
            optim=_optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=self.config.report_to,
            remove_unused_columns=True,
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            label_names=["labels"],
            dataloader_pin_memory=_use_gpu,
        )

        # Build the data collator — SpecAugment is applied only during training
        # (the collator detects eval/test via torch.is_grad_enabled()).
        if self.config.spec_augment:
            collator = SpecAugmentCollator(
                mask_time_count=self.config.mask_time_count,
                mask_time_ratio=self.config.mask_time_ratio,
                mask_freq_count=self.config.mask_freq_count,
                mask_freq_ratio=self.config.mask_freq_ratio,
            )
            print(
                f"[INFO] SpecAugment enabled — "
                f"time masks: {self.config.mask_time_count} × {self.config.mask_time_ratio:.0%}, "
                f"freq masks: {self.config.mask_freq_count} × {self.config.mask_freq_ratio:.0%}"
            )
        else:
            collator = default_data_collator
            print("[INFO] SpecAugment disabled.")

        # Initialize and return the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
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
        # Use the loaded checkpoint's run folder when available so no new
        # timestamped folder is created for test-only runs.
        _eval_output_dir = (
            str(Path(checkpoint_folder).parent)
            if checkpoint_folder
            else os.path.join(os.getcwd(), "models", "eval_output")
        )
        test_args = TrainingArguments(
            output_dir=_eval_output_dir,
            report_to="none",
        )
        test_trainer = Trainer(
            model=model_to_test,
            args=test_args,
            eval_dataset=self.dataset['test'],
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics,
        )

        import numpy as np

        pred_output = test_trainer.predict(self.dataset['test'])
        metrics     = pred_output.metrics

        # Log and save metrics via the evaluation trainer
        if test_trainer.state.is_world_process_zero:
            test_trainer.log_metrics("test", metrics)
            test_trainer.save_metrics("test", metrics)

        logits      = pred_output.predictions
        labels      = pred_output.label_ids
        predictions = np.argmax(logits, axis=-1)

        # Resolve human-readable class names.
        # Priority: model config (set by map_labels_to_ids and saved in checkpoint)
        # → runtime label_to_id built during this session
        # → fall back to generic "class N" labels.
        cfg_id2label = getattr(getattr(model_to_test, "config", None), "id2label", None) or {}
        runtime_id2label = (
            {v: k for k, v in self.label_to_id.items()}
            if hasattr(self, "label_to_id") and self.label_to_id
            else {}
        )
        # Use config names only when they look real (not "LABEL_N" placeholders).
        if cfg_id2label and not any(v.startswith("LABEL_") for v in cfg_id2label.values()):
            id2label = cfg_id2label
        elif runtime_id2label:
            id2label = runtime_id2label
        else:
            id2label = cfg_id2label

        accuracy        = metrics.get("test_accuracy", metrics.get("eval_accuracy", 0))
        random_baseline = metrics.get("test_random_baseline", metrics.get("eval_random_baseline", 0))
        gain            = metrics.get("test_gain_over_random", metrics.get("eval_gain_over_random", 0))
        loss            = metrics.get("test_loss", metrics.get("eval_loss", 0))

        total   = len(labels)
        correct = int((predictions == labels).sum())

        print("\n[RESULTS] ─────────────────────────────────")
        print(f"  Accuracy:         {accuracy:.2f}%  ({correct} / {total} correct)")
        print(f"  Random baseline:  {random_baseline:.2f}%")
        print(f"  Gain over random: +{gain:.2f} pp")
        print(f"  Loss:             {loss:.4f}")
        print()

        # Per-class breakdown
        class_ids = sorted(set(labels.tolist()))
        col_w = max((len(id2label.get(i, f"class {i}")) for i in class_ids), default=8)
        header = f"  {'Class':<{col_w}}   Total   Correct   Wrong   Accuracy"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for cid in class_ids:
            name      = id2label.get(cid, f"class {cid}")
            mask      = labels == cid
            cls_total = int(mask.sum())
            cls_ok    = int((predictions[mask] == cid).sum())
            cls_bad   = cls_total - cls_ok
            cls_acc   = 100.0 * cls_ok / cls_total if cls_total else 0.0
            print(f"  {name:<{col_w}}   {cls_total:>5}   {cls_ok:>7}   {cls_bad:>5}   {cls_acc:.1f}%")
        print("[RESULTS] ─────────────────────────────────\n")
        return metrics


    # =========================================================================
    # EVALUATE ON TEST (DETAIL): Per-segment inference with confidence scores.
    # =========================================================================
    def evaluate_on_test_detail(self, checkpoint_folder=None, filter_by=None, split="test", aggregate_songs=False):
        """
        Runs inference on any dataset split and prints a per-segment table
        showing true label, predicted label, per-class confidence scores,
        and whether each prediction was correct.

        Parameters
        ----------
        checkpoint_folder : str | None
            Same semantics as evaluate_on_test — None uses the in-memory model.
        split : str
            Which dataset split to run inference on: "test" (default), "eval", or "train".
        filter_by : None | str | int | tuple | list
            Controls which segments within the split to show:
              None          → all segments
              "mysong"      → all segments whose song_id contains "mysong"
              42            → single segment with global index 42
              (10, 30)      → segments 10–29
              [5, 12, 99]   → specific segments by global index
        aggregate_songs : bool
            When True, sums confidence scores across all segments of the same
            song and shows one row per song (late fusion / score aggregation).
            When False (default), shows one row per segment.
        """
        import numpy as np
        import torch

        # ── Load model ────────────────────────────────────────────────────────
        if checkpoint_folder:
            from pathlib import Path
            checkpoint_path = Path(checkpoint_folder)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(os.getcwd()) / checkpoint_folder
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"[ERROR] Checkpoint not found: {checkpoint_path}"
                )
            print(f"[INFO] Loading model from: {checkpoint_path}")
            model_to_test = AutoModelForAudioClassification.from_pretrained(
                checkpoint_path, local_files_only=True,
            )
        else:
            print("[INFO] No checkpoint specified — using the in-memory model.")
            model_to_test = self.model

        # ── Resolve label names ───────────────────────────────────────────────
        cfg_id2label = getattr(getattr(model_to_test, "config", None), "id2label", None) or {}
        runtime_id2label = (
            {v: k for k, v in self.label_to_id.items()}
            if hasattr(self, "label_to_id") and self.label_to_id else {}
        )
        if cfg_id2label and not any(v.startswith("LABEL_") for v in cfg_id2label.values()):
            id2label = cfg_id2label
        elif runtime_id2label:
            id2label = runtime_id2label
        else:
            id2label = cfg_id2label

        # ── Build filtered index list ─────────────────────────────────────────
        # "all" means concatenate all splits.
        if not split or split == "all":
            from datasets import concatenate_datasets
            test_ds  = concatenate_datasets(list(self.dataset.values()))
        elif split not in self.dataset:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.dataset.keys())}")
        else:
            test_ds  = self.dataset[split]
        n_total   = len(test_ds)
        song_ids  = test_ds["song_id"] if "song_id" in test_ds.column_names else [""] * n_total

        if filter_by is None:
            indices = list(range(n_total))
        elif isinstance(filter_by, str):
            indices = [i for i, s in enumerate(song_ids) if filter_by in s]
        elif isinstance(filter_by, int):
            indices = [filter_by]
        elif isinstance(filter_by, tuple) and len(filter_by) == 2:
            indices = list(range(filter_by[0], filter_by[1]))
        elif isinstance(filter_by, list):
            indices = filter_by
        else:
            raise TypeError(f"filter_by must be None, str, int, tuple, or list — got {type(filter_by)}")

        if not indices:
            print("[WARN] No test segments matched the filter.")
            return

        subset_ds = test_ds.select(indices)

        # ── Run inference ─────────────────────────────────────────────────────
        _eval_output_dir = (
            str(Path(checkpoint_folder).parent)
            if checkpoint_folder
            else os.path.join(os.getcwd(), "models", "eval_output")
        )
        test_args = TrainingArguments(
            output_dir=_eval_output_dir,
            report_to="none",
        )
        test_trainer = Trainer(
            model=model_to_test,
            args=test_args,
            eval_dataset=subset_ds,
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics,
        )

        pred_output  = test_trainer.predict(subset_ds)
        logits       = pred_output.predictions          # (N, num_classes)
        true_labels  = pred_output.label_ids            # (N,)
        predictions  = np.argmax(logits, axis=-1)       # (N,)

        # Softmax for confidence scores
        exp_l     = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs     = exp_l / exp_l.sum(axis=-1, keepdims=True)

        # ── Shared formatting helpers ─────────────────────────────────────────
        class_names = [id2label.get(i, f"class_{i}") for i in range(logits.shape[-1])]
        col_w  = max(len(n) for n in class_names)
        name_w = max((len(s) for s in song_ids), default=8)
        conf_headers = "  ".join(f"{n:>{col_w}}" for n in class_names)

        split_label = "all" if not split or split == "all" else split.upper()
        print(f"\n[DETAIL] Split: {split_label}  |  filter: {filter_by!r}  "
              f"|  aggregate: {aggregate_songs}  ({len(indices)} / {n_total} segments)")

        def _print_summary(correct_count, total, true_arr, pred_arr):
            wrong = total - correct_count
            acc   = 100.0 * correct_count / total if total else 0.0
            sep_s = "  " + "─" * 68
            print(sep_s)
            print(f"  {'Total':<20}  {total}")
            print(f"  {'Correct  ✓':<20}  {correct_count}  ({acc:.1f}%)")
            print(f"  {'Wrong    ✗':<20}  {wrong}  ({100 - acc:.1f}%)")
            print()
            cls_head = f"  {'Class':<{col_w}}   Total   Correct   Wrong   Accuracy"
            print(cls_head)
            print("  " + "─" * (len(cls_head) - 2))
            for cid in sorted(set(true_arr.tolist())):
                name    = id2label.get(int(cid), f"class_{cid}")
                mask    = true_arr == cid
                cls_tot = int(mask.sum())
                cls_ok  = int((pred_arr[mask] == cid).sum())
                cls_acc = 100.0 * cls_ok / cls_tot if cls_tot else 0.0
                print(f"  {name:<{col_w}}   {cls_tot:>5}   {cls_ok:>7}   {cls_tot-cls_ok:>5}   {cls_acc:.1f}%")
            print(sep_s + "\n")

        if not aggregate_songs:
            # ── Per-segment table ─────────────────────────────────────────────
            header = (f"\n  {'idx':>4}  {'song_id':<{name_w}}  "
                      f"{'true':<{col_w}}  {'predicted':<{col_w}}  "
                      f"{conf_headers}  ok?")
            sep = "  " + "─" * (len(header) - 2)
            print(header)
            print(sep)

            correct_count = 0
            for global_idx, true_id, pred_id, prob_row in zip(
                indices, true_labels, predictions, probs
            ):
                song   = song_ids[global_idx]
                true_n = id2label.get(int(true_id), str(true_id))
                pred_n = id2label.get(int(pred_id), str(pred_id))
                ok     = "✓" if true_id == pred_id else "✗"
                if true_id == pred_id:
                    correct_count += 1
                conf_str = "  ".join(f"{p:>{col_w}.2%}" for p in prob_row)
                print(f"  {global_idx:>4}  {song:<{name_w}}  "
                      f"{true_n:<{col_w}}  {pred_n:<{col_w}}  "
                      f"{conf_str}  {ok}")

            _print_summary(correct_count, len(indices), true_labels, predictions)

        else:
            # ── Per-song aggregated table (late fusion) ───────────────────────
            from collections import OrderedDict
            songs_data = OrderedDict()
            for global_idx, true_id, prob_row in zip(indices, true_labels, probs):
                base = song_ids[global_idx].rsplit("_seg", 1)[0]
                if base not in songs_data:
                    songs_data[base] = {"first_idx": global_idx,
                                        "true_id": int(true_id),
                                        "probs": []}
                songs_data[base]["probs"].append(prob_row)

            agg_true = []
            agg_pred = []

            header = (f"\n  {'idx':>4}  {'song_id':<{name_w}}  "
                      f"{'true':<{col_w}}  {'predicted':<{col_w}}  "
                      f"{conf_headers}  segs  ok?")
            sep = "  " + "─" * (len(header) - 2)
            print(header)
            print(sep)

            correct_count = 0
            for base, data in songs_data.items():
                summed   = np.sum(data["probs"], axis=0)
                pred_id  = int(np.argmax(summed))
                true_id  = data["true_id"]
                ok       = "✓" if pred_id == true_id else "✗"
                if pred_id == true_id:
                    correct_count += 1
                agg_true.append(true_id)
                agg_pred.append(pred_id)
                true_n   = id2label.get(true_id, str(true_id))
                pred_n   = id2label.get(pred_id, str(pred_id))
                n_segs   = len(data["probs"])
                conf_str = "  ".join(f"{s:>{col_w}.2f}" for s in summed)
                print(f"  {data['first_idx']:>4}  {base:<{name_w}}  "
                      f"{true_n:<{col_w}}  {pred_n:<{col_w}}  "
                      f"{conf_str}  {n_segs:>4}  {ok}")

            _print_summary(correct_count, len(songs_data),
                           np.array(agg_true), np.array(agg_pred))


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
