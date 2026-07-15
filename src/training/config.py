from dataclasses import dataclass, field
from datetime import datetime
import os

@dataclass
class TrainingConfig:
    # --- Model & Data Configuration ---
    model_ckpt: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    max_length: int = 512
    batch_size: int = 32
    num_labels: int = 3

    # --- Training Hyperparameters ---
    learning_rate: float = 2e-6 #1e-5
    num_train_epochs: int = 8
    logging_steps: int = 50

    # --- Strategy Configuration ---
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    # Maximum number of checkpoints to keep on disk at any time.
    # The Trainer always preserves the best checkpoint (tracked via
    # trainer_state.json) and fills remaining slots with the most recent
    # saves, deleting the rest.  2 = keep best + latest.
    save_total_limit: int = 3

    # --- Infrastructure Configuration ---
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch_fused"
    report_to: str = "tensorboard"

    # --- Run Behaviour ---
    # Set debug=True to run integrity checks (inspect samples, leakage tests, etc.).
    debug: bool = False
    # Execution mode: "train" runs training only, "test" runs evaluation only,
    # "both" (default) runs training then evaluates on the test set.
    mode: str = "both" #"train" / "test" / "both"

    # --- WandB ---
    wandb_project: str = "audio-mood-classifier"

    # Steps completed in previous sessions of the same lineage (set automatically).
    step_offset: int = 0

    # --- Checkpoint Resumption ---
    # Leave both empty for a clean new run (the default).
    # Fill both to load weights from a previous checkpoint before training.
    # A fresh timestamped output folder is always created regardless.
    parent_run_folder: str = "mood_classifier_2026-07-15_14-39"      # e.g. "mood_classifier_2026-07-12_20-20"
    resume_checkpoint_name: str = "checkpoint-496" # e.g. "checkpoint-248"

    # --- Run Folder ---
    # A fresh timestamped folder is ALWAYS created for each training session.
    # Set RESUME_RUN_FOLDER / RESUME_CKPT_NAME in main.py only to load weights —
    # they do NOT change where the new checkpoints are saved.
    run_name: str = field(default_factory=lambda: f"mood_classifier_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")

    def __post_init__(self):
        # Captured once so logging_dir stays stable across multiple property accesses.
        self._session_timestamp = datetime.now().strftime('%H-%M')

    @property
    def output_dir(self) -> str:
        """New timestamped folder created for every training session, inside models/."""
        return os.path.abspath(os.path.join(os.getcwd(), "models", self.run_name))

    @property
    def session_name(self) -> str:
        """Unique label for this specific training session."""
        return f"{self.run_name}_{self._session_timestamp}"

    @property
    def logging_dir(self) -> str:
        """Per-session TensorBoard log folder (separate line per session)."""
        return os.path.abspath(os.path.join(os.getcwd(), "runs", self.session_name))

    # --- Checkpoint Resumption ---
    # Leave both empty for a clean new run (the default).
    # Fill both to load weights from a previous checkpoint before training.
    # A fresh timestamped output folder is always created regardless.
    @property
    def checkpoint_to_load(self) -> str:
        """Full path to the checkpoint to load weights from. Empty if starting fresh."""
        if self.parent_run_folder and self.resume_checkpoint_name:
            return os.path.abspath(
                os.path.join(os.getcwd(), "models", self.parent_run_folder, self.resume_checkpoint_name)
            )
        return ""

    @property
    def continuous_logging_dir(self) -> str:
        """
        Single directory that accumulates all sessions in the same training
        lineage with correct step offsets, giving one continuous TensorBoard curve.

        Resumed sessions group under the parent run's name so the full history
        is visible in one place.

        View continuous progress:  tensorboard --logdir runs/continuous
        """
        lineage = self.parent_run_folder if self.parent_run_folder else self.run_name
        return os.path.abspath(os.path.join(os.getcwd(), "runs", "continuous", lineage))
