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
    learning_rate: float = 4e-7 #1e-6 #1e-5
    num_train_epochs: int = 10
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

    # --- Layer Freezing ---
    # Number of encoder layers to unfreeze for training, counting from the top
    # (output end) of the transformer.  The classifier head is always trainable.
    #   0  → classifier head only         (fastest, ~3 K params)   ← current default
    #   1  → layer 11 + layernorm + head  (~7 M params)
    #   2  → layers 10-11 + layernorm + head
    #   12 → all encoder layers + head    (full fine-tune, slowest)
    # When resuming from a checkpoint, set this higher and lower the learning_rate.
    num_unfrozen_layers: int = 3

    # --- Infrastructure Configuration ---
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch_fused"
    report_to: str = "tensorboard"

    # --- SpecAugment ---
    # Applied on-the-fly to training batches only (eval/test are always clean).
    # Set spec_augment=False to disable entirely.
    #
    # mask_time_count   — how many time masks to apply per spectrogram (Google paper uses 2)
    # mask_time_ratio   — each mask covers at most this fraction of the time axis (0.1 = 10%)
    # mask_freq_count   — how many frequency masks to apply per spectrogram
    # mask_freq_ratio   — each mask covers at most this fraction of the frequency axis
    #
    # Conservative starting point (tune up if val/test gap keeps widening):
    spec_augment: bool  = True
    mask_time_count: int   = 2
    mask_time_ratio: float = 0.10
    mask_freq_count: int   = 2
    mask_freq_ratio: float = 0.10

    # --- Run Behaviour ---
    # Set debug=True to run integrity checks (inspect samples, leakage tests, etc.).
    debug: bool = False
    # Execution mode:
    #   "train"       → training only
    #   "test"        → evaluate on test set (summary + per-class table)
    #   "both"        → train then evaluate on test set
    #   "test_detail" → per-segment inference on test set with confidence scores
    mode: str = "both" #"train" / "test" / "both" / "test_detail"

    # --- Test Detail Mode ---
    # Used when mode="test_detail". Runs per-segment inference with confidence scores.
    # The checkpoint to load is still controlled by parent_run_folder +
    # resume_checkpoint_name, exactly like mode="test".
    #
    # detail_split_test: which dataset split to run inference on.
    #   "test"   → test set (default)
    #   "eval"   → validation set
    #   "train"  → training set
    #   "all"    → all splits combined (train + eval + test)
    #
    # detail_filter_test: which segments within that split to show.
    #   None           → all segments (default)
    #   "mysong"       → all segments whose song_id contains "mysong"
    #   42             → single segment with global index 42
    #   (10, 30)       → segments 10–29
    #   [5, 12, 99]    → specific segments by global index
    detail_split_test: str     = "test"
    detail_filter_test: object = None
    # detail_aggregate_songs: when True, sums the confidence scores of all
    # segments belonging to the same song and shows one row per song instead
    # of one row per segment. The predicted label is the class with the highest
    # total score across all segments (late fusion / score aggregation).
    #   False → one row per segment (default)
    #   True  → one row per song, scores summed across segments
    detail_aggregate_songs: bool = True

    # --- Segment Inspection ---
    # Set inspect_segments=True to print a flat table of segments after the
    # dataset is prepared (no inference — read-only view of the data).
    #
    # inspect_segments_split: which split(s) to include in the table.
    #   "all"   → all splits combined: train + eval + test (default)
    #   "train" → training set only
    #   "eval"  → validation set only
    #   "test"  → test set only
    #
    # inspect_segment_ids: which rows to show (by global index across the chosen split(s)).
    #   None          → all segments (default)
    #   42            → only segment 42
    #   (10, 30)      → segments 10–29
    #   [5, 12, 99]   → specific segments by global ID
    inspect_segments: bool       = False
    inspect_segments_split: str  = "test"
    inspect_segment_ids: object  = None

    # --- WandB ---
    wandb_project: str = "audio-mood-classifier"

    # Steps completed in previous sessions of the same lineage (set automatically).
    step_offset: int = 0

    # --- Checkpoint Resumption ---
    # Leave both empty for a clean new run (the default).
    # Fill both to load weights from a previous checkpoint before training.
    # A fresh timestamped output folder is always created regardless.
    parent_run_folder: str = "mood_classifier_2026-07-16_09-27"      # e.g. "mood_classifier_2026-07-12_20-20"
    resume_checkpoint_name: str = "checkpoint-372" # e.g. "checkpoint-248"

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
