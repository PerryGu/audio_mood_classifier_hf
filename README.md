# Audio Mood Classifier

A fine-tuned [MIT Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) model for 3-class music mood classification, built end-to-end on a custom MP3 dataset and trained with the Hugging Face `Trainer` API.

---

## Overview

Music carries mood — but labeling it automatically is a hard problem. This project builds a complete, end-to-end pipeline to do exactly that: given a short audio clip, the model predicts whether the music feels **calm and melancholic**, **moderate and neutral**, or **energetic and upbeat**.

The goal was to go from a hand-curated list of songs all the way to a trained and evaluated classifier, without relying on any pre-labeled public dataset. Instead, the dataset was constructed from scratch:

1. A personal music library was scanned against a curated song catalog, and each matched track was sampled at **6 evenly-spaced positions** throughout the song. At every position a **10-second clip** was extracted, skipping the first and last 30 seconds of the track to avoid intros and outros. This gives 6 labeled segments per song, each capturing a different moment in the track — maximizing dataset size and variety while keeping each clip representative of the song's overall mood. Each segment also had EBU R128 loudness normalization applied to keep volume levels consistent across clips.
2. Those segments were loaded, resampled to 16 kHz, and converted to spectrograms — 2D frequency-over-time representations of the audio — which were then fed into the model.

**The model** is a fine-tuned [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), developed by MIT and pre-trained on AudioSet. AST applies the standard Vision Transformer architecture directly to audio spectrograms, treating each spectrogram as an "image" and processing it with self-attention across frequency and time. Starting from a model already pre-trained on a large and diverse audio dataset gives a strong foundation — the fine-tuning step only needs to teach it the mood-specific distinctions. By default only the final classification head is re-trained for the 3 mood classes, but `num_unfrozen_layers` in `config.py` lets you progressively unfreeze the top encoder layers for deeper fine-tuning.

The pipeline handles the full workflow: data generation, feature extraction, group-aware dataset splitting (ensuring all segments from the same song stay in the same split, to prevent leakage between train/test/eval), training with checkpoint resumption, and evaluation with accuracy reported relative to the random baseline.

The project runs in two environments — **locally** (with a GPU or CPU) and in **Google Colab** (free T4 GPU). Checkpoints are saved to Google Drive and can be copied back locally to continue training, keeping both environments in sync via a simple `git push` / `git pull`.

---

## Mood Classes

| Label | Description |
|---|---|
| `calm_melancholic` | Slow, introspective, melancholic |
| `moderate_neutral` | Balanced, mid-energy, neutral feel |
| `energetic_upbeat` | Fast, high-energy, upbeat |

---

## Project Structure

```
audio_mood_classifier_hf/
├── main.py                              # Entry point — runs the full ML pipeline
├── requirements.txt
├── .env                                 # HF_TOKEN (local auth)
├── audio_mood_classifier_hf.ipynb       # Google Colab notebook (3-cell setup)
│
├── demos/                               # Gradio / HF Spaces deployment
│   ├── upload_space.py                  # One-time script to publish the Space to HF
│   └── audio_mood_classifier/           # The Space itself (uploaded as-is)
│       ├── app.py                       # Gradio inference app
│       ├── requirements.txt             # Space dependencies
│       └── README.md                    # HF Spaces metadata + description
│
├── data/
│   ├── mp3_data/                        # Raw MP3 segments, organized by mood class
│   │   ├── calm_melancholic/
│   │   ├── energetic_upbeat/
│   │   └── moderate_neutral/
│   └── processed_dataset/               # Cached HF Dataset with extracted input_values
│
├── data_generation/                     # Data pipeline scripts (run once to build dataset)
│   ├── prepare_dataset.py               # Step 1 — match catalog against local music library
│   ├── generate_songs_sagmens.py        # Step 2 — extract MP3 segments from matched tracks
│   └── generate_spectrograms.py         # (Optional) Generate Mel-spectrogram PNGs instead
│
├── docs/
│   ├── songs_catalog.md                 # Hand-curated track list by category
│   └── catalog_with_paths.md            # Auto-generated: catalog + matched file paths
│
├── models/
│   ├── ast_pretrained/                  # Base pre-trained AST model files (not committed to git)
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   └── model.safetensors
│   └── mood_classifier_<timestamp>/     # Training checkpoints (auto-created per session)
│       ├── checkpoint-N/
│       ├── training_info.json
│       └── test_performance.txt
│
├── runs/                                # TensorBoard logs (auto-created per session)
│   └── continuous/                      # Aggregated multi-session logs for one curve
│
└── src/
    ├── pipeline_manager.py              # Central state manager / pipeline orchestrator
    ├── config.py                        # TrainingConfig dataclass (all hyperparameters & flags)
    ├── data_processing/
    │   ├── data_loader.py               # Load MP3s with librosa, build HF Dataset
    │   ├── data_processor.py            # Feature extraction & group-shuffle splitting
    │   ├── dataset.py                   # AudioDataset class
    │   └── augmentation.py              # SpecAugmentCollator — on-the-fly training augmentation
    └── utils/
        ├── load_model.py                # AST model & feature extractor initialization
        ├── tests.py                     # Integrity checks, leakage detection, debug tools
        ├── get_model_params.py
        └── data_uploader.py
```

### Google Drive folder structure

The Drive folder mirrors the local project structure exactly:

```
MyDrive/audio_mood_classifier_hf/
├── models/
│   ├── ast_pretrained/              # Base AST model files (upload once, reused every session)
│   └── mood_classifier_<timestamp>/ # Auto-backed-up after each training run
├── runs/                            # Auto-backed-up TensorBoard logs
└── mp3_data.zip                     # Zipped MP3 dataset (extracted once on first run)
```

> The Colab notebook (`audio_mood_classifier_hf.ipynb`) clones the repo from GitHub, mounts Drive, and extracts the MP3 data automatically — you never need to manually copy code files to Drive.

---

## Quick Start

### Running locally

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

> Requires `ffmpeg` on `PATH` for MP3 encoding (used by `pydub` during data generation only — not needed for training).

**2. Set up credentials**

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

**3. Run the pipeline**

```bash
# Train and evaluate (default)
python main.py

# Train only
python main.py --mode train

# Evaluate only — summary + per-class table (loads from configured checkpoint)
python main.py --mode test

# Per-segment detail evaluation with confidence scores
python main.py --mode test_detail

# Enable debug / integrity checks
python main.py --debug
```

---

### Running in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PerryGu/audio_mood_classifier_hf/blob/main/audio_mood_classifier_hf.ipynb)

Open the notebook directly in Colab using the badge above, or find it at [`audio_mood_classifier_hf.ipynb`](audio_mood_classifier_hf.ipynb) in the project root.

The notebook has **3 cells**. Run them in order.

---

**Before you start — one-time Drive setup**

Upload these files to your Google Drive under `MyDrive/audio_mood_classifier_hf/`:

| What | Where on Drive |
|---|---|
| `mp3_data.zip` | `MyDrive/audio_mood_classifier_hf/mp3_data.zip` |
| Base model files (`model.safetensors`, `config.json`, `preprocessor_config.json`) | `MyDrive/audio_mood_classifier_hf/models/ast_pretrained/` |

> After the first successful local run, `models/ast_pretrained/` is populated automatically. You can copy it from there to Drive instead of hunting for the HF cache folder.

Also add your HF token: open the **🔑 Secrets** panel in Colab, add a secret named `HF_TOKEN`, and enable **Notebook access**.

Select a **GPU runtime** before running: **Runtime → Change runtime type → T4 GPU**.

**Colab path variables** (top of `main.py`)

| Variable | Purpose |
|---|---|
| `COLAB_PROJECT_PATH` | Absolute path to the cloned repo in Colab |
| `COLAB_MP3_ZIP_PATH` | Drive path to `mp3_data.zip` |
| `COLAB_DRIVE_MODEL_PATH` | Drive path to `models/ast_pretrained/` (copied locally on first run) |
| `COLAB_DRIVE_MODELS_BASE` | Drive path to your `models/` folder — set once and never touch again; the checkpoint path is auto-derived from this base + `config.parent_run_folder` + `config.resume_checkpoint_name` |

---

**Cell 1 — Project setup** (clone repo, mount Drive, extract MP3 data)

```python
from google.colab import userdata, drive
import zipfile, os
from pathlib import Path

# 1. Clone the repository
git_token = userdata.get("GIT_TOKEN")
repo_url = f"https://{git_token}@github.com/PerryGu/audio_mood_classifier_hf.git"
os.system("rm -rf audio_mood_classifier_hf")
os.system(f"git clone {repo_url}")
os.chdir("audio_mood_classifier_hf")

# 2. Mount Google Drive
drive.mount("/content/drive")

# 3. Extract MP3 data if not already present
ZIP_PATH   = "/content/drive/MyDrive/audio_mood_classifier_hf/mp3_data.zip"
EXTRACT_TO = "data/mp3_data"

if os.path.exists("data/processed_dataset") and os.listdir("data/processed_dataset"):
    print("Processed dataset already in repo — no MP3 extraction needed.")
else:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        top_level = {Path(n).parts[0] for n in zf.namelist() if n.strip("/")}
        if len(top_level) == 1:
            zf.extractall("data")
            extracted = Path("data") / next(iter(top_level))
            if extracted.resolve() != Path(EXTRACT_TO).resolve():
                extracted.rename(EXTRACT_TO)
        else:
            os.makedirs(EXTRACT_TO, exist_ok=True)
            zf.extractall(EXTRACT_TO)
    mp3_count = len(list(Path(EXTRACT_TO).rglob("*.mp3")))
    print(f"Extraction complete — {mp3_count} MP3 files ready.")
```

---

**Cell 2 — Sync latest code** (run after any code update)

```python
!git pull origin main
```

---

**Cell 3 — Run the pipeline**

```python
from google.colab import userdata
import os
from huggingface_hub import login

token = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = token
login(token=token, add_to_git_credential=False)
print("Authenticated ✓")

!python main.py
```

When `main.py` runs it will automatically:
- Set the working directory so all relative paths resolve correctly
- Install any missing packages (`librosa`, `pyloudnorm`, `pydub`, `mutagen`, `python-dotenv`)
- Read `HF_TOKEN` from the environment
- Resolve the base model source in this priority order:
  1. `models/ast_pretrained/` already present locally → use it directly
  2. Drive folder set in `COLAB_DRIVE_MODEL_PATH` has weights → copy to `models/ast_pretrained/` → use it
  3. Neither → download from HuggingFace Hub → save to `models/ast_pretrained/` for future runs
- Extract MP3 data from Drive if not already present
- If resuming from a checkpoint and `COLAB_DRIVE_CHECKPOINT_PATH` is set → copy checkpoint from Drive to `models/` before training
- Run training (with terminal noise from `load_best_model_at_end` suppressed) and evaluation
- **Back up the checkpoint folder and `runs/` to Google Drive automatically when done**

---

## Training Configuration

All hyperparameters and run-behaviour flags live in **`src/config.py`** — the single source of truth. Edit them there; `main.py` applies them automatically. CLI flags (`--debug`, `--mode`) override config values for one-off runs without touching the file.

### Hyperparameters

| Field | Default | Description |
|---|---|---|
| `learning_rate` | `1e-5` | AdamW learning rate |
| `batch_size` | `32` | Per-device training batch size |
| `num_train_epochs` | `5` | Total training epochs |
| `logging_steps` | `50` | TensorBoard log frequency |
| `optim` | `adamw_torch_fused` | Optimizer — auto-switched to `adamw_torch` on CPU |
| `report_to` | `tensorboard` | `"tensorboard"` \| `"wandb"` \| `"all"` |
| `wandb_project` | `audio-mood-classifier` | WandB project name |

### CPU / GPU auto-detection

`get_trainer()` detects `torch.cuda.is_available()` at runtime and automatically applies safe CPU fallbacks when no GPU is found:

| Setting | GPU | CPU (auto) |
|---|---|---|
| `optim` | `adamw_torch_fused` | `adamw_torch` |
| `batch_size` | `config.batch_size` (32) | `min(config.batch_size, 4)` |
| `dataloader_pin_memory` | `True` | `False` |

### Layer freezing

Controlled by `num_unfrozen_layers` in `config.py`. The classifier head is always trainable; this setting additionally unfreezes encoder layers counting from the top (output) end of the 12-layer transformer.

| `num_unfrozen_layers` | Trainable scope | Approx params | Recommended LR |
|---|---|---|---|
| `0` | Classifier head only *(default)* | ~3 K | `1e-5` |
| `1` | Layer 11 + layernorm + head | ~7 M | `2e-6` |
| `2` | Layers 10–11 + layernorm + head | ~14 M | `1e-6` |
| `4` | Layers 8–11 + layernorm + head | ~28 M | `5e-7` |
| `12` | All encoder layers + head | ~87 M | `1e-7` |

> When unfreezing layers on top of a checkpoint, always lower `learning_rate` significantly — too high a value will destroy the pre-trained representations rather than adapting them.

### SpecAugment (data augmentation)

Applied **on-the-fly to training batches only** — eval and test batches are always clean. Implemented in `src/data_processing/augmentation.py` as `SpecAugmentCollator`, which randomly zeros rectangular strips on the time and frequency axes of each spectrogram before it reaches the model.

| Field | Default | Description |
|---|---|---|
| `spec_augment` | `True` | Enable / disable SpecAugment |
| `mask_time_count` | `2` | Number of time masks applied per spectrogram |
| `mask_time_ratio` | `0.10` | Maximum fraction of the time axis each mask may cover |
| `mask_freq_count` | `2` | Number of frequency masks applied per spectrogram |
| `mask_freq_ratio` | `0.10` | Maximum fraction of the frequency axis each mask may cover |

> The collator auto-detects training vs. eval mode via `torch.is_grad_enabled()` — no separate collator is needed for evaluation.

### Run behaviour

| Field | Default | Description |
|---|---|---|
| `mode` | `"both"` | `"train"` / `"test"` / `"both"` / `"test_detail"` — overridable with `--mode` |
| `debug` | `False` | Enable integrity checks — overridable with `--debug` |

### Segment inspection (read-only dataset view)

Prints a flat table of every segment after the dataset is prepared — no inference, just a view of what's in each split. Useful for auditing labels and checking which songs landed in which split.

| Field | Default | Description |
|---|---|---|
| `inspect_segments` | `False` | `True` to enable the table |
| `inspect_segments_split` | `"all"` | `"all"` / `"train"` / `"eval"` / `"test"` — which split(s) to show |
| `inspect_segment_ids` | `None` | `None` = all rows; `42` = single row; `(10,30)` = range; `[5,12,99]` = specific |

### Test detail mode

Runs per-segment inference and prints a table showing true label, predicted label, per-class confidence scores, and ✓/✗ for each row. Activated by `mode="test_detail"`.

| Field | Default | Description |
|---|---|---|
| `detail_split_test` | `"test"` | Which split to run inference on: `"test"` / `"eval"` / `"train"` / `"all"` |
| `detail_filter_test` | `None` | `None` = all; `"songname"` = filter by song; `42` / `(10,30)` / `[5,12]` = by index |
| `detail_aggregate_songs` | `False` | When `True`, sums confidence scores across all segments of the same song and shows **one row per song** (late fusion / score aggregation) instead of one row per segment |

**Song-level aggregation** (`detail_aggregate_songs=True`) typically gives 2–4 pp higher accuracy than segment-level evaluation because noise from individual ambiguous segments averages out across the 6 segments of each song. This is also the strategy recommended for the Gradio inference app.

### Checkpoint management

| Field | Default | Description |
|---|---|---|
| `save_total_limit` | `2` | Max checkpoints kept on disk at any time |
| `load_best_model_at_end` | `True` | Reload best checkpoint into memory after training |
| `metric_for_best_model` | `"loss"` | Metric used to rank checkpoints |

With `save_total_limit=2` the Trainer always preserves the **best** checkpoint (tracked in `trainer_state.json` → `best_model_checkpoint`) and fills the remaining slot with the most recent save; older non-best checkpoints are deleted automatically.

To find the best checkpoint after a run, read `models/<run>/checkpoint-N/trainer_state.json`:

```json
{
  "best_model_checkpoint": ".../checkpoint-496",
  "best_metric": 0.7278
}
```

### Checkpoint resumption

Set these fields in `config.py` to load weights from a previous checkpoint before starting a new training session. A **new timestamped output folder** is always created; the optimizer restarts fresh.

```python
# src/training/config.py
parent_run_folder      = "mood_classifier_2026-07-15_14-39"  # folder under models/
resume_checkpoint_name = "checkpoint-496"                     # checkpoint inside that folder
```

In **Colab**, `COLAB_DRIVE_MODELS_BASE` at the top of `main.py` is the only variable you need — set it once and never touch it again. The pipeline automatically derives the full checkpoint path from it using `config.parent_run_folder` and `config.resume_checkpoint_name`:

```python
# main.py  (Colab config section — set once)
COLAB_DRIVE_MODELS_BASE = "/content/drive/MyDrive/audio_mood_classifier_hf/models"
```

From then on, to resume from a different checkpoint you only update `config.py`:

```python
# src/config.py  (the only file you edit between runs)
parent_run_folder      = "mood_classifier_2026-07-15_14-39"
resume_checkpoint_name = "checkpoint-496"
```

Leave `parent_run_folder` and `resume_checkpoint_name` both empty for a clean new run (no resumption from Drive).

---

## Data Generation Pipeline

Run these scripts **once** to build the MP3 dataset from a local music library.

### Step 1 — Match catalog to library

```bash
python data_generation/prepare_dataset.py
```

- Reads `docs/songs_catalog.md` (hand-curated track list).
- Walks your local music library (`MUSIC_LIBRARY_PATH` in the script).
- Writes `docs/catalog_with_paths.md` with matched audio file paths and similarity scores.

> Set `MUSIC_LIBRARY_PATH` inside `prepare_dataset.py` before running.

### Step 2 — Extract MP3 segments

```bash
python data_generation/generate_songs_sagmens.py
```

- Reads `docs/catalog_with_paths.md`.
- Runs an interactive verification step — review and confirm the track list before committing.
- Extracts **6 × 10-second segments** per track, evenly distributed across the interior of the track (with a 30-second guard buffer at each end).
- Applies **EBU R128 loudness normalization** (`-20 LUFS`) to every segment.
- Saves segments as `<TrackName>_seg1.mp3 … _seg6.mp3` under `data/mp3_data/<category>/`.

### (Optional) Step 2b — Generate Mel-spectrogram PNGs

```bash
python data_generation/generate_spectrograms.py
```

Produces `128-bin dB-scaled Mel-spectrogram PNGs` (400×400 px, magma colormap) alongside the MP3s. Not required for the main training pipeline.

---

## ML Pipeline (detailed)

`main.py` drives a single `PipelineManager` instance through these ordered stages:

```
setup_environment()              ← install missing packages (Colab) + load HF_TOKEN
                                    HF Hub login is non-fatal — a 504 timeout is caught and
                                    logged as a warning; local checkpoints still work fine
run_data_loading()               ← load MP3s with librosa → HF Dataset (cached to disk)
run_model_loading()              ← resolve model source (priority order below), load AST,
                                    freeze/unfreeze layers per num_unfrozen_layers,
                                    re-init classification head for num_labels classes;
                                    if downloaded from HF Hub → save to models/ast_pretrained/
                                    so future runs load locally without re-downloading;
                                    id2label / label2id saved into model config so checkpoints
                                    carry human-readable class names (not LABEL_0, LABEL_1 …)
prepare_dataset()                ← extract input_values via ASTFeatureExtractor (cached)
                                    uses 1 process + batch_size=16 in Colab to avoid stalling
split_dataset(test_size=0.3)    ← group-shuffle split: 70% train / 15% test / 15% eval
map_labels_to_ids()              ← convert string labels → integer IDs

[inspect_segments=True]
           inspect_segment_table()        read-only table of every segment: ID, song, label,
                                          split — filtered by inspect_segment_ids /
                                          inspect_segments_split; grouped by split then label

[train]    _ensure_resume_checkpoint()    if resuming (copies from Drive in Colab)
           load_model_from_checkpoint()   if resuming (loads weights, re-applies freeze)
           save_training_info()           → training_info.json
           run_training(trainer)          → trainer.train() with terminal noise suppressed;
                                          SpecAugmentCollator applied to training batches
                                          when spec_augment=True (eval/test always clean)
           save_session_steps()           → session_log.json (for continuous TensorBoard)

[test]     evaluate_on_test()            → summary + per-class breakdown table:
                                            correct / wrong counts and % for each class
                                          → test_performance.txt

[test_detail]
           evaluate_on_test_detail()     → per-segment inference table with true label,
                                            predicted label, per-class confidence scores,
                                            and ✓/✗; filtered by detail_filter_test /
                                            detail_split_test
           (detail_aggregate_songs=True) → sum confidence scores across all segments of
                                            the same song → one prediction per song
                                            (late fusion; typically 2–4 pp better than
                                            per-segment accuracy)

[colab]    backup_to_drive()             → copy models/<run>/ and runs/ to Drive
```

### Dataset splitting strategy

Splitting is performed with `GroupShuffleSplit` (scikit-learn), grouping by **song name** so all 6 segments of a song always land in the same split. This prevents data leakage between train, test, and eval sets.

### Model architecture

- **Base model:** `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer, 12 encoder layers)
- **Classification head:** re-initialized for `num_labels=3`, always trainable
- **Frozen layers:** controlled by `num_unfrozen_layers` in `config.py` (default `0` = full backbone frozen, only head trains; set higher to unfreeze top encoder layers for deeper fine-tuning)
- **Input:** 16 kHz mono audio → `ASTFeatureExtractor` → 2D spectrogram (`input_values`)
- **Model source priority:** `models/ast_pretrained/` (local) → Drive copy (Colab) → HF Hub download (cached to `models/ast_pretrained/` for future runs)

### Evaluation metrics

```
accuracy          — % of correct predictions
random_baseline   — 33.33% (1/3 for 3 classes)
gain_over_random  — accuracy − random_baseline (percentage points)
loss              — cross-entropy
```

`evaluate_on_test()` (mode `"test"`) additionally prints a **per-class breakdown table** showing the number and percentage of correct vs. incorrect segments for each mood class — useful for diagnosing which category the model struggles with the most.

`evaluate_on_test_detail()` (mode `"test_detail"`) provides a **per-segment inference table** with one row per segment (or per song when `detail_aggregate_songs=True`):

| Column | Description |
|---|---|
| `id` | Index of the segment in the dataset |
| `song` | Song name (from `song_id`) |
| `split` | Which split the segment belongs to |
| `true` | Ground-truth label |
| `pred` | Model prediction |
| `✓/✗` | Correct / incorrect |
| `conf_<class>` | Softmax confidence for each class |

At the bottom of the table a summary line shows total ✓ / ✗ counts and the accuracy percentage.

When `detail_aggregate_songs=True` the confidence columns are **summed** across all segments of the same song before taking `argmax`. This late-fusion approach is significantly more robust than per-segment prediction for short clips.

---

## TensorBoard

**Locally:**

```bash
# Current session only
tensorboard --logdir runs

# Full continuous history (all sessions in a lineage)
tensorboard --logdir runs/continuous
```

**In Colab:**

```python
%load_ext tensorboard

# Current session only
%tensorboard --logdir runs

# Full continuous history
%tensorboard --logdir runs/continuous
```

Each training session writes to its own timestamped folder under `runs/`. The `continuous/` directory accumulates logs from resumed sessions with correct global step offsets so all sessions appear as a single unbroken learning curve. In Colab, both folders are automatically backed up to Drive after each training run.

---

## Output Files

Each training session creates a folder under `models/mood_classifier_<YYYY-MM-DD_HH-MM>/` containing:

| File | Description |
|---|---|
| `checkpoint-N/` | Saved model weights + optimizer state. At most `save_total_limit` checkpoints are kept — the Trainer always preserves the best one and fills remaining slots with the most recent saves. |
| `checkpoint-N/trainer_state.json` | Full per-epoch metric history + `best_model_checkpoint` field identifying the winning checkpoint by name. |
| `training_info.json` | Run metadata (LR, epochs, resume source, timestamp) |
| `all_results.json` | Combined train + eval metrics for the session |
| `test_performance.txt` | Final test-set evaluation metrics |
| `test_results.json` | Same metrics in JSON format |

### Google Drive backup (Colab only)

After training completes, `backup_to_drive()` runs automatically and copies:

| Source (Colab) | Destination (Drive) |
|---|---|
| `models/mood_classifier_<timestamp>/` | `MyDrive/audio_mood_classifier_hf/models/mood_classifier_<timestamp>/` |
| `runs/` | `MyDrive/audio_mood_classifier_hf/runs/` |

On a local machine this step is skipped entirely.

---

## Dependencies

| Package | Version |
|---|---|
| `torch` | 2.5.1+cu121 |
| `transformers` | 5.13.1 |
| `datasets` | 5.0.0 |
| `librosa` | 0.11.0 |
| `scikit_learn` | 1.9.0 |
| `pyloudnorm` | 0.2.0 |
| `pydub` | 0.25.1 |
| `mutagen` | 1.48.1 |
| `python-dotenv` | 1.2.2 |

Full pinned list: [`requirements.txt`](requirements.txt)
