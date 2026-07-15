# Audio Mood Classifier

A fine-tuned [MIT Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) model for 3-class music mood classification, built end-to-end on a custom MP3 dataset and trained with the Hugging Face `Trainer` API.

---

## Overview

Music carries mood — but labeling it automatically is a hard problem. This project builds a complete, end-to-end pipeline to do exactly that: given a short audio clip, the model predicts whether the music feels **calm and melancholic**, **moderate and neutral**, or **energetic and upbeat**.

The goal was to go from a hand-curated list of songs all the way to a trained and evaluated classifier, without relying on any pre-labeled public dataset. Instead, the dataset was constructed from scratch:

1. A personal music library was scanned against a curated song catalog, and each matched track was sampled at **6 evenly-spaced positions** throughout the song. At every position a **10-second clip** was extracted, skipping the first and last 30 seconds of the track to avoid intros and outros. This gives 6 labeled segments per song, each capturing a different moment in the track — maximizing dataset size and variety while keeping each clip representative of the song's overall mood. Each segment also had EBU R128 loudness normalization applied to keep volume levels consistent across clips.
2. Those segments were loaded, resampled to 16 kHz, and converted to spectrograms — 2D frequency-over-time representations of the audio — which were then fed into the model.

**The model** is a fine-tuned [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), developed by MIT and pre-trained on AudioSet. AST applies the standard Vision Transformer architecture directly to audio spectrograms, treating each spectrogram as a "image" and processing it with self-attention across frequency and time. Starting from a model already pre-trained on a large and diverse audio dataset gives a strong foundation — the fine-tuning step only needs to teach it the mood-specific distinctions. To keep training efficient, the entire transformer backbone is frozen and only the final classification head is re-trained for the 3 mood classes.

The pipeline handles the full workflow: data generation, feature extraction, group-aware dataset splitting (ensuring all segments from the same song stay in the same split, to prevent leakage between train/test/eval), training with checkpoint resumption, and evaluation with accuracy reported relative to the random baseline.

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
├── main.py                          # Entry point — runs the full ML pipeline
├── requirements.txt
├── .env                             # HF_TOKEN (local auth)
│
├── data/
│   ├── mp3_data/                    # Raw MP3 segments, organized by mood class
│   │   ├── calm_melancholic/
│   │   ├── energetic_upbeat/
│   │   └── moderate_neutral/
│   └── processed_dataset/           # Cached HF Dataset with extracted input_values
│
├── data_generation/                 # Data pipeline scripts (run once to build dataset)
│   ├── prepare_dataset.py           # Step 1 — match catalog against local music library
│   ├── generate_songs_sagmens.py    # Step 2 — extract MP3 segments from matched tracks
│   └── generate_spectrograms.py    # (Optional) Generate Mel-spectrogram PNGs instead
│
├── docs/
│   ├── songs_catalog.md             # Hand-curated track list by category
│   └── catalog_with_paths.md        # Auto-generated: catalog + matched file paths
│
├── models/
│   ├── ast_pretrained/              # Base pre-trained AST model files (not committed to git)
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   └── model.safetensors
│   └── mood_classifier_<timestamp>/ # Training checkpoints (auto-created per session)
│       ├── checkpoint-N/
│       ├── training_info.json
│       └── test_performance.txt
│
├── runs/                            # TensorBoard logs (auto-created per session)
│   └── continuous/                  # Aggregated multi-session logs for one curve
│
└── src/
    ├── pipeline_manager.py          # Central state manager / pipeline orchestrator
    ├── data_processing/
    │   ├── data_loader.py           # Load MP3s with librosa, build HF Dataset
    │   ├── data_processor.py        # Feature extraction & group-shuffle splitting
    │   └── dataset.py               # AudioDataset class
    ├── training/
    │   └── config.py                # TrainingConfig dataclass (hyperparameters)
    └── utils/
        ├── load_model.py            # AST model & feature extractor initialization
        ├── tests.py                 # Integrity checks, leakage detection, debug tools
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

# Evaluate only (loads from configured checkpoint)
python main.py --mode test

# Enable debug / integrity checks
python main.py --debug
```

---

### Running in Google Colab

The notebook has **3 cells**. Run them in order.

---

**Before you start — one-time Drive setup**

Upload these files to your Google Drive under `MyDrive/audio_mood_classifier_hf/`:

| What | Where on Drive |
|---|---|
| `mp3_data.zip` | `MyDrive/audio_mood_classifier_hf/mp3_data.zip` |
| Base model files (`model.safetensors`, `config.json`, `preprocessor_config.json`) | `MyDrive/audio_mood_classifier_hf/models/ast_pretrained/` |

The base model files are found locally at:
```
C:\Users\<you>\.cache\huggingface\hub\models--MIT--ast-finetuned-audioset-10-10-0.4593\snapshots\<hash>\
```

Also add your HF token: open the **🔑 Secrets** panel in Colab, add a secret named `HF_TOKEN`, and enable **Notebook access**.

Select a **GPU runtime** before running: **Runtime → Change runtime type → T4 GPU**.

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
  2. Drive folder set in `COLAB_DRIVE_MODEL_PATH` (in `main.py`) has weights → copy to `models/ast_pretrained/` → use it
  3. Neither → download from HuggingFace Hub
- Extract MP3 data from Drive if not already present
- Run training and evaluation
- **Back up the checkpoint folder and `runs/` to Google Drive automatically when done**

---

## Training Configuration

Hyperparameters are set at the top of `main.py` and applied to the `TrainingConfig` object:

| Parameter | Default | Description |
|---|---|---|
| `BASE_LEARNING_RATE` | `1e-5` | Learning rate for AdamW |
| `BATCH_SIZE` | `32` | Per-device training batch size |
| `NUM_TRAIN_EPOCHS` | `8` | Total training epochs |
| `OPTIMIZER_NAME` | `adamw_torch_fused` / `adamw_torch` | Fused AdamW on GPU, standard AdamW on CPU (auto-detected) |
| `LOGGING_STEPS` | `50` | TensorBoard log frequency |
| `REPORT_TO` | `tensorboard` | `"tensorboard"` \| `"wandb"` \| `"all"` |

### Checkpoint Resumption

To resume weights from a previous run, set in `main.py`:

```python
RESUME_RUN_FOLDER = "mood_classifier_2026-07-13_13-22"  # folder under models/
RESUME_CKPT_NAME  = "checkpoint-62"                      # checkpoint inside that folder
```

A **new timestamped output folder** is always created regardless of whether weights are resumed. The optimizer always restarts fresh.

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
setup_environment()         ← install missing packages (Colab) + load HF_TOKEN
run_data_loading()          ← load MP3s with librosa → HF Dataset (cached to disk)
run_model_loading()         ← load AST from models/ast_pretrained/ or HF Hub
                               freeze backbone, re-init 3-class head
prepare_dataset()           ← extract input_values via ASTFeatureExtractor (cached)
                               uses 1 process + batch_size=16 in Colab to avoid stalling
split_dataset(test_size=0.3)← group-shuffle split: 70% train / 15% test / 15% eval
map_labels_to_ids()         ← convert string labels → integer IDs
[--debug]  inspect_dataset_samples()
[train]    load_model_from_checkpoint() if resuming
           save_training_info()  → training_info.json
           trainer.train()
           save_session_steps()  → session_log.json (for continuous TensorBoard)
[test]     evaluate_on_test()    → test_performance.txt
           backup_to_drive()     → copy models/<run>/ and runs/ to Drive (Colab only)
```

### Dataset splitting strategy

Splitting is performed with `GroupShuffleSplit` (scikit-learn), grouping by **song name** so all 6 segments of a song always land in the same split. This prevents data leakage between train, test, and eval sets.

### Model architecture

- **Base model:** `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer)
- **Classification head:** re-initialized for `num_labels=3`, trained from scratch
- **Frozen layers:** All transformer backbone layers (`classifier` excluded)
- **Input:** 16 kHz mono audio → `ASTFeatureExtractor` → 2D spectrogram (`input_values`)

### Evaluation metrics

```
accuracy          — % of correct predictions
random_baseline   — 33.33% (1/3 for 3 classes)
gain_over_random  — accuracy − random_baseline (percentage points)
loss              — cross-entropy
```

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
| `checkpoint-N/` | Model weights + config saved every epoch |
| `training_info.json` | Run metadata (LR, epochs, resume source, timestamp) |
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
