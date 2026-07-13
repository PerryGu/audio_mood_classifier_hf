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
├── models/                          # Training checkpoints (auto-created per session)
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

**1. Get the project into Colab**

You have two options:

| Option | How | `COLAB_PROJECT_PATH` to set |
|---|---|---|
| Upload directly | Upload the folder via *Files → Upload* or `!unzip` | `"/content/audio_mood_classifier_hf"` |
| Keep it in Drive | Copy the folder to your Google Drive | `"/content/drive/MyDrive/audio_mood_classifier_hf"` |

The project folder must contain `data/mp3_data/` with your labeled MP3 segments. If you already have a `data/processed_dataset/` cache from a previous run, only that folder is needed — the raw `mp3_data/` can be omitted.

**2. If using Google Drive — mount it first**

`drive.mount()` requires the IPython kernel and must be called from a **notebook cell**, not from a script. Run this in a cell before executing `main.py`:

```python
from google.colab import drive
drive.mount("/content/drive")
```

If the project is uploaded directly to `/content/`, skip this step entirely.

**3. Set your project path**

Open `main.py` and update the single line at the top to match where your project lives:

```python
# Direct upload:
COLAB_PROJECT_PATH = "/content/audio_mood_classifier_hf"

# In Google Drive:
COLAB_PROJECT_PATH = "/content/drive/MyDrive/audio_mood_classifier_hf"
```

**4. Add your Hugging Face token to Colab Secrets**

In the Colab left sidebar open the **🔑 Secrets** panel and add a secret named `HF_TOKEN`. The code reads it automatically — no `.env` file needed.

**5. Select a GPU runtime**

Go to **Runtime → Change runtime type** and select a GPU (T4 or better). The optimizer automatically falls back to standard AdamW if no GPU is detected, but training on CPU will be extremely slow.

**6. Run**

```python
!python main.py
```

When `main.py` starts it will automatically:
- Change the working directory to your project folder (so all relative paths resolve)
- Install the few packages not bundled with Colab (`librosa`, `pyloudnorm`, `pydub`, `mutagen`, `python-dotenv`)
- Read your `HF_TOKEN` from Secrets
- Then proceed with the normal pipeline

All output folders (`models/`, `runs/`) are written into your project folder and persist in Drive across sessions.

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
run_model_loading()         ← load MIT AST + ASTFeatureExtractor (freeze base, train head)
prepare_dataset()           ← extract input_values via ASTFeatureExtractor (cached)
split_dataset(test_size=0.3)← group-shuffle split: 70% train / 15% test / 15% eval
map_labels_to_ids()         ← convert string labels → integer IDs
[--debug]  inspect_dataset_samples()
[train]    load_model_from_checkpoint() if resuming
           save_training_info()  → training_info.json
           trainer.train()
           save_session_steps()  → session_log.json (for continuous TensorBoard)
[test]     evaluate_on_test()    → test_performance.txt
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

Each training session writes to its own timestamped folder under `runs/`. The `continuous/` directory accumulates logs from resumed sessions with correct global step offsets so all sessions appear as a single unbroken learning curve. In Colab, both folders live in your Drive and remain available across sessions.

---

## Output Files

Each training session creates a folder under `models/mood_classifier_<YYYY-MM-DD_HH-MM>/` containing:

| File | Description |
|---|---|
| `checkpoint-N/` | Model weights + config saved every epoch |
| `training_info.json` | Run metadata (LR, epochs, resume source, timestamp) |
| `test_performance.txt` | Final test-set evaluation metrics |
| `test_results.json` | Same metrics in JSON format |

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
