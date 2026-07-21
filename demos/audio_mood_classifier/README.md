---
title: Audio Mood Classifier
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
fullWidth: false
license: apache-2.0
short_description: Classify the mood of an uploaded song
suggested_hardware: zero-a10g
startup_duration_timeout: 45m
preload_from_hub:
  - guyPerry/audio-mood-classifier
models:
  - guyPerry/audio-mood-classifier
---

# 🎵 Audio Mood Classifier 🎵

> Can a machine feel the mood of a song?

Upload any song (MP3) and get an instant mood prediction from a fine-tuned deep learning model — no lyrics, no metadata, just the raw audio.

The model is a fine-tuned [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) pre-trained by MIT on AudioSet. Because mood is inherently subjective, it targets broad emotional character rather than precise genre.

**Links:** [Model card](https://huggingface.co/guyPerry/audio-mood-classifier) · [Project on GitHub](https://github.com/PerryGu/audio_mood_classifier_hf)

---

## How this demo works

For each upload, the app:

1. Skips the first and last **30 seconds** of the track (to avoid intros and outros).
2. Extracts **3 evenly-spaced clips**, **10 seconds** each, from the remaining audio.
3. Resamples to **16 kHz** mono and applies **EBU R128 loudness normalization** (−20 LUFS), matching the training pipeline.
4. Classifies each clip independently, then **combines the confidence scores** (late fusion) into one final prediction.

Training used **6 segments per song**; the demo uses **3** for a faster response while keeping the same sampling logic.

---

## Training dataset

The model was trained on a **custom dataset built from scratch** — a hand-curated catalog matched against a personal music library. Each song was manually assigned to one mood category.

| | |
|---|---|
| **Total songs** | 250 |
| **Mood categories** | 3 |
| **Segments per song (training)** | 6 |
| **Segment length** | 10 seconds |
| **Guard buffer** | 30 s skipped at each end of every track |
| **Total training segments** | up to 1,500 (250 songs × 6 segments) |
| **Audio preprocessing** | 16 kHz mono · EBU R128 loudness norm (−20 LUFS) |
| **Train / eval / test split** | ~70% / 15% / 15% (group-aware — all segments from the same song stay in the same split) |

### Songs per category

| Category | Songs in catalog |
|---|---|
| 🌧 `calm_melancholic` | 84 |
| 😐 `moderate_neutral` | 74 |
| ⚡ `energetic_upbeat` | 90 |
| **Total** | **250** |

---

## Mood classes

| Label | Description |
|---|---|
| 🌧 `calm_melancholic` | Slow, introspective, melancholic |
| 😐 `moderate_neutral` | Balanced, mid-energy, neutral feel |
| ⚡ `energetic_upbeat` | Fast, high-energy, upbeat |
