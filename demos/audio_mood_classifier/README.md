---
title: Audio Mood Classifier
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎵 Audio Mood Classifier 🎵

> Can a machine feel the mood of a song?

This small demo lets you upload any song and get an instant mood prediction from a fine-tuned deep learning model. No lyrics, no metadata — just the raw audio.

Under the hood, the model listens to three 10-second snapshots taken from different points in the song (skipping the first and last 30 seconds to avoid intros and outros), classifies each snapshot independently using an Audio Spectrogram Transformer, and combines the results into a single confident answer.

The model is a fine-tuned [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) pre-trained by MIT on AudioSet, adapted to a hand-curated personal music library with each song manually labeled across three mood categories. Because mood is inherently subjective, the model targets broad emotional character rather than precise genre — so a song that *feels* heavy and slow will read as calm/melancholic even if it technically belongs to a different genre.

---

Upload a song (MP3) and the model will predict its mood.

The model samples 3 evenly-spaced 10-second clips from the song (skipping the
first and last 30 seconds), runs each clip through a fine-tuned
[Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593),
and aggregates the confidence scores across all three clips to produce a single
final prediction.

## Mood classes 

| Label | Description |
|---|---|
| 🌧 `calm_melancholic` | Slow, introspective, melancholic |
| 😐 `moderate_neutral` | Balanced, mid-energy, neutral feel |
| ⚡ `energetic_upbeat` | Fast, high-energy, upbeat |
