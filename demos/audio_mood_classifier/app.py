"""
app.py — Audio Mood Classifier: Gradio inference app.

Pipeline:
  Stage 1 — Sample the uploaded MP3 at 3 evenly-spaced positions.
  Stage 2 — Extract features via ASTFeatureExtractor.
  Stage 3 — Run model inference → per-segment logits.
  Stage 4 — Aggregate scores across segments (late fusion) → final prediction.
  Stage 5 — Gradio UI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import mutagen
import numpy as np
import pyloudnorm as pyln
import torch
import gradio as gr
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

# ── Configuration ─────────────────────────────────────────────────────────────
# HuggingFace model repository to load (update once the model is uploaded).
MODEL_ID = "your-hf-username/audio-mood-classifier"

# Audio settings — must match the training pipeline exactly.
SAMPLE_RATE      = 16_000   # Hz — required by ASTFeatureExtractor
SEGMENT_DURATION = 10.0     # seconds per clip
NUM_SEGMENTS     = 3        # clips sampled per song
GUARD_BUFFER     = 30.0     # seconds skipped at each end of the track
TARGET_LUFS      = -20.0    # EBU R128 integrated loudness target


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Audio sampling
# ═════════════════════════════════════════════════════════════════════════════

def sample_song(mp3_path: str | Path) -> list[np.ndarray]:
    """
    Load an MP3 file and extract NUM_SEGMENTS evenly-spaced 10-second clips.

    The first GUARD_BUFFER seconds and last GUARD_BUFFER seconds of the track
    are skipped, exactly as the training dataset was constructed.  Each clip is:
      • resampled to SAMPLE_RATE (16 kHz) mono
      • loudness-normalised to TARGET_LUFS (EBU R128 / BS.1770)
    """
    mp3_path = Path(mp3_path)

    meta = mutagen.File(str(mp3_path))
    if meta is None or meta.info is None:
        raise ValueError(f"Could not read audio metadata from '{mp3_path}'.")
    total_secs = float(meta.info.length)

    offsets     = _segment_offsets(total_secs)
    seg_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
    meter       = pyln.Meter(SAMPLE_RATE)
    segments: list[np.ndarray] = []

    for offset in offsets:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(
                str(mp3_path),
                sr=SAMPLE_RATE,
                mono=True,
                offset=offset,
                duration=SEGMENT_DURATION,
            )

        if len(y) < seg_samples:
            y = np.pad(y, (0, seg_samples - len(y)))

        try:
            loudness = meter.integrated_loudness(y.astype(np.float64))
            y = pyln.normalize.loudness(
                y.astype(np.float64), loudness, TARGET_LUFS
            ).astype(np.float32)
        except Exception:
            pass

        y = np.clip(y, -1.0, 1.0)
        segments.append(y)

    return segments


def _segment_offsets(total_secs: float) -> list[float]:
    """Evenly-spaced start positions within [GUARD_BUFFER, total - GUARD_BUFFER]."""
    n         = NUM_SEGMENTS
    seg       = SEGMENT_DURATION
    buf       = GUARD_BUFFER
    threshold = 2.0 * buf + n * seg

    if total_secs >= threshold:
        anchors = np.linspace(buf, total_secs - buf - seg, n)
    else:
        step    = total_secs / n
        anchors = [i * step for i in range(n)]

    return [float(a) for a in anchors]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Feature extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(segments: list[np.ndarray]) -> dict:
    """
    Run ASTFeatureExtractor on all segments and return a batched pt tensor dict.
    The feature extractor is loaded once at module level (see bottom of file).
    """
    inputs = feature_extractor(
        [s.tolist() for s in segments],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    return inputs


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Model inference
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(inputs: dict) -> np.ndarray:
    """
    Forward pass through the AST model.
    Returns raw logits as a numpy array of shape (NUM_SEGMENTS, num_classes).
    """
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (NUM_SEGMENTS, num_classes)

    return logits.cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Score aggregation (late fusion)
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_scores(logits: np.ndarray) -> dict[str, float]:
    """
    Convert logits to probabilities, sum across all segments (late fusion),
    and return a {label: score} dict that Gradio's gr.Label expects.

    Summing probabilities across segments means that classes consistently
    preferred across all 3 clips score highest — this matches the
    detail_aggregate_songs logic used during training evaluation.
    """
    # Softmax per segment
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)   # (N, num_classes)

    # Sum across segments then re-normalise to get a final probability.
    summed = probs.sum(axis=0)                           # (num_classes,)
    summed /= summed.sum()

    id2label = model.config.id2label
    return {id2label[i]: float(summed[i]) for i in range(len(summed))}


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Gradio UI
# ═════════════════════════════════════════════════════════════════════════════

def classify_mood(audio_path: str) -> dict[str, float]:
    """
    Full pipeline: MP3 path → {mood_label: confidence}.
    This is the function Gradio calls on every user upload.
    """
    if audio_path is None:
        return {}

    segments = sample_song(audio_path)
    inputs   = extract_features(segments)
    logits   = run_inference(inputs)
    scores   = aggregate_scores(logits)
    return scores


description = """
Upload a song (MP3) and the model will predict its overall mood.

Three 10-second clips are sampled from different points in the song
(skipping the first and last 30 seconds), each clip is classified
independently, and the confidence scores are combined to give a final answer.
"""

demo = gr.Interface(
    fn=classify_mood,
    inputs=gr.Audio(type="filepath", label="Upload a song (MP3)"),
    outputs=gr.Label(num_top_classes=3, label="Predicted mood"),
    title="🎵 Audio Mood Classifier",
    description=description,
)


# ── Load model and feature extractor once at startup ─────────────────────────
# Loaded here (module level) so they are shared across all Gradio requests
# without being re-initialised on every call.
print(f"Loading model from '{MODEL_ID}' ...")
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
print("Model ready.")


if __name__ == "__main__":
    demo.launch()
