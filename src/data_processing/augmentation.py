"""
SpecAugment collator for audio spectrogram training.

Applies random time and frequency masks on-the-fly to training batches.
Evaluation and test batches are passed through unchanged — the collator
detects the no-grad context that the HuggingFace Trainer uses during
eval/predict and skips augmentation automatically.
"""

import torch
from transformers import default_data_collator


class SpecAugmentCollator:
    """
    Data collator that applies SpecAugment on-the-fly to training batches.

    During evaluation / test the Trainer calls the model with
    ``torch.no_grad()``, so ``torch.is_grad_enabled()`` is False.
    We use that flag to skip augmentation automatically — no separate
    collator is needed for eval/test.

    Parameters
    ----------
    mask_time_count : int
        Number of time masks applied per spectrogram.
    mask_time_ratio : float
        Maximum fraction of the time axis each mask may cover.
    mask_freq_count : int
        Number of frequency masks applied per spectrogram.
    mask_freq_ratio : float
        Maximum fraction of the frequency axis each mask may cover.
    """

    def __init__(
        self,
        mask_time_count: int   = 2,
        mask_time_ratio: float = 0.10,
        mask_freq_count: int   = 2,
        mask_freq_ratio: float = 0.10,
    ):
        self.mask_time_count = mask_time_count
        self.mask_time_ratio = mask_time_ratio
        self.mask_freq_count = mask_freq_count
        self.mask_freq_ratio = mask_freq_ratio

    @staticmethod
    def _apply_masks(spec: torch.Tensor, count: int, max_ratio: float, axis: int) -> torch.Tensor:
        """Apply `count` rectangular masks along `axis` (0 = freq, 1 = time)."""
        size = spec.shape[axis]
        max_width = max(1, int(size * max_ratio))
        for _ in range(count):
            width = torch.randint(1, max_width + 1, ()).item()
            start = torch.randint(0, max(1, size - width + 1), ()).item()
            if axis == 0:
                spec[start : start + width, :] = 0.0
            else:
                spec[:, start : start + width] = 0.0
        return spec

    def __call__(self, features: list) -> dict:
        batch = default_data_collator(features)

        # Skip augmentation during eval/test (no_grad context).
        if not torch.is_grad_enabled():
            return batch

        # input_values shape: (B, freq_bins, time_frames)
        inputs = batch["input_values"]
        augmented = []
        for spec in inputs:
            spec = spec.clone()
            spec = self._apply_masks(spec, self.mask_time_count, self.mask_time_ratio, axis=1)
            spec = self._apply_masks(spec, self.mask_freq_count, self.mask_freq_ratio, axis=0)
            augmented.append(spec)
        batch["input_values"] = torch.stack(augmented)
        return batch
