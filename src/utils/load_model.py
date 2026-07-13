"""
This script handles the initial loading of the MIT AST model.
It clears memory, configures the device, and initializes 
the feature extractor and model for classification.
"""

import gc
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification, logging


def load_ast_model(model_ckpt: str, device: str, num_labels: int):
    print(f"[INFO] Loading feature extractor for: {model_ckpt}")
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_ckpt)

    print(f"[INFO] Loading AST model...")
    # Apply num_labels and ignore_mismatched_sizes for custom classification head
    model = ASTForAudioClassification.from_pretrained(
        model_ckpt, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    model.to(device)
    model.eval()

    return feature_extractor, model