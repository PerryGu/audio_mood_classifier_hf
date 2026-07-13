"""
## Model Parameter Diagnostic

This helper function provides a breakdown of the model's parameters.
It differentiates between trainable parameters (those updated during 
fine-tuning) and non-trainable ones (frozen layers), allowing you 
to verify that your fine-tuning strategy is applied correctly.
"""

def get_model_num_params(model):
    """
    Returns the number of trainable, non-trainable and total parameters of a PyTorch model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    return {
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params": total_params
    }
