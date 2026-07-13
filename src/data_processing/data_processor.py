"""
Data Processor
This module contains utility functions for data preprocessing and management.
It encapsulates the logic required to transform raw audio data into 
model-ready features (e.g., spectrograms) and handles dataset splitting.
By keeping this logic decoupled from the PipelineManager, we ensure a clean, 
modular design that is easy to test and maintain.
"""
from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset

def preprocess_audio(examples, feature_extractor):
    """
    Standalone function to process audio arrays into model-ready features.
    """
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        return_tensors="np"
    )
    return inputs


def split_data(dataset, test_size=0.2, use_eval=True):
    """Splits the dataset ensuring no song segments leak between sets."""
    song_names = [x.rsplit('_seg', 1)[0] for x in dataset['song_id']]
    
    # 1. First split: Train vs Rest (80/20)
    gss_train = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, rest_idx = next(gss_train.split(dataset, groups=song_names))
    
    train_ds = dataset.select(train_idx)
    rest_ds = dataset.select(rest_idx)
    
    # Re-extract groups for the remaining data
    rest_song_names = [x.rsplit('_seg', 1)[0] for x in rest_ds['song_id']]
    
    if use_eval:
        # 2. Second split: Test vs Eval (50/50 of the 30% → 15%/15% of total)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        test_idx, eval_idx = next(gss_test.split(rest_ds, groups=rest_song_names))
        
        return train_ds, rest_ds.select(test_idx), rest_ds.select(eval_idx)
    else:
        # Only Train and Test
        return train_ds, rest_ds, None