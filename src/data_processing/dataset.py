"""
AudioDataset: The Data Interface for the Model
This class serves as the bridge between the processed dataset and the model.
It transforms raw audio segments into tensor-based features while mapping 
categorical labels to numerical IDs. 

Responsibilities:
- Lazy loading: Retrieves audio segments from the dataset dynamically by index.
- Feature Transformation: Converts audio arrays to model-ready tensors using the feature extractor.
- Label Encoding: Translates string labels into indices expected by the loss function.

This decoupling allows the trainer to iterate through batches efficiently 
without loading the entire processed dataset into memory simultaneously.
"""

import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, dataset, feature_extractor, label_to_id):
        """
        Initializes the dataset wrapper.
        :param dataset: The processed Hugging Face dataset containing audio and labels.
        :param feature_extractor: The pre-trained processor to convert audio to model features.
        :param label_to_id: A dictionary mapping string labels to numerical IDs.
        """
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.label_to_id = label_to_id

    def __len__(self):
        """Returns the total number of segments in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_array = item["audio"]["array"]
        label_id = self.label_to_id[item['label']]

        # Extract features
        inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Here is the fix: Explicitly convert to torch.tensor
        # Because we saw it is currently coming out as a plain list.
        return {
            "input_values": torch.tensor(inputs["input_values"]).squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }