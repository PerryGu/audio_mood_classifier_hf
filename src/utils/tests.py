import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#tensorboard --logdir runs --port 6006

def inspect_dataset(dataset, num_samples=3):
    print(f"\n--- Inspection: Inspecting {num_samples} Samples ---")
    # Calculate total elements dynamically
   
    # Check if the dataset is a dictionary (split) or a single object
    if isinstance(dataset, dict):
        # Pick the first split (usually 'train') to inspect
        split_name = list(dataset.keys())[0]
        data_to_inspect = dataset[split_name]
        print(f"[INFO] Dataset is split. Inspecting split: '{split_name}'")
    else:
        data_to_inspect = dataset
        print(f"[INFO] Dataset is not split. Inspecting entire dataset.")

    print(f"\n--- Total number of elements in Dataset: {len(data_to_inspect)} Samples ---")
    for i in range(num_samples):
        sample = data_to_inspect[i]
        print(f"\nIndex {i}:")
        print(f"Song ID: {sample['song_id']}")
        print(f"Label: {sample['label']}")
        print(f"Audio array type: {type(sample['audio']['array'])}")
        print(f"Audio array length: {len(sample['audio']['array'])}")
        print(f"First 5 audio samples: {sample['audio']['array'][:5]}")
        print(f"Sample label after mapping: {sample['label']} (Type: {type(sample['label'])})")
        
        # This part prints all columns cleanly and dynamically
        for col in data_to_inspect.column_names:
            if col not in ['song_id', 'label', 'audio']:
                print(f"Column_name: {col}")
                data = sample[col]
                
                # Check for nested list (like input_values: 1024x128)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    print(f"  Type: Nested List")
                    print(f"  Outer length (pixels): {len(data)}")
                    print(f"  Inner length (pixels): {len(data[0])}")
                    print(f"  First 5 elements of inner list: {data[0][:5]}...")
                
                # Check for standard list/array
                elif hasattr(data, '__len__') and len(data) > 5:
                    print(f"  Type: List/Array")
                    print(f"  Length (pixels): {len(data)}")
                    print(f"  First 5 elements: {list(data)[:5]}...")
                
                # Print simple values as-is
                else:
                    print(f"  Data: {data} (Type: {type(data).__name__})")


    print(f"\n---------------------------------------------")



def test_spectrogram_integrity(dataset):
    print(f"\n--- Detailed Spectrogram Integrity Test ---")
    
    # Pick a random sample
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    # Print available metadata (assuming these keys exist in your dataset)
    print(f"Song ID: {sample.get('song_id', 'N/A')}")
    print(f"Category: {sample.get('label', 'N/A')}")
    print(f"Input values shape: {np.array(sample['input_values']).shape}")
    
    # Visualize the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(np.array(sample['input_values']).squeeze(), aspect='auto', origin='lower')
    plt.title(f"Spectrogram of {sample.get('song_id', 'Song')}")
    plt.colorbar(label='Amplitude')
    plt.show()



def check_for_leakage(split_dataset):
    """Checks for song overlap between all dataset sets (train, test, eval)."""
    sets = list(split_dataset.keys())
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            set1_name, set2_name = sets[i], sets[j]
            
            songs1 = set(split_dataset[set1_name]['song_id'])
            songs2 = set(split_dataset[set2_name]['song_id'])
            
            leakage = songs1.intersection(songs2)
            if leakage:
                print(f"[ALERT] Leakage found between {set1_name} and {set2_name}! {len(leakage)} songs are in both.")
            else:
                print(f"[SUCCESS] No song leakage detected between {set1_name} and {set2_name}.")


def generate_split_summary(split_dataset, ids=None):
    """Generates a summary for all sets present in the dataset."""
    print("\n--- Detailed Split Summary ---")
    for split_name in split_dataset.keys(): # Loop through whatever sets exist
        # 1. Take ALL data for this set
        full_df = pd.DataFrame({
            'song_base': [x.rsplit('_seg', 1)[0] for x in split_dataset[split_name]['song_id']],
            'label': split_dataset[split_name]['label']
        })
        
        # 2. Calculate counts
        summary = full_df.groupby('song_base').agg(
            segment_count=('song_base', 'size'),
            category=('label', 'first')
        ).reset_index()
        
        # 3. Apply filter
        if ids is not None and split_name in ids:
            summary = summary.iloc[ids[split_name]]
        
        print(f"\nSet: {split_name.upper()}")
        print(f"Total Unique Songs: {len(summary)}")
        print(summary.to_string())


def validate_dataset_preparation(mgr, label_to_id):
    """
    Orchestrates the creation and validation of the AudioDataset.
    """
    from src.data_processing.dataset import AudioDataset
    
    # Creation
    train_ds = AudioDataset(
        dataset=mgr.dataset['train'],
        feature_extractor=mgr.feature_extractor,
        label_to_id=label_to_id
    )
    
    # Validation
    test_dataset_item(train_ds)
    return train_ds

    
def test_dataset_item(dataset_instance):
    """
    Validates that a single item from the AudioDataset has the correct format.
    """
    try:
        sample = dataset_instance[0]
        assert "input_values" in sample, "Missing input_values"
        assert "labels" in sample, "Missing labels"
        print(f"[SUCCESS] Dataset item integrity verified. Shape: {sample['input_values'].shape}")
    except Exception as e:
        print(f"[ERROR] Dataset item integrity check failed: {e}")

