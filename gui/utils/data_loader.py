# utils/data_loader.py
import os
import json
import numpy as np

def load_ecg_data(ecg_folder_path):
    """
    Loads the numpy signal and the JSON metadata if available.
    Ensures safe handling if metadata is completely missing.
    """
    signal_path = os.path.join(ecg_folder_path, "signal.npy")
    meta_path = os.path.join(ecg_folder_path, "metadata.json")
    
    # 1. Load Signal
    if not os.path.exists(signal_path):
        raise FileNotFoundError(f"No signal.npy found in {ecg_folder_path}")
    signal = np.load(signal_path) # Shape: (1000, 12) or (Time, Leads)
    
    # 2. Try Loading Metadata
    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return signal, metadata

def get_available_test_samples(test_signal_dir):
    """Scans the test directory to list available patient samples."""
    if not os.path.exists(test_signal_dir):
        return []
    return sorted([d for d in os.listdir(test_signal_dir) if d.startswith("ECG_")])