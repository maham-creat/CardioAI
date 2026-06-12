import os
import torch
import numpy as np
from torch.utils.data import Dataset

class UniversalECGDataset(Dataset):
    def __init__(self, dataframe, label_cols=None, is_train=True):
        self.df = dataframe.reset_index(drop=True)
        self.label_cols = label_cols
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load preprocessed, standardized relative numpy paths directly 
        signal = np.load(row["standardized_signal_path"]).astype(np.float32)
        
        # Consistent Z-Score normalization per track channel
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-8
        signal = (signal - mean) / std
        
        signal_tensor = torch.tensor(signal) # Format: (Channels, Length)

        if self.is_train and self.label_cols is not None:
            labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
            return signal_tensor, labels
        else:
            return signal_tensor, str(row["unique_record_id"])