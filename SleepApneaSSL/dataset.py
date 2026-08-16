import os
import torch
from torch.utils.data import Dataset

class SleepApneaSSLDataset(Dataset):
    def __init__(self, processed_root, subjects, window_size_sec=30, sfreq=100, transform=None):
        self.processed_root = processed_root
        self.samples_per_window = int(window_size_sec * sfreq)
        self.transform = transform
        self.windows = []
        self.data_cache = {}
        
        print("Loading preprocessed tensors into memory pointers...")
        for sub in subjects:
            pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
            if not os.path.exists(pt_path):
                continue
                
            # Use mmap=True if RAM gets tight, otherwise standard load is fine for 35GB across 32GB RAM systems
            tensor_data = torch.load(pt_path, weights_only=True)
            self.data_cache[sub] = tensor_data
            
            total_samples = tensor_data.shape[1]
            for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
                self.windows.append((sub, start, start + self.samples_per_window))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sub, start, end = self.windows[idx]
        data = self.data_cache[sub][:, start:end]
        
        # Standardize
        data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-6)
        
        if self.transform:
            return self.transform(data)
        return data