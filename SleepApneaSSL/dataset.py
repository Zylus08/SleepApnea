import os
import torch
from torch.utils.data import Dataset

class SleepApneaSSLDataset(Dataset):
    def __init__(self, processed_root, subjects, window_size_sec=30, sfreq=100, transform=None):
        self.processed_root = processed_root
        self.samples_per_window = int(window_size_sec * sfreq)
        self.transform = transform
        self.windows = []
        
        # This will hold the file streams permanently open
        self.mmap_handles = {}
        
        print("Locking memory-mapped file handles permanently (Zero-RAM Mode)...")
        for sub in subjects:
            pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
            if not os.path.exists(pt_path):
                continue
                
            try:
                # Open the file in mmap mode and keep the handle ALIVE
                mmap_tensor = torch.load(pt_path, weights_only=True, mmap=True)
                self.mmap_handles[sub] = mmap_tensor
                
                total_samples = mmap_tensor.shape[1]
                for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
                    self.windows.append((sub, start, start + self.samples_per_window))
            except Exception as e:
                print(f"Skipped {sub}: {e}")
                
        print(f"Total sleep windows mapped: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sub, start, end = self.windows[idx]
        
        # Slicing from an already-open handle is instantaneous and causes zero RAM spikes
        data = self.mmap_handles[sub][:, start:end].clone()
        
        # Standardize
        data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-6)
        
        if self.transform:
            return self.transform(data)
        return data