import torch
from torch.utils.data import Dataset
import mne
from mne_bids import BIDSPath, read_raw_bids

class SleepApneaSSLDataset(Dataset):
    def __init__(self, bids_root, subjects, window_size_sec=30, sfreq=100, transform=None):
        """
        bids_root: str, path to the extracted ds008108 directory
        subjects: list of str, subject IDs (e.g. ['001', '002'])
        window_size_sec: int, length of the EEG crop for the SSL encoder
        sfreq: int, target sampling frequency
        """
        self.bids_root = bids_root
        self.window_size_sec = window_size_sec
        self.sfreq = sfreq
        self.transform = transform
        
        self.samples_per_window = int(window_size_sec * sfreq)
        self.windows = []  # Stores tuples of (subject_id, start_sample, end_sample)
        self.raw_cache = {} 
        
        print(f"Indexing {len(subjects)} subjects...")
        
        for sub in subjects:
            # Target the continuous nocturnal sleep sessions
            bids_path = BIDSPath(
                subject=sub, 
                session='nightSleep', 
                task='sleep',
                datatype='eeg', 
                root=self.bids_root,
                check=False 
            )
            
            try:
                # preload=False is the default, but we declare it for safety.
                # This reads only the headers, zero memory footprint.
                raw = read_raw_bids(bids_path, verbose=False)
                
                # Resampling is computationally expensive to do on the fly. 
                # If the dataset varies, you should pre-process and save to .fif. 
                # Assuming uniform sampling for this sprint:
                
                self.raw_cache[sub] = raw
                total_samples = raw.n_times
                
                # Map out valid sequential windows across the entire night
                for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
                    self.windows.append((sub, start, start + self.samples_per_window))
                    
            except Exception as e:
                print(f"Skipping sub-{sub}: {e}")
                
        print(f"Dataset ready. Total 30-second epochs mapped: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sub, start, end = self.windows[idx]
        raw = self.raw_cache[sub]
        
        # The slice operation triggers disk I/O only for this specific window
        data, _ = raw[:, start:end]  
        
        # data shape is (n_channels, n_samples)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        
        # Here is where your SimCLR augmentations will run
        if self.transform:
            tensor_data = self.transform(tensor_data)
            
        return tensor_data


if __name__ == "__main__":
    import os
    bids_root = os.path.join(os.path.dirname(__file__), "ds008108")
    subjects = ["001", "002"]
    dataset = SleepApneaSSLDataset(bids_root=bids_root, subjects=subjects)
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample 0 shape: {sample.shape}")
