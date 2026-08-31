import torch
import mne
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T

class SHHSDataset(Dataset):
    def __init__(self, edf_paths, original_ch_names, target_freq=100, window_len=30):
        self.edf_paths = edf_paths
        self.target_freq = target_freq
        self.window_len = window_len
        self.samples_per_window = target_freq * window_len
        self.original_ch_names = original_ch_names # List of the 20 channels the model expects
        
        # We will populate these dynamically
        self.windows = [] 
        self.labels = []
        
        # Load and preprocess all EDFs
        self._prepare_data()

    def _prepare_data(self):
        for path in self.edf_paths:
            # 1. Load raw data silently
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            data = raw.get_data() # Shape: (11, Total_Samples)
            orig_freq = int(raw.info['sfreq'])
            
            # 2. Resample if necessary (e.g., 250Hz -> 100Hz)
            if orig_freq != self.target_freq:
                resampler = T.Resample(orig_freq=orig_freq, new_freq=self.target_freq)
                data = resampler(torch.tensor(data)).numpy()
                
            # 3. Map to 20-channel layout
            mapped_data = self._map_channels(data, raw.ch_names)
            
            # 4. Windowing logic (sliding 30s windows)
            # ... [Windowing and Label parsing logic goes here] ...

    def _map_channels(self, shhs_data, shhs_ch_names):
        """Maps the 11 SHHS channels to the 20-channel STFT model layout."""
        mapped = np.zeros((20, shhs_data.shape[1]), dtype=np.float32)
        
        # Example mapping dictionary (Needs to be updated with your actual mapping)
        mapping_dict = {
            'EEG': 'EEG_C4_M1',      # Match SHHS 'EEG' to your original name
            'EOG(L)': 'EOG_LOC',
            'SaO2': 'SpO2'
            # ...
        }
        
        for i, shhs_name in enumerate(shhs_ch_names):
            if shhs_name in mapping_dict:
                target_name = mapping_dict[shhs_name]
                if target_name in self.original_ch_names:
                    idx = self.original_ch_names.index(target_name)
                    mapped[idx, :] = shhs_data[i, :]
                    
        return mapped

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), torch.tensor(self.labels[idx])