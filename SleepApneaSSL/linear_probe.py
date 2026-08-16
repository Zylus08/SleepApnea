import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import mne
from mne_bids import BIDSPath
from model import EEGEncoder

class LinearProbeDataset(Dataset):
    def __init__(self, bids_root, subjects, window_size_sec=30, sfreq=100):
        self.bids_root = bids_root
        self.window_size_sec = window_size_sec
        self.sfreq = sfreq
        self.samples_per_window = int(window_size_sec * sfreq)
        
        # Read the clinical labels
        participants_file = os.path.join(bids_root, 'participants.tsv')
        df = pd.read_csv(participants_file, sep='\t')
        
        # Dynamically find the participant_id and group columns, ignoring case/spaces
        col_names = {col.strip().lower(): col for col in df.columns}
        group_col = col_names.get('group')
        pid_col = col_names.get('participant_id')
        
        if not group_col:
            raise ValueError(f"Could not find a 'group' column. Available headers: {df.columns.tolist()}")

        # Map labels (handling potential lowercase entries)
        label_map = {'control': 0, 'osa': 1}
        self.subject_labels = {}
        
        for _, row in df.iterrows():
            sub_id = str(row[pid_col]).replace('sub-', '').strip()
            group_val = str(row[group_col]).strip().lower()
            
            if group_val in label_map:
                self.subject_labels[sub_id] = label_map[group_val]

        self.windows = [] 
        self.raw_cache = {} 
        
        print("Mapping downstream dataset...")
        for sub in subjects:
            if sub not in self.subject_labels:
                continue
                
            bids_path = BIDSPath(subject=sub, session='nightSleep', datatype='eeg', root=self.bids_root, check=False)
            try:
                edf_path = bids_path.fpath
                if not os.path.exists(edf_path) or (os.path.getsize(edf_path) / (1024*1024)) < 1.0:
                    continue
                    
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                self.raw_cache[sub] = raw
                total_samples = raw.n_times
                
                # Extract windows and pair them with the subject's clinical label
                for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
                    self.windows.append((sub, start, start + self.samples_per_window, self.subject_labels[sub]))
            except Exception:
                pass

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sub, start, end, label = self.windows[idx]
        raw = self.raw_cache[sub]
        
        # Define the exact 20 channels we want to extract (Standard 10-20 layout)
        target_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                           'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'Oz']
        
        # MNE channel names might have prefixes/suffixes (e.g. 'EEG Fp1-REF')
        # We need to map our target names to the actual names in the EDF file
        actual_ch_names = raw.ch_names
        mapped_channels = []
        
        for target in target_channels:
            # Find the first channel that contains the target name (ignoring case)
            matched = next((ch for ch in actual_ch_names if target.lower() in ch.lower()), None)
            if matched:
                mapped_channels.append(matched)
                
        # If we can't find all 20, we have to pad with zeros to maintain tensor shape
        # Or, safer: just grab the first 20 channels if exact matching fails. 
        # For this sprint, we'll force exact shapes by taking the first 20 valid EEG channels.
        
        # Safest quick-fix for uneven channels in a sprint:
        eeg_channels = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        
        # Force exactly 20 channels
        if len(eeg_channels) >= 20:
            picked_indices = eeg_channels[:20]
        else:
            # Fallback if somehow a patient has less than 20 EEG channels
            picked_indices = eeg_channels
            
        data, _ = raw[picked_indices, start:end]
        
        # Pad with zeros if we strictly needed 20 but got less (rare, but prevents crashes)
        if data.shape[0] < 20:
            padding = np.zeros((20 - data.shape[0], data.shape[1]))
            data = np.vstack([data, padding])
        # Truncate if we got more than 20
        elif data.shape[0] > 20:
            data = data[:20, :]
            
        # Standardize
        tensor_data = torch.tensor(data, dtype=torch.float32)
        tensor_data = (tensor_data - tensor_data.mean(dim=1, keepdim=True)) / (tensor_data.std(dim=1, keepdim=True) + 1e-6)
        
        return tensor_data, torch.tensor(label, dtype=torch.long)


class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        
        # FREEZE THE ENCODER
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        embed_dim = self.encoder.fc.out_features
        # The Linear Probe
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

def train_linear_probe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = LinearProbeDataset(
        bids_root='E:/SleepApnea', 
        subjects=[f"{i:03d}" for i in range(1, 143)]
    )
    
    # Train/Test split for the downstream task (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize frozen encoder and linear head
    encoder = EEGEncoder(in_channels=20)
    # TODO: Load your saved pre-trained weights from train.py here if you saved them
    # encoder.load_state_dict(torch.load('simclr_encoder.pth'))
    
    model = LinearClassifier(encoder).to(device)
    
    # Only optimizing the linear layer
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Probe Loss: {total_loss/len(train_loader):.4f} | Train Accuracy: {acc:.2f}%")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    print(f"\nFinal Test Accuracy: {100. * correct / total:.2f}%")

if __name__ == '__main__':
    train_linear_probe()