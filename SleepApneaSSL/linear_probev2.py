import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import EEGEncoder

class LinearProbeDataset(Dataset):
    def __init__(self, processed_root, subjects, window_size_sec=30, sfreq=100):
        self.processed_root = processed_root
        self.samples_per_window = int(window_size_sec * sfreq)
        self.windows = []
        self.mmap_handles = {}
        
        print("Locking memory-mapped file handles (Zero-RAM Mode)...")
        for sub in subjects:
            pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
            if not os.path.exists(pt_path):
                continue
                
            # --- UPDATE THIS ---
            # You must replace this placeholder logic to read from your actual clinical labels
            label = self._get_label_for_subject(sub)
            
            try:
                mmap_tensor = torch.load(pt_path, weights_only=True, mmap=True)
                self.mmap_handles[sub] = mmap_tensor
                
                total_samples = mmap_tensor.shape[1]
                for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
                    self.windows.append((sub, start, start + self.samples_per_window, label))
            except Exception as e:
                print(f"Skipped {sub}: {e}")
                
        print(f"Total labeled windows ready: {len(self.windows)}")

    def _get_label_for_subject(self, sub):
        # PLACEHOLDER LOGIC: Replace with your pandas CSV/TSV lookup
        # Return 1 for OSA, 0 for Control
        return int(sub) % 2 

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sub, start, end, label = self.windows[idx]
        
        data = self.mmap_handles[sub][:, start:end].clone()
        data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-6)
        
        return data, torch.tensor(label, dtype=torch.long)


class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        
        # FREEZE the encoder entirely. We only train the final layer.
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Linear(encoder.fc.out_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)


def train_linear_probe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use the same 40 subjects to avoid RAM overload
    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    subset_subjects = all_processed[:40]

    dataset = LinearProbeDataset(
        processed_root='E:/SleepApneaProcessed', 
        subjects=subset_subjects
    )
    
    # We can safely bump batch size slightly here since the encoder is frozen
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    # 1. Load the architecture
    encoder = EEGEncoder(in_channels=20)
    
    # 2. Inject your converged pre-trained weights
    weights_path = 'E:/SleepApnea/SleepApneaSSL/simclr_encoder.pth'
    encoder.load_state_dict(torch.load(weights_path, weights_only=True))
    print("Pre-trained SimCLR weights loaded successfully.")

    # 3. Build the probe
    model = LinearClassifier(encoder).to(device)
    
    # Notice we ONLY pass the classifier parameters to the optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 15 # Linear probes converge very quickly
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    train_linear_probe()