import os
import glob
import re
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from model import STFTEncoder2D, SimCLR

# --- 2D SPECTROGRAM AUGMENTATIONS FOR SIMCLR ---
class SpectrogramAugmenter:
    """Applies random scale and noise augmentations to 1D raw waveforms before STFT."""
    def __call__(self, x):
        x_aug = x.clone()
        if torch.rand(1).item() > 0.5:
            scale = torch.empty(x_aug.size(0), 1).uniform_(0.8, 1.2)
            x_aug = x_aug * scale
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x_aug) * 0.05
            x_aug = x_aug + noise
        return x_aug

class StreamingWindowDataset(IterableDataset):
    def __init__(self, data_dir='E:/SleepApneaProcessed'):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
            
        all_files = glob.glob(os.path.join(data_dir, '*.pt'))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
            
        # Reserved test set subject IDs as integers
        test_ids = {36, 75, 95, 97, 4, 114, 117, 9, 15, 19, 83, 101, 2, 102, 115, 72, 8, 76, 68, 16, 113, 111, 65}
        
        self.valid_files = []
        
        for f in all_files:
            fname = os.path.basename(f)
            digits = re.findall(r'\d+', fname)
            if digits:
                pid = int(digits[0])
                if pid not in test_ids:
                    self.valid_files.append(f)

        self.augmenter = SpectrogramAugmenter()
        print(f"[+] Streaming Dataset initialized with {len(self.valid_files)} training subjects. Zero memory overhead.")

    def __iter__(self):
        # Shuffle patient order at the start of every epoch
        random.shuffle(self.valid_files)
        
        for f in self.valid_files:
            # Load exactly one patient at a time
            bag = torch.load(f, map_location='cpu', weights_only=True)
            
            # --- ROBUST TENSOR SHAPE PARSING ---
            if isinstance(bag, torch.Tensor):
                if bag.dim() == 2:
                    # It's a continuous 2D signal. Shape: (20, Total_Samples)
                    if bag.size(0) == 20:
                        n_windows = bag.size(1) // 3000
                        # Trim any excess trailing samples and reshape to (N_windows, 20, 3000)
                        bag = bag[:, :n_windows * 3000].view(20, n_windows, 3000).permute(1, 0, 2)
                    elif bag.size(1) == 20:
                        # Shape: (Total_Samples, 20)
                        n_windows = bag.size(0) // 3000
                        bag = bag[:n_windows * 3000, :].view(n_windows, 3000, 20).permute(0, 2, 1)
                    else:
                        continue  # Malformed tensor, skip patient
                        
                elif bag.dim() == 3:
                    # It's already chunked into 3D windows
                    if bag.size(0) == 20:
                        # Shape: (20, N_windows, 3000) -> Convert to (N_windows, 20, 3000)
                        bag = bag.permute(1, 0, 2)
                    # Else we assume it's already (N_windows, 20, 3000)
            else:
                continue # Unrecognized format, skip patient
            
            # Now `bag` is guaranteed to be shape (N_windows, 20, 3000)
            n_windows = bag.size(0)
            window_indices = list(range(n_windows))
            random.shuffle(window_indices)  # Shuffle windows for contrastive variance
            
            for w in window_indices:
                x = bag[w].float()  # Extracts a single (20, 3000) window
                
                x1 = self.augmenter(x)
                x2 = self.augmenter(x)
                
                yield x1, x2
            
            # Force memory cleanup before the next patient
            del bag
            gc.collect()
            
# --- NT-Xent LOSS ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = self.cosine_sim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        negatives = sim_matrix[mask].view(2 * batch_size, -1)
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        return nn.CrossEntropyLoss()(logits, labels)

def train_simclr_2d():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = StreamingWindowDataset()
    # Note: IterableDataset does not support shuffle=True in DataLoader. 
    # We already handle shuffling internally inside __iter__().
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=True)
    
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    simclr_model = SimCLR(encoder, projection_dim=64).to(device)
    
    optimizer = optim.AdamW(simclr_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = NTXentLoss(temperature=0.5)
    
    epochs = 15
    print("\n--- STARTING 2D STFT SIMCLR PRE-TRAINING ---")
    simclr_model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            _, z1 = simclr_model(x1)
            _, z2 = simclr_model(x2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print intermediate progress every 50 batches since we don't have len(loader)
            if batch_count % 50 == 0:
                print(f"Epoch {epoch:02d} | Batch {batch_count:04d} | Current Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / max(1, batch_count)
        print(f">>> End of Epoch {epoch:02d}/{epochs:02d} | Average SimCLR Loss: {avg_loss:.4f}\n")
        
    torch.save(encoder.state_dict(), 'E:/SleepApnea/SleepApneaSSL/stft_pretrained_encoder.pth')
    print("\n[+] Saved pre-trained 2D encoder to 'stft_pretrained_encoder.pth'")

if __name__ == '__main__':
    train_simclr_2d()