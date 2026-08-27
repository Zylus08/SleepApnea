import os
import glob
import re
import gc
import random
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
from model import STFTEncoder2D, SimCLR
from physio_clr import SpectralSubbandMasking, PhysioCLRLoss

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
            
            # Sequential windows for Temporal Continuity Loss
            for w in range(n_windows):
                x = bag[w].float()  # Extracts a single (20, 3000) window
                is_boundary = 1 if w == 0 else 0  # 1 indicates start of a new patient
                yield x, is_boundary
            
            # Force memory cleanup before the next patient
            del bag
            gc.collect()

def train_simclr_2d():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = StreamingWindowDataset()
    # Note: IterableDataset does not support shuffle=True in DataLoader. 
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=True)
    
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    simclr_model = SimCLR(encoder, projection_dim=64).to(device)
    
    optimizer = optim.AdamW(simclr_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysioCLRLoss(temperature=0.07, lambda_temporal=0.15)
    masker = SpectralSubbandMasking(p=0.5).to(device)
    scaler = GradScaler('cuda')
    
    epochs = 15
    print("\n--- STARTING PHYSIO-CLR PRE-TRAINING ---")
    simclr_model.train()
    masker.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_contrastive = 0.0
        total_temporal = 0.0
        batch_count = 0
        
        for x, is_boundary in loader:
            x = x.to(device)
            is_boundary = is_boundary.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. Generate STFT Spectrogram
                with torch.no_grad():
                    specs = simclr_model.encoder.stft(x)
                
                # 2. Spectral Subband Masking Augmentation
                specs1 = masker(specs)
                specs2 = masker(specs)
                
                # 3. Compute Projections
                _, z1 = simclr_model(specs1, input_is_spec=True)
                _, z2 = simclr_model(specs2, input_is_spec=True)
                
                # 4. Compute Physio-CLR Loss
                loss, metrics = criterion(z1, z2, is_boundary)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += metrics['loss_total']
            total_contrastive += metrics['loss_contrastive']
            total_temporal += metrics['loss_temporal']
            batch_count += 1
            
            if batch_count % 50 == 0:
                print(f"Epoch {epoch:02d} | Batch {batch_count:04d} | "
                      f"Total Loss: {metrics['loss_total']:.4f} | "
                      f"Contrastive: {metrics['loss_contrastive']:.4f} | "
                      f"Temporal: {metrics['loss_temporal']:.4f}")
            
        avg_loss = total_loss / max(1, batch_count)
        avg_contrastive = total_contrastive / max(1, batch_count)
        avg_temporal = total_temporal / max(1, batch_count)
        
        print(f">>> End of Epoch {epoch:02d}/{epochs:02d} | "
              f"Avg Total: {avg_loss:.4f} | "
              f"Avg Contrastive: {avg_contrastive:.4f} | "
              f"Avg Temporal: {avg_temporal:.4f}\n")
        
    torch.save(encoder.state_dict(), 'E:/SleepApnea/SleepApneaSSL/stft_pretrained_encoder.pth')
    print("\n[+] Saved pre-trained 2D encoder to 'stft_pretrained_encoder.pth'")

if __name__ == '__main__':
    train_simclr_2d()