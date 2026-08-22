import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
# Import your components
from clinical_finetune import SubjectSplitDataset # Ensure this yields (data, label, sub_idx, time_idx)
from model import EEGEncoder
from augmentations import ICLRTimeTransform
from temporal_loss import TemporalNTXentLoss

def train_ssl():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # 1. Hyperparameters
    batch_size = 64
    epochs = 20
    lr = 3e-4
    
    # 2. Setup Dataset with Augmentations
    processed_dir = 'E:/SleepApneaProcessed'
    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    # Use all available subjects for self-supervised pre-training (no labels needed!)
    print(f"Found {len(all_processed)} subjects for self-supervised pre-training.")
    
    # Dummy label dict since SSL doesn't use clinical labels
    dummy_labels = {s: 0 for s in all_processed}
    
    dataset = SubjectSplitDataset(processed_dir, all_processed, dummy_labels, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    # 3. Initialize Model, Loss, and Optimizer
    encoder = EEGEncoder(in_channels=20).to(device)
    
    # Projection head mapping encoder output to contrastive latent space (e.g., 128-dim)
    projection_head = torch.nn.Sequential(
        torch.nn.Linear(128, 128), # Adjust input dim based on your model.py final encoder output
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64)
    ).to(device)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projection_head.parameters()), lr=lr, weight_decay=1e-4)
    criterion = TemporalNTXentLoss(temperature=0.5, lambda_decay=0.05)
    scaler = GradScaler('cuda')
    augmenter = ICLRTimeTransform()

    print("\n--- STARTING ICLR-GRADE TEMPORAL SSL PRE-TRAINING ---")
    
    for epoch in range(epochs):
        encoder.train()
        projection_head.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, (data, _, sub_ids, time_idxs) in enumerate(dataloader):
            # 1. Move raw batch to GPU immediately
            data = data.to(device)
            sub_ids = sub_ids.to(device)
            time_idxs = time_idxs.to(device)

            # 2. Apply complex FFTs and masking instantly on the GPU
            view1, view2 = augmenter(data)

            optimizer.zero_grad()

            with autocast('cuda'):
                # Forward pass through encoder and projection head
                h1 = encoder(view1)
                h2 = encoder(view2)
                
                z1 = projection_head(h1)
                z2 = projection_head(h2)

                # Compute Novel Temporal Contrastive Loss
                loss = criterion(z1, z2, sub_ids, time_idxs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                # Calculate elapsed time and speed
                elapsed_time = time.time() - epoch_start_time
                batches_completed = batch_idx + 1
                batches_remaining = len(dataloader) - batches_completed
                
                # Predict remaining time
                time_per_batch = elapsed_time / batches_completed
                eta_seconds = batches_remaining * time_per_batch
                
                # Format into HH:MM:SS
                eta_string = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f} | Batch Time: {time_per_batch:.3f}s | ETA: {eta_string}")

        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}\n")

    # Save the pre-trained weights
    save_path = 'E:/SleepApnea/SleepApneaSSL/iclr_pretrained_encoder.pth'
    torch.save(encoder.state_dict(), save_path)
    print(f"Pre-trained ICLR encoder saved successfully to {save_path}")

if __name__ == '__main__':
    train_ssl()