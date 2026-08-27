import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
from torch.amp import autocast, GradScaler
from model import STFTEncoder2D, SimCLR
import gc

# --- 1. PATIENT-LEVEL DATASET (THE "BAG") ---
class PatientBagDataset(Dataset):
    def __init__(self, processed_root, subjects, label_dict, is_training=True):
        self.processed_root = processed_root
        self.subjects = subjects
        self.label_dict = label_dict
        self.samples_per_window = 3000
        self.is_training = is_training
        self.crop_windows = 480 # Standardize exactly 4 hours of sleep

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        label = self.label_dict[sub]
        pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
        
        tensor_data = torch.load(pt_path, weights_only=True)
        total_samples = tensor_data.shape[1]
        total_windows = total_samples // self.samples_per_window
        
        # --- PHASE 2: SEQUENCE STANDARDIZATION ---
        if self.is_training and total_windows > self.crop_windows:
            # Randomly crop a continuous 4-hour segment for data augmentation
            start_window = random.randint(0, total_windows - self.crop_windows)
            start_idx = start_window * self.samples_per_window
            end_idx = start_idx + (self.crop_windows * self.samples_per_window)
            tensor_data = tensor_data[:, start_idx:end_idx]
        else:
            # For testing, just take the first 4 hours to ensure standard tensor sizes
            tensor_data = tensor_data[:, :self.crop_windows * self.samples_per_window]

        windows = []
        for start in range(0, tensor_data.shape[1], self.samples_per_window):
            window = tensor_data[:, start:start + self.samples_per_window].clone()
            # Normalize
            window = (window - window.mean(dim=1, keepdim=True)) / (window.std(dim=1, keepdim=True) + 1e-6)
            
            # --- PHASE 3: ARTIFACT MASKING ---
            if self.is_training:
                # Inject 10% Gaussian Noise to force the model to ignore sensor glitches
                noise = torch.randn_like(window) * 0.1 
                window = window + noise
                
            windows.append(window)
            
        bag = torch.stack(windows)
        
        del tensor_data 
        gc.collect() 
        
        return bag, torch.tensor([label], dtype=torch.float32)

# --- 2. MEMORY-SAFE AB-MIL MODEL ---
class AttentionMIL(nn.Module):
    def __init__(self, encoder, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        
        # Gated Attention Mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        # Patient-level classifier
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, bag):
        # bag shape: (480, 20, 3000)
        num_windows = bag.size(0)
        chunk_size = 32  # Micro-batch size to keep VRAM < 1.5 GB
        
        embeddings = []
        for i in range(0, num_windows, chunk_size):
            chunk = bag[i:i + chunk_size]
            emb = self.encoder(chunk) # Shape: (32, 128)
            embeddings.append(emb)
            
        # Stack back into a single matrix H: (480, 128)
        H = torch.cat(embeddings, dim=0)
        
        # Calculate Attention Weights
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U) # Element-wise multiplication (Gated Attention)
        A = torch.transpose(A, 1, 0)     # (1, 480)
        A = torch.softmax(A, dim=1)      # Normalize weights across all 480 windows
        
        # Aggregate bag representation (weighted sum)
        M = torch.mm(A, H)               # (1, 128)
        
        # Predict patient-level probability logit
        logits = self.classifier(M)      # (1, 1)
        
        return logits, A
# --- 3. TRAINING LOOP WITH GRADIENT ACCUMULATION ---
def train_mil():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Labels
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {str(row['participant_id']).replace('sub-', '').zfill(3): (1 if str(row['group']).strip().lower() == 'osa' else 0) for _, row in df.iterrows()}

    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    valid_subjects = [s for s in all_processed if s in label_dict]
    
    osa_subs = [s for s in valid_subjects if label_dict[s] == 1]
    ctrl_subs = [s for s in valid_subjects if label_dict[s] == 0]
    
    # 2. Strict 80/20 Stratified Split
    random.seed(42)
    random.shuffle(osa_subs)
    random.shuffle(ctrl_subs)
    
    train_osa_split = int(len(osa_subs) * 0.8)
    train_ctrl_split = int(len(ctrl_subs) * 0.8)
    
    train_subs = osa_subs[:train_osa_split] + ctrl_subs[:train_ctrl_split]
    test_subs = osa_subs[train_osa_split:] + ctrl_subs[train_ctrl_split:]
    
    random.shuffle(train_subs)
    
    print("\n--- DATASET SCALING ---")
    print(f"Total Valid Subjects: {len(valid_subjects)}")
    print(f"Training on: {len(train_subs)} subjects")
    print(f"Reserved for Testing: {len(test_subs)} subjects")
    
    # Batch size is strictly 1 (One patient per step)
    train_dataset = PatientBagDataset('E:/SleepApneaProcessed', train_subs, label_dict, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 3. Initialize Models
    base_encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    # base_encoder.load_state_dict(torch.load('E:/SleepApnea/SleepApneaSSL/iclr_pretrained_encoder.pth', weights_only=True))
    
    model = AttentionMIL(base_encoder).to(device)
    checkpoint_path = 'E:/SleepApnea/SleepApneaSSL/mil_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print("\n[+] Found existing checkpoint! Resuming training from saved weights...")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    # 4. Scaled Training Parameters
    epochs = 50
    accumulation_steps = 16 # Simulates a batch size of 16 patients to perfectly smooth the gradients
    
    print("\n--- STARTING PRODUCTION AB-MIL TRAINING ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (bag, label) in enumerate(train_loader):
            # bag shape is (1, N_windows, 20, 3000). Squeeze to remove batch dim.
            bag = bag.squeeze(0).to(device)
            label = label.to(device)
            
            with autocast('cuda'):
                logits, attention_weights = model(bag)
                loss = criterion(logits, label)
                loss = loss / accumulation_steps # Normalize loss
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * accumulation_steps

        print(f"Epoch {epoch+1}/{epochs} | Avg Patient Loss: {total_loss/len(train_loader):.4f}")
        
        # --- NEW: SAVE A CHECKPOINT EVERY EPOCH ---
        torch.save(model.state_dict(), 'E:/SleepApnea/SleepApneaSSL/mil_checkpoint.pth')

    # (Keep the final production save at the very end outside the loop)
    torch.save(model.state_dict(), 'E:/SleepApnea/SleepApneaSSL/mil_production_model.pth')
    print("\nProduction AB-MIL model saved successfully!")

if __name__ == '__main__':
    train_mil()