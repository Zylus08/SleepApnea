import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
from torch.amp import autocast, GradScaler
from model import EEGEncoder
import gc

# --- 1. PATIENT-LEVEL DATASET (THE "BAG") ---
class PatientBagDataset(Dataset):
    def __init__(self, processed_root, subjects, label_dict):
        self.processed_root = processed_root
        self.subjects = subjects
        self.label_dict = label_dict
        self.samples_per_window = 3000

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        label = self.label_dict[sub]
        pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
        
        # Load entire patient directly from disk
        tensor_data = torch.load(pt_path, weights_only=True)
        total_samples = tensor_data.shape[1]
        
        # Chop the night into 30-second windows
        windows = []
        for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
            window = tensor_data[:, start:start + self.samples_per_window].clone()
            # Normalize each window
            window = (window - window.mean(dim=1, keepdim=True)) / (window.std(dim=1, keepdim=True) + 1e-6)
            windows.append(window)
            
        # Shape: (N_windows, 20, 3000)
        bag = torch.stack(windows)
        del tensor_data
        gc.collect()
        return bag, torch.tensor([label], dtype=torch.float32)

# --- 2. MEMORY-SAFE AB-MIL MODEL ---
class AttentionMIL(nn.Module):
    def __init__(self, encoder, input_dim=128, attention_dim=64):
        super().__init__()
        self.encoder = encoder
        
        # Unfreeze encoder for end-to-end learning
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        self.attention_V = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        
        # Binary classification output (1 node for Probability of OSA)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x, chunk_size=64):
        # Memory-safe encoding: process the night in chunks
        embeddings = []
        for i in range(0, x.size(0), chunk_size):
            chunk = x[i:i+chunk_size]
            embeddings.append(self.encoder(chunk))
            
        h = torch.cat(embeddings, dim=0) # Shape: (N_windows, 128)
        
        # Attention Mechanism
        A_V = self.attention_V(h) 
        A_U = self.attention_U(h) 
        A = self.attention_w(A_V * A_U) 
        A = torch.softmax(A, dim=0)  # Weights sum to 1.0
        
        # Aggregate and Classify
        patient_vector = torch.mm(A.t(), h) # Shape: (1, 128)
        logits = self.classifier(patient_vector) # Shape: (1, 1)
        
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
    train_dataset = PatientBagDataset('E:/SleepApneaProcessed', train_subs, label_dict)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 3. Initialize Models
    base_encoder = EEGEncoder(in_channels=20)
    base_encoder.load_state_dict(torch.load('E:/SleepApnea/SleepApneaSSL/iclr_pretrained_encoder.pth', weights_only=True))
    
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