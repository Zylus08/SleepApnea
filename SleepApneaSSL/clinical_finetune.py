import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from functools import lru_cache
from torch.amp import autocast, GradScaler
import random
import gc
import pandas as pd
from model import EEGEncoder

class SubjectSplitDataset(Dataset):
    def __init__(self, processed_root, subjects, label_dict, transform=None): # <-- Added label_dict
        self.processed_root = processed_root
        self.subjects = subjects
        self.label_dict = label_dict
        self.transform = transform
        self.samples_per_window = 3000
        self.windows = []
        
        print(f"Scanning {len(subjects)} subjects (Sequential I/O Mode)...")
        for sub in subjects:
            pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
            if not os.path.exists(pt_path):
                continue
                
            # --- THE CLINICAL FIX ---
            # Now we look up the real label from your clinical file
            if sub not in label_dict:
                continue
            label = label_dict[sub]
            
            try:
                temp_tensor = torch.load(pt_path, weights_only=True, mmap=True)
                total_samples = temp_tensor.shape[1]
                del temp_tensor
                
                sub_windows = []
                for start in range(0, total_samples - self.samples_per_window, self.samples_per_window):
                    sub_windows.append((sub, start, start + self.samples_per_window, label))
                
                random.shuffle(sub_windows)
                self.windows.extend(sub_windows)
                
            except Exception:
                pass
        
        print(f"Total windows ready: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sub, start, end, label = self.windows[idx]
        pt_path = os.path.join(self.processed_root, f'sub-{sub}_eeg.pt')
        
        tensor_data = torch.load(pt_path, weights_only=True, mmap=True)
        data = tensor_data[:, start:end].clone()
        del tensor_data
        
        data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-6)
        
        # Convert subject string to a unique integer ID for tensor math
        sub_idx = int(sub) 
        time_idx = start // self.samples_per_window
        
        if self.transform:
            views = self.transform(data)
            return views, torch.tensor(label, dtype=torch.long), torch.tensor(sub_idx, dtype=torch.long), torch.tensor(time_idx, dtype=torch.long)

        return data.contiguous(), torch.tensor(label, dtype=torch.long), torch.tensor(sub_idx, dtype=torch.long), torch.tensor(time_idx, dtype=torch.long)

class ClinicalClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder            
        self.classifier = nn.Linear(encoder.fc.out_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

from torch.amp import autocast, GradScaler

def finetune_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # --- 1. LOAD CLINICAL METADATA FIRST ---
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {}
    for _, row in df.iterrows():
        sub_id = str(row['participant_id']).replace('sub-', '').zfill(3)
        group_label = str(row['group']).strip().lower()
        # Exactly matching the strings from your terminal
        label_dict[sub_id] = 1 if group_label == 'osa' else 0

    # --- 2. CREATE A BALANCED 50/50 SUBSET ---
    # --- 2. CREATE A BALANCED STRATIFIED SUBSET ---
    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    valid_subjects = [s for s in all_processed if s in label_dict]
    
    osa_subs = [s for s in valid_subjects if label_dict[s] == 1]
    ctrl_subs = [s for s in valid_subjects if label_dict[s] == 0]
    
    print(f"Available in pool -> OSA: {len(osa_subs)}, Control: {len(ctrl_subs)}")
    
    import random
    # Shuffle the pools first so we don't always grab the same patients
    random.seed(42)
    random.shuffle(osa_subs)
    random.shuffle(ctrl_subs)
    
    # Force a strict 80/20 Stratified Split:
    # Train: 16 OSA + 16 Control = 32
    # Test: 4 OSA + 4 Control = 8
    train_subs = osa_subs[:16] + ctrl_subs[:16]
    test_subs = osa_subs[16:20] + ctrl_subs[16:20]
    
    # Shuffle the final lists so the batches get a mix of both classes
    random.shuffle(train_subs)
    random.shuffle(test_subs)
    
    print("\n--- DATASET SPLIT ---")
    print(f"Training Subjects: {len(train_subs)}")
    print(f"Testing Subjects: {len(test_subs)}")

    # Pass the real labels into the datasets
    train_dataset = SubjectSplitDataset('E:/SleepApneaProcessed', train_subs, label_dict)
    test_dataset = SubjectSplitDataset('E:/SleepApneaProcessed', test_subs, label_dict)

    # shuffle=False is strictly required here to prevent the black-screen IO crash
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=0)

    # 2. Load Pre-trained Encoder & Build Model
    encoder = EEGEncoder(in_channels=20).to(device)
    weights_path = 'E:/SleepApnea/SleepApneaSSL/iclr_pretrained_encoder.pth'
    encoder.load_state_dict(torch.load(weights_path, weights_only=True))
    
    # Load the state dict and move the parameters to GPU
    state_dict = torch.load(weights_path, weights_only=True)
    
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        encoder.load_state_dict(new_state_dict)
    else:
        encoder.load_state_dict(state_dict)


    model = ClinicalClassifier(encoder).to(device)
    
    # 3. Full Fine-Tuning Setup with AMP
    # Explicitly only pass the classifier's weights to the optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the AMP Gradient Scaler
    scaler = GradScaler('cuda')

    print("\n--- STARTING FULL FINE-TUNING (AMP ENABLED) ---")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        model.encoder.eval()
        total_loss, correct, total = 0, 0, 0
        
        for batch_idx, (data, labels, _, _) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Cast operations to float16 to save massive amounts of VRAM
            with autocast('cuda'):
                outputs = model(data)
                loss = criterion(outputs, labels)
            
            # Scale loss and step optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")
        
        # --- ADD THIS RAM FLUSH ---
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Clinical Evaluation on Unseen Subjects
    print("\n--- RUNNING CLINICAL EVALUATION ON UNSEEN SUBJECTS ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, _, _ in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Use AMP for inference too
            with autocast('cuda'):
                outputs = model(data)
                
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report (Precision, Recall, F1):")
    print(classification_report(all_labels, all_preds, target_names=['Control (0)', 'OSA (1)']))

    torch.save(model.state_dict(), 'E:/SleepApnea/SleepApneaSSL/clinical_finetuned_model.pth')
    print("Fine-tuned model saved successfully.")
    
if __name__ == '__main__':
    finetune_and_evaluate()