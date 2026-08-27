import os
import glob
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from torch.amp import autocast

# Import from existing codebase
from downstream_finetune import load_bids_labels, LabeledStreamingDataset, BinaryFocalLossWithLogits, SleepApneaClassifier
from model import STFTEncoder2D

def get_few_shot_subset(train_ids, label_map, fraction, seed=42):
    """
    Subsets the training subject IDs for Few-Shot transfer, ensuring both
    OSA and Control classes are represented.
    """
    random.seed(seed)
    
    osa_ids = [pid for pid in train_ids if label_map[pid] == 1.0]
    ctrl_ids = [pid for pid in train_ids if label_map[pid] == 0.0]
    
    # Shuffle
    random.shuffle(osa_ids)
    random.shuffle(ctrl_ids)
    
    # Calculate counts
    n_osa = max(1, int(len(osa_ids) * fraction))
    n_ctrl = max(1, int(len(ctrl_ids) * fraction))
    
    # In case fraction is 1.0, don't overshoot
    if fraction == 1.0:
        n_osa = len(osa_ids)
        n_ctrl = len(ctrl_ids)
        
    subset_ids = osa_ids[:n_osa] + ctrl_ids[:n_ctrl]
    return set(subset_ids)

@torch.no_grad()
def evaluate_benchmark(model, loader, device):
    """Runs inference and calculates Patient-Level metrics."""
    model.eval()
    
    patient_probs = {}
    patient_targets = {}
    
    window_probs = []
    window_targets = []
    
    for x, y, pids in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.numpy()
        p_np = pids.numpy()
        
        for prob, label, pid in zip(probs, y_np, p_np):
            pid = int(pid)
            if pid not in patient_probs:
                patient_probs[pid] = []
            patient_probs[pid].append(float(prob))
            patient_targets[pid] = float(label)
            
            window_probs.append(float(prob))
            window_targets.append(float(label))

    # Window Level AUC
    if len(set(window_targets)) > 1:
        win_auc = roc_auc_score(window_targets, window_probs)
    else:
        win_auc = 0.5
        
    # Patient Level Metrics
    pat_ids = list(patient_probs.keys())
    pat_probs_arr = np.array([np.mean(patient_probs[pid]) for pid in pat_ids])
    pat_targets_arr = np.array([patient_targets[pid] for pid in pat_ids])
    
    if len(set(pat_targets_arr)) > 1:
        pat_auc = roc_auc_score(pat_targets_arr, pat_probs_arr)
        
        # Calculate optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(pat_targets_arr, pat_probs_arr)
        best_idx = np.argmax(tpr - fpr)
        threshold = thresholds[best_idx]
    else:
        pat_auc = 0.5
        threshold = 0.5
        
    preds = (pat_probs_arr >= threshold).astype(float)
    pat_acc = accuracy_score(pat_targets_arr, preds)
    
    return win_auc, pat_auc, pat_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r'E:\SleepApneaProcessed'
    tsv_path = r'E:\SleepApnea\participants.tsv'
    weights_path = r'E:\SleepApnea\SleepApneaSSL\stft_pretrained_encoder.pth'
    
    # 1. Parse BIDS Labels and Split Data
    label_map = load_bids_labels(tsv_path)
    all_files = glob.glob(os.path.join(data_dir, '*.pt'))
    
    available_pids = set()
    pid_to_file = {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                available_pids.add(pid)
                pid_to_file[pid] = f
                
    available_pids = list(available_pids)
    available_labels = [label_map[pid] for pid in available_pids]
    
    # Fixed Train/Test Split
    train_ids_list, test_ids_list = train_test_split(
        available_pids, test_size=0.2, stratify=available_labels, random_state=42
    )
    
    test_files = [pid_to_file[pid] for pid in test_ids_list]
    test_loader = DataLoader(
        LabeledStreamingDataset(test_files, label_map, is_train=False),
        batch_size=128, shuffle=False
    )
    
    print(f"\n[+] Baseline Test Set: {len(test_ids_list)} patients")
    
    # 2. Benchmark fractions
    fractions = [0.01, 0.05, 0.10, 1.0]
    results = []
    epochs = 12
    
    # Criterion
    criterion = BinaryFocalLossWithLogits(alpha=0.55, gamma=1.0)
    
    print("\n" + "=" * 60)
    print("  CROSS-COHORT FEW-SHOT TRANSFER BENCHMARK")
    print("=" * 60)
    
    for frac in fractions:
        print(f"\n--- Training on {frac * 100:.0f}% of Source Data ---")
        
        # Subsetting
        subset_ids = get_few_shot_subset(train_ids_list, label_map, frac, seed=123)
        train_files = [pid_to_file[pid] for pid in subset_ids]
        
        total_train_patients = len(subset_ids)
        osa_count = sum(1 for pid in subset_ids if label_map[pid] == 1.0)
        ctrl_count = sum(1 for pid in subset_ids if label_map[pid] == 0.0)
        print(f"[i] Few-Shot Subset: {total_train_patients} patients (OSA: {osa_count} | Control: {ctrl_count})")
        
        train_loader = DataLoader(
            LabeledStreamingDataset(train_files, label_map, is_train=True),
            batch_size=128, shuffle=False
        )
        
        # Determine approx number of windows (optional, for logging)
        # We can just record the number of patients for now.
        
        # Re-initialize Encoder & Classifier for each fraction run
        encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
        if os.path.exists(weights_path):
            encoder.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        else:
            print(f"[!] Warning: SSL weights not found at {weights_path}.")
            
        model = SleepApneaClassifier(encoder, embed_dim=128).to(device)
        
        # Only optimise the classifier MLP head
        optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_pat_auc = 0.0
        best_pat_acc = 0.0
        best_win_auc = 0.0
        
        # Training Loop
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            count = 0
            
            for x, y, _ in train_loader:
                x = x.to(device)
                y = y.to(device).float().view(-1)
                
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                
            train_loss = total_loss / max(1, count)
            
            # Evaluate
            win_auc, pat_auc, pat_acc = evaluate_benchmark(model, test_loader, device)
            
            if pat_auc > best_pat_auc:
                best_pat_auc = pat_auc
                best_pat_acc = pat_acc
                best_win_auc = win_auc
                
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                  f"Test Pat-AUC: {pat_auc:.4f} | Test Pat-Acc: {pat_acc:.4f}")
            
        print(f">>> {frac * 100:.0f}% Data Benchmark Complete | Best Pat-AUC: {best_pat_auc:.4f}")
        
        # Save results
        results.append({
            'Fraction': frac,
            'Train_Patients': total_train_patients,
            'Window_AUC': best_win_auc,
            'Patient_AUC': best_pat_auc,
            'Patient_Acc': best_pat_acc
        })
        
    # Generate Output
    df = pd.DataFrame(results)
    df.to_csv('E:/SleepApnea/SleepApneaSSL/transfer_results.csv', index=False)
    print("\n[+] Results saved to 'transfer_results.csv'")
    print(df.to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(df['Fraction'] * 100, df['Patient_AUC'], marker='o', color='royalblue', lw=2)
    plt.xscale('log')
    plt.xticks([1, 5, 10, 100], ['1%', '5%', '10%', '100%'])
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Patient-Level AUC')
    plt.title('Few-Shot Transfer Learning: Sample Efficiency')
    plt.grid(True, alpha=0.3)
    
    plot_path = 'E:/SleepApnea/SleepApneaSSL/few_shot_curve.png'
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Sample efficiency curve saved to '{plot_path}'")

if __name__ == '__main__':
    main()
