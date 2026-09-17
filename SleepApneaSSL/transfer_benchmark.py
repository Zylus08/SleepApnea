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
    
    # Fixed TrainVal/Test Split (80/20)
    trainval_ids, test_ids = train_test_split(
        available_pids, test_size=0.2, stratify=available_labels, random_state=42
    )
    
    test_files = [pid_to_file[pid] for pid in test_ids]
    test_loader = DataLoader(
        LabeledStreamingDataset(test_files, label_map, is_train=False),
        batch_size=128, shuffle=False
    )
    
    print(f"\n[+] Baseline Test Set: {len(test_ids)} patients")
    
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
        
        # Subsetting for few-shot Train/Val
        subset_ids = get_few_shot_subset(trainval_ids, label_map, frac, seed=123)
        subset_ids = list(subset_ids)
        subset_labels = [label_map[p] for p in subset_ids]
        
        # Split subset into train and validation (90/10 of subset if enough, else 50/50, or just use train if too small)
        if len(subset_ids) > 4:
            try:
                train_ids, val_ids = train_test_split(subset_ids, test_size=0.15, stratify=subset_labels, random_state=123)
            except:
                train_ids, val_ids = train_test_split(subset_ids, test_size=0.25, random_state=123)
        else:
            train_ids = subset_ids
            val_ids = subset_ids  # Fallback for extremely small 1% sets

        train_files = [pid_to_file[pid] for pid in train_ids]
        val_files = [pid_to_file[pid] for pid in val_ids]
        
        total_train_patients = len(train_ids)
        osa_count = sum(1 for pid in train_ids if label_map[pid] == 1.0)
        ctrl_count = sum(1 for pid in train_ids if label_map[pid] == 0.0)
        print(f"[i] Few-Shot Train: {total_train_patients} patients (OSA: {osa_count} | Control: {ctrl_count})")
        print(f"[i] Few-Shot Val: {len(val_ids)} patients")
        
        train_loader = DataLoader(
            LabeledStreamingDataset(train_files, label_map, is_train=True),
            batch_size=128, shuffle=False
        )
        val_loader = DataLoader(
            LabeledStreamingDataset(val_files, label_map, is_train=False),
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
        
        best_val_auc = 0.0
        best_model_state = None
        
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
            
            # Evaluate on Validation Set for Model Selection
            _, val_auc, _ = evaluate_benchmark(model, val_loader, device)
            
            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Pat-AUC: {val_auc:.4f}")

        # Final Evaluation on Held-out Test Set
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
        test_win_auc, test_pat_auc, test_pat_acc = evaluate_benchmark(model, test_loader, device)
            
        print(f">>> {frac * 100:.0f}% Data Benchmark Complete | Test Pat-AUC: {test_pat_auc:.4f} (Val-selected)")
        
        # Save results
        results.append({
            'Fraction': frac,
            'Train_Patients': total_train_patients,
            'Window_AUC': test_win_auc,
            'Patient_AUC': test_pat_auc,
            'Patient_Acc': test_pat_acc
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
