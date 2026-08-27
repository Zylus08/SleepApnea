import os
import glob
import re
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.amp import autocast
from collections import defaultdict

# Import architecture and dataset from the refactored downstream pipeline
from downstream_finetune import LabeledStreamingDataset, SleepApneaClassifier, load_bids_labels
from model import STFTEncoder2D


def evaluate_and_tune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading test data and model...")

    data_dir = r'E:\SleepApneaProcessed'
    tsv_path = r'E:\SleepApnea\participants.tsv'

    # 1. Load BIDS Metadata (int-keyed label_map, consistent with downstream_finetune)
    label_map = load_bids_labels(tsv_path)

    # Discover all .pt files and extract unique subject IDs
    all_files = glob.glob(os.path.join(data_dir, '*.pt'))
    pid_to_files = defaultdict(list)
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pid_to_files[pid].append(f)

    available_pids = sorted(pid_to_files.keys())

    osa_pids = [p for p in available_pids if label_map[p] == 1.0]
    ctrl_pids = [p for p in available_pids if label_map[p] == 0.0]

    random.seed(42)
    random.shuffle(osa_pids)
    random.shuffle(ctrl_pids)

    # 8 Validation/Test Subjects (same split logic as original)
    test_pids = osa_pids[16:20] + ctrl_pids[16:20]
    random.shuffle(test_pids)

    # Gather all .pt files belonging to test subjects
    test_files = []
    for pid in test_pids:
        test_files.extend(pid_to_files[pid])

    print(f"[i] Test subjects ({len(test_pids)}): {test_pids}")
    print(f"[i] Test files: {len(test_files)}")

    test_dataset = LabeledStreamingDataset(test_files, label_map, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 2. Load the saved checkpoint
    checkpoint_path = r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth'
    if not os.path.exists(checkpoint_path):
        # Fallback if named differently
        checkpoint_path = r'E:\SleepApnea\SleepApneaSSL\clinical_finetuned_model.pth'

    print(f"[+] Loading model weights from: {checkpoint_path}")
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    model = SleepApneaClassifier(encoder, embed_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Extract Predictions and Map to Subjects
    window_labels = []
    window_probs = []

    patient_probs_dict = defaultdict(list)
    patient_labels_dict = {}

    print("Running inference on Test Set...")
    with torch.no_grad():
        for data, labels, pids in test_loader:
            data = data.to(device)
            with autocast('cuda'):
                logits = model(data)
                # Model outputs scalar logits (shape [batch_size])
                probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            labels_np = labels.numpy()
            pids_np = pids.numpy()

            window_probs.extend(probs_np)
            window_labels.extend(labels_np)

            # Map predictions to corresponding patient IDs
            for i, pid in enumerate(pids_np):
                pid = int(pid)
                patient_probs_dict[pid].append(float(probs_np[i]))
                patient_labels_dict[pid] = float(labels_np[i])

    # Aggregate window probabilities into patient averages
    patient_labels = []
    patient_probs = []
    for pid, probs_list in patient_probs_dict.items():
        patient_labels.append(patient_labels_dict[pid])
        patient_probs.append(float(np.mean(probs_list)))
        print(f"  Patient {pid}: {len(probs_list)} windows | "
              f"Mean Prob: {np.mean(probs_list):.4f} | Label: {int(patient_labels_dict[pid])}")

    # 4. Display Window-Level Analysis
    fpr_w, tpr_w, thresholds_w = roc_curve(window_labels, window_probs)
    auc_w = auc(fpr_w, tpr_w)
    idx_w = np.argmax(tpr_w - fpr_w)
    thresh_w = thresholds_w[idx_w]
    preds_w = [1 if p >= thresh_w else 0 for p in window_probs]

    print("\n=========================================")
    print(" WINDOW-LEVEL ANALYSIS")
    print("=========================================")
    print(f"Total Windows: {len(window_labels)}")
    print(f"Area Under Curve (AUC): {auc_w:.4f}")
    print(f"Optimal Tuned Threshold: {thresh_w:.4f}")
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(window_labels, preds_w))
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(window_labels, preds_w, target_names=['Control (0)', 'OSA (1)']))

    # 5. Display Patient-Level Analysis
    fpr_p, tpr_p, thresholds_p = roc_curve(patient_labels, patient_probs)
    auc_p = auc(fpr_p, tpr_p)
    idx_p = np.argmax(tpr_p - fpr_p)
    thresh_p = thresholds_p[idx_p]
    preds_p = [1 if p >= thresh_p else 0 for p in patient_probs]

    print("\n=========================================")
    print(" PATIENT-LEVEL ANALYSIS")
    print("=========================================")
    print(f"Total Patients: {len(patient_labels)}")
    print(f"Area Under Curve (AUC): {auc_p:.4f}")
    print(f"Optimal Tuned Threshold: {thresh_p:.4f}")
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(patient_labels, preds_p))
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(patient_labels, preds_p, target_names=['Control (0)', 'OSA (1)']))

    # 6. Plot ROC Curve
    plt.figure(figsize=(9, 7))
    plt.plot(fpr_w, tpr_w, color='royalblue', lw=2, linestyle=':', label=f'Window-Level (AUC = {auc_w:.3f})')
    plt.scatter([fpr_w[idx_w]], [tpr_w[idx_w]], color='royalblue', marker='x', s=80, label=f'Window Cutoff ({thresh_w:.2f})')

    plt.plot(fpr_p, tpr_p, color='darkorange', lw=2.5, label=f'Patient-Level (AUC = {auc_p:.3f})')
    plt.scatter([fpr_p[idx_p]], [tpr_p[idx_p]], color='red', marker='o', s=100, label=f'Patient Cutoff ({thresh_p:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('ROC - Sleep Apnea Detection (Window vs Patient Level)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    plot_path = 'E:/SleepApnea/SleepApneaSSL/roc_curve_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Saved dual ROC curve to: {plot_path}")


if __name__ == '__main__':
    evaluate_and_tune()