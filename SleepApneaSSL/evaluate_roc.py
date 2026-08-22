import os
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.amp import autocast

# Import the architecture and dataset from your finetuning script
from clinical_finetune import SubjectSplitDataset, ClinicalClassifier
from model import EEGEncoder

def evaluate_and_tune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading test data and model...")

    # 1. Recreate the EXACT SAME 8 Test Subjects using your Seed (42)
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {}
    for _, row in df.iterrows():
        sub_id = str(row['participant_id']).replace('sub-', '').zfill(3)
        group_label = str(row['group']).strip().lower()
        label_dict[sub_id] = 1 if group_label == 'osa' else 0

    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    valid_subjects = [s for s in all_processed if s in label_dict]
    
    osa_subs = [s for s in valid_subjects if label_dict[s] == 1]
    ctrl_subs = [s for s in valid_subjects if label_dict[s] == 0]
    
    random.seed(42)
    random.shuffle(osa_subs)
    random.shuffle(ctrl_subs)
    
    # Grab the exact same Test Subjects
    test_subs = osa_subs[16:20] + ctrl_subs[16:20]
    random.shuffle(test_subs)
    
    test_dataset = SubjectSplitDataset('E:/SleepApneaProcessed', test_subs, label_dict)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 2. Load the Saved Model
    encoder = EEGEncoder(in_channels=20)
    model = ClinicalClassifier(encoder).to(device)
    model.load_state_dict(torch.load('E:/SleepApnea/SleepApneaSSL/clinical_finetuned_model.pth', weights_only=True))
    model.eval()

    # 3. Extract Raw Probabilities (Not Hard Predictions)
    all_labels = []
    all_probs = [] 

    print("Running inference on Test Set...")
    with torch.no_grad():
        for data, labels, _, _ in test_loader:
            data, labels = data.to(device), labels.to(device)
            with autocast('cuda'):
                outputs = model(data)
                # Apply softmax to convert raw logits into percentages (0.0 to 1.0)
                probs = torch.softmax(outputs, dim=1)[:, 1] 
                
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate ROC and Optimal Threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Youden's J Statistic to find the perfect balance
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n--- ROC ANALYSIS ---")
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Default PyTorch Threshold: 0.5000")
    print(f"Optimal Tuned Threshold: {optimal_threshold:.4f}")

    # 5. Apply the New Threshold
    tuned_preds = [1 if p >= optimal_threshold else 0 for p in all_probs]

    print("\n--- NEW TUNED CONFUSION MATRIX ---")
    print(confusion_matrix(all_labels, tuned_preds))
    print("\n--- NEW TUNED CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, tuned_preds, target_names=['Control (0)', 'OSA (1)']))

    # 6. Plot and Save the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', marker='o', s=100, label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic - Sleep Apnea Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plot_path = 'E:/SleepApnea/SleepApneaSSL/roc_curve.png'
    plt.savefig(plot_path)
    print(f"\nROC Curve saved to: {plot_path}")

if __name__ == '__main__':
    evaluate_and_tune()