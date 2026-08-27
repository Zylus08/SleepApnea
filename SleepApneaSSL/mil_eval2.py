import os
import torch
import pandas as pd
import numpy as np
import random
import gc
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.amp import autocast

# Import the architecture and dataset directly from your training script
from model import STFTEncoder2D
from mil_train import PatientBagDataset, AttentionMIL

def evaluate_production_mil():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Labels and Recreate the EXACT 80/20 Stratified Split
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {str(row['participant_id']).replace('sub-', '').zfill(3): (1 if str(row['group']).strip().lower() == 'osa' else 0) for _, row in df.iterrows()}

    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    valid_subjects = [s for s in all_processed if s in label_dict]
    
    osa_subs = [s for s in valid_subjects if label_dict[s] == 1]
    ctrl_subs = [s for s in valid_subjects if label_dict[s] == 0]
    
    random.seed(42)
    random.shuffle(osa_subs)
    random.shuffle(ctrl_subs)
    
    train_osa_split = int(len(osa_subs) * 0.8)
    train_ctrl_split = int(len(ctrl_subs) * 0.8)
    
    # Isolate the 20% Test Subjects
    test_subs = osa_subs[train_osa_split:] + ctrl_subs[train_ctrl_split:]
    random.shuffle(test_subs)
    
    print(f"\n--- DATASET SCALING ---")
    print(f"Evaluating strictly on {len(test_subs)} UNSEEN Test Subjects")
    
    test_dataset = PatientBagDataset('E:/SleepApneaProcessed', test_subs, label_dict, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. Load the Production AB-MIL Model
    base_encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    model = AttentionMIL(base_encoder).to(device)
    checkpoint = torch.load('E:/SleepApnea/SleepApneaSSL/mil_checkpoint.pth', weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_probs = []
    all_labels = []

    print("\n--- RUNNING AB-MIL EVALUATION ---")
    with torch.no_grad():
        for step, (bag, label) in enumerate(test_loader):
            # Squeeze out the batch dimension so bag is (N_windows, 20, 3000)
            bag = bag.squeeze(0).to(device)
            
            with autocast('cuda'):
                logits, attention = model(bag)
                # Convert raw logit to a 0.0 - 1.0 probability
                prob = torch.sigmoid(logits).item() 
            
            true_label = int(label.item())
            sub_id = test_subs[step]
            status = "OSA" if true_label == 1 else "Control"
            
            print(f"Patient {sub_id} ({status}) -> Predicted OSA Probability: {prob:.4f}")
            
            all_probs.append(prob)
            all_labels.append(true_label)
            
            # Clean up GPU memory after every massive patient bag
            del bag, logits, attention
            torch.cuda.empty_cache()
            gc.collect()

    # 3. Calculate Final AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n==========================================")
    print(f" FINAL PRODUCTION PATIENT-LEVEL AUC: {roc_auc:.4f}")
    print(f"==========================================")

    # Calculate optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    tuned_preds = [1 if p >= optimal_threshold else 0 for p in all_probs]
    
    print(f"\nOptimal Operating Threshold: {optimal_threshold:.4f}")
    print("\nPatient-Level Confusion Matrix:")
    print(confusion_matrix(all_labels, tuned_preds))
    print("\nPatient-Level Classification Report:")
    print(classification_report(all_labels, tuned_preds, target_names=['Control (0)', 'OSA (1)']))

if __name__ == '__main__':
    evaluate_production_mil()