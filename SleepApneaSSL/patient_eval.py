import os
import torch
import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.amp import autocast

# Import the architecture and dataset from your finetuning script
from clinical_finetune import SubjectSplitDataset, ClinicalClassifier
from model import EEGEncoder

def evaluate_patient_level():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading test data and model...")

    # 1. Recreate the EXACT SAME 8 Test Subjects using Seed (42)
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
    
    # Grab the exact same 8 Test Subjects
    test_subs = osa_subs[16:20] + ctrl_subs[16:20]
    random.shuffle(test_subs)
    
    test_dataset = SubjectSplitDataset('E:/SleepApneaProcessed', test_subs, label_dict)
    # Crank batch size up to 16 for faster inference
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 2. Load the Saved Model
    encoder = EEGEncoder(in_channels=20)
    model = ClinicalClassifier(encoder).to(device)
    model.load_state_dict(torch.load('E:/SleepApnea/SleepApneaSSL/clinical_finetuned_model.pth', weights_only=True))
    model.eval()

    # 3. Extract Raw Probabilities AND Subject IDs
    all_labels = []
    all_probs = []
    all_sub_ids = []

    print(f"Running inference on {len(test_subs)} Test Subjects to extract window signals...")
    with torch.no_grad():
        for data, labels, sub_ids, _ in test_loader:
            data = data.to(device)
            with autocast('cuda'):
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)[:, 1] 
                
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_sub_ids.extend(sub_ids.numpy())

    # 4. Aggregate by Patient
    print("\n--- AGGREGATING PATIENT-LEVEL METRICS ---")
    patient_probs = {}
    patient_labels = {}

    for sub_id, prob, label in zip(all_sub_ids, all_probs, all_labels):
        if sub_id not in patient_probs:
            patient_probs[sub_id] = []
            # The label is identical for all windows belonging to the same patient
            patient_labels[sub_id] = label 
        patient_probs[sub_id].append(prob)

    final_patient_probs = []
    final_patient_labels = []

    for sub_id in patient_probs.keys():
        # Calculate the Mean Probability across the entire night
        mean_prob = np.percentile(patient_probs[sub_id], 95)
        
        final_patient_probs.append(mean_prob)
        final_patient_labels.append(patient_labels[sub_id])
        
        status = "OSA" if patient_labels[sub_id] == 1 else "Control"
        print(f"Patient ID {sub_id} ({status}) -> Nightly OSA Probability Score: {mean_prob:.4f}")

    # 5. Calculate Final Patient-Level ROC and AUC
    fpr, tpr, thresholds = roc_curve(final_patient_labels, final_patient_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\n==========================================")
    print(f" FINAL CLINICAL PATIENT-LEVEL AUC: {roc_auc:.4f}")
    print(f"==========================================")
    
    # Calculate optimal patient-level threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    tuned_preds = [1 if p >= optimal_threshold else 0 for p in final_patient_probs]
    
    print("\nPatient-Level Confusion Matrix:")
    print(confusion_matrix(final_patient_labels, tuned_preds))
    print("\nPatient-Level Classification Report:")
    print(classification_report(final_patient_labels, tuned_preds, target_names=['Control (0)', 'OSA (1)']))

if __name__ == '__main__':
    evaluate_patient_level()