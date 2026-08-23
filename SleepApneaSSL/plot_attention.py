import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mil_train import PatientBagDataset, AttentionMIL
from model import EEGEncoder
from torch.amp import autocast

def plot_patient_attention(target_patient_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load labels
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {str(row['participant_id']).replace('sub-', '').zfill(3): (1 if str(row['group']).strip().lower() == 'osa' else 0) for _, row in df.iterrows()}
    
    if target_patient_id not in label_dict:
        print(f"Patient {target_patient_id} not found.")
        return

    # Load just the target patient
    test_dataset = PatientBagDataset('E:/SleepApneaProcessed', [target_patient_id], label_dict)
    
    # Load Model
    base_encoder = EEGEncoder(in_channels=20)
    model = AttentionMIL(base_encoder).to(device)
    model.load_state_dict(torch.load('E:/SleepApnea/SleepApneaSSL/mil_production_model.pth', weights_only=True))
    model.eval()

    print(f"Extracting attention weights for Patient {target_patient_id}...")
    
    bag, label = test_dataset[0]
    bag = bag.to(device)
    
    with torch.no_grad():
        with autocast('cuda'):
            logits, attention = model(bag)
            prob = torch.sigmoid(logits).item()
            
    attention = attention.cpu().numpy().flatten()
    true_label = "OSA" if label.item() == 1 else "Control"
    
    print(f"True Label: {true_label}")
    print(f"Predicted Probability: {prob:.4f}")

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.plot(attention, color='crimson', linewidth=1.5)
    plt.title(f'Neural Attention Weights Over Time - Patient {target_patient_id} ({true_label}) | Pred: {prob:.4f}')
    plt.xlabel('Time (30-second Windows)')
    plt.ylabel('Attention Weight (Importance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f'E:/SleepApnea/SleepApneaSSL/attention_pt_{target_patient_id}.png'
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    # Let's look at the massive False Positive
    plot_patient_attention('136')
    # Let's look at a perfect True Positive
    plot_patient_attention('018')