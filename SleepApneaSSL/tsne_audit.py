import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# Import the architecture and dataset
from model import EEGEncoder
from mil_train import PatientBagDataset

def run_tsne_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading base encoder for latent space audit...")
    
    # 1. Load the Pre-trained SSL Base Encoder
    encoder = EEGEncoder(in_channels=20).to(device)
    encoder.load_state_dict(torch.load('E:/SleepApnea/SleepApneaSSL/iclr_pretrained_encoder.pth', map_location=device, weights_only=True))
    encoder.eval()

    # 2. Load Labels & Subjects
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {str(row['participant_id']).replace('sub-', '').zfill(3): (1 if str(row['group']).strip().lower() == 'osa' else 0) for _, row in df.iterrows()}

    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    valid_subjects = [s for s in all_processed if s in label_dict]
    
    osa_subs = [s for s in valid_subjects if label_dict[s] == 1]
    ctrl_subs = [s for s in valid_subjects if label_dict[s] == 0]

    # Grab a strict sample of 5 OSA and 5 Control patients to manage RAM/compute during t-SNE
    random.seed(42)
    audit_subs = random.sample(osa_subs, min(5, len(osa_subs))) + random.sample(ctrl_subs, min(5, len(ctrl_subs)))

    # is_training=False ensures we don't inject the Gaussian noise from the sanitizer phase
    dataset = PatientBagDataset('E:/SleepApneaProcessed', audit_subs, label_dict, is_training=False)

    all_embeddings = []
    all_labels = []

    print(f"Extracting 128-D embeddings from {len(audit_subs)} patients...")
    with torch.no_grad():
        for i in range(len(dataset)):
            bag, label_tensor = dataset[i]
            bag = bag.to(device)
            
            # Pass directly through the base encoder, bypassing the MIL head
            embeddings = encoder(bag) 
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Broadly label every 30s window with the patient's overarching label
            true_label = int(label_tensor.item())
            window_labels = np.full((embeddings.shape[0],), true_label)
            all_labels.append(window_labels)

    # 3. Concatenate and run t-SNE
    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"Total windows extracted: {X.shape[0]}")
    print("Running t-SNE dimensionality reduction (this might take a minute)...")
    
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    # 4. Plot the Latent Space
    plt.figure(figsize=(12, 10))
    
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='#3498db', label='Control Windows', alpha=0.4, s=15, edgecolors='none')
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='#e74c3c', label='OSA Windows', alpha=0.4, s=15, edgecolors='none')
    
    plt.title('t-SNE Projection of 128-D SSL Encoder Latent Space', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(markerscale=3)
    plt.grid(True, alpha=0.2)
    
    save_path = 'E:/SleepApnea/SleepApneaSSL/tsne_latent_space.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nt-SNE plot saved successfully to: {save_path}")

if __name__ == '__main__':
    run_tsne_audit()