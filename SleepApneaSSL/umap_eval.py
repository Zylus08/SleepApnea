import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score

try:
    import umap.umap_ as umap
except ImportError:
    print("[!] UMAP not installed. Please run: pip install umap-learn")
    exit(1)

# Import your architecture and dataset
# Adjust these imports based on your actual file names
from clinical_finetune import SubjectSplitDataset 
from model import STFTEncoder2D

def evaluate_latent_space():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # 1. Load BIDS Metadata & Test Set
    print("[+] Loading BIDS metadata...")
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {
        str(row['participant_id']).replace('sub-', '').zfill(3): (1 if str(row['group']).strip().lower() == 'osa' else 0)
        for _, row in df.iterrows()
    }

    all_processed = list(set([f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]))
    valid_subjects = [s for s in all_processed if s in label_dict]
    
    # Isolate the exact test subjects used in your benchmark
    test_subs = [s for s in valid_subjects if s in ['096', '109', '048', '056', '017', '007', '069', '118']]
    if not test_subs: # Fallback just in case zeros are dropped
        test_subs = [s for s in valid_subjects if int(s) in [96, 109, 48, 56, 17, 7, 69, 118]]

    print(f"[i] Extracting embeddings for Test Subjects: {test_subs}")
    test_dataset = SubjectSplitDataset('E:/SleepApneaProcessed', test_subs, label_dict)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Load Frozen Encoder
    checkpoint_path = r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth'
    print(f"[+] Loading pre-trained SSL encoder from: {checkpoint_path}")
    
    encoder = STFTEncoder2D(in_channels=20) # Adjust channels if needed
    
    # Load state dict safely (ignoring downstream MLP weights if they got saved here)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder' in k or 'mlp' not in k}
    
    try:
        encoder.load_state_dict(encoder_state, strict=False)
    except:
        encoder.load_state_dict(state_dict, strict=False)
        
    encoder.to(device)
    encoder.eval()
    
    # Freeze weights
    for param in encoder.parameters():
        param.requires_grad = False

    # 3. Extract Latent Vectors
    latents = []
    labels = []
    
    print("[*] Forward passing test set through frozen encoder...")
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) >= 3:
                x_batch, y_batch, sub_ids = batch[:3]
            else:
                x_batch, y_batch = batch[0], batch[1]
                
            x_batch = x_batch.to(device)
            
            with torch.amp.autocast('cuda'):
                # Extract projection/bottleneck vector
                z = encoder(x_batch) 
            
            latents.append(z.cpu().numpy())
            labels.append(y_batch.numpy())

    X_latents = np.vstack(latents)
    Y_labels = np.concatenate(labels)
    
    print(f"[+] Extracted latent shape: {X_latents.shape}")

    # 4. Calculate Silhouette Score
    print("[*] Calculating Silhouette Score...")
    sil_score = silhouette_score(X_latents, Y_labels, metric='cosine')
    print(f"[=>] High-Dimensional Silhouette Score: {sil_score:.4f}")

    # 5. UMAP Projection
    print("[*] Running UMAP dimensionality reduction (this may take a minute)...")
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=30, 
        min_dist=0.3, 
        metric='cosine', 
        random_state=42
    )
    X_umap = reducer.fit_transform(X_latents)

    # 6. Visualization
    print("[*] Generating publication-ready plot...")
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Map labels to text
    class_names = ['Healthy (Control)' if y == 0 else 'OSA' for y in Y_labels]
    
    # Plot using a high-contrast palette
    sns.scatterplot(
        x=X_umap[:, 0], 
        y=X_umap[:, 1], 
        hue=class_names, 
        palette={'Healthy (Control)': '#3498db', 'OSA': '#e74c3c'},
        s=30, 
        alpha=0.8, 
        edgecolor=None
    )

    # Format Axes
    plt.title('UMAP Projection of Frozen Physio-CLR Embeddings', fontsize=14, fontweight='bold', pad=20)
    plt.xticks([])  # Remove arbitrary UMAP ticks
    plt.yticks([])
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    # Add Silhouette Score annotation
    plt.text(
        0.05, 0.95, f'Silhouette Score (High-Dim): {sil_score:.3f}', 
        transform=plt.gca().transAxes, 
        fontsize=12, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8)
    )

    plt.legend(title="Clinical Class", loc='lower right', frameon=True)
    plt.tight_layout()

    # Save
    plot_path = 'E:/SleepApnea/SleepApneaSSL/latent_umap_projection.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[+] UMAP plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    evaluate_latent_space()