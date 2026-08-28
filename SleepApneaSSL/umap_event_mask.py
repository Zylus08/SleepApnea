import os
import torch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

try:
    import umap.umap_ as umap
except ImportError:
    print("[!] UMAP not installed. Please run: pip install umap-learn")
    exit(1)

from clinical_finetune import SubjectSplitDataset 
from model import STFTEncoder2D

def load_bids_events(bids_root, subject_id):
    """Parses full BIDS events.tsv including sleep stages and respiratory events."""
    clean_id = re.sub(r'\D', '', str(subject_id)).zfill(3)
    
    event_path = os.path.join(
        bids_root, 
        f"sub-{clean_id}", 
        "ses-nightSleep", 
        "eeg", 
        f"sub-{clean_id}_ses-nightSleep_task-sleep_events.tsv"
    )
    
    if not os.path.exists(event_path):
        return pd.DataFrame()
        
    df = pd.read_csv(event_path, sep='\t')
    return df

def assign_window_labels(subject_ids, window_start_times, bids_root='E:/SleepApnea'):
    """Maps 30s windows to sleep stages, prioritizing Apnea ('A') annotations."""
    event_labels = []
    
    for sub_id, w_start in zip(subject_ids, window_start_times):
        events_df = load_bids_events(bids_root, sub_id)
        
        if events_df.empty:
            event_labels.append("Unannotated")
            continue
            
        w_end = w_start + 30.0 # 30-second window
        
        # Check overlapping annotations in this 30s window
        overlapping = events_df[
            (events_df['onset'] < w_end) & 
            ((events_df['onset'] + events_df['duration']) > w_start)
        ]
        
        if not overlapping.empty:
            types = overlapping['trial_type'].astype(str).tolist()
            # Prioritize Apnea if present during the window
            if 'A' in types:
                event_labels.append("Apnea Event")
            else:
                # Otherwise assign primary sleep stage
                event_labels.append(types[0])
        else:
            event_labels.append("Unannotated")
            
    return np.array(event_labels)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # 1. Load Data
    print("[+] Loading BIDS metadata...")
    df = pd.read_csv('E:/SleepApnea/participants.tsv', sep='\t')
    label_dict = {
        str(row['participant_id']).replace('sub-', '').zfill(3): (1 if str(row['group']).strip().lower() == 'osa' else 0)
        for _, row in df.iterrows()
    }

    all_processed = list(set([f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]))
    valid_subjects = [s for s in all_processed if s in label_dict]
    test_subs = [s for s in valid_subjects if s in ['096', '109', '048', '056', '017', '007', '069', '118']]
    
    test_dataset = SubjectSplitDataset('E:/SleepApneaProcessed', test_subs, label_dict)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Load Frozen Encoder
    checkpoint_path = r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth'
    encoder = STFTEncoder2D(in_channels=20) 
    
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder' in k or 'mlp' not in k}
    
    try:
        encoder.load_state_dict(encoder_state, strict=False)
    except:
        encoder.load_state_dict(state_dict, strict=False)
        
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters(): param.requires_grad = False

    # 3. Extract Latents & Track Timestamps
    latents, Y_subjects, W_times = [], [], []
    time_tracker = {}
    
    print("[*] Extracting representations and tracking temporal windows...")
    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch[0].to(device)
            sub_ids = batch[2] if len(batch) >= 3 else [test_subs[0]] * x_batch.size(0)
            
            with torch.amp.autocast('cuda'):
                z = encoder(x_batch)
            latents.append(z.cpu().numpy())
            
            for sid in sub_ids:
                sid_str = str(sid)
                if sid_str not in time_tracker:
                    time_tracker[sid_str] = 0.0
                W_times.append(time_tracker[sid_str])
                Y_subjects.append(sid_str)
                time_tracker[sid_str] += 30.0

    X_latents = np.vstack(latents)
    
    # 4. Generate Stage & Event Labels
    print("[*] Mapping sleep architecture and respiratory events...")
    event_labels = assign_window_labels(Y_subjects, W_times)
    
    # 5. UMAP Projection
    print("[*] Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
    X_umap = reducer.fit_transform(X_latents)

    # 6. Plotting Multi-Class Manifold
    print("[*] Generating physiological manifold visualization...")
    plt.figure(figsize=(11, 8.5))
    sns.set_theme(style="white")
    
    # Categorical Color Palette
    palette = {
        'Wake': '#e41a1c',        # Crimson
        'REM': '#ff7f00',         # Orange
        'N1': '#377eb8',          # Light Blue
        'N2': '#4daf4a',          # Green
        'N3': '#984ea3',          # Purple
        'Apnea Event': '#000000', # High-contrast Black
        'Unannotated': '#d9d9d9'  # Grey
    }
    
    # Layering order: lower priority rendered first, high priority (Apnea/Wake) rendered on top
    z_order = ['Unannotated', 'N2', 'N3', 'N1', 'REM', 'Wake', 'Apnea Event']
    sort_order = np.argsort([z_order.index(lbl) if lbl in z_order else 0 for lbl in event_labels])
    
    sns.scatterplot(
        x=X_umap[sort_order, 0], 
        y=X_umap[sort_order, 1], 
        hue=np.array(event_labels)[sort_order], 
        palette=palette,
        s=30, alpha=0.85, edgecolor=None
    )

    plt.title('UMAP Projection: Sleep Architecture & Respiratory Events', fontsize=14, fontweight='bold', pad=20)
    plt.xticks([])  
    plt.yticks([])
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title="Physiological Stage", loc='lower right', frameon=True, bbox_to_anchor=(1.0, 0.0))
    plt.tight_layout()

    save_path = r'E:\SleepApnea\SleepApneaSSL\umap_full_stage_manifold.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Full stage manifold saved to: {save_path}")

if __name__ == "__main__":
    main()