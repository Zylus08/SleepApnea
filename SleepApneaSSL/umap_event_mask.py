import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_bids_events(bids_root, subject_id):
    """
    Parses the BIDS events.tsv file to extract Apnea and Hypopnea timestamps.
    Returns a dataframe of events.
    """
    event_path = os.path.join(bids_root, f"sub-{subject_id}", "eeg", f"sub-{subject_id}_task-sleep_events.tsv")
    
    if not os.path.exists(event_path):
        return pd.DataFrame()
        
    df = pd.read_csv(event_path, sep='\t')
    # Filter for respiratory events
    resp_events = df[df['trial_type'].str.contains('apnea|hypopnea', case=False, na=False)].copy()
    return resp_events

def assign_window_labels(latents, subject_ids, window_start_times, bids_root='E:/SleepApnea'):
    """
    Iterates through the extracted windows and masks them based on overlapping annotations.
    """
    event_labels = []
    
    for idx, (sub_id, w_start) in enumerate(zip(subject_ids, window_start_times)):
        events_df = load_bids_events(bids_root, sub_id)
        
        if events_df.empty:
            event_labels.append("Normal Sleep")
            continue
            
        w_end = w_start + 30.0 # 30-second window
        
        # Check if any event overlaps with this 30s window
        overlapping = events_df[
            (events_df['onset'] < w_end) & 
            ((events_df['onset'] + events_df['duration']) > w_start)
        ]
        
        if not overlapping.empty:
            # If multiple events overlap, take the most severe (Apnea > Hypopnea)
            event_types = overlapping['trial_type'].str.lower().tolist()
            if any('apnea' in e for e in event_types):
                event_labels.append("Apnea Event")
            elif any('hypopnea' in e for e in event_types):
                event_labels.append("Hypopnea Event")
            else:
                event_labels.append("Other Event")
        else:
            event_labels.append("Normal Sleep")
            
    return np.array(event_labels)

def plot_event_masked_umap(X_umap, event_labels, save_path='E:/SleepApnea/SleepApneaSSL/umap_event_mask.png'):
    """
    Generates the recolored UMAP projection.
    """
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Custom high-contrast palette for clinical events
    palette = {
        "Normal Sleep": "#e0e0e0",    # Light grey for background normal sleep
        "Hypopnea Event": "#f39c12",  # Orange for partial blockages
        "Apnea Event": "#c0392b"      # Deep red for full apneas
    }
    
    # Sort so apneas are plotted on top of normal sleep
    sort_order = np.argsort([list(palette.keys()).index(lbl) if lbl in palette else 0 for lbl in event_labels])
    
    sns.scatterplot(
        x=X_umap[sort_order, 0], 
        y=X_umap[sort_order, 1], 
        hue=event_labels[sort_order], 
        palette=palette,
        s=35, 
        alpha=0.9, 
        edgecolor=None
    )

    plt.title('UMAP Projection: Window-Level Respiratory Events', fontsize=14, fontweight='bold', pad=20)
    plt.xticks([])  
    plt.yticks([])
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    plt.legend(title="Window Physiology", loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Event-masked UMAP saved to: {save_path}")

# ==========================================
# Execution Block (Run this after loading your X_umap array)
# ==========================================
# Assuming you have X_umap, Y_subjects (array of subject IDs), and W_times (array of start times in seconds)
# event_labels = assign_window_labels(X_umap, Y_subjects, W_times)
# plot_event_masked_umap(X_umap, event_labels)