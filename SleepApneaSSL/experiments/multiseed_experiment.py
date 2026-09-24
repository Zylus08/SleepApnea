"""
multiseed_experiment.py
=======================
Runs A0 and A1 across multiple seeds [42, 123, 2025].
"""
import os
import sys
import glob
import re
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss_ablation import (
    VanillaNTXentLoss,
    TemporalNTXentWrapper,
    pretrain_ssl,
    finetune_downstream,
    TSV_PATH,
    DATA_DIR,
    set_seed
)
from downstream_finetune import load_bids_labels
from sklearn.model_selection import train_test_split

OUT_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'
os.makedirs(OUT_DIR, exist_ok=True)

# SEEDS = [42, 123, 2025]
SEEDS = [7, 13]

def main():
    print("=== MULTI-SEED EXPERIMENT (A0 vs A1) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    label_map = load_bids_labels(TSV_PATH)
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))
    
    pids = []
    pid_to_file = {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pids.append(pid)
                pid_to_file[pid] = f
    
    pids = sorted(list(set(pids)))
    if len(pids) < 10:
        print("Not enough data.")
        return

    labels_for_split = [label_map[p] for p in pids]
    
    results = {}
    
    for seed in SEEDS:
        print(f"\n>>> Running Seed {seed} <<<")
        set_seed(seed)
        
        # Consistent splitting methodology (using the seed for split too, or keep data split fixed?
        # Usually data split is kept fixed (seed 42) and only model init/shuffle varies, or both.
        # Let's keep data split fixed across seeds to allow paired tests!
        p_train_val, p_test, l_train_val, _ = train_test_split(
            pids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=42
        )
        p_train, p_val = train_test_split(
            p_train_val, test_size=0.25, stratify=l_train_val, random_state=42
        )

        train_files = [pid_to_file[p] for p in p_train]
        val_files   = [pid_to_file[p] for p in p_val]
        test_files  = [pid_to_file[p] for p in p_test]
        ssl_files = train_files

        ablations = {
            'A0_VanillaNTXent': VanillaNTXentLoss(temperature=0.5),
            'A1_TemporalNTXent': TemporalNTXentWrapper(temperature=0.5, lambda_decay=0.1),
        }
        
        results[seed] = {}
        for name, loss_fn in ablations.items():
            print(f"\n--- {name} (Seed {seed}) ---")
            
            # Sub-directory for seed to avoid overwriting weights
            seed_dir = os.path.join(OUT_DIR, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            enc_path = os.path.join(seed_dir, f"{name}_encoder.pth")
            pred_path = os.path.join(seed_dir, f"{name}_predictions.npz")
            
            # Re-seed exactly before SSL
            set_seed(seed)
            pretrain_ssl(name, loss_fn, ssl_files, device, enc_path)
            
            # Re-seed exactly before DS
            set_seed(seed)
            metrics = finetune_downstream(
                enc_path, label_map, train_files, val_files, test_files, device,
                predictions_save_path=pred_path, seed=seed
            )
            print(f"  [+] AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
            results[seed][name] = metrics

    metrics_path = os.path.join(OUT_DIR, "multiseed_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone! Multi-seed results saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
