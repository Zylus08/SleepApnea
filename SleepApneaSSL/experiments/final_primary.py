"""
final_primary.py
================
Final canonical protocol for the SleepApneaSSL primary ablation.
Runs A0 (Vanilla), A1 (Temporal), A2 (TemporalReg), A3 (PhysioCLR).
Uses seed 42, corrected TemporalNTXent, and true subject-IDs.

Output:
  - results/final_primary/*_predictions.npz
  - results/final_primary/metrics.json
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
    NTXentPlusTempReg,
    PhysioCLRWrapper,
    pretrain_ssl,
    finetune_downstream,
    TSV_PATH,
    DATA_DIR,
    SEED,
    set_seed
)
from downstream_finetune import load_bids_labels

OUT_DIR = r'E:\SleepApnea\SleepApneaSSL\results\final_primary'
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=== FINAL PRIMARY EXPERIMENT (SEED 42) ===")
    set_seed(SEED)
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

    # Deterministic split
    from sklearn.model_selection import train_test_split
    labels_for_split = [label_map[p] for p in pids]
    
    # 60/20/20
    p_train_val, p_test, l_train_val, _ = train_test_split(
        pids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=SEED
    )
    p_train, p_val = train_test_split(
        p_train_val, test_size=0.25, stratify=l_train_val, random_state=SEED
    )

    train_files = [pid_to_file[p] for p in p_train]
    val_files   = [pid_to_file[p] for p in p_val]
    test_files  = [pid_to_file[p] for p in p_test]

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    # SSL Pretraining Uses TRAIN ONLY (strictly no val/test leakage in representation)
    ssl_files = train_files

    ablations = {
        'A0_VanillaNTXent': VanillaNTXentLoss(temperature=0.5),
        'A1_TemporalNTXent': TemporalNTXentWrapper(temperature=0.5, lambda_decay=0.1),
        'A2_NTXent_TempReg': NTXentPlusTempReg(temperature=0.5, lambda_temporal=0.15),
        'A3_PhysioCLR': PhysioCLRWrapper(temperature=0.07, lambda_temporal=0.15),
    }

    results = {}
    for name, loss_fn in ablations.items():
        print(f"\n--- Running {name} ---")
        enc_path = os.path.join(OUT_DIR, f"{name}_encoder.pth")
        pred_path = os.path.join(OUT_DIR, f"{name}_predictions.npz")
        
        pretrain_ssl(name, loss_fn, ssl_files, device, enc_path)
        metrics = finetune_downstream(
            enc_path, label_map, train_files, val_files, test_files, device,
            predictions_save_path=pred_path, seed=SEED
        )
        print(f"  [+] Downstream AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
        results[name] = metrics

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone! Results saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
