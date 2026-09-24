import os, sys, glob, re, json, time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from experiments.loss_ablation import (
    TemporalNTXentWrapper,
    pretrain_ssl,
    finetune_downstream,
    TSV_PATH,
    DATA_DIR,
    set_seed,
)
from downstream_finetune import load_bids_labels

SEEDS = [42, 123, 2025, 7, 13]
RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\kernel_ablation'
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    label_map = load_bids_labels(TSV_PATH)
    all_files  = glob.glob(os.path.join(DATA_DIR, '*.pt'))

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
    labels_for_split = [label_map[p] for p in pids]

    # Equivalent kernel scales to exp(lambda=0.1)
    kernels = {
        'exp_lambda0.1':   TemporalNTXentWrapper(kernel_type='exponential', lambda_decay=0.1),
        'linear_alpha0.1': TemporalNTXentWrapper(kernel_type='linear', kernel_alpha=0.1),
        'cutoff_10.0':     TemporalNTXentWrapper(kernel_type='cutoff', kernel_cutoff=10.0)
    }

    for seed in SEEDS:
        print(f"\n>>> Running Kernel Ablation Seed {seed} <<<")
        p_train_val, p_test, l_train_val, _ = train_test_split(
            pids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=42
        )
        p_train, p_val = train_test_split(
            p_train_val, test_size=0.25, stratify=l_train_val, random_state=42
        )

        train_files = [pid_to_file[p] for p in p_train]
        val_files   = [pid_to_file[p] for p in p_val]
        test_files  = [pid_to_file[p] for p in p_test]
        
        seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for name, loss_fn in kernels.items():
            print(f"\n--- Kernel: {name} (Seed {seed}) ---")
            
            enc_path  = os.path.join(seed_dir, f"{name}_encoder.pth")
            pred_path = os.path.join(seed_dir, f"{name}_predictions.npz")

            if not os.path.exists(enc_path):
                set_seed(seed)
                pretrain_ssl(name, loss_fn, train_files, device, enc_path)
            else:
                print(f"Skipping pretraining, {enc_path} exists")

            set_seed(seed)
            metrics = finetune_downstream(
                enc_path, label_map, train_files, val_files, test_files, device,
                predictions_save_path=pred_path
            )
            print(f"Result {name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")

if __name__ == '__main__':
    main()
