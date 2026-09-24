import os
import sys
import glob
import re
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.loss_ablation import (
    PhysioCLRWrapper,
    pretrain_ssl,
    finetune_downstream,
    TSV_PATH,
    DATA_DIR,
    set_seed
)
from downstream_finetune import load_bids_labels
from experiments.paired_permutation_test import paired_permutation_test, load_and_align

SEEDS = [42, 123, 2025, 7, 13]
RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\A3_multiseed'
OLD_MULTISEED_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_predictions(path):
    d = np.load(path)
    return {
        'patient_ids': d['patient_ids'].astype(int),
        'targets': d['targets'].astype(float),
        'probs': d['probs'].astype(float)
    }

def align_three(path_a3, path_a0, path_a1):
    d3 = load_predictions(path_a3)
    d0 = load_predictions(path_a0)
    d1 = load_predictions(path_a1)

    common_ids = sorted(set(d3['patient_ids']) & set(d0['patient_ids']) & set(d1['patient_ids']))
    
    t, p3, p0, p1 = [], [], [], []
    for pid in common_ids:
        i3 = np.where(d3['patient_ids'] == pid)[0][0]
        i0 = np.where(d0['patient_ids'] == pid)[0][0]
        i1 = np.where(d1['patient_ids'] == pid)[0][0]
        
        t.append(d3['targets'][i3])
        p3.append(d3['probs'][i3])
        p0.append(d0['probs'][i0])
        p1.append(d1['probs'][i1])
        
    return np.array(t), np.array(p3), np.array(p0), np.array(p1)


def main():
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
    labels_for_split = [label_map[p] for p in pids]

    results = {}

    for seed in SEEDS:
        print(f"\n>>> Running A3 Seed {seed} <<<")
        
        # Consistent data split (random_state=42)
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
        
        seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        enc_path = os.path.join(seed_dir, "A3_PhysioCLR_encoder.pth")
        pred_path = os.path.join(seed_dir, "A3_PhysioCLR_predictions.npz")
        
        loss_fn = PhysioCLRWrapper(temperature=0.07, lambda_temporal=0.15)
        
        # SSL
        set_seed(seed)
        pretrain_ssl('A3_PhysioCLR', loss_fn, ssl_files, device, enc_path)
        
        # Downstream
        set_seed(seed)
        metrics = finetune_downstream(
            enc_path, label_map, train_files, val_files, test_files, device,
            predictions_save_path=pred_path, seed=seed
        )
        
        results[seed] = metrics
        
    print("\n=== RUN COMPLETE. AGGREGATING RESULTS ===")
    
    auroc_list = [results[s]['auroc'] for s in SEEDS]
    auprc_list = [results[s]['auprc'] for s in SEEDS]
    
    mean_auroc, std_auroc = np.mean(auroc_list), np.std(auroc_list, ddof=1)
    mean_auprc, std_auprc = np.mean(auprc_list), np.std(auprc_list, ddof=1)
    med_auroc, med_auprc = np.median(auroc_list), np.median(auprc_list)
    
    print(f"A3 AUROC: {mean_auroc:.4f} +- {std_auroc:.4f} (Median: {med_auroc:.4f})")
    print(f"A3 AUPRC: {mean_auprc:.4f} +- {std_auprc:.4f} (Median: {med_auprc:.4f})")
    
    diff_results = []
    
    for seed in SEEDS:
        path_a3 = os.path.join(RESULTS_DIR, f"seed_{seed}", "A3_PhysioCLR_predictions.npz")
        path_a0 = os.path.join(OLD_MULTISEED_DIR, f"seed_{seed}", "A0_VanillaNTXent_predictions.npz")
        path_a1 = os.path.join(OLD_MULTISEED_DIR, f"seed_{seed}", "A1_TemporalNTXent_predictions.npz")
        
        t, p3, p0, p1 = align_three(path_a3, path_a0, path_a1)
        
        diff_a3_a0_roc, p_roc_0 = paired_permutation_test(t, p3, p0, roc_auc_score, n_perm=10000, seed=42)
        diff_a3_a1_roc, p_roc_1 = paired_permutation_test(t, p3, p1, roc_auc_score, n_perm=10000, seed=42)
        diff_a3_a0_prc, p_prc_0 = paired_permutation_test(t, p3, p0, average_precision_score, n_perm=10000, seed=42)
        diff_a3_a1_prc, p_prc_1 = paired_permutation_test(t, p3, p1, average_precision_score, n_perm=10000, seed=42)
        
        diff_results.append({
            'seed': seed,
            'A3-A0_AUROC': diff_a3_a0_roc, 'p_A3-A0_AUROC': p_roc_0,
            'A3-A1_AUROC': diff_a3_a1_roc, 'p_A3-A1_AUROC': p_roc_1,
            'A3-A0_AUPRC': diff_a3_a0_prc, 'p_A3-A0_AUPRC': p_prc_0,
            'A3-A1_AUPRC': diff_a3_a1_prc, 'p_A3-A1_AUPRC': p_prc_1,
        })
        
    num_a3_gt_a0 = sum([1 for r in diff_results if r['A3-A0_AUROC'] > 0])
    num_a3_gt_a1 = sum([1 for r in diff_results if r['A3-A1_AUROC'] > 0])
    
    print(f"Seeds A3 > A0 (AUROC): {num_a3_gt_a0}")
    print(f"Seeds A3 > A1 (AUROC): {num_a3_gt_a1}")
    
    out_data = {
        'A3_aggregate': {
            'mean_auroc': mean_auroc, 'std_auroc': std_auroc, 'median_auroc': med_auroc,
            'mean_auprc': mean_auprc, 'std_auprc': std_auprc, 'median_auprc': med_auprc
        },
        'diffs': diff_results,
        'raw_metrics': results,
        'num_a3_gt_a0': num_a3_gt_a0,
        'num_a3_gt_a1': num_a3_gt_a1
    }
    
    with open(os.path.join(RESULTS_DIR, "A3_analysis.json"), "w") as f:
        json.dump(out_data, f, indent=4)
        
    # Write report stub
    report_path = r'E:\SleepApnea\SleepApneaSSL\experiment_reports\A3_FIVE_SEED_REPORT.md'
    with open(report_path, "w") as f:
        f.write(f"# A3 Five-Seed Multiseed Report\n\n")
        f.write(f"## A3 Aggregate Metrics\n")
        f.write(f"- Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f} (Median: {med_auroc:.4f})\n")
        f.write(f"- Mean AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f} (Median: {med_auprc:.4f})\n\n")
        
        f.write(f"## Seed vs A0 and A1\n")
        f.write(f"- Number of seeds A3 > A0 (AUROC): {num_a3_gt_a0}\n")
        f.write(f"- Number of seeds A3 > A1 (AUROC): {num_a3_gt_a1}\n\n")
        
        f.write(f"## Per-Seed Breakdown (Permutation test n=10000)\n")
        for r in diff_results:
            f.write(f"### Seed {r['seed']}\n")
            f.write(f"- A3-A0 AUROC: {r['A3-A0_AUROC']:+.4f} (p={r['p_A3-A0_AUROC']:.4f})\n")
            f.write(f"- A3-A1 AUROC: {r['A3-A1_AUROC']:+.4f} (p={r['p_A3-A1_AUROC']:.4f})\n")
            f.write(f"- A3-A0 AUPRC: {r['A3-A0_AUPRC']:+.4f} (p={r['p_A3-A0_AUPRC']:.4f})\n")
            f.write(f"- A3-A1 AUPRC: {r['A3-A1_AUPRC']:+.4f} (p={r['p_A3-A1_AUPRC']:.4f})\n\n")
            
if __name__ == '__main__':
    main()
