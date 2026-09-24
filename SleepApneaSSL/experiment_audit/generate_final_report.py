"""
generate_final_report.py
========================
Generates the authoritative A0 vs A1 final statistical report.
"""
import os, sys, glob, re, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import STFTEncoder2D
from downstream_finetune import load_bids_labels, LabeledStreamingDataset
from experiments.paired_permutation_test import paired_permutation_test
from experiments.paired_bootstrap_ci import paired_bootstrap

TSV_PATH = r'E:\SleepApnea\participants.tsv'
DATA_DIR = r'E:\SleepApneaProcessed'

SEEDS = [42, 123, 2025, 7, 13]
METHODS = ['A0_VanillaNTXent', 'A1_TemporalNTXent']
RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'
REP_DIR = r'E:\SleepApnea\SleepApneaSSL\results\representation_analysis'
os.makedirs(REP_DIR, exist_ok=True)

def calc_erank(z):
    # z: (N, D)
    # SVD on Z
    U, S, V = torch.svd(z, compute_uv=False)
    # p_i = sigma_i / sum(sigma_j)
    p = S / S.sum()
    # erank = exp(-sum(p_i * log p_i))
    ent = -(p * torch.log(p + 1e-9)).sum()
    return torch.exp(ent).item(), S.cpu().numpy()

def local_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    np.random.seed(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        # Ensure both classes exist
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if len(scores) == 0:
        return np.nan, np.nan
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    stats = {}
    for method in METHODS:
        stats[method] = {'auroc': [], 'auprc': [], 'acc': [], 'f1': []}
    stats['diff'] = {'auroc': [], 'auprc': []}
    
    seed_metrics = []
    
    for seed in SEEDS:
        path_a0 = os.path.join(RESULTS_DIR, f"seed_{seed}", f"A0_VanillaNTXent_predictions.npz")
        path_a1 = os.path.join(RESULTS_DIR, f"seed_{seed}", f"A1_TemporalNTXent_predictions.npz")
        
        d_a0 = np.load(path_a0)
        d_a1 = np.load(path_a1)
        
        np.testing.assert_array_equal(d_a0['patient_ids'], d_a1['patient_ids'])
        np.testing.assert_array_equal(d_a0['targets'], d_a1['targets'])
        
        targets = d_a0['targets']
        probs_a0 = d_a0['probs']
        probs_a1 = d_a1['probs']
        
        auroc_a0 = roc_auc_score(targets, probs_a0)
        auprc_a0 = average_precision_score(targets, probs_a0)
        acc_a0 = accuracy_score(targets, (probs_a0 > 0.5).astype(int))
        f1_a0 = f1_score(targets, (probs_a0 > 0.5).astype(int))
        
        auroc_a1 = roc_auc_score(targets, probs_a1)
        auprc_a1 = average_precision_score(targets, probs_a1)
        acc_a1 = accuracy_score(targets, (probs_a1 > 0.5).astype(int))
        f1_a1 = f1_score(targets, (probs_a1 > 0.5).astype(int))
        
        diff_auroc, p_auroc = paired_permutation_test(targets, probs_a1, probs_a0, roc_auc_score, n_perm=10000, seed=42)
        diff_auprc, p_auprc = paired_permutation_test(targets, probs_a1, probs_a0, average_precision_score, n_perm=10000, seed=42)
        
        ci_a0_auroc = local_bootstrap_ci(targets, probs_a0, roc_auc_score)
        ci_a1_auroc = local_bootstrap_ci(targets, probs_a1, roc_auc_score)
        ci_diff_auroc = paired_bootstrap(targets, probs_a1, probs_a0, roc_auc_score, n=1000, seed=42)[1:3]
        
        row = {
            'seed': seed, 'n_patients': len(targets), 'pos_rate': targets.mean(),
            'a0_auroc': auroc_a0, 'a1_auroc': auroc_a1, 'diff_auroc': auroc_a1 - auroc_a0, 'p_auroc': p_auroc,
            'a0_auprc': auprc_a0, 'a1_auprc': auprc_a1, 'diff_auprc': auprc_a1 - auprc_a0, 'p_auprc': p_auprc,
            'a0_acc': acc_a0, 'a1_acc': acc_a1, 'a0_f1': f1_a0, 'a1_f1': f1_a1,
            'ci_a0_auroc': ci_a0_auroc, 'ci_a1_auroc': ci_a1_auroc, 'ci_diff_auroc': ci_diff_auroc
        }
        seed_metrics.append(row)
        
        for k, v in zip(['auroc', 'auprc', 'acc', 'f1'], [auroc_a0, auprc_a0, acc_a0, f1_a0]):
            stats['A0_VanillaNTXent'][k].append(v)
        for k, v in zip(['auroc', 'auprc', 'acc', 'f1'], [auroc_a1, auprc_a1, acc_a1, f1_a1]):
            stats['A1_TemporalNTXent'][k].append(v)
            
        stats['diff']['auroc'].append(auroc_a1 - auroc_a0)
        stats['diff']['auprc'].append(auprc_a1 - auprc_a0)

    print("Extracting representations...")
    label_map = load_bids_labels(TSV_PATH)
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))
    pid_to_file = {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pid_to_file[pid] = f

    erank_stats = []
    spectra_a0 = []
    spectra_a1 = []
    
    from sklearn.model_selection import train_test_split
    pids = sorted(list(set(pid_to_file.keys())))
    labels_for_split = [label_map[p] for p in pids]

    for seed in SEEDS:
        print(f"Seed {seed} representations...")
        p_train_val, p_test, l_train_val, _ = train_test_split(
            pids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=42
        )
        test_files = [pid_to_file[p] for p in p_test]
        
        test_dataset = LabeledStreamingDataset(test_files, label_map, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        for method in METHODS:
            enc_path = os.path.join(RESULTS_DIR, f"seed_{seed}", f"{method}_encoder.pth")
            encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
            encoder.load_state_dict(torch.load(enc_path, map_location='cpu', weights_only=True))
            encoder = encoder.to(device)
            encoder.eval()
            
            all_z = []
            with torch.no_grad():
                for x, _, _ in test_loader:
                    x = x.to(device)
                    # Use stft first if needed, as in pretrain_ssl
                    # Actually STFTEncoder2D usually takes (B,C,T) and applies stft internally if input_is_spec is False
                    # Check if forward takes input_is_spec
                    # To be safe:
                    try:
                        z = encoder(x, input_is_spec=False)
                    except TypeError:
                        try:
                            z = encoder(x)
                        except RuntimeError:
                            spec = encoder.stft(x)
                            z = encoder(spec)
                            
                    all_z.append(z)
            Z = torch.cat(all_z, dim=0) # (Total_windows, 128)
            erank, S = calc_erank(Z)
            
            if method == 'A0_VanillaNTXent':
                spectra_a0.append(S / S.sum())
                a0_er = erank
            else:
                spectra_a1.append(S / S.sum())
                a1_er = erank
        
        erank_stats.append({
            'seed': seed,
            'a0_erank': a0_er,
            'a1_erank': a1_er,
            'diff_erank': a1_er - a0_er
        })

    plt.figure(figsize=(8,6))
    mean_s_a0 = np.mean(spectra_a0, axis=0)
    mean_s_a1 = np.mean(spectra_a1, axis=0)
    std_s_a0 = np.std(spectra_a0, axis=0)
    std_s_a1 = np.std(spectra_a1, axis=0)
    dims = np.arange(1, len(mean_s_a0)+1)
    
    plt.plot(dims, mean_s_a0, label='A0 (Vanilla NT-Xent)', color='red')
    plt.fill_between(dims, mean_s_a0 - std_s_a0, mean_s_a0 + std_s_a0, color='red', alpha=0.2)
    plt.plot(dims, mean_s_a1, label='A1 (Temporal NT-Xent)', color='blue')
    plt.fill_between(dims, mean_s_a1 - std_s_a1, mean_s_a1 + std_s_a1, color='blue', alpha=0.2)
    
    plt.title('Normalized Singular Value Spectrum (Mean over 5 seeds)')
    plt.xlabel('Principal Component')
    plt.ylabel('Normalized Variance Explained')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REP_DIR, 'A0_A1_singular_spectrum.png'))
    plt.close()

    os.makedirs('experiment_reports', exist_ok=True)
    with open('experiment_reports/temp_stats.json', 'w') as f:
        json.dump({'seed_metrics': seed_metrics, 'stats': stats, 'erank_stats': erank_stats}, f)

if __name__ == '__main__':
    main()
