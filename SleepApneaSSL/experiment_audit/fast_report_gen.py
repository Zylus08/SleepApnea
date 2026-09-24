import os, sys, glob, re, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import STFTEncoder2D
from downstream_finetune import load_bids_labels, LabeledStreamingDataset
from experiments.paired_permutation_test import paired_permutation_test
from experiments.paired_bootstrap_ci import paired_bootstrap

TSV_PATH = r'E:\SleepApnea\participants.tsv'
DATA_DIR = r'E:\SleepApneaProcessed'
SEEDS = [42, 123, 2025, 7, 13]
RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def local_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    np.random.seed(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if not scores: return np.nan, np.nan
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def calc_erank(z):
    U, S, V = torch.svd(z, compute_uv=False)
    p = S / S.sum()
    ent = -(p * torch.log(p + 1e-9)).sum()
    return torch.exp(ent).item()

def main():
    print("PHASE 1-4: Statistical Metrics")
    seed_metrics = []
    stats = {
        'A0_VanillaNTXent': {'auroc':[], 'auprc':[], 'acc':[], 'f1':[]},
        'A1_TemporalNTXent': {'auroc':[], 'auprc':[], 'acc':[], 'f1':[]},
        'diff': {'auroc':[], 'auprc':[]}
    }
    
    for seed in SEEDS:
        path_a0 = os.path.join(RESULTS_DIR, f"seed_{seed}", "A0_VanillaNTXent_predictions.npz")
        path_a1 = os.path.join(RESULTS_DIR, f"seed_{seed}", "A1_TemporalNTXent_predictions.npz")
        d_a0 = np.load(path_a0); d_a1 = np.load(path_a1)
        
        targets = d_a0['targets']; probs_a0 = d_a0['probs']; probs_a1 = d_a1['probs']
        
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
        
        ci_a0 = local_bootstrap_ci(targets, probs_a0, roc_auc_score)
        ci_a1 = local_bootstrap_ci(targets, probs_a1, roc_auc_score)
        ci_diff = paired_bootstrap(targets, probs_a1, probs_a0, roc_auc_score, n=1000, seed=42)[1:3]
        
        seed_metrics.append({
            'seed': seed, 'n_patients': len(targets), 'pos_rate': targets.mean(),
            'a0_auroc': auroc_a0, 'a1_auroc': auroc_a1, 'diff_auroc': auroc_a1 - auroc_a0, 'p_auroc': p_auroc,
            'a0_auprc': auprc_a0, 'a1_auprc': auprc_a1, 'diff_auprc': auprc_a1 - auprc_a0, 'p_auprc': p_auprc,
            'ci_a0_auroc': ci_a0, 'ci_a1_auroc': ci_a1, 'ci_diff_auroc': ci_diff
        })
        
        stats['A0_VanillaNTXent']['auroc'].append(auroc_a0)
        stats['A0_VanillaNTXent']['auprc'].append(auprc_a0)
        stats['A1_TemporalNTXent']['auroc'].append(auroc_a1)
        stats['A1_TemporalNTXent']['auprc'].append(auprc_a1)
        stats['diff']['auroc'].append(auroc_a1 - auroc_a0)
        stats['diff']['auprc'].append(auprc_a1 - auprc_a0)

    print("PHASE 5-8: Effective Rank Extraction (Optimized)")
    label_map = load_bids_labels(TSV_PATH)
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))
    pid_to_file = {int(re.findall(r'\d+', os.path.basename(f))[0]): f for f in all_files if int(re.findall(r'\d+', os.path.basename(f))[0]) in label_map}
    pids = sorted(list(pid_to_file.keys()))
    labels_for_split = [label_map[p] for p in pids]
    
    erank_stats = []
    for seed in SEEDS:
        print(f"  Extracting Seed {seed}...")
        p_train_val, p_test, l_train_val, _ = train_test_split(pids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=42)
        test_files = [pid_to_file[p] for p in p_test]
        test_loader = DataLoader(LabeledStreamingDataset(test_files, label_map, is_train=False), batch_size=256, shuffle=False)

        enc0 = STFTEncoder2D(in_channels=20, embed_dim=128).eval()
        enc1 = STFTEncoder2D(in_channels=20, embed_dim=128).eval()
        enc0.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f'seed_{seed}', 'A0_VanillaNTXent_encoder.pth'), map_location='cpu', weights_only=True))
        enc1.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f'seed_{seed}', 'A1_TemporalNTXent_encoder.pth'), map_location='cpu', weights_only=True))
        
        all_z0 = []; all_z1 = []
        with torch.no_grad():
            for x, _, _ in test_loader:
                spec = enc0.stft(x)
                all_z0.append(enc0(spec, input_is_spec=True))
                all_z1.append(enc1(spec, input_is_spec=True))
                
        er0 = calc_erank(torch.cat(all_z0, dim=0))
        er1 = calc_erank(torch.cat(all_z1, dim=0))
        erank_stats.append({'seed': seed, 'a0_erank': er0, 'a1_erank': er1, 'diff_erank': er1 - er0})

    print("Dumping to JSON...")
    os.makedirs('experiment_reports', exist_ok=True)
    with open('experiment_reports/temp_stats.json', 'w') as f:
        json.dump({'seed_metrics': seed_metrics, 'stats': stats, 'erank_stats': erank_stats}, f, cls=NpEncoder)
    print("Done!")

if __name__ == '__main__':
    main()
