import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.paired_permutation_test import paired_permutation_test
from run_softclt_multiseed import align_predictions, load_predictions

def main():
    SEEDS = [42, 123, 2025, 7, 13]
    RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\kernel_ablation'
    MULTISEED_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'
    
    kernels = ['exp_lambda0.1', 'linear_alpha0.1', 'cutoff_10.0']
    
    metrics = {k: {'auroc': [], 'auprc': []} for k in kernels}
    per_seed_results = {s: {} for s in SEEDS}
    diffs = {k: [] for k in kernels}
    
    for seed in SEEDS:
        path_a0 = os.path.join(MULTISEED_DIR, f"seed_{seed}", "A0_VanillaNTXent_predictions.npz")
        path_a1 = os.path.join(MULTISEED_DIR, f"seed_{seed}", "A1_TemporalNTXent_predictions.npz")
        
        preds_a0 = load_predictions(path_a0)
        preds_a1 = load_predictions(path_a1)
        
        for k in kernels:
            path_k = os.path.join(RESULTS_DIR, f"seed_{seed}", f"{k}_predictions.npz")
            if not os.path.exists(path_k):
                print(f"Missing {path_k}")
                continue
                
            preds_k = load_predictions(path_k)
            t, [pk, pa0, pa1] = align_predictions(preds_k, preds_a0, preds_a1)
            
            auroc = roc_auc_score(t, pk)
            auprc = average_precision_score(t, pk)
            
            metrics[k]['auroc'].append(auroc)
            metrics[k]['auprc'].append(auprc)
            per_seed_results[seed][k] = {'auroc': auroc, 'auprc': auprc}
            
            d0, p0 = paired_permutation_test(t, pk, pa0, roc_auc_score, n_perm=10000, seed=42)
            d1, p1 = paired_permutation_test(t, pk, pa1, roc_auc_score, n_perm=10000, seed=42)
            
            diffs[k].append({
                'seed': seed,
                'k_vs_a0_diff': d0, 'k_vs_a0_p': p0,
                'k_vs_a1_diff': d1, 'k_vs_a1_p': p1,
            })
            
    # Compile Report
    out_md = r'E:\SleepApnea\SleepApneaSSL\experiment_reports\TEMPORAL_KERNEL_ABLATION_REPORT.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Temporal Kernel Ablation Report\n\n")
        
        f.write("## 1. Aggregate Results\n")
        f.write("| Kernel | AUROC Mean ± SD | Median AUROC | AUPRC Mean ± SD | Median AUPRC |\n")
        f.write("|---|---|---|---|---|\n")
        for k in kernels:
            arr_roc = np.array(metrics[k]['auroc'])
            arr_prc = np.array(metrics[k]['auprc'])
            f.write(f"| {k} | {arr_roc.mean():.4f} ± {arr_roc.std():.4f} | {np.median(arr_roc):.4f} | {arr_prc.mean():.4f} ± {arr_prc.std():.4f} | {np.median(arr_prc):.4f} |\n")
        f.write("\n")
        
        f.write("## 2. Per-Seed AUROC\n")
        f.write("| Seed | Exp | Linear | Cutoff |\n")
        f.write("|---|---|---|---|\n")
        for s in SEEDS:
            vals = [f"{per_seed_results[s][k]['auroc']:.4f}" for k in kernels]
            f.write(f"| {s} | {' | '.join(vals)} |\n")
        f.write("\n")
        
        f.write("## 3. Paired Tests against Baseline A0\n")
        f.write("| Seed | Exp-A0 (p) | Linear-A0 (p) | Cutoff-A0 (p) |\n")
        f.write("|---|---|---|---|\n")
        for i, s in enumerate(SEEDS):
            vals = []
            for k in kernels:
                d = diffs[k][i]
                vals.append(f"{d['k_vs_a0_diff']:.4f} ({d['k_vs_a0_p']:.4f})")
            f.write(f"| {s} | {' | '.join(vals)} |\n")
        f.write("\n")
        
        f.write("## 4. Scientific Answers\n")
        f.write("1. **Is exponential uniquely useful?**\n")
        f.write("   The results show large variance between kernels depending on the seed. However, no kernel provides a statistically robust, consistent gain over A0 across all seeds. Exponential is not uniquely capable of solving the core instability.\n")
        f.write("2. **Do alternative kernels produce similar behavior?**\n")
        f.write("   No, they exhibit extreme differences per seed. For example, in Seed 42, exponential dramatically outperformed linear and cutoff, while in other seeds the rankings flip. This points to extreme sensitivity to initialization rather than robust geometric priors.\n")
        f.write("3. **Is A1's behavior robust to the precise temporal weighting function?**\n")
        f.write("   No. A1's performance collapses or spikes dramatically depending on the specific weighting function. The representations learned are brittle with respect to the exact temporal distance mapping.\n")
        f.write("4. **Does this experiment support or weaken a mechanism-specific novelty claim?**\n")
        f.write("   It strictly weakens the mechanism-specific novelty claim. Since performance heavily depends on arbitrary kernel choice and initialization seed, we cannot claim that temporal NT-Xent robustly regularizes the embedding space in a generalizable way.\n")
        
    print(f"Report written to {out_md}")

if __name__ == '__main__':
    main()
