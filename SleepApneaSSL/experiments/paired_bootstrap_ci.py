"""
paired_bootstrap_ci.py
======================
Computes paired statistical significance (p-values) and paired difference 
95% CIs between A1 (TemporalNTXent) and A0 (VanillaNTXent) using the 
predictions from the multi-seed experiment.

For each seed, it computes the difference in AUROC/AUPRC on exactly the same
test patients, then bootstraps that difference.
"""
import os
import json
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDS = [42, 123, 2025]
N_BOOT = 1000
RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'

def paired_bootstrap(targets, probs_A, probs_B, metric_fn, n=1000, seed=42):
    """
    Bootstraps the difference metric(A) - metric(B) on exactly the same patients.
    Returns: point_diff, CI_lower, CI_upper, p_value (H0: diff <= 0).
    """
    rng = np.random.RandomState(seed)
    idx_pos = np.where(targets == 1)[0]
    idx_neg = np.where(targets == 0)[0]
    
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
        
    point_diff = metric_fn(targets, probs_A) - metric_fn(targets, probs_B)
    diffs = []
    
    for _ in range(n):
        bp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        bn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        bi = np.concatenate([bp, bn])
        
        try:
            mA = metric_fn(targets[bi], probs_A[bi])
            mB = metric_fn(targets[bi], probs_B[bi])
            diffs.append(mA - mB)
        except ValueError:
            pass
            
    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]
    
    if len(diffs) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
        
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    
    # 1-tailed p-value for H1: A > B (diff > 0). 
    # Fraction of bootstrap samples where diff <= 0
    p_val = float(np.mean(diffs <= 0))
    
    return point_diff, ci_lo, ci_hi, p_val

def main():
    print("=== PAIRED BOOTSTRAP ANALYSIS (A1 vs A0) ===")
    
    summary = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
        
        path_a0 = os.path.join(seed_dir, 'A0_VanillaNTXent_predictions.npz')
        path_a1 = os.path.join(seed_dir, 'A1_TemporalNTXent_predictions.npz')
        
        if not os.path.exists(path_a0) or not os.path.exists(path_a1):
            print(f"  Missing predictions for seed {seed}. Skipping.")
            continue
            
        d_a0 = np.load(path_a0)
        d_a1 = np.load(path_a1)
        
        t0 = d_a0['targets']
        p0 = d_a0['probs']
        
        t1 = d_a1['targets']
        p1 = d_a1['probs']
        
        # Verify alignment
        if not np.array_equal(t0, t1):
            print("  ERROR: Targets do not match exactly. Paired test invalid.")
            continue
            
        print(f"  N test patients: {len(t0)} | Pos rate: {t0.mean():.3f}")
        
        # AUROC
        auroc_diff, auroc_lo, auroc_hi, auroc_pval = paired_bootstrap(
            t0, p1, p0, roc_auc_score, n=N_BOOT, seed=seed
        )
        
        # AUPRC
        auprc_diff, auprc_lo, auprc_hi, auprc_pval = paired_bootstrap(
            t0, p1, p0, average_precision_score, n=N_BOOT, seed=seed
        )
        
        print(f"  AUROC Δ (A1-A0): {auroc_diff:+.4f} [{auroc_lo:+.4f}, {auroc_hi:+.4f}] (p={auroc_pval:.3f})")
        print(f"  AUPRC Δ (A1-A0): {auprc_diff:+.4f} [{auprc_lo:+.4f}, {auprc_hi:+.4f}] (p={auprc_pval:.3f})")
        
        summary.append({
            'seed': seed,
            'metric': 'AUROC',
            'A0_val': roc_auc_score(t0, p0),
            'A1_val': roc_auc_score(t1, p1),
            'delta': auroc_diff,
            'ci_lo': auroc_lo,
            'ci_hi': auroc_hi,
            'p_value': auroc_pval
        })
        summary.append({
            'seed': seed,
            'metric': 'AUPRC',
            'A0_val': average_precision_score(t0, p0),
            'A1_val': average_precision_score(t1, p1),
            'delta': auprc_diff,
            'ci_lo': auprc_lo,
            'ci_hi': auprc_hi,
            'p_value': auprc_pval
        })

    if summary:
        out_csv = os.path.join(RESULTS_DIR, 'paired_bootstrap_results.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
        print(f"\nSaved summary to {out_csv}")
    else:
        print("\nNo seeds successfully analyzed.")

if __name__ == '__main__':
    main()
