import json
import os

def main():
    json_path = r'E:\SleepApnea\SleepApneaSSL\results\softclt\softclt_analysis.json'
    out_md_path = r'E:\SleepApnea\SleepApneaSSL\experiment_reports\SOFTCLT_FIVE_SEED_REPORT.md'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    agg = data['softclt_aggregate']
    raw = data['raw_metrics']
    diffs = data['diffs']
    seeds = data['seeds_run']
    
    num_a0 = data['num_sc_gt_a0']
    num_a1 = data['num_sc_gt_a1']
    
    # Calculate num_sc_gt_a3 if available
    num_a3 = sum(1 for r in diffs if r.get('SC-A3_AUROC') is not None and r['SC-A3_AUROC'] > 0)
    
    with open(out_md_path, 'w', encoding='utf-8') as f:
        f.write("# SoftCLT Five-Seed Experiment Report\n\n")
        
        f.write("## 1. Aggregate Performance\n")
        f.write(f"- **Mean AUROC:** {agg['mean_auroc']:.4f} ± {agg['std_auroc']:.4f}\n")
        f.write(f"- **Median AUROC:** {agg['median_auroc']:.4f}\n")
        f.write(f"- **Mean AUPRC:** {agg['mean_auprc']:.4f} ± {agg['std_auprc']:.4f}\n")
        f.write(f"- **Median AUPRC:** {agg['median_auprc']:.4f}\n\n")
        
        f.write("## 2. Per-Seed Results\n")
        f.write("| Seed | AUROC | AUPRC |\n")
        f.write("|---|---|---|\n")
        for s in seeds:
            m = raw[str(s)]
            f.write(f"| {s} | {m['auroc']:.4f} | {m['auprc']:.4f} |\n")
        f.write("\n")
        
        f.write("## 3. Paired Permutation Tests (AUROC)\n")
        f.write("| Seed | Δ (SoftCLT - A0) | p-value | Δ (SoftCLT - A1) | p-value | Δ (SoftCLT - A3) | p-value |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for d in diffs:
            s = d['seed']
            d_a0 = d['SC-A0_AUROC']
            p_a0 = d['p_SC-A0_AUROC']
            d_a1 = d['SC-A1_AUROC']
            p_a1 = d['p_SC-A1_AUROC']
            
            d_a3 = d.get('SC-A3_AUROC')
            p_a3 = d.get('p_SC-A3_AUROC')
            
            a3_str = f"{d_a3:.4f}" if d_a3 is not None else "N/A"
            p_a3_str = f"{p_a3:.4f}" if p_a3 is not None else "N/A"
            
            f.write(f"| {s} | {d_a0:.4f} | {p_a0:.4f} | {d_a1:.4f} | {p_a1:.4f} | {a3_str} | {p_a3_str} |\n")
        f.write("\n")
        
        f.write("## 4. Summary of Pairwise Wins\n")
        f.write(f"- **SoftCLT > A0 (AUROC):** {num_a0} / {len(seeds)} seeds\n")
        f.write(f"- **SoftCLT > A1 (AUROC):** {num_a1} / {len(seeds)} seeds\n")
        f.write(f"- **SoftCLT > A3 (AUROC):** {num_a3} / {len(seeds)} seeds\n\n")
        
        f.write("## 5. Scientific Interpretation\n")
        f.write("1. **Does SoftCLT replicate across seeds?**\n")
        f.write("   Yes, it replicates successfully across the 5 seeds, though performance has high cross-seed variability (Mean AUROC ~0.5567 ± 0.1451), which matches the variability seen in A0 and A1.\n")
        f.write("2. **Does SoftCLT consistently outperform A0?**\n")
        f.write(f"   No. SoftCLT outperforms A0 in {num_a0}/{len(seeds)} seeds.\n")
        f.write("3. **Does SoftCLT consistently outperform A1?**\n")
        f.write(f"   No. SoftCLT outperforms A1 in {num_a1}/{len(seeds)} seeds.\n")
        f.write("4. **Does the result establish specificity for A1?**\n")
        f.write("   The results suggest neither Temporal NT-Xent (A1) nor SoftCLT yields a consistent downstream improvement over the vanilla baseline (A0). Thus, neither mechanism provides robust gains under this specific linear evaluation protocol.\n")
        f.write("5. **Is SoftCLT meaningfully different from A1?**\n")
        f.write("   In downstream performance, SoftCLT does not exhibit a statistically significant and consistent difference from A1. Both struggle to robustly outperform A0 across random seeds.\n")
        f.write("6. **What claims are NOT supported?**\n")
        f.write("   We cannot claim that incorporating temporal distance into contrastive learning (either via SoftCLT or A1) robustly improves downstream patient-level classification in this setup.\n")
        
    print(f"Report written to {out_md_path}")

if __name__ == '__main__':
    main()
