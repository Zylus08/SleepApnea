# SoftCLT Five-Seed Experiment Report

## 1. Aggregate Performance
- **Mean AUROC:** 0.5567 ± 0.1451
- **Median AUROC:** 0.6083
- **Mean AUPRC:** 0.5300 ± 0.1389
- **Median AUPRC:** 0.5927

## 2. Per-Seed Results
| Seed | AUROC | AUPRC |
|---|---|---|
| 42 | 0.5583 | 0.6112 |
| 123 | 0.6083 | 0.5927 |
| 2025 | 0.6750 | 0.6227 |
| 7 | 0.6333 | 0.5343 |
| 13 | 0.3083 | 0.2891 |

## 3. Paired Permutation Tests (AUROC)
| Seed | Δ (SoftCLT - A0) | p-value | Δ (SoftCLT - A1) | p-value | Δ (SoftCLT - A3) | p-value |
|---|---|---|---|---|---|---|
| 42 | 0.0167 | 0.9467 | -0.2083 | 0.3182 | -0.0333 | 0.8943 |
| 123 | -0.1083 | 0.5737 | 0.0167 | 0.9207 | 0.1083 | 0.6485 |
| 2025 | 0.2333 | 0.3303 | 0.3250 | 0.1697 | 0.1000 | 0.6826 |
| 7 | 0.0500 | 0.8312 | -0.1333 | 0.5948 | 0.0167 | 0.9570 |
| 13 | -0.2167 | 0.4110 | -0.2000 | 0.4341 | -0.5000 | 0.0820 |

## 4. Summary of Pairwise Wins
- **SoftCLT > A0 (AUROC):** 3 / 5 seeds
- **SoftCLT > A1 (AUROC):** 2 / 5 seeds
- **SoftCLT > A3 (AUROC):** 3 / 5 seeds

## 5. Scientific Interpretation
1. **Does SoftCLT replicate across seeds?**
   Yes, it replicates successfully across the 5 seeds, though performance has high cross-seed variability (Mean AUROC ~0.5567 ± 0.1451), which matches the variability seen in A0 and A1.
2. **Does SoftCLT consistently outperform A0?**
   No. SoftCLT outperforms A0 in 3/5 seeds.
3. **Does SoftCLT consistently outperform A1?**
   No. SoftCLT outperforms A1 in 2/5 seeds.
4. **Does the result establish specificity for A1?**
   The results suggest neither Temporal NT-Xent (A1) nor SoftCLT yields a consistent downstream improvement over the vanilla baseline (A0). Thus, neither mechanism provides robust gains under this specific linear evaluation protocol.
5. **Is SoftCLT meaningfully different from A1?**
   In downstream performance, SoftCLT does not exhibit a statistically significant and consistent difference from A1. Both struggle to robustly outperform A0 across random seeds.
6. **What claims are NOT supported?**
   We cannot claim that incorporating temporal distance into contrastive learning (either via SoftCLT or A1) robustly improves downstream patient-level classification in this setup.
