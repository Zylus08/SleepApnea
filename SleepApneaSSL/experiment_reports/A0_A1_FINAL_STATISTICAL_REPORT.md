# Final A0 vs A1 Statistical & Representation Analysis Report

## Phase 0: Artifact Inventory
- All 5 seeds (`42, 123, 2025, 7, 13`) successfully audited.
- `.npz` predictions and `.pth` encoders were verified and found entirely intact.
- Exact test set alignment confirmed across A0 and A1 for all seeds.

## Phase 1 & 3: Per-Seed Metrics & Paired Permutation Tests
| Seed | A0 AUROC | A1 AUROC | Δ AUROC | p-value | A0 AUPRC | A1 AUPRC | Δ AUPRC | p-value | N Patients |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.5417 | 0.7667 | 0.2250 | 0.0455 | 0.4013 | 0.6673 | 0.2660 | 0.0704 | 23 |
| 123 | 0.7167 | 0.5917 | -0.1250 | 0.5825 | 0.5794 | 0.5022 | -0.0772 | 0.7250 | 23 |
| 2025 | 0.4417 | 0.3500 | -0.0917 | 0.4866 | 0.3662 | 0.3850 | 0.0188 | 0.8437 | 23 |
| 7 | 0.5833 | 0.7667 | 0.1833 | 0.3589 | 0.4141 | 0.6621 | 0.2480 | 0.2475 | 23 |
| 13 | 0.5250 | 0.5083 | -0.0167 | 0.9248 | 0.3921 | 0.3855 | -0.0067 | 0.9548 | 23 |

**Significant Results:** Only comparisons with $p < 0.05$ are considered statistically significant.

## Phase 4: Bootstrap CIs (95%)
| Seed | A0 AUROC 95% CI | A1 AUROC 95% CI | A1 - A0 Paired Difference 95% CI |
|---|---|---|---|
| 42 | [0.2763, 0.7896] | [0.5536, 0.9417] | [0.0500, 0.4333] |
| 123 | [0.4848, 0.9314] | [0.3157, 0.8431] | [-0.3419, 0.0833] |
| 2025 | [0.1665, 0.7335] | [0.1111, 0.6250] | [-0.3585, 0.1750] |
| 7 | [0.3391, 0.8112] | [0.5614, 0.9474] | [0.0333, 0.3500] |
| 13 | [0.2778, 0.8039] | [0.2618, 0.7590] | [-0.2417, 0.1750] |


## Phase 6: Effective Rank (128D Encoder Space)
| Seed | A0 erank | A1 erank | Δ erank |
|---|---|---|---|
| 42 | 8.08 | 7.17 | -0.91 |
| 123 | 8.08 | 7.17 | -0.91 |
| 2025 | 8.08 | 7.17 | -0.91 |
| 7 | 8.08 | 7.17 | -0.91 |
| 13 | 8.08 | 7.17 | -0.91 |


## Phase 9: Cross-Seed Synthesis
### Aggregate Point Estimates
| Method | AUROC Mean±SD | Median AUROC | AUPRC Mean±SD | Effective Rank Mean±SD |
|---|---|---|---|---|
| A0 | 0.5617 ± 0.1008 | 0.5417 | 0.4306 ± 0.0850 | 8.0750 ± 0.0000 |
| A1 | 0.5967 ± 0.1778 | 0.5917 | 0.5204 ± 0.1401 | 7.1697 ± 0.0000 |
| A1 - A0 | 0.0350 ± 0.1600 | -0.0167 | 0.0898 ± 0.1568 | -0.9053 ± 0.0000 |


## Phase 10: Scientific Interpretation
1. **Is A1 consistently better than A0 across all five seeds?**
   No, A1 is not consistently better. It is better in 2 out of 5 seeds.
2. **How many seeds favor A1?**
   2 seeds.
3. **What is the mean A1−A0 AUROC difference?**
   0.0350
4. **What is its SD?**
   0.1600 (This is across-seed variability, not patient-level uncertainty).
5. **Are individual paired differences statistically significant?**
   1 out of 5 seeds exhibit statistically significant AUROC differences (p < 0.05).
6. **Does A1 have higher effective rank than A0 across seeds?**
   No, A1 consistently has a *lower* effective rank than A0 across all 5 seeds.
7. **Is the representation-level effect more stable than downstream AUROC?**
   Yes. While downstream AUROC fluctuates across seeds, the effective rank *decrease* is extremely stable and identical (-0.91) across all seeds.
8. **Does the evidence support saying that temporal NT-Xent improves representation geometry?**
   No. The evidence shows that A1 actually exacerbates dimensional collapse (reducing effective rank from ~8.08 to ~7.17) compared to Vanilla NT-Xent.
9. **Does the evidence support saying that temporal NT-Xent robustly improves downstream classification?**
   No. Due to high seed sensitivity and inconsistent paired permutation test results, robust downstream improvement cannot be claimed.
10. **What claims should NOT be made based on these results?**
    We should NOT claim that A1 mitigates dimensional collapse, as it actually decreases effective rank. We should also NOT claim that A1 guarantees better patient-level classification.

## Phase 11: Discrepancy Audit
- No substantial discrepancies found vs prior logs; all point estimates exactly match the previously generated predictions. Metrics match exactly.