# Temporal Kernel Ablation Report

## 1. Aggregate Results
| Kernel | AUROC Mean ± SD | Median AUROC | AUPRC Mean ± SD | Median AUPRC |
|---|---|---|---|---|
| exp_lambda0.1 | 0.6133 ± 0.1071 | 0.6500 | 0.5620 ± 0.0668 | 0.5405 |
| linear_alpha0.1 | 0.4900 ± 0.1596 | 0.4917 | 0.4794 ± 0.1580 | 0.4222 |
| cutoff_10.0 | 0.5467 ± 0.1473 | 0.5833 | 0.4542 ± 0.1014 | 0.4199 |

## 2. Per-Seed AUROC
| Seed | Exp | Linear | Cutoff |
|---|---|---|---|
| 42 | 0.7667 | 0.5250 | 0.3750 |
| 123 | 0.6583 | 0.7667 | 0.6750 |
| 2025 | 0.5333 | 0.3167 | 0.3750 |
| 7 | 0.6500 | 0.4917 | 0.7250 |
| 13 | 0.4583 | 0.3500 | 0.5833 |

## 3. Paired Tests against Baseline A0
| Seed | Exp-A0 (p) | Linear-A0 (p) | Cutoff-A0 (p) |
|---|---|---|---|
| 42 | 0.2250 (0.0455) | -0.0167 (0.8679) | -0.1667 (0.5011) |
| 123 | -0.0583 (0.7857) | 0.0500 (0.8269) | -0.0417 (0.8821) |
| 2025 | 0.0917 (0.3926) | -0.1250 (0.2104) | -0.0667 (0.4382) |
| 7 | 0.0667 (0.7618) | -0.0917 (0.6812) | 0.1417 (0.5311) |
| 13 | -0.0667 (0.7543) | -0.1750 (0.5023) | 0.0583 (0.7574) |

## 4. Scientific Answers
1. **Is exponential uniquely useful?**
   The results show large variance between kernels depending on the seed. However, no kernel provides a statistically robust, consistent gain over A0 across all seeds. Exponential is not uniquely capable of solving the core instability.
2. **Do alternative kernels produce similar behavior?**
   No, they exhibit extreme differences per seed. For example, in Seed 42, exponential dramatically outperformed linear and cutoff, while in other seeds the rankings flip. This points to extreme sensitivity to initialization rather than robust geometric priors.
3. **Is A1's behavior robust to the precise temporal weighting function?**
   No. A1's performance collapses or spikes dramatically depending on the specific weighting function. The representations learned are brittle with respect to the exact temporal distance mapping.
4. **Does this experiment support or weaken a mechanism-specific novelty claim?**
   It strictly weakens the mechanism-specific novelty claim. Since performance heavily depends on arbitrary kernel choice and initialization seed, we cannot claim that temporal NT-Xent robustly regularizes the embedding space in a generalizable way.
