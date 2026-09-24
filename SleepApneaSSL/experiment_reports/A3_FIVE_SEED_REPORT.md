# A3 Five-Seed Multiseed Report (PhysioCLR)

**Experiment:** A3 PhysioCLR (corrected temporal continuity logic) across seeds 42, 123, 2025, 7, and 13.
**Protocol:** Identical to A0/A1 ablation protocol.

---

## 1. Aggregate Results

**A3 (PhysioCLR)**
- **Mean AUROC:** 0.6183 ± 0.1148 (Median: 0.5917)
- **Mean AUPRC:** 0.5474 ± 0.1039 (Median: 0.5608)

**Comparison against Baselines**
- Seeds where A3 > A0 (AUROC): 4 out of 5
- Seeds where A3 > A1 (AUROC): 2 out of 5

---

## 2. Per-Seed Breakdown (Permutation test, n=10,000)

| Seed | A3 AUROC | A3 AUPRC | vs A0 AUROC | vs A1 AUROC | vs A0 AUPRC | vs A1 AUPRC |
|---|---|---|---|---|---|---|
| **42** | 0.5917 | 0.5608 | +0.0500 (p=0.742) | -0.1750 (p=0.201) | +0.1595 (p=0.306) | -0.1065 (p=0.468) |
| **123** | 0.5000 | 0.4015 | -0.2167 (p=0.397) | -0.0917 (p=0.705) | -0.1779 (p=0.463) | -0.1007 (p=0.649) |
| **2025** | 0.5750 | 0.5159 | +0.1333 (p=0.560) | +0.2250 (p=0.320) | +0.1497 (p=0.384) | +0.1309 (p=0.433) |
| **7** | 0.6167 | 0.5691 | +0.0333 (p=0.870) | -0.1500 (p=0.549) | +0.1550 (p=0.396) | -0.0929 (p=0.708) |
| **13** | 0.8083 | 0.6898 | +0.2833 (p=0.115) | +0.3000 (p=0.156) | +0.2977 (p=0.105) | +0.3043 (p=0.206) |

*(Note: No paired permutation test achieved p < 0.05 significance due to small test set size and high variance.)*

---

## 3. Conclusions

**1. Does A3 replicate across seeds?**
**No.** A3's performance exhibits massive variance depending on the random seed. AUROC ranges from random chance (0.5000 in seed 123) to strong performance (0.8083 in seed 13). The standard deviation (±0.1148) is extremely high, indicating that the method (or this specific evaluation protocol) is highly unstable and deeply dependent on initialization and batch ordering.

**2. Does A3 behave similarly to A1?**
**Yes, in its instability.** Like A1, A3 utilizes a temporal continuity mechanism, and like A1, it shows high variance across different seeds (A3 beats A1 in 2/5 seeds, and loses in 3/5). Neither method provides a robust, seed-invariant improvement.

**3. Does A1 appear specifically superior to A3?**
**No.** A1 is not specifically superior. While A1 beats A3 in 3 out of 5 seeds, A3 beats A1 substantially in others (e.g., +0.225 in seed 2025, +0.300 in seed 13). Because neither method consistently outperforms the other and neither achieves statistical significance in paired tests across the multi-seed evaluation, a claim that A1's specific InfoNCE decay mechanism is inherently superior to A3's explicit MSE continuity term is unsupported by this data. 

**4. Or do all temporal formulations exhibit substantial seed sensitivity?**
**Yes.** The data clearly demonstrates that adding temporal constraints (whether via A1's temporal NT-Xent weighting or A3's explicit MSE continuity) introduces massive seed sensitivity. The underlying representations being learned are brittle and highly dependent on the stochasticity of the data loader, initialization, and masking. 

**5. What claim about mechanism specificity is actually supported?**
**None.** The results do not support a claim that the precise mathematical formulation of A1 (Temporal NT-Xent) is uniquely responsible for downstream performance improvements over A3 (PhysioCLR), because the variance *within* a single method across seeds dwarfs the variance *between* the methods. The only scientifically defensible claim is that temporal constraints influence the learned representation, but the current limited-data regime and high variance prevent concluding that one specific temporal mechanism is superior.
