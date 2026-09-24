# SoftCLT Protocol Audit

**Date:** 2026-09-23
**Phase:** Phase 3 — Pre-training protocol verification

---

## Verification Table

| Component | A0 | A1 | SoftCLT | A3 | Consistent? |
|---|---|---|---|---|---|
| Dataset | ds008108 | ds008108 | ds008108 | ds008108 | YES |
| Encoder (STFTEncoder2D, embed_dim=128) | YES | YES | YES | YES | YES |
| Projection head (SimCLR, dim=64) | YES | YES | YES | YES | YES |
| Augmentation (SpectralSubbandMasking p=0.5) | YES | YES | YES | YES | YES |
| Optimizer (AdamW, lr=1e-3, wd=1e-4) | YES | YES | YES | YES | YES |
| SSL epochs | 5 | 5 | 5 | 5 | YES |
| Downstream epochs | 8 | 8 | 8 | 8 | YES |
| Batch size | 64 | 64 | 64 | 64 | YES |
| Patient-level train/val/test split | 60/20/20, `random_state=42` | Same | Same | Same | YES |
| SSL trains on train-split subjects only | YES | YES | YES | YES | YES |
| Downstream fine-tunes on train-split subjects | YES | YES | YES | YES | YES |
| Validation for checkpoint selection | YES | YES | YES | YES | YES |
| Patient-level prediction aggregation | YES | YES | YES | YES | YES |

## SHHS / External Data Confirmation

SoftCLT uses ONLY:
- Raw EEG windows from ds008108 train-split subjects
- Pairwise Euclidean distances computed **within each training batch** (no offline cache, no external data)

SoftCLT does NOT receive:
- SHHS data: NO
- SHHS labels: NO
- Any target-domain calibration data: NO
- Test-subject information during SSL pretraining: NO

## Hyperparameters (SoftCLT-specific)

| Hyperparameter | Value | Source |
|---|---|---|
| τ (contrastive temperature) | 0.5 | Consistent with A0/A1/A2 |
| τ_I (soft assignment sharpness) | 2.0 | Paper default (App. C) |
| α (upper bound for soft weights) | 0.5 | Paper Table 5c best result |
| Distance metric | Euclidean L2 (min-max normalized) | Paper Table 5d shows metrics equivalent |
| λ (instance vs temporal trade-off) | N/A — temporal component not implemented | Architectural incompatibility |

**None of these hyperparameters were tuned against the test set.** τ_I=2 and α=0.5 are taken directly from the paper's published best values.

## Protocol Audit PASSED

All conditions verified. Proceed to Phase 4 pilot run (seed=42).
