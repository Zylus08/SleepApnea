# SoftCLT Implementation Compatibility Audit

**Date:** 2026-09-23
**Phase:** Phase 1

---

## 1. Required Tensors

| Tensor | SoftCLT Needs | Source in Our Pipeline |
|---|---|---|
| z_i (projections, aug 1) | (N, D) | `SimCLR.forward` → z1 |
| z_j (projections, aug 2) | (N, D) | `SimCLR.forward` → z2 |
| raw EEG windows (for distance) | (N, C, T) = (N, 20, 3000) | x from DataLoader |
| sub_ids | (N,) | SSLStreamingDataset yield |
| time_idx | (N,) | SSLStreamingDataset yield |

**Sub_ids / time_idx:** Used by A1. SoftCLT does NOT need them. But they are yielded anyway (no code change required to the data pipeline).

**Raw EEG windows (x):** SoftCLT needs the pre-STFT raw windows to compute data-space distances. These are currently available in the training loop at `x = x.to(device)` (loss_ablation.py:L145) BEFORE the STFT is applied. However, the current loss function signatures do NOT receive `x` — they only receive `z1, z2, sub_ids, time_idx, is_boundary`.

**Required change to the training loop:** Pass `x_raw` to the SoftCLT loss function. This is a targeted, contained change to the training loop ONLY.

---

## 2. Dataset Semantics — No Changes Required

- `SSLStreamingDataset` yields `(x, sub_ids, time_idx, is_boundary)`. No change needed.
- Patient ID extraction uses `re.findall(r'\d+', os.path.basename(f))` from filenames. No change needed.
- Time indices are per-file window indices (0, 1, 2, ...). No change needed.
- Train/val/test patient split uses `random_state=42` throughout. No change needed.

---

## 3. Reuse of A1 Infrastructure

| Component | Reuse? | Notes |
|---|---|---|
| `SSLStreamingDataset` | YES | No change |
| `STFTEncoder2D` | YES | No change |
| `SimCLR` (projector) | YES | No change |
| `SpectralSubbandMasking` | YES | No change |
| `pretrain_ssl` loop | YES (with x_raw pass) | Minor change: pass `x` to loss |
| `finetune_downstream` | YES | No change |
| Patient-level prediction aggregation | YES | No change |
| `set_seed` | YES | No change |
| Optimizer (AdamW), LR, epochs, batch size | YES | No change |

---

## 4. Affected Files (Minimal)

1. **`temporal_loss.py`** — Add `SoftCLTLoss` class
2. **`experiments/loss_ablation.py`** — Add `SoftCLTWrapper` (with x_raw handling); minor change to `pretrain_ssl` to pass `x` raw to wrapper

---

## 5. No Interference with A0/A1/A3

SoftCLT does NOT touch:
- `VanillaNTXentLoss`
- `TemporalNTXentLoss`
- `PhysioCLRLoss`
- Any result directories for A0/A1/A3
