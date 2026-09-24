# SoftCLT Formulation Audit
## Lee et al., ICLR 2024 — "Soft Contrastive Learning for Time Series"

**Date:** 2026-09-23
**Source:** arXiv:2312.16424v4; Official code at github.com/seunghan96/softclt

---

## A. Original SoftCLT Equations

SoftCLT defines two complementary soft contrastive objectives:

### A1. Soft Instance-Wise Contrastive Loss

**Soft assignment between time-series instances i and i':**

```
w_I(i, i') = α * σ(-(D(x_i, x_i') - 1) / τ_I)     if i ≠ i'  (Eq. 1)
w_I(i, i') = 1                                        if i = i'  (positive pair)
```

where:
- D(·,·) is a min-max normalized distance metric (DTW is default)
- σ(a) = 1 / (1 + exp(-a))  is the sigmoid function
- τ_I is sharpness hyperparameter
- α ∈ [0,1] is upper bound to distinguish same-TS from close-but-different TS
  - α=0.5 is the paper default

**Softmax probability for anchor i at timestamp t:**

```
p_I(i,t) = exp(r_{i,t} ∘ r̃_{j,t}) / Σ_{j} exp(r_{i,t} ∘ r_{j,t})    (Eq. 2)
```

**Soft instance-wise loss for anchor i at timestamp t:**

```
ℓ_I(i,t) = -log p_I(i,t) - Σ_{i' ≠ i} w_I(i,i') * log p_I(i',t)      (Eq. 3)
```

### A2. Soft Temporal Contrastive Loss

**Soft assignment between timestamps t and t':**

```
w_T(t, t') = σ(-(|t - t'| - 1) / τ_T)                                   (Eq. 4)
```

where τ_T controls sharpness (paper default: τ_T=2, adjusted at depth k by τ_T=m^k * τ̃_T).

**Softmax probability for instance i at timestamp t:**

```
p_T(i,t) = exp(r_{i,t} ∘ r̃_{i,t}) / Σ_{t'} exp(r_{i,t} ∘ r_{i,t'})   (Eq. 5)
```

**Soft temporal loss for instance i at timestamp t:**

```
ℓ_T(i,t) = -log p_T(i,t) - Σ_{t' ≠ t} w_T(t,t') * log p_T(i,t')        (Eq. 6)
```

### A3. Total SoftCLT Loss

```
L = L_I + λ * L_T    (Eq. 7)
```

where λ=0.5 is the paper default, and L_I and L_T are the mean of ℓ_I and ℓ_T over all anchors/timestamps.

The loss is hierarchically computed: representations are max-pooled along the temporal axis at each depth level d, with τ_T adjusted by m^d where m=2 (kernel size of pooling).

---

## B. Definitions of Every Term

| Symbol | Description | Paper Default |
|---|---|---|
| x_i | Time series instance i | — |
| r_{i,t} | Embedding of x_i at timestamp t | — |
| r̃_{i,t} = r_{i+N,t} | Embedding of augmented view of x_i | — |
| D(·,·) | Min-max normalized instance distance (DTW) | DTW |
| τ_I | Sharpness for instance assignment | tuned per dataset |
| τ_T | Sharpness for temporal assignment | 2 |
| α | Upper bound for instance assignment | 0.5 |
| λ | Trade-off between instance and temporal loss | 0.5 |
| m | Max-pool kernel size for hierarchical loss | 2 |
| k | Depth level in hierarchical loss | — |

---

## C. Exact Implementation Mapping to Our Tensors

### Critical structural difference vs. paper

The original SoftCLT was designed for TS2Vec, which:
1. Operates at the **timestamp level within a single long time series** (e.g., T=10000 timestamps)
2. Computes **temporal contrastive loss across timestamps** within one recording
3. Has a hierarchical architecture with dilated convolutions + max-pooling

Our codebase operates at the **window level**:
1. Each batch element is a **pre-segmented 30-second EEG window**
2. The encoder produces a **single fixed-length embedding per window** (no temporal dimension in output)
3. There is no hierarchical pooling architecture

### Consequence

**Soft Temporal CL (Section 3.3 of the paper) CANNOT be directly applied to our architecture.**

The temporal CL in SoftCLT assigns weights based on proximity of *timestamps within a single time series instance*, iterating over all t, t' pairs within r_i. Our encoder outputs r ∈ R^D (not R^(T×M)), so there is no intra-instance temporal axis to contrast.

**Soft Instance-Wise CL (Section 3.2 of the paper) CAN be adapted**, where:
- "instances" are 30-second EEG windows
- "distance between instances" in the data space requires a choice: Euclidean distance or DTW on raw EEG windows
- soft assignments are based on inter-window similarity

### Mapping for Instance-Wise Component Only

| SoftCLT term | Our pipeline |
|---|---|
| x_i | EEG window tensor, shape (20, 3000) |
| D(x_i, x_i') | Euclidean (or DTW) distance between windows |
| r_{i,t} → r_i | Global projection z_i ∈ R^64 (single embed per window) |
| r̃_{i,t} → z̃_i | Projection of augmented view z̃_i ∈ R^64 |
| w_I(i, i') | Soft assignment = α * σ(-(D_norm(i,i') - 1) / τ_I) |
| L_I | Soft instance-wise InfoNCE across batch |

### Key Reference Code (`soft_losses.py:temp_CL_soft`)

The reference `temp_CL_soft` function (for temporal contrastive loss) uses:
- `z = torch.cat([z1, z2], dim=1)` → concatenating **along the time axis**
- Shape is `(B, 2T, C)` where T is the temporal dimension

This confirms the temporal CL operates on per-timestamp embeddings, not per-window embeddings.

---

## D. Difference Between SoftCLT and A1 (Temporal NT-Xent)

| Dimension | SoftCLT | A1 (Temporal NT-Xent) |
|---|---|---|
| **Unit of contrast** | Timestamps within a long TS (for temporal CL) OR instances (for instance CL) | Windows (fixed 30s segments) |
| **Temporal similarity** | Sigmoid of timestamp difference within ONE time series | Sigmoid of window index difference within ONE patient's recording |
| **Subject/patient structure** | Not modelled — instances within a batch are all considered | Explicitly separates same-patient vs cross-patient pairs |
| **Positive pair definition** | Augmentation-based (same augmented view) | Augmentation-based (same augmented window) |
| **Negative weighting** | w_I based on data-space distance OR w_T based on timestamp proximity | w based on temporal distance, gated by same-patient identity |
| **Positive in denominator** | YES (via `F.log_softmax` without masking the positive) | NO (positive masked out of denominator) |
| **Hierarchical computation** | YES (repeated max-pooling) | NO |
| **Instance distance** | DTW / Euclidean on raw data | NOT used — only temporal indices |

### Critical Distinction:
A1 (Temporal NT-Xent) explicitly models the **patient (subject) identity** in the negative weighting: same-patient pairs get down-weighted, cross-patient pairs always get w=1. SoftCLT does **not** model subject identity — it only models:
- (Instance CL) Raw data-space proximity between windows
- (Temporal CL) Timestamp proximity within a single time series

This is a **genuine structural difference**, not a superficial one.

---

## E. Ambiguities and Open Questions

### E1. Temporal CL is inapplicable to our architecture. What is the right adaptation?

**Option A (Pure Instance CL only):** Implement SoftCLT instance-wise loss only (w_I based on Euclidean distance between pre-computed EEG windows). This is the cleanest faithful adaptation.

**Option B (Temporal as inter-window CL):** Reinterpret "temporal CL" as contrasting across windows from different time positions, using window indices as the temporal coordinate. This is what A1 does — making "SoftCLT-temporal" essentially isomorphic to A1's structure, but using sigmoid(|Δt|-1)/τ instead of 1-exp(-λ|Δt|), and NOT gating by patient identity.

**Scientific verdict:** Only Option A constitutes a faithful SoftCLT implementation for our setting.

### E2. Distance metric for instance-wise CL

DTW on raw EEG windows (20 channels × 3000 samples) would be extremely expensive. FastDTW or Euclidean distance must be used. This is explicitly permitted by the paper (Table 5d shows COS, EUC, DTW, TAM all perform similarly, and the paper states results are robust to distance metric choice).

**Chosen default:** Euclidean distance (L2) on flattened raw EEG windows, min-max normalized per batch.

### E3. Hyperparameters τ_I and α not specified in the paper for non-UCR datasets

The paper states: "As the degree of closeness between timestamps varies across datasets, we tune τ_T to control the degree of soft assignments." For instance-wise CL, τ_I is also dataset-specific.

**Default adopted from paper:** τ_I = 2, α = 0.5 (from Table 5c, best result).

**Scientific risk:** τ_I is NOT tuned on our test set. It will be set to the paper's default of τ_I=2 throughout. This is the only scientifically defensible choice.

---

## F. Assumptions for EEG Window Setting

1. **No hierarchical computation** — our encoder is a single forward pass (STFTEncoder2D + projection), not a hierarchical temporal encoder. We do not apply the hierarchical max-pooling from TS2Vec/SoftCLT.

2. **Temporal CL component is dropped** — cannot be applied to our architecture without fundamentally changing the encoder. We implement only the instance-wise soft contrastive loss.

3. **Distance metric** — Euclidean L2 distance between raw EEG windows (20, 3000) computed per-batch, min-max normalized. This matches the paper's fallback distance metrics.

4. **Positive pair** — the augmented pair (SpectralSubbandMasking view pair), identical to A0/A1/A3.

5. **Patient/subject structure** — NOT incorporated into SoftCLT assignment weights. This is intentional — SoftCLT does not model subjects. Our A1 does. This is the primary controlled difference.

6. **Positive in denominator** — SoftCLT's reference code (`soft_losses.py`) does NOT mask the positive from the denominator (it uses `F.log_softmax` over all j). We preserve this convention exactly. This means SoftCLT shares the A0 denominator convention, differing from A1.

---

## G. Summary Verdict

**CAN implement faithfully:** Soft Instance-Wise Contrastive Loss from SoftCLT, applied at the window level, with Euclidean distance between raw EEG windows, τ_I=2, α=0.5.

**CANNOT implement faithfully without architecture change:** Soft Temporal Contrastive Loss (requires per-timestamp embeddings from a hierarchical encoder like TS2Vec's dilated CNN).

**Implemented SoftCLT = Soft Instance-Wise CL only.** This is the correct, maximally faithful adaptation. The paper's ablation (Table 5a) shows that the instance-wise component alone provides a meaningful gain over the hard baseline.

This adaptation is sufficient to answer the reviewer question: does weighting negatives by data-space similarity (SoftCLT's approach) provide similar or greater benefit than weighting same-patient negatives by temporal proximity (A1's approach)?
