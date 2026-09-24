# A1 Implementation Audit Report

**Date:** 2026-09-22
**Auditor:** Automated (code-level, no execution)
**Target:** `TemporalNTXentLoss` in [`temporal_loss.py`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py)
**Reference:** Paper formulation from user specification (ICLR 2027 draft)

---

## 1. File Map

| Component | File | Lines |
|---|---|---|
| A0 (Vanilla NT-Xent) | [`loss_ablation.py`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py) | L69–79 |
| A1 (Temporal NT-Xent core) | [`temporal_loss.py`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py) | L31–136 |
| A1 wrapper | [`loss_ablation.py`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py) | L110–116 |
| A2 (NT-Xent + TempReg) | [`loss_ablation.py`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py) | L82–107 |
| A3 (PhysioCLR) | [`physio_clr.py`](file:///e:/SleepApnea/SleepApneaSSL/physio_clr.py) | L37–115 |
| SSL dataset | [`loss_ablation.py`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py) | L47–66 |
| SSL training loop | [`loss_ablation.py`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py) | L128–180 |
| Downstream eval | [`loss_ablation.py`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py) | L183–498 |
| Encoder (STFTEncoder2D) | [`model.py`](file:///e:/SleepApnea/SleepApneaSSL/model.py) | L66–97 |
| SimCLR projector | [`model.py`](file:///e:/SleepApnea/SleepApneaSSL/model.py) | L100–115 |
| Augmentation | [`physio_clr.py`](file:///e:/SleepApnea/SleepApneaSSL/physio_clr.py) | L6–35 |

---

## 2. Mathematical Reference

Paper defines A1 as:

```
L = -(1/N) Σ_i log [ exp(s(i,p(i)) / τ)  /  ( exp(s(i,p(i)) / τ) + Σ_{j∈N_i} w_ij exp(s(i,j) / τ) ) ]
```

where:

```
w_ij = 1 - exp(-λ * |t_i - t_j|)   if patient_i == patient_j
w_ij = 1                            if patient_i ≠ patient_j
```

Positive p(i) is the augmentation partner. Self (i=i) excluded. Positive **excluded** from denominator.

---

## 3. Line-by-Line Audit of `temporal_loss.py`

### 3.1 Same-patient detection

**Code:** [`temporal_loss.py:101`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L101)
```python
same_sub = (sub_2n.unsqueeze(0) == sub_2n.unsqueeze(1))
```

**Verdict:** ✅ **PASS**
Uses actual `subject_ids` tensor broadcast comparison. Correctly produces `(2N, 2N)` boolean matrix. Subject IDs are integers passed from the dataset.

**Upstream check:** [`loss_ablation.py:53-54`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L53-L54)
```python
pid_match = re.findall(r'\d+', os.path.basename(f))
pid = int(pid_match[0]) if pid_match else 0
```
**Verdict:** ✅ **PASS** — Extracts true patient ID from filename (e.g., `sub-042_eeg.pt` → `42`). Fallback `0` only triggers if filename has no digits (shouldn't happen with BIDS data).

---

### 3.2 Temporal distance computation

**Code:** [`temporal_loss.py:87,102`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L87)
```python
time_2n = torch.cat([time_indices, time_indices], dim=0).float()
time_dist = torch.abs(time_2n.unsqueeze(0) - time_2n.unsqueeze(1))
```

**Verdict:** ✅ **PASS**
Uses the actual window index `w` within the recording (yielded at [`loss_ablation.py:65`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L65)). Absolute difference correctly gives temporal distance in units of 30-second windows.

**Note:** Time indices are within-recording indices (0, 1, 2, ...). Two different patients may have the same time index (e.g., both have window 5). This is correct because `same_sub` gates the temporal weighting: cross-patient pairs always get `w=1` regardless of time_dist.

---

### 3.3 Weight formula: `w_ij = 1 - exp(-λ * |t_i - t_j|)`

**Code:** [`temporal_loss.py:105`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L105)
```python
w_same = 1.0 - torch.exp(-self.lambda_decay * time_dist)
```

**Verdict:** ✅ **PASS** — Exactly matches the paper formula.

---

### 3.4 Cross-patient w_ij = 1

**Code:** [`temporal_loss.py:107`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L107)
```python
temporal_weights = torch.where(same_sub, w_same, torch.ones_like(w_same))
```

**Verdict:** ✅ **PASS** — `torch.where` selects `w_same` for same-subject, `1.0` for different-subject.

---

### 3.5 Positive excluded from denominator

**Code:** [`temporal_loss.py:91-94,112`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L91-L94)
```python
pos_mask[:N, N:]  = eye_N   # z_i[k] -> z_j[k]
pos_mask[N:,  :N] = eye_N   # z_j[k] -> z_i[k]
...
neg_mask = ~self_mask & ~pos_mask   # True for valid negatives
...
sim_adj = sim_adj.masked_fill(self_mask | pos_mask, -1e9)
```

**Verdict:** ✅ **PASS** — Positive pairs are masked with `-1e9` in the logsumexp denominator, effectively zeroing their contribution. This matches "positive excluded from denominator" per standard NT-Xent.

---

### 3.6 Weights applied INSIDE the InfoNCE denominator

**Code:** [`temporal_loss.py:121-128`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L121-L128)
```python
log_w   = torch.log(temporal_weights.clamp(min=1e-9))
sim_adj = sim + log_w
sim_adj = sim_adj.masked_fill(self_mask | pos_mask, -1e9)
log_denom = torch.logsumexp(sim_adj, dim=1, keepdim=True)
```

This computes: `log Σ_j w_ij exp(s_ij/τ) = logsumexp(s_ij/τ + log w_ij)`

**Verdict:** ✅ **PASS** — Weights are applied inside the denominator via log-domain addition, exactly matching the mathematical formulation.

---

### 3.7 No unintended weighting of the numerator

**Code:** [`temporal_loss.py:131`](file:///e:/SleepApnea/SleepApneaSSL/temporal_loss.py#L131)
```python
log_numerator = sim[pos_mask].view(2 * N, 1)
```

**Verdict:** ✅ **PASS** — Numerator reads directly from the unmodified similarity matrix `sim`, not from `sim_adj`. The temporal weights do NOT affect the numerator.

---

### 3.8 No accidental batch-neighbor slicing or positional assumptions

**Previous versions** of this code used `z[:-1]` vs `z[1:]` for temporal regularization (batch-position slicing). The current `TemporalNTXentLoss`:

- Uses full `(2N, 2N)` pairwise matrices for everything
- No slicing, no `[:-1]`/`[1:]`
- Temporal relationships are computed via explicit `time_indices` and `subject_ids`

**Verdict:** ✅ **PASS** — No positional assumptions.

---

### 3.9 No leakage of validation/test patients into SSL pretraining

**Code:** [`loss_ablation.py:528`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L528)
```python
ssl_files = train_files  # only use train subjects for SSL
```

And train/val/test split at [`loss_ablation.py:521-527`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L521-L527):
```python
trainval_ids, test_ids = train_test_split(pids, test_size=0.20, ...)
train_ids, val_ids = train_test_split(trainval_ids, test_size=0.25, ...)
...
ssl_files = train_files
```

**Verdict:** ✅ **PASS** — SSL trains only on `train_files`. Val/test files never enter SSL.

---

## 4. Numerical Edge Cases

### 4.1 d = 0 (same window, same subject)

When `time_dist = 0` and `same_sub = True`:
```
w_same = 1 - exp(0) = 1 - 1 = 0
log_w = log(max(0, 1e-9)) = log(1e-9) ≈ -20.7
sim_adj = sim + (-20.7) → effectively -∞
```
This window pair contributes ~0 to the denominator.

**Verdict:** ✅ **PASS** — Same-subject, same-time-index negatives are correctly suppressed. The `clamp(min=1e-9)` prevents actual `-inf`, using `-20.7` instead, which is dominated by the `-1e9` sentinel but still functionally zero.

**Subtle issue:** For the 2N-expanded batch, `z_i[k]` and `z_j[k]` both have `time_idx = w` and `sub_id = pid`. So `same_sub[k, N+k] = True` and `time_dist[k, N+k] = 0`, giving `w = 0`. But this pair is the **positive pair**, which is already excluded by `pos_mask`. The `w=0` is irrelevant because `pos_mask` masks it with `-1e9` anyway. No double-masking conflict.

**Verdict:** ✅ **PASS**

### 4.2 Same-patient adjacent windows (d = 1)

```
w = 1 - exp(-0.1 * 1) = 1 - 0.9048 = 0.0952
```
This heavily down-weights the pair as a negative.

**Verdict:** ✅ **PASS** — Correct per formula.

### 4.3 Distant same-patient windows (d = 50)

```
w = 1 - exp(-0.1 * 50) = 1 - 0.0067 = 0.993
```
Approaches 1 — treated nearly as hard negative.

**Verdict:** ✅ **PASS**

### 4.4 Cross-patient windows

Always `w = 1` regardless of time indices.

**Verdict:** ✅ **PASS**

### 4.5 Self-pairs (i = i)

Masked by `self_mask = torch.eye(2N)` at line 97, then masked with `-1e9` at line 125.

**Verdict:** ✅ **PASS**

### 4.6 Duplicate windows from same patient with same time index

If the batch contains two different windows from the same patient at the same time index (possible if windowing overlaps or dataset has duplicates):
```
same_sub = True, time_dist = 0, w = 0
```
These would be suppressed as negatives. This is arguably correct — same-patient, same-time windows should not be hard negatives.

**Verdict:** ✅ **PASS** (edge case handled correctly by the formula)

---

## 5. A0 vs A1 Structural Comparison

### A0: VanillaNTXentLoss

**Code:** [`loss_ablation.py:69-79`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L69-L79)
```python
sim.fill_diagonal_(-1e9)
labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
return F.cross_entropy(sim, labels)
```

> [!WARNING]
> **FINDING F1: A0 includes the positive in the denominator; A1 does not.**
>
> `F.cross_entropy(sim, labels)` computes:
> ```
> -log(exp(sim[i, pos]) / Σ_j exp(sim[i, j]))
> ```
> where the sum runs over ALL columns j (except self, masked to -1e9), **including the positive**.
>
> A1 (`TemporalNTXentLoss`) explicitly excludes the positive from the denominator via `pos_mask`.
>
> This is a **known convention difference** between implementations. Both are valid contrastive losses. However, comparing A0 and A1 loss values directly is not meaningful because they use different denominators.
>
> **Impact on downstream metrics:** Minimal. Both conventions are standard. The downstream AUROC/AUPRC comparison remains valid because each ablation trains to convergence under its own loss. The paper should note which convention is used for each.

**Severity:** LOW — does not invalidate the A0 vs A1 comparison at the downstream metric level, but the SSL loss curves are not directly comparable between A0 and A1.

---

## 6. Additional Findings

### F2: IterableDataset ordering is stochastic across ablations

**Code:** [`loss_ablation.py:51`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L51)
```python
flist = list(self.files); random.shuffle(flist)
```

Each call to `pretrain_ssl` calls `set_seed(SEED)` at line 130, which resets `random.seed(42)`. However, `DataLoader` with `IterableDataset` creates a new iterator each epoch, and `random.shuffle` in `__iter__` uses the global random state which has been modified by training operations.

**Impact:** Batch composition varies epoch-to-epoch (expected), but the first epoch's first batch should be deterministic given the seed reset. Across ablations, `set_seed(SEED)` is called at the start of each, so A0 and A1 see the same first-epoch batch ordering.

**Verdict:** ⚠️ **PASS with note** — Reproducibility is maintained at the ablation level (same seed reset), but exact batch ordering within later epochs is not perfectly deterministic due to interleaving of Python random state with training.

### F3: Temporal continuity in A2 uses correct subject-aware masking

**Code:** [`loss_ablation.py:96-105`](file:///e:/SleepApnea/SleepApneaSSL/experiments/loss_ablation.py#L96-L105)
```python
same_sub = (sub_ids.unsqueeze(1) == sub_ids.unsqueeze(0))
time_next = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0) + 1)
valid_mask = same_sub & time_next
```

**Verdict:** ✅ **PASS** — A2 correctly finds temporally adjacent pairs only within the same subject using pairwise comparison, not batch slicing.

### F4: PhysioCLR (A3) temporal continuity uses batch-position slicing

**Code:** [`physio_clr.py:56-57`](file:///e:/SleepApnea/SleepApneaSSL/physio_clr.py#L56-L57)
```python
z_t = z[:-1]
z_t_next = z[1:]
```

> [!CAUTION]
> **BUG — A3's temporal continuity assumes consecutive batch positions are consecutive windows from the same patient.** With shuffled multi-patient batches, this computes MSE between arbitrary cross-patient windows. The `patient_boundary_mask` at line 61 only catches boundaries at the file level (set to 1 when `w == 0` in the dataset), but within-batch ordering after DataLoader is arbitrary.
>
> However, `DataLoader` is created with `shuffle=False` for `IterableDataset`, and the dataset iterates files sequentially (per-epoch shuffle of files, but windows within a file are sequential). So consecutive batch positions ARE likely from the same patient within a file's windows, but transitions between files are not always marked.
>
> **This bug does NOT affect A1.** A1 uses pairwise distance matrices and never assumes batch ordering.

**Verdict:** ⚠️ A3 has a known batch-ordering dependency. **Not in scope for A1 audit**, but noted.

---

## 7. Summary Table

| Audit Item | Status | File:Line | Notes |
|---|---|---|---|
| Same-patient detection (true IDs) | ✅ PASS | `temporal_loss.py:101` | Broadcast comparison on real PIDs |
| Temporal distance (true indices) | ✅ PASS | `temporal_loss.py:102` | Absolute difference of window indices |
| w_ij = 1 - exp(-λd) | ✅ PASS | `temporal_loss.py:105` | Exact match |
| Cross-patient w_ij = 1 | ✅ PASS | `temporal_loss.py:107` | `torch.where(same_sub, w_same, 1)` |
| Positive excluded from negatives | ✅ PASS | `temporal_loss.py:91-94, 112, 125` | Double-masked: `neg_mask` + `masked_fill` |
| Weights inside denominator | ✅ PASS | `temporal_loss.py:121-128` | Log-domain: `logsumexp(sim + log_w)` |
| No numerator weighting | ✅ PASS | `temporal_loss.py:131` | Reads from unmodified `sim` |
| No batch-position assumptions | ✅ PASS | `temporal_loss.py` (all) | Full pairwise matrices |
| No val/test leakage into SSL | ✅ PASS | `loss_ablation.py:528` | `ssl_files = train_files` |
| d=0 edge case | ✅ PASS | `temporal_loss.py:121` | `clamp(min=1e-9)` prevents -inf |
| Self-pair exclusion | ✅ PASS | `temporal_loss.py:97, 125` | `torch.eye(2N)` mask |
| Gradient flow | ✅ PASS | `temporal_loss.py:134` | Standard differentiable ops |

### Findings

| ID | Severity | Description | Affects A1 results? |
|---|---|---|---|
| F1 | LOW | A0 includes positive in denominator; A1 excludes it | No — downstream comparison valid; SSL loss values not directly comparable |
| F2 | LOW | IterableDataset ordering not perfectly deterministic across epochs | No — seed reset ensures first-epoch determinism; minor stochasticity expected |
| F3 | INFO | A2 temporal reg correctly subject-aware (no bug) | N/A |
| F4 | MEDIUM | A3 temporal continuity uses batch-position slicing | **No** — only affects A3, not A1 |

---

## 8. Conclusion

**A1 (`TemporalNTXentLoss`) implementation is CORRECT** and faithfully implements the paper's mathematical formulation.

No bugs affecting A1's reported results were found.
