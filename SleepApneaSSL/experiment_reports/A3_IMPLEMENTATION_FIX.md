# A3 Implementation Fix Report (PhysioCLR)

**Date:** 2026-09-22
**Auditor:** Automated
**Target:** `PhysioCLRLoss` in `physio_clr.py` and its wrapper in `loss_ablation.py`.
**Reference:** Task to replace batch-position assumptions with explicit subject/time-aware formulation.

---

## 1. Inspection Findings (Task 1)

**How patient IDs and time indices entered the A3 loss:**
Previously, `PhysioCLRLoss.forward` did *not* accept patient IDs (`sub_ids`) or time indices (`time_idx`). It only accepted `patient_boundary_mask`. In `loss_ablation.py`, `PhysioCLRWrapper` completely dropped `sub_ids` and `time_idx`, dynamically building an `is_boundary` mask based on index 0 before calling `self.loss`.

**How the continuity term was constructed:**
`physio_clr.py` contained this logic:
```python
z_t = z[:-1]
z_t_next = z[1:]
```
This hardcoded assumption treated adjacent batch indices as adjacent time windows. While `DataLoader` iterations without shuffling may yield consecutive windows for a single file, cross-file batch boundaries and multi-worker loading break this assumption.

**Equivalence to A2:**
The A2 implementation (`NTXentPlusTempReg` in `loss_ablation.py:96-105`) already solved this correctly by using explicit pairwise tracking:
```python
same_sub = (sub_ids.unsqueeze(1) == sub_ids.unsqueeze(0))
time_next = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0) + 1)
valid_mask = same_sub & time_next
```
We were able to lift this exact, tested logic into `PhysioCLRLoss`.

---

## 2. Exact Files Changed

1. **`e:\SleepApnea\SleepApneaSSL\physio_clr.py`**
   - **Modified:** `PhysioCLRLoss.compute_temporal_loss` (L44-64)
   - Removed batch slicing (`z[:-1]`, `z[1:]`).
   - Replaced with pairwise distance matrix and boolean gating based on `sub_ids` and `time_idx`, directly matching A2's implementation.
   - **Modified:** `PhysioCLRLoss.forward` (L66) to accept `sub_ids` and `time_idx` instead of `patient_boundary_mask`.

2. **`e:\SleepApnea\SleepApneaSSL\experiments\loss_ablation.py`**
   - **Modified:** `PhysioCLRWrapper.forward` (L123-125)
   - Removed logic that fabricated `is_boundary`.
   - Now passes `sub_ids` and `time_idx` straight through to `self.loss`.

---

## 3. Exact Mathematical Definition Now Implemented

The temporal continuity term for a batch of representations $z$ is now defined as:

$$
\mathcal{L}_{temp} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} ||z_i - z_j||_2^2
$$

where $\mathcal{V}$ is the set of valid temporal pairs in the batch, defined as:
$$
(i,j) \in \mathcal{V} \iff \text{patient}(i) = \text{patient}(j) \text{ AND } t_j = t_i + 1
$$

This perfectly replicates the A2 continuity term logic, ensuring only genuinely adjacent windows from the *same* patient contribute to the loss, completely independent of their position within the batch.

---

## 4. Unit Test Results

**File:** `experiment_audit/test_physio_clr_temporal.py`

| Test Case | Status | Notes |
| :--- | :--- | :--- |
| 1. Adjacent same-patient windows selected | ✅ PASS | Verified explicit mask generation `valid_mask[i,j]` |
| 2. Adjacent windows diff patient NOT selected | ✅ PASS | Evaluated mask cross-patient |
| 3. Non-adjacent same-patient NOT selected | ✅ PASS | Evaluated mask for non-adjacent time windows |
| 4. Shuffled batch ordering does not change pairs | ✅ PASS | Loss value invariant to batch index permutation |
| 5. Explicit pairwise MSE matches | ✅ PASS | Value exact match to manual `F.mse_loss` calculation |
| 6. Returns 0 when no valid pairs exist | ✅ PASS | Handled empty case gracefully without NaNs |

---

## 5. Impact on Existing Results

**Are existing A3 seed-42 results invalidated?**
**YES.** The previous A3 ablation results were generated using the batch-slicing continuity term. Because batches often contain cross-patient boundaries (especially with `batch_size=64`), the old implementation erroneously penalized differences between completely unrelated patients, effectively acting as an unintended uniform smoothing term across the batch. 

**Must A3 be completely rerun?**
**YES.** To accurately compare A3 against A0, A1, and A2 in the planned five-seed experiment, A3 must be rerun using the corrected, structurally sound temporal continuity term. The previous A3 results should be discarded.
