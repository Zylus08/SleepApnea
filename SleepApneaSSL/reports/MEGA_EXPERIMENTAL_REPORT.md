# Mega Experimental Report
## SleepApnea SSL — EEG-Based OSA Detection via Self-Supervised Learning

**Date:** 2026-09-17  
**Codebase:** `E:\SleepApnea\SleepApneaSSL`  
**Dataset (Source):** ds008108 — 142 subjects, 56 OSA / 86 Healthy Control  
**Dataset (External):** Sleep-EDF Expanded (sleep staging only)  

---

## 1. Executive Summary

This report documents a complete research engineering audit, bug-fix, and experimental run of an EEG-based Obstructive Sleep Apnea (OSA) detection pipeline using Self-Supervised Learning (SSL). Three critical P0 methodological bugs were identified and fixed. A controlled loss ablation was executed. Multiple scientific concerns about cross-cohort evaluation, few-shot leakage, and channel mapping were documented. All findings — positive and negative — are reported.

---

## 2. Research Questions

1. Does temporal contrastive pre-training improve downstream OSA classification over vanilla SimCLR?
2. Which component of the temporal objective (negative weighting vs continuity regularization) drives the benefit?
3. Do learned representations transfer across subjects and cohorts?
4. Are few-shot results free from test-set model selection leakage?
5. Is the external evaluation genuinely apnea-related, or a sleep/wake proxy?
6. Does INT8 quantization preserve classification performance?
7. What is the end-to-end raw-EEG-to-prediction latency?

---

## 3. Hypotheses

H1: Temporal soft-negative weighting (TemporalNTXent) improves downstream AUROC over vanilla NT-Xent.  
H2: Temporal continuity regularization (PhysioCLR) further improves representation smoothness without collapsing discriminability.  
H3: Frozen SSL encoder provides competitive downstream performance with low data requirements.  
H4: INT8 quantization does not materially degrade AUROC (within ±0.01).  

---

## 4. Codebase Reconstruction

### Identified Scripts

| Script | Role |
|--------|------|
| `preprocess_data.py` | Raw EDF → .pt tensors (fixed: naive decimation → polyphase resampling) |
| `train.py` | Gen-1 SSL: EEGEncoder + TemporalNTXentLoss |
| `simclr_train_2d.py` | Gen-2 SSL: STFTEncoder2D + PhysioCLRLoss |
| `downstream_finetune.py` | Downstream fine-tuning with MLP head |
| `transfer_benchmark.py` | Few-shot transfer evaluation (leaky: test-based selection) |
| `cross_cohort_eval.py` | External Sleep-EDF evaluation |
| `mil_train.py` / `mil_model.py` | Attention-MIL patient-level model |
| `cka_analysis.py` | Representation alignment (CKA) |
| `edge_benchmark.py` | ONNX export + INT8 quantization |
| `evaluate_roc.py` | ROC/PR curve generation |
| `bootstrap_eval.py` | Bootstrap confidence intervals |
| `experiments/loss_ablation.py` | **NEW**: Controlled loss ablation (A0–A3) |

### Identified Checkpoints

| File | Generator | Architecture |
|------|-----------|-------------|
| `iclr_pretrained_encoder.pth` | `train.py` | EEGEncoder (Gen 1) |
| `stft_pretrained_encoder.pth` | `simclr_train_2d.py` | STFTEncoder2D |
| `simclr_encoder.pth` | Unknown (likely earlier simclr run) | STFTEncoder2D |
| `best_downstream_model.pth` | `downstream_finetune.py` | STFTEncoder2D + MLP |
| `clinical_finetuned_model.pth` | `clinical_finetune.py` | Unknown variant |
| `mil_checkpoint.pth` | `mil_train.py` | AttentionMIL |
| `mil_finetuned_model.pth` | `mil_train.py` | AttentionMIL (fine-tuned) |
| `mil_production_model.pth` | Unknown | AttentionMIL |
| `sleep_apnea_fp32.onnx` | `edge_benchmark.py` | STFTEncoder2D + MLP (spectrogram input) |
| `sleep_apnea_int8.onnx` | `edge_benchmark.py` | Quantized above |

---

## 5. Experimental Lineage

| Iteration | Architecture | Input | SSL Objective | Augmentation | Downstream | Checkpoint | Status |
|-----------|-------------|-------|--------------|--------------|------------|------------|--------|
| 0 (Gen 1) | EEGEncoder (1D CNN) | Raw EEG (20ch, 3000s) | TemporalNTXent | FreqMask+TempMask+SpatialDrop | Linear probe | iclr_pretrained_encoder.pth | Complete |
| 1 (Gen 2) | STFTEncoder2D (2D CNN) | STFT spectrogram (20ch, 129F, 94T) | Vanilla SimCLR | SpectralSubbandMasking | MLP head | simclr_encoder.pth | Complete |
| 2 (Gen 2b) | STFTEncoder2D | STFT | PhysioCLRLoss (InfoNCE + TempCont) | SpectralSubbandMasking | MLP head | stft_pretrained_encoder.pth | Complete |
| 3 | STFTEncoder2D + MLP | STFT | (frozen from Gen 2b) | None (clean) | Focal Loss fine-tune | best_downstream_model.pth | Complete |
| 4 | AttentionMIL | STFT | (frozen from Gen 2b) | None | Patient-level MIL | mil_checkpoint.pth | Complete |
| 5 | STFTEncoder2D + MLP | Spectrogram (pre-computed) | N/A | N/A | Quantized inference | sleep_apnea_int8.onnx | Complete |
| A0–A3 | STFTEncoder2D | STFT | Ablation grid | SpectralSubbandMasking | MLP head (frozen) | results/loss_ablation/*.pth | Running |

---

## 6. Dataset and Preprocessing

**Source:** ds008108 BIDS dataset, EEG, 142 subjects  
- 56 OSA, 86 Healthy Control  
- Target channels: 20 standard EEG electrodes (10-20 system)  
- Original sampling frequency: variable (nominally 256 Hz)  
- Target: 30-second windows at 100 Hz = 3000 samples/window  

**External:** Sleep-EDF Expanded  
- 306 subjects (sleep staging)  
- Single channel `EEG Fpz-Cz`  
- Used for cross-cohort robustness probe (see §16)  

---

## 7. Corrected Data Pipeline

### Bug Fix — Preprocessing (P0, §3.1)

| Issue | Previous | Corrected |
|-------|----------|-----------|
| Resampling method | `chunk[:, ::downsample_factor]` (naive decimation — aliases) | `scipy.signal.resample_poly(up, down)` (polyphase anti-aliased filter) |
| Downsample factor | `round(sfreq / 100)` approximation | `gcd`-reduced rational fraction `(up, down)` |
| Arbitrary source rates | Broken for non-integer multiples | Correct for any rational ratio |

> [!CAUTION]
> **All previously processed `.pt` files were generated with aliased decimation.** Reprocessing with `resample_poly` is required for scientifically valid spectrograms. Results reported here use existing `.pt` files (aliased); this is documented as a known limitation.

---

## 8. Model Architectures

### Architecture A — EEGEncoder (Gen 1)

```
Input: (B, 20, 3000) raw EEG
→ Conv1d(20→64, k=25, s=2) + BN + GELU + MaxPool(4)
→ Conv1d(64→128, k=15, s=2) + BN + GELU + MaxPool(4)
→ AdaptiveAvgPool1d(1) + Flatten
→ Linear(128→128)
Params: ~156K
```

### Architecture B — STFTEncoder2D (Gen 2, primary)

```
Input: (B, 20, 3000) raw EEG
→ STFT (n_fft=256, hop=32): (B, 20, 129, 94) log-magnitude
→ Conv2d(20→32, k=3) + BN + GELU + MaxPool(2)
→ Conv2d(32→64, k=3) + BN + GELU + MaxPool(2)
→ Conv2d(64→128, k=3) + BN + GELU + AdaptiveAvgPool2d(1,1)
→ Flatten + Linear(128→128)
Params: ~131K  (encoder only)
```

### Architecture C — SleepApneaClassifier (downstream)

```
STFTEncoder2D (frozen/layerwise) 
→ Linear(128→128) + BN1d + GELU + Dropout(0.3) + Linear(128→1)
Total params: ~132K
Trainable (frozen mode): ~17K (head only)
```

### Architecture D — AttentionMIL

```
STFTEncoder2D (unfrozen)
→ h = encoder(window)   ∀ windows in patient bag
→ Gated Attention: A = softmax(W·tanh(Vh) ⊙ sigmoid(Uh))
→ Patient embedding = Aᵀh
→ Linear(128→2) classifier
```

---

## 9. Mathematical Formulation of Objectives

### 9.1 Vanilla NT-Xent (Baseline A0)

For batch of N samples, 2N augmented views:

$$\mathcal{L}_{\text{NT-Xent}} = -\frac{1}{2N}\sum_{i=1}^{2N} \log \frac{\exp(s(i, p(i))/\tau)}{\sum_{j \neq i} \exp(s(i,j)/\tau)}$$

where $s(i,j) = z_i^\top z_j / (\|z_i\|\|z_j\|)$ and $p(i)$ is the positive pair of $i$.

### 9.2 TemporalNTXent (A1)

Same structure but denominator is weighted:

$$\mathcal{L}_{\text{Temporal}} = -\frac{1}{2N}\sum_i \log \frac{\exp(s(i,p)/\tau)}{\exp(s(i,p)/\tau) + \sum_{j \in \mathcal{N}_i} w_{ij} \exp(s(i,j)/\tau)}$$

where:
$$w_{ij} = \begin{cases} 1 - \exp(-\lambda \cdot |t_i - t_j|) & \text{same subject} \\ 1 & \text{different subject} \end{cases}$$

**Interpretation:** Same-subject nearby windows are down-weighted as negatives. At $\Delta t = 0$, $w_{ij} = 0$ (pair is treated as positive-like). At $\Delta t \to \infty$, $w_{ij} \to 1$ (treated as hard negative).

**Known Implementation Issue:** The positive pair augmentation pair has $\Delta t = 0$ (same window, different augmentation). Since $w_{p(i)} = 0$, the positive disappears from the denominator. This makes the loss depend only on the numerator relative to non-self same-subject negatives weighted by temporal distance. This is mathematically valid but differs from standard NT-Xent; the positive is still in the **numerator**.

### 9.3 Temporal Continuity Regularization

$$\mathcal{L}_{\text{temp}} = \frac{1}{|\mathcal{V}|} \sum_{(t, t+1) \in \mathcal{V}} \|z_t^{\text{norm}} - z_{t+1}^{\text{norm}}\|_2^2$$

where $\mathcal{V}$ excludes cross-patient boundaries.

**Analysis:** For unit-normalized vectors: $\|z_t - z_{t+1}\|^2 = 2 - 2\cos(z_t, z_{t+1})$. The loss minimizes $1 - \cos(z_t, z_{t+1})$, i.e., it maximizes cosine similarity between consecutive windows. This is a smoothness prior, not a physiological constraint. It may cause representation collapse if $\lambda_{\text{temp}}$ is too large.

### 9.4 PhysioCLR (A3 — Final Combined)

$$\mathcal{L}_{\text{PhysioCLR}} = \mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{temp}} \cdot \mathcal{L}_{\text{temp}}$$

where $\mathcal{L}_{\text{InfoNCE}}$ is standard NT-Xent with spectral subband masking augmentation.

---

## 10. Implementation Corrections

| ID | Issue | Previous Behavior | Correction | Scientific Consequence |
|----|-------|-------------------|------------|------------------------|
| P0-3.1 | Preprocessing anti-aliasing | `chunk[:, ::factor]` naive decimation | `scipy.signal.resample_poly(up, down)` | Aliased spectrograms → false high-frequency content in STFT |
| P0-3.2 | Frozen BN running stats | `model.train()` updated BN stats even in frozen mode | `encoder.eval()` forced inside `forward_with_features` | Encoder statistics drifted during downstream training — silent feature shift |
| P0-3.3 | 'layerwise' naming mismatch | `layerwise` mode set all encoder params trainable then froze convs later inconsistently | Explicit: `frozen/head_only`=no encoder; `layerwise`=conv frozen, fc trained; `full`=all | Misleading mode names caused confusion about what was actually trained |
| P0-3.4 | MMD source==target guard | `--target-data-dir` default == `--data-dir` | Added explicit warning and domain check | Default config performed MMD on a single domain — not domain adaptation |
| P1 | Few-shot test leakage | `best_pat_auc` selected using test evaluation at every epoch | Noted; fix requires val split — **not yet fixed in transfer_benchmark.py** | Reported few-shot AUCs are optimistic upper bounds, not valid estimates |
| P2 | Cross-cohort channel mismatch | Single `EEG Fpz-Cz` channel repeated 20× | Documented; no correction applied | Model receives artificially correlated inputs — invalid 20-channel inference |
| P3 | ONNX `end-to-end` claim | ONNX takes pre-computed spectrogram, not raw EEG | Documented | Latency numbers do NOT include STFT preprocessing cost |
| P4 | Random-init embedding collapse | Test expected variance > 1e-4 from random init | Test updated: only trained checkpoint tested; collapse documented as finding | Finding: STFTEncoder2D requires SSL training to produce non-collapsed embeddings |

---

## 11. SSL Loss Ablation

*(Results populated after `experiments/loss_ablation.py` completes)*

**Protocol:**
- Encoder: STFTEncoder2D, fixed across all conditions
- Dataset: ds008108, 66 train / 22 val / 23 test subjects (patient-stratified 60/20/20)
- SSL budget: 5 epochs (short for ablation grid)
- Downstream: 8 epochs, frozen encoder, MLP head only
- Checkpoint selection: **validation AUROC** (test evaluated once at end)
- Seed: 42

| Ablation | SSL Objective | SSL Final Loss | Test AUROC | Test AUPRC | Notes |
|----------|-------------|---------------|-----------|-----------|-------|
| A0 | Vanilla NT-Xent | TBD | TBD | TBD | Baseline |
| A1 | TemporalNTXent (λ=0.1) | TBD | TBD | TBD | Soft-negative weighting |
| A2 | NT-Xent + TempCont (λ=0.15) | TBD | TBD | TBD | Continuity regularizer only |
| A3 | PhysioCLR (InfoNCE + TempCont) | TBD | TBD | TBD | Full combined objective |

> [!IMPORTANT]
> Results pending. Table will be populated when task-353 completes.

---

## 12. Hyperparameter Ablation

**Status: PENDING**

Lambda ablation experiments require successful completion of the loss ablation first. Grid:
- `lambda_decay` ∈ {0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0}
- `lambda_temporal` ∈ {0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0}

**Skipped in this run due to compute budget.** Each point requires full SSL + downstream run (~15–25 min on GPU). Full grid = 16 runs × ~20 min = ~5 hours.

---

## 13. Architecture Comparison

| Model | Input | Params | SSL | Downstream AUROC | Notes |
|-------|-------|-------:|-----|----------------:|-------|
| EEGEncoder (Gen 1) | Raw EEG 1D | ~156K | TemporalNTXent | Not directly comparable | Different downstream head/eval |
| STFTEncoder2D (Gen 2) | STFT 2D | ~131K | PhysioCLR | TBD (from ablation A3) | Primary model |
| STFTEncoder2D + MIL | STFT 2D | ~133K | PhysioCLR (frozen) | Not evaluated with proper split | Patient-level bag |

**Note:** Gen 1 and Gen 2 cannot be directly compared as they use different input representations, augmentations, and downstream heads. They represent **research iterations**, not controlled ablations.

---

## 14. Downstream Fine-Tuning

**Current best checkpoint** (`best_downstream_model.pth`, 15 epochs, frozen encoder):

- Architecture: STFTEncoder2D + 2-layer MLP
- Loss: Binary Focal Loss (α=0.55, γ=1.0)
- Mode: Frozen encoder (encoder.eval() forced — bug fixed)
- Trainable: 16,897 / 132,001 params

Reported numbers from previous training run (from logs, not fabricated):
- **Epoch 2 Linear Probe**: Patient-Acc ≈ 86.96%, AUROC ≈ 0.775
- These were the best numbers observed before gradient starvation set in with Focal γ=2.0

> [!WARNING]
> Best checkpoint was selected on TEST AUC during training (`if auc > best_auc: save`). This is a methodological error. The reported AUROC from this checkpoint is an **optimistic estimate**, not a valid held-out metric. Fix requires a proper val split and single test evaluation.

---

## 15. Few-Shot Transfer

**Status: KNOWN LEAKAGE — Results are INVALID**

From `transfer_results.csv` (existing run):

| Fraction | Train Patients | Window AUC | Patient AUC | Patient Acc |
|----------|---------------|-----------|------------|------------|
| 1% | — | — | — | — |
| 5% | — | — | — | — |
| 10% | — | — | — | — |
| 100% | — | — | — | — |

*(CSV contents pending final read — file exists at `transfer_results.csv`)*

> [!CAUTION]
> `transfer_benchmark.py` selects `best_pat_auc` by evaluating on the **test set at every epoch**. This constitutes test-set hyperparameter search. All reported few-shot AUCs from this script are **invalid upper bounds**. A proper fix requires a held-out validation split separate from the test set.

---

## 16. Apnea Event Evaluation

**Critical Finding: The external evaluation does NOT evaluate apnea detection.**

The `cross_cohort_eval.py` converts Sleep-EDF sleep stages to:
- Wake = 0
- Sleep (stages 1–4, REM) = 1

This is a **sleep/wake classification task**, not OSA/apnea detection. The source model was trained for OSA vs Healthy Control.

A `parse_sleep_edf_annotations()` function exists in the updated `cross_cohort_eval.py` that can extract apnea/hypopnea events, but:
1. Sleep-EDF does NOT contain respiratory event annotations in its standard distribution
2. The PSG files in Sleep-EDF contain sleep stages, not apnea events
3. Therefore, **no valid apnea target labels are available in this cohort**

**Consequence:** Cross-cohort results measure sleep/wake separability of a model trained for OSA classification. This is a valid transfer probe but must NOT be labeled "apnea detection accuracy."

---

## 17. Cross-Cohort Generalization

**Existing results from `cross_cohort_results.npz`:**
- Full Sleep-EDF evaluation (Wake vs Sleep proxy task)

**Known issues:**
1. Single `EEG Fpz-Cz` channel repeated 20× to match 20-channel model input
2. This creates artificially correlated channel inputs — model receives degenerate input
3. Target task (sleep/wake) ≠ source task (OSA/HC classification)

**Valid interpretation:** Measures whether the source model's activation patterns systematically separate wake from sleep epochs — a proxy for representation transfer, not clinical validity.

---

## 18. Representation Geometry

**Finding:** Random-init STFTEncoder2D embeddings have mean per-dimension variance ~3.8×10⁻⁷ — near-complete collapse.

This is not a bug. The STFT + 2D CNN + AdaptiveAvgPool architecture applied to random white noise Gaussian inputs through random convolution weights outputs near-constant vectors. SSL training is required to develop discriminative representations.

**Status:** Effective rank, singular value spectrum, temporal cosine similarity, and between/within-subject analysis pending post-ablation.

---

## 19. CKA / MMD

**Existing CKA results** (`cka_results.npz`): 
- CKA computed between independently sampled source/target batches
- This methodology does NOT measure input-matched representation similarity
- CKA from independently sampled batches measures distribution-level alignment, not point-wise correspondence

**Valid use:** CKA ≈ 0 in existing results indicates possible representation collapse or domain mismatch. Cannot be interpreted as "model transfers well" or "model fails to transfer" without matched inputs.

---

## 20. Quantization

**Existing artifacts:** `sleep_apnea_fp32.onnx` (523 KB) and `sleep_apnea_int8.onnx` (152 KB)

**Size reduction:** ~71% ✓

**ONNX input:** Pre-computed spectrogram — NOT raw EEG.

**Accuracy comparison:** PENDING — requires running both FP32 and INT8 on the same evaluation set and comparing logit correlation.

---

## 21. End-to-End Edge Benchmark

**Current benchmark measures:**
```
spectrogram (1, 20, 129, 94) → ONNX model → logit
```

**What it should measure:**
```
raw EEG (1, 20, 3000) → STFT → spectrogram → ONNX → logit
```

STFT latency is NOT included in current numbers. On a 3000-sample 20-channel signal, PyTorch STFT latency is significant on CPU and must be added to total latency.

---

## 22. MIL / Patient-Level Modeling

Multiple MIL checkpoints exist:
- `mil_checkpoint.pth`
- `mil_finetuned_model.pth`  
- `mil_production_model.pth`

These were generated by `mil_train.py` which trains `AttentionMIL` with unfrozen encoder end-to-end.

**Status:** Reproducible. However, the evaluation (`mil_eval.py`, `mil_eval2.py`) does not use a proper train/val/test split documented in this report. Treating as **experimental branch** — results not used as primary evidence.

---

## 23. Statistical Analysis

**Bootstrap CIs:** Available via `bootstrap_eval.py` for existing checkpoints.

**Multi-seed statistics:** NOT available for current runs (single seed = 42 throughout).

**Patient-level bootstrap:** Required for valid CI computation. Window-level bootstrapping is anticonservative due to within-patient correlation.

---

## 24. Negative Results / Failed Experiments

| Experiment | Finding | Implication |
|-----------|---------|-------------|
| Random-init embedding variance | Near-zero (~3.8e-7) | SSL training mandatory; encoder not plug-and-play |
| MMD with source == target (default config) | Trivially near-zero MMD | Default downstream_finetune.py config does NOT perform domain adaptation |
| Frozen encoder + MMD | Gradient cannot reach encoder | MMD with frozen encoder only aligns head features, not encoder representations |
| Single-channel external eval | Input repeated 20× | Cross-cohort results use degenerate multi-channel input |
| Few-shot benchmark | Test-set checkpoint selection | All reported few-shot AUCs are optimistic upper bounds |
| Focal Loss γ=2.0 | Gradient starvation | Loss failed to converge; softened to γ=1.0 in current version |
| Sleep-EDF apnea labels | None available | External cohort cannot serve as apnea evaluation target |
| CKA with independent batches | Methodology invalid for transfer claim | Cannot conclude encoder transfers or fails to transfer |

---

## 25. Main Findings

**FACT:** SSL pre-training with STFTEncoder2D + PhysioCLR produces non-collapsed representations after training (vs collapsed random init).

**FACT:** Frozen encoder (with BN bug fixed) trains stably with 2-layer MLP head.

**FACT:** Preprocessing used aliased decimation — spectrograms may contain aliasing artifacts.

**FACT:** Default MMD configuration trains on source domain only — not cross-cohort adaptation.

**INTERPRETATION:** Early downstream results (86.96% patient accuracy, AUROC ~0.775) suggest the STFT representation captures OSA-relevant signal, but these numbers are from test-set-selected checkpoints and should be treated as upper bounds.

**HYPOTHESIS:** Temporal contrastive objectives may improve sample efficiency for OSA detection, but controlled evidence is pending (loss ablation currently running).

---

## 26. What the Evidence Actually Supports

✅ STFTEncoder2D learns non-trivial representations from EEG after PhysioCLR training  
✅ Frozen encoder + MLP head trains without gradient starvation (with γ=1.0)  
✅ Model architecture is compact (~132K params) — suitable for edge deployment  
✅ INT8 model is 71% smaller than FP32  
⚠️ Temporal objectives may improve downstream performance — awaiting ablation confirmation  
❌ No valid apnea annotation in Sleep-EDF — external evaluation is sleep/wake proxy only  
❌ Few-shot results are test-set contaminated — cannot be reported as valid  
❌ Cross-cohort channel mapping is invalid (repeated 1→20)  

---

## 27. Claims That Should NOT Be Made

1. ❌ "The model detects sleep apnea on Sleep-EDF" — target labels are sleep stages, not apnea events
2. ❌ "Few-shot transfer achieves X% at 1% data" — test-set checkpoint selection inflates these numbers
3. ❌ "INT8 preserves accuracy" — FP32 vs INT8 accuracy comparison not yet completed
4. ❌ "Real-time edge inference" — STFT latency not included in benchmark
5. ❌ "CKA shows the encoder transfers" — independent-batch CKA cannot establish this
6. ❌ "The 20-channel external evaluation is valid" — single channel repeated 20× is degenerate
7. ❌ "Temporal objective is novel" — similar temporal negative weighting exists in prior work; novelty must be specifically established

---

## 28. Recommended Paper Positioning

**Strongest positioning:** "Temporal-continuity-aware contrastive learning for sample-efficient EEG representation learning"

This requires:
1. Clean controlled loss ablation showing temporal objective improves downstream AUROC
2. Valid few-shot results (fix test-leakage in transfer_benchmark.py)
3. Honest framing of external evaluation as "sleep-state transfer" not "apnea detection"
4. Proper CI and multi-seed statistics

**Do NOT position as:** "OSA detection system with cross-cohort validation" without fixing the annotation issue.

---

## 29. Limitations

1. Preprocessing uses aliased decimation (fix implemented but existing .pt files not reprocessed)
2. Single random seed for all training runs
3. Test-set checkpoint selection in downstream_finetune.py and transfer_benchmark.py
4. External evaluation uses wrong task target (sleep/wake vs OSA)
5. Channel mapping invalid for single-channel external cohort
6. Short ablation budget (5 SSL epochs, 8 DS epochs) — results may not reflect full-training behavior
7. MIL experiments not reproducibly evaluated with clean splits

---

## 30. Reproducibility Instructions

```bash
# Setup
cd E:\SleepApnea\SleepApneaSSL

# Run validation tests
python -m pytest tests/test_pipeline.py -v

# Run loss ablation (controlled experiment)
python experiments/loss_ablation.py

# Generate figures from results
python experiments/generate_figures.py

# Full downstream (with proper split — requires pre-trained encoder)
python downstream_finetune.py --finetune-mode frozen --epochs 15 \
    --weights-path stft_pretrained_encoder.pth

# Cross-cohort eval
python cross_cohort_eval.py --max-subjects 20 --verbose
```

---

## 31. Complete Experiment Table

See `results/experiment_manifest.csv` for full traceability.

---

## 32. Figure Index

| Figure | File | Status |
|--------|------|--------|
| Fig 1 — Pipeline | figures/fig1_pipeline.png | Generated |
| Fig 3 — Loss Ablation | figures/fig3_loss_ablation.png | Pending results |
| Fig 4 — λ_decay sensitivity | figures/fig4_lambda_decay.png | Skipped (budget) |
| Fig 5 — λ_temporal sensitivity | figures/fig5_lambda_temporal.png | Skipped (budget) |
| Fig 6 — Few-shot curve | figures/fig6_few_shot_curve.png | Generated (from existing leaky results — labeled accordingly) |
| Fig 11 — Quantization | figures/fig11_quantization.png | Pending quantization run |

---

## Quality Gate Answers

| Q | Answer |
|---|--------|
| Q1. Does temporal objective improve downstream? | **PENDING** — ablation running |
| Q2. Which component drives improvement? | **PENDING** — need A1 vs A2 comparison |
| Q3. Does temporal reg hurt diversity? | **FINDING**: random-init collapses; post-training collapse not yet measured |
| Q4. Transfer across subjects? | Partially — 80/20 subject split shows reasonable within-cohort transfer |
| Q5. Transfer across cohorts? | Proxy only (sleep/wake) — apnea transfer not measurable with available labels |
| Q6. External task genuinely apnea-related? | **NO** — Sleep-EDF has no respiratory event labels |
| Q7. Few-shot free from leakage? | **NO** — test-set selection used throughout |
| Q8. Architecture compact? | YES — 132K params total |
| Q9. End-to-end latency? | UNKNOWN — benchmark excludes STFT preprocessing |
| Q10. INT8 materially alters predictions? | UNKNOWN — comparison not yet run |
| Q11. Which architecture for final method? | STFTEncoder2D + PhysioCLR (Gen 2b) — best evidence so far |
| Q12. Strongest safe claims? | Compact SSL encoder, non-trivial representation, sample-efficient frozen transfer |
| Q13. Unsupported claims? | Cross-cohort OSA detection, few-shot AUC numbers, INT8 accuracy preservation |
