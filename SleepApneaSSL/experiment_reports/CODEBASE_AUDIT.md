# Codebase Audit: Pre-Experiment Readiness

**Date:** 2026-09-23
**Auditor:** Automated

This report assesses the codebase's readiness for the next set of reviewer-critical experiments, identifying what already exists, what is missing, and what action is required before proceeding.

---

## 1. Final A0/A1 Statistical Package
**Goal:** Generate authoritative multi-seed A0 vs A1 report (AUROC, AUPRC, mean ± SD, median, deltas, permutation tests, bootstrap CIs).

* **Status:** **PARTIAL / EXISTS (No retraining required)**
* **Relevant Files:** `experiments/multiseed_experiment.py`, `experiments/analyze_seed_Stability.py`, `experiments/paired_permutation_test.py`, `experiments/paired_bootstrap_ci.py`, `experiments/fast_bootstrap_ci.py`.
* **Artifacts Sufficient:** **YES.** The prediction arrays (`A0_VanillaNTXent_predictions.npz` and `A1_TemporalNTXent_predictions.npz`) exist for all 5 seeds. No expensive retraining is needed to generate point estimates, deltas, or permutation tests.
* **Missing Component:** 
  - **Representation effective rank:** Missing. The saved `encoder.pth` files exist for all seeds, but there is currently no script to load them, extract test representations, and compute the effective rank or singular value spectrum.
* **Recommended Next Action:** Write an aggregation script to compile the final statistical report using existing `.npz` files, and write an analysis script to compute representation effective rank from the saved `.pth` files. Do not rerun model training.

---

## 2. SoftCLT Baseline Audit (Highest Priority)
**Goal:** Run a soft contrastive learning temporal baseline.

* **Status:** **MISSING**
* **Relevant Files:** Searched codebase for `SoftCLT`, `soft contrastive`, and `soft_contrastive`. None exist. 
* **Implementation Feasibility:** The exact formulation can easily be implemented using existing infrastructure. `TemporalNTXentLoss` already computes a `(2N, 2N)` pairwise absolute time distance matrix (`time_dist`) and applies subject gating. This exact machinery can be used to generate soft continuous target distributions (e.g., via a Gaussian or exponential kernel over `time_dist`) rather than binary positives/negatives, which can then be passed to a Kullback-Leibler divergence or soft cross-entropy loss against the log-softmaxed similarities.
* **Code Additions Required:** 
  1. A new `SoftCLTLoss` module in `temporal_loss.py`.
  2. A new `SoftCLTWrapper` in `experiments/loss_ablation.py`.
* **Experimental Consistency:** YES. The exact same encoder, dataset splits, masking, pretraining budget, downstream fine-tuning protocol, and seeds can be reused.
* **Estimated Runtime:** ~45-50 minutes per seed (pretraining + fine-tuning). Total runtime for 5 seeds is estimated at ~4 hours on GPU.
* **Recommended Next Action:** Define the exact mathematical formulation of the soft target distribution (e.g., standard deviation for a Gaussian kernel) with the user to prevent ambiguity, then implement the loss module and add it to the multi-seed pipeline.

---

## 3. Temporal Kernel Ablation Audit
**Goal:** Ablate the temporal weighting function in A1 (exponential vs linear vs hard cutoff).

* **Status:** **PARTIAL (Hardcoded)**
* **Relevant Files:** `temporal_loss.py:105`
* **Implementation Feasibility:** The current temporal weight is hardcoded: `w_same = 1.0 - torch.exp(-self.lambda_decay * time_dist)`. It is structurally trivial to parameterize this. We can inject a `kernel_type` string argument to `TemporalNTXentLoss` and branch the weight calculation:
  - Exponential: `1.0 - exp(-λd)`
  - Linear: `min(1.0, αd)`
  - Cutoff: `1.0 if d > cutoff else 0.0`
  - None (A0): `1.0`
* **Experimental Consistency:** YES. Can be run under exactly the same protocol.
* **Recommended Next Action:** Do not run yet. Parameterize the loss class. These require full pipeline runs per variant, so await user approval on specific kernel hyperparameters before initiating training.

---

## 4. 5-Fold Patient-Level CV Audit
**Goal:** Evaluate robustness via 5-fold cross-validation.

* **Status:** **MISSING**
* **Relevant Files:** `experiments/loss_ablation.py` (L515-532)
* **Architecture Limitations:** Currently, the code performs a single fixed split: 60% Train, 20% Val, 20% Test. SSL pretraining runs ONLY on the Train subjects. 
* **Scientific Validity:** To constitute a scientifically valid comparison, the cross-validation loop must wrap the *entire* pipeline (both SSL pretraining and Downstream fine-tuning). If SSL was pretrained on all data and only the downstream classifier was cross-validated, the test folds would suffer from massive data leakage (the SSL encoder would have already seen the test subjects).
* **Cost Implication:** 5 true folds × 5 seeds = 25 full pipeline runs (pretraining + fine-tuning) per method. For A0 and A1 alone, this is 50 runs (~40 hours compute time).
* **Recommended Next Action:** Acknowledge the extreme computational cost. Do not implement a naive CV over the downstream head alone. Determine if true 5-fold CV is strictly necessary for the rebuttal, or if the current 5-seed initialization/batching variance on a fixed holdout is sufficient.

---

## 5. Quantization Reproducibility Audit
**Goal:** Reproduce calibration, INT8, W8A8, and accuracy parity metrics for edge deployment.

* **Status:** **PARTIAL / MISSING CRITICAL EVALUATION**
* **Relevant Files:** `edge_benchmark.py`
* **Existing Components:**
  - INT8 conversion: EXISTS (`quantize_dynamic` from ONNXRuntime is used).
  - Model-size footprint: EXISTS.
  - Latency benchmarking: EXISTS.
* **Missing Components:**
  - Calibration set construction: MISSING. The code uses *dynamic* quantization (`QuantType.QUInt8` on weights only), which scales activations dynamically at runtime and does not require a calibration dataset. 
  - W8A8 configuration: MISSING. (Because it uses dynamic quantization, activations remain in FP32 format and are scaled on the fly, not strictly W8A8 static).
  - Cosine similarity, logit error, flip-rate computation: MISSING. `edge_benchmark.py` runs dummy noise (`torch.randn`) through the network 500 times strictly to measure latency. It never loads the test dataset or computes classification accuracy/AUROC.
* **Scientific Risks:** Any claims in the paper stating that "INT8 quantization does not materially degrade AUROC" are currently unreproducible from the codebase, as the ONNX INT8 model is never actually evaluated on real patient data.
* **Recommended Next Action:** Write an `evaluate_onnx.py` script that loads the test dataset, runs inference using the `sleep_apnea_int8.onnx` model, and computes exact AUROC/AUPRC parity, flip-rates, and logit MSE against the PyTorch FP32 baseline.
