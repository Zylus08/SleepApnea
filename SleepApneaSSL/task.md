# Task Tracker — ICLR Cross-Cohort Pipeline Refactor

## Component 1: Cross-Cohort Evaluation Pipeline
- [x] Refactor inference loop to collect raw logits + sigmoid probs
- [x] Remove hardcoded binary thresholding in inference
- [x] Save raw outputs (logits, probs, subject_ids) to `.npz`
- [x] Implement AUROC, Macro AUPRC, Youden's J dynamic metrics
- [x] Remove debug diagnostics from inference loop
- [x] Scale to full dataset (remove `[:2]` slice)
- [x] Add memory-efficient inference (gc, cuda empty cache)
- [x] Add `--max-subjects` CLI flag for quick testing

## Component 2: Downstream Fine-Tuning Improvements
- [x] Add `set_finetune_mode()` to `SleepApneaClassifier`
- [x] Implement layer-wise discriminative learning rates
- [x] Add `WeightedBCEWithLogits` as alternative loss
- [x] Wire finetune mode selection in `main()`

## Component 3: Bootstrap Confidence Intervals
- [x] Create `bootstrap_eval.py` with stratified bootstrap resampling
- [x] Compute AUROC/AUPRC per bootstrap iteration
- [x] Report 95% CI (mean, 2.5th, 97.5th percentile)
- [x] LaTeX table row output

## Component 4: CKA Representation Geometry
- [x] Create `cka_analysis.py` with linear CKA implementation
- [x] Layer-wise feature extraction (post-conv1, post-conv2, post-FC)
- [x] MMD with RBF kernel as complementary metric
- [x] Heatmap visualization

## Component 5: Cleanup
- [x] Delete `test.py` and `test2.py`

## Verification
- [x] Syntax check all modified/new files
- [x] Dry-run cross-cohort eval on 2 subjects
