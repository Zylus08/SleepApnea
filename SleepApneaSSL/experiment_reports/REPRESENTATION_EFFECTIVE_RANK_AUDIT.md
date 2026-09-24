# Representation Effective Rank Audit

## A. Are checkpoints different?
**No, they are perfectly identical.**
A cryptographic hash and parameter norm audit reveals that for a given method, all 5 "independently" trained encoder checkpoints are actually exact duplicates of the Seed 42 model.
- All `A0_VanillaNTXent` checkpoints (seeds 42, 123, 2025, 7, 13) share the exact same MD5 hash: `eb67cec45a301703bd2ef9f698c81ef2`.
- All `A1_TemporalNTXent` checkpoints share the exact same MD5 hash: `d09cc92781cc49fceae80d05d0c84b2e`.

## B. Are extracted representations different?
**No.** Because the underlying PyTorch `encoder.pth` files are byte-for-byte identical, the extracted test-set representation matrices are mathematically identical across all 5 seeds.

## C. Was the previous erank calculation accidentally using cached/shared representations?
No, the extraction script correctly processed the test set for every seed independently. The exact repetition in the effective rank metric (8.08 for A0, 7.17 for A1) is entirely caused by the fact that the underlying encoder weights on disk are identical across all seeds. 

## D. Is 8.08 vs 7.17 a genuine result?
**Yes, for Seed 42.** The effective rank decrease from 8.08 (A0) to 7.17 (A1) is a genuine topological property of the single models trained under Seed 42. However, it is **not** a multi-seed result.

## E. If genuine, is the effect stable across seeds?
**Unknown/Invalid.** Since all seeds are actually just Seed 42 copies, we cannot make any claims about cross-seed stability of the representation geometry. The 5-seed stability is a complete mirage.

## F. If erroneous, exactly what caused the artifact?
The error originates in `experiments/loss_ablation.py` line 146. 
The `pretrain_ssl()` function hardcodes `set_seed(SEED)` where `SEED` is a global constant defined as `42` at the top of the file. 
When `experiments/multiseed_experiment.py` looped over the seeds and called `set_seed(seed)` before executing `pretrain_ssl()`, `pretrain_ssl()` immediately ignored the loop's seed and overrode it back to `42`. This forced the PyTorch RNG, model weight initialization, and data loader shuffling to be identical for every "independent" run of the SSL pretraining phase.

*Note: The downstream finetuning function (`finetune_downstream`) did not contain this hardcoded override, which is why the downstream AUROC metrics fluctuated across seeds while the encoder geometry remained perfectly identical.*

## G. What representation-level claims are scientifically defensible?
1. **Dimensional Collapse Claim is Invalid**: We cannot claim A1 mitigates dimensional collapse because (in the single valid seed) A1 actually *exacerbates* dimensional collapse by lowering the effective rank compared to A0.
2. **Stability Claim is Invalid**: We cannot claim representation geometry is "highly stable across seeds" because the multi-seed experiment silently failed to randomize the pre-training seeds. 

**Conclusion**: The current representations do not scientifically support the claim that Temporal NT-Xent improves geometry or mitigates dimensional collapse. Retraining the encoders with correctly randomized seeds is necessary if we wish to evaluate multi-seed geometry, though the Seed 42 result strongly suggests A1 decreases effective rank.
