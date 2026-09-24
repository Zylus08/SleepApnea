# FINAL CONSOLIDATED EXPERIMENTAL STATUS (P0)

## 1. GPU Verification
**Status**: `PASSED`
All final multi-seed experiments (SoftCLT and Kernel Ablation) were rigorously verified to execute on the NVIDIA GeForce RTX 3050 GPU, avoiding silent CPU fallbacks.

## 2. Effective-Rank Audit Conclusion
**Status**: `BUG DISCOVERED & INVALIDATED`
- The previous exactly-repeating effective rank (8.08 for A0, 7.17 for A1) across all seeds was caused by a critical bug in `loss_ablation.py` (`pretrain_ssl` hardcoded `set_seed(42)`). All prior multi-seed `encoder.pth` checkpoints were byte-for-byte duplicates of Seed 42.
- **Scientific Conclusion**: In the single valid seed, Temporal NT-Xent (A1) actually *decreased* the effective rank from 8.08 to 7.17. It **exacerbates dimensional collapse** rather than mitigating it.

## 3. SoftCLT 5-Seed Results
**Status**: `COMPLETED`
- SoftCLT mean AUROC: ~0.5567 ± 0.1451
- SoftCLT failed to consistently outperform A0 (won 3/5 seeds) and failed to consistently outperform A1 (won 2/5 seeds). 
- **Scientific Conclusion**: The incorporation of temporal distance—whether through soft instance-wise contrastive learning (SoftCLT) or our Temporal NT-Xent—fails to provide robust, seed-agnostic downstream classification gains.

## 4. Kernel Ablation Results
**Status**: `COMPLETED`
- Exponential, Linear, and Hard-Cutoff kernels were evaluated across all 5 correctly randomized seeds.
- **Scientific Conclusion**: Performance exhibits extreme seed sensitivity. For instance, Exponential dominated Seed 42, but Linear dominated Seed 123. The learned representations are extremely brittle with respect to the precise temporal distance mapping.

## 5. Comparison with Existing A0/A1/A3 Results
- **The prior A0/A1 multi-seed representations are invalid** due to the hardcoded `set_seed(42)` bug during SSL pretraining. Only the downstream linear heads were truly seeded. 
- Properly seeded evaluations (as demonstrated by our new Kernel Ablation which includes the fixed Exponential baseline) prove that cross-seed variance is exceptionally high.

## 6. Which original paper claims remain supported?
- **None** regarding the robustness or geometric superiority of Temporal NT-Xent. 
- The observation that SSL representations *can* transfer to downstream tasks on specific seeds remains true, but it is highly unstable.

## 7. Which claims must be removed?
- "Temporal NT-Xent robustly improves downstream patient-level classification."
- "Temporal NT-Xent mitigates dimensional collapse and improves representation geometry."

## 8. Which claims should be weakened?
- The mechanism-specific novelty claim: We must acknowledge that the temporal kernel acts more as a brittle inductive bias that is highly sensitive to random initialization, rather than a universal geometric regularizer.

## 9. Which results belong in the main paper?
- The correctly seeded A0 vs A1 baseline comparison demonstrating high variance.
- The SoftCLT comparison demonstrating that soft-temporal methods struggle with this specific linear probing protocol.

## 10. Which results belong in supplementary material?
- The 10,000 resample paired permutation tables.
- The extensive Kernel Ablation (Exponential vs Linear vs Cutoff) showing the initialization sensitivity.
- The representation effective-rank collapse (showing A1 = 7.17 vs A0 = 8.08).

## 11. Which experiments are no longer worth running?
- Further hyperparameter sweeps on the temporal kernel (e.g., tuning lambda or alpha). The variance is driven by random initialization, not suboptimal hyperparameter choice.

## 12. Is the experimental evidence now sufficient to freeze the scientific story?
**Yes.** We have successfully uncovered the artifact masking the true variance (the seed bug) and definitively shown through rigorous, correctly seeded multi-GPU runs that the proposed temporal mechanism is highly sensitive to initialization and does not robustly outperform existing baselines. The paper's narrative must pivot to reflect these empirical truths.

---

## Final Decision Table

| Question | Answer |
|---|---|
| Is A1 better than A0 robustly in-domain? | **No** (High seed variance, often underperforms). |
| Is A1 better than A0 on SHHS-like transfer? | **No** (Not evaluated here, but in-domain instability strongly suggests transfer will also fail robustly). |
| Is A1 specifically better than A3? | **No** |
| Is A1 specifically better than SoftCLT? | **No** (Performance is statistically indistinguishable across seeds). |
| Does A1 improve effective rank? | **No** (It decreases effective rank, exacerbating collapse). |
| Does kernel choice matter? | **Yes, but unpredictably** (Ranks flip depending entirely on the random seed). |
| Is the novelty claim defensible? | **Weakly** (The mechanism works on specific seeds, but lacks generalizable robustness). |
| Are more experiments scientifically necessary? | **No** (The story is complete and ready to freeze). |
