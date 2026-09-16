import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    f1_score,
)

RESULTS_PATH = r"E:\SleepApnea\SleepApneaSSL\cross_cohort_results.npz"

def run_diagnostics():
    if not os.path.exists(RESULTS_PATH):
        print(f"[!] File not found: {RESULTS_PATH}")
        return

    data = np.load(RESULTS_PATH)
    targets = data["targets"]          # 0: Wake, 1: Sleep
    logits = data["logits"]
    probs = data["probs"]
    subject_ids = data["subject_ids"]

    print(f"[*] Loaded {len(targets):,} epochs across {len(np.unique(subject_ids))} subjects.\n")

    # 1. Global Standard vs. Inverted Calibration Check
    auroc_std = roc_auc_score(targets, probs)
    auroc_inv = roc_auc_score(targets, 1.0 - probs)
    auprc_std = average_precision_score(targets, probs)
    auprc_inv = average_precision_score(1 - targets, 1.0 - probs)

    print("==================================================")
    print(" 1. GLOBAL POLARITY & INVERSION DIAGNOSTICS")
    print("==================================================")
    print(f" Standard AUROC  (P=1 as Sleep) : {auroc_std:.4f}")
    print(f" Inverted AUROC  (P=1 as Wake)  : {auroc_inv:.4f}")
    print(f" Standard AUPRC                 : {auprc_std:.4f}")
    print(f" Inverted AUPRC                 : {auprc_inv:.4f}\n")

    # 2. Logit Conditioning Analysis
    wake_logits = logits[targets == 0]
    sleep_logits = logits[targets == 1]

    print("==================================================")
    print(" 2. CLASS LOGIT DISTRIBUTION DETAILED STATS")
    print("==================================================")
    print(f" Wake (0)  -> Mean: {np.mean(wake_logits):.4f} | Std: {np.std(wake_logits):.4f} | Median: {np.median(wake_logits):.4f}")
    print(f" Sleep (1) -> Mean: {np.mean(sleep_logits):.4f} | Std: {np.std(sleep_logits):.4f} | Median: {np.median(sleep_logits):.4f}")
    print(f" Mean Delta (Sleep - Wake): {np.mean(sleep_logits) - np.mean(wake_logits):.4f}\n")

    # 3. Youden-Optimal Report under Inverted Construct
    inv_probs = 1.0 - probs
    # Best threshold for inverted target mapping
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(1 - targets, (inv_probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("==================================================")
    print(f" 3. INVERTED TARGET REPORT (Optimal Threshold = {best_thresh:.4f})")
    print("==================================================")
    preds_inv = (inv_probs >= best_thresh).astype(int)
    print(classification_report(1 - targets, preds_inv, target_names=["Sleep (0)", "Wake (1)"], zero_division=0))

if __name__ == "__main__":
    run_diagnostics()