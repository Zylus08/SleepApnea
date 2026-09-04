"""
Cross-Cohort Evaluation Pipeline — ICLR Submission
====================================================
Evaluates an OSA detection model (trained on ds008108) against the
Sleep-EDF Expanded dataset as a *domain-shift robustness probe*.

The model was trained for binary sleep apnea classification (OSA vs Control).
Here we repurpose it on a different task (Wake vs Sleep staging) to measure
how well the SSL encoder's representations transfer across cohorts and
task semantics.

Outputs
-------
- cross_cohort_results.npz : raw logits, probabilities, targets, subject IDs
- prediction_distribution.png : logit distribution conditioned on ground truth
- Console: AUROC, AUPRC, Youden-optimal classification report

Usage
-----
    python cross_cohort_eval.py                    # Full 306-subject evaluation
    python cross_cohort_eval.py --max-subjects 5   # Quick sanity check
    python cross_cohort_eval.py --verbose           # Enable per-batch diagnostics
"""

import argparse
import gc
import glob
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Project imports
from downstream_finetune import SleepApneaClassifier
from model import STFTEncoder2D

# Suppress MNE and runtime warnings globally
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DATASET_DIR = r"E:\sleep-edf-database-expanded-1.0.0\sleep-cassette"
TARGET_CHANNEL = "EEG Fpz-Cz"
TARGET_SFREQ = 100
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth"
OUTPUT_DIR = r"E:\SleepApnea\SleepApneaSSL"

# Memory management: flush CUDA cache every N subjects
CUDA_FLUSH_INTERVAL = 10

# Stage Mapping (Sleep-EDF 8-stage → AASM 5-stage)
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


# ==========================================
# 2. FILE PAIRING LOGIC
# ==========================================
def match_sleep_edf_files(data_dir):
    """Discover and pair PSG signal files with their corresponding Hypnogram annotation files."""
    psg_files = glob.glob(
        os.path.join(data_dir, "**", "*PSG.edf"), recursive=True
    ) + glob.glob(os.path.join(data_dir, "**", "*PSG.EDF"), recursive=True)

    hyp_files = glob.glob(
        os.path.join(data_dir, "**", "*Hypnogram.edf"), recursive=True
    ) + glob.glob(os.path.join(data_dir, "**", "*Hypnogram.EDF"), recursive=True)

    print(
        f"[*] Raw search found {len(psg_files)} PSG files and {len(hyp_files)} Hypnogram files."
    )

    pairs = []
    for psg in psg_files:
        prefix = os.path.basename(psg)[:5]
        matching_hyp = [
            h for h in hyp_files if os.path.basename(h).startswith(prefix)
        ]

        if matching_hyp:
            pairs.append((psg, matching_hyp[0]))

    print(f"[*] Successfully paired {len(pairs)} PSG-Hypnogram sets.")
    return pairs


# ==========================================
# 3. DATA EXTRACTION PER SUBJECT
# ==========================================
def extract_epochs(psg_path, hyp_path):
    """
    Extract 30-second EEG epochs from a single PSG/Hypnogram pair.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, 1, 3000) — z-scored EEG signal
    y : np.ndarray, shape (n_epochs,) — AASM stage labels (0–4)
    subject_id : str — filename prefix identifying the subject
    """
    subject_id = os.path.basename(psg_path)[:5]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

        # Channel selection: prefer target, fall back to any EEG channel
        if TARGET_CHANNEL in raw.ch_names:
            raw.pick([TARGET_CHANNEL])
        else:
            eeg_chans = [c for c in raw.ch_names if "EEG" in c]
            if not eeg_chans:
                return None, None, subject_id
            raw.pick([eeg_chans[0]])

        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, verbose=False)

        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations, emit_warning=False)

        events, event_id = mne.events_from_annotations(
            raw, event_id=STAGE_MAP, chunk_duration=30.0, verbose=False
        )

        if len(events) == 0:
            return None, None, subject_id

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=0,
            tmax=30.0 - (1 / TARGET_SFREQ),
            baseline=None,
            preload=True,
            verbose=False,
        )

        X = epochs.get_data(units="uV")

        # Per-subject z-score normalization
        X_mean = np.mean(X)
        X_std = np.std(X)
        X = (X - X_mean) / (X_std + 1e-8)

        y = epochs.events[:, -1]
        return X, y, subject_id

    except Exception as e:
        print(f"[!] Error processing {os.path.basename(psg_path)}: {e}")
        return None, None, subject_id


# ==========================================
# 4. DYNAMIC EVALUATION METRICS
# ==========================================
def compute_youden_threshold(targets, probs):
    """
    Find the optimal decision threshold using Youden's J statistic.

    J = max(TPR - FPR) over all thresholds on the ROC curve.

    Returns
    -------
    threshold : float — optimal decision boundary
    auroc : float — area under the ROC curve
    """
    try:
        auroc = roc_auc_score(targets, probs)
        fpr, tpr, thresholds = roc_curve(targets, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(auroc)
    except ValueError:
        # Happens when only one class is present
        return 0.5, float("nan")


def compute_all_metrics(targets, probs, logits):
    """
    Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    targets : np.ndarray — binary ground-truth labels
    probs : np.ndarray — sigmoid probabilities
    logits : np.ndarray — raw model logits (pre-sigmoid)

    Returns
    -------
    metrics : dict — all computed metrics
    """
    metrics = {}

    # 1. AUROC
    try:
        metrics["auroc"] = roc_auc_score(targets, probs)
    except ValueError:
        metrics["auroc"] = float("nan")

    # 2. Macro AUPRC (Average Precision)
    try:
        metrics["auprc"] = average_precision_score(targets, probs)
    except ValueError:
        metrics["auprc"] = float("nan")

    # 3. Youden's J optimal threshold
    threshold, _ = compute_youden_threshold(targets, probs)
    metrics["youden_threshold"] = threshold

    # 4. Classification at optimal threshold
    preds_optimal = (probs >= threshold).astype(int)
    metrics["macro_f1"] = f1_score(targets, preds_optimal, average="macro", zero_division=0)

    # 5. Class distribution
    metrics["n_wake"] = int(np.sum(targets == 0))
    metrics["n_sleep"] = int(np.sum(targets == 1))
    metrics["prevalence_sleep"] = float(np.mean(targets))

    # 6. Logit statistics (for diagnosing collapse)
    metrics["logit_mean"] = float(np.mean(logits))
    metrics["logit_std"] = float(np.std(logits))
    metrics["logit_min"] = float(np.min(logits))
    metrics["logit_max"] = float(np.max(logits))

    return metrics, preds_optimal


# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_prediction_distribution(targets, logits, probs, output_path):
    """Generate a publication-quality figure showing logit and probability distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Raw logit distribution
    ax = axes[0]
    ax.hist(logits[targets == 0], bins=60, alpha=0.6, label="Wake (0)", color="#4C72B0", density=True)
    ax.hist(logits[targets == 1], bins=60, alpha=0.6, label="Sleep (1)", color="#DD8452", density=True)
    ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.5, label="Logit = 0")
    ax.set_xlabel("Raw Logit Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(A) Model Logit Distribution by Ground Truth", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel B: Sigmoid probability distribution
    ax = axes[1]
    ax.hist(probs[targets == 0], bins=60, alpha=0.6, label="Wake (0)", color="#4C72B0", density=True)
    ax.hist(probs[targets == 1], bins=60, alpha=0.6, label="Sleep (1)", color="#DD8452", density=True)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="p = 0.5")
    ax.set_xlabel("Sigmoid Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(B) Probability Distribution by Ground Truth", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Distribution plot saved to {output_path}")


# ==========================================
# 6. MAIN EVALUATION LOOP
# ==========================================
def run_external_evaluation(model, data_pairs, verbose=False):
    """
    Run inference on all Sleep-EDF subjects and compute cross-cohort metrics.

    Collects raw logits (not binary predictions) and applies dynamic
    thresholding post-hoc for rigorous evaluation.
    """
    model.eval()

    # Accumulators — raw continuous values, NOT binary
    all_logits = []      # Raw model output (pre-sigmoid)
    all_probs = []       # Sigmoid probabilities
    all_targets = []     # Ground-truth AASM labels (0–4)
    all_subject_ids = [] # Subject identifier per epoch

    n_subjects = len(data_pairs)
    n_skipped = 0
    start_time = time.time()

    print(f"[*] Processing {n_subjects} subjects, running inference...")
    for idx, (psg, hyp) in enumerate(tqdm(data_pairs, desc="Subjects")):
        X, y, subject_id = extract_epochs(psg, hyp)
        if X is None or len(X) == 0:
            n_skipped += 1
            continue

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Match dimensions to model input expectations (Batch, Channels=20, Time=3000)
        if X_tensor.ndim == 3 and X_tensor.shape[1] == 1:
            X_tensor = X_tensor.repeat(1, 20, 1)
        elif X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1).repeat(1, 20, 1)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        subject_logits = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(DEVICE)
                outputs = model(batch_x)  # Raw logits, shape: [batch_size]

                if verbose and len(subject_logits) == 0:
                    print(f"  [{subject_id}] batch_x: {batch_x.shape} → outputs: {outputs.shape}")

                # Collect raw logits on CPU immediately to avoid GPU memory accumulation
                subject_logits.append(outputs.cpu().numpy())

                del batch_x, outputs

        subject_logits = np.concatenate(subject_logits)

        if len(subject_logits) != len(y):
            print(
                f"[!] Warning: Mismatch in {subject_id}: "
                f"{len(subject_logits)} preds vs {len(y)} targets"
            )
            n_skipped += 1
            continue

        # Accumulate raw continuous values
        all_logits.append(subject_logits)
        all_probs.append(1.0 / (1.0 + np.exp(-subject_logits)))  # Sigmoid
        all_targets.append(y)
        all_subject_ids.append(np.full(len(y), idx, dtype=np.int32))

        # Memory management: explicit cleanup every N subjects
        del X, y, X_tensor, dataset, loader, subject_logits
        gc.collect()
        if torch.cuda.is_available() and (idx + 1) % CUDA_FLUSH_INTERVAL == 0:
            torch.cuda.empty_cache()

        # ETA estimation
        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (n_subjects - idx - 1)
            print(f"  [{idx+1}/{n_subjects}] Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # ==============================================
    # AGGREGATE AND EVALUATE
    # ==============================================
    if not all_logits:
        print("[!] No evaluation samples gathered.")
        return

    all_logits_arr = np.concatenate(all_logits)
    all_probs_arr = np.concatenate(all_probs)
    all_targets_arr = np.concatenate(all_targets)
    all_subject_ids_arr = np.concatenate(all_subject_ids)

    # Convert multi-class targets to binary: Wake (0) vs Sleep (1–4)
    all_targets_binary = (all_targets_arr > 0).astype(int)

    elapsed_total = time.time() - start_time
    print(f"\n[*] Inference complete. {n_subjects - n_skipped} subjects processed, "
          f"{n_skipped} skipped. Total time: {elapsed_total/60:.1f} minutes.")
    print(f"[*] Total epochs evaluated: {len(all_targets_binary)}")

    # ----- Save raw results for offline analysis -----
    npz_path = os.path.join(OUTPUT_DIR, "cross_cohort_results.npz")
    np.savez(
        npz_path,
        targets=all_targets_binary,
        targets_multiclass=all_targets_arr,
        logits=all_logits_arr,
        probs=all_probs_arr,
        subject_ids=all_subject_ids_arr,
    )
    print(f"[+] Raw results saved to {npz_path}")

    # ----- Compute and display metrics -----
    metrics, preds_optimal = compute_all_metrics(
        all_targets_binary, all_probs_arr, all_logits_arr
    )

    print("\n" + "=" * 60)
    print("  CROSS-COHORT EVALUATION RESULTS (Sleep-EDF)")
    print("  Domain-shift robustness probe: ds008108 -> Sleep-EDF")
    print("=" * 60)
    print(f"  Epochs:  {metrics['n_wake']} Wake + {metrics['n_sleep']} Sleep "
          f"(prevalence: {metrics['prevalence_sleep']:.1%})")
    print(f"  AUROC:                 {metrics['auroc']:.4f}")
    print(f"  AUPRC (Avg Precision): {metrics['auprc']:.4f}")
    print(f"  Youden's J Threshold:  {metrics['youden_threshold']:.4f}")
    print(f"  Macro F1 (at optimal): {metrics['macro_f1']:.4f}")
    print(f"  Logit stats:  mean={metrics['logit_mean']:.4f}  std={metrics['logit_std']:.4f}  "
          f"range=[{metrics['logit_min']:.4f}, {metrics['logit_max']:.4f}]")

    # Collapse warning
    if metrics["logit_std"] < 1e-3:
        print("\n  ! WARNING: Near-zero logit variance detected.")
        print("    The encoder likely suffers from representation collapse on this cohort.")
        print("    Consider layer-wise fine-tuning (see downstream_finetune.py --finetune-mode layerwise).")

    print("\n" + "-" * 60)
    print("  Classification Report (at Youden-optimal threshold)")
    print("-" * 60)
    print(
        classification_report(
            all_targets_binary,
            preds_optimal,
            target_names=["Wake (0)", "Sleep (1)"],
            zero_division=0,
        )
    )

    # ----- Visualization -----
    plot_path = os.path.join(OUTPUT_DIR, "prediction_distribution.png")
    plot_prediction_distribution(all_targets_binary, all_logits_arr, all_probs_arr, plot_path)


# ==========================================
# 7. CLI ENTRY POINT
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-cohort evaluation: ds008108 model → Sleep-EDF dataset"
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="Limit evaluation to N subjects for quick testing (default: all)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable per-batch diagnostic output."
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        help=f"Path to model checkpoint (default: {MODEL_PATH})."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    file_pairs = match_sleep_edf_files(DATASET_DIR)

    if args.max_subjects is not None:
        file_pairs = file_pairs[: args.max_subjects]
        print(f"[*] Limited to {args.max_subjects} subjects for quick evaluation.")

    print("[*] Loading Model...")
    base_encoder = STFTEncoder2D()
    model = SleepApneaClassifier(base_encoder).to(DEVICE)

    state_dict = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[+] Model loaded from {args.model_path}")

    if len(file_pairs) > 0:
        run_external_evaluation(model, file_pairs, verbose=args.verbose)
    else:
        print("[!] No files found. Evaluation stopped.")