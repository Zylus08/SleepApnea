"""
Bootstrap Confidence Interval Estimation — ICLR Submission
===========================================================
Computes 95% CIs for AUROC and AUPRC via stratified bootstrap
resampling from saved cross-cohort evaluation results.

Supports two modes:
    1. Single-seed: Bootstrap resample within a single .npz file.
    2. Multi-seed:  Aggregate across multiple .npz files from
                    independent training runs (different random seeds).

Usage
-----
    # Single-seed bootstrap (1000 iterations)
    python bootstrap_eval.py --input cross_cohort_results.npz

    # Multi-seed aggregate (5 training runs)
    python bootstrap_eval.py --input-dir results/ --seeds 5

    # LaTeX table output
    python bootstrap_eval.py --input cross_cohort_results.npz --latex

Output
------
    - Console:  metric mean [95% CI lower, upper]
    - Optional: LaTeX-formatted table row for direct paper inclusion
"""

import argparse
import glob
import os
import sys

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)


# ==========================================
# 1. CORE BOOTSTRAP ENGINE
# ==========================================
def stratified_bootstrap_ci(
    targets: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
):
    """
    Compute a confidence interval for a given metric via stratified
    bootstrap resampling.

    Stratification ensures each bootstrap sample preserves the class
    ratio of the original dataset, preventing degenerate samples where
    only one class is present (which would make AUROC undefined).

    Parameters
    ----------
    targets : np.ndarray
        Binary ground-truth labels (0 or 1).
    scores : np.ndarray
        Continuous predictions (probabilities or logits).
    metric_fn : callable
        Function(targets, scores) -> float.
    n_iterations : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    point_estimate : float
        Metric computed on the full dataset.
    ci_lower : float
        Lower bound of the CI.
    ci_upper : float
        Upper bound of the CI.
    bootstrap_distribution : np.ndarray
        All bootstrap metric values (for histogram plotting).
    """
    rng = np.random.RandomState(random_state)

    # Indices for each class
    idx_pos = np.where(targets == 1)[0]
    idx_neg = np.where(targets == 0)[0]

    # Point estimate on the full dataset
    point_estimate = metric_fn(targets, scores)

    # Bootstrap resampling
    bootstrap_values = np.zeros(n_iterations)
    n_failures = 0

    for i in range(n_iterations):
        # Stratified resample: sample with replacement within each class
        boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        boot_targets = targets[boot_idx]
        boot_scores = scores[boot_idx]

        try:
            bootstrap_values[i] = metric_fn(boot_targets, boot_scores)
        except ValueError:
            # Edge case: degenerate sample despite stratification
            bootstrap_values[i] = np.nan
            n_failures += 1

    if n_failures > 0:
        print(f"  [!] {n_failures}/{n_iterations} bootstrap iterations failed (degenerate samples).")

    # Remove NaN values before computing percentiles
    valid_values = bootstrap_values[~np.isnan(bootstrap_values)]

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(valid_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid_values, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper, bootstrap_values


# ==========================================
# 2. METRIC FUNCTIONS
# ==========================================
def auroc_metric(targets, scores):
    """Area Under the Receiver Operating Characteristic Curve."""
    return roc_auc_score(targets, scores)


def auprc_metric(targets, scores):
    """Area Under the Precision-Recall Curve (Average Precision)."""
    return average_precision_score(targets, scores)


def youden_f1_metric(targets, scores):
    """Macro F1 at the Youden's J optimal threshold."""
    fpr, tpr, thresholds = roc_curve(targets, scores)
    best_idx = np.argmax(tpr - fpr)
    threshold = thresholds[best_idx]
    preds = (scores >= threshold).astype(int)
    return f1_score(targets, preds, average="macro", zero_division=0)


# ==========================================
# 3. SINGLE-FILE EVALUATION
# ==========================================
def evaluate_single_file(npz_path: str, n_iterations: int = 1000, verbose: bool = True):
    """
    Load a saved .npz and compute bootstrapped CIs for all metrics.

    Returns
    -------
    results : dict
        Keys: metric name -> (point, ci_lower, ci_upper)
    """
    data = np.load(npz_path)

    # Support both old format (targets, preds) and new format (targets, probs, logits)
    targets = data["targets"]
    if "probs" in data:
        scores = data["probs"]
    elif "logits" in data:
        # Apply sigmoid to logits
        logits = data["logits"]
        scores = 1.0 / (1.0 + np.exp(-logits))
    elif "preds" in data:
        scores = data["preds"].astype(float)
    else:
        raise KeyError(f"Cannot find prediction scores in {npz_path}. "
                       f"Available keys: {list(data.keys())}")

    if verbose:
        print(f"[*] Loaded {npz_path}")
        print(f"    Samples: {len(targets)} | Positive rate: {np.mean(targets):.3f}")
        print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    metrics = {
        "AUROC": auroc_metric,
        "AUPRC": auprc_metric,
        "Macro F1 (Youden)": youden_f1_metric,
    }

    results = {}
    for name, fn in metrics.items():
        point, ci_lo, ci_hi, _ = stratified_bootstrap_ci(
            targets, scores, fn, n_iterations=n_iterations
        )
        results[name] = (point, ci_lo, ci_hi)

        if verbose:
            print(f"    {name:25s}: {point:.4f}  [95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")

    return results


# ==========================================
# 4. MULTI-SEED AGGREGATION
# ==========================================
def evaluate_multi_seed(input_dir: str, n_iterations: int = 1000):
    """
    Aggregate results across multiple training seeds.

    Expects files named like: cross_cohort_results_seed0.npz, ..._seed4.npz
    or cross_cohort_results.npz (single file).
    """
    # Find all result files
    patterns = [
        os.path.join(input_dir, "cross_cohort_results_seed*.npz"),
        os.path.join(input_dir, "cross_cohort_results*.npz"),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    if not files:
        print(f"[!] No result files found in {input_dir}")
        return

    print(f"[*] Found {len(files)} result files for multi-seed aggregation.\n")

    # Collect per-seed metrics
    all_seed_results = []
    for f in files:
        print(f"--- Seed: {os.path.basename(f)} ---")
        results = evaluate_single_file(f, n_iterations=n_iterations, verbose=True)
        all_seed_results.append(results)
        print()

    # Aggregate across seeds
    print("=" * 60)
    print("  MULTI-SEED AGGREGATE (mean ± std across seeds)")
    print("=" * 60)

    metric_names = list(all_seed_results[0].keys())
    for name in metric_names:
        points = [r[name][0] for r in all_seed_results]
        mean_val = np.mean(points)
        std_val = np.std(points)

        # Also report the range of per-seed CIs
        ci_los = [r[name][1] for r in all_seed_results]
        ci_his = [r[name][2] for r in all_seed_results]

        print(f"  {name:25s}: {mean_val:.4f} ± {std_val:.4f}  "
              f"(seed range: [{min(points):.4f}, {max(points):.4f}])")


# ==========================================
# 5. LATEX OUTPUT
# ==========================================
def format_latex_row(results: dict, model_name: str = "Ours") -> str:
    """
    Generate a LaTeX table row for direct paper inclusion.

    Format: Model & AUROC & AUPRC & F1 \\
    With 95% CI in subscript.
    """
    parts = [model_name]
    for name in ["AUROC", "AUPRC", "Macro F1 (Youden)"]:
        if name in results:
            point, ci_lo, ci_hi = results[name]
            parts.append(
                f"${point:.3f}_{{[{ci_lo:.3f},\\,{ci_hi:.3f}]}}$"
            )
        else:
            parts.append("—")

    row = " & ".join(parts) + r" \\"
    return row


# ==========================================
# 6. CLI ENTRY POINT
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap CI estimation for cross-cohort evaluation results."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a single .npz result file."
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing multiple seed result files."
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap iterations (default: 1000)."
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Print LaTeX-formatted table row."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input is not None:
        print("=" * 60)
        print("  BOOTSTRAP CONFIDENCE INTERVAL ESTIMATION")
        print("=" * 60)

        results = evaluate_single_file(args.input, n_iterations=args.n_bootstrap)

        if args.latex:
            print("\n[LaTeX Table Row]")
            print(format_latex_row(results))

    elif args.input_dir is not None:
        evaluate_multi_seed(args.input_dir, n_iterations=args.n_bootstrap)

    else:
        # Default: try to find the result file in the current directory
        default_path = os.path.join(
            r"E:\SleepApnea\SleepApneaSSL", "cross_cohort_results.npz"
        )
        if os.path.exists(default_path):
            print("=" * 60)
            print("  BOOTSTRAP CONFIDENCE INTERVAL ESTIMATION")
            print("=" * 60)
            results = evaluate_single_file(default_path, n_iterations=args.n_bootstrap)

            if args.latex:
                print("\n[LaTeX Table Row]")
                print(format_latex_row(results))
        else:
            print("[!] No input specified. Use --input or --input-dir.")
            sys.exit(1)
