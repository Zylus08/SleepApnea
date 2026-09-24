"""
paired_permutation_test.py
===========================
Patient-level paired permutation test: A1 vs A0.

For each test patient, randomly swap A0/A1 predictions with probability 0.5.
This preserves the paired structure and tests whether the observed
performance difference is larger than expected under the null.

Runs independently for seeds 42, 123, 2025.
"""

import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# SEEDS = [42, 123, 2025]
SEEDS = [7, 13]
N_PERM = 10000

RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'


def paired_permutation_test(
    targets,
    probs_a1,
    probs_a0,
    metric_fn,
    n_perm=10000,
    seed=42
):
    """
    Returns:
        observed_diff,
        p_value_two_sided
    """

    rng = np.random.RandomState(seed)

    observed = (
        metric_fn(targets, probs_a1)
        - metric_fn(targets, probs_a0)
    )

    perm_diffs = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):

        # Randomly swap A0/A1 predictions independently
        # for each patient.
        swap = rng.randint(0, 2, size=len(targets)).astype(bool)

        perm_a1 = np.where(
            swap,
            probs_a0,
            probs_a1
        )

        perm_a0 = np.where(
            swap,
            probs_a1,
            probs_a0
        )

        try:
            perm_diffs[i] = (
                metric_fn(targets, perm_a1)
                - metric_fn(targets, perm_a0)
            )
        except ValueError:
            perm_diffs[i] = np.nan

    perm_diffs = perm_diffs[~np.isnan(perm_diffs)]

    # Two-sided randomization p-value with +1 correction
    extreme = np.sum(
        np.abs(perm_diffs) >= abs(observed)
    )

    p_value = (extreme + 1) / (len(perm_diffs) + 1)

    return float(observed), float(p_value)


def load_and_align(path_a0, path_a1):
    """
    Load predictions and explicitly align A0/A1 by patient ID.
    """

    d0 = np.load(path_a0)
    d1 = np.load(path_a1)

    ids0 = d0['patient_ids']
    ids1 = d1['patient_ids']

    targets0 = d0['targets']
    targets1 = d1['targets']

    probs0 = d0['probs']
    probs1 = d1['probs']

    # Build patient -> index mapping
    map0 = {
        int(pid): i
        for i, pid in enumerate(ids0)
    }

    map1 = {
        int(pid): i
        for i, pid in enumerate(ids1)
    }

    common_ids = sorted(set(map0) & set(map1))

    if len(common_ids) == 0:
        raise RuntimeError("No common patient IDs found.")

    targets = []
    p0 = []
    p1 = []

    for pid in common_ids:

        i0 = map0[pid]
        i1 = map1[pid]

        if not np.isclose(targets0[i0], targets1[i1]):
            raise RuntimeError(
                f"Target mismatch for patient {pid}"
            )

        targets.append(targets0[i0])
        p0.append(probs0[i0])
        p1.append(probs1[i1])

    return (
        np.array(common_ids, dtype=np.int64),
        np.array(targets, dtype=np.float32),
        np.array(p1, dtype=np.float32),
        np.array(p0, dtype=np.float32)
    )


def main():

    print("=== PAIRED PERMUTATION TEST (A1 vs A0) ===")
    print(f"Permutations: {N_PERM}")

    summary = []

    for seed in SEEDS:

        print(f"\n--- Seed {seed} ---")

        seed_dir = os.path.join(
            RESULTS_DIR,
            f"seed_{seed}"
        )

        path_a0 = os.path.join(
            seed_dir,
            'A0_VanillaNTXent_predictions.npz'
        )

        path_a1 = os.path.join(
            seed_dir,
            'A1_TemporalNTXent_predictions.npz'
        )

        if not os.path.exists(path_a0):
            print("  Missing A0 predictions. Skipping.")
            continue

        if not os.path.exists(path_a1):
            print("  Missing A1 predictions. Skipping.")
            continue

        ids, targets, p1, p0 = load_and_align(
            path_a0,
            path_a1
        )

        print(
            f"  N test patients: {len(ids)} | "
            f"Pos rate: {targets.mean():.3f}"
        )

        # AUROC
        auroc_diff, auroc_p = paired_permutation_test(
            targets,
            p1,
            p0,
            roc_auc_score,
            n_perm=N_PERM,
            seed=seed
        )

        # AUPRC
        auprc_diff, auprc_p = paired_permutation_test(
            targets,
            p1,
            p0,
            average_precision_score,
            n_perm=N_PERM,
            seed=seed + 1000
        )

        a0_auc = roc_auc_score(targets, p0)
        a1_auc = roc_auc_score(targets, p1)

        a0_ap = average_precision_score(targets, p0)
        a1_ap = average_precision_score(targets, p1)

        print(
            f"  AUROC: A0={a0_auc:.4f} | "
            f"A1={a1_auc:.4f} | "
            f"Δ={auroc_diff:+.4f} | "
            f"p={auroc_p:.4f}"
        )

        print(
            f"  AUPRC: A0={a0_ap:.4f} | "
            f"A1={a1_ap:.4f} | "
            f"Δ={auprc_diff:+.4f} | "
            f"p={auprc_p:.4f}"
        )

        summary.append({
            'seed': seed,
            'n_patients': len(ids),
            'positive_rate': float(targets.mean()),

            'A0_AUROC': a0_auc,
            'A1_AUROC': a1_auc,
            'AUROC_delta': auroc_diff,
            'AUROC_p_two_sided': auroc_p,

            'A0_AUPRC': a0_ap,
            'A1_AUPRC': a1_ap,
            'AUPRC_delta': auprc_diff,
            'AUPRC_p_two_sided': auprc_p,
        })

    if summary:

        out_csv = os.path.join(
            RESULTS_DIR,
            'paired_permutation_results.csv'
        )

        with open(out_csv, 'w', newline='') as f:

            writer = csv.DictWriter(
                f,
                fieldnames=summary[0].keys()
            )

            writer.writeheader()
            writer.writerows(summary)

        print(
            f"\nSaved results to:\n{out_csv}"
        )

    else:
        print("\nNo valid seeds found.")


if __name__ == '__main__':
    main()