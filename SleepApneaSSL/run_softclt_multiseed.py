"""
run_softclt_multiseed.py
========================
Runs SoftCLT (Lee et al., ICLR 2024 — soft instance-wise CL) across
seeds [42, 123, 2025, 7, 13] using exactly the same protocol as A0/A1/A3.

Phase 4: set SEEDS = [42] to pilot on seed 42 only.
Phase 5: set SEEDS = [42, 123, 2025, 7, 13] for the full five-seed experiment.

Results saved to: results/softclt/seed_{seed}/
"""
import os, sys, glob, re, json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.loss_ablation import (
    SoftCLTWrapper,
    pretrain_ssl,
    finetune_downstream,
    TSV_PATH,
    DATA_DIR,
    set_seed,
)
from downstream_finetune import load_bids_labels
from experiments.paired_permutation_test import paired_permutation_test

SEEDS = [42, 123, 2025, 7, 13]
RESULTS_DIR    = r'E:\SleepApnea\SleepApneaSSL\results\softclt'
MULTISEED_DIR  = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'  # A0/A1
A3_DIR         = r'E:\SleepApnea\SleepApneaSSL\results\A3_multiseed'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_predictions(path):
    d = np.load(path)
    return {
        'patient_ids': d['patient_ids'].astype(int),
        'targets':     d['targets'].astype(float),
        'probs':       d['probs'].astype(float),
    }


def align_predictions(*dicts):
    """Return common patients and aligned targets/probs for each dict."""
    common_ids = sorted(set.intersection(*[set(d['patient_ids']) for d in dicts]))
    targets = None
    probs_list = []
    for d in dicts:
        p_ordered, t_ordered = [], []
        for pid in common_ids:
            idx = np.where(d['patient_ids'] == pid)[0][0]
            p_ordered.append(d['probs'][idx])
            t_ordered.append(d['targets'][idx])
        probs_list.append(np.array(p_ordered))
        if targets is None:
            targets = np.array(t_ordered)
    return targets, probs_list


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    label_map = load_bids_labels(TSV_PATH)
    all_files  = glob.glob(os.path.join(DATA_DIR, '*.pt'))

    pids = []
    pid_to_file = {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pids.append(pid)
                pid_to_file[pid] = f

    pids = sorted(list(set(pids)))
    labels_for_split = [label_map[p] for p in pids]

    raw_metrics = {}

    for seed in SEEDS:
        print(f"\n>>> Running SoftCLT Seed {seed} <<<")

        # Fixed split: random_state=42 (identical to A0/A1/A3)
        p_train_val, p_test, l_train_val, _ = train_test_split(
            pids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=42
        )
        p_train, p_val = train_test_split(
            p_train_val, test_size=0.25, stratify=l_train_val, random_state=42
        )

        train_files = [pid_to_file[p] for p in p_train]
        val_files   = [pid_to_file[p] for p in p_val]
        test_files  = [pid_to_file[p] for p in p_test]
        ssl_files   = train_files

        seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        enc_path  = os.path.join(seed_dir, "SoftCLT_encoder.pth")
        pred_path = os.path.join(seed_dir, "SoftCLT_predictions.npz")

        loss_fn = SoftCLTWrapper(temperature=0.5, tau_I=2.0, alpha=0.5)

        # SSL pretraining
        if not os.path.exists(enc_path):
            set_seed(seed)
            pretrain_ssl('SoftCLT', loss_fn, ssl_files, device, enc_path)
        else:
            print(f"Skipping pretraining, {enc_path} already exists")

        # Downstream fine-tuning
        set_seed(seed)
        metrics = finetune_downstream(
            enc_path, label_map, train_files, val_files, test_files, device,
            predictions_save_path=pred_path, seed=seed
        )

        raw_metrics[seed] = metrics

    # ── Aggregation ─────────────────────────────────────────────────────────
    print("\n=== AGGREGATING RESULTS ===")
    auroc_list = [raw_metrics[s]['auroc'] for s in SEEDS]
    auprc_list = [raw_metrics[s]['auprc'] for s in SEEDS]

    mean_auroc = np.mean(auroc_list)
    std_auroc  = np.std(auroc_list, ddof=1) if len(SEEDS) > 1 else 0.0
    med_auroc  = np.median(auroc_list)
    mean_auprc = np.mean(auprc_list)
    std_auprc  = np.std(auprc_list, ddof=1) if len(SEEDS) > 1 else 0.0
    med_auprc  = np.median(auprc_list)

    print(f"SoftCLT AUROC: {mean_auroc:.4f} +- {std_auroc:.4f} (Median: {med_auroc:.4f})")
    print(f"SoftCLT AUPRC: {mean_auprc:.4f} +- {std_auprc:.4f} (Median: {med_auprc:.4f})")

    # ── Permutation tests vs A0 / A1 / A3 ───────────────────────────────────
    diff_results = []

    for seed in SEEDS:
        path_sc = os.path.join(RESULTS_DIR, f"seed_{seed}", "SoftCLT_predictions.npz")
        path_a0 = os.path.join(MULTISEED_DIR, f"seed_{seed}", "A0_VanillaNTXent_predictions.npz")
        path_a1 = os.path.join(MULTISEED_DIR, f"seed_{seed}", "A1_TemporalNTXent_predictions.npz")
        path_a3 = os.path.join(A3_DIR, f"seed_{seed}", "A3_PhysioCLR_predictions.npz")

        row = {'seed': seed}
        t, [p_sc, p_a0, p_a1] = align_predictions(
            load_predictions(path_sc),
            load_predictions(path_a0),
            load_predictions(path_a1),
        )
        row['n_patients']   = len(t)
        row['pos_rate']     = t.mean()

        diff, pval = paired_permutation_test(t, p_sc, p_a0, roc_auc_score, n_perm=10000, seed=42)
        row['SC-A0_AUROC'] = diff; row['p_SC-A0_AUROC'] = pval
        diff, pval = paired_permutation_test(t, p_sc, p_a1, roc_auc_score, n_perm=10000, seed=42)
        row['SC-A1_AUROC'] = diff; row['p_SC-A1_AUROC'] = pval
        diff, pval = paired_permutation_test(t, p_sc, p_a0, average_precision_score, n_perm=10000, seed=42)
        row['SC-A0_AUPRC'] = diff; row['p_SC-A0_AUPRC'] = pval
        diff, pval = paired_permutation_test(t, p_sc, p_a1, average_precision_score, n_perm=10000, seed=42)
        row['SC-A1_AUPRC'] = diff; row['p_SC-A1_AUPRC'] = pval

        # vs A3 (if available)
        if os.path.exists(path_a3):
            t3, [p_sc3, p_a33] = align_predictions(
                load_predictions(path_sc),
                load_predictions(path_a3),
            )
            diff, pval = paired_permutation_test(t3, p_sc3, p_a33, roc_auc_score, n_perm=10000, seed=42)
            row['SC-A3_AUROC'] = diff; row['p_SC-A3_AUROC'] = pval
            diff, pval = paired_permutation_test(t3, p_sc3, p_a33, average_precision_score, n_perm=10000, seed=42)
            row['SC-A3_AUPRC'] = diff; row['p_SC-A3_AUPRC'] = pval
        else:
            row['SC-A3_AUROC'] = None; row['p_SC-A3_AUROC'] = None
            row['SC-A3_AUPRC'] = None; row['p_SC-A3_AUPRC'] = None

        diff_results.append(row)

    num_sc_gt_a0 = sum(1 for r in diff_results if r['SC-A0_AUROC'] is not None and r['SC-A0_AUROC'] > 0)
    num_sc_gt_a1 = sum(1 for r in diff_results if r['SC-A1_AUROC'] is not None and r['SC-A1_AUROC'] > 0)

    print(f"Seeds SoftCLT > A0 (AUROC): {num_sc_gt_a0}/{len(SEEDS)}")
    print(f"Seeds SoftCLT > A1 (AUROC): {num_sc_gt_a1}/{len(SEEDS)}")

    analysis = {
        'softclt_aggregate': {
            'mean_auroc': mean_auroc, 'std_auroc': std_auroc, 'median_auroc': med_auroc,
            'mean_auprc': mean_auprc, 'std_auprc': std_auprc, 'median_auprc': med_auprc,
        },
        'diffs': diff_results,
        'raw_metrics': {str(s): raw_metrics[s] for s in SEEDS},
        'num_sc_gt_a0': num_sc_gt_a0,
        'num_sc_gt_a1': num_sc_gt_a1,
        'seeds_run': SEEDS,
    }

    out_json = os.path.join(RESULTS_DIR, "softclt_analysis.json")
    with open(out_json, 'w') as f:
        json.dump(analysis, f, indent=4)
    print(f"[+] Analysis saved to {out_json}")


if __name__ == '__main__':
    main()
