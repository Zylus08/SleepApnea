"""
analyze_seed_stability.py
=========================

Patient-level diagnostic analysis of A0 vs A1 across seeds.

Uses already-generated prediction files:
    seed_42/
    seed_123/
    seed_2025/

No model training is performed.

Outputs:
    - patient_level_predictions.csv
    - seed_summary.csv
    - patient_consistency.csv
    - per_seed_rank_changes.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'

SEEDS = [42, 123, 2025]

METHODS = {
    'A0': 'A0_VanillaNTXent',
    'A1': 'A1_TemporalNTXent'
}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(seed, method):

    path = os.path.join(
        RESULTS_DIR,
        f'seed_{seed}',
        f'{METHODS[method]}_predictions.npz'
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing prediction file:\n{path}'
        )

    d = np.load(path)

    required = {
        'patient_ids',
        'targets',
        'probs'
    }

    missing = required - set(d.files)

    if missing:
        raise RuntimeError(
            f'{path} is missing keys: {missing}'
        )

    return {
        'patient_ids': d['patient_ids'].astype(int),
        'targets': d['targets'].astype(float),
        'probs': d['probs'].astype(float)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== PATIENT-LEVEL SEED STABILITY ANALYSIS ===")

    all_data = {}

    for seed in SEEDS:

        print(f"\nLoading seed {seed}...")

        all_data[seed] = {}

        for method in METHODS:

            data = load_predictions(
                seed,
                method
            )

            all_data[seed][method] = data

            print(
                f"  {method}: "
                f"{len(data['patient_ids'])} patients"
            )

    # ========================================================
    # VERIFY PATIENT ALIGNMENT
    # ========================================================

    reference_ids = all_data[SEEDS[0]]['A0']['patient_ids']
    reference_ids = set(reference_ids.tolist())

    for seed in SEEDS:

        for method in METHODS:

            ids = set(
                all_data[seed][method]['patient_ids'].tolist()
            )

            if ids != reference_ids:

                raise RuntimeError(
                    f"Patient mismatch in seed={seed}, "
                    f"method={method}"
                )

    print("\n[OK] All six prediction files contain identical patient sets.")

    # ========================================================
    # BUILD MASTER PATIENT TABLE
    # ========================================================

    rows = []

    for pid in sorted(reference_ids):

        row = {
            'patient_id': pid
        }

        # Label
        ref = all_data[42]['A0']

        idx = np.where(
            ref['patient_ids'] == pid
        )[0][0]

        row['target'] = int(
            ref['targets'][idx]
        )

        # Predictions from every seed/method
        for seed in SEEDS:

            for method in METHODS:

                data = all_data[seed][method]

                idx = np.where(
                    data['patient_ids'] == pid
                )[0][0]

                row[
                    f'{method}_seed{seed}'
                ] = float(
                    data['probs'][idx]
                )

        rows.append(row)

    df = pd.DataFrame(rows)

    # ========================================================
    # A1 - A0 DELTAS
    # ========================================================

    for seed in SEEDS:

        df[
            f'delta_A1_minus_A0_seed{seed}'
        ] = (
            df[f'A1_seed{seed}']
            - df[f'A0_seed{seed}']
        )

    # ========================================================
    # CORRECTNESS
    # ========================================================

    for seed in SEEDS:

        for method in METHODS:

            prob_col = f'{method}_seed{seed}'

            pred_col = f'{method}_correct_seed{seed}'

            df[pred_col] = (
                (
                    df[prob_col] >= 0.5
                ).astype(int)
                ==
                df['target']
            )

    # ========================================================
    # SAVE MASTER TABLE
    # ========================================================

    master_path = os.path.join(
        RESULTS_DIR,
        'patient_level_predictions.csv'
    )

    df.to_csv(
        master_path,
        index=False
    )

    print(
        f"\n[+] Saved:\n{master_path}"
    )

    # ========================================================
    # SEED SUMMARY
    # ========================================================

    summary_rows = []

    for seed in SEEDS:

        for method in METHODS:

            y = df['target'].values
            p = df[f'{method}_seed{seed}'].values

            auc = roc_auc_score(y, p)
            ap = average_precision_score(y, p)

            accuracy = np.mean(
                (
                    p >= 0.5
                ).astype(int)
                == y
            )

            summary_rows.append({
                'seed': seed,
                'method': method,
                'AUROC': auc,
                'AUPRC': ap,
                'accuracy_at_0.5': accuracy
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        'seed_summary.csv'
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"[+] Saved:\n{summary_path}"
    )

    # ========================================================
    # PATIENT CONSISTENCY
    # ========================================================

    consistency_rows = []

    for _, row in df.iterrows():

        pid = int(row['patient_id'])
        target = int(row['target'])

        a0_probs = np.array([
            row[f'A0_seed{s}']
            for s in SEEDS
        ])

        a1_probs = np.array([
            row[f'A1_seed{s}']
            for s in SEEDS
        ])

        a0_preds = (
            a0_probs >= 0.5
        ).astype(int)

        a1_preds = (
            a1_probs >= 0.5
        ).astype(int)

        consistency_rows.append({

            'patient_id': pid,
            'target': target,

            'A0_mean_prob': a0_probs.mean(),
            'A0_std_prob': a0_probs.std(
                ddof=1
            ),
            'A0_min_prob': a0_probs.min(),
            'A0_max_prob': a0_probs.max(),
            'A0_correct_count': int(
                np.sum(a0_preds == target)
            ),

            'A1_mean_prob': a1_probs.mean(),
            'A1_std_prob': a1_probs.std(
                ddof=1
            ),
            'A1_min_prob': a1_probs.min(),
            'A1_max_prob': a1_probs.max(),
            'A1_correct_count': int(
                np.sum(a1_preds == target)
            ),

            'A1_minus_A0_mean': (
                a1_probs.mean()
                - a0_probs.mean()
            ),

            'A1_minus_A0_std': (
                a1_probs - a0_probs
            ).std(ddof=1)
        })

    consistency_df = pd.DataFrame(
        consistency_rows
    )

    consistency_path = os.path.join(
        RESULTS_DIR,
        'patient_consistency.csv'
    )

    consistency_df.to_csv(
        consistency_path,
        index=False
    )

    print(
        f"[+] Saved:\n{consistency_path}"
    )

    # ========================================================
    # RANK CHANGES
    # ========================================================

    rank_rows = []

    for seed in SEEDS:

        a0 = df[
            ['patient_id', 'target', f'A0_seed{seed}']
        ].copy()

        a1 = df[
            ['patient_id', 'target', f'A1_seed{seed}']
        ].copy()

        a0['A0_rank'] = a0[
            f'A0_seed{seed}'
        ].rank(
            ascending=False,
            method='average'
        )

        a1['A1_rank'] = a1[
            f'A1_seed{seed}'
        ].rank(
            ascending=False,
            method='average'
        )

        merged = a0.merge(
            a1,
            on=['patient_id', 'target']
        )

        merged['rank_change_A1_minus_A0'] = (
            merged['A1_rank']
            - merged['A0_rank']
        )

        for _, r in merged.iterrows():

            rank_rows.append({
                'seed': seed,
                'patient_id': int(
                    r['patient_id']
                ),
                'target': int(
                    r['target']
                ),
                'A0_rank': r['A0_rank'],
                'A1_rank': r['A1_rank'],
                'rank_change_A1_minus_A0':
                    r['rank_change_A1_minus_A0']
            })

    rank_df = pd.DataFrame(
        rank_rows
    )

    rank_path = os.path.join(
        RESULTS_DIR,
        'per_seed_rank_changes.csv'
    )

    rank_df.to_csv(
        rank_path,
        index=False
    )

    print(
        f"[+] Saved:\n{rank_path}"
    )

    # ========================================================
    # CONSOLE DIAGNOSTICS
    # ========================================================

    print("\n" + "=" * 65)
    print("SEED PERFORMANCE")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f'{x:.4f}'
        )
    )

    print("\n" + "=" * 65)
    print("PATIENT-LEVEL STABILITY")
    print("=" * 65)

    for method in METHODS:

        mean_std = consistency_df[
            f'{method}_std_prob'
        ].mean()

        correct_counts = consistency_df[
            f'{method}_correct_count'
        ]

        print(
            f"{method}: "
            f"mean probability SD = {mean_std:.4f} | "
            f"patients correct in all 3 seeds = "
            f"{np.sum(correct_counts == 3)} / {len(df)}"
        )

    print("\n" + "=" * 65)
    print("A1 VS A0 — PER SEED")
    print("=" * 65)

    for seed in SEEDS:

        delta = df[
            f'delta_A1_minus_A0_seed{seed}'
        ]

        print(
            f"Seed {seed}: "
            f"mean Δprob = {delta.mean():+.4f} | "
            f"median Δprob = {delta.median():+.4f} | "
            f"A1 increased probability for "
            f"{np.sum(delta > 0)} / {len(delta)} patients"
        )

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == '__main__':
    main()