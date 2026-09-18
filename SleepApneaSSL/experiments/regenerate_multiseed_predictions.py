# regenerate_multiseed_predictions.py

import os
import sys
import glob
import re
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from loss_ablation import (
    finetune_downstream,
    TSV_PATH,
    DATA_DIR,
    set_seed
)

from downstream_finetune import load_bids_labels


OUT_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'

SEEDS = [42, 123, 2025]


def main():

    print("=== REGENERATING MULTI-SEED PREDICTIONS ===")

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Device: {device}")

    label_map = load_bids_labels(TSV_PATH)

    all_files = glob.glob(
        os.path.join(DATA_DIR, '*.pt')
    )

    pids = []
    pid_to_file = {}

    for f in all_files:

        digits = re.findall(
            r'\d+',
            os.path.basename(f)
        )

        if digits:

            pid = int(digits[0])

            if pid in label_map:

                pids.append(pid)
                pid_to_file[pid] = f

    pids = sorted(list(set(pids)))

    labels_for_split = [
        label_map[p]
        for p in pids
    ]

    # ---------------------------------------------------------
    # SAME FIXED PATIENT SPLIT AS ORIGINAL MULTI-SEED RUN
    # ---------------------------------------------------------
    p_train_val, p_test, l_train_val, _ = train_test_split(
        pids,
        labels_for_split,
        test_size=0.2,
        stratify=labels_for_split,
        random_state=42
    )

    p_train, p_val = train_test_split(
        p_train_val,
        test_size=0.25,
        stratify=l_train_val,
        random_state=42
    )

    train_files = [
        pid_to_file[p]
        for p in p_train
    ]

    val_files = [
        pid_to_file[p]
        for p in p_val
    ]

    test_files = [
        pid_to_file[p]
        for p in p_test
    ]

    print(
        f"Train: {len(train_files)} | "
        f"Val: {len(val_files)} | "
        f"Test: {len(test_files)}"
    )

    methods = [
        'A0_VanillaNTXent',
        'A1_TemporalNTXent'
    ]

    for seed in SEEDS:

        print(f"\n========== SEED {seed} ==========")

        seed_dir = os.path.join(
            OUT_DIR,
            f"seed_{seed}"
        )

        for method in methods:

            print(f"\n--- {method} ---")

            encoder_path = os.path.join(
                seed_dir,
                f'{method}_encoder.pth'
            )

            prediction_path = os.path.join(
                seed_dir,
                f'{method}_predictions.npz'
            )

            if not os.path.exists(encoder_path):
                print(
                    f"[!] Missing encoder: "
                    f"{encoder_path}"
                )
                continue

            set_seed(seed)

            metrics = finetune_downstream(
                encoder_path=encoder_path,
                label_map=label_map,
                train_files=train_files,
                val_files=val_files,
                test_files=test_files,
                device=device,
                predictions_save_path=prediction_path,
                seed=seed
            )

            print(
                f"  AUROC: {metrics['auroc']:.4f} | "
                f"AUPRC: {metrics['auprc']:.4f}"
            )

    print("\n=== DONE ===")


if __name__ == '__main__':
    main()