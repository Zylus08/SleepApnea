"""
experiments/ablation_bootstrap_ci.py
=====================================
Re-evaluates each saved ablation encoder on the test set,
collects raw patient-level probs+targets, saves as .npz,
then runs stratified bootstrap CI (n=1000) for AUROC and AUPRC.

Requires:
  results/loss_ablation/encoder_*.pth   (from loss_ablation.py)
  E:\\SleepApneaProcessed               (preprocessed .pt files)
  E:\\SleepApnea\\participants.tsv

Usage:
  conda run -n reswork python experiments/ablation_bootstrap_ci.py
"""
import os, sys, glob, re, csv, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from model import STFTEncoder2D
from downstream_finetune import (
    load_bids_labels, LabeledStreamingDataset,
    SleepApneaClassifier, BinaryFocalLossWithLogits
)
import torch.optim as optim

SEED      = 42
DS_EPOCHS = 8
BATCH     = 128
DATA_DIR  = r'E:\SleepApneaProcessed'
TSV_PATH  = r'E:\SleepApnea\participants.tsv'
OUT_DIR   = r'E:\SleepApnea\SleepApneaSSL\results\loss_ablation'
N_BOOT    = 1000

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Bootstrap engine ──────────────────────────────────────────────────────────
def stratified_bootstrap_ci(targets, scores, metric_fn, n=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(targets == 1)[0]
    idx_neg = np.where(targets == 0)[0]
    point   = metric_fn(targets, scores)
    vals    = []
    for _ in range(n):
        bp  = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        bn  = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        bi  = np.concatenate([bp, bn])
        try:
            vals.append(metric_fn(targets[bi], scores[bi]))
        except ValueError:
            vals.append(np.nan)
    vals = np.array(vals)
    valid = vals[~np.isnan(vals)]
    alpha = 1.0 - ci
    return point, float(np.percentile(valid, 100*alpha/2)), float(np.percentile(valid, 100*(1-alpha/2)))

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    pat_probs = {}; pat_targets = {}
    for x, y, pids in loader:
        x = x.to(device)
        logits = model(x)
        probs  = torch.sigmoid(logits).cpu().numpy()
        for prob, lbl, pid in zip(probs, y.numpy(), pids.numpy()):
            pid = int(pid)
            pat_probs.setdefault(pid, []).append(float(prob))
            pat_targets[pid] = float(lbl)
    ids  = list(pat_probs)
    pp   = np.array([np.mean(pat_probs[p]) for p in ids])
    pt   = np.array([pat_targets[p] for p in ids])
    return pt, pp

def finetune_and_collect(encoder_path, label_map, train_files, val_files, test_files, device):
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
    model   = SleepApneaClassifier(encoder, finetune_mode='frozen').to(device)
    optim_  = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    crit    = BinaryFocalLossWithLogits(alpha=0.55, gamma=1.0)

    train_loader = DataLoader(LabeledStreamingDataset(train_files, label_map, is_train=True),  batch_size=BATCH, shuffle=False)
    val_loader   = DataLoader(LabeledStreamingDataset(val_files,   label_map, is_train=False), batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(LabeledStreamingDataset(test_files,  label_map, is_train=False), batch_size=BATCH, shuffle=False)

    best_val_auc = 0.; best_state = None
    for epoch in range(1, DS_EPOCHS+1):
        model.train(); model.encoder.eval()
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device).float().view(-1)
            optim_.zero_grad()
            logits = model(x)
            loss   = crit(logits, y)
            loss.backward(); optim_.step()
        tgt, sc = collect_predictions(model, val_loader, device)
        if len(set(tgt)) > 1:
            val_auc = roc_auc_score(tgt, sc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    targets, probs = collect_predictions(model, test_loader, device)
    return targets, probs

def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_map = load_bids_labels(TSV_PATH)
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))

    pids, pid_to_file = [], {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pids.append(pid); pid_to_file[pid] = f
    pids = list(set(pids))
    if len(pids) < 10:
        print("SKIP: insufficient data"); return

    labels = [label_map[p] for p in pids]
    trainval_ids, test_ids = train_test_split(pids, test_size=0.20, stratify=labels, random_state=SEED)
    tv_labels = [label_map[p] for p in trainval_ids]
    train_ids, val_ids   = train_test_split(trainval_ids, test_size=0.25, stratify=tv_labels, random_state=SEED)

    train_files = [pid_to_file[p] for p in train_ids]
    val_files   = [pid_to_file[p] for p in val_ids]
    test_files  = [pid_to_file[p] for p in test_ids]

    ablation_names = [
        'A0_VanillaNTXent',
        'A1_TemporalNTXent',
        'A2_NTXent_TempReg',
        'A3_PhysioCLR',
    ]

    summary = []
    for name in ablation_names:
        enc_path = os.path.join(OUT_DIR, f'encoder_{name}.pth')
        if not os.path.exists(enc_path):
            print(f"SKIP {name}: encoder not found"); continue

        print(f"\n{'='*60}\n  BOOTSTRAP CI: {name}\n{'='*60}")
        targets, probs = finetune_and_collect(enc_path, label_map, train_files, val_files, test_files, device)

        # Save raw predictions
        npz_path = os.path.join(OUT_DIR, f'{name}_predictions.npz')
        np.savez(npz_path, targets=targets, probs=probs)

        if len(set(targets)) < 2:
            print(f"  SKIP: only one class in test set"); continue

        auroc_pt, auroc_lo, auroc_hi = stratified_bootstrap_ci(
            targets, probs, roc_auc_score, n=N_BOOT)
        auprc_pt, auprc_lo, auprc_hi = stratified_bootstrap_ci(
            targets, probs, average_precision_score, n=N_BOOT)

        print(f"  AUROC: {auroc_pt:.4f} [95% CI: {auroc_lo:.4f}–{auroc_hi:.4f}]")
        print(f"  AUPRC: {auprc_pt:.4f} [95% CI: {auprc_lo:.4f}–{auprc_hi:.4f}]")
        print(f"  Patients: {len(targets)} | Pos rate: {targets.mean():.3f}")

        # LaTeX row
        latex = (f"{name} & ${auroc_pt:.3f}_{{[{auroc_lo:.3f},\\,{auroc_hi:.3f}]}}$"
                 f" & ${auprc_pt:.3f}_{{[{auprc_lo:.3f},\\,{auprc_hi:.3f}]}}$ \\\\")
        print(f"  LaTeX: {latex}")

        row = {
            'ablation': name,
            'n_patients': int(len(targets)),
            'pos_rate': float(targets.mean()),
            'auroc_point': auroc_pt,
            'auroc_ci_lo': auroc_lo,
            'auroc_ci_hi': auroc_hi,
            'auprc_point': auprc_pt,
            'auprc_ci_lo': auprc_lo,
            'auprc_ci_hi': auprc_hi,
            'n_bootstrap': N_BOOT,
        }
        summary.append(row)

        # Save per-ablation CI json
        with open(os.path.join(OUT_DIR, f'{name}_bootstrap_ci.json'), 'w') as f:
            json.dump(row, f, indent=2)

    # Summary CSV
    if summary:
        ci_csv = os.path.join(OUT_DIR, 'bootstrap_ci_summary.csv')
        with open(ci_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
        print(f"\nBootstrap CI summary saved to {ci_csv}")

    print("\n=== BOOTSTRAP CI COMPLETE ===")
    for r in summary:
        print(f"  {r['ablation']:30s}  AUROC={r['auroc_point']:.4f} [{r['auroc_ci_lo']:.4f}–{r['auroc_ci_hi']:.4f}]"
              f"  AUPRC={r['auprc_point']:.4f} [{r['auprc_ci_lo']:.4f}–{r['auprc_ci_hi']:.4f}]")

if __name__ == '__main__':
    main()
