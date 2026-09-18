"""
experiments/loss_ablation.py
Controlled loss ablation: A0 (vanilla NT-Xent), A1 (TemporalNTXent),
A2 (NT-Xent + temporal continuity), A3 (PhysioCLR).

SAME: encoder, dataset, batch size, lr, epochs, seed, augmentations.
ONLY VARIABLE: loss function.

Outputs saved to results/loss_ablation/
"""
import os, sys, json, time, random, glob, re, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from model import STFTEncoder2D, SimCLR, EEGEncoder
from physio_clr import SpectralSubbandMasking, PhysioCLRLoss
from temporal_loss import TemporalNTXentLoss
from downstream_finetune import (
    load_bids_labels, LabeledStreamingDataset,
    SleepApneaClassifier, BinaryFocalLossWithLogits
)

SEED       = 42
BATCH      = 64
LR         = 1e-3
SSL_EPOCHS = 5          # short for ablation grid
DS_EPOCHS  = 8
DATA_DIR   = r'E:\SleepApneaProcessed'
TSV_PATH   = r'E:\SleepApnea\participants.tsv'
OUT_DIR    = r'E:\SleepApnea\SleepApneaSSL\results\loss_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ── SSL Dataset ──────────────────────────────────────────────────────────────
class SSLStreamingDataset(IterableDataset):
    def __init__(self, files):
        self.files = files
    def __iter__(self):
        flist = list(self.files); random.shuffle(flist)
        for f in flist:
            pid_match = re.findall(r'\d+', os.path.basename(f))
            pid = int(pid_match[0]) if pid_match else 0
            
            bag = torch.load(f, map_location='cpu', weights_only=True)
            if isinstance(bag, torch.Tensor) and bag.dim() == 2 and bag.shape[0] == 20:
                n_win = bag.shape[1] // 3000
                bag = bag[:, :n_win*3000].view(20, n_win, 3000).permute(1, 0, 2)
            elif not isinstance(bag, torch.Tensor) or bag.dim() != 3:
                continue
            n_win = bag.shape[0]
            for w in range(n_win):
                is_boundary = int(w == 0)
                yield bag[w].float(), torch.tensor(pid, dtype=torch.long), torch.tensor(w, dtype=torch.long), torch.tensor(is_boundary)
            del bag; gc.collect()

# ── Vanilla NT-Xent ──────────────────────────────────────────────────────────
class VanillaNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.T = temperature
    def forward(self, z_i, z_j, *args, **kwargs):
        B = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], 0), dim=1)
        sim = (torch.mm(z, z.t()) / self.T).float()  # cast to fp32 to avoid half overflow
        sim.fill_diagonal_(-1e9)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, labels)

# ── NT-Xent + Temporal Continuity Reg ────────────────────────────────────────
class NTXentPlusTempReg(nn.Module):
    def __init__(self, temperature=0.5, lambda_temporal=0.15):
        super().__init__()
        self.ntxent = VanillaNTXentLoss(temperature)
        self.lam = lambda_temporal
    def forward(self, z_i, z_j, sub_ids=None, time_idx=None, is_boundary=None):
        l_nt = self.ntxent(z_i, z_j)
        # temporal continuity on z_i
        z_n = F.normalize(z_i, dim=1)
        l_temp = z_n.new_tensor(0.)
        if sub_ids is not None and time_idx is not None:
            B = z_n.shape[0]
            if B >= 2:
                # Find valid temporally adjacent pairs within the batch
                same_sub = (sub_ids.unsqueeze(1) == sub_ids.unsqueeze(0))
                time_next = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0) + 1)
                valid_mask = same_sub & time_next
                
                diff = z_n.unsqueeze(1) - z_n.unsqueeze(0)  # (B, B, D)
                mse_dist = diff.pow(2).mean(dim=-1)         # (B, B)
                
                n_valid = valid_mask.sum()
                if n_valid > 0:
                    l_temp = (mse_dist * valid_mask).sum() / n_valid
        total = l_nt + self.lam * l_temp
        return total, {'loss_contrastive': l_nt.item(), 'loss_temporal': l_temp.item() if isinstance(l_temp, torch.Tensor) else l_temp}

# ── Wrapper to give TemporalNTXentLoss same signature ─────────────────────────
class TemporalNTXentWrapper(nn.Module):
    def __init__(self, temperature=0.5, lambda_decay=0.1):
        super().__init__()
        self.loss = TemporalNTXentLoss(temperature=temperature, lambda_decay=lambda_decay)
    def forward(self, z_i, z_j, sub_ids, time_idx, is_boundary=None):
        l = self.loss(z_i, z_j, sub_ids, time_idx)
        return l, {'loss_contrastive': l.item(), 'loss_temporal': 0.0}

# ── PhysioCLR wrapper ─────────────────────────────────────────────────────────
class PhysioCLRWrapper(nn.Module):
    def __init__(self, temperature=0.07, lambda_temporal=0.15):
        super().__init__()
        self.loss = PhysioCLRLoss(temperature=temperature, lambda_temporal=lambda_temporal)
    def forward(self, z_i, z_j, sub_ids=None, time_idx=None, is_boundary=None):
        if is_boundary is None:
            is_boundary = torch.zeros(z_i.shape[0], dtype=torch.long, device=z_i.device)
        return self.loss(z_i, z_j, is_boundary)

# ── SSL pretraining ───────────────────────────────────────────────────────────
def pretrain_ssl(loss_name, loss_fn, ssl_files, device, out_path):
    set_seed(SEED)
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128).to(device)
    model = SimCLR(encoder, projection_dim=64).to(device)
    masker = SpectralSubbandMasking(p=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    dataset = SSLStreamingDataset(ssl_files)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, drop_last=True)

    history = []
    for epoch in range(1, SSL_EPOCHS+1):
        model.train(); masker.train()
        ep_loss = ep_cont = ep_temp = 0.; n = 0
        for x, sub_ids, time_idx, is_boundary in loader:
            x = x.to(device)
            sub_ids = sub_ids.to(device)
            time_idx = time_idx.to(device)
            is_boundary = is_boundary.to(device)

            with torch.no_grad():
                specs = model.encoder.stft(x)
            s1 = masker(specs); s2 = masker(specs)

            optimizer.zero_grad()
            ctx = autocast('cuda') if scaler else torch.enable_grad()
            with ctx:
                _, z1 = model(s1, input_is_spec=True)
                _, z2 = model(s2, input_is_spec=True)
                out = loss_fn(z1, z2, sub_ids, time_idx, is_boundary)
                if isinstance(out, tuple):
                    loss, metrics = out
                else:
                    loss, metrics = out, {'loss_contrastive': out.item(), 'loss_temporal': 0.0}

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            ep_loss += loss.item(); ep_cont += metrics['loss_contrastive']
            ep_temp += metrics['loss_temporal']; n += 1

        avg = {'epoch': epoch, 'total': ep_loss/max(1,n),
               'contrastive': ep_cont/max(1,n), 'temporal': ep_temp/max(1,n)}
        print(f"  [SSL {loss_name}] Ep{epoch}: total={avg['total']:.4f} cont={avg['contrastive']:.4f} temp={avg['temporal']:.4f}")
        history.append(avg)

    torch.save(encoder.state_dict(), out_path)
    return history

# ── Downstream evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_downstream(model, loader, device, return_raw=False):
    """
    Evaluate downstream classifier at the PATIENT level.

    Each patient's window-level probabilities are averaged first,
    and AUROC/AUPRC are then computed across patients.

    Parameters
    ----------
    model : torch.nn.Module
        Trained downstream model.
    loader : DataLoader
        Evaluation DataLoader yielding (x, y, pid).
    device : torch.device
        Evaluation device.
    return_raw : bool, default=False
        If False:
            Returns {'auroc': ..., 'auprc': ...}

        If True:
            Returns (patient_ids, targets, probabilities)

    Returns
    -------
    dict OR tuple
        Patient-level metrics, or raw patient-level predictions.
    """

    model.eval()

    # Collect window-level predictions grouped by patient ID
    pat_probs = {}
    pat_targets = {}

    with torch.no_grad():
        for x, y, pids in loader:
            x = x.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            y_np = y.detach().cpu().numpy()
            pid_np = pids.detach().cpu().numpy()

            for prob, lbl, pid in zip(probs, y_np, pid_np):
                pid = int(pid)

                pat_probs.setdefault(pid, []).append(float(prob))
                pat_targets[pid] = float(lbl)

    # No predictions available
    if len(pat_probs) == 0:
        if return_raw:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32)
            )

        return {
            'auroc': float('nan'),
            'auprc': float('nan')
        }

    # Sort patient IDs explicitly for deterministic ordering
    pat_ids = sorted(pat_probs.keys())

    # Patient-level mean probability
    patient_probs = np.array(
        [np.mean(pat_probs[pid]) for pid in pat_ids],
        dtype=np.float32
    )

    # Patient-level ground-truth labels
    patient_targets = np.array(
        [pat_targets[pid] for pid in pat_ids],
        dtype=np.float32
    )

    patient_ids = np.array(
        pat_ids,
        dtype=np.int64
    )

    # Return raw patient-level predictions for bootstrap/statistical analysis
    if return_raw:
        return patient_ids, patient_targets, patient_probs

    # Metrics require both classes
    if len(np.unique(patient_targets)) < 2:
        return {
            'auroc': float('nan'),
            'auprc': float('nan')
        }

    auroc = roc_auc_score(
        patient_targets,
        patient_probs
    )

    auprc = average_precision_score(
        patient_targets,
        patient_probs
    )

    return {
        'auroc': float(auroc),
        'auprc': float(auprc)
    }

def finetune_downstream(
    encoder_path,
    label_map,
    train_files,
    val_files,
    test_files,
    device,
    predictions_save_path=None,
    seed=None
):
    """
    Downstream frozen-encoder evaluation.

    Protocol:
      1. Explicitly seed downstream training.
      2. Train classifier on TRAIN patients only.
      3. Select checkpoint using VALIDATION AUROC only.
      4. Evaluate TEST exactly once.
      5. Aggregate window predictions to PATIENT level.
      6. Save patient IDs, targets, and probabilities.
    """

    if seed is not None:
        set_seed(seed)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    encoder = STFTEncoder2D(
        in_channels=20,
        embed_dim=128
    )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {encoder_path}"
        )

    encoder.load_state_dict(
        torch.load(
            encoder_path,
            map_location='cpu',
            weights_only=True
        )
    )

    model = SleepApneaClassifier(
        encoder,
        finetune_mode='frozen'
    ).to(device)

    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    criterion = BinaryFocalLossWithLogits(
        alpha=0.55,
        gamma=1.0
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_loader = DataLoader(
        LabeledStreamingDataset(
            train_files,
            label_map,
            is_train=True
        ),
        batch_size=128,
        shuffle=False
    )

    val_loader = DataLoader(
        LabeledStreamingDataset(
            val_files,
            label_map,
            is_train=False
        ),
        batch_size=128,
        shuffle=False
    )

    test_loader = DataLoader(
        LabeledStreamingDataset(
            test_files,
            label_map,
            is_train=False
        ),
        batch_size=128,
        shuffle=False
    )

    # ---------------------------------------------------------
    # Validation-based model selection
    # ---------------------------------------------------------
    best_val_auc = -float('inf')
    best_model_state = None

    import copy

    for epoch in range(1, DS_EPOCHS + 1):

        model.train()

        # Encoder remains frozen, including BN statistics
        model.encoder.eval()

        for x, y, _ in train_loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device).float().view(-1)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        # Validation only — never touch test here
        val_m = evaluate_downstream(
            model,
            val_loader,
            device
        )

        if (
            not np.isnan(val_m['auroc'])
            and val_m['auroc'] > best_val_auc
        ):
            best_val_auc = val_m['auroc']
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    if best_model_state is None:
        raise RuntimeError(
            "No valid validation checkpoint was selected. "
            "Check validation labels and evaluation pipeline."
        )

    # ---------------------------------------------------------
    # FINAL TEST EVALUATION — EXACTLY ONCE
    # ---------------------------------------------------------
    model.load_state_dict(best_model_state)

    test_patient_ids, test_targets, test_probs = evaluate_downstream(
        model,
        test_loader,
        device,
        return_raw=True
    )

    # ---------------------------------------------------------
    # Patient-level metrics
    # ---------------------------------------------------------
    if len(test_targets) == 0:
        raise RuntimeError(
            "Test evaluation returned zero patients."
        )

    if len(np.unique(test_targets)) < 2:
        best_test_metrics = {
            'auroc': float('nan'),
            'auprc': float('nan')
        }

    else:
        best_test_metrics = {
            'auroc': float(
                roc_auc_score(
                    test_targets,
                    test_probs
                )
            ),
            'auprc': float(
                average_precision_score(
                    test_targets,
                    test_probs
                )
            )
        }

    # ---------------------------------------------------------
    # Save patient-level predictions
    # ---------------------------------------------------------
    if predictions_save_path is not None:

        np.savez(
            predictions_save_path,
            patient_ids=test_patient_ids,
            targets=test_targets,
            probs=test_probs
        )

        print(
            f"  [+] Patient-level predictions saved to "
            f"{predictions_save_path}"
        )

    return best_test_metrics

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    label_map = load_bids_labels(TSV_PATH)
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))
    pids, labels = [], []
    pid_to_file = {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pids.append(pid); labels.append(label_map[pid]); pid_to_file[pid] = f
    pids = list(set(pids))
    if len(pids) < 10:
        print("SKIP: insufficient processed files"); return

    # 60/20/20 split by subject
    labels_for_split = [label_map[p] for p in pids]
    trainval_ids, test_ids = train_test_split(pids, test_size=0.20, stratify=labels_for_split, random_state=SEED)
    tv_labels = [label_map[p] for p in trainval_ids]
    train_ids, val_ids   = train_test_split(trainval_ids, test_size=0.25, stratify=tv_labels, random_state=SEED)

    train_files = [pid_to_file[p] for p in train_ids]
    val_files   = [pid_to_file[p] for p in val_ids]
    test_files  = [pid_to_file[p] for p in test_ids]
    ssl_files   = train_files  # only use train subjects for SSL

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)} subjects")

    ablations = [
        ('A0_VanillaNTXent',    VanillaNTXentLoss(temperature=0.5)),
        ('A1_TemporalNTXent',   TemporalNTXentWrapper(temperature=0.5, lambda_decay=0.1)),
        ('A2_NTXent_TempReg',   NTXentPlusTempReg(temperature=0.5, lambda_temporal=0.15)),
        ('A3_PhysioCLR',        PhysioCLRWrapper(temperature=0.07, lambda_temporal=0.15)),
    ]

    all_results = []
    for name, loss_fn in ablations:
        print(f"\n{'='*60}\n  ABLATION: {name}\n{'='*60}")
        enc_path = os.path.join(OUT_DIR, f'encoder_{name}.pth')

        t0 = time.time()
        ssl_hist = pretrain_ssl(name, loss_fn, ssl_files, device, enc_path)
        ssl_time = time.time() - t0

        t1 = time.time()
        pred_path = os.path.join(OUT_DIR, f'{name}_predictions.npz')
        test_metrics = finetune_downstream(enc_path, label_map, train_files, val_files, test_files, device,
                                           predictions_save_path=pred_path)
        ds_time = time.time() - t1

        row = {
            'ablation': name,
            'ssl_final_loss': ssl_hist[-1]['total'],
            'ssl_contrastive': ssl_hist[-1]['contrastive'],
            'ssl_temporal': ssl_hist[-1]['temporal'],
            'test_auroc': test_metrics.get('auroc', float('nan')),
            'test_auprc': test_metrics.get('auprc', float('nan')),
            'ssl_time_s': round(ssl_time, 1),
            'ds_time_s': round(ds_time, 1),
        }
        print(f"  >> {name}: Test AUROC={row['test_auroc']:.4f}, AUPRC={row['test_auprc']:.4f}")
        all_results.append(row)

        # save per-ablation
        with open(os.path.join(OUT_DIR, f'{name}_metrics.json'), 'w') as f:
            json.dump({'row': row, 'ssl_history': ssl_hist}, f, indent=2)

    # Save manifest
    import csv
    with open(os.path.join(OUT_DIR, 'ablation_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

    print("\n\n=== LOSS ABLATION SUMMARY ===")
    for r in all_results:
        print(f"  {r['ablation']:30s}  AUROC={r['test_auroc']:.4f}  AUPRC={r['test_auprc']:.4f}  SSL_loss={r['ssl_final_loss']:.4f}")

if __name__ == '__main__':
    main()
