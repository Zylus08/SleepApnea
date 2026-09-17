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
            bag = torch.load(f, map_location='cpu', weights_only=True)
            if isinstance(bag, torch.Tensor) and bag.dim() == 2 and bag.shape[0] == 20:
                n_win = bag.shape[1] // 3000
                bag = bag[:, :n_win*3000].view(20, n_win, 3000).permute(1, 0, 2)
            elif not isinstance(bag, torch.Tensor) or bag.dim() != 3:
                continue
            n_win = bag.shape[0]
            for w in range(n_win):
                is_boundary = int(w == 0)
                yield bag[w].float(), torch.tensor(w, dtype=torch.long), torch.tensor(is_boundary)
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
        if is_boundary is not None:
            B = z_n.shape[0]
            if B < 2: return l_nt, {'loss_contrastive': l_nt.item(), 'loss_temporal': 0.0}
            valid = (1.0 - is_boundary[1:].float())
            mse = F.mse_loss(z_n[:-1], z_n[1:], reduction='none').mean(1)
            n_valid = valid.sum()
            l_temp = (mse * valid).sum() / n_valid if n_valid > 0 else z_n.new_tensor(0.)
        else:
            l_temp = z_n.new_tensor(0.)
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
        for x, time_idx, is_boundary in loader:
            x = x.to(device)
            is_boundary = is_boundary.to(device)
            sub_ids = torch.zeros(x.shape[0], dtype=torch.long, device=device)  # single cohort
            time_idx = time_idx.to(device)

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
def evaluate_downstream(model, loader, device):
    model.eval()
    pat_probs = {}; pat_targets = {}
    for x, y, pids in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        for prob, lbl, pid in zip(probs, y.numpy(), pids.numpy()):
            pid = int(pid)
            pat_probs.setdefault(pid, []).append(float(prob))
            pat_targets[pid] = float(lbl)
    pat_ids = list(pat_probs)
    pp = np.array([np.mean(pat_probs[p]) for p in pat_ids])
    pt = np.array([pat_targets[p] for p in pat_ids])
    if len(set(pt)) < 2:
        return {'auroc': float('nan'), 'auprc': float('nan')}
    return {'auroc': roc_auc_score(pt, pp), 'auprc': average_precision_score(pt, pp)}

def finetune_downstream(encoder_path, label_map, train_files, val_files, test_files, device):
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
    model = SleepApneaClassifier(encoder, finetune_mode='frozen').to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = BinaryFocalLossWithLogits(alpha=0.55, gamma=1.0)

    train_loader = DataLoader(LabeledStreamingDataset(train_files, label_map, is_train=True), batch_size=128, shuffle=False)
    val_loader   = DataLoader(LabeledStreamingDataset(val_files,   label_map, is_train=False), batch_size=128, shuffle=False)
    test_loader  = DataLoader(LabeledStreamingDataset(test_files,  label_map, is_train=False), batch_size=128, shuffle=False)

    best_val_auc = 0.; best_test_metrics = {}
    for epoch in range(1, DS_EPOCHS+1):
        model.train(); model.encoder.eval()
        for x, y, _ in train_loader:
            x = x.to(device); y = y.to(device).float().view(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        val_m = evaluate_downstream(model, val_loader, device)
        if val_m['auroc'] > best_val_auc:
            best_val_auc = val_m['auroc']
            best_test_metrics = evaluate_downstream(model, test_loader, device)
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
        test_metrics = finetune_downstream(enc_path, label_map, train_files, val_files, test_files, device)
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
