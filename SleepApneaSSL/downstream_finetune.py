import os
import glob
import re
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import encoder model definition
from model import STFTEncoder2D


# ---------------------------------------------------------------------------
# BIDS METADATA PARSER
# ---------------------------------------------------------------------------
def load_bids_labels(tsv_path):
    """
    Parses BIDS participants.tsv and maps integer participant IDs to binary labels.
    1.0 = OSA Patient, 0.0 = Healthy Control (HC).
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"[!] BIDS metadata file not found at: '{tsv_path}'")

    print(f"[+] Loading BIDS metadata from '{tsv_path}'...")
    df = pd.read_csv(tsv_path, sep='\t')

    # Auto-detect subject and diagnosis column names
    id_col = next(
        (col for col in df.columns if 'participant' in col.lower() or 'sub' in col.lower()),
        df.columns[0]
    )
    group_col = next(
        (col for col in df.columns if any(k in col.lower() for k in ['group', 'diagnosis', 'condition', 'status'])),
        None
    )

    if group_col is None:
        group_col = df.columns[1]  # Default fallback if keyword not found

    print(f"[i] Using Subject ID Column: '{id_col}' | Diagnosis Column: '{group_col}'")

    label_map = {}
    for _, row in df.iterrows():
        participant_str = str(row[id_col])
        digits = re.findall(r'\d+', participant_str)
        if digits:
            pid = int(digits[0])
            condition = str(row[group_col]).strip().upper()
            # Map OSA/Apnea diagnosis to 1.0 and Healthy Control to 0.0
            is_osa = 1.0 if any(k in condition for k in ['OSA', 'PATIENT', 'APNEA', 'DISEASE']) else 0.0
            label_map[pid] = is_osa

    pos_cnt = sum(1 for v in label_map.values() if v == 1.0)
    print(f"[+] Metadata loaded for {len(label_map)} subjects -> OSA: {pos_cnt} | Healthy: {len(label_map) - pos_cnt}")
    return label_map


# ---------------------------------------------------------------------------
# FOCAL LOSS
# ---------------------------------------------------------------------------
class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with Logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weight for the positive (OSA) class. Higher value
                       penalises false-negatives more. Default: 0.55.
        gamma (float): Focusing parameter that down-weights easy examples.
                       Softened to 1.0 to prevent gradient starvation.
                       Default: 1.0.
        reduction (str): 'mean' | 'sum' | 'none'.
    """
    def __init__(self, alpha: float = 0.55, gamma: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Numerically stable BCE as the base loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t: probability of the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # alpha_t: per-sample class weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal modulation
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# DOWNSTREAM CLASSIFIER MODEL
# ---------------------------------------------------------------------------
class SleepApneaClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=128):
        super().__init__()
        self.encoder = encoder

        # Feature-extractor mode: encoder stays frozen throughout training
        print("[+] Feature Extractor Mode: Freezing SSL encoder weights permanently.")
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 2-layer MLP projection head (trainable)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.classifier(features)
        return logits.view(-1)  # Output shape: [batch_size]


# ---------------------------------------------------------------------------
# STREAMING LABELED DATASET  (yields pid alongside x, y)
# ---------------------------------------------------------------------------
# NOTE: Online augmentations (Gaussian noise, channel cutout) have been
# intentionally removed so the encoder sees clean STFT time-frequency
# patterns during downstream fine-tuning.
# ---------------------------------------------------------------------------
class LabeledStreamingDataset(IterableDataset):
    def __init__(self, file_paths, label_map, is_train=True):
        self.file_paths = file_paths
        self.label_map = label_map
        self.is_train = is_train

    def __iter__(self):
        paths = list(self.file_paths)
        if self.is_train:
            random.shuffle(paths)

        for f in paths:
            filename = os.path.basename(f)
            digits = re.findall(r'\d+', filename)
            if not digits:
                continue

            pid = int(digits[0])
            if pid not in self.label_map:
                print(f"[!] Warning: Participant ID {pid} from '{filename}' not found in TSV metadata. Skipping.")
                continue

            subject_label = self.label_map[pid]
            data = torch.load(f, map_location='cpu', weights_only=True)

            # Extract raw signal array
            if isinstance(data, dict):
                signals = data.get('x', data.get('signals', None))
            elif isinstance(data, (tuple, list)):
                signals = data[0]
            else:
                signals = data[:20] if data.shape[0] > 20 else data

            if signals is None:
                continue

            # Ensure 3D tensor shape (N_windows, 20, 3000)
            if signals.dim() == 2:
                if signals.size(0) == 20:
                    n_win = signals.size(1) // 3000
                    signals = signals[:, :n_win * 3000].view(20, n_win, 3000).permute(1, 0, 2)
                else:
                    n_win = signals.size(0) // 3000
                    signals = signals[:n_win * 3000, :].view(n_win, 3000, 20).permute(0, 2, 1)

            n_windows = signals.size(0)
            indices = list(range(n_windows))
            if self.is_train:
                random.shuffle(indices)

            for w in indices:
                x = signals[w].float()
                y = torch.tensor(subject_label, dtype=torch.float32).reshape(())
                pid_tensor = torch.tensor(pid, dtype=torch.long)
                yield x, y, pid_tensor

            del data, signals
            gc.collect()


# ---------------------------------------------------------------------------
# TRAINING ENGINE
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0

    for x, y, _pid in loader:
        x = x.to(device)
        y = y.to(device).float().view(-1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------
def _youden_threshold(targets: np.ndarray, probs: np.ndarray):
    """Return (best_threshold, auc) using Youden's J statistic."""
    try:
        auc = roc_auc_score(targets, probs)
        fpr, tpr, thresholds = roc_curve(targets, probs)
        best_idx = np.argmax(tpr - fpr)
        return thresholds[best_idx], auc
    except ValueError:
        return 0.5, 0.5


def _compute_metrics(targets: np.ndarray, probs: np.ndarray, label: str = ""):
    """Compute and print Acc / Sens / Spec / F1 / AUC at the given prediction level."""
    threshold, auc = _youden_threshold(targets, probs)
    preds = (probs >= threshold).astype(float)

    acc  = accuracy_score(targets, preds)
    sens = recall_score(targets, preds, zero_division=0)
    tn_fp = np.sum(targets == 0)
    spec = np.sum((preds == 0) & (targets == 0)) / tn_fp if tn_fp > 0 else 0.0
    f1   = f1_score(targets, preds, zero_division=0)

    if label:
        print(f"  [{label}] Threshold: {threshold:.4f} | Acc: {acc*100:.2f}% | "
              f"Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")

    return acc, sens, spec, f1, auc


# ---------------------------------------------------------------------------
# EVALUATION ENGINE  (window-level + patient-level)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """
    Runs inference and logs both window-level and patient-level metrics.

    Patient-level prediction = mean probability across all windows for that patient.
    Returns patient-level (acc, sens, spec, f1, auc) for scheduler/checkpoint logic.
    """
    model.eval()

    # Accumulators keyed by participant ID
    patient_probs   = {}  # pid -> list of window probabilities
    patient_targets = {}  # pid -> ground-truth label

    for x, y, pids in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_np  = y.numpy()
        p_np  = pids.numpy()

        for prob, label, pid in zip(probs, y_np, p_np):
            pid = int(pid)
            if pid not in patient_probs:
                patient_probs[pid] = []
            patient_probs[pid].append(float(prob))
            patient_targets[pid] = float(label)

    # Window-level metrics (all windows flattened)
    all_win_probs = np.concatenate([v for v in patient_probs.values()])
    all_win_targets = np.concatenate([
        np.full(len(patient_probs[pid]), patient_targets[pid])
        for pid in patient_probs
    ])
    _compute_metrics(all_win_targets, all_win_probs, label="Window-Level")

    # Patient-level metrics (mean probability aggregation per patient)
    pat_ids     = list(patient_probs.keys())
    pat_probs   = np.array([np.mean(patient_probs[pid]) for pid in pat_ids])
    pat_targets = np.array([patient_targets[pid]        for pid in pat_ids])
    acc, sens, spec, f1, auc = _compute_metrics(pat_targets, pat_probs, label="Patient-Level")

    return acc, sens, spec, f1, auc


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # File Paths
    data_dir             = r'E:\SleepApneaProcessed'
    tsv_path             = r'E:\SleepApnea\participants.tsv'
    weights_path         = r'E:\SleepApnea\SleepApneaSSL\stft_pretrained_encoder.pth'
    checkpoint_save_path = r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth'

    # 1. Parse BIDS Labels from TSV
    label_map = load_bids_labels(tsv_path)

    # 2. Train / Test Split by Subject ID (Stratified 80/20)
    all_files = glob.glob(os.path.join(data_dir, '*.pt'))

    available_pids = set()
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                available_pids.add(pid)

    available_pids   = list(available_pids)
    available_labels = [label_map[pid] for pid in available_pids]

    train_ids_list, test_ids_list = train_test_split(
        available_pids, test_size=0.2, stratify=available_labels, random_state=42
    )
    train_ids = set(train_ids_list)
    test_ids  = set(test_ids_list)

    train_files, test_files = [], []
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in train_ids:
                train_files.append(f)
            elif pid in test_ids:
                test_files.append(f)

    # 3. Initialize Encoder & Load Pre-trained SSL Weights
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    if os.path.exists(weights_path):
        encoder.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[+] Successfully loaded SSL pre-trained weights from '{weights_path}'")
    else:
        print(f"[!] Warning: Pre-trained weights file '{weights_path}' not found. Initializing randomly.")

    # 4. Focal Loss Criterion (softened to prevent gradient starvation)
    train_labels = [label_map[pid] for pid in train_ids]
    num_healthy  = train_labels.count(0.0)
    num_osa      = train_labels.count(1.0)
    print(f"[i] Class distribution in Train set -> Healthy: {num_healthy} | OSA: {num_osa}")

    criterion = BinaryFocalLossWithLogits(alpha=0.55, gamma=1.0)
    print(f"[+] Using Binary Focal Loss (alpha=0.55, gamma=1.0)")

    # 5. Data Loaders (no online augmentations — clean STFT patterns)
    train_loader = DataLoader(
        LabeledStreamingDataset(train_files, label_map, is_train=True),
        batch_size=128, shuffle=False
    )
    test_loader = DataLoader(
        LabeledStreamingDataset(test_files, label_map, is_train=False),
        batch_size=128, shuffle=False
    )

    # -----------------------------------------------------------------------
    # FEATURE EXTRACTOR TRAINING (encoder frozen, MLP head only)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FEATURE EXTRACTOR: MLP HEAD TRAINING (Encoder Frozen)")
    print("=" * 60)
    model = SleepApneaClassifier(encoder, embed_dim=128).to(device)

    # Only optimise the classifier MLP head
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[i] Trainable parameters: {n_trainable:,} / {n_total:,} total")

    best_auc = 0.0
    num_epochs = 15

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | LR: {current_lr:.2e}")

        acc, sens, spec, f1, auc = evaluate(model, test_loader, device)
        scheduler.step()

        # Checkpoint on best patient-level AUC
        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_save_path)
            print(f"    --> New best patient-level checkpoint saved (AUC: {best_auc:.4f})")

    print(f"\n[+] Training complete. Best patient-level AUC: {best_auc:.4f}")
    print(f"[+] Best checkpoint saved to '{checkpoint_save_path}'")


if __name__ == '__main__':
    main()