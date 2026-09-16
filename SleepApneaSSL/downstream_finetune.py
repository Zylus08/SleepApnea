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
import argparse
from itertools import cycle

# Import encoder model definition
from model import STFTEncoder2D
from mmd_loss import GaussianMMDLoss


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


class WeightedBCEWithLogits(nn.Module):
    """
    Binary Cross-Entropy with class-frequency-derived pos_weight.

    Automatically up-weights the minority class to compensate for
    class imbalance without the focusing mechanism of Focal Loss.

    Args:
        pos_weight (float): Weight for the positive (OSA) class.
                            Typically ``n_negative / n_positive``.
        reduction (str): 'mean' | 'sum' | 'none'.
    """
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


# ---------------------------------------------------------------------------
# DOWNSTREAM CLASSIFIER MODEL
# ---------------------------------------------------------------------------
class SleepApneaClassifier(nn.Module):
    """
    Binary classifier that wraps an SSL-pretrained encoder with a
    2-layer MLP head.

    Supports three fine-tuning modes:
        - ``frozen``    : Encoder weights fixed; only the MLP head trains.
        - ``layerwise`` : Encoder gradually unfrozen with discriminative LR.
        - ``full``      : All parameters trainable at a single LR.
    """
    VALID_MODES = ('frozen', 'layerwise', 'full')

    def __init__(self, encoder, embed_dim=128, finetune_mode='frozen'):
        super().__init__()
        self.encoder = encoder
        self._finetune_mode = None  # Will be set by set_finetune_mode()

        # 2-layer MLP projection head (always trainable)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Apply initial finetune mode
        self.set_finetune_mode(finetune_mode)

    def set_finetune_mode(self, mode: str):
        """
        Configure which parameters are trainable.

        Parameters
        ----------
        mode : str
            One of 'frozen', 'layerwise', 'full'.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"finetune_mode must be one of {self.VALID_MODES}, got '{mode}'")

        self._finetune_mode = mode

        if mode == 'frozen':
            print("[+] Finetune mode: FROZEN — encoder weights permanently fixed.")
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif mode == 'layerwise':
            print("[+] Finetune mode: LAYERWISE — encoder unfrozen with discriminative LR.")
            for param in self.encoder.parameters():
                param.requires_grad = True

        elif mode == 'full':
            print("[+] Finetune mode: FULL — all parameters trainable at single LR.")
            for param in self.encoder.parameters():
                param.requires_grad = True

    def get_layerwise_param_groups(self, lr_conv: float, lr_fc: float, lr_head: float):
        """
        Build parameter groups with discriminative learning rates.
        """
        conv_params = []
        fc_params = []

        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if 'fc' in name:
                fc_params.append(param)
            else:
                # Freeze conv blocks by default to preserve universal time-frequency filters
                param.requires_grad = False

        groups = [
            {'params': fc_params, 'lr': lr_fc, 'name': 'encoder_fc'},
            {'params': self.classifier.parameters(), 'lr': lr_head, 'name': 'classifier'},
        ]
        if conv_params:
            groups.insert(0, {'params': conv_params, 'lr': lr_conv, 'name': 'encoder_conv'})
            
        return groups
        
    def forward_with_features(self, x):
        if self._finetune_mode == 'frozen':
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
            
        # Hook post-fc embeddings after GELU, strictly before Dropout
        h = features
        for i in range(3): # Linear -> BatchNorm -> GELU
            h = self.classifier[i](h)
            
        post_fc_embed = h
        
        for i in range(3, len(self.classifier)): # Dropout -> Linear
            h = self.classifier[i](h)
            
        logits = h.view(-1)
        return post_fc_embed, logits

    def forward(self, x):
        _, logits = self.forward_with_features(x)
        return logits


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
def train_one_epoch(model, source_loader, target_loader, optimizer, criterion, mmd_criterion, lambda_mmd, device):
    model.train()
    total_task_loss = 0.0
    total_mmd_loss = 0.0
    total_loss = 0.0
    count = 0

    target_iter = cycle(target_loader)

    for src_x, src_y, _ in source_loader:
        src_x = src_x.to(device)
        src_y = src_y.to(device).float().view(-1)
        
        # Get next unsupervised target batch
        tgt_x, _, _ = next(target_iter)
        tgt_x = tgt_x.to(device)

        optimizer.zero_grad()
        
        # Forward pass source
        features_src, logits_src = model.forward_with_features(src_x)
        
        # Forward pass target
        features_tgt, _ = model.forward_with_features(tgt_x)
        
        # Task Loss (on source)
        l_task = criterion(logits_src, src_y)
        
        # MMD Loss (source vs target representations)
        l_mmd = mmd_criterion(features_src, features_tgt)
        
        # Total Loss
        loss = l_task + lambda_mmd * l_mmd
        
        loss.backward()
        optimizer.step()

        total_task_loss += l_task.item()
        total_mmd_loss += l_mmd.item()
        total_loss += loss.item()
        count += 1

    return total_task_loss / max(1, count), total_mmd_loss / max(1, count), total_loss / max(1, count)


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
def _select_criterion(loss_type: str, num_healthy: int, num_osa: int):
    """
    Instantiate the loss function based on the selected type.

    Parameters
    ----------
    loss_type : str
        One of 'focal' or 'weighted_bce'.
    num_healthy : int
        Number of healthy (negative) subjects in training set.
    num_osa : int
        Number of OSA (positive) subjects in training set.
    """
    if loss_type == 'focal':
        criterion = BinaryFocalLossWithLogits(alpha=0.55, gamma=1.0)
        print(f"[+] Using Binary Focal Loss (alpha=0.55, gamma=1.0)")
    elif loss_type == 'weighted_bce':
        pos_weight = num_healthy / max(num_osa, 1)
        criterion = WeightedBCEWithLogits(pos_weight=pos_weight)
        print(f"[+] Using Weighted BCE Loss (pos_weight={pos_weight:.2f})")
    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Use 'focal' or 'weighted_bce'.")
    return criterion


def main():
    parser = argparse.ArgumentParser(description="Downstream Fine-Tuning for Sleep Apnea Classification")
    parser.add_argument('--finetune-mode', type=str, default='frozen', choices=['frozen', 'layerwise', 'full'],
                        help="Fine-tuning mode: 'frozen', 'layerwise', or 'full'")
    parser.add_argument('--loss-type', type=str, default='focal', choices=['focal', 'weighted_bce'],
                        help="Loss type: 'focal' or 'weighted_bce'")
    parser.add_argument('--lr-conv', type=float, default=1e-5, help="Base learning rate for conv block")
    parser.add_argument('--lr-post-fc', type=float, default=1e-5, help="Learning rate for post_fc layer")
    parser.add_argument('--lr-head', type=float, default=1e-3, help="Learning rate for MLP classifier head")
    parser.add_argument('--mmd-lambda', type=float, default=0.15, help="Weight for the MMD loss")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--data-dir', type=str, default=r'E:\SleepApneaProcessed', help="Directory with processed .pt files (source)")
    parser.add_argument('--target-data-dir', type=str, default=r'E:\SleepApneaProcessed', help="Directory with processed .pt files (target)")
    parser.add_argument('--tsv-path', type=str, default=r'E:\SleepApnea\participants.tsv', help="Path to BIDS participants.tsv")
    parser.add_argument('--weights-path', type=str, default=r'E:\SleepApnea\SleepApneaSSL\stft_pretrained_encoder.pth', help="Path to pretrained encoder weights")
    parser.add_argument('--checkpoint-path', type=str, default=r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth', help="Path to save the best model")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------------
    data_dir             = args.data_dir
    tsv_path             = args.tsv_path
    weights_path         = args.weights_path
    checkpoint_save_path = args.checkpoint_path

    # Finetune mode: 'frozen' | 'layerwise' | 'full'
    finetune_mode = args.finetune_mode

    # Loss type: 'focal' | 'weighted_bce'
    loss_type = args.loss_type

    # Training hyperparameters

    num_epochs = args.epochs

    # -----------------------------------------------------------------------
    # 1. Parse BIDS Labels from TSV
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 3. Initialize Encoder & Load Pre-trained SSL Weights
    # -----------------------------------------------------------------------
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    if os.path.exists(weights_path):
        encoder.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"[+] Successfully loaded SSL pre-trained weights from '{weights_path}'")
    else:
        print(f"[!] Warning: Pre-trained weights file '{weights_path}' not found. Initializing randomly.")

    # -----------------------------------------------------------------------
    # 4. Loss Criterion
    # -----------------------------------------------------------------------
    train_labels = [label_map[pid] for pid in train_ids]
    num_healthy  = train_labels.count(0.0)
    num_osa      = train_labels.count(1.0)
    print(f"[i] Class distribution in Train set -> Healthy: {num_healthy} | OSA: {num_osa}")

    criterion = _select_criterion(loss_type, num_healthy, num_osa)

    # -----------------------------------------------------------------------
    # 5. Data Loaders
    # -----------------------------------------------------------------------
    train_loader = DataLoader(
        LabeledStreamingDataset(train_files, label_map, is_train=True),
        batch_size=128, shuffle=False
    )
    test_loader = DataLoader(
        LabeledStreamingDataset(test_files, label_map, is_train=False),
        batch_size=128, shuffle=False
    )
    
    # Target Dataloader for Sleep-EDF
    target_files = glob.glob(os.path.join(args.target_data_dir, '*.pt'))
    # Use a dummy label map since labels aren't used for MMD
    dummy_label_map = {int(re.findall(r'\d+', os.path.basename(f))[0]): 0.0 for f in target_files if re.findall(r'\d+', os.path.basename(f))}
    target_loader = DataLoader(
        LabeledStreamingDataset(target_files, dummy_label_map, is_train=True),
        batch_size=128, shuffle=False
    )
    
    mmd_criterion = GaussianMMDLoss(kernel_mul=2.0, kernel_num=5)

    # -----------------------------------------------------------------------
    # 6. Model & Optimizer (mode-aware)
    # -----------------------------------------------------------------------
    mode_label = {
        'frozen':    'FEATURE EXTRACTOR: MLP HEAD TRAINING (Encoder Frozen)',
        'layerwise': 'LAYERWISE FINE-TUNING (Discriminative LR)',
        'full':      'FULL FINE-TUNING (All Parameters)',
    }
    print("\n" + "=" * 60)
    print(f"  {mode_label[finetune_mode]}")
    print("=" * 60)

    model = SleepApneaClassifier(encoder, embed_dim=128, finetune_mode=finetune_mode).to(device)

    if finetune_mode == 'layerwise':
        param_groups = model.get_layerwise_param_groups(args.lr_conv, args.lr_post_fc, args.lr_head)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        for pg in param_groups:
            print(f"  [{pg.get('name', '?')}] LR: {pg['lr']:.2e} | Params: {sum(p.numel() for p in pg['params']):,}")
    elif finetune_mode == 'full':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_head, weight_decay=1e-4)
    else:  # frozen
        optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr_head, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[i] Trainable parameters: {n_trainable:,} / {n_total:,} total")

    # -----------------------------------------------------------------------
    # 7. Training Loop
    # -----------------------------------------------------------------------
    best_auc = 0.0

    for epoch in range(1, num_epochs + 1):
        t_task, t_mmd, t_total = train_one_epoch(model, train_loader, target_loader, optimizer, criterion, mmd_criterion, args.mmd_lambda, device)
        current_lr = optimizer.param_groups[-1]['lr'] # Get head LR
        print(f"\nEpoch {epoch:02d}/{num_epochs:02d} | Task Loss: {t_task:.4f} | MMD Loss: {t_mmd:.4f} | Total Loss: {t_total:.4f} | Head LR: {current_lr:.2e}")

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