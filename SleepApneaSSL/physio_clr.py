import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SpectralSubbandMasking(nn.Module):
    """
    Randomly zeroes out a physiological EEG sub-band with probability p=0.5.
    Assumes STFT input of shape (Batch, Channels, Freq_Bins, Time_Steps).
    Freq bins derived from sr=100 Hz, N_fft=256 -> bin resolution ≈ 0.39 Hz.
    Total bins = 129 (0 to 50 Hz).
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        # Bin indices for each sub-band based on ~0.39 Hz resolution
        self.bands = {
            'delta': (1, 10),   # 0.5 - 4 Hz
            'theta': (10, 20),  # 4 - 8 Hz
            'alpha': (20, 31),  # 8 - 12 Hz
            'beta': (31, 77)    # 12 - 30 Hz
        }

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        x_masked = x.clone()
        band_name = random.choice(list(self.bands.keys()))
        f_start, f_end = self.bands[band_name]
        
        # Zero out the selected frequency band across all channels and time steps
        x_masked[:, :, f_start:f_end, :] = 0.0
        
        return x_masked

class PhysioCLRLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_temporal=0.15):
        super().__init__()
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def compute_temporal_loss(self, z, sub_ids, time_idx):
        """
        Computes the Temporal Continuity Loss (MSE between consecutive representations)
        using explicit subject and time indices to find valid temporally adjacent pairs.
        z: (Batch, Dim)
        sub_ids: (Batch,)
        time_idx: (Batch,)
        """
        B = z.size(0)
        if B < 2 or sub_ids is None or time_idx is None:
            return torch.tensor(0.0, device=z.device)

        # Find valid temporally adjacent pairs within the batch
        same_sub = (sub_ids.unsqueeze(1) == sub_ids.unsqueeze(0))
        time_next = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0) + 1)
        valid_mask = same_sub & time_next
        
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # (B, B, Dim)
        mse_dist = diff.pow(2).mean(dim=-1)     # (B, B)
        
        n_valid = valid_mask.sum()
        if n_valid > 0:
            return (mse_dist * valid_mask).sum() / n_valid
        return torch.tensor(0.0, device=z.device)

    def forward(self, z1, z2, sub_ids=None, time_idx=None):
        """
        z1, z2: Projected representations (Batch, Dim)
        sub_ids: (Batch,) patient IDs
        time_idx: (Batch,) temporal window indices
        """
        batch_size = z1.size(0)
        
        # --- InfoNCE Loss ---
        # L2 normalize
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        z = torch.cat([z1_norm, z2_norm], dim=0)
        sim_matrix = self.cosine_sim(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        negatives = sim_matrix[mask].view(2 * batch_size, -1)
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss_contrastive = F.cross_entropy(logits, labels)
        
        # --- Temporal Continuity Loss ---
        loss_temporal_1 = self.compute_temporal_loss(z1_norm, sub_ids, time_idx)
        loss_temporal_2 = self.compute_temporal_loss(z2_norm, sub_ids, time_idx)
        loss_temporal = (loss_temporal_1 + loss_temporal_2) / 2.0
        
        # Total Loss
        loss_total = loss_contrastive + self.lambda_temporal * loss_temporal
        
        return loss_total, {
            'loss_total': loss_total.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_temporal': loss_temporal.item()
        }
