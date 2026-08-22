import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, lambda_decay=0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_decay = lambda_decay

    def forward(self, z_i, z_j, subject_ids, time_indices):
        """
        z_i, z_j: Augmented views of the batch (Batch_Size, Latent_Dim)
        subject_ids: (Batch_Size)
        time_indices: (Batch_Size)
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # Combine augmented views
        z = torch.cat([z_i, z_j], dim=0) # (2N, D)
        z = F.normalize(z, dim=1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature # (2N, 2N)
        
        # Expand subject IDs and time indices for the 2N augmented batch
        sub_ids_2n = torch.cat([subject_ids, subject_ids], dim=0)
        time_idx_2n = torch.cat([time_indices, time_indices], dim=0).float()
        
        # Create Subject Match Mask: 1 if same subject, 0 otherwise
        sub_mask = (sub_ids_2n.unsqueeze(0) == sub_ids_2n.unsqueeze(1)).float()
        
        # Compute Temporal Distance Matrix: |t_i - t_j|
        time_dist = torch.abs(time_idx_2n.unsqueeze(0) - time_idx_2n.unsqueeze(1))
        
        # Compute Temporal Weights: W = 1 - exp(-lambda * delta_t)
        # If different subjects, weight defaults to 1.0
        temporal_weights = 1.0 - torch.exp(-self.lambda_decay * time_dist)
        temporal_weights = torch.where(sub_mask == 1, temporal_weights, torch.ones_like(temporal_weights))
        
        # Ensure self-similarity is masked out (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(mask, -1e4)
        
        # Standard positive targets (z_i paired with z_j)
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
        
        positives = sim_matrix[pos_mask].view(2 * batch_size, -1)
        
        # Apply the temporal penalty to the denominator (negatives)
        # Instead of summing exp(sim) for all negatives, we sum (exp(sim) * temporal_weights)
        exp_sim = torch.exp(sim_matrix) * temporal_weights
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Final Log-Sum-Exp contrastive loss
        loss = -torch.log(torch.exp(positives) / (denominator + 1e-8)).mean()
        
        return loss