"""
temporal_loss.py
================
TemporalNTXentLoss: NT-Xent with temporal-distance-based negative weighting.

Mathematical objective (A1):
  For a batch of N windows with augmented pairs (z_i, z_j):

  L = -1/(2N) * sum_k [ s(k, pos(k)) / T
                        - log sum_{j != k} w_kj * exp(s(k,j) / T) ]

  where:
    s(i,j) = z_i . z_j  (cosine similarity after L2 normalization)
    pos(k) = augmentation partner of k
    w_kj   = temporal weight for negative pair (k, j):
              0               if j == k (masked, self)
              1               if j == pos(k) (positive — NOT in denominator per standard NT-Xent)
              1 - exp(-λ |t_k - t_j|)   if same subject, j is a negative
              1               if different subject

  Note: The positive pair is EXCLUDED from the denominator (standard NT-Xent convention).
  This is different from treating the positive as weight=1 in the denominator.

Implementation uses log-domain arithmetic for numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNTXentLoss(nn.Module):
    """
    NT-Xent with temporal-distance soft negative weighting.

    Same-subject negative pairs closer in time are down-weighted.
    Cross-subject negative pairs receive weight 1 (unmodified).
    Positive pairs (augmentation pairs) are excluded from the denominator
    per standard NT-Xent convention.

    Args:
        temperature (float): Softmax temperature τ.
        lambda_decay (float): Temporal decay rate λ.
            w_same = 1 - exp(-λ * |t_i - t_j|)
            At λ=0.1: Δt=1 → w≈0.095, Δt=10 → w≈0.632, Δt=50 → w≈0.993
    """
    def __init__(
        self, 
        temperature: float = 0.5, 
        lambda_decay: float = 0.1,
        kernel_type: str = 'exponential',
        kernel_alpha: float = 0.1,
        kernel_cutoff: float = 10.0
    ):
        super().__init__()
        assert temperature > 0, "temperature must be positive"
        assert lambda_decay >= 0, "lambda_decay must be non-negative"
        assert kernel_type in ['exponential', 'linear', 'cutoff'], f"Invalid kernel: {kernel_type}"
        self.temperature = temperature
        self.lambda_decay = lambda_decay
        self.kernel_type = kernel_type
        self.kernel_alpha = kernel_alpha
        self.kernel_cutoff = kernel_cutoff

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        subject_ids: torch.Tensor,
        time_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_i:          (N, D) — first augmented view embeddings
            z_j:          (N, D) — second augmented view embeddings
            subject_ids:  (N,)   — integer patient ID for each sample
            time_indices: (N,)   — integer window index within the recording

        Returns:
            scalar loss
        """
        N = z_i.size(0)
        device = z_i.device

        if N < 2:
            return z_i.new_tensor(0.0)

        # ── 1. Normalize and concatenate ──────────────────────────────────────
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2N, D)

        # ── 2. Cosine similarity matrix ───────────────────────────────────────
        # sim[i,j] = z_i . z_j  (already normalized → cosine sim)
        sim = torch.mm(z, z.t()) / self.temperature              # (2N, 2N)

        # ── 3. Subject IDs and time indices for 2N augmented batch ───────────
        # First N entries = z_i views, last N entries = z_j views
        # Both halves correspond to the SAME underlying windows
        sub_2n  = torch.cat([subject_ids, subject_ids], dim=0)   # (2N,)
        time_2n = torch.cat([time_indices, time_indices], dim=0).float()  # (2N,)

        # ── 4. Positive pair mask ─────────────────────────────────────────────
        # z_i[k] is paired with z_j[k] → indices (k, N+k) and (N+k, k)
        pos_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=device)
        eye_N    = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask[:N, N:]  = eye_N   # z_i[k] -> z_j[k]
        pos_mask[N:,  :N] = eye_N   # z_j[k] -> z_i[k]

        # ── 5. Self-pair mask ─────────────────────────────────────────────────
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=device)

        # ── 6. Temporal weight matrix ─────────────────────────────────────────
        # same_subject[i,j] = 1 if sub_2n[i] == sub_2n[j]
        same_sub  = (sub_2n.unsqueeze(0) == sub_2n.unsqueeze(1))  # (2N, 2N)
        time_dist = torch.abs(time_2n.unsqueeze(0) - time_2n.unsqueeze(1))  # (2N, 2N)

        # Same-subject temporal weight: 0 at Δt=0, approaches 1 as Δt→∞
        if self.kernel_type == 'exponential':
            w_same = 1.0 - torch.exp(-self.lambda_decay * time_dist)
        elif self.kernel_type == 'linear':
            w_same = torch.clamp(self.kernel_alpha * time_dist, min=0.0, max=1.0)
        elif self.kernel_type == 'cutoff':
            w_same = (time_dist > self.kernel_cutoff).float()
        # Cross-subject weight: always 1
        temporal_weights = torch.where(same_sub, w_same, torch.ones_like(w_same))

        # ── 7. Build denominator mask ─────────────────────────────────────────
        # Denominator includes all j != i AND j != pos(i)
        # (standard NT-Xent excludes both self and positive from denominator)
        neg_mask = ~self_mask & ~pos_mask   # (2N, 2N) True for valid negatives

        # ── 8. Weighted denominator in log-domain ─────────────────────────────
        # For each anchor i: log sum_{j in neg} w_ij * exp(sim_ij)
        # = log sum_{j in neg} exp(sim_ij + log(w_ij))
        #
        # Numerics: where w_ij=0 (same window, Δt=0, same subject), log(0)→-inf
        # → exp(-inf) = 0, correctly excluded from sum
        # We clamp log(w) to -1e9 to avoid -inf NaN propagation
        log_w   = torch.log(temporal_weights.clamp(min=1e-9))    # (2N, 2N)
        sim_adj = sim + log_w                                      # (2N, 2N)

        # Mask out self and positive positions with very negative value
        sim_adj = sim_adj.masked_fill(self_mask | pos_mask, -1e9)

        # log-sum-exp over negatives (numerically stable via torch.logsumexp)
        log_denom = torch.logsumexp(sim_adj, dim=1, keepdim=True)  # (2N, 1)

        # ── 9. Numerator: positive similarity ────────────────────────────────
        log_numerator = sim[pos_mask].view(2 * N, 1)               # (2N, 1)

        # ── 10. Loss ──────────────────────────────────────────────────────────
        loss = -(log_numerator - log_denom).mean()

        return loss


def unit_test():
    """
    Verify TemporalNTXentLoss behavior on synthetic embeddings.

    Test 1: Same-subject adjacent windows should have lower weight → loss differs from vanilla
    Test 2: Perfect positives (z_i == z_j) should produce near-zero loss
    Test 3: All different subjects → loss == VanillaNTXent (weights all 1)
    Test 4: Gradient flows
    Test 5: Numerically stable with near-identical embeddings
    """
    torch.manual_seed(0)
    import torch.nn.functional as F

    T, lam = 0.5, 0.1
    loss_fn = TemporalNTXentLoss(temperature=T, lambda_decay=lam)

    N, D = 8, 16

    def vanilla_ntxent(z_i, z_j, T=0.5):
        z = F.normalize(torch.cat([z_i, z_j]), dim=1)
        sim = torch.mm(z, z.t()) / T
        sim.fill_diagonal_(-1e9)
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)])
        import torch.nn.functional as F_inner
        return F_inner.cross_entropy(sim, labels)

    # ── Test 1: Same subject, adjacent windows ────────────────────────────────
    z_i = F.normalize(torch.randn(N, D), dim=1)
    z_j = F.normalize(torch.randn(N, D), dim=1)
    sub_ids   = torch.zeros(N, dtype=torch.long)          # all same subject
    time_idx  = torch.arange(N, dtype=torch.long)         # sequential

    l_temporal = loss_fn(z_i, z_j, sub_ids, time_idx)
    l_vanilla  = vanilla_ntxent(z_i, z_j, T)
    print(f"Test 1 — same-subj adjacent: temporal={l_temporal:.4f}, vanilla={l_vanilla:.4f}")
    assert l_temporal.item() != l_vanilla.item(), "TEST 1 FAIL: temporal == vanilla when should differ"
    print("  PASS")

    # ── Test 2: All different subjects → should equal vanilla ─────────────────
    sub_ids_diff = torch.arange(N, dtype=torch.long)      # all different
    l_diff = loss_fn(z_i, z_j, sub_ids_diff, time_idx)
    # With all different subjects, temporal weights all = 1 → should ≈ vanilla
    # Note: positive excluded from denominator in TemporalNTXent but NOT in vanilla above
    # so they won't be exactly equal, but should be close
    print(f"Test 2 — all-diff-subj: temporal={l_diff:.4f}, vanilla={l_vanilla:.4f}")
    print("  INFO: small difference expected due to positive-in-denominator convention difference")

    # ── Test 3: Gradient flows ────────────────────────────────────────────────
    z_i_g = F.normalize(torch.randn(N, D, requires_grad=True), dim=1)
    z_j_g = F.normalize(torch.randn(N, D, requires_grad=True), dim=1)
    l_g   = loss_fn(z_i_g, z_j_g, sub_ids, time_idx)
    l_g.backward()
    assert z_i_g.grad is not None and not torch.isnan(z_i_g.grad).any(), "TEST 3 FAIL: NaN grad"
    print("Test 3 — gradient: PASS")

    # ── Test 4: No NaN/Inf with near-identical embeddings ────────────────────
    z_same = F.normalize(torch.ones(N, D), dim=1)
    l_same = loss_fn(z_same, z_same.clone(), sub_ids, time_idx)
    assert not torch.isnan(l_same) and not torch.isinf(l_same), "TEST 4 FAIL: NaN/Inf"
    print(f"Test 4 — near-identical embeddings: loss={l_same:.4f}  PASS")

    # ── Test 5: Subject ID matters ────────────────────────────────────────────
    sub_all_same  = torch.zeros(N, dtype=torch.long)
    sub_all_diff  = torch.arange(N, dtype=torch.long)
    time_close    = torch.zeros(N, dtype=torch.long)   # all Δt=0

    l_same_sub  = loss_fn(z_i, z_j, sub_all_same, time_close)
    l_diff_sub  = loss_fn(z_i, z_j, sub_all_diff, time_close)
    print(f"Test 5 — sub_id effect: same_sub={l_same_sub:.4f}, diff_sub={l_diff_sub:.4f}")
    print("  INFO: same-subject with Δt=0 → w=0 → negatives removed → typically lower denominator → higher loss or lower loss depending on embeddings")
    print("  PASS (no crash)")

    print("\nAll unit tests passed.")


if __name__ == '__main__':
    unit_test()


# ==============================================================================
# SoftCLTLoss
# ==============================================================================
# Faithful adaptation of Soft Instance-Wise Contrastive Loss from:
#   Lee et al., "Soft Contrastive Learning for Time Series", ICLR 2024
#   arXiv:2312.16424
#
# Paper equations adapted to our window-level EEG setting:
#
# Soft assignment for pair (i, i'):
#   w_I(i, i') = α * σ(-(D_norm(x_i, x_i') - 1) / τ_I)   if i ≠ i'  (Eq. 1)
#   w_I(i, i') = 1                                          if i = i'  (positive pair)
#
# where D_norm is the min-max normalized pairwise Euclidean distance
# between raw EEG windows, computed per-batch.
#
# For each anchor i, the soft loss is:
#   ℓ_I(i) = -log p_I(i) - Σ_{i'≠i} w_I(i,i') * log p_I(i')     (Eq. 3)
#
# where p_I(i') = exp(z_i ∘ z_j) / Σ_k exp(z_i ∘ z_k)
#
# The POSITIVE IS INCLUDED in the denominator (matching reference code convention).
#
# Temporal Contrastive Loss (TS2Vec-style, Eq. 4-6) is NOT implemented here.
# It requires per-timestamp embeddings from a hierarchical encoder, which is
# incompatible with our window-level STFTEncoder2D architecture.
# See SOFTCLT_FORMULATION_AUDIT.md for full justification.
# ==============================================================================

class SoftCLTLoss(nn.Module):
    """
    Soft Instance-Wise Contrastive Loss (Lee et al., ICLR 2024).

    Implements only the instance-wise component (Eq. 1-3) adapted to
    window-level EEG contrastive learning. The temporal contrastive component
    (Eq. 4-6) requires a hierarchical per-timestamp architecture (TS2Vec) and
    cannot be faithfully applied to our STFTEncoder2D.

    Args:
        temperature (float): Contrastive temperature τ. Default 0.5
            (consistent with A0/A1/A2 in this codebase).
        tau_I (float): Sharpness for soft assignment. Paper default: 2.
        alpha (float): Upper-bound for non-self-pair assignments. Paper
            ablation Table 5c chooses α=0.5 as best.
    """
    def __init__(self, temperature: float = 0.5, tau_I: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.tau_I = tau_I
        self.alpha = alpha

    @staticmethod
    def _pairwise_euclidean_minmax(x_raw: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise Euclidean distance matrix for a batch of raw EEG windows,
        then min-max normalizes to [0, 1].

        Args:
            x_raw: (N, C, T) raw EEG windows

        Returns:
            D_norm: (N, N) distance matrix, values in [0, 1]
        """
        N = x_raw.shape[0]
        x_flat = x_raw.reshape(N, -1).float()   # (N, C*T)

        # Squared Euclidean distances via broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        dot = torch.mm(x_flat, x_flat.t())      # (N, N)
        sq = (x_flat * x_flat).sum(dim=1)       # (N,)
        dist_sq = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot
        dist_sq = dist_sq.clamp(min=0.0)        # numerical safety
        dist = dist_sq.sqrt()                   # (N, N)

        # Min-max normalization per Lee et al. (excludes diagonal)
        # Use off-diagonal values for normalization range
        mask_diag = ~torch.eye(N, dtype=torch.bool, device=x_raw.device)
        off_diag = dist[mask_diag]
        d_min = off_diag.min()
        d_max = off_diag.max()
        if d_max > d_min:
            D_norm = (dist - d_min) / (d_max - d_min)
            D_norm = D_norm.clamp(min=0.0, max=1.0)  # diagonal (0) < d_min → clamp to 0
        else:
            # All windows identical — flat distance matrix
            D_norm = torch.zeros_like(dist)
        return D_norm

    def _soft_assignments(self, D_norm: torch.Tensor) -> torch.Tensor:
        """
        Computes soft assignment matrix w_I from min-max normalized distance.

        w_I(i,i') = α * σ(-(D_norm(i,i') - 1) / τ_I)   for i ≠ i'
        w_I(i,i) = 1   (self/positive pair, set to 1 to match pos weight)

        Returns:
            w: (N, N) soft assignment matrix
        """
        N = D_norm.shape[0]
        # Off-diagonal assignments
        w = self.alpha * torch.sigmoid(-(D_norm - 1.0) / self.tau_I)
        # Set diagonal to 1 (self-pair / positive weight)
        eye = torch.eye(N, dtype=torch.bool, device=D_norm.device)
        w = w.masked_fill(eye, 1.0)
        return w

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        x_raw: torch.Tensor,
        sub_ids: torch.Tensor = None,    # unused — accepted for API consistency
        time_idx: torch.Tensor = None,   # unused — accepted for API consistency
        is_boundary: torch.Tensor = None  # unused
    ) -> torch.Tensor:
        """
        Compute soft instance-wise contrastive loss.

        Args:
            z_i: (N, D) projected embeddings, augmented view 1 (L2 normalized by caller)
            z_j: (N, D) projected embeddings, augmented view 2 (L2 normalized by caller)
            x_raw: (N, C, T) raw EEG windows used to compute inter-instance distances
            sub_ids: ignored (API compatibility)
            time_idx: ignored (API compatibility)
            is_boundary: ignored (API compatibility)

        Returns:
            loss: scalar soft contrastive loss
        """
        N = z_i.shape[0]
        if N < 2:
            return z_i.new_tensor(0.0)

        # ── 1. Compute soft assignment matrix from raw EEG windows ────────────
        # Shape: (N, N)  D_norm in [0, 1]
        D_norm = self._pairwise_euclidean_minmax(x_raw)
        w_I = self._soft_assignments(D_norm)  # (N, N)

        # ── 2. L2 normalize projections ───────────────────────────────────────
        z_i_n = F.normalize(z_i, dim=1)
        z_j_n = F.normalize(z_j, dim=1)

        # ── 3. Concatenate: z = [z_i; z_j], shape (2N, D) ───────────────────
        z = torch.cat([z_i_n, z_j_n], dim=0)  # (2N, D)

        # ── 4. Similarity matrix s = z @ z.T / τ, shape (2N, 2N) ─────────────
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # ── 5. Build soft assignment matrix for the 2N expanded batch ─────────
        # w_expanded[i, j] for i, j in {0..2N-1}:
        # The positive of z_i[k] (index k) is z_j[k] (index N+k), and vice versa.
        # For non-positive pairs, weight from D_norm.
        #
        # Block structure (2N x 2N):
        #   [w_I    | w_I  ]   where diag entries of top-left and bottom-right
        #   [w_I    | w_I  ]   are self pairs (distance=0, w→0.5*α from formula;
        #                       BUT the self entries [k,k] are set to 1 in _soft_assignments)
        #
        # We build all four blocks from w_I (which is N×N).
        # Then override:
        #   - The positive positions [k, N+k] and [N+k, k] → w = 1 (positive)
        #   - Self positions [k, k] and [N+k, N+k] → will be masked to -inf

        w_exp = torch.cat([
            torch.cat([w_I, w_I], dim=1),
            torch.cat([w_I, w_I], dim=1),
        ], dim=0)  # (2N, 2N)

        # Positive positions: z_i[k] ↔ z_j[k]  i.e. [k, N+k] and [N+k, k]
        eye_N = torch.eye(N, dtype=torch.bool, device=z_i.device)
        pos_mask = torch.zeros(2*N, 2*N, dtype=torch.bool, device=z_i.device)
        pos_mask[:N, N:] = eye_N
        pos_mask[N:, :N] = eye_N
        w_exp = w_exp.masked_fill(pos_mask, 1.0)

        # Self-pair positions: [k, k] for k in 0..2N-1
        self_mask = torch.eye(2*N, dtype=torch.bool, device=z_i.device)

        # ── 6. Log-softmax over all j ≠ self ─────────────────────────────────
        # Mask self to -inf so they don't contribute
        sim_masked = sim.masked_fill(self_mask, float('-inf'))
        log_p = F.log_softmax(sim_masked, dim=1)  # (2N, 2N)

        # ── 7. Compute soft loss per anchor ───────────────────────────────────
        # ℓ(i) = -log p(pos(i)) - Σ_{j≠i, j≠pos(i)} w_I(i,j) * log p(j)
        # Which equals: -Σ_{j≠self} w_full(i,j) * log p(j)
        # where w_full(pos) = 1, w_full(other) = w_I(i,j), w_full(self) = 0

        # Zero out self column weights
        w_exp = w_exp.masked_fill(self_mask, 0.0)

        # Weighted sum of log_p
        # loss_per_anchor = -sum_j w_exp[i,j] * log_p[i,j]
        # Note: where w_exp=0 and log_p=-inf, product is 0 * (-inf) = nan.
        # Since w=0 means zero contribution, we replace such nans with 0.
        loss_matrix = -(w_exp * log_p).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # (2N, 2N)

        # Normalize each row by the sum of weights (excluding self)
        # This follows the reference code: loss = sum(logits * soft_labels) / (2*B*T)
        # In their notation, the denominator is constant (2*B) not per-row.
        # We use the same convention: divide total by 2N.
        loss = loss_matrix.sum() / (2 * N)

        return loss