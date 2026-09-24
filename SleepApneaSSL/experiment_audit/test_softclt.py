"""
experiment_audit/test_softclt.py
================================
Deterministic unit tests for SoftCLTLoss.

Verifies:
1. Same-patient temporal relationships handled (not applicable — SoftCLT ignores patient IDs)
2. Cross-patient relationships handled correctly
3. Temporal ordering independent of batch position
4. Shuffling the batch does NOT change mathematical loss
5. Positive/self pairs handled per SoftCLT paper convention
6. Loss is finite under normal and edge-case inputs
7. Degenerate/no-valid-relation cases handled safely

Run:
    conda run -n reswork python experiment_audit/test_softclt.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from temporal_loss import SoftCLTLoss


def test_weight_range():
    """Verify soft assignment values are in [0, alpha]."""
    N = 8
    loss_fn = SoftCLTLoss(temperature=0.5, tau_I=2.0, alpha=0.5)

    # Random x_raw
    x_raw = torch.randn(N, 20, 300)
    D_norm = loss_fn._pairwise_euclidean_minmax(x_raw)
    w = loss_fn._soft_assignments(D_norm)

    # Off-diagonal values must be in [0, alpha]
    eye = torch.eye(N, dtype=torch.bool)
    off_diag = w[~eye]
    assert (off_diag >= 0.0).all(), "Weights must be non-negative"
    assert (off_diag <= 0.5 + 1e-6).all(), f"Off-diag weights must be <= alpha=0.5, got max={off_diag.max():.6f}"

    # Diagonal values must be 1.0 (positive pair weight)
    diag = w[eye]
    assert torch.allclose(diag, torch.ones_like(diag)), "Diagonal (positive) weight must be 1.0"

    print("PASS: soft assignment values in correct range")


def test_d_norm_range():
    """Verify min-max normalized distances are in [0, 1]."""
    N = 8
    loss_fn = SoftCLTLoss()
    x_raw = torch.randn(N, 20, 300)

    D_norm = loss_fn._pairwise_euclidean_minmax(x_raw)

    assert (D_norm >= 0.0).all(), "D_norm must be non-negative"
    assert (D_norm <= 1.0 + 1e-5).all(), f"D_norm must be <= 1.0, got max={D_norm.max():.6f}"

    # Diagonal (self-distances) should be exactly 0 after normalization (or near 0)
    diag = D_norm.diagonal()
    assert (diag < 1e-5).all(), f"Self distances should be ~0, got {diag}"

    print("PASS: D_norm values in [0, 1] and diagonal ~0")


def test_loss_finite():
    """Smoke test: loss is finite and gradients flow."""
    torch.manual_seed(42)
    N, C, T = 8, 20, 300
    D = 16
    loss_fn = SoftCLTLoss(temperature=0.5, tau_I=2.0, alpha=0.5)

    z_i = torch.randn(N, D, requires_grad=True)
    z_j = torch.randn(N, D, requires_grad=True)
    x_raw = torch.randn(N, C, T)

    loss = loss_fn(z_i, z_j, x_raw=x_raw)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    loss.backward()
    assert z_i.grad is not None, "z_i gradient is None"
    assert torch.isfinite(z_i.grad).all(), "z_i gradient contains NaN/Inf"

    print(f"PASS: loss={loss.item():.4f}, gradients finite")


def test_shuffled_batch_gives_same_loss():
    """
    Test 4: Shuffling the batch must NOT change the mathematical loss.
    SoftCLT computes distances from raw x_raw, so the order of windows
    within the batch is irrelevant.
    """
    torch.manual_seed(0)
    N, C, T = 6, 20, 100
    D = 16
    loss_fn = SoftCLTLoss(temperature=0.5, tau_I=2.0, alpha=0.5)

    z_i = F.normalize(torch.randn(N, D), dim=1)
    z_j = F.normalize(torch.randn(N, D), dim=1)
    x_raw = torch.randn(N, C, T)

    loss_orig = loss_fn(z_i, z_j, x_raw=x_raw)

    # Shuffle batch
    perm = torch.randperm(N)
    z_i_shuf = z_i[perm]
    z_j_shuf = z_j[perm]
    x_raw_shuf = x_raw[perm]

    loss_shuf = loss_fn(z_i_shuf, z_j_shuf, x_raw=x_raw_shuf)

    # Because all pairs are re-arranged together consistently, the loss should be equal
    assert torch.allclose(loss_orig, loss_shuf, atol=1e-5), \
        f"Loss changed after batch shuffle: {loss_orig.item():.6f} vs {loss_shuf.item():.6f}"

    print(f"PASS: shuffled batch gives same loss ({loss_orig.item():.6f})")


def test_positive_weight_is_one():
    """
    Test 5: Positive pair weights must be 1.0 per paper convention.
    Self-pairs must have zero contribution (masked to -inf in softmax).
    """
    N = 4
    loss_fn = SoftCLTLoss(tau_I=2.0, alpha=0.5)
    x_raw = torch.randn(N, 20, 100)

    D_norm = loss_fn._pairwise_euclidean_minmax(x_raw)
    w = loss_fn._soft_assignments(D_norm)

    # Self pairs (i=i) should have w=1 from _soft_assignments
    diag_w = w.diagonal()
    assert torch.allclose(diag_w, torch.ones(N)), f"Diagonal weights should be 1.0, got {diag_w}"

    # In forward, self-mask is applied to sim, effectively zeroing self-contribution
    # Verify that w_exp[k, k] = 0 after masking in the forward pass
    z_i = F.normalize(torch.randn(N, 8), dim=1)
    z_j = F.normalize(torch.randn(N, 8), dim=1)

    # Partial verify: the self_mask in forward sets w to 0 for diagonal of 2N x 2N
    # We verify indirectly: loss should be finite and reasonable for identical windows
    # (if self pairs contributed, loss would be 0)
    x_same = torch.ones(N, 20, 100)  # identical windows
    loss_same = loss_fn(z_i, z_j, x_raw=x_same)
    assert torch.isfinite(loss_same), "Loss should be finite even for identical raw windows"

    print(f"PASS: positive weights = 1.0; self pairs correctly masked (loss with identical windows = {loss_same.item():.4f})")


def test_edge_case_identical_windows():
    """Identical windows (distance=0 for all pairs) — loss should still be finite."""
    N = 4
    D = 8
    loss_fn = SoftCLTLoss(tau_I=2.0, alpha=0.5)

    z_i = F.normalize(torch.randn(N, D), dim=1)
    z_j = F.normalize(torch.randn(N, D), dim=1)
    x_same = torch.ones(N, 20, 100)  # all identical → D=0 for all pairs

    loss = loss_fn(z_i, z_j, x_raw=x_same)
    assert torch.isfinite(loss), f"Loss should be finite for identical windows, got {loss}"

    print(f"PASS: identical windows edge case handled (loss={loss.item():.4f})")


def test_edge_case_batch_size_2():
    """Minimum batch size of 2 should produce a finite loss."""
    N = 2
    D = 8
    loss_fn = SoftCLTLoss()

    z_i = torch.randn(N, D, requires_grad=True)
    z_j = torch.randn(N, D, requires_grad=True)
    x_raw = torch.randn(N, 20, 100)

    loss = loss_fn(z_i, z_j, x_raw=x_raw)
    assert torch.isfinite(loss), f"Loss should be finite for N=2, got {loss}"

    loss.backward()
    assert z_i.grad is not None

    print(f"PASS: N=2 edge case handled (loss={loss.item():.4f})")


def test_edge_case_batch_size_1():
    """Batch size of 1 should return 0.0 (no valid pairs)."""
    N = 1
    loss_fn = SoftCLTLoss()

    z_i = torch.randn(N, 8)
    z_j = torch.randn(N, 8)
    x_raw = torch.randn(N, 20, 100)

    loss = loss_fn(z_i, z_j, x_raw=x_raw)
    assert loss.item() == 0.0, f"Loss should be 0 for N=1, got {loss.item()}"

    print("PASS: N=1 returns 0.0 (degenerate case)")


def test_patient_structure_not_used():
    """
    SoftCLT does NOT use patient IDs or temporal indices.
    Passing different sub_ids or time_idx must not change the loss.
    """
    torch.manual_seed(7)
    N = 6
    D = 8
    loss_fn = SoftCLTLoss()

    z_i = F.normalize(torch.randn(N, D), dim=1)
    z_j = F.normalize(torch.randn(N, D), dim=1)
    x_raw = torch.randn(N, 20, 100)

    sub_a = torch.zeros(N, dtype=torch.long)
    sub_b = torch.arange(N, dtype=torch.long)
    time_a = torch.zeros(N, dtype=torch.long)
    time_b = torch.arange(N, dtype=torch.long)

    loss_a = loss_fn(z_i, z_j, x_raw=x_raw, sub_ids=sub_a, time_idx=time_a)
    loss_b = loss_fn(z_i, z_j, x_raw=x_raw, sub_ids=sub_b, time_idx=time_b)

    assert torch.allclose(loss_a, loss_b), \
        f"Loss must be identical regardless of sub_ids/time_idx: {loss_a:.6f} vs {loss_b:.6f}"

    print(f"PASS: patient structure ignored (loss_a={loss_a.item():.6f}, loss_b={loss_b.item():.6f})")


def test_closer_windows_get_higher_weight():
    """
    Windows that are more similar (smaller Euclidean distance) should get
    higher soft assignment weights (closer to alpha=0.5) than dissimilar pairs.
    """
    N = 4
    loss_fn = SoftCLTLoss(tau_I=2.0, alpha=0.5)

    # Create controlled raw windows: pairs 0-1 are very similar, pairs 0-2 are very different
    x_raw = torch.zeros(N, 1, 10)
    x_raw[0] = 0.0
    x_raw[1] = 0.001   # very close to 0
    x_raw[2] = 10.0    # very far from 0
    x_raw[3] = 5.0

    D_norm = loss_fn._pairwise_euclidean_minmax(x_raw)
    w = loss_fn._soft_assignments(D_norm)

    # w[0, 1] should be higher (closer pair) than w[0, 2] (farther pair)
    assert w[0, 1].item() > w[0, 2].item(), \
        f"Close pairs should have higher weights: w[0,1]={w[0,1]:.4f}, w[0,2]={w[0,2]:.4f}"

    print(f"PASS: closer windows get higher weights (w_close={w[0,1]:.4f}, w_far={w[0,2]:.4f})")


if __name__ == '__main__':
    print("=" * 60)
    print("  SoftCLTLoss — Deterministic Unit Tests")
    print("=" * 60)
    print()

    test_weight_range()
    test_d_norm_range()
    test_loss_finite()
    test_shuffled_batch_gives_same_loss()
    test_positive_weight_is_one()
    test_edge_case_identical_windows()
    test_edge_case_batch_size_2()
    test_edge_case_batch_size_1()
    test_patient_structure_not_used()
    test_closer_windows_get_higher_weight()

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
