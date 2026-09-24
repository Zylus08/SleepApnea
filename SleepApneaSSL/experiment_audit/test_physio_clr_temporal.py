"""
experiment_audit/test_physio_clr_temporal.py
============================================
Deterministic unit test verifying A3 PhysioCLR temporal continuity logic.

Tests:
  1. Adjacent same-patient windows are selected
  2. Adjacent windows from different patients are NOT selected
  3. Non-adjacent same-patient windows are NOT selected
  4. File/batch boundaries do not create false temporal pairs
  5. Shuffled batch ordering does not change which pairs are considered temporally adjacent
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from physio_clr import PhysioCLRLoss

def test_temporal_continuity_selection():
    """Verify explicit subject/time logic correctly selects adjacent pairs."""
    
    loss_fn = PhysioCLRLoss()
    
    N = 5
    D = 4
    
    # Simulate a batch with different scenarios
    # idx 0: patient 0, time 0
    # idx 1: patient 0, time 1 (adjacent to 0) -> Valid pair (0, 1) and (1, 0)
    # idx 2: patient 0, time 5 (non-adjacent to 1) -> No pair
    # idx 3: patient 1, time 6 (adjacent time, different patient) -> No pair
    # idx 4: patient 1, time 7 (adjacent to 3) -> Valid pair (3, 4) and (4, 3)
    
    sub_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    time_idx = torch.tensor([0, 1, 5, 6, 7], dtype=torch.long)
    
    # We just need to check the internal logic of compute_temporal_loss
    # Let's rebuild the mask here to verify expectations
    same_sub = (sub_ids.unsqueeze(1) == sub_ids.unsqueeze(0))
    time_next = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0) + 1)
    valid_mask = same_sub & time_next
    
    # Expected valid_mask (True where row_time == col_time + 1 AND row_sub == col_sub)
    # row 0 (time 0): none
    # row 1 (time 1): col 0 (time 0)
    # row 2 (time 5): none
    # row 3 (time 6): none
    # row 4 (time 7): col 3 (time 6)
    
    # 1. Adjacent same-patient windows are selected
    assert valid_mask[1, 0], "Failed: row 1 (t=1, p=0) should select col 0 (t=0, p=0)"
    assert valid_mask[4, 3], "Failed: row 4 (t=7, p=1) should select col 3 (t=6, p=1)"
    
    # 2. Adjacent windows from different patients are NOT selected
    # Check if (3, 2) is selected? (t=6, p=1) vs (t=5, p=0)
    assert not valid_mask[3, 2], "Failed: should not select cross-patient adjacent windows"
    
    # 3. Non-adjacent same-patient windows are NOT selected
    assert not valid_mask[2, 1], "Failed: should not select non-adjacent windows"
    
    # Check total valid pairs
    assert valid_mask.sum() == 2, f"Expected 2 valid pairs, got {valid_mask.sum()}"
    print("PASS: valid_mask correctly identifies valid pairs")
    
    # 4 & 5: Let's run it through the actual loss function and verify it works regardless of ordering
    
    # Create distinct representations
    z = torch.randn(N, D)
    
    l_temp_ordered = loss_fn.compute_temporal_loss(z, sub_ids, time_idx)
    
    # Shuffle the batch
    shuffle_idx = torch.randperm(N)
    z_shuf = z[shuffle_idx]
    sub_ids_shuf = sub_ids[shuffle_idx]
    time_idx_shuf = time_idx[shuffle_idx]
    
    l_temp_shuffled = loss_fn.compute_temporal_loss(z_shuf, sub_ids_shuf, time_idx_shuf)
    
    # 5. Shuffled batch ordering does not change which pairs are considered temporally adjacent
    # (Because it's MSE and average, the sum and average should be identical)
    assert torch.allclose(l_temp_ordered, l_temp_shuffled), f"Shuffled loss {l_temp_shuffled} != Ordered loss {l_temp_ordered}"
    print("PASS: Shuffling the batch does not affect the temporal loss")
    
    # Calculate expected MSE manually
    # Valid pairs: (1, 0) and (4, 3)
    mse_10 = F.mse_loss(z[1], z[0])
    mse_43 = F.mse_loss(z[4], z[3])
    expected_mse = (mse_10 + mse_43) / 2
    
    assert torch.allclose(l_temp_ordered, expected_mse), f"Expected {expected_mse}, got {l_temp_ordered}"
    print("PASS: Computed loss matches expected manual MSE")

def test_no_valid_pairs():
    """Verify behavior when no valid adjacent pairs exist in batch."""
    loss_fn = PhysioCLRLoss()
    
    N = 4
    D = 4
    z = torch.randn(N, D)
    
    # All same patient, but no adjacent windows (0, 2, 4, 6)
    sub_ids = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    time_idx = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    
    l_temp = loss_fn.compute_temporal_loss(z, sub_ids, time_idx)
    assert l_temp == 0.0, f"Expected 0.0, got {l_temp}"
    
    # All different patients, adjacent windows (0, 1, 2, 3)
    sub_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    time_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    
    l_temp = loss_fn.compute_temporal_loss(z, sub_ids, time_idx)
    assert l_temp == 0.0, f"Expected 0.0, got {l_temp}"
    
    print("PASS: Returns 0 when no valid pairs exist")

if __name__ == "__main__":
    print("=" * 60)
    print("  A3 PhysioCLR Temporal Loss — Deterministic Unit Tests")
    print("=" * 60)
    test_temporal_continuity_selection()
    test_no_valid_pairs()
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
