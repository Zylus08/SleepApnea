import os
import torch
import numpy as np

def test_preprocessing():
    processed_dir = 'E:/SleepApneaProcessed_V2'
    files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    if not files:
        print("No processed files found yet.")
        return

    # Check a few random files
    for fname in files[:5]:
        fpath = os.path.join(processed_dir, fname)
        data = torch.load(fpath, weights_only=True)
        
        # Verify shape (20, N)
        assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D"
        assert data.size(0) == 20, f"Expected 20 channels, got {data.size(0)}"
        
        # Verify sample count is divisible by 3000 (30 seconds at 100 Hz)
        assert data.size(1) % 3000 == 0, f"Expected multiple of 3000 samples, got {data.size(1)}"
        
        # Verify no NaNs
        assert not torch.isnan(data).any(), f"Found NaNs in {fname}"
        
        print(f"File {fname}: Shape={data.shape} - Validated.")

if __name__ == "__main__":
    test_preprocessing()
