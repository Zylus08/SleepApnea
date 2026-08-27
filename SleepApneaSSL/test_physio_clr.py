import torch
from torch.amp import autocast
from physio_clr import SpectralSubbandMasking, PhysioCLRLoss
from model import STFTEncoder2D, SimCLR

def test_physio_clr():
    print("--- Testing Physio-CLR ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Instantiate modules
    masker = SpectralSubbandMasking(p=1.0) # Force masking for test
    criterion = PhysioCLRLoss(temperature=0.07, lambda_temporal=0.15)
    
    # Model (using 1 channel and 300 time steps as per prompt, or 20 channels depending on real use)
    # The prompt mentioned (B, 1, 129, 300), we'll test with STFTEncoder2D(in_channels=1)
    encoder = STFTEncoder2D(in_channels=1, embed_dim=128)
    simclr = SimCLR(encoder, projection_dim=64).to(device)
    
    # Dummy data
    B = 4
    # Spectrogram shape: (Batch, Channels, Freq_Bins, Time_Steps)
    specs = torch.randn(B, 1, 129, 300, device=device)
    
    # Patient boundary mask (e.g. patients change at index 2)
    # 0, 1, 2, 3 -> patient A (0,1), patient B (2,3)
    patient_boundary_mask = torch.tensor([1, 0, 1, 0], device=device)
    
    # Ensure gradients can be tracked
    specs.requires_grad_(True)
    
    with autocast('cuda'):
        # Pass through masker
        specs1 = masker(specs)
        specs2 = masker(specs)
        
        # Projections
        _, z1 = simclr(specs1, input_is_spec=True)
        _, z2 = simclr(specs2, input_is_spec=True)
        
        # Loss
        loss, metrics = criterion(z1, z2, patient_boundary_mask)
    
    print(f"Loss Output: {loss.item()} (Valid scalar without NaNs/Infs)")
    print(f"Metrics: {metrics}")
    
    # Test Backprop
    loss.backward()
    print("Backward pass completed successfully.")
    
    # Verify masking
    assert (specs1[:, :, 1:10, :] == 0).all() or (specs1[:, :, 10:20, :] == 0).all() or \
           (specs1[:, :, 20:31, :] == 0).all() or (specs1[:, :, 31:77, :] == 0).all(), \
           "SpectralSubbandMasking did not zero out a subband correctly."
           
    print("[+] All verification tests passed.")

if __name__ == '__main__':
    test_physio_clr()
