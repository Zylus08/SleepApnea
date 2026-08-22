import torch
import torch.fft

class FrequencyMasking:
    def __init__(self, max_mask_pct=0.15):
        self.max_mask_pct = max_mask_pct

    def __call__(self, x):
        # x is now shape (Batch, Channels, Time)
        fft_x = torch.fft.rfft(x, dim=-1)
        seq_len = fft_x.size(-1)
        
        mask_len = int(seq_len * torch.rand(1).item() * self.max_mask_pct)
        if mask_len > 0:
            mask_start = torch.randint(0, seq_len - mask_len, (1,)).item()
            fft_x[:, :, mask_start:mask_start+mask_len] = 0
            
        return torch.fft.irfft(fft_x, n=x.size(-1), dim=-1)

class SpatialDropout:
    def __init__(self, drop_ratio=0.15):
        self.drop_ratio = drop_ratio
        
    def __call__(self, x):
        # Generates a random mask for the channels and broadcasts across the batch
        b, c, t = x.shape
        mask = torch.rand((1, c, 1), device=x.device) > self.drop_ratio
        return x * mask.float()

class TemporalMasking:
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
        
    def __call__(self, x):
        b, c, t = x.shape
        mask_len = int(t * self.mask_ratio)
        start = torch.randint(0, t - mask_len, (1,)).item()
        x_aug = x.clone()
        x_aug[:, :, start:start+mask_len] = 0.0
        return x_aug

class ICLRTimeTransform:
    def __init__(self):
        self.freq_mask = FrequencyMasking(max_mask_pct=0.15)
        self.temp_mask = TemporalMasking(mask_ratio=0.15)
        self.spatial_drop = SpatialDropout(drop_ratio=0.15)
        
    def apply_transforms(self, x):
        if torch.rand(1).item() > 0.5:
            x = self.freq_mask(x)
        if torch.rand(1).item() > 0.5:
            x = self.temp_mask(x)
        if torch.rand(1).item() > 0.5:
            x = self.spatial_drop(x)
            
        noise = torch.randn_like(x) * 0.02
        return x + noise
        
    def __call__(self, x):
        view1 = self.apply_transforms(x)
        view2 = self.apply_transforms(x)
        return view1, view2