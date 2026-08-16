import torch
import numpy as np

class TemporalMasking:
    def __init__(self, mask_ratio=0.2):
        self.mask_ratio = mask_ratio
        
    def __call__(self, x):
        # x shape: (channels, time)
        c, t = x.shape
        mask_len = int(t * self.mask_ratio)
        start = np.random.randint(0, t - mask_len)
        x_aug = x.clone()
        x_aug[:, start:start+mask_len] = 0.0
        return x_aug

class SpatialDropout:
    def __init__(self, drop_ratio=0.15):
        self.drop_ratio = drop_ratio
        
    def __call__(self, x):
        # x shape: (channels, time)
        c, t = x.shape
        drop_count = int(c * self.drop_ratio)
        drop_indices = np.random.choice(c, drop_count, replace=False)
        x_aug = x.clone()
        x_aug[drop_indices, :] = 0.0
        return x_aug

class EEGContrastiveTransform:
    def __init__(self):
        self.temp_mask = TemporalMasking(mask_ratio=0.2)
        self.spatial_drop = SpatialDropout(drop_ratio=0.15)
        
    def apply_transforms(self, x):
        x = self.temp_mask(x)
        x = self.spatial_drop(x)
        # Add subtle Gaussian noise to prevent catastrophic overfitting
        noise = torch.randn_like(x) * 0.05
        return x + noise
        
    def __call__(self, x):
        # SimCLR requires two independent augmented views of the same anchor
        view1 = self.apply_transforms(x)
        view2 = self.apply_transforms(x)
        return view1, view2