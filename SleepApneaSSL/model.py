import torch
import torch.nn as nn

class EEGEncoder(nn.Module):
    def __init__(self, in_channels=20, embed_dim=128):
        super().__init__()
        # Fast, memory-efficient 1D CNN blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # --- THE VRAM SAVER ---
        # This collapses the entire temporal sequence down to 1 value per channel.
        # This prevents the Linear layer from exploding to 10+ GB.
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.fc.out_features
        
        # Lightweight Projection Head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z