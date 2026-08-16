import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    def __init__(self, in_channels=32, embed_dim=128):
        super().__init__()
        # 1D CNN to extract spatial-temporal features
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        # Non-linear projection head (crucial for SimCLR performance)
        embed_dim = self.encoder.fc.out_features
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, x):
        representation = self.encoder(x)
        projection = self.projector(representation)
        return representation, projection