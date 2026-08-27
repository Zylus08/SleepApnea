import torch
import torch.nn as nn

class EEGSTFTTransform(nn.Module):
    def __init__(self, n_fft=256, hop_length=32, win_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, x):
        """
        Input x:  (B, C, S) -> (Batch, 20 channels, 3000 samples)
        Output:   (B, C, F, T) -> (Batch, 20 channels, 129 freq_bins, 94 time_frames)
        """
        B, C, S = x.shape
        x_flat = x.view(B * C, S)
        
        stft_out = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        mag = torch.abs(stft_out)
        log_spec = torch.log1p(mag)
        
        _, F, T = log_spec.shape
        log_spec = log_spec.view(B, C, F, T)
        
        mean = log_spec.mean(dim=(-2, -1), keepdim=True)
        std = log_spec.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_spec - mean) / std


class EEGEncoder(nn.Module):
    def __init__(self, in_channels=20, embed_dim=128):
        super().__init__()
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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class STFTEncoder2D(nn.Module):
    def __init__(self, in_channels=20, embed_dim=128):
        super().__init__()
        self.stft = EEGSTFTTransform()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x, input_is_spec=False):
        if not input_is_spec:
            specs = self.stft(x)
        else:
            specs = x
        feats = self.conv_blocks(specs)
        feats = torch.flatten(feats, 1)
        return self.fc(feats)


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.fc.out_features
        
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, x, input_is_spec=False):
        h = self.encoder(x, input_is_spec=input_is_spec)
        z = self.projector(h)
        return h, z

if __name__ == '__main__':
    dummy_input = torch.randn(2, 20, 3000) # Batch of 2, 20 channels, 30s at 100Hz
    model = STFTEncoder2D()
    output = model(dummy_input)
    print(f"STFT Output Shape: {output.shape}") # Expects torch.Size([2, 128])