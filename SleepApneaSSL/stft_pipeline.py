import torch
import torch.nn as nn

class EEGSTFTTransform(nn.Module):
    def __init__(self, n_fft=256, hop_length=32, win_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Register Hann window as a buffer so it moves to GPU automatically
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, x):
        """
        Input x:  (B, C, S) -> (Batch, 20 channels, 3000 samples)
        Output:   (B, C, F, T) -> (Batch, 20 channels, 129 freq_bins, 94 time_frames)
        """
        B, C, S = x.shape
        x_flat = x.view(B * C, S)
        
        # Compute Short-Time Fourier Transform
        stft_out = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        # Compute Log-Power Spectrogram: log(1 + |STFT|)
        mag = torch.abs(stft_out)
        log_spec = torch.log1p(mag)
        
        # Reshape back to (B, C, F, T)
        _, F, T = log_spec.shape
        log_spec = log_spec.view(B, C, F, T)
        
        # Per-channel z-score standardization across time/freq dimensions
        mean = log_spec.mean(dim=(-2, -1), keepdim=True)
        std = log_spec.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_spec - mean) / std