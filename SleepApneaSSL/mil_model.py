import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, encoder, input_dim=128, attention_dim=64, num_classes=2):
        super().__init__()
        self.encoder = encoder
        
        # We unfreeze the encoder so the whole network learns end-to-end
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        # --- GATED ATTENTION NETWORK ---
        # Learns complex, non-linear relationships between windows
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attention_dim, 1)
        
        # --- FINAL PATIENT CLASSIFIER ---
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape is now: (N_windows, Channels, Time) 
        # A single "batch" is all the windows for ONE patient.
        
        # 1. Map every window to the latent space simultaneously
        h = self.encoder(x)  # Shape: (N_windows, 128)
        
        # 2. Calculate the raw attention score for every window
        A_V = self.attention_V(h)  # Shape: (N_windows, attention_dim)
        A_U = self.attention_U(h)  # Shape: (N_windows, attention_dim)
        A = self.attention_w(A_V * A_U)  # Shape: (N_windows, 1)
        
        # 3. Softmax forces all attention weights to sum to exactly 1.0
        A = torch.softmax(A, dim=0)  
        
        # 4. Matrix multiplication aggregates the bag into one vector
        # A.t() is (1, N_windows) * h is (N_windows, 128) = (1, 128)
        patient_vector = torch.mm(A.t(), h)  
        
        # 5. Output a single prediction for the entire patient
        logits = self.classifier(patient_vector)  # Shape: (1, 2)
        
        return logits, A