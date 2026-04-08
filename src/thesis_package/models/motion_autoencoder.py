import torch.nn as nn
import torch.nn.functional as F

class MotionAE(nn.Module):
    """
    1D CNN Autoencoder for short invariant motion sequences.
    Forces temporal compression to create a fixed-size, low-dimensional latent space.
    """
    def __init__(self, in_channels=5, latent_dim=8, window_size=45):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        
        # --- ENCODER ---
        # Input: (Batch, 5, T)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2), # Halves T: e.g., 45 -> 22
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            # Crush the remaining temporal dimension completely
            nn.AdaptiveAvgPool1d(1), # Output: (Batch, 32, 1)
            nn.Flatten()             # Output: (Batch, 32)
        )
        
        # Latent projection
        self.fc_mean = nn.Linear(32, latent_dim)
        
        # --- DECODER ---
        # Project latent back up to a sequence
        self.dec_proj = nn.Linear(latent_dim, 32 * (window_size // 2))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), # Doubles T
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(16, in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # PyTorch Conv1d expects (Batch, Channels, Time)
        # Input 'x' from dataset is (Batch, Time, Channels)
        x = x.transpose(1, 2)
        
        # Encode
        encoded = self.encoder(x)
        latent = self.fc_mean(encoded) # (Batch, latent_dim)
        
        # Decode
        # 1. Project and reshape back to sequence form
        T_half = self.window_size // 2
        dec_in = self.dec_proj(latent).view(-1, 32, T_half)
        
        # 2. Upsample
        out = self.decoder(dec_in)
        
        # 3. Handle odd/even frame length mismatches
        if out.shape[2] != self.window_size:
            out = F.interpolate(out, size=self.window_size, mode='linear', align_corners=False)
            
        # Return to (Batch, Time, Channels) for loss calculation
        return out.transpose(1, 2), latent