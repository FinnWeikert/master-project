# Autoencoder architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionAutoencoder(nn.Module):
    """
    Conv1D Autoencoder for motion sequences.

    Input shape: (batch, seq_len, feature_dim)
    Output shape: same as input
    Embedding shape: latent_dim
    """

    def __init__(self, feature_dim, latent_dim=16, hidden_channels=[16, 32]):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder
        layers = []
        in_ch = feature_dim
        for out_ch in hidden_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # Compute the output length after pooling dynamically
        self._encoded_length = None  # Will set in forward()

        # Fully connected latent
        self.fc_enc = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_channels[-1])

        # Decoder (upsampling)
        dec_layers = []
        reversed_channels = hidden_channels[::-1]
        for i in range(len(reversed_channels)-1):
            dec_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            dec_layers.append(nn.Conv1d(reversed_channels[i], reversed_channels[i+1], kernel_size=3, padding=1))
            dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        dec_layers.append(nn.Conv1d(reversed_channels[-1], feature_dim, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        """
        # Convert to (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Encoder
        z = self.encoder(x)  # (batch, hidden_ch[-1], seq_len//2^n)
        self._encoded_length = z.shape[2]

        # Global average pooling over time
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)  # (batch, hidden_ch[-1])

        # Latent vector
        latent = self.fc_enc(z)

        # Decoder
        z_dec = self.fc_dec(latent).unsqueeze(-1).repeat(1, 1, self._encoded_length)
        out = self.decoder(z_dec)
        out = out.permute(0, 2, 1)  # back to (batch, seq_len, feature_dim)

        return out, latent

# for later
class GlobalGRU(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x: (batch=1, seq_len, latent_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_dim)
        return h_n[-1]  # return last layer's hidden state (1, hidden_dim)
