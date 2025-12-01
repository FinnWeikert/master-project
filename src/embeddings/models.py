# Autoencoder architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionAutoencoder(nn.Module):
    """
    Conv1D Autoencoder for motion sequences with masking support.

    Input shape: (batch, seq_len, feature_dim)
        where last channel is "valid" mask ∈ {0,1}

    Decoder reconstructs only motion_dim = feature_dim - 1

    Returns:
        out: (batch, seq_len, motion_dim)
        latent: (batch, latent_dim)
    """

    def __init__(self, feature_dim, latent_dim=16, hidden_channels=[16, 32],
                 dropout_prob=0.0):
        super().__init__()

        self.feature_dim = feature_dim
        self.motion_dim = feature_dim - 1     # exclude "valid"
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob      # motion dropout augmentation

        # =============== ENCODER ===============

        layers = []
        in_ch = self.motion_dim  # << use only motion channels
        for out_ch in hidden_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        self._encoded_length = None

        # Latent
        self.fc_enc = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_channels[-1])

        # =============== DECODER ===============

        dec_layers = []
        reversed_channels = hidden_channels[::-1]
        for i in range(len(reversed_channels)-1):
            dec_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            dec_layers.append(nn.Conv1d(reversed_channels[i], reversed_channels[i+1],
                                        kernel_size=3, padding=1))
            dec_layers.append(nn.ReLU())

        # Final layer → output is only motion channels (NOT valid)
        dec_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        dec_layers.append(nn.Conv1d(reversed_channels[-1], self.motion_dim, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(*dec_layers)

    # ======================================================
    # FORWARD
    # ======================================================
    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        """

        # Split motion vs validity
        motion = x[..., :-1]           # (b, T, motion_dim)
        valid_mask = x[..., -1].unsqueeze(-1)  # (b, T, 1)

        # -------- optional motion dropout augmentation ------
        if self.training and self.dropout_prob > 0:
            dropout_mask = torch.rand_like(motion) < self.dropout_prob
            motion = motion.masked_fill(dropout_mask, 0.0)

        # ====== Encoder ======
        z = motion.permute(0, 2, 1)     # (b, motion_dim, T)
        z = self.encoder(z)             # (b, H, T')
        self._encoded_length = z.shape[2]

        # Global pooling
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)   # (b, H)

        # Latent
        latent = self.fc_enc(z)         # (b, latent_dim)

        # ====== Decoder ======
        z_dec = self.fc_dec(latent).unsqueeze(-1)
        z_dec = z_dec.repeat(1, 1, self._encoded_length)

        recon = self.decoder(z_dec)     # (b, motion_dim, T)
        recon = recon.permute(0, 2, 1)  # (b, T, motion_dim)

        # Mask decoder output so we don't penalize invalid frames
        recon = recon * valid_mask

        return recon, latent


# for later
class GlobalGRU(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x: (batch=1, seq_len, latent_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_dim)
        return h_n[-1]  # return last layer's hidden state (1, hidden_dim)
