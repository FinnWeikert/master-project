# Autoencoder architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionAutoencoder(nn.Module):
    """
    Lightweight Conv1D Autoencoder for low-fps motion sequences with dilated convolutions.
    CPU-friendly and prevents zero outputs for short windows.
    """

    def __init__(self, feature_dim, latent_dim=16,
                 hidden_channels=[16, 32], kernel_size=5,
                 dilations=[1, 2], dropout_prob=0.0):
        super().__init__()

        self.feature_dim = feature_dim
        self.motion_dim = feature_dim - 1
        self.latent_dim = latent_dim

        # --------- ENCODER ---------
        enc_layers = []
        in_ch = self.motion_dim
        for out_ch, dil in zip(hidden_channels, dilations):
            enc_layers += [
                nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=1,                  # no aggressive downsampling
                    padding=dil*(kernel_size//2),
                    dilation=dil
                ),
                nn.ReLU()
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # Latent
        self.fc_enc = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_channels[-1])

        # --------- DECODER ---------
        dec_layers = []
        reversed_channels = hidden_channels[::-1]
        reversed_dilations = dilations[::-1]

        for i in range(len(reversed_channels) - 1):
            dec_layers += [
                nn.Conv1d(
                    reversed_channels[i], reversed_channels[i+1],
                    kernel_size=kernel_size,
                    padding=reversed_dilations[i]*(kernel_size//2),
                    dilation=reversed_dilations[i]
                ),
                nn.ReLU()
            ]

        # Final conv to motion_dim
        dec_layers += [
            nn.Conv1d(reversed_channels[-1], self.motion_dim,
                      kernel_size=kernel_size, padding=kernel_size//2)
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        motion = x[..., :-1]  # (B,T,D-1)
        mask   = x[..., -1:]  # (B,T,1)

        # Encoder
        z = motion.transpose(1, 2)       # (B,C,T)
        z_enc = self.encoder(z)          # (B,C,T)
        z_pool = z_enc.mean(dim=-1)      # (B,C)
        latent = self.fc_enc(z_pool)     # (B,latent_dim)

        # Decoder
        z_dec = self.fc_dec(latent).unsqueeze(-1).repeat(1, 1, z_enc.shape[-1])
        out = self.decoder(z_dec)        # (B,C,T)
        out = out.transpose(1, 2)        # (B,T,C)

        # Crop/pad to match input
        T = motion.shape[1]
        if out.shape[1] > T:
            out = out[:, :T]
        elif out.shape[1] < T:
            pad = T - out.shape[1]
            out = F.pad(out, (0,0,0,pad))

        out = out * mask
        return out, latent





class MotionAutoencoderOld(nn.Module):
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










class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2  # SAME padding

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.relu(x + res)


class MotionAutoencoderTCN(nn.Module):
    """
    TCN Autoencoder for motion sequences with masking support.

    Input:  (batch, seq_len, feature_dim)
    Output: (batch, seq_len, motion_dim), latent (batch, latent_dim)
    """

    def __init__(self, feature_dim, latent_dim=16,
                 channels=[16, 32, 64, 64],
                 kernel_size=3,
                 dilations=[1, 2, 4, 8],
                 dropout_prob=0.1,
                 motion_dropout=0.0):
        super().__init__()

        self.feature_dim = feature_dim
        self.motion_dim = feature_dim - 1
        self.latent_dim = latent_dim
        self.motion_dropout = motion_dropout

        assert len(channels) == len(dilations)

        # =====================
        # TCN ENCODER
        # =====================
        layers = []
        in_ch = self.motion_dim
        for out_ch, dil in zip(channels, dilations):
            layers.append(TCNBlock(
                in_ch, out_ch,
                kernel_size=kernel_size,
                dilation=dil,
                dropout=dropout_prob
            ))
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # =====================
        # LATENT PROJECTION
        # =====================
        self.fc_enc = nn.Linear(channels[-1], latent_dim)
        self.fc_dec = nn.Linear(latent_dim, channels[-1])

        # =====================
        # DECODER (mirrored TCN)
        # =====================
        dec_layers = []
        rev_channels = channels[::-1]
        rev_dil = dilations[::-1]

        for i in range(len(rev_channels) - 1):
            dec_layers.append(TCNBlock(
                rev_channels[i], rev_channels[i+1],
                kernel_size=kernel_size,
                dilation=rev_dil[i],
                dropout=dropout_prob
            ))

        # final projection to motion_dim
        dec_layers.append(nn.Conv1d(
            rev_channels[-1], self.motion_dim, kernel_size=1))

        self.decoder = nn.Sequential(*dec_layers)

    # =====================
    # FORWARD
    # =====================
    def forward(self, x):
        """
        x: (batch, seq_len, feature_dim)
        """

        motion = x[..., :-1]
        valid_mask = x[..., -1].unsqueeze(-1)

        # ====== Motion dropout augmentation ======
        if self.training and self.motion_dropout > 0:
            drop = torch.rand_like(motion) < self.motion_dropout
            motion = motion.masked_fill(drop, 0.0)

        # ====== ENCODER ======
        z = motion.permute(0, 2, 1)         # (b, C, T)
        z = self.encoder(z)                 # (b, H, T)

        # Global average pooling
        z_pool = z.mean(dim=-1)             # (b, H)
        latent = self.fc_enc(z_pool)        # (b, latent_dim)

        # ====== DECODER ======
        z_dec = self.fc_dec(latent).unsqueeze(-1)
        z_dec = z_dec.repeat(1, 1, z.shape[-1])

        recon = self.decoder(z_dec)         # (b, motion_dim, T)
        recon = recon.permute(0, 2, 1)      # (b, T, motion_dim)

        # Mask output
        recon = recon * valid_mask

        return recon, latent

