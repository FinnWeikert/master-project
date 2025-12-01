# generate embeddings for full videos
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingExtractor:
    """
    Extract latent embeddings from full motion sequences using a trained AE.
    """

    def __init__(self, model, window_size=20, step_size=5, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.window_size = window_size
        self.step_size = step_size
        self.device = device

    def extract_embeddings(self, sequence, pad_last=True):
        """
        sequence: torch.Tensor (seq_len, feature_dim)
        Returns:
            embeddings: (num_windows, latent_dim)
            centers:    (num_windows,)  indices of window centers
        """
        sequence = sequence.float()
        seq_len = sequence.shape[0]

        embeddings = []
        centers = []

        with torch.no_grad():
            # Iterate windows
            for start in range(0, seq_len - self.window_size + 1, self.step_size):
                end = start + self.window_size

                window = sequence[start:end].unsqueeze(0).to(self.device)
                _, latent = self.model(window)

                embeddings.append(latent)
                centers.append(start + self.window_size // 2)

            # Optional: pad last window if sequence is short or leftover exists
            if pad_last and (seq_len < self.window_size or (seq_len - self.window_size) % self.step_size != 0):
                pad_start = max(0, seq_len - self.window_size)
                window = sequence[pad_start:].unsqueeze(0).to(self.device)

                # pad if needed
                T = window.shape[1]
                if T < self.window_size:
                    pad_amt = self.window_size - T
                    pad_tensor = torch.zeros((1, pad_amt, window.shape[2]), device=self.device)
                    window = torch.cat([pad_tensor, window], dim=1)

                _, latent = self.model(window)
                embeddings.append(latent)
                centers.append(seq_len - 1)

        embeddings = torch.cat(embeddings, dim=0)  # (N, latent_dim)
        centers = torch.tensor(centers, dtype=torch.long)

        return embeddings.cpu(), centers

