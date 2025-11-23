# generate embeddings for full videos
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingExtractor:
    """
    Extract embeddings from full motion sequences using a trained MotionAutoencoder.

    Supports sliding windows and returns embeddings per window.
    """

    def __init__(self, model, window_size=30, step_size=10, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.window_size = window_size
        self.step_size = step_size
        self.device = device

    def extract_embeddings(self, sequence):
        """
        sequence: torch.Tensor, shape (seq_len, feature_dim)
        Returns: torch.Tensor of embeddings, shape (num_windows, latent_dim)
        """
        seq_len, feature_dim = sequence.shape
        embeddings = []

        with torch.no_grad():
            for start in range(0, seq_len - self.window_size + 1, self.step_size):
                window = sequence[start:start+self.window_size].unsqueeze(0).to(self.device)  # (1, T, F)
                _, latent = self.model(window)
                embeddings.append(latent.cpu())

        embeddings = torch.cat(embeddings, dim=0)  # (num_windows, latent_dim)
        return embeddings
