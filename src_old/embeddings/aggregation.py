import numpy as np
from sklearn.decomposition import PCA
import torch


class EmbeddingAggregator:
    """
    Aggregate a sequence of window embeddings into a single fixed-size vector.
    
    Supported methods:
        - "mean": average embedding (baseline)
        - "std": standard deviation only
        - "mean_std": concatenation of mean and std (recommended)           TRY FIRST
        - "max": max-pooling
        - "avg_max": concatenation of mean and max
        - "pca": PCA trajectory summary (first component mean)
        - "pca_mean_var": (mean PCA coords, PCA explained variance)         TRY SECOND
        - "trajectory": flattened PCA trajectory (optional; dim-limited)
    """

    def __init__(self, method="mean_std", pca_dim=3, max_traj_len=50):
        assert method in (
            "mean", "std", "mean_std",
            "max", "avg_max",
            "pca", "pca_mean_var", "trajectory"
        ), f"Unknown aggregation method: {method}"

        self.method = method
        self.pca_dim = pca_dim
        self.max_traj_len = max_traj_len  # for 'trajectory' only

    def __call__(self, embeddings):
        """
        embeddings: numpy array of shape (N, latent_dim)
        Returns: numpy array (aggregated embedding vector)
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        assert embeddings.ndim == 2, "Expected shape (N, latent_dim)"

        if self.method == "mean":
            return self._mean(embeddings)

        if self.method == "std":
            return self._std(embeddings)

        if self.method == "mean_std":
            return self._mean_std(embeddings)

        if self.method == "max":
            return self._max(embeddings)

        if self.method == "avg_max":
            return self._avg_max(embeddings)

        if self.method == "pca":
            return self._pca_first_component_mean(embeddings)

        if self.method == "pca_mean_var":
            return self._pca_mean_and_variance(embeddings)

        if self.method == "trajectory":
            return self._pca_trajectory(embeddings)

    # --------------------------
    # BASIC METHODS
    # --------------------------
    def _mean(self, E):
        return E.mean(axis=0)

    def _std(self, E):
        return E.std(axis=0)

    def _mean_std(self, E):
        return np.concatenate([E.mean(axis=0), E.std(axis=0)], axis=0)

    def _max(self, E):
        return E.max(axis=0)

    def _avg_max(self, E):
        return np.concatenate([E.mean(axis=0), E.max(axis=0)], axis=0)

    # --------------------------
    # PCA-BASED AGGREGATION
    # --------------------------
    def _fit_pca(self, E):
        pca = PCA(n_components=min(self.pca_dim, E.shape[1], E.shape[0]))
        E_pca = pca.fit_transform(E)
        return pca, E_pca

    def _pca_first_component_mean(self, E):
        """
        Summaries the motion trajectory by taking the mean of the 1st PCA component.
        (latent_dim stays small)
        """
        pca, E_pca = self._fit_pca(E)
        comp1 = E_pca[:, 0]  # (N,)
        return np.array([comp1.mean()], dtype=np.float32)

    def _pca_mean_and_variance(self, E):
        """
        Returns: concatenate(mean PCA coords, PCA explained variance)
        Dimensionality: pca_dim * 2
        """
        pca, E_pca = self._fit_pca(E)
        mean_pca = E_pca.mean(axis=0)               # (pca_dim,)
        var_pca = pca.explained_variance_ratio_     # (pca_dim,)
        return np.concatenate([mean_pca, var_pca], axis=0)

    # --------------------------
    # PCA TRAJECTORY (OPTIONAL)
    # --------------------------
    def _pca_trajectory(self, E):
        """
        Flattened PCA trajectory up to max_traj_len timesteps.
        WARNING: long vectors (not recommended unless needed).
        """
        pca, E_pca = self._fit_pca(E)

        # truncate or pad trajectory length
        T = len(E_pca)
        d = E_pca.shape[1]
        if T > self.max_traj_len:
            E_pca = E_pca[:self.max_traj_len]
        else:
            pad = np.zeros((self.max_traj_len - T, d))
            E_pca = np.vstack([E_pca, pad])

        return E_pca.flatten()
