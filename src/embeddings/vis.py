# imports 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ======================================================
# 1. PCA TRAJECTORY FOR A SINGLE VIDEO (N × latent_dim)
# ======================================================

def plot_latent_trajectory(
    embeddings,
    title="Latent PCA trajectory",
    save_path=None,
    show=True,
    point_size=20,
    cmap="viridis"
):
    """
    embeddings: numpy array or torch.Tensor, shape (N, latent_dim)
    
    Plots PCA projection of the latent trajectory:
        x-axis = PCA component 1
        y-axis = PCA component 2
        color = time
    """
    if "torch" in str(type(embeddings)):
        embeddings = embeddings.detach().cpu().numpy()

    assert embeddings.ndim == 2, "Expected shape (N, latent_dim)"
    N = embeddings.shape[0]

    # PCA to 2D
    pca = PCA(n_components=2)
    Z = pca.fit_transform(embeddings)

    # Color by time
    t = np.arange(N)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=t, cmap=cmap, s=point_size)
    plt.colorbar(label="time")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


# ======================================================
# 2. UMAP / T-SNE FOR MANY VIDEOS
# ======================================================

def plot_embedding_map(
    embedding_dict,
    method="umap",  # "umap" or "tsne"
    labels=None,
    title="Embedding Map",
    save_path=None,
    show=True,
    point_size=35,
):
    """
    embedding_dict: dict
        video_id → embedding (D,)

    method: "umap" or "tsne"
    labels: list or dict video_id → label, or None

    Example:
        embedding_dict = {
            "vid1": emb1,  # (D,)
            "vid2": emb2,
            ...
        }
    """
    video_ids = list(embedding_dict.keys())
    X = np.stack([embedding_dict[vid] for vid in video_ids], axis=0)

    # Determine colors
    if labels is None:
        # color by index
        y = np.arange(len(video_ids))
    elif isinstance(labels, dict):
        y = np.array([labels[vid] for vid in video_ids])
    else:
        y = np.array(labels)

    # --------------------------
    # Dimensionality reduction
    # --------------------------
    if method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not installed. Run: pip install umap-learn")
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="euclidean")
        Z = reducer.fit_transform(X)

    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=20, learning_rate="auto")
        Z = reducer.fit_transform(X)

    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    # --------------------------
    # Plot
    # --------------------------
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=point_size, cmap="Spectral")
    plt.colorbar(sc, label="Label")

    for i, vid in enumerate(video_ids):
        plt.text(Z[i, 0], Z[i, 1], str(vid), fontsize=7)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


# ======================================================
# 3. Helper for comparing PCA trajectories across videos
# ======================================================

def plot_multiple_trajectories(
    dict_of_embeddings,
    title="PCA Trajectories Across Videos",
    max_videos=10,
    save_path=None,
    show=True
):
    """
    dict_of_embeddings: video_id → (N, latent_dim)
    
    Plots PCA(2D) trajectories from multiple videos in one figure.
    Useful for seeing style differences.
    """
    plt.figure(figsize=(8, 7))

    for i, (vid, emb) in enumerate(list(dict_of_embeddings.items())[:max_videos]):
        if "torch" in str(type(emb)):
            emb = emb.cpu().numpy()

        pca = PCA(n_components=2)
        Z = pca.fit_transform(emb)
        
        plt.plot(Z[:, 0], Z[:, 1], label=str(vid), alpha=0.8)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()
