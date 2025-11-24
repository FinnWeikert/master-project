# training loop for autoencoder
import torch
from torch.utils.data import DataLoader
from src.embeddings.models import MotionAutoencoder


def train_ae(
    dataset,
    feature_dim=4,
    latent_dim=16,
    batch_size=16,
    epochs=50,
    lr=1e-3,
    device="cpu",
    patience=10,          # set None to disable early stopping
    delta=1e-5            # minimum improvement
):
    from copy import deepcopy
    import torch
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ae = MotionAutoencoder(feature_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    wait = 0    # number of epochs without improvement

    ae.train()
    for epoch in range(epochs):

        epoch_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            out, _ = ae(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.6f}")

        # ------------------------
        # Early stopping logic
        # ------------------------
        if patience is not None:
            if epoch_loss < best_loss - delta:
                best_loss = epoch_loss
                best_state = deepcopy(ae.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                    break

    # Load best weights if early stopped
    if best_state is not None:
        ae.load_state_dict(best_state)

    return ae, best_loss


