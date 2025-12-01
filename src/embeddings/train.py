# training loop for autoencoder
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from src.embeddings.models import MotionAutoencoder

def train_ae(
    dataset,
    feature_dim=5,              # now includes "valid" channel
    latent_dim=16,
    batch_size=16,
    epochs=50,
    lr=1e-3,
    device="cpu",
    patience=10,
    delta=1e-5,
    lambda_aug=0.0,             # set > 0 to enable augmentation consistency loss
):
    """
    Train MotionAutoencoder with masked loss and optional synthetic gap augmentation.
    Assumes last feature in input is 'valid' mask.
    """
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ae = MotionAutoencoder(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        dropout_prob=0.1 if lambda_aug > 0 else 0.0
    ).to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    wait = 0

    ae.train()
    for epoch in range(epochs):

        epoch_loss = 0.0

        for x, _ in dataloader:

            x = x.to(device)

            motion = x[..., :-1]          # real motion
            valid = x[..., -1:]           # (batch, T, 1)

            optimizer.zero_grad()

            # ------ PASS 1: normal input ------
            out, latent_clean = ae(x)     # recon motion only

            # masked MSE
            diff = (out - motion) * valid
            mse_recon = (diff ** 2).sum() / (valid.sum() + 1e-6)

            # ------ PASS 2: augmented input ------
            if lambda_aug > 0:
                x_aug = x.clone()
                B, T, D = x_aug.shape

                # randomly drop some valid frames (fake tracking loss)
                drop_mask = (torch.rand(B, T, 1, device=device) < 0.1)
                x_aug[..., :-1] = x_aug[..., :-1] * (~drop_mask)   # zero out motion at dropped frames
                # keep valid mask unchanged

                _, latent_aug = ae(x_aug)

                # consistency loss
                mse_aug = torch.mean((latent_clean - latent_aug)**2)
            else:
                mse_aug = 0.0

            total_loss = mse_recon + lambda_aug * mse_aug
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1:03d} — loss: {epoch_loss:.6f}")

        # ---- early stopping ----
        if epoch_loss < best_loss - delta:
            best_loss = epoch_loss
            best_state = deepcopy(ae.state_dict())
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                print("\nEarly stopping.")
                break

    if best_state is not None:
        ae.load_state_dict(best_state)

    return ae, best_loss