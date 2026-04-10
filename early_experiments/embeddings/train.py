# training loop for autoencoder
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from src.embeddings.models import MotionAutoencoder, MotionAutoencoderTCN
from tqdm import tqdm

import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm


def train_autoencoder(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=1e-3,
    device="cpu",
    patience=10,
    delta=1e-5,
    lambda_aug=0.01,
    lambda_cons=0.01,   # <-- NEW for consistency loss (overlapping windows should have a similar latent representation)
):
    """
    Generic training loop for motion autoencoders.

    Assumes:
        - input shape is (B, T, D)
        - last feature is a VALID MASK ∈ {0,1}
        - model(x) returns (recon, latent)
        - recon has shape (B, T, D-1)
        - batch is temporally ordered (for consistency loss)
    """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        recon_loss = 0.0
        aug_loss = 0.0
        cons_loss = 0.0

        if epoch == 2:
            d = 1

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}"):
            x = x.to(device)
            motion = x[..., :-1]
            valid = x[..., -1:]

            optimizer.zero_grad()

            # ===== PASS 1 — clean input =====
            out, latent_clean = model(x)
            diff = (out - motion) * valid
            mse_recon = (diff ** 2).sum() / (valid.sum() + 1e-6)

            # ===== PASS 2 — augmented (optional) =====
            if lambda_aug > 0:
                x_aug = x.clone()
                B, T, _ = x.shape
                drop = (torch.rand(B, T, 1, device=device) < 0.1)
                x_aug[..., :-1] *= (~drop)

                _, latent_aug = model(x_aug)
                mse_aug = torch.mean((latent_clean - latent_aug) ** 2)
            else:
                mse_aug = 0.0

            # ===== PASS 3 — latent consistency (NEW) =====
            if lambda_cons > 0 and latent_clean.shape[0] > 1:
                # Normalize to prevent trivial collapse
                z = torch.nn.functional.normalize(latent_clean, dim=-1)
                consistency_loss = torch.mean((z[1:] - z[:-1]) ** 2)
            else:
                consistency_loss = 0.0

            # ===== TOTAL LOSS =====
            loss = (
                mse_recon
                + lambda_aug * mse_aug
                + lambda_cons * consistency_loss
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            recon_loss += mse_recon.item() * x.size(0)
            aug_loss += mse_aug.item() * x.size(0)
            cons_loss += consistency_loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        recon_loss /= len(train_loader.dataset)
        aug_loss /= len(train_loader.dataset)
        cons_loss /= len(train_loader.dataset)

        # -----------------------------
        # Validation
        # -----------------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    motion = x[..., :-1]
                    valid = x[..., -1:]

                    out, _ = model(x)
                    diff = (out - motion) * valid
                    val_loss += (diff ** 2).sum().item() / (valid.sum() + 1e-6)

            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1:03d} — train: {train_loss:.6f} (rec:{recon_loss:.6f}, aug:{aug_loss:.6f}, cons:{cons_loss:.6f}) | val: {val_loss:.6f}")
            current_loss = val_loss
        else:
            print(f"Epoch {epoch+1:03d} — train loss: {train_loss:.6f}")
            current_loss = train_loss

        # -----------------------------
        # Early stopping
        # -----------------------------
        if current_loss < best_loss - delta:
            best_loss = current_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                print("\nEarly stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss




def train_tcn_ae(
    dataset,
    feature_dim=5,
    latent_dim=16,
    batch_size=16,
    epochs=50,
    lr=1e-3,
    device="cpu",
    patience=10,
    delta=1e-5,
    lambda_aug=0.1,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ae = MotionAutoencoderTCN(
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

        # tdqm for progress bar
        for _, (x, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1:03d}")):
            x = x.to(device)
            motion = x[..., :-1]
            valid = x[..., -1:]

            optimizer.zero_grad()

            # ===== normal pass =====
            out, latent_clean = ae(x)

            diff = (out - motion) * valid
            num_valid = valid.sum() * motion.shape[-1]
            mse_recon = (diff**2).sum() / (num_valid + 1e-6)

            # ===== smoothness penalty =====
            vel_out = out[:,1:] - out[:,:-1]
            vel_gt  = motion[:,1:] - motion[:,:-1]
            smooth = ((vel_out - vel_gt)**2 * valid[:,1:]).sum()
            smooth /= (valid[:,1:].sum() * motion.shape[-1] + 1e-6)

            # ===== augmentation =====
            if lambda_aug > 0:
                x_aug = x.clone()
                B, T, D = x_aug.shape
                drop_mask = (torch.rand(B, T, 1, device=device) < 0.05)

                x_aug[..., :-1] *= (~drop_mask)
                x_aug[..., -1:] *= (~drop_mask)

                _, latent_aug = ae(x_aug)
                mse_aug = torch.mean((latent_clean - latent_aug)**2)
            else:
                mse_aug = 0.0

            total_loss = mse_recon + 0.1 * smooth + lambda_aug * mse_aug
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            optimizer.step()

            epoch_loss += total_loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1:03d} — loss: {epoch_loss:.6f}")

        # ===== early stop =====
        if epoch_loss < best_loss - delta:
            best_loss = epoch_loss
            best_state = deepcopy(ae.state_dict())
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        ae.load_state_dict(best_state)

    return ae, best_loss
