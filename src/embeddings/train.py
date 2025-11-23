# training loop

import torch
from torch.utils.data import DataLoader
import os
if os.getcwd().endswith("embeddings"):
    os.chdir("../") # set cwd to src/
from embeddings.models import MotionAutoencoder
from embeddings.dataset import WindowDataset

# Parameters
window_size = 30
step_size = 10
batch_size = 16
epochs = 50
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset & DataLoader
dataset = WindowDataset(df_dict, hand="Right", feature_mode="pos_vel", 
                        window_size=window_size, step_size=step_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Autoencoder
feature_dim = 4
latent_dim = 32
ae = MotionAutoencoder(feature_dim, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# Training loop
ae.train()
for epoch in range(epochs):
    epoch_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        out, _ = ae(x)
        loss = criterion(out, x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataset):.6f}")
