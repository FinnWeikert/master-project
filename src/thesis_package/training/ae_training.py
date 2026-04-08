import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

class MotionAETrainer:
    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=1e-4):
        """
        Trainer for the BottleneckMotionAE.
        Uses AdamW to help regularize the small dataset and prevent overfitting.
        """
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.HuberLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {'train_loss': [], 'test_loss': []}

    def evaluate(self, dataloader):
        """ Computes reconstruction loss on a validation/test set. """
        self.model.eval()
        total_loss = 0.0
        batches = 0
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                x_recon, _ = self.model(x)
                loss = self.criterion(x_recon, x)
                total_loss += loss.item()
                batches += 1
        
        return total_loss / batches if batches > 0 else 0

    def train(self, dataloader, test_dataloader=None, epochs=50, verbose=True):
        """ Executes the training loop with optional validation. """

        pbar_epoch = tqdm(range(epochs), desc="Training MotionAE", disable=verbose)

        for epoch in pbar_epoch:
            # --- Training Phase ---
            self.model.train()
            epoch_train_loss = 0.0
            train_batches = 0
            
            # Progress bar for the training set
            
            for x, _ in dataloader:
                x = x.to(self.device)
                
                self.optimizer.zero_grad()
                x_recon, _ = self.model(x)
                loss = self.criterion(x_recon, x)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = epoch_train_loss / train_batches
            self.history['train_loss'].append(avg_train_loss)

            # --- Validation Phase (Optional) ---
            log_str = f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}"
            
            if test_dataloader is not None:
                avg_test_loss = self.evaluate(test_dataloader)
                self.history['test_loss'].append(avg_test_loss)
                log_str += f" | Test Loss: {avg_test_loss:.4f}"
            
            if verbose:
                print(log_str)
                
        return self.history

    def extract_latents(self, dataloader):
            self.model.eval()
            all_latents = []
            all_vids = []
            all_surgeons = []
            
            with torch.no_grad():
                for x, vids in tqdm(dataloader, desc="Extracting Latents"):
                    # x: [Batch, 45, 5]
                    # vids[0]: List of strings (video_ids)
                    # vids[1]: Tensor of ints (surgeon_ids)
                    
                    x = x.to(self.device)
                    _, latent = self.model(x)
                    
                    all_latents.append(latent.cpu().numpy())
                    all_vids.extend(vids[0])
                    
                    # FIX: Convert the tensor of surgeon_ids to a numpy array or list
                    # .cpu() moves it from GPU, .numpy() converts to array
                    surgeon_ids = vids[1].cpu().numpy()
                    all_surgeons.extend(surgeon_ids)
                    
            latents_array = np.concatenate(all_latents, axis=0)
            
            # Create the DataFrame
            latent_cols = [f"latent_{i}" for i in range(latents_array.shape[1])]
            df_latents = pd.DataFrame(latents_array, columns=latent_cols)
            
            df_latents['video_id'] = all_vids
            df_latents['surgeon_id'] = all_surgeons
            
            return df_latents