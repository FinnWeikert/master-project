import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

# --- 1. MILBagDataset (UPDATED for TPL) ---

class MILBagDataset(Dataset):
    """
    Dataset optimized for Hybrid Attention-based MIL training (Stage 2).
    
    This version requires the TPL scalar to be included in the extracted_data list.
    """
    def __init__(self, extracted_data):
        """
        Args:
            extracted_data (list): A list of tuples: (bag_tensor, score_tensor, unique_key, TPL_scalar)
                                   - TPL_scalar must be a 1-D torch.tensor (e.g., [45.2])
        """
        self.data = extracted_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve the Bag, the score, the unique key, and the TPL scalar
        bag_tensor, score_tensor, TPL_scalar, unique_key = self.data[idx] 
        
        # Returns: (Bag of Embeddings, GRS Score, TPL Scalar, Unique ID)
        return bag_tensor, score_tensor, TPL_scalar, unique_key


# --- 2. HybridAttentionMIL Model (NEW) ---

class HybridAttentionMIL(nn.Module):
    """
    Hybrid Attention-based Multiple Instance Learning (MIL) model.
    Fuses the learned Bag Representation (32D) with a Handcrafted Metric (TPL, 1D).
    """
    def __init__(self, embed_dim=32, hidden_dim=16):
        super().__init__()
        
        # --- Attention Mechanism Layers (Identical to Pure MIL) ---
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), # 32 -> 16
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), # 32 -> 16
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )
        self.attention_weights = nn.Linear(hidden_dim, 1) # 16 -> 1

        # --- Hybrid Fusion and Final Regression Head ---
        # Input size is embed_dim (32) + 1 (TPL) = 33
        # We add a small hidden layer here to allow for non-linear interaction between B and TPL
        self.hybrid_regression = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim), # 33 -> 16
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)              # 16 -> 1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, H, TPL):
        """
        Input: 
          H: (N, D) Bag of instance embeddings. (e.g., [271, 32] tensor)
          TPL: (1, 1) TPL scalar for the current bag.
          
        Output: score: (1, 1) Predicted GRS score.
        """
        # 1. Attention Mechanism (same as pure MIL)
        V = self.attention_V(H) 
        U = self.attention_U(H) 
        G = V * U               
        
        A_raw = self.attention_weights(G) 
        alpha = F.softmax(A_raw.T, dim=1) 
        
        B = torch.matmul(alpha, H) # Bag Representation (1, 32)
        
        # 2. Hybrid Fusion
        # Concatenate Bag Representation (B) and TPL
        # Fusion vector shape: (1, 32 + 1) = (1, 33)
        B_fused = torch.cat([B, TPL], dim=1) 
        
        # 3. Final Regression
        score = self.hybrid_regression(B_fused)
        
        return score, alpha


# --- 3. Custom Collate Function (UPDATED for TPL) ---

def mil_collate_fn(batch):
    """
    Custom collate function for Hybrid MIL, handling TPL scalar.
    """
    windows, scores, tpls, keys = zip(*batch)
    windows = torch.stack(windows, 0)
    scores = torch.stack(scores, 0)
    tpls = torch.stack(tpls, 0)
    keys_list = list(keys)
    return windows, scores, tpls, keys_list


# --- 4. Helper Training and Validation Functions (UPDATED for TPL) ---

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for _, (bag_tensor, score_tensor, tpl_scalar, _) in enumerate(loader):
        # Data preparation
        H = bag_tensor.squeeze(0).to(device) # [N_windows, 32]
        Y_true = score_tensor.to(device)     # [1, 1]
        TPL = tpl_scalar.to(device)          # [1, 1]

        optimizer.zero_grad()
        # Pass both H and TPL to the model
        Y_pred, _ = model(H, TPL) 
        
        loss = criterion(Y_pred, Y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bag_tensor, score_tensor, tpl_scalar, _ in loader:
            H = bag_tensor.squeeze(0).to(device) 
            Y_true = score_tensor.to(device)
            TPL = tpl_scalar.to(device)
            
            # Pass both H and TPL to the model
            Y_pred, _ = model(H, TPL)

            loss = criterion(Y_pred, Y_true)
            val_loss += loss.item()

            all_preds.append(Y_pred.item())
            all_targets.append(Y_true.item())

    # Calculate Metrics
    val_mse = val_loss / len(loader)
    val_rmse = np.sqrt(val_mse)
    
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    val_mae = np.mean(np.abs(all_preds_np - all_targets_np))
    
    # Calculate Pearson Correlation
    val_corr, _ = spearmanr(all_targets, all_preds)

    return val_mse, val_rmse, val_mae, val_corr


# --- 5. Main Execution Function (UPDATED to use HybridAttentionMIL) ---

def run_mil_training(extracted_data_train, extracted_data_val, num_epochs=50, lr=1e-4):
    """
    Main function to run the Hybrid MIL training pipeline. Optimizes for RMSE.
    
    NOTE: extracted_data must now include the TPL scalar for each bag.
    """
    
    # 1. Data Setup
    mil_train_data = MILBagDataset(extracted_data_train)
    mil_val_data = MILBagDataset(extracted_data_val)
    
    train_loader = DataLoader(mil_train_data, batch_size=1, shuffle=True, collate_fn=mil_collate_fn)
    val_loader = DataLoader(mil_val_data, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    # 2. Model, Loss, and Optimizer Setup
    device = torch.device("cpu") 
    
    # *** MODEL INSTANTIATION IS NOW HybridAttentionMIL ***
    model = HybridAttentionMIL(embed_dim=32, hidden_dim=16).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) 

    best_val_rmse = float('inf') 
    
    # 3. Training Loop
    print(f"Starting HYBRID MIL Training on CPU for {num_epochs} epochs...")
    print(f"Training Bags: {len(mil_train_data)} | Validation Bags: {len(mil_val_data)}")

    i = 0
    for epoch in range(1, num_epochs + 1):
        i += 1
        if i % 20 == 0:
            d=1
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_rmse, val_mae, val_corr = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss (MSE): {val_mse:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Val Corr: {val_corr:.4f}")

        # Save the Best Model based on Validation RMSE (MINIMIZATION)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            #torch.save(model.state_dict(), f'best_hybrid_mil_rmse_{best_val_rmse:.4f}.pth')
            #print(">>> Saved new best HYBRID model checkpoint based on RMSE.")

    print(f"\nTraining Complete. Best Validation RMSE: {best_val_rmse:.4f}")
    
    return model, best_val_rmse


def prepare_hybrid_mil_bags(embeddings_list, metrics_df, metric_column_name, mean=None, std=None):
    """
    Transforms the 3-element bag list into the 4-element hybrid list
    required by the HybridAttentionMIL model.
    """
    new_embeddings = []
    
    # Iterate over the original (bag_tensor, score_tensor, unique_key) tuples
    for bag_tensor, score_tensor, unique_key in embeddings_list:
        
        # 1. Extract the bag identifier
        bag_name = unique_key[0]
        
        # 2. Look up the metric value in the DataFrame
        try:
            # We use .values[0] to reliably get the scalar metric value
            metric_value = metrics_df[metrics_df['Vid_Name'] == bag_name][metric_column_name].values[0]
        except IndexError:
            # Handle cases where the video name is missing (critical error handling)
            print(f"Error: Metric not found for video {bag_name}. Skipping.")
            continue
        
        if mean is not None and std is not None:
            # Normalize the metric value if mean and std are provided
            metric_value = (metric_value - mean) / std
        
        # 3. Create the TPL scalar tensor (must be [1, 1] for collate_fn to stack it as [B, 1, 1])
        # We ensure it's a float32 tensor
        tpl_scalar = torch.tensor([metric_value], dtype=torch.float32)
        
        # 4. Append the new 4-element tuple
        new_embeddings.append((bag_tensor, score_tensor, tpl_scalar, unique_key))
        
    return new_embeddings