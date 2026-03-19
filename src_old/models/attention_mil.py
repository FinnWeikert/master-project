import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """
    Pure Attention-based Multiple Instance Learning (MIL) model.
    Takes a Bag of TCN embeddings and outputs a single regression score.
    """
    def __init__(self, embed_dim=32, hidden_dim=16):
        super().__init__()
        
        # 1. Instance Feature Transformation Layers (V and U)
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), # 32 -> 16
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), # 32 -> 16
            nn.Sigmoid()
        )

        # 2. Attention Weight Layer (W_w)
        self.attention_weights = nn.Linear(hidden_dim, 1) # 16 -> 1

        # 3. Final Regression Head
        self.final_regression = nn.Linear(embed_dim, 1) # 32 -> 1

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, H):
        """
        Input: H: (N, D) Bag of instance embeddings. 
        Output: score: (1, 1) Predicted GRS score, alpha: (1, N) Attention weights
        """
        # Gating Mechanism (V * U)
        V = self.attention_V(H) 
        U = self.attention_U(H) 
        G = V * U               
        
        # Attention Score
        A_raw = self.attention_weights(G) 
        alpha = F.softmax(A_raw.T, dim=1) # Softmax across N instances
        
        # Bag Representation (Weighted Sum)
        B = torch.matmul(alpha, H)
        
        # Final Regression (with Scaling)
        
        # 1. Get the raw logit from the linear layer
        logit = self.final_regression(B)
        
        # 2. Apply Sigmoid to squash output to [0, 1]
        score_01 = torch.sigmoid(logit) 
        
        # 3. Rescale and Shift to target range [30, 65]
        MIN_SCORE = 25.0
        MAX_SCORE = 70.0
        score = (MAX_SCORE - MIN_SCORE) * score_01 + MIN_SCORE
        
        return score, alpha

    

import torch
from torch.utils.data import Dataset

class MILBagDataset(Dataset):
    """
    Dataset optimized for Attention-based MIL training (Stage 2 - Pure MIL).
    
    Loads pre-extracted, grouped feature bags for each video.
    """
    def __init__(self, extracted_data):
        """
        Args:
            extracted_data (list): A list of tuples: (bag_tensor, score_tensor, unique_key)
        """
        self.data = extracted_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve the Bag, the score, and the unique key
        bag_tensor, score_tensor, unique_key = self.data[idx] 
        
        # Returns: (Bag of Embeddings, GRS Score, Unique ID)
        return bag_tensor, score_tensor, unique_key
    


from torch.utils.data import DataLoader
from src.utils.utils import mil_collate_fn

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for i, (bag_tensor, score_tensor, _) in enumerate(loader):
        # bag_tensor shape: [1, N_windows, 32] (due to batch_size=1 and collate_fn)
        # score_tensor shape: [1, 1]
        
        # 1. Prepare Data
        # Squeeze the unnecessary batch dimension (B=1)
        H = bag_tensor.squeeze(0).to(device) # H shape: [N_windows, 32]
        Y_true = score_tensor.to(device)     # Y_true shape: [1, 1]

        # 2. Zero Gradients
        optimizer.zero_grad()

        # 3. Forward Pass
        # Pass the Bag of embeddings (H) to the model
        Y_pred, _ = model(H) 

        # 4. Calculate Loss
        loss = criterion(Y_pred, Y_true)
        
        # 5. Backpropagation and Optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(loader)


from scipy.stats import pearsonr

def validate_epoch(model, loader, criterion=nn.MSELoss(), device=torch.device("cpu")):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bag_tensor, score_tensor, _ in loader:
            # 1. Prepare Data (Same as training)
            H = bag_tensor.squeeze(0).to(device) 
            Y_true = score_tensor.to(device)
            
            # 2. Forward Pass
            Y_pred, alpha = model(H)

            # 3. Calculate Loss
            loss = criterion(Y_pred, Y_true)
            val_loss += loss.item()

            # 4. Store predictions for correlation metric
            all_preds.append(Y_pred.item())
            all_targets.append(Y_true.item())

    # 5. Calculate Pearson Correlation
    val_corr, _ = spearmanr(all_targets, all_preds)
    
    return val_loss / len(loader), val_corr


def run_mil_training(extracted_data_train, extracted_data_val, num_epochs=50, lr=1e-4):
    """
    Main function to run the MIL training pipeline.
    
    Args:
        extracted_data_train (list): Output of generate_mil_bags_cpu for training.
        extracted_data_val (list): Output of generate_mil_bags_cpu for validation.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate for the Adam optimizer.
    """
    
    # 1. Data Setup
    mil_train_data = MILBagDataset(extracted_data_train)
    mil_val_data = MILBagDataset(extracted_data_val)
    
    # CRITICAL: batch_size MUST be 1 for variable-length bags
    train_loader = DataLoader(mil_train_data, batch_size=1, shuffle=True, collate_fn=mil_collate_fn)
    val_loader = DataLoader(mil_val_data, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    # 2. Model, Loss, and Optimizer Setup
    device = torch.device("cpu") 
    model = AttentionMIL(embed_dim=32, hidden_dim=16).to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) 

    best_val_corr = -1.0
    
    # 3. Training Loop
    print(f"Starting MIL Training on CPU for {num_epochs} epochs...")
    print(f"Training Bags: {len(mil_train_data)} | Validation Bags: {len(mil_val_data)}")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_corr = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Corr: {val_corr:.4f}")

        # Save the Best Model based on Validation Correlation
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            # Ensure the directory exists if you plan to save outside current folder
            torch.save(model.state_dict(), f'best_attention_mil_corr_{best_val_corr:.4f}.pth')
            print(">>> Saved new best model checkpoint.")

    print(f"\nTraining Complete. Best Validation Correlation: {best_val_corr:.4f}")
    
    return model, best_val_corr







################################################################################################################
############################ Training compatible with crafted_window_features.ipynb ############################
################################################################################################################
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for bags, labels, _ in loader:
        optimizer.zero_grad()
        
        # Initialize loss accumulator for the batch
        total_batch_loss = 0.0 
        batch_size = len(bags)

        # --- Process each bag in the batch independently ---
        for i in range(batch_size):
            bag = bags[i].to(device)       # Shape (N, 5)
            label = labels[i].to(device)   # Shape (1,)
            
            # Forward pass
            score, alpha = model(bag)      # score: (1, 1)
            
            # Calculate full loss for this single bag
            # We reshape label to match score (1, 1)
            loss_i = criterion(score, label.view(1, 1))
            
            # Accumulate the loss for the batch
            # We keep track of the full loss here. 
            # We'll normalize by batch_size later when calling .backward()
            total_batch_loss += loss_i
            
            # NOTE: NO loss.backward() inside this loop!

        # --- Update Weights (Once per batch) ---
        
        # 1. Normalize the total batch loss (Average loss across all bags)
        avg_batch_loss = total_batch_loss / batch_size
        
        # 2. Backward pass (Call backward once on the average loss)
        avg_batch_loss.backward()
        
        # 3. Step the optimizer
        optimizer.step()
        
        # For tracking, use the item of the average loss
        running_loss += avg_batch_loss.item() 
        
    avg_epoch_loss = running_loss / len(loader)
    return avg_epoch_loss



def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for bags, labels, _ in loader:
            batch_size = len(bags)
            
            for i in range(batch_size):
                bag = bags[i].to(device)
                label = labels[i].to(device)
                
                score, alpha = model(bag)
                loss = criterion(score, label.view(1, 1))
                
                running_loss += loss.item()
                
                # Store for metrics
                all_preds.append(score.item())
                all_labels.append(label.item())

    # Calculate Metrics
    avg_loss = running_loss / len(loader.dataset) # Average over total samples
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics: MAE, Pearson, Spearman
    mae = np.mean(np.abs(all_preds - all_labels))
    pearson_corr, _ = pearsonr(all_preds, all_labels)
    spearman_corr, _ = spearmanr(all_preds, all_labels)
    
    metrics = {
        "val_loss": avg_loss,
        "mae": mae,
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }
    
    return metrics

def run_training(train_loader, val_loader, input_dim=6, epochs=50, lr=1e-3, device="cpu"):
    
    # 1. Initialize Model, Optimizer, Loss
    model = AttentionMIL(embed_dim=input_dim, hidden_dim=16).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Added weight decay for regularization
    criterion = nn.MSELoss()
    
    # Learning Rate Scheduler (Optional but recommended)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_mae = float('inf')
    history = []

    print(f"Starting training on {device} for {epochs} epochs...")
    print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val MAE':<10} | {'Spearman':<10}")
    print("-" * 65)

    i = 0
    for epoch in range(epochs):
        i += 1
        if i % 50 == 0:
            d=1
        # --- Train ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # --- Validate ---
        metrics = validate(model, val_loader, criterion, device)
        val_loss = metrics['val_loss']
        val_mae = metrics['mae']
        val_rho = metrics['spearman']
        
        # --- Update Scheduler ---
        #scheduler.step(val_loss)
        
        # --- Logging ---
        print(f"{epoch+1:<5} | {train_loss:.4f}     | {val_loss:.4f}     | {val_mae:.4f}     | {val_rho:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **metrics
        })

        # --- Save Best Model ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_mil_model.pth")
            # print("  -> Saved new best model")

    print(f"\nTraining Complete. Best Val MAE: {best_val_mae:.4f}")
    return model, history