import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttentionMIL(nn.Module):
    def __init__(self, local_dim=7, global_dim=4, hidden_dim=16, dropout=0.25):
        """
        local_dim: Dimension of window features (e.g., 7)
        global_dim: Dimension of video-level global features (e.g., 4)
        """
        super().__init__()
        
        # --- 1. Attention Mechanism (Gated) ---
        self.attention_V = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # --- 2. Regression Head (Non-Linear) ---
        # Input size is Local_Embedding (local_dim) + Global_Features (global_dim)
        fusion_dim = local_dim + global_dim
        
        # Regression MLP with 100-200 parameters
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 16), # might want to reduce it further
            nn.ReLU(),
            nn.Dropout(dropout), # Small N=80 dataset
            nn.Linear(16, 1)     # Output raw score
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bag, global_feat):
        """
        bag: (N, local_dim) - The windows
        global_feat: (1, global_dim) - The PC1, Vel_Corr, etc.
        """
        # A. Calculate Attention
        V = self.attention_V(bag) 
        U = self.attention_U(bag) 
        G = V * U               
        
        A_raw = self.attention_weights(G) 
        alpha = F.softmax(A_raw, dim=0) # Softmax over N instances (dim=0)
        
        # B. Aggregate Bag (Weighted Sum)
        # alpha is (N, 1), bag is (N, L). Result is (1, L)
        bag_embedding = torch.matmul(alpha.T, bag) 
        
        # C. Late Fusion
        # Concatenate: [MIL_Vector, Global_Vector]
        # shape: (1, local_dim + global_dim)
        fused = torch.cat([bag_embedding, global_feat], dim=1)
        
        # D. Predict
        score = self.regressor(fused)
        
        return score, alpha
    



########### Training Functions ###########
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for bag_feats, global_feats, labels, vid_id in loader:
        optimizer.zero_grad()
        total_batch_loss = 0.0 
        batch_size = len(bag_feats)

        for i in range(batch_size):
            bag = bag_feats[i].to(device)
            glob = global_feats[i].unsqueeze(0).to(device) 
            label = labels[i].to(device)
            
            score, alpha = model(bag, glob)
            
            loss_i = criterion(score, label.view(1, 1))
            total_batch_loss += loss_i

            # Detach gradients for metrics to save memory
            all_preds.append(score.item())
            all_labels.append(label.item())
            
        # Optimization step
        avg_batch_loss = total_batch_loss / batch_size
        avg_batch_loss.backward()
        optimizer.step()
        
        running_loss += avg_batch_loss.item()
    
    # FIX 1: Convert to numpy arrays before subtraction
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))
    std = np.std(np.abs(all_preds - all_labels))
        
    return running_loss / len(loader), mae, std

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for bag_feats, global_feats, labels, video_id in loader:
            batch_size = len(bag_feats)
            
            for i in range(batch_size):
                bag = bag_feats[i].to(device)
                glob = global_feats[i].unsqueeze(0).to(device)
                label = labels[i].to(device)
                
                score, alpha = model(bag, glob)
                loss = criterion(score, label.view(1, 1))
                
                running_loss += loss.item()
                
                all_preds.append(score.item())
                all_labels.append(label.item())

    # Average loss over total SAMPLES (standard for validation)
    avg_loss = running_loss / len(loader.dataset) 
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mae = np.mean(np.abs(all_preds - all_labels))
    std = np.std(np.abs(all_preds - all_labels))
    
    # FIX 2: Add correlations to the dictionary
    metrics = {
        "val_loss": avg_loss,
        "val_mae": mae,
        'val_std': std,
    }
    
    return metrics

def mil_collate_fn(batch):
    # This function is correct
    bags = [item[0] for item in batch]
    globals = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    vid_ids = [item[3] for item in batch]
    return bags, globals, labels, vid_ids

def run_training(model, train_loader, val_loader, input_dim=6, epochs=50, lr=1e-3, device="cpu"):
    
    # 1. Initialize Model, Optimizer, Loss
    model.to(device)
    
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
        train_loss, train_mae, train_std = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # --- Validate ---
        metrics = validate_one_epoch(model, val_loader, criterion, device)
        val_loss = metrics['val_loss']
        val_mae = metrics['mae']
        
        # --- Update Scheduler ---
        #scheduler.step(val_loss)
        
        # --- Logging ---
        print(f"{epoch+1:<5} | {train_loss:.4f}     | {val_loss:.4f}     | {val_mae:.4f}     | {val_rho:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'test_std': train_std
            **metrics
        })

        # --- Save Best Model ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_mil_model.pth")
            # print("  -> Saved new best model")

    print(f"\nTraining Complete. Best Val MAE: {best_val_mae:.4f}")
    return model, history




def mil_collate_fn(batch):
    """
    Custom collate function to handle variable-sized bags.
    Batch is a list of tuples: [(bag, global_feat, label, vid_id), ...]
    """
    bags = [item[0] for item in batch]        # List of Tensors (variable size)
    globals = torch.stack([item[1] for item in batch]) # Stacked Tensor (Batch, D_global)
    labels = torch.stack([item[2] for item in batch])  # Stacked Tensor (Batch, )
    vid_ids = [item[3] for item in batch]     # List of IDs
    
    return bags, globals, labels, vid_ids