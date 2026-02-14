import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class HybridAttentionMIL(nn.Module):
    def __init__(self, local_dim=1, global_dim=2, attention_hidden_dim=4, 
                 mlp_hidden_dim=4, dropout=0.25, use_feature_extractor=False,
                 temperature=0.1): # <--- Added temperature
        super().__init__()
        self.temperature = temperature
        
        if use_feature_extractor:
            self.feature_extractor = nn.Sequential(
                nn.Linear(local_dim, local_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_extractor = nn.Identity()

        self.attention_V = nn.Sequential(
            nn.Linear(local_dim, attention_hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(local_dim, attention_hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_hidden_dim, 1)

        fusion_dim = local_dim + global_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bag, global_feat, ablation=None):
        # A. Process local features
        bag_processed = self.feature_extractor(bag) 

        # B. Calculate Attention
        V = self.attention_V(bag_processed) 
        U = self.attention_U(bag_processed) 
        G = V * U               
        
        A_raw = self.attention_weights(G) 
        
        # --- TEMPERATURE SCALING ---
        # Dividing by T < 1 makes the softmax "sharper" (more peaky)
        # Dividing by T > 1 makes it "flatter" (more uniform)
        alpha = F.softmax(A_raw / self.temperature, dim=0) 
        
        # C. Aggregate Bag (Weighted Sum)
        bag_embedding = torch.matmul(alpha.T, bag_processed) 

        if ablation == 'global_only':
            bag_embedding = torch.zeros_like(bag_embedding)
        elif ablation == 'mil_only':
            global_feat = torch.zeros_like(global_feat)
        
        # D. Fusion & Prediction
        # Ensure correct shapes for concatenation (1, feat_dim)
        fused = torch.cat([bag_embedding, global_feat.view(1, -1)], dim=1)
        score = self.regressor(fused)
        
        return score, alpha
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np

class HybridAttentionMIL_new(nn.Module):
    def __init__(self, local_dim, global_dim, attention_hidden_dim=8, 
                 mlp_hidden_dim=8, dropout=0.25, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
        # 1. Normalization layers to stabilize attention
        self.local_norm = nn.LayerNorm(local_dim)
        self.global_norm = nn.LayerNorm(global_dim)

        # 2. Gated Attention Mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(local_dim, attention_hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(local_dim, attention_hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_hidden_dim, 1)

        # 3. Final Regressor (Fusion)
        fusion_dim = local_dim + global_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bag, global_feat, ablation=None):
        # Normalize inputs to stabilize gradients
        bag = self.local_norm(bag)
        global_feat = self.global_norm(global_feat)

        # Calculate Attention
        V = self.attention_V(bag) 
        U = self.attention_U(bag) 
        G = V * U               
        
        A_raw = self.attention_weights(G) 
        
        # Softmax with temperature and a small epsilon for numerical stability
        alpha = F.softmax(A_raw / self.temperature, dim=0) + 1e-10
        alpha = alpha / alpha.sum() 
        
        # Aggregate Bag (Weighted Sum)
        bag_embedding = torch.matmul(alpha.T, bag) # (1, local_dim)

        if ablation == 'global_only':
            bag_embedding = torch.zeros_like(bag_embedding)
        elif ablation == 'mil_only':
            global_feat = torch.zeros_like(global_feat)
        
        # Fusion & Prediction
        fused = torch.cat([bag_embedding, global_feat.view(1, -1)], dim=1)
        score = self.regressor(fused)
        
        return score, alpha


########### Training Functions ###########
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, score_scaler=None, ablation=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    alpha_ratios = []
    max_alpha_feat_windows = []
    
    for bag_feats, global_feats, labels, vid_id in loader:
        optimizer.zero_grad()
        
        # These lists will store scores for the entire batch
        batch_scores = []
        
        # Move batch of labels to device
        labels = labels.to(device).view(-1, 1) # Shape: (batch_size, 1)

        for i in range(len(bag_feats)):
            bag = bag_feats[i].to(device)
            glob = global_feats[i].unsqueeze(0).to(device) 
            
            score, alpha = model(bag, glob, ablation=ablation)
            batch_scores.append(score) # Keep the tensor for backprop

            # Metrics for logging
            all_preds.append(score.item())
            all_labels.append(labels[i].item())
            alpha_ratios.append((alpha.max() / alpha.min()).item())
            max_alpha_feat_windows.append(bag[alpha.argmax()].cpu().numpy().tolist())

        # Concatenate all scores into a single tensor [batch_size, 1]
        batch_scores_tensor = torch.cat(batch_scores, dim=0)
        
        # Calculate loss on the WHOLE batch at once
        loss = criterion(batch_scores_tensor, labels)
        
        loss.backward() # Clean backprop on batch loss
        optimizer.step()
    
    # Process metrics
    preds_arr = np.array(all_preds).reshape(-1, 1)
    labels_arr = np.array(all_labels).reshape(-1, 1)

    # Inverse scale if scaler is provided
    if score_scaler is not None:
        preds_arr = score_scaler.inverse_transform(preds_arr)
        labels_arr = score_scaler.inverse_transform(labels_arr)
    
    # Calculate real-world metrics
    errors = np.abs(preds_arr - labels_arr)
    mae = np.mean(errors)

    max_alpha_feature_dist = pd.DataFrame(max_alpha_feat_windows, columns=[f"feat_{i}" for i in range(bag.shape[1])]).describe()
    alpha_ratios_dist = pd.Series(alpha_ratios).describe()
        
    return loss, mae, max_alpha_feature_dist, alpha_ratios_dist

def validate_one_epoch(model, loader, criterion, device, label_scaler=None, ablation=None):
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
                
                score, alpha = model(bag, glob, ablation=ablation)
                loss = criterion(score, label.view(1, 1))
                
                running_loss += loss.item()
                
                all_preds.append(score.item())
                all_labels.append(label.item())

    # Average loss over total SAMPLES (standard for validation)
    avg_loss = running_loss / len(loader.dataset) 
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if label_scaler is not None:
        # 2. INVERSE TRANSFORM: Move from ~0-1 back to ~25-70
        all_preds_real = label_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_labels_real = label_scaler.inverse_transform(all_labels.reshape(-1, 1)).flatten()
        
        # 3. Calculate REAL WORLD metrics
        mae = np.mean(np.abs(all_preds_real - all_labels_real))
        std = np.std(np.abs(all_preds_real - all_labels_real))
    else:
        mae = np.mean(np.abs(all_preds - all_labels))
        std = np.std(np.abs(all_preds - all_labels))
    
    metrics = {
        "val_loss": avg_loss,
        "val_mae": mae,
        'val_std': std,
    }
    
    return metrics



import copy
from collections import deque

def run_training_unbiased(model, train_loader, test_loader, epochs=600, lr=1e-3, 
                          device="cpu", score_scaler=None, patience=50, 
                          verbose=True, ablation=None, use_weight_averaging=True, 
                          avg_window=20, train_mae_threshold=4.4):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40
    )

    history = []
    best_train_mae = float('inf')
    bm_test_mae = float('inf')
    epochs_no_improve = 0
    
    # Buffer to store the last N model states for averaging
    state_buffer = deque(maxlen=avg_window)

    print(f"Starting training (Fixed/Train-based Stopping)...")
    if verbose:
        print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Train MAE':<10} | {'Test Loss':<10} | {'Test MAE (Ref)':<15} | {'LR':<8}")
        print("-" * 85)

    for epoch in range(epochs):
        # 1. Train
        if epoch % 100 == 0:
            d=1
        train_loss, train_mae, max_alpha_feature_dist, alpha_ratios_dist = train_one_epoch(
            model, train_loader, optimizer, criterion, device, score_scaler=score_scaler, ablation=ablation
        )
        
        # 2. Validate (Purely for logging, NOT for decisions)
        metrics = validate_one_epoch(model, test_loader, criterion, device, label_scaler=score_scaler, ablation=ablation)
        test_mae = metrics['val_mae']
        test_loss = metrics['val_loss']
        
        # 3. Update Scheduler based on TRAIN loss
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 4. Collect state for averaging
        if use_weight_averaging:
            # We store a CPU copy to save GPU memory
            state_dict_copy = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            state_buffer.append(state_dict_copy)
        
        # 5. Logging
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
             print(f"{epoch+1:<5} | {train_loss:.3f}     | {train_mae:.3f}     | {test_loss:.3f}     | {test_mae:.4f}          | {current_lr:.1e}")
        
        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_mae': train_mae,
            'test_mae': test_mae, 'lr': current_lr
        })

        # 6. Convergence Check
        if epoch >= 100:
            if train_mae < best_train_mae - 0.001:
                best_train_mae = train_mae
                bm_test_mae = test_mae
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if best_train_mae < train_mae_threshold:
                break
            
        if epochs_no_improve >= patience:
            print(f"\n[Terminated] Training converged at epoch {epoch+1}.")
            break
    
    # DEBUG TO Remove later
    print("max alpha feature windows\n", max_alpha_feature_dist)
    print("alpha ratios distribution\n", alpha_ratios_dist)
        

    # 7. Apply Weight Averaging
    if use_weight_averaging and len(state_buffer) > 1:
        print(f"Applying weight averaging over the last {len(state_buffer)} epochs...")
        avg_state = {}
        for key in state_buffer[0].keys():
            # Stack and mean across the stored snapshots
            avg_state[key] = torch.stack([sd[key] for sd in state_buffer]).mean(dim=0)
        
        model.load_state_dict(avg_state)
        
        # Final test pass with the averaged model
        _, train_mae, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, score_scaler=score_scaler, ablation=ablation)
        final_metrics = validate_one_epoch(model, test_loader, criterion, device, label_scaler=score_scaler, ablation=ablation)
        final_test_mae = final_metrics['val_mae']
        final_train_mae = train_mae
        print(f"Averaged Model Test MAE: {final_test_mae:.4f} (compared to final epoch {test_mae:.4f})")
    else:
        final_test_mae = test_mae
        final_train_mae = train_mae

    print(f"Training Complete. Final Test MAE: {final_test_mae:.4f} at Epoch {epoch+1}, Train MAE: {final_train_mae:.4f}")
    return model, history



import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def run_nested_training(
    model, 
    full_train_dataset,  # The N-1 surgeons
    test_loader,         # The 1 held-out surgeon
    val_size=0.2,        # Adjustable: % of training surgeons used for early stopping
    batch_size=8,
    epochs=1000,
    lr=1e-3,
    patience=50,
    device="cpu",
    score_scaler=None,
):
    """
    Performs 'Honest' training by splitting the training pool into 
    Internal Train and Internal Val.
    """
    
    # 1. Internal Split (Avoiding Data Leakage)
    # We split indices so we can use Subset
    indices = np.arange(len(full_train_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, shuffle=True)
    
    inner_train_loader = DataLoader(Subset(full_train_dataset, train_idx), batch_size=batch_size, shuffle=True)
    inner_val_loader = DataLoader(Subset(full_train_dataset, val_idx), batch_size=batch_size, shuffle=False)
    
    # 2. Setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Scheduler monitors the INTERNAL validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25
    )

    best_inner_val_mae = float('inf')
    epochs_no_improve = 0
    history = []

    print(f"Inner Train size: {len(train_idx)} | Inner Val size: {len(val_idx)}")
    
    for epoch in range(epochs):
        # --- TRAIN on Inner Train ---
        train_loss, train_mae, _ = train_one_epoch(
            model, inner_train_loader, optimizer, criterion, device, score_scaler=score_scaler
        )
        
        # --- VALIDATE on Inner Val (Decides Early Stopping) ---
        val_metrics = validate_one_epoch(
            model, inner_val_loader, criterion, device, label_scaler=score_scaler
        )
        inner_val_mae = val_metrics['val_mae']
        
        # --- TEST on Held-out Surgeon (Purely for tracking, NOT for decisions) ---
        test_metrics = validate_one_epoch(
            model, test_loader, criterion, device, label_scaler=score_scaler
        )
        held_out_mae = test_metrics['val_mae']

        scheduler.step(inner_val_mae)
        
        # --- Early Stopping Logic (Based ONLY on Inner Val) ---
        if inner_val_mae < best_inner_val_mae:
            best_inner_val_mae = inner_val_mae
            # Save weights that worked best for the INTERNAL validation
            torch.save(model.state_dict(), "inner_best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 20 == 0:
            print(f"Ep {epoch+1:03d} | InnerVal MAE: {inner_val_mae:.3f} | HeldOut MAE: {held_out_mae:.3f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered by Internal Validation at epoch {epoch+1}")
            break

    # 3. Final Evaluation
    # Load the model that performed best on the Internal Val
    model.load_state_dict(torch.load("inner_best_model.pth"))
    
    final_test_metrics = validate_one_epoch(
        model, test_loader, criterion, device, label_scaler=score_scaler
    )
    
    print(f"\n>> Final Performance on Held-Out Surgeon: {final_test_metrics['val_mae']:.4f}")
    return model, final_test_metrics



from sklearn.model_selection import KFold

def run_inner_cv_ensemble(model_class, full_train_dataset, test_loader, n_inner_folds=5, **kwargs):
    """
    Trains 5 models on different internal splits. 
    Tests the ENSEMBLE of these 5 models on the held-out surgeon.
    """
    indices = np.arange(len(full_train_dataset))
    kf = KFold(n_splits=n_inner_folds, shuffle=True)
    
    ensemble_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Starting Inner Fold {fold+1}/{n_inner_folds} ---")
        
        # Create a fresh instance of the model for this inner fold
        model = model_class() 
        
        # Train this specific model using the internal val for early stopping
        # (Using a version of run_training that saves to 'best_model_fold_X.pth')
        trained_model, _ = run_nested_training(
            model, 
            Subset(full_train_dataset, train_idx), 
            Subset(full_train_dataset, val_idx),
            # ... other params ...
        )
        ensemble_models.append(trained_model)

    # --- FINAL EVALUATION ON HELD-OUT SURGEON ---
    all_ensemble_preds = []
    
    for test_bag, test_glob, test_label, _ in test_loader:
        # Get predictions from ALL 5 models
        batch_preds = []
        for m in ensemble_models:
            m.eval()
            with torch.no_grad():
                pred, _ = m(test_bag, test_glob)
                batch_preds.append(pred)
        
        # Average the predictions (Ensemble)
        avg_pred = torch.stack(batch_preds).mean(dim=0)
        all_ensemble_preds.append(avg_pred)
    
    # Calculate final MAE based on the averaged predictions
    # ... return final metrics ...










# old 

def run_training(model, train_loader, val_loader, epochs=1000, lr=1e-3, 
                 device="cpu", score_scaler=None, patience=50, min_delta=0.01, 
                 required_train_mae=5.5, min_epochs_before_scheduling=100, verbose=True, ablation=None):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    # Scheduler: Reduces LR by half (factor=0.5) if val_loss doesn't improve for 20 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    best_val_mae = float('inf')
    bm_train_mae = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    print(f"Starting training on {device} (Max Epochs: {epochs}, Patience: {patience})")

    if verbose:
        print(f"{'Epoch':<5} | {'Train MAE':<10} | {'Val Loss':<10} | {'Val MAE':<10} | {'LR':<8}")
        print("-" * 75)

    for epoch in range(epochs):
        # 1. Train
        train_loss, train_mae, train_std = train_one_epoch(
            model, train_loader, optimizer, criterion, device, score_scaler=score_scaler, ablation=ablation
        )
        
        # 2. Validate
        if epoch % 100 == 0:
            d=1
        metrics = validate_one_epoch(model, val_loader, criterion, device, label_scaler=score_scaler, ablation=ablation)
        val_loss = metrics['val_loss']
        val_mae = metrics['val_mae']
        
        # 3. Update Scheduler (Uses val_loss to decide when to drop LR)
        if epoch >= min_epochs_before_scheduling:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 4. Logging
        if verbose:
            if (epoch + 1) % 5 == 0 or epoch == 0: # Print every 5 epochs to keep it clean
                print(f"{epoch+1:<5} | {train_mae:.3f}     | {val_loss:.4f}     | {val_mae:.4f}     | {current_lr:.1e}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'lr': current_lr,
            **metrics
        })

        # 5. Early Stopping Logic
        if train_mae < required_train_mae and epoch >= min_epochs_before_scheduling:
            if val_mae < (best_val_mae - min_delta):
                best_val_mae = val_mae
                bm_train_mae = train_mae
                best_epoch = epoch
                torch.save(model.state_dict(), "best_mil_model.pth")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            epochs_no_improve = 0  # Reset if train MAE not yet good enough
                
            
        if epochs_no_improve >= patience:
            print(f"\n[Terminated] Early stopping at epoch {epoch+1}.Model saved at Epoch {best_epoch+1}. Best Val MAE: {best_val_mae:.4f} with train MAE: {bm_train_mae:.4f}")
            break
    
    if epoch == epochs - 1:
        print(f"\n[Completed] Reached max epochs ({epochs}). Best Val MAE: {best_val_mae:.4f} with train MAE: {bm_train_mae:.4f} at Epoch {best_epoch+1}")

    # Load best weights
    model.load_state_dict(torch.load("best_mil_model.pth", weights_only=True))
    return model, history
