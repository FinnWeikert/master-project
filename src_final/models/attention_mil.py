import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class HybridAttentionMILold(nn.Module):
    def __init__(self, local_dim=1, global_dim=2, attention_hidden_dim=8, 
                 mlp_hidden_dim=8, n_hidden=1, dropout=0.25, 
                 use_feature_extractor=False, temperature=1,
                 feature_extractor_dim=8): # <--- Added temperature
        super().__init__()
        self.temperature = temperature
        
        if use_feature_extractor:
            self.feature_extractor = nn.Sequential(
                nn.Linear(local_dim, feature_extractor_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            local_dim = feature_extractor_dim  
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
        if n_hidden == 1:
            self.regressor = nn.Sequential(
                nn.Linear(fusion_dim, mlp_hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, 1)
            )
        else:
            # Keeps width consistent to preserve feature interactions
            layers = []
            curr_dim = fusion_dim
            for _ in range(n_hidden):
                layers.extend([
                    nn.Linear(curr_dim, mlp_hidden_dim),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(dropout)
                ])
                curr_dim = mlp_hidden_dim
            layers.append(nn.Linear(mlp_hidden_dim, 1))
            self.regressor = nn.Sequential(*layers)
        
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
        
        if ablation == 'no_attention' or ablation == 'no_attention_and_global':
            alpha = torch.ones_like(alpha) / alpha.size(0)  # Uniform weights

        # C. Aggregate Bag (Weighted Sum)
        bag_embedding = torch.matmul(alpha.T, bag_processed) 

        if ablation == 'global_only':
            bag_embedding = torch.zeros_like(bag_embedding)
        elif ablation == 'mil_only' or ablation == 'no_attention_and_global':
            global_feat = torch.zeros_like(global_feat)
        
        # D. Fusion & Prediction
        # Ensure correct shapes for concatenation (1, feat_dim)
        fused = torch.cat([bag_embedding, global_feat.view(1, -1)], dim=1)
        score = self.regressor(fused)
        
        return score, alpha




class HybridAttentionMIL(nn.Module):
    def __init__(self, local_dim=1, global_dim=2, attention_hidden_dim=8, 
                 mlp_hidden_dim=8, n_hidden=1, dropout=0.25, 
                 use_feature_extractor=False, temperature=1,
                 feature_extractor_dim=8):
        super().__init__()
        self.temperature = temperature
        
        # --- LOCAL BRANCH ---
        if use_feature_extractor:
            self.feature_extractor = nn.Sequential(
                nn.Linear(local_dim, feature_extractor_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            local_dim = feature_extractor_dim  
        else:
            self.feature_extractor = nn.Identity()

        # Attention Mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(local_dim, attention_hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(local_dim, attention_hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_hidden_dim, 1)

        # Separate processing for the aggregated Local Embedding
        self.local_head = nn.Sequential(
            nn.Linear(local_dim, mlp_hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout)
        )

        # --- GLOBAL BRANCH ---
        # This branch ensures the global signal is processed on its own 
        # terms before meeting the local features
        self.global_head = nn.Sequential(
            nn.Linear(global_dim, mlp_hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout)
        )

        # --- FINAL FUSION ---
        # The heads meet here at the very end
        self.final_regressor = nn.Linear(mlp_hidden_dim * 2, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bag, global_feat, ablation=None):
        # 1. LOCAL PATHWAY
        if ablation == 'global_only':
            # Skip local pathway entirely for speed
            local_out = torch.zeros((1, self.local_head[0].out_features), device=bag.device)
            alpha = None # No attention calculated
        else:
            bag_processed = self.feature_extractor(bag) 
            
            # OPTIMIZATION: Skip attention computation if not needed
            if ablation in ['no_attention', 'no_attention_and_global']:
                # Mean Pooling: Simple average across the window dimension
                bag_embedding = torch.mean(bag_processed, dim=0, keepdim=True)
                alpha = torch.ones((bag.size(0), 1), device=bag.device) / bag.size(0)
            else:
                # Full Attention Mechanism
                V = self.attention_V(bag_processed) 
                U = self.attention_U(bag_processed) 
                G = V * U               
                A_raw = self.attention_weights(G) 
                alpha = F.softmax(A_raw / self.temperature, dim=0) 
                bag_embedding = torch.matmul(alpha.T, bag_processed) # (1, local_dim)
            
            # Process local embedding through its dedicated head
            local_out = self.local_head(bag_embedding) # (1, mlp_hidden_dim)

        # 2. GLOBAL PATHWAY
        if ablation in ['mil_only', 'no_attention_and_global']:
            # Skip global processing
            global_out = torch.zeros((1, self.global_head[0].out_features), device=bag.device)
        else:
            # Process global features through its dedicated head
            global_out = self.global_head(global_feat.view(1, -1)) 

        # 3. FINAL FUSION
        fused = torch.cat([local_out, global_out], dim=1)
        score = self.final_regressor(fused)
        
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

            if alpha is not None:
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

    if ablation != 'global_only':  # Only calculate if attention was used
        max_alpha_feature_dist = pd.DataFrame(max_alpha_feat_windows, columns=[f"feat_{i}" for i in range(bag.shape[1])]).describe()
        alpha_ratios_dist = pd.Series(alpha_ratios).describe()
    else:
        max_alpha_feature_dist = None
        alpha_ratios_dist = None
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

    # alternative to try
    """
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=patience,   
        gamma=0.5      
    )"""

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
        #scheduler.step()
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



### NESTED training with inner validation set


import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import GroupKFold
from copy import deepcopy


class CrossValidatedTrainer:
    def __init__(
        self,
        model_fn,                 # lambda: HybridAttentionMIL(...)
        lr=2e-3,
        weight_decay=1e-3,
        epochs=600,
        n_splits=4,
        device="cpu",
        ablation=None,
        random_state=42,
        patience=50,
        plot_history=False
    ):
        self.model_fn = model_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.n_splits = n_splits
        self.device = device
        self.ablation = ablation
        self.random_state = random_state
        self.patience = patience
        
        self.stopping_epoch_ = None
        self.plot_history= plot_history
        self.cv_history_ = []
        self.history = []
        

    # ------------------------------------------------------------
    # Full batch training for ONE model (used internally)
    # ------------------------------------------------------------
    def _train_full_batch(self, model, loader, optimizer, criterion, validation_loader=None, test_loader=None):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        for bags, globs, labels, _ in loader:
            optimizer.zero_grad()
            
            labels = labels.to(self.device).view(-1, 1)  # Shape: [batch_size, 1]
            batch_preds = []

            for i in range(len(bags)):
                bag = [b.to(self.device) for b in bags[i]] if isinstance(bags[i], list) else bags[i].to(self.device)
                glob = globs[i].unsqueeze(0).to(self.device)
                
                pred, _ = model(bag, glob, ablation=self.ablation)
                batch_preds.append(pred)

            # Stack all bag predictions into a tensor [batch_size, 1]
            batch_preds_tensor = torch.cat(batch_preds, dim=0)
            
            # Compute loss over entire batch
            loss = criterion(batch_preds_tensor, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_mae += torch.abs(batch_preds_tensor - labels).sum().item()
            n += len(labels)

        # Optionally compute metrics on validation/test set
        if validation_loader is not None:
            val_loss, val_mae = self._evaluate(model, validation_loader, criterion)
            self.cv_history_[-1].append({
                'train_loss': total_loss / n,
                'train_mae': total_mae / n,
                'val_loss': val_loss,
                'val_mae': val_mae
            })
        elif test_loader is not None:
            test_loss, test_mae = self._evaluate(model, test_loader, criterion)
            self.history.append({
                'train_loss': total_loss / n,
                'train_mae': total_mae / n,
                'test_loss': test_loss,
                'test_mae': test_mae
            })

        return total_loss / n, total_mae / n


    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        with torch.no_grad():
            for bags, globs, labels, _ in loader:
                labels = labels.to(self.device).view(-1, 1)
                batch_preds = []

                for i in range(len(bags)):
                    bag = [b.to(self.device) for b in bags[i]] if isinstance(bags[i], list) else bags[i].to(self.device)
                    glob = globs[i].unsqueeze(0).to(self.device)
                    
                    pred, _ = model(bag, glob, ablation=self.ablation)
                    batch_preds.append(pred)

                batch_preds_tensor = torch.cat(batch_preds, dim=0)
                loss = criterion(batch_preds_tensor, labels)

                total_loss += loss.item() * len(labels)
                total_mae += torch.abs(batch_preds_tensor - labels).sum().item()
                n += len(labels)

        return total_loss / n, total_mae / n



    # ------------------------------------------------------------
    # Estimate stopping epoch via grouped CV
    # ------------------------------------------------------------
    def _estimate_stopping_epoch(self, dataset, groups):
        """
        Estimate stopping epoch via internal grouped CV, with early stopping based on validation MAE.
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        fold_best_epochs = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(dataset)), groups=groups)):

            self.cv_history_.append([])  # For plotting CV history later

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=len(train_subset), shuffle=False, collate_fn=dataset.mil_collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=len(val_subset), shuffle=False, collate_fn=dataset.mil_collate_fn
            )

            model = self.model_fn().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            criterion = nn.MSELoss()

            best_val_mae = np.inf
            best_epoch = 0
            epochs_no_improve = 0

            for epoch in range(self.epochs):
                self._train_full_batch(model, train_loader, optimizer, criterion, validation_loader=val_loader)
                _, val_mae = self._evaluate(model, val_loader, criterion)

                if epoch > self.patience:
                    if val_mae < best_val_mae - 1e-6:  # small epsilon to avoid float issues
                        best_val_mae = val_mae
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    # Early stopping if no improvement for `patience` epochs
                    if epochs_no_improve >= self.patience:
                        print(f"[Fold {fold}] Early stopping at epoch {epoch+1} (best val MAE: {best_val_mae:.4f})")
                        break
            
            if self.plot_history:
                self._plot_cv_history(fold)

            fold_best_epochs.append(best_epoch)

        # Use median for robustness
        stopping_epoch = int(np.median(fold_best_epochs))
        self.stopping_epoch_ = stopping_epoch
        
        return stopping_epoch

    # ------------------------------------------------------------
    # Public Fit
    # ------------------------------------------------------------
    def fit(self, train_dataset, groups, test_dataset=None):
        torch.manual_seed(self.random_state)

        print("Estimating stopping epoch via internal grouped CV...")
        stopping_epoch = self._estimate_stopping_epoch(train_dataset, groups)
        print(f"Selected stopping epoch (median across folds): {stopping_epoch}")

        # Retrain on FULL training set
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=train_dataset.mil_collate_fn)
        model = self.model_fn().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        if test_dataset is not None:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.mil_collate_fn)

        for epoch in range(stopping_epoch):
            self._train_full_batch(model, loader, optimizer, criterion, test_loader=test_loader)

        self.model_ = model
        return model

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    def predict(self, loader):
        self.model_.eval()
        preds = []

        with torch.no_grad():
            for bags, globs, _, _ in loader:
                batch_preds = []

                for i in range(len(bags)):
                    bag = [b.to(self.device) for b in bags[i]] if isinstance(bags[i], list) else bags[i].to(self.device)
                    glob = globs[i].unsqueeze(0).to(self.device)

                    p, _ = self.model_(bag, glob, ablation=self.ablation)
                    # Ensure p is at least 1D
                    batch_preds.append(p.view(-1))

                # Concatenate predictions safely
                batch_preds_tensor = torch.cat(batch_preds, dim=0)
                preds.extend(batch_preds_tensor.cpu().numpy())

        return np.array(preds)

    
    def _plot_cv_history(self, fold):
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.cv_history_[fold]) + 1)
        train_mae = [h['train_mae'] for h in self.cv_history_[fold]]
        test_mae = [h['val_mae'] for h in self.cv_history_[fold]]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_mae, label='Train MAE')
        plt.plot(epochs, test_mae, label='Test MAE')
        #plt.axvline(self.stopping_epoch_, color='red', linestyle='--', label='Selected Stopping Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title(f'Cross-Validation MAE Across Epochs (Fold {fold})')
        plt.legend()
        plt.grid()
        plt.show()
