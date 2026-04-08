# training function for adjustments here, move later
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

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

    max_alpha_feature_dist = None
    alpha_ratios_dist = None
    
    if alpha is not None:
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

def run_training_unbiased(
    model, train_loader, test_loader,
    epochs=600, lr=1e-3, device="cpu",
    score_scaler=None, patience=50,
    verbose=True, ablation=None,
    use_weight_averaging=True,
    avg_window=20,
    train_mae_threshold=4.4
):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience
    )

    best_train_mae = float('inf')
    epochs_no_improve = 0

    state_buffer = deque(maxlen=avg_window)

    if verbose:
        print(f"{'Epoch':<5} | {'Train MAE':<10} | {'Test MAE':<10} | {'LR':<8}")
        print("-" * 60)

    history = []

    for epoch in range(epochs):
        if epoch % 100 == 0:
            d=1

        # --- TRAIN ---
        train_loss, train_mae, max_alpha_feature_dist, alpha_ratios_dist  = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, score_scaler=score_scaler, ablation=ablation
        )

        # --- VALIDATE (LOGGING ONLY) ---
        metrics = validate_one_epoch(
            model, test_loader, criterion,
            device, label_scaler=score_scaler, ablation=ablation
        )

        test_mae = metrics['val_mae']
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # --- STORE WEIGHTS FOR AVERAGING ---
        if use_weight_averaging:
            state_dict_copy = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            state_buffer.append(state_dict_copy)

        # --- LOGGING ---
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"{epoch+1:<5} | {train_mae:.3f}     | {test_mae:.3f}     | {current_lr:.1e}")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_mae': train_mae,
            'test_mae': test_mae, 'lr': current_lr
        })

        # --- STOPPING LOGIC ---
        if train_mae < best_train_mae - 1e-3:
            best_train_mae = train_mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # threshold stopping (your idea, kept)
        if best_train_mae < train_mae_threshold:
            if verbose:
                print(f"\n[Stopped] Train MAE threshold reached.")
            break

        # patience stopping
        if epochs_no_improve >= patience:
            if verbose:
                print(f"\n[Stopped] No improvement for {patience} epochs.")
            break

    # --- APPLY WEIGHT AVERAGING ---
    if use_weight_averaging and len(state_buffer) > 1:
        if verbose:
            print(f"Applying weight averaging over {len(state_buffer)} epochs...")
        
        avg_state = {}
        for key in state_buffer[0].keys():
            avg_state[key] = torch.stack([sd[key] for sd in state_buffer]).mean(dim=0)

        model.load_state_dict(avg_state)

    # --- FINAL EVALUATION (NO TRAINING!) ---
    final__test_metrics = validate_one_epoch(
        model, test_loader, criterion,
        device, label_scaler=score_scaler, ablation=ablation
    )

    final_train_metrics = validate_one_epoch(
        model, train_loader, criterion,
        device, label_scaler=score_scaler, ablation=ablation
    )

    final_test_mae = final__test_metrics['val_mae']
    final_train_mae = final_train_metrics['val_mae']

    # for interpretation 
    #print("max alpha feature windows\n", max_alpha_feature_dist)
    #print("alpha ratios distribution\n", alpha_ratios_dist)

    print(f"Training stopped after {epoch+1} epochs. Final Test MAE: {final_test_mae:.4f} and Train MAE: {final_train_mae:.4f}")

    return model, history


from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_training_two_phase(
    model, train_loader, test_loader,
    epochs=600, lr=1e-3, device="cpu",
    score_scaler=None, patience=50,
    verbose=True, 
    use_weight_averaging=True,
    avg_window=20,
    mil_mae_threshold=4.8,     
    hybrid_mae_threshold=4.4,
    warmup_epochs=0,             # <--- NEW PARAMETER
    freeze_local_branch=False
):
    model.to(device)
    criterion = nn.MSELoss()
    history = []

    # Helper to freeze/unfreeze the local branch
    def set_local_requires_grad(requires_grad):
        local_components = [
            model.feature_extractor,
            model.attention_V,
            model.attention_U,
            model.attention_weights,
            model.local_head
        ]
        for component in local_components:
            for param in component.parameters():
                param.requires_grad = requires_grad

    # ==========================================
    # PHASE 1: PRETRAIN MIL ONLY
    # ==========================================
    if verbose: print("\n=== PHASE 1: Pretraining MIL Branch ===")
    
    optimizer_p1 = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode='min', factor=0.5, patience=patience)

    best_train_mae_p1 = float('inf')
    epochs_no_improve_p1 = 0
    state_buffer_p1 = deque(maxlen=avg_window)

    for epoch in range(epochs):
        train_loss, train_mae, _, _ = train_one_epoch(
            model, train_loader, optimizer_p1, criterion,
            device, score_scaler=score_scaler, ablation='mil_only'
        )

        metrics = validate_one_epoch(model, test_loader, criterion, device, label_scaler=score_scaler, ablation='mil_only')
        scheduler_p1.step(train_loss)
        
        if use_weight_averaging:
            state_buffer_p1.append({k: v.clone().cpu() for k, v in model.state_dict().items()})

        history.append({
            'phase': 1, 'epoch': epoch, 'train_loss': train_loss, 
            'train_mae': train_mae, 'test_mae': metrics['val_mae'], 'lr': optimizer_p1.param_groups[0]['lr']
        })

        if train_mae < best_train_mae_p1 - 1e-3:
            best_train_mae_p1 = train_mae
            epochs_no_improve_p1 = 0
        else:
            epochs_no_improve_p1 += 1

        if best_train_mae_p1 < mil_mae_threshold:
            if verbose: print(f"[Phase 1 Stopped] Threshold ({mil_mae_threshold}) reached at epoch {epoch+1}.")
            break
        if epochs_no_improve_p1 >= patience:
            if verbose: print(f"[Phase 1 Stopped] No improvement for {patience} epochs.")
            break

    # Apply Phase 1 weights
    if use_weight_averaging and len(state_buffer_p1) > 1:
        avg_state = {}
        for key in state_buffer_p1[0].keys():
            avg_state[key] = torch.stack([sd[key] for sd in state_buffer_p1]).mean(dim=0)
        model.load_state_dict(avg_state)

    # ==========================================
    # PHASE 2: FULL HYBRID TRAINING
    # ==========================================
    if verbose: print("\n=== PHASE 2: Full Model Training ===")
    
    # Optional Warm-up: Freeze the local branch initially
    if freeze_local_branch:
        if verbose: print("-> Freezing local branch for entire Phase 2.")
        set_local_requires_grad(False)
    if warmup_epochs > 0:
        if verbose: print(f"-> Freezing local branch for {warmup_epochs} epochs.")
        set_local_requires_grad(False)

    optimizer_p2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='min', factor=0.5, patience=patience)

    best_train_mae_p2 = float('inf')
    epochs_no_improve_p2 = 0
    state_buffer_p2 = deque(maxlen=avg_window)

    for epoch in range(epochs):
        # --- ADJUSTED UNFREEZE LOGIC ---
        # Only unfreeze if we are NOT permanently freezing, and warmup is complete
        if not freeze_local_branch and warmup_epochs > 0 and epoch == warmup_epochs:
            if verbose: print(f"-> Epoch {epoch}: Warm-up complete. Unfreezing local branch.")
            set_local_requires_grad(True)
            optimizer_p2 = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='min', factor=0.5, patience=patience)
        # -------------------------------

        train_loss, train_mae, _, _ = train_one_epoch(
            model, train_loader, optimizer_p2, criterion,
            device, score_scaler=score_scaler, ablation=None
        )

        metrics = validate_one_epoch(model, test_loader, criterion, device, label_scaler=score_scaler, ablation=None)
        scheduler_p2.step(train_loss)

        if use_weight_averaging and (warmup_epochs == 0 or epoch >= warmup_epochs):
            # Only start averaging Phase 2 weights AFTER the warmup is done
            state_buffer_p2.append({k: v.clone().cpu() for k, v in model.state_dict().items()})

        history.append({
            'phase': 2, 'epoch': epoch, 'train_loss': train_loss, 
            'train_mae': train_mae, 'test_mae': metrics['val_mae'], 'lr': optimizer_p2.param_groups[0]['lr']
        })

        if train_mae < best_train_mae_p2 - 1e-3:
            best_train_mae_p2 = train_mae
            epochs_no_improve_p2 = 0
        else:
            epochs_no_improve_p2 += 1

        if best_train_mae_p2 < hybrid_mae_threshold:
            if verbose: print(f"[Phase 2 Stopped] Threshold ({hybrid_mae_threshold}) reached at epoch {epoch+1}.")
            break
        if epochs_no_improve_p2 >= patience:
            if verbose: print(f"[Phase 2 Stopped] No improvement for {patience} epochs.")
            break

    # Apply Phase 2 final weights
    if use_weight_averaging and len(state_buffer_p2) > 1:
        avg_state = {}
        for key in state_buffer_p2[0].keys():
            avg_state[key] = torch.stack([sd[key] for sd in state_buffer_p2]).mean(dim=0)
        model.load_state_dict(avg_state)

    return model, history