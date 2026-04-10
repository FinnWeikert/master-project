import numpy as np
import torch
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader, Subset
from torch import nn

def train_tcn(
    model,
    train_loader,
    val_loader,
    num_epochs=40,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    device="cpu"
):
    """
    Training loop for surgical skill assessment using Masked TCN.
    - train_loader/val_loader: returns (x_features, y_target, surgeon_id)
    """

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_corr = -1
    best_state = None

    for epoch in range(1, num_epochs + 1):

        # -----------------------
        # TRAINING
        # -----------------------
        model.train()
        train_losses = []

        # x_features contains: [kinematics | mask] (C_kinematic + 1 channels total)
        for x_features, y_target, _ in train_loader: 
            
            y = y_target.to(device).float()           # (B, 1)

            #x_features = x_features.permute(0, 2, 1).to(device)
            optimizer.zero_grad()

            pred = model(x_features)  # Pass the entire features tensor

            loss = criterion(pred.squeeze(), y.squeeze())
            loss.backward()

            # Optional safety: gradient clipping for stability
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # -----------------------
        # VALIDATION
        # -----------------------
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_features, y_target, _ in val_loader:
                
                # Separate kinematics (features) from the mask (last channel)
                y = y_target.to(device).float()
                #x_features = x_features.permute(0, 2, 1).to(device)

                pred = model(x_features)
                loss = criterion(pred.squeeze(), y.squeeze())

                val_losses.append(loss.item())
                all_preds.extend(pred.squeeze().cpu().numpy().tolist())
                all_targets.extend(y.squeeze().cpu().numpy().tolist())

        avg_val_loss = sum(val_losses) / len(val_losses)

        # Compute Pearson correlation (for your thesis)
        try:
            val_corr, _ = pearsonr(all_preds, all_targets)
        except:
            # Handle case where one array is constant (e.g., all predictions are the same)
            val_corr = 0.0 

        # Track best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = model.state_dict()

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Corr: {val_corr:.3f}")

    print("\nTraining complete.")
    print(f"Best validation correlation: {best_val_corr:.3f}")

    # Restore the best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_corr


def run_losocv(
    model_class,         # Uninstantiated TCN model class (e.g., SurgicalTCN)
    dataset,             # Full MotionFeatureDataset instance
    model_params,        # Dictionary of parameters for model_class init
    train_tcn_params,    # Dictionary of parameters for train_tcn function
    device="cpu"
):
    """
    Performs Leave-One-Surgeon-Out Cross-Validation (LOSO-CV).

    Assumes dataset.index_map has structure:
        (sample_key, start, grs_score, surgeon_id)
    """
    
    # 1. Extract Grouping IDs
    # Get the surgeon_id for every single window index in the dataset.
    # item[3] corresponds to the surgeon_id in the index_map tuple.
    groups = np.array([item[3] for item in dataset.index_map])
    
    # 2. Setup CV Splitter
    logo = LeaveOneGroupOut()
    unique_surgeons = np.unique(groups)
    
    # 3. Initialize storage
    fold_correlations = []
    
    print(f"Starting LOSO-CV with {len(unique_surgeons)} folds (Surgeons: {unique_surgeons})...")
    
    # Iterate through folds (each unique surgeon ID is the test group once)
    for fold, (train_index, val_index) in enumerate(logo.split(X=np.arange(len(dataset)), groups=groups)):

        # Skip fold if validation set is empty (shouldn't happen with correct grouping)
        if len(val_index) == 0:
            continue

        # 4. Create Subset DataLoaders for the current fold
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)
        
        # Batch sizes are typically smaller for TCNs
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # 5. Instantiate a FRESH model for each fold (Crucial for independence)
        current_model = model_class(**model_params)
        
        # 6. Get the ID of the surgeon left out for logging
        left_out_id = groups[val_index[0]] # ID of the first window's surgeon in the validation set
        
        print(f"\n--- Fold {fold + 1:02d} | Leaving out Surgeon ID: {left_out_id} ---")
        
        # 7. Train the model using the train_tcn function
        # Note: train_loader returns (x, y, surgeon_id). train_tcn must handle this.
        _, best_corr = train_tcn(
            model=current_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **train_tcn_params
        )
        
        fold_correlations.append(best_corr)

    # 8. Report Final Results
    mean_corr = np.mean(fold_correlations)
    std_corr = np.std(fold_correlations)
    
    print("\n--- Final LOSO-CV Results ---")
    print(f"Individual fold correlations: {np.round(fold_correlations, 3)}")
    print(f"Mean Correlation across {len(fold_correlations)} folds: **{mean_corr:.4f}**")
    print(f"Standard Deviation: {std_corr:.4f}")
    print("-----------------------------")
    
    return mean_corr, fold_correlations