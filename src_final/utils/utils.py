import torch
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler



def leakage_free_correlation_analysis(
    df,
    candidate_features,
    target='QRS_Overal',
    surgeon_col='Participant Number',
    n = 5
):
    """
    Evaluates candidate features by identifying their contribution to residuals
    using a Leave-One-Surgeon-Out cross-validation approach.
    """
    unique_surgeons = df[surgeon_col].unique()
    fold_results = []

    for surgeon_out in tqdm(unique_surgeons, desc="Analyzing Folds"):
        # Split Data
        df_train = df[df[surgeon_col] != surgeon_out].dropna(subset=[target] + candidate_features).copy()
        
        # Correlation between candidate features and target
        for feat in candidate_features:
            r, _ = pearsonr(df_train[feat], df_train[target])
            fold_results.append({
                'Fold_Surgeon_Out': surgeon_out,
                'Feature': feat,
                'Corr_with_Target': r
            })

    # ==============================
    # 4. Aggregate Results across all folds
    # ==============================
    results_df = pd.DataFrame(fold_results)
    
    # We want to know:
    # 1. Mean Partial R2
    # 2. Selection Frequency (how often is it in the top 10?)
    summary = results_df.groupby('Feature').agg({
        'Corr_with_Target': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    summary.columns = ['Feature', 'Mean_Corr_with_Target', 'Std_Corr_with_Target', 'Min_Corr_with_Target', 'Max_Corr_with_Target']

    results_df['Corr_with_Target'] = results_df['Corr_with_Target'].abs()
    
    # Calculate "Stability": How often was this feature in the top 10 of its fold?
    results_df['Rank'] = results_df.groupby('Fold_Surgeon_Out')['Corr_with_Target'].rank(ascending=False)
    stability = results_df[results_df['Rank'] <= n]['Feature'].value_counts() / len(unique_surgeons)
    stability = stability.rename('Selection_Stability').reset_index().rename(columns={'index': 'Feature'})
    
    final_summary = summary.merge(stability, on='Feature', how='left').fillna(0)
    
    return final_summary.sort_values(by='Mean_Corr_with_Target', key=abs, ascending=False)



def leakage_free_residual_analysis(
    df,
    pca_features,
    candidate_features,
    target='QRS_Overal',
    surgeon_col='Participant Number',
):
    """
    Evaluates candidate features by identifying their contribution to residuals
    using a Leave-One-Surgeon-Out cross-validation approach.
    """
    unique_surgeons = df[surgeon_col].unique()
    fold_results = []

    for surgeon_out in tqdm(unique_surgeons, desc="Analyzing Folds"):
        # Split Data
        df_train = df[df[surgeon_col] != surgeon_out].dropna(subset=[target] + candidate_features + pca_features).copy()
        
        # 1. Baseline PCA & Model (Fit on Train ONLY)
        scaler = StandardScaler()
        X_train_pca_raw = scaler.fit_transform(df_train[pca_features])
        
        pca = PCA(n_components=len(pca_features))
        train_pcs = pca.fit_transform(X_train_pca_raw)
        
        # Define baseline columns
        pc_cols = [f'PC{i+1}' for i in range(train_pcs.shape[1])]
        df_train[pc_cols] = train_pcs
        used_pc_cols = [pc_cols[i] for i in pcs_to_use]
        
        # 2. Baseline Model (Train residuals)
        y_train = df_train[target].values
        pc_model = Ridge(alpha=0.5)
        pc_model.fit(df_train[used_pc_cols], y_train)
        
        train_residuals = y_train - pc_model.predict(df_train[used_pc_cols])
        pc_resid_var = np.var(train_residuals)

        # 3. Evaluate Candidate Features on this Fold
        for feat in candidate_features:
            # Partial R2 calculation on train set
            X_full = df_train[used_pc_cols + [feat]]
            full_model = Ridge(alpha=0.5)
            full_model.fit(X_full, y_train)
            
            full_resid_var = np.var(y_train - full_model.predict(X_full))
            partial_r2 = 1 - (full_resid_var / pc_resid_var)
            
            # Correlation with residuals
            r, _ = pearsonr(df_train[feat], train_residuals)
            
            fold_results.append({
                'Fold_Surgeon_Out': surgeon_out,
                'Feature': feat,
                'Partial_R2': partial_r2,
                'Resid_Corr': r
            })

    # ==============================
    # 4. Aggregate Results across all folds
    # ==============================
    results_df = pd.DataFrame(fold_results)
    
    # We want to know:
    # 1. Mean Partial R2
    # 2. Selection Frequency (how often is it in the top 10?)
    summary = results_df.groupby('Feature').agg({
        'Partial_R2': ['mean', 'std'],
        'Resid_Corr': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['Feature', 'Mean_Partial_R2', 'Std_Partial_R2', 'Mean_Resid_Corr', 'Std_Resid_Corr']

    results_df['Resid_Corr'] = results_df['Resid_Corr'].abs()
    
    # Calculate "Stability": How often was this feature in the top 10 of its fold?
    results_df['Rank'] = results_df.groupby('Fold_Surgeon_Out')['Resid_Corr'].rank(ascending=False)
    stability = results_df[results_df['Rank'] <= 3]['Feature'].value_counts() / len(unique_surgeons)
    stability = stability.rename('Selection_Stability').reset_index().rename(columns={'index': 'Feature'})
    
    final_summary = summary.merge(stability, on='Feature', how='left').fillna(0)
    
    return final_summary.sort_values('Mean_Partial_R2', ascending=False)






























#OLD
###########################################################################
################ helper functions for MIL dataset creation ################
###########################################################################

def mil_collate_fn(batch):
    """
    Custom collate function to handle the unique sample_key tuple.
    
    Input: batch (list of tuples: [(window_tensor, score_tensor, sample_key), ...])
    Output: (collated_windows, collated_scores, sample_keys_list)
    """
    # 1. Separate the components
    windows, scores, keys = zip(*batch)
    
    # 2. Collate the tensors (default behavior)
    windows = torch.stack(windows, 0)
    scores = torch.stack(scores, 0)
    
    # 3. CRUCIAL: Keep the keys as a simple list of tuples
    # 'keys' is already a list of tuples due to the zip(*batch) operation.
    keys_list = list(keys)
    
    return windows, scores, keys_list

def generate_mil_bags_cpu(data_loader, encoder):
    """
    Processes the window-level DataLoader to generate video-level bag embeddings (CPU-Only).
    
    Expected DataLoader return: (window, score, sample_key)
    (sample_key is the tuple: (video_id, surgeon_id))
    """
    grouped_data = {}
    encoder.to("cpu") 

    with torch.no_grad():
        # Unpack the batch to get windows, scores, AND the unique keys
        # Note the loop now unpacks 3 items: (windows, scores, batch_keys)
        for windows, scores, batch_keys in data_loader:
            
            # Ensure input tensors are on CPU
            windows = windows.to("cpu")
            
            # Generate the window embeddings (Instance Features)
            # embeddings_batch shape: (B, 32)
            embeddings_batch = encoder(windows)

            # --- Grouping Logic (Using the sample_key tuple as the dictionary key) ---
            for j in range(embeddings_batch.size(0)):
                # batch_keys[j] is the tuple (video_id, surgeon_id)
                key = batch_keys[j]
                
                score = scores[j].item()
                embedding = embeddings_batch[j].cpu().numpy()
                
                if key not in grouped_data:
                    # Initialize storage for the new video bag
                    grouped_data[key] = {
                        'embeddings': [],
                        'score': score,
                    }
                
                grouped_data[key]['embeddings'].append(embedding)

    # --- Final List Construction (The extracted_data list) ---
    extracted_data = []
    
    # Iterate over the unique video/surgeon keys
    for key, data in grouped_data.items():
        # Stack all window embeddings for a video into a single Bag tensor
        # Bag tensor shape: (N_windows, 32)
        bag_tensor = torch.tensor(np.stack(data['embeddings'], axis=0), dtype=torch.float32)
        score_tensor = torch.tensor(data['score'], dtype=torch.float32).unsqueeze(0)
        
        # The output tuple for the MILBagDataset: (Bag, Score, Unique_Key)
        extracted_data.append((bag_tensor, score_tensor, key))
        
    return extracted_data