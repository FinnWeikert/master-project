import torch
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import random



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



import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

def leakage_free_residual_analysis(
    df,
    pca_features,
    candidate_features,
    base_features=None, 
    target='QRS_Overal',
    surgeon_col='Participant Number',
    test_other_pcs=False,
    top_n=3,
    perform_shuffle=False, 
    seed=42
):
    """
    Evaluates candidate features by checking if they can explain the residuals 
    of a baseline model (PC1 + Base Features).
    
    Significance Testing (Permutation):
    1. A single global permutation of INDICES is defined at the start.
    2. In each fold, the baseline model is trained on the training set.
    3. That model predicts the ENTIRE dataset to generate 'Current Global Residuals'.
    4. These residuals are shuffled using the fixed index permutation.
    5. The shuffled residuals corresponding to the TRAINING set are used as the null target.
    """
    if base_features is None:
        base_features = []
        
    # --- Step 1: Strict Cleanup & Global Indexing ---
    # We drop NaNs first to ensure integer indexing [0..N] is continuous and valid.
    all_required_cols = list(set([target, surgeon_col] + pca_features + candidate_features + base_features))
    df_clean = df.dropna(subset=all_required_cols).copy().reset_index(drop=True)
    
    unique_surgeons = df_clean[surgeon_col].unique()
    
    # Define the GLOBAL permutation of indices (Fixed Mapping)
    # If Row 5 maps to Row 10 here, it will ALWAYS map to Row 10 in every fold.
    rng = np.random.RandomState(seed)
    global_perm_indices = rng.permutation(len(df_clean))
    
    fold_results = []
    
    for surgeon_out in unique_surgeons:
        # --- A. Define Split ---
        # We use boolean indexing for logic, but need integer indices for the shuffle mapping
        train_mask = df_clean[surgeon_col] != surgeon_out
        train_idx = df_clean.index[train_mask].values
        
        y_all = df_clean[target].values
        y_train = y_all[train_idx]

        # --- B. Scale Features (Fit on TRAIN, Transform ALL) ---
        # We need the whole dataset scaled according to the training fold's statistics
        # so we can predict residuals for the whole dataset later.
        scaler = StandardScaler()
        cols_to_scale = pca_features + base_features + candidate_features
        cols_to_scale = [col for col in cols_to_scale if 'case' not in col.lower()]  # Exclude case identifier columns
        
        # Create a copy for this fold to avoid overwriting global data
        df_fold = df_clean.copy()
        
        # Fit on Train
        scaler.fit(df_fold.loc[train_idx, cols_to_scale])
        # Transform All
        df_fold[cols_to_scale] = scaler.transform(df_fold[cols_to_scale])

        # --- C. PCA (Fit on TRAIN, Transform ALL) ---
        X_train_pca_raw = df_fold.loc[train_idx, pca_features].values
        X_all_pca_raw = df_fold[pca_features].values
        
        n_components = len(pca_features)
        pca = PCA(n_components=n_components)
        
        # Fit on Train
        train_pcs = pca.fit_transform(X_train_pca_raw)
        
        # Transform All (Generate PCs for the whole dataset based on Train projection)
        all_pcs = pca.transform(X_all_pca_raw)
        pc_col_names = [f'PCA_Comp_{i+1}' for i in range(n_components)]

        pca_scaler = StandardScaler()
        all_pcs_scaled = pca_scaler.fit(train_pcs).transform(all_pcs)

        df_fold[pc_col_names] = all_pcs_scaled  

        # --- D. Define Baseline & Candidates ---
        baseline_cols = ['PCA_Comp_1'] + base_features
        
        remaining_pc_cols = pc_col_names[1:] 
        if test_other_pcs:
            current_fold_candidates = candidate_features + remaining_pc_cols
        else:
            current_fold_candidates = candidate_features

        # --- E. Fit Baseline Model (Train Only) ---
        baseline_model = RidgeCV(alphas=np.logspace(-2, 0.5, 20))
        baseline_model.fit(df_fold.loc[train_idx, baseline_cols], y_train)
        
        # Get Predictions for Train (for real analysis)
        train_preds = baseline_model.predict(df_fold.loc[train_idx, baseline_cols])
        train_residuals = y_train - train_preds
        baseline_resid_var = np.var(train_residuals)

        # --- F. Construct Null Target (The "Global Shuffle" Logic) ---
        y_train_shuffled = None
        baseline_resid_var_shuff = None
        
        if perform_shuffle:
            # 1. Predict on ENTIRE dataset using the fold's model
            #    (This gives us residuals that reflect the current fold's model performance)
            all_preds = baseline_model.predict(df_fold[baseline_cols])
            all_residuals = y_all - all_preds
            
            # 2. Shuffle the global residuals using the FIXED global indices
            #    (Row i gets the residual from Row global_perm_indices[i])
            all_residuals_shuffled = all_residuals[global_perm_indices]
            
            # 3. Subset back to Training set
            #    (We extract the "noise" assigned to the training rows)
            train_noise = all_residuals_shuffled[train_idx]
            
            # 4. Create Null Target
            y_train_shuffled = train_preds + train_noise
            baseline_resid_var_shuff = np.var(train_noise)

        # --- G. Evaluate Candidates ---
        for feat in current_fold_candidates:
            # Prepare Feature Data (Train Only)
            X_feat_train = df_fold.loc[train_idx, baseline_cols + [feat]].values
            
            # 1. Real Analysis
            combined_model = RidgeCV(alphas=np.logspace(-2, 0.5, 20))
            combined_model.fit(X_feat_train, y_train)
            combined_preds = combined_model.predict(X_feat_train)
            combined_resid_var = np.var(y_train - combined_preds)
            
            if baseline_resid_var > 1e-9:
                partial_r2 = 1 - (combined_resid_var / baseline_resid_var)
            else:
                partial_r2 = 0.0
                
            r_corr, _ = pearsonr(df_fold.loc[train_idx, feat], train_residuals)
            
            # 2. Shuffled Analysis (Null Hypothesis)
            partial_r2_shuffled = np.nan
            if perform_shuffle:
                combined_model_shuff = RidgeCV(alphas=np.logspace(-2, 0.5, 20))
                # Fit against the Null Target (Fixed Baseline + Scrambled Noise)
                combined_model_shuff.fit(X_feat_train, y_train_shuffled)
                
                combined_preds_shuff = combined_model_shuff.predict(X_feat_train)
                combined_resid_var_shuff = np.var(y_train_shuffled - combined_preds_shuff)
                
                if baseline_resid_var_shuff > 1e-9:
                    partial_r2_shuffled = 1 - (combined_resid_var_shuff / baseline_resid_var_shuff)
                else:
                    partial_r2_shuffled = 0.0

            fold_results.append({
                'Fold_Surgeon_Out': surgeon_out,
                'Feature': feat,
                'Partial_R2': partial_r2,
                'Shuffled_R2': partial_r2_shuffled,
                'Resid_Corr': r_corr,
            })

    # --- H. Aggregate Results ---
    results_df = pd.DataFrame(fold_results)
    
    agg_dict = {
        'Partial_R2': ['mean', 'std', 'min'],
        'Resid_Corr': ['mean']
    }
    if perform_shuffle:
        agg_dict['Shuffled_R2'] = ['mean', 'std']

    summary = results_df.groupby(['Feature']).agg(agg_dict).reset_index()
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    
    # Stability
    col = 'Shuffled_R2' if perform_shuffle else 'Partial_R2'
    results_df['Rank'] = results_df.groupby('Fold_Surgeon_Out')[col].rank(ascending=False)
    stability_counts = results_df[results_df['Rank'] <= top_n]['Feature'].value_counts()
    stability_score = (stability_counts / len(unique_surgeons)).rename('Selection_Stability')
    
    final_summary = summary.merge(stability_score, left_on='Feature', right_index=True, how='left').fillna(0)
    
    # Signal Quality Metric
    if perform_shuffle:
        # Z-Score estimate: (Mean Real - Mean Null) / Std Null
        # This tells you how many standard deviations the real performance is away from the random noise performance.
        final_summary['Z_Score_Est'] = (final_summary['Partial_R2_mean'] - final_summary['Shuffled_R2_mean']) / (final_summary['Shuffled_R2_std'] + 1e-9)

    if perform_shuffle:
        return final_summary.sort_values('Shuffled_R2_mean', ascending=False)
    else:
        return final_summary.sort_values('Partial_R2_mean', ascending=False)
    