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
    base_features=None,  # NEW: Features included in the baseline alongside PC1
    target='QRS_Overal',
    surgeon_col='Participant Number',
    test_other_pcs=False,
    top_n=3
):
    """
    Sets a baseline using PC1 + any provided 'base_features'.
    Then evaluates remaining PCs and candidate_features against those residuals.
    """
    if base_features is None:
        base_features = []
        
    unique_surgeons = df[surgeon_col].unique()
    fold_results = []

    for surgeon_out in tqdm(unique_surgeons, desc="Analyzing Folds"):
        # --- A. Strict LOSO Splitting ---
        required_cols = [target] + pca_features + candidate_features + base_features
        df_train = df[df[surgeon_col] != surgeon_out].dropna(subset=required_cols).copy()
        y_train = df_train[target].values

        # --- B. Compute PCA (Fit on Train ONLY) ---
        scaler = StandardScaler()
        X_train_pca_raw = scaler.fit_transform(df_train[pca_features])
        
        n_components = len(pca_features)
        pca = PCA(n_components=n_components)
        train_pcs = pca.fit_transform(X_train_pca_raw)
        
        pc_col_names = [f'PCA_Comp_{i+1}' for i in range(n_components)]
        df_train[pc_col_names] = train_pcs

        # --- C. Define Baseline & Candidates ---
        # Baseline is PC1 + any fixed additional features (e.g., Bimanual)
        baseline_cols = ['PCA_Comp_1'] + base_features
        
        # Remaining PCs
        remaining_pc_cols = pc_col_names[1:] 
        
        if test_other_pcs:
            current_fold_candidates = candidate_features + remaining_pc_cols
        else:
            current_fold_candidates = candidate_features

        # --- D. Fit Baseline Model ---
        baseline_model = Ridge(alpha=0.5)
        # We must scale base_features inside the fold to prevent leakage
        # (PC1 is already scaled by nature of PCA on scaled data)
        baseline_model.fit(df_train[baseline_cols], y_train)
        
        baseline_preds = baseline_model.predict(df_train[baseline_cols])
        train_residuals = y_train - baseline_preds
        baseline_resid_var = np.var(train_residuals)

        # --- E. Evaluate Candidates against Residuals ---
        for feat in current_fold_candidates:
            # 1. Partial R2 calculation
            # y ~ (PC1 + Base_Features) + Candidate_Feature
            X_combined = df_train[baseline_cols + [feat]]
            scaler_full = StandardScaler()
            X_combined = scaler_full.fit_transform(X_combined)

            combined_model = Ridge(alpha=0.5)
            combined_model.fit(X_combined, y_train)
            
            combined_preds = combined_model.predict(X_combined)
            combined_resid_var = np.var(y_train - combined_preds)
            
            # Improvement over the stronger baseline
            if baseline_resid_var > 1e-9: # Avoid division by zero
                partial_r2 = 1 - (combined_resid_var / baseline_resid_var)
            else:
                partial_r2 = 0.0
            
            # Correlation with residual
            r, _ = pearsonr(df_train[feat], train_residuals)
            
            fold_results.append({
                'Fold_Surgeon_Out': surgeon_out,
                'Feature': feat,
                'Partial_R2': partial_r2,
                'Resid_Corr': r,
                'Type': 'PC' if feat in remaining_pc_cols else 'External'
            })

    # --- F. Aggregate Results ---
    results_df = pd.DataFrame(fold_results)
    
    summary = results_df.groupby(['Feature', 'Type']).agg({
        'Partial_R2': ['mean', 'std'],
        'Resid_Corr': ['mean']
    }).reset_index()
    summary.columns = ['Feature', 'Type', 'Mean_Partial_R2', 'Std_Partial_R2', 'Mean_Resid_Corr']
    
    # Stability Calculation
    results_df['Rank'] = results_df.groupby('Fold_Surgeon_Out')['Partial_R2'].rank(ascending=False)
    stability_counts = results_df[results_df['Rank'] <= top_n]['Feature'].value_counts()
    stability_score = (stability_counts / len(unique_surgeons)).rename('Selection_Stability')
    
    final_summary = summary.merge(stability_score, left_on='Feature', right_index=True, how='left').fillna(0)
    
    return final_summary.sort_values('Mean_Partial_R2', ascending=False)