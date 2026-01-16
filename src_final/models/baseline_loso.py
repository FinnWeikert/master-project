import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

def evaluate_loso_model(
    df,
    primary_features,
    target_col='QRS_Overal',
    surgeon_col='Participant Number',
    case_col='Case_Number', # NEW: Column to group by for scaling
    extra_features=None,
    model_class=Ridge,
    model_params={'alpha': 0.5},
    pca_components=None, 
    scale_by_case=False,    # NEW: Toggle for case-wise standardization
    scale_features=True,    # Global scaling (if scale_by_case is False)
    verbose=True
):
    if extra_features is None:
        extra_features = []
    
    unique_surgeons = df[surgeon_col].unique()
    all_preds, all_true, all_surgeons = [], [], []
    fold_metrics = {}
    use_pca = pca_components is not None and len(pca_components) > 0
    all_cols = primary_features + extra_features

    iterator = tqdm(unique_surgeons, desc="LOSOCV Folds") if verbose else unique_surgeons

    for surgeon_out in iterator:
        # 1. Split Data
        train_mask = df[surgeon_col] != surgeon_out
        test_mask = df[surgeon_col] == surgeon_out
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        y_train = df_train[target_col].values
        y_test = df_test[target_col].values

        # 2. Scaling Logic
        if scale_by_case and case_col in df.columns:
            # --- CASE-WISE STANDARDIZATION ---
            # Standardize training data within each case
            for case in df_train[case_col].unique():
                case_train_idx = df_train[df_train[case_col] == case].index
                case_test_idx = df_test[df_test[case_col] == case].index
                
                scaler = StandardScaler()
                # Fit on training surgeons for this specific case
                df_train.loc[case_train_idx, all_cols] = scaler.fit_transform(df_train.loc[case_train_idx, all_cols])
                
                # Apply that case-specific scaler to the test surgeon
                if not df_test.loc[case_test_idx].empty:
                    df_test.loc[case_test_idx, all_cols] = scaler.transform(df_test.loc[case_test_idx, all_cols])
        
        elif scale_features:
            # --- GLOBAL STANDARDIZATION ---
            scaler = StandardScaler()
            df_train[all_cols] = scaler.fit_transform(df_train[all_cols])
            df_test[all_cols] = scaler.transform(df_test[all_cols])

        # 3. Prepare Feature Matrices
        X_train_prim = df_train[primary_features].values
        X_test_prim = df_test[primary_features].values
        X_train_extra = df_train[extra_features].values if extra_features else np.empty((len(df_train), 0))
        X_test_extra = df_test[extra_features].values if extra_features else np.empty((len(df_test), 0))

        # 4. Optional PCA on Primary Features
        if use_pca:
            pca = PCA(n_components=max(pca_components) + 1)
            X_train_pca = pca.fit_transform(X_train_prim)
            X_test_pca = pca.transform(X_test_prim)
            X_train_final = X_train_pca[:, pca_components]
            X_test_final = X_test_pca[:, pca_components]
        else:
            X_train_final = X_train_prim
            X_test_final = X_test_prim

        # 5. Concatenate Extra Features
        if extra_features:
            X_train_final = np.hstack((X_train_final, X_train_extra))
            X_test_final = np.hstack((X_test_final, X_test_extra))

        # 6. Train & Predict
        model = model_class(**model_params)
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)

        # 7. Metrics
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_corr = pearsonr(y_test, y_pred)[0] if len(y_test) > 1 else np.nan
        fold_r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
        
        fold_metrics[surgeon_out] = {'MAE': fold_mae, 'Corr': fold_corr, 'R2': fold_r2}
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_surgeons.extend([surgeon_out] * len(y_test))

    # --- 8. Aggregate ---
    predictions_df = pd.DataFrame({'Surgeon_ID': all_surgeons, 'True_Score': all_true, 'Predicted_Score': all_preds})
    overall_mae = mean_absolute_error(all_true, all_preds)
    overall_std = np.std(np.abs(np.array(all_true) - np.array(all_preds)))
    overall_corr, _ = spearmanr(all_true, all_preds)
    overall_r2 = r2_score(all_true, all_preds)
    
    summary = {
        'Overall_MAE': overall_mae,
        'Overall_MAE_STD': overall_std,
        'Overall_Spearman_R': overall_corr,
        'Overall_R2': overall_r2
    }
    
    if verbose:
        scaling_type = f"By Case ({case_col})" if scale_by_case else "Global"
        print(f"\n=== LOSOCV Results ({scaling_type} Scaling) ===")
        print(f"R: {overall_corr:.4f} | MAE: {overall_mae:.4f} | MAE STD: {overall_std:.4f} | R2: {overall_r2:.4f}")
    
    plot_loso_results(predictions_df, title=f"LOSOCV: {model_class.__name__} | Scaling: {scaling_type}")
    
    return summary, pd.DataFrame.from_dict(fold_metrics, orient='index'), predictions_df

def plot_loso_results(pred_df, title="Model Performance"):
    """
    Standardized plotting function for the results.
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter with color coding
    unique_ids = pred_df['Surgeon_ID'].unique()
    # Map IDs to integers for coloring
    id_map = {sid: i for i, sid in enumerate(unique_ids)}
    colors = pred_df['Surgeon_ID'].map(id_map)
    
    scatter = plt.scatter(
        pred_df['True_Score'], 
        pred_df['Predicted_Score'], 
        c=colors, 
        cmap='Spectral', 
        alpha=0.8, 
        edgecolor='k',
        s=60
    )
    
    # Perfect fit line
    min_val = min(pred_df['True_Score'].min(), pred_df['Predicted_Score'].min())
    max_val = max(pred_df['True_Score'].max(), pred_df['Predicted_Score'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    # Annotations (optional, can get crowded)
    # for i, row in pred_df.iterrows():
    #     plt.text(row['True_Score'], row['Predicted_Score'], str(row['Surgeon_ID']), fontsize=8)

    plt.xlabel("True GRS Score")
    plt.ylabel("Predicted Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()