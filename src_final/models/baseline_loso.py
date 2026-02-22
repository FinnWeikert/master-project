import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from src_final.models.analysis import NestedFeatureSelector
from src_final.models.mlp_regressor import PyTorchMLPEnsemble

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
    verbose=True,
    print_fold_metrics=False
):
    if extra_features is None:
        extra_features = []
    
    unique_surgeons = df[surgeon_col].unique()
    all_preds, all_true, all_surgeons = [], [], []
    fold_metrics = {}
    use_pca = pca_components is not None and len(pca_components) > 0
    all_cols = primary_features + extra_features

    all_cols_to_scale = [col for col in all_cols if 'case' not in col.lower()]  # Exclude case identifier columns

    all_fold_weights = []
    feature_names = ['bias'] + primary_features + extra_features

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
            # Cast columns to float to avoid dtype warnings during scaling
            df[all_cols_to_scale] = df[all_cols_to_scale].astype(float)
            # --- CASE-WISE STANDARDIZATION ---
            # Standardize training data within each case
            for case in df_train[case_col].unique():
                case_train_idx = df_train[df_train[case_col] == case].index
                case_test_idx = df_test[df_test[case_col] == case].index
                
                scaler = StandardScaler()
                # Fit on training surgeons for this specific case
                df_train.loc[case_train_idx, all_cols_to_scale] = scaler.fit_transform(df_train.loc[case_train_idx, all_cols_to_scale])
                
                # Apply that case-specific scaler to the test surgeon
                if not df_test.loc[case_test_idx].empty:
                    df_test.loc[case_test_idx, all_cols_to_scale] = scaler.transform(df_test.loc[case_test_idx, all_cols_to_scale])
        
        elif scale_features:
            # --- GLOBAL STANDARDIZATION ---
            scaler = StandardScaler()
            df_train[all_cols_to_scale] = scaler.fit_transform(df_train[all_cols_to_scale])
            df_test[all_cols_to_scale] = scaler.transform(df_test[all_cols_to_scale])

        # 3. Prepare Feature Matrices
        X_train_prim = df_train[primary_features].values
        X_test_prim = df_test[primary_features].values
        X_train_extra = df_train[extra_features].values if extra_features else np.empty((len(df_train), 0))
        X_test_extra = df_test[extra_features].values if extra_features else np.empty((len(df_test), 0))

        # 4. Optional PCA on Primary Features
        if use_pca:
            feature_names = ['bias'] + [f"PC{idx+1}" for idx in pca_components] + extra_features
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

        # capture weights
        if isinstance(model.intercept_, float):  # single feature case
            intercept_and_coef = np.concatenate([[model.intercept_], model.coef_])
        else:
            intercept_and_coef = np.concatenate((model.intercept_, model.coef_[0]))
        all_fold_weights.append(intercept_and_coef)
        
        y_pred = model.predict(X_test_final)

        # 7. Metrics
        fold_train_mae = mean_absolute_error(y_train, model.predict(X_train_final))
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_corr = pearsonr(y_test, y_pred)[0] if len(y_test) > 1 else np.nan
        fold_r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
        
        fold_metrics[surgeon_out] = {'Train_MAE': fold_train_mae, 'Test_MAE': fold_mae, 'Test_Corr': fold_corr}
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_surgeons.extend([surgeon_out] * len(y_test))

    # --- 8. Aggregate ---
    predictions_df = pd.DataFrame({'Surgeon_ID': all_surgeons, 'True_Score': all_true, 'Predicted_Score': all_preds})
    overall_mae = mean_absolute_error(all_true, all_preds)
    overall_std = np.std(np.abs(np.array(all_true) - np.array(all_preds)))
    overall_corr, _ = spearmanr(all_true, all_preds)
    overall_r2 = r2_score(all_true, all_preds)
    adjusted_r2 = 1 - (1 - overall_r2) * (len(all_true) - 1) / (len(all_true) - len(feature_names) - 1)

    summary = {
        'Overall_MAE': overall_mae,
        'Overall_MAE_STD': overall_std,
        'Overall_Spearman_R': overall_corr,
        'Overall_R2': overall_r2,
        'Overall_Adj_R2': adjusted_r2
    }

    avg_weights = np.mean(all_fold_weights, axis=0)
    std_weights = np.std(all_fold_weights, axis=0)

    weight_report = pd.DataFrame({
        'Feature': feature_names,
        'Average_Weight': avg_weights,
        'Std_Weight': std_weights
    })
    fold_results_df = pd.DataFrame.from_dict(fold_metrics, orient='index')

    if verbose:
        scaling_type = f"By Case ({case_col})" if scale_by_case else "Global"
        print(f"\n=== LOSOCV Results ({scaling_type} Scaling) ===")
        print(f"R: {overall_corr:.4f} | MAE: {overall_mae:.4f} | MAE STD: {overall_std:.4f} | R2: {overall_r2:.4f} | Adj R2: {adjusted_r2:.4f}")
        print("\nFeature Weights:")
        print(weight_report)
        print(f"\n--- Per-Fold Performance Summary ---")
        if print_fold_metrics:
            print(fold_results_df)
    
    plot_loso_results(predictions_df, title=f"LOSOCV: {model_class.__name__} | Scaling: {scaling_type}")
    
    return summary, fold_results_df, predictions_df

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




def evaluate_loso_mlp_ensemble(
    model_class, # an instance of PyTorchMLPEnsemble
    df,
    primary_features,
    target_col='QRS_Overal',
    surgeon_col='Participant Number',
    case_col='Case_Number',
    extra_features=None,
    model_params={'hidden_dim': 16, 'n_models': 5}, # Parameters for the Ensemble class
    pca_components=None, 
    scale_by_case=False,
    scale_features=True,
    verbose=True,
    print_fold_metrics=False
):
    if extra_features is None:
        extra_features = []
    
    unique_surgeons = df[surgeon_col].unique()
    all_preds, all_true, all_surgeons = [], [], []
    fold_metrics = {}
    use_pca = pca_components is not None and len(pca_components) > 0
    all_cols = primary_features + extra_features
    all_cols_to_scale = [col for col in all_cols if 'case' not in col.lower()]  # Exclude case identifier columns

    iterator = tqdm(unique_surgeons, desc="LOSOCV Ensemble Folds") if verbose else unique_surgeons

    for surgeon_out in unique_surgeons:
        # 1. Split Data
        train_mask = df[surgeon_col] != surgeon_out
        test_mask = df[surgeon_col] == surgeon_out
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        y_train = df_train[target_col].values
        y_test = df_test[target_col].values

        score_scaler = StandardScaler()
        y_train = score_scaler.fit_transform(y_train.reshape(-1, 1))

        # 2. Scaling Logic (Same as your Ridge function)
        if scale_by_case and case_col in df.columns:
            for case in df_train[case_col].unique():
                case_train_idx = df_train[df_train[case_col] == case].index
                case_test_idx = df_test[df_test[case_col] == case].index
                scaler = StandardScaler()
                df_train.loc[case_train_idx, all_cols_to_scale] = scaler.fit_transform(df_train.loc[case_train_idx, all_cols_to_scale])
                if not df_test.loc[case_test_idx].empty:
                    df_test.loc[case_test_idx, all_cols_to_scale] = scaler.transform(df_test.loc[case_test_idx, all_cols_to_scale])
        elif scale_features:
            scaler = StandardScaler()
            df_train[all_cols_to_scale] = scaler.fit_transform(df_train[all_cols_to_scale])
            df_test[all_cols_to_scale] = scaler.transform(df_test[all_cols_to_scale])

        # 3. Prepare Feature Matrices
        X_train_prim = df_train[primary_features].values
        X_test_prim = df_test[primary_features].values
        X_train_extra = df_train[extra_features].values if extra_features else np.empty((len(df_train), 0))
        X_test_extra = df_test[extra_features].values if extra_features else np.empty((len(df_test), 0))

        # 4. Optional PCA
        if use_pca:
            pca = PCA(n_components=max(pca_components) + 1)
            X_train_pca = pca.fit_transform(X_train_prim)
            X_test_pca = pca.transform(X_test_prim)
            X_train_final = X_train_pca[:, pca_components]
            X_test_final = X_test_pca[:, pca_components]
        else:
            X_train_final = X_train_prim
            X_test_final = X_test_prim

        if extra_features:
            X_train_final = np.hstack((X_train_final, X_train_extra))
            X_test_final = np.hstack((X_test_final, X_test_extra))

        # 5. Initialize and Train Ensemble
        # We inject the input_dim here because it depends on PCA/Extra features
        input_dim = X_train_final.shape[1]
        model = model_class(input_dim=input_dim, **model_params)
        
        model.fit(X_train_final, y_train)
        y_train = score_scaler.inverse_transform(y_train).flatten()
        
        # 6. Predict
        y_pred_scaled = model.predict(X_test_final)
        y_pred = score_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # 7. Metrics
        y_train_pred = score_scaler.inverse_transform(model.predict(X_train_final).reshape(-1, 1)).flatten()
        fold_train_mae = mean_absolute_error(y_train, y_train_pred)
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_corr = pearsonr(y_test, y_pred)[0] if len(y_test) > 1 else np.nan
        
        fold_metrics[surgeon_out] = {'Train_MAE': fold_train_mae, 'Test_MAE': fold_mae, 'Test_Corr': fold_corr}

        if print_fold_metrics:
            print(f"Surgeon {surgeon_out} | Train MAE: {fold_train_mae:.4f} | Test MAE: {fold_mae:.4f} | Test Corr: {fold_corr:.4f}")
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_surgeons.extend([surgeon_out] * len(y_test))

    # --- 8. Aggregate ---
    predictions_df = pd.DataFrame({'Surgeon_ID': all_surgeons, 'True_Score': all_true, 'Predicted_Score': all_preds})
    overall_mae = mean_absolute_error(all_true, all_preds)
    overall_std = np.std(np.abs(np.array(all_true) - np.array(all_preds)))
    overall_corr, _ = spearmanr(all_true, all_preds)
    overall_r2 = r2_score(all_true, all_preds)
    overall_adj_r2 = 1 - (1 - overall_r2) * (len(all_true) - 1) / (len(all_true) - X_train_final.shape[1] - 1)

    summary = {
        'Overall_MAE': overall_mae,
        'Overall_MAE_STD': overall_std,
        'Overall_Spearman_R': overall_corr,
        'Overall_R2': overall_r2,
        'Overall_Adjusted_R2': overall_adj_r2
    }

    fold_results_df = pd.DataFrame.from_dict(fold_metrics, orient='index')

    if verbose:
        scaling_type = f"By Case" if scale_by_case else "Global"
        print(f"\n=== LOSOCV Ensemble MLP Results ({scaling_type} Scaling) ===")
        print(f"Spearman R: {overall_corr:.4f} | MAE: {overall_mae:.4f} | MAE STD: {overall_std:.4f} | R2: {overall_r2:.4f} | Adjusted R2: {overall_adj_r2:.4f}")
        print(f"\n--- Per-Fold Performance Summary ---")
    
    plot_loso_results(predictions_df, title=f"LOSOCV: MLP Ensemble (N={model_params.get('n_models', 5)})")
    
    return summary, fold_results_df, predictions_df


def run_nested_loso(
    df,
    primary_features,     # To be compressed into PC1
    candidate_features,   # To be selected via Partial R2
    extra_features=None,  # Included without selection (e.g. Case IDs)
    target_col='QRS_Overal',
    surgeon_col='Participant Number',
    case_col='Case_Number',
    model_type='ridge',   # 'ridge' or 'mlp'
    model_params=None,
    top_n=2,
    pr2_threshold=0.05,
    corr_threshold=0.75,
    scale_by_case=False,
    print_fold_metrics=True
):
    if extra_features is None: extra_features = []
    if model_params is None: model_params = {}
    
    unique_surgeons = df[surgeon_col].unique()
    selector = NestedFeatureSelector(top_n=top_n, pr2_threshold=pr2_threshold, corr_threshold=corr_threshold)
    
    all_preds, all_true, all_surgeons = [], [], []
    all_preds_train, all_true_train = [], []
    selection_history = [] # To track stability

    for surgeon_out in unique_surgeons:#tqdm(unique_surgeons, desc=f"Nested LOSO ({model_type})"):
        # --- 1. Split ---
        train_idx = df[df[surgeon_col] != surgeon_out].index
        test_idx = df[df[surgeon_col] == surgeon_out].index
        
        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()
        
        # --- 2. Scale ---
        cols_to_scale = primary_features + candidate_features
        # Only scale the extra features that aren't dummy/categorical
        scale_extras = [c for c in extra_features if df[c].nunique() > 2]
        cols_to_scale += scale_extras

        if scale_by_case:
            for case in df[case_col].unique():
                c_tr = df_train[df_train[case_col] == case].index
                c_te = df_test[df_test[case_col] == case].index
                if len(c_tr) > 0:
                    scaler = StandardScaler()
                    df_train.loc[c_tr, cols_to_scale] = scaler.fit_transform(df_train.loc[c_tr, cols_to_scale])
                    if len(c_te) > 0:
                        df_test.loc[c_te, cols_to_scale] = scaler.transform(df_test.loc[c_te, cols_to_scale])
        else:
            scaler = StandardScaler()
            df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
            df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

        # --- 3. Baseline Preparation (PC1 + Extras) ---
        pca = PCA(n_components=1)
        pc1_train = pca.fit_transform(df_train[primary_features])
        pc1_test = pca.transform(df_test[primary_features])
        
        X_tr_base = np.hstack([pc1_train, df_train[extra_features].values])
        X_te_base = np.hstack([pc1_test, df_test[extra_features].values])

        # --- 4. Nested Selection ---
        selected_candidates, selected_pr2s = selector.select_features(
            X_tr_base, df_train[target_col].values, df_train[candidate_features]
        )
        selection_history.append({'Surgeon_Out': surgeon_out, 'Selected': selected_candidates})

        # --- 5. Final Feature Matrix Assembly ---
        X_train_final = np.hstack([X_tr_base, df_train[selected_candidates].values])
        X_test_final = np.hstack([X_te_base, df_test[selected_candidates].values])
        
        y_train = df_train[target_col].values
        y_test = df_test[target_col].values

        # --- 6. Model Training ---
        if model_type == 'ridge':
            model = Ridge(alpha=model_params.get('alpha', 0.5))
            model.fit(X_train_final, y_train)
            preds = model.predict(X_test_final)
            preds_train = model.predict(X_train_final)
        
        elif model_type == 'mlp':
            # Handle Score Scaling for MLP
            y_scaler = StandardScaler()
            y_tr_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
            
            # Assuming PyTorchMLPEnsemble is imported and available
            model = PyTorchMLPEnsemble(input_dim=X_train_final.shape[1], **model_params)
            model.fit(X_train_final, y_tr_scaled)
            
            preds_scaled = model.predict(X_test_final)
            preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

            preds_train_scaled = model.predict(X_train_final)
            preds_train = y_scaler.inverse_transform(preds_train_scaled.reshape(-1, 1)).flatten()

        fold_mae = mean_absolute_error(y_test, preds)
        fold_train_mae = mean_absolute_error(y_train, preds_train)

        all_preds.extend(preds)
        all_true.extend(y_test)
        all_preds_train.extend(preds_train)
        all_true_train.extend(y_train)
        all_surgeons.extend([surgeon_out] * len(y_test))

        if print_fold_metrics:
            print(f"Surgeon {surgeon_out} | Train MAE: {fold_train_mae:.4f} | Test MAE: {fold_mae:.4f} | Selected Features: {selected_candidates} | Selected PR2s: {[round(pr2,4) for pr2 in selected_pr2s]}")

    # --- 7. Aggregation & Reporting ---
    results_df = pd.DataFrame({'Surgeon_ID': all_surgeons, 'True': all_true, 'Pred': all_preds})
    stability_df = pd.DataFrame(selection_history)
    
    # Calculate Stability Statistics for the Thesis Table
    all_selected = [item for sublist in stability_df['Selected'] for item in sublist]
    stability_stats = pd.Series(all_selected).value_counts() / len(unique_surgeons)
    overall_r2 = r2_score(all_true, all_preds)
    overall_adj_r2 = 1 - (1 - overall_r2) * (len(all_true) - 1) / (len(all_true) - X_train_final.shape[1] - 1)
    
    print(f"\n=== Nested LOSO Summary ({model_type}) ===")
    print(f"MAE: {mean_absolute_error(all_true, all_preds):.4f} +/- {np.std(np.abs(np.array(all_true) - np.array(all_preds))):.4f}")
    print(f"Train MAE: {mean_absolute_error(all_true_train, all_preds_train):.4f}")
    print(f"Spearman R: {spearmanr(all_true, all_preds)[0]:.4f}")
    print(f"Overall R2: {overall_r2:.4f}")
    print(f"Overall Adjusted R2: {overall_adj_r2:.4f}")
    print("\nFeature Selection Stability:")
    print(stability_stats)
    
    return results_df, stability_stats




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

def run_automated_nested_loso(
    df,
    feature_pool,         # All possible features to consider
    extra_features=None,  # Included without selection (e.g. Case IDs)
    target_col='QRS_Overal',
    surgeon_col='Participant Number',
    case_col='Case_Number',
    model_type='ridge',
    model_params=None,
    top_n=2,
    global_corr_threshold=0.6, # Threshold for initial global feature filtering
    global_redundancy_threshold=0.95, # Threshold to remove redundant global features
    pr2_threshold=0.05,
    corr_threshold=0.75, # Threshold used inside the Partial R2 selector
    print_fold_metrics=True,
    top_n_globals=5

):
    if extra_features is None: extra_features = []
    if model_params is None: model_params = {}
    
    unique_surgeons = df[surgeon_col].unique()
    # Assuming NestedFeatureSelector is defined elsewhere in your code
    selector = NestedFeatureSelector(top_n=top_n, pr2_threshold=pr2_threshold, corr_threshold=corr_threshold)
    
    all_preds, all_true, all_surgeons = [], [], []
    all_preds_train, all_true_train = [], []
    selection_history = [] 

    for surgeon_out in unique_surgeons:
        # --- 1. Split ---
        train_idx = df[df[surgeon_col] != surgeon_out].index
        test_idx = df[df[surgeon_col] == surgeon_out].index
        
        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()
        
        # --- 2. Scaling ---
        # Scale all numeric features in the pool + non-categorical extras
        scale_extras = [c for c in extra_features if df[c].nunique() > 2]
        cols_to_scale = list(set(feature_pool + scale_extras))

        scaler = StandardScaler()
        df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
        df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

        # --- 3. Automated "Global" Selection ---
        # Calculate correlation with target for all features in the pool
        correlations = df_train[feature_pool].corrwith(df_train[target_col]).abs()
        if global_corr_threshold is not None:
            # corr threshdold
            high_corr_features = correlations[correlations > global_corr_threshold].sort_values(ascending=False).index.tolist()
        else:
            # top n
            high_corr_features = correlations.sort_values(ascending=False).head(top_n_globals).index.tolist()

        # Remove redundant features (inter-correlation > 0.95)
        # Because they are sorted by target-corr, we keep the better predictor
        final_globals = []
        to_ignore = set()
        for i, f1 in enumerate(high_corr_features):
            if f1 in to_ignore: continue
            final_globals.append(f1)
            for f2 in high_corr_features[i+1:]:
                if abs(df_train[f1].corr(df_train[f2])) > global_redundancy_threshold:
                    to_ignore.add(f2)

        # --- 4. PCA of Globals ---
        if len(final_globals) > 0:
            pca = PCA(n_components=1)
            pc1_train = pca.fit_transform(df_train[final_globals])
            pc1_test = pca.transform(df_test[final_globals])
        else:
            # Fallback if no feature hits the 0.6 threshold
            pc1_train = np.zeros((len(df_train), 1))
            pc1_test = np.zeros((len(df_test), 1))
            print(f"Warning: No Global features found for Surgeon {surgeon_out}")

        # Baseline: PC1 + Extras
        X_tr_base = np.hstack([pc1_train, df_train[extra_features].values])
        X_te_base = np.hstack([pc1_test, df_test[extra_features].values])

        # --- 5. Candidate Feature Selection (Locals) ---
        # Candidates are anything NOT used in the Global PCA
        candidate_features = [f for f in feature_pool if f not in final_globals]
        
        selected_candidates, selected_pr2s = selector.select_features(
            X_tr_base, df_train[target_col].values, df_train[candidate_features]
        )
        
        selection_history.append({
            'Surgeon_Out': surgeon_out, 
            'Globals': final_globals, 
            'Selected_Locals': selected_candidates
        })

        # --- 6. Final Matrix & Model ---
        X_train_final = np.hstack([X_tr_base, df_train[selected_candidates].values])
        X_test_final = np.hstack([X_te_base, df_test[selected_candidates].values])
        
        y_train = df_train[target_col].values
        y_test = df_test[target_col].values

        if model_type == 'ridge':
            model = Ridge(alpha=model_params.get('alpha', 0.5))
            model.fit(X_train_final, y_train)
            preds = model.predict(X_test_final)
            preds_train = model.predict(X_train_final)
        
        elif model_type == 'mlp':
            y_scaler = StandardScaler()
            y_tr_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
            model = PyTorchMLPEnsemble(input_dim=X_train_final.shape[1], **model_params)
            model.fit(X_train_final, y_tr_scaled)
            
            preds_scaled = model.predict(X_test_final)
            preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            preds_train_scaled = model.predict(X_train_final)
            preds_train = y_scaler.inverse_transform(preds_train_scaled.reshape(-1, 1)).flatten()

        # Statistics Gathering
        all_preds.extend(preds)
        all_true.extend(y_test)
        all_preds_train.extend(preds_train)
        all_true_train.extend(y_train)
        all_surgeons.extend([surgeon_out] * len(y_test))

        if print_fold_metrics:
            print(f"Surgeon {surgeon_out} | Globals: {len(final_globals)},  PC1 Explained Var: {pca.explained_variance_ratio_[0]:.4f} | Locals: {selected_candidates} with PR2s {[round(pr2,4) for pr2 in selected_pr2s]}")

    # --- 7. Aggregation ---
    results_df = pd.DataFrame({'Surgeon_ID': all_surgeons, 'True': all_true, 'Pred': all_preds})
    
    # Calculate Global Stability
    all_globals = [item for sublist in [d['Globals'] for d in selection_history] for item in sublist]
    global_stability = pd.Series(all_globals).value_counts() / len(unique_surgeons)
    
    # Calculate Local Stability
    all_locals = [item for sublist in [d['Selected_Locals'] for d in selection_history] for item in sublist]
    local_stability = pd.Series(all_locals).value_counts() / len(unique_surgeons)

    print(f"\n=== Fully Automated Nested LOSO Summary ({model_type}) ===")
    print(f"MAE: {mean_absolute_error(all_true, all_preds):.4f}")
    print(f"Error STD: {np.std(np.abs(np.array(all_true) - np.array(all_preds))):.4f}")
    print(f"Spearman R: {spearmanr(all_true, all_preds)[0]:.4f}")
    print(f"Overall R2: {r2_score(all_true, all_preds):.4f}")
    
    return results_df, global_stability, local_stability