import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any
from tqdm import tqdm

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr, spearmanr


# =========================================================
#                       CORE HELPERS
# =========================================================

def _safe_pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan


def _safe_spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else np.nan


def _metrics(y_true, y_pred, n_features=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAE_STD": np.std(np.abs(y_true - y_pred)),
        "Pearson_R": _safe_pearson(y_true, y_pred),
        "Spearman_R": _safe_spearman(y_true, y_pred),
        "R2": r2,
    }

    if n_features is not None and len(y_true) > n_features + 1 and not np.isnan(r2):
        out["Adj_R2"] = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1)
    else:
        out["Adj_R2"] = np.nan

    return out


def _make_group_splits(groups, test_size=1, seed=42):
    """
    groups: 1D iterable of surgeon IDs, one per row in the global df.
    Returns list of (train_group_values, test_group_values).
    """
    unique_groups = np.array(sorted(pd.Series(groups).dropna().unique()))

    if test_size == 1:
        return [([g for g in unique_groups if g != held_out], [held_out]) for held_out in unique_groups]

    n_groups = len(unique_groups)
    n_splits = int(1 / test_size) if isinstance(test_size, float) else n_groups // test_size
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for tr_idx, te_idx in kf.split(unique_groups):
        splits.append((unique_groups[tr_idx].tolist(), unique_groups[te_idx].tolist()))
    return splits


def _split_df(df, group_col, train_groups, test_groups):
    train_mask = df[group_col].isin(train_groups)
    test_mask = df[group_col].isin(test_groups)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def _fit_transform_columns(df_train, df_test, cols, scaler_cls=StandardScaler):
    if not cols:
        return df_train, df_test, None
    
    scaler = scaler_cls()
    
    # Cast to float first to avoid the dtype mismatch warning
    df_train[cols] = df_train[cols].astype(float)
    df_test[cols] = df_test[cols].astype(float)
    
    df_train.loc[:, cols] = scaler.fit_transform(df_train[cols])
    df_test.loc[:, cols] = scaler.transform(df_test[cols])
    
    return df_train, df_test, scaler


def _fit_transform_target(df_train, df_test, target_col):
    scaler = StandardScaler()
    y_train_raw = df_train[target_col].values.reshape(-1, 1)
    y_test_raw = df_test[target_col].values.reshape(-1, 1)

    y_train_scaled = scaler.fit_transform(y_train_raw).ravel()
    y_test_scaled = scaler.transform(y_test_raw).ravel()

    return y_train_raw.ravel(), y_test_raw.ravel(), y_train_scaled, y_test_scaled, scaler


def _build_baseline_features(
    df_train,
    df_test,
    primary_features,
    extra_features=None,
    pca_components=None,
    extra_scale_rule: Optional[Callable[[str], bool]] = None,
):
    """
    Standard pattern used across your LOSO functions:
    - scale primary + selected extras
    - optional PCA on primary
    - optional standardization of PCA outputs
    - concatenate extras
    """
    extra_features = extra_features or []

    if extra_scale_rule is None:
        extra_scale_rule = lambda c: ("case" not in c.lower()) and ("dummy" not in c.lower())

    extra_to_scale = [c for c in extra_features if extra_scale_rule(c)]
    cols_to_scale = list(primary_features) + extra_to_scale

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train, df_test, _ = _fit_transform_columns(df_train, df_test, cols_to_scale, StandardScaler)

    X_train_prim = df_train[primary_features].values
    X_test_prim = df_test[primary_features].values

    if pca_components is None:
        X_train_base = X_train_prim
        X_test_base = X_test_prim
        feature_names = list(primary_features)
    else:
        if isinstance(pca_components, int):
            pca = PCA(n_components=pca_components)
            X_train_base = pca.fit_transform(X_train_prim)
            X_test_base = pca.transform(X_test_prim)
            feature_names = [f"PC{i+1}" for i in range(pca_components)]
        else:
            pca = PCA(n_components=max(pca_components) + 1)
            X_train_pca = pca.fit_transform(X_train_prim)
            X_test_pca = pca.transform(X_test_prim)
            X_train_base = X_train_pca[:, pca_components]
            X_test_base = X_test_pca[:, pca_components]
            feature_names = [f"PC{i+1}" for i in pca_components]

        pca_scaler = StandardScaler()
        X_train_base = pca_scaler.fit_transform(X_train_base)
        X_test_base = pca_scaler.transform(X_test_base)

    if extra_features:
        X_train_extra = df_train[extra_features].values
        X_test_extra = df_test[extra_features].values
        X_train = np.hstack([X_train_base, X_train_extra])
        X_test = np.hstack([X_test_base, X_test_extra])
        feature_names = feature_names + list(extra_features)
    else:
        X_train, X_test = X_train_base, X_test_base

    return X_train, X_test, feature_names, df_train, df_test


def _extract_linear_weights(model, feature_names):
    if not hasattr(model, "coef_"):
        return None

    coef = np.ravel(model.coef_)
    intercept = np.array([np.ravel(model.intercept_)[0] if np.ndim(model.intercept_) else model.intercept_])
    values = np.concatenate([intercept, coef])

    return pd.DataFrame({
        "Feature": ["bias"] + list(feature_names),
        "Weight": values
    })


def _parse_video_surgeon(video_id):
    # matches your previous x[1] logic, but safer
    # assumes video ids like 'S3_...' or similar where the 2nd char is surgeon id
    s = str(video_id)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits[0]) if digits else s


# =========================================================
#                  MAIN EVALUATOR CLASS
# =========================================================

@dataclass
class EvaluationConfig:
    target_col: str = "QRS_Overal"
    surgeon_col: str = "Participant Number"
    video_col: str = "video_id"
    device: str = "cpu"
    seed: int = 42


class LOSOEvaluator:
    def __init__(self, config: EvaluationConfig = EvaluationConfig()):
        self.cfg = config

    # -----------------------------------------------------
    # 1) GENERIC TABULAR LOSO / GROUP-CV
    # -----------------------------------------------------
    def evaluate_tabular(
        self,
        df: pd.DataFrame,
        primary_features: Sequence[str],
        model,
        extra_features: Optional[Sequence[str]] = None,
        group_col: Optional[str] = None,
        test_size=1,
        pca_components=None,
        selector=None,
        scale_target=False,
        verbose=True,
        collect_weights=True,
        print_fold_metrics=False,
        primary_feature_corr_threshold=None,   # NEW
        min_primary_features=1,                # NEW safeguard
    ):
        """
        Covers classic Ridge / MLP-style LOSO tabular functions.

        selector API expected:
            selected_names, selected_scores = selector.select_features(
                X_train_base, y_train, df_train[candidate_cols]
            )

        If selector is None -> baseline + optional extras.
        If selector is provided -> extra_features are treated as candidate features.

        New:
        ----
        primary_feature_corr_threshold : float or None
            If not None, select primary features inside each fold using absolute
            Pearson correlation with y_train. Only features with abs(corr) >= threshold
            are retained.

        min_primary_features : int
            Safeguard so that at least this many primary features are kept.
            If too few pass threshold, the top-k correlated features are used instead.
        """
        extra_features = list(extra_features or [])
        primary_features = list(primary_features)
        group_col = group_col or self.cfg.surgeon_col

        splits = _make_group_splits(df[group_col], test_size=test_size, seed=self.cfg.seed)

        all_preds, all_true, all_groups = [], [], []
        fold_rows = []
        weight_tables = []
        selection_history = []
        primary_selection_history = []

        iterator = tqdm(splits, desc="Tabular CV") if verbose else splits

        for fold_idx, (train_groups, test_groups) in enumerate(iterator, start=1):
            df_train, df_test = _split_df(df, group_col, train_groups, test_groups)

            y_train = df_train[self.cfg.target_col].values
            y_test = df_test[self.cfg.target_col].values

            # -------------------------------------------------
            # NEW: fold-wise selection of primary features
            # -------------------------------------------------
            if primary_feature_corr_threshold is not None:
                corrs = {}
                for feat in primary_features:
                    x = df_train[feat].values
                    if np.std(x) == 0 or np.isnan(x).all():
                        corrs[feat] = 0.0
                    else:
                        c = np.corrcoef(x, y_train)[0, 1]
                        corrs[feat] = 0.0 if np.isnan(c) else abs(c)

                selected_primary_features = [
                    feat for feat, corr in corrs.items()
                    if corr >= primary_feature_corr_threshold
                ]

                # safeguard: keep at least top-k if threshold is too strict
                if len(selected_primary_features) < min_primary_features:
                    selected_primary_features = [
                        feat for feat, _ in sorted(
                            corrs.items(), key=lambda kv: kv[1], reverse=True
                        )[:max(1, min_primary_features)]
                    ]

                primary_selection_history.append({
                    "Fold": fold_idx,
                    "Test_Groups": test_groups,
                    "Selected_Primary": list(selected_primary_features),
                    "Primary_Correlations": {k: float(v) for k, v in corrs.items()},
                })
            else:
                selected_primary_features = primary_features

            # if PCA requested, make sure n_components is valid for selected primaries
            current_pca_components = pca_components
            if current_pca_components is not None:
                n_selected_primary = len(selected_primary_features)

                if n_selected_primary == 0:
                    raise ValueError("No primary features available after correlation filtering.")

                if isinstance(current_pca_components, int):
                    current_pca_components = min(current_pca_components, n_selected_primary)
                else:
                    valid_components = [c for c in current_pca_components if c < n_selected_primary]
                    if len(valid_components) == 0:
                        valid_components = [0]
                    current_pca_components = valid_components

            X_train_base, X_test_base, base_feature_names, df_train, df_test = _build_baseline_features(
                df_train=df_train,
                df_test=df_test,
                primary_features=selected_primary_features,
                extra_features=[],
                pca_components=current_pca_components,
            )

            selected_names, selected_scores = [], []
            if selector is not None and extra_features:
                selected_names, selected_scores = selector.select_features(
                    X_train_base, y_train, df_train[extra_features]
                )
                X_train = np.hstack([X_train_base, df_train[selected_names].values])
                X_test = np.hstack([X_test_base, df_test[selected_names].values])
                feature_names = base_feature_names + list(selected_names)

                selection_history.append({
                    "Fold": fold_idx,
                    "Test_Groups": test_groups,
                    "Selected": list(selected_names),
                    "Scores": list(selected_scores),
                })
            else:
                # baseline + optional extras without selection
                if extra_features:
                    df_train, df_test, _ = _fit_transform_columns(
                        df_train,
                        df_test,
                        [c for c in extra_features if "case" not in c.lower() and "dummy" not in c.lower()],
                        StandardScaler
                    )
                    X_train = np.hstack([X_train_base, df_train[extra_features].values])
                    X_test = np.hstack([X_test_base, df_test[extra_features].values])
                    feature_names = base_feature_names + list(extra_features)
                else:
                    X_train, X_test, feature_names = X_train_base, X_test_base, base_feature_names

            if scale_target:
                y_train_raw, y_test_raw, y_train_scaled, _, y_scaler = _fit_transform_target(
                    df_train, df_test, self.cfg.target_col
                )
                model_fit = clone(model) if hasattr(model, "get_params") else model.__class__(**model.get_params())
                model_fit.fit(X_train, y_train_scaled)
                y_pred = y_scaler.inverse_transform(model_fit.predict(X_test).reshape(-1, 1)).ravel()
                y_train_pred = y_scaler.inverse_transform(model_fit.predict(X_train).reshape(-1, 1)).ravel()
                y_train_eval = y_train_raw
                y_test_eval = y_test_raw
            else:
                model_fit = clone(model) if hasattr(model, "get_params") else model
                model_fit.fit(X_train, y_train)
                y_pred = np.ravel(model_fit.predict(X_test))
                y_train_pred = np.ravel(model_fit.predict(X_train))
                y_train_eval = y_train
                y_test_eval = y_test

            fold_metric = {
                "Fold": fold_idx,
                "Train_MAE": mean_absolute_error(y_train_eval, y_train_pred),
                "Test_MAE": mean_absolute_error(y_test_eval, y_pred),
                "N_Primary_Used": len(selected_primary_features),
                "Primary_Used": list(selected_primary_features),
                "Test_Groups": test_groups,
            }

            if print_fold_metrics:
                print(
                    f"Fold {fold_idx} - Test Group {test_groups}: "
                    f"Test MAE: {fold_metric['Test_MAE']:.4f}, "
                    f"Train MAE: {fold_metric['Train_MAE']:.4f}, "
                    f"N primary: {fold_metric['N_Primary_Used']}"
                )

            fold_rows.append(fold_metric)

            all_preds.extend(y_pred)
            all_true.extend(y_test_eval)
            all_groups.extend([test_groups[0] if len(test_groups) == 1 else str(test_groups)] * len(y_pred))

            if collect_weights:
                wt = _extract_linear_weights(model_fit, feature_names)
                if wt is not None:
                    wt = wt.rename(columns={"Weight": f"Fold_{fold_idx}"})
                    weight_tables.append(wt)

        results_df = pd.DataFrame({
            "Group": all_groups,
            "True": all_true,
            "Pred": all_preds
        })
        fold_df = pd.DataFrame(fold_rows)
        summary = _metrics(all_true, all_preds, n_features=len(feature_names))

        if weight_tables:
            # Set 'Feature' as the index for all dataframes to align them automatically
            indexed_tables = [df.set_index("Feature") for df in weight_tables]
            
            # Join them all at once (axis=1 is like a multi-way outer join)
            merged = pd.concat(indexed_tables, axis=1)
            
            # Calculate stats directly on the columns
            weight_report = pd.DataFrame({
                "Average_Weight": merged.mean(axis=1),
                "Std_Weight": merged.std(axis=1)
            }).reset_index() # Bring 'Feature' back as a column
        else:
            weight_report = None

        selection_df = pd.DataFrame(selection_history) if selection_history else None
        primary_selection_df = pd.DataFrame(primary_selection_history) if primary_selection_history else None

        return {
            "summary": summary,
            "fold_results": fold_df,
            "predictions": results_df,
            "weights": weight_report,
            "selection_history": selection_df,
            "primary_selection_history": primary_selection_df,   # NEW
        }

    # -----------------------------------------------------
    # 2) MIL ENSEMBLE CV
    # -----------------------------------------------------
    def evaluate_mil(
        self,
        df_global,
        df_windows,
        window_feature_cols,
        pca_global_cols,
        additional_global,
        mil_dataset_cls,
        mil_model_cls,
        train_fn,
        mil_feature_scaler_cls,
        n_ensemble=3,
        test_size=1,
        epochs=600,
        log_feats=None,
        plot=False,
        training_mode="direct",
        model_kwargs=None,
        train_kwargs=None,
    ):
        """
        Unifies:
        - run_cv_mil_direct_ensemble
        - run_cv_mil_two_phase

        Required external pieces are injected:
        - mil_dataset_cls
        - mil_model_cls
        - train_fn                  (run_training_unbiased or run_training_two_phase)
        - mil_feature_scaler_cls    (your MILFeatureScaler)

        training_mode is only informational here; behavior comes from train_fn/train_kwargs.
        """
        log_feats = log_feats or []
        model_kwargs = model_kwargs or {}
        train_kwargs = train_kwargs or {}

        df_global = df_global.copy()
        if self.cfg.surgeon_col not in df_global.columns:
            df_global[self.cfg.surgeon_col] = df_global[self.cfg.video_col].apply(_parse_video_surgeon)

        splits = _make_group_splits(df_global[self.cfg.surgeon_col], test_size=test_size, seed=self.cfg.seed)

        all_preds, all_true = [], []
        fold_rows = []

        for fold_idx, (train_groups, test_groups) in enumerate(tqdm(splits, desc="MIL CV"), start=1):
            df_g_train, df_g_test = _split_df(df_global, self.cfg.surgeon_col, train_groups, test_groups)

            train_videos = set(df_g_train[self.cfg.video_col])
            test_videos = set(df_g_test[self.cfg.video_col])

            df_w_train = df_windows[df_windows[self.cfg.video_col].isin(train_videos)].copy()
            df_w_test = df_windows[df_windows[self.cfg.video_col].isin(test_videos)].copy()

            y_train_raw, y_test_raw, y_train_scaled, y_test_scaled, score_scaler = _fit_transform_target(
                df_g_train, df_g_test, self.cfg.target_col
            )
            df_g_train["target_scaled"] = y_train_scaled
            df_g_test["target_scaled"] = y_test_scaled

            dummy_cols = [c for c in additional_global if ("case" in c.lower()) or ("dummy" in c.lower())]
            cont_additional = [c for c in additional_global if c not in dummy_cols]
            cont_global = list(pca_global_cols) + cont_additional

            df_g_train, df_g_test, _ = _fit_transform_columns(df_g_train, df_g_test, cont_global, StandardScaler)

            pca = PCA(n_components=1)
            pc1_train = pca.fit_transform(df_g_train[pca_global_cols])
            pc1_test = pca.transform(df_g_test[pca_global_cols])

            pc1_scaler = StandardScaler()
            df_g_train["pca_feat"] = pc1_scaler.fit_transform(pc1_train)
            df_g_test["pca_feat"] = pc1_scaler.transform(pc1_test)

            global_input_cols = ["pca_feat"] + cont_additional + dummy_cols

            mil_scaler = mil_feature_scaler_cls(
                feature_cols=window_feature_cols,
                log_features=log_feats,
                method="robust",
            )
            mil_scaler.fit(df_w_train)
            w_train_s = mil_scaler.transform(df_w_train)
            w_test_s = mil_scaler.transform(df_w_test)

            train_ds = mil_dataset_cls(
                w_train_s, df_g_train, window_feature_cols, global_input_cols, label_col="target_scaled"
            )
            test_ds = mil_dataset_cls(
                w_test_s, df_g_test, window_feature_cols, global_input_cols, label_col="target_scaled"
            )

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=len(train_ds), shuffle=True, collate_fn=train_ds.mil_collate_fn
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.mil_collate_fn
            )

            ensemble_preds = []
            for i in range(n_ensemble):
                torch.manual_seed(self.cfg.seed + i)

                model = mil_model_cls(
                    local_dim=len(window_feature_cols),
                    global_dim=len(global_input_cols),
                    **model_kwargs
                ).to(self.cfg.device)

                model, history = train_fn(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=epochs,
                    score_scaler=score_scaler,
                    verbose=False,
                    **train_kwargs,
                )

                if plot:
                    try:
                        plot_training_history(pd.DataFrame(history))
                    except Exception:
                        pass

                model.eval()
                cur_preds = []
                with torch.no_grad():
                    for b_bags, b_globs, _, _ in test_loader:
                        pred_scaled, _ = model(
                            b_bags[0].to(self.cfg.device),
                            b_globs[0].unsqueeze(0).to(self.cfg.device),
                            ablation=train_kwargs.get("ablation", None),
                        )
                        pred = score_scaler.inverse_transform(
                            pred_scaled.detach().cpu().numpy().reshape(-1, 1)
                        ).ravel()[0]
                        cur_preds.append(pred)

                ensemble_preds.append(np.array(cur_preds))

            avg_preds = np.mean(ensemble_preds, axis=0)

            all_preds.extend(avg_preds)
            all_true.extend(y_test_raw)

            fold_rows.append({
                "Fold": fold_idx,
                "Test_Groups": test_groups,
                "Test_MAE": mean_absolute_error(y_test_raw, avg_preds),
                "Test_Pearson": _safe_pearson(y_test_raw, avg_preds),
                "Test_Spearman": _safe_spearman(y_test_raw, avg_preds),
            })

        return {
            "summary": _metrics(all_true, all_preds),
            "fold_results": pd.DataFrame(fold_rows),
            "predictions": pd.DataFrame({"True": all_true, "Pred": all_preds}),
        }

    # -----------------------------------------------------
    # 3) VOCABULARY LOSO
    # -----------------------------------------------------
    def evaluate_vocabulary(
        self,
        df_videos,
        df_windows,
        vocab,
        model=None,
        primary_features=None,
        extra_features=None,
        selector=None,
        bow_features=None,
        leakage_free=True,
        use_baseline=True,
    ):
        """
        Unifies:
        - run_hybrid_vocabulary_loso
        - run_leaky_vocabulary_loso
        """
        model = model if model is not None else RidgeCV(np.logspace(-1, 0.5, 20))
        primary_features = list(primary_features or [])
        extra_features = list(extra_features or [])

        splits = _make_group_splits(df_videos[self.cfg.surgeon_col], test_size=1, seed=self.cfg.seed)

        all_preds, all_true, all_groups = [], [], []
        selected_profiles = []
        nb_selected_per_fold = []
        baseline_selection_history = []
        fold_centroids = []

        centers = None
        if not leakage_free and hasattr(vocab, "model"):
            centers = vocab.model.cluster_centers_ if vocab.model_type == "kmeans" else vocab.model.means_

        for fold_idx, (train_groups, test_groups) in enumerate(tqdm(splits, desc="Vocabulary LOSO"), start=1):
            df_tr, df_te = _split_df(df_videos, self.cfg.surgeon_col, train_groups, test_groups)

            if leakage_free:
                df_tr_win = df_windows[df_windows[self.cfg.video_col].isin(df_tr[self.cfg.video_col])]
                df_te_win = df_windows[df_windows[self.cfg.video_col].isin(df_te[self.cfg.video_col])]

                vocab.fit(df_tr_win)
                X_tr_bow_raw = vocab.transform(df_tr_win)
                X_te_bow_raw = vocab.transform(df_te_win)
                centers = vocab.model.cluster_centers_ if vocab.model_type == "kmeans" else vocab.model.means_

                y_tr = df_tr.set_index(self.cfg.video_col).loc[X_tr_bow_raw.index, self.cfg.target_col].values
                y_te = df_te.set_index(self.cfg.video_col).loc[X_te_bow_raw.index, self.cfg.target_col].values
            else:
                if bow_features is None:
                    raise ValueError("bow_features must be provided when leakage_free=False")
                X_tr_bow_raw = df_tr[bow_features]
                X_te_bow_raw = df_te[bow_features]
                y_tr = df_tr[self.cfg.target_col].values
                y_te = df_te[self.cfg.target_col].values

            bow_scaler = StandardScaler()
            X_tr_bow = pd.DataFrame(
                bow_scaler.fit_transform(X_tr_bow_raw),
                columns=X_tr_bow_raw.columns,
                index=X_tr_bow_raw.index,
            )
            X_te_bow = pd.DataFrame(
                bow_scaler.transform(X_te_bow_raw),
                columns=X_te_bow_raw.columns,
                index=X_te_bow_raw.index,
            )

            if use_baseline:
                X_tr_base, X_te_base, base_names, _, _ = _build_baseline_features(
                    df_train=df_tr,
                    df_test=df_te,
                    primary_features=primary_features,
                    extra_features=extra_features,
                    pca_components=1,
                )
                base_names = ["PC1"] + list(extra_features)
            else:
                X_tr_base = np.empty((len(X_tr_bow), 0))
                X_te_base = np.empty((len(X_te_bow), 0))
                base_names = []

            if selector is not None:
                selected_bow, selected_scores = selector.select_features(X_tr_base, y_tr, X_tr_bow)
            else:
                selected_bow = list(X_tr_bow.columns)
                selected_scores = None

            X_train = np.hstack([X_tr_base, X_tr_bow[selected_bow].values])
            X_test = np.hstack([X_te_base, X_te_bow[selected_bow].values])

            model_fit = clone(model) if hasattr(model, "get_params") else model
            model_fit.fit(X_train, y_tr)
            preds = np.ravel(model_fit.predict(X_test))

            all_preds.extend(preds)
            all_true.extend(y_te)
            all_groups.extend([test_groups[0]] * len(preds))

            fold_centroids.append(centers)

            coefs = np.ravel(model_fit.coef_) if hasattr(model_fit, "coef_") else None
            n_base = X_tr_base.shape[1]

            if use_baseline and coefs is not None:
                baseline_selection_history.append({
                    name: int(i < len(coefs) and abs(coefs[i]) > 0.1)
                    for i, name in enumerate(base_names)
                })

            if selector is not None:
                nb_selected_per_fold.append(len(selected_bow))
                for feat_name, score in zip(selected_bow, selected_scores):
                    cluster_idx = int(str(feat_name).split("_")[-1])
                    profile = {
                        "Fold": fold_idx,
                        "Surgeon_Out": test_groups[0],
                        "Feature_Name": feat_name,
                        "Importance": score,
                        "Type": "Selector",
                    }
                    for j, col in enumerate(vocab.feature_cols):
                        profile[col] = centers[cluster_idx][j]
                    selected_profiles.append(profile)
            elif coefs is not None:
                chosen = 0
                for i, coef_val in enumerate(coefs[n_base:], start=n_base):
                    if abs(coef_val) > 0.01:
                        feat_name = selected_bow[i - n_base]
                        cluster_idx = int(str(feat_name).split("_")[-1])
                        profile = {
                            "Fold": fold_idx,
                            "Surgeon_Out": test_groups[0],
                            "Feature_Name": feat_name,
                            "Importance": coef_val,
                            "Type": "Model_Weight",
                        }
                        for j, col in enumerate(vocab.feature_cols):
                            profile[col] = centers[cluster_idx][j]
                        selected_profiles.append(profile)
                        chosen += 1
                nb_selected_per_fold.append(chosen)

        baseline_stats = pd.DataFrame(baseline_selection_history).mean() if baseline_selection_history else None

        return {
            "summary": _metrics(all_true, all_preds),
            "predictions": pd.DataFrame({"Surgeon": all_groups, "True": all_true, "Pred": all_preds}),
            "profiles": pd.DataFrame(selected_profiles),
            "baseline_stability": baseline_stats,
            "fold_centroids": fold_centroids,
            "avg_n_selected": float(np.mean(nb_selected_per_fold)) if nb_selected_per_fold else np.nan,
        }