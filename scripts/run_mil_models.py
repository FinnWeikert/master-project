"""
Reproduce the final Multiple Instance Learning (MIL) model results.

This script:
1. Loads processed trajectory data and extracts window features.
2. Loads global PC1 features.
3. Evaluates 4 MIL-based configurations under LOSO-CV (MIL-only, Mean Pooling, Hybrid MIL, Hybrid Mean Pooling).
4. Allows loading pre-trained models or training from scratch via --retrain.
5. Saves summary results to results/tables/mil_results.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from thesis_package.data.loaders import (
    get_eligible_files,
    load_processed_dict,
    load_scores_df,
)
from thesis_package.features.attention_mil_dataset import AttentionMILDataset
from thesis_package.features.local_feature_extractor import WindowFeatureExtractor
from thesis_package.features.mil_scaler import MILFeatureScaler
from thesis_package.models.attention_mil import HybridAttentionMIL
from thesis_package.training.evaluation import EvaluationConfig, LOSOEvaluator
from thesis_package.training.mil_training import run_training_unbiased
from thesis_package.utils.script_utils import ensure_output_dirs, print_performance, build_results_row


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MIL Models for Surgical Skill Assessment")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="If set, ignores saved models and trains all configurations from scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- Paths ----------------
    # Assuming the script is run from the project root folder
    processed_dir = "data/processed/landmark_dataframes2/"
    vid_name_map_path = "data/scores/vid_name_map.csv"
    pc1_features_path = "data/metrics/pc1_features.csv"
    
    saved_models_dir = Path("saved_models")
    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    
    ensure_output_dirs(saved_models_dir, results_dir, tables_dir)
    output_csv = tables_dir / "mil_results.csv"

    # ---------------- Data Loading ----------------
    print("Loading trajectory data and extracting window features...")
    eligible = get_eligible_files(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
    )
    df_dict = load_processed_dict(eligible)

    extractor = WindowFeatureExtractor(hand="Right", window_sec=1.5, step_sec=0.5)
    df_window_features = extractor.extract_features(df_dict)

    # Remove idle windows
    df_window_features['is_idle'] = (df_window_features['total_path'] <= 45).astype(int)
    df_window_features = df_window_features[df_window_features['is_idle'] == 0]

    # Load PC1 Features
    print("Loading PC1 features...")
    df_pc1_features = pd.read_csv(pc1_features_path)
    df_pc1_features['video_id'] = df_window_features['video_id'].unique()

    # One-hot encode Case_Number
    if 'Case_Number' in df_pc1_features.columns and not any('Case_1' in col for col in df_pc1_features.columns):
        df_case_onehot = pd.get_dummies(df_pc1_features['Case_Number'], prefix='Case').astype(int)
        df_pc1_features = pd.concat([df_pc1_features, df_case_onehot], axis=1)

    # ---------------- Feature Definitions ----------------
    window_features = ['spatial_spread', 'path_ratio', 'palm_area_cv', 'sparc']
    
    # Inferring PC1 global features (adjust if your column naming differs)
    pca_global_features = [col for col in df_pc1_features.columns if "(R)" in col]
    if not pca_global_features:
        print("Warning: No PC1 features found. Check column names in pc1_features.csv")

    # ---------------- Evaluator Setup ----------------
    evaluator = LOSOEvaluator(EvaluationConfig(
        target_col="QRS_Overal",
        surgeon_col="Participant Number",
        video_col="video_id",
        device="cpu",
        seed=42,
    ))

    model_kwargs = {
        'mlp_hidden_dim': 12,
        'attention_hidden_dim': 12,
        'dropout': 0.0,
    }

    # Helper function to handle the --retrain logic
    def get_load_path(filename):
        return None if args.retrain else str(saved_models_dir / filename)

    results_rows = []

    # ==========================================
    # Model 1: MIL-Only (Window Features)
    # ==========================================
    train_kwargs_mil = {
        'epochs': 600, 'lr': 2e-3, 'train_mae_threshold': 5.5,
        'ablation': 'mil_only', 'patience': 100, 'avg_window': 10,
    }
    
    print("\nEvaluating Model 1: MIL-Only...")
    mil_only_results = evaluator.evaluate_mil(
        df_global=df_pc1_features, df_windows=df_window_features,
        window_feature_cols=window_features, pca_global_cols=pca_global_features,
        additional_global=[], mil_dataset_cls=AttentionMILDataset,
        mil_model_cls=HybridAttentionMIL, mil_feature_scaler_cls=MILFeatureScaler,
        log_feats=window_features, train_fn=run_training_unbiased,
        model_kwargs=model_kwargs, train_kwargs=train_kwargs_mil,                       
        n_ensemble=5, test_size=1, save_models=True,
        load_path=get_load_path("mil_only.pt")
    )
    # Pass the entire results dict instead of ['summary'] to match the shared util
    print_performance(mil_only_results, 'MIL-Only (Window Features)')
    results_rows.append(build_results_row("MIL-Only (Window Features)", mil_only_results))


    # ==========================================
    # Model 2: Mean Pooling Only
    # ==========================================
    train_kwargs_mean = {
        'epochs': 600, 'lr': 2e-3, 'train_mae_threshold': 5.5,
        'ablation': 'no_attention_and_global', 'patience': 100, 'avg_window': 10,
    }
    
    print("\nEvaluating Model 2: Mean Pooling Only...")
    mean_pool_results = evaluator.evaluate_mil(
        df_global=df_pc1_features, df_windows=df_window_features,
        window_feature_cols=window_features, pca_global_cols=pca_global_features,
        additional_global=[], mil_dataset_cls=AttentionMILDataset,
        mil_model_cls=HybridAttentionMIL, mil_feature_scaler_cls=MILFeatureScaler,
        log_feats=window_features, train_fn=run_training_unbiased,
        model_kwargs=model_kwargs, train_kwargs=train_kwargs_mean,                       
        n_ensemble=5, test_size=1, save_models=True,
        load_path=get_load_path("mean_pooling_only.pt")
    )
    print_performance(mean_pool_results, 'Mean Pooling Only (Window Features)')
    results_rows.append(build_results_row("Mean Pooling Only", mean_pool_results))


    # ==========================================
    # Model 3: Hybrid MIL (Window + PC1)
    # ==========================================
    train_kwargs_hybrid_mil = {
        'epochs': 600, 'lr': 2e-3, 'train_mae_threshold': 4.2,
        'ablation': None, 'patience': 100, 'avg_window': 10,
    }
    
    print("\nEvaluating Model 3: Hybrid MIL...")
    hybrid_mil_results = evaluator.evaluate_mil(
        df_global=df_pc1_features, df_windows=df_window_features,
        window_feature_cols=window_features, pca_global_cols=pca_global_features,
        additional_global=[], mil_dataset_cls=AttentionMILDataset,
        mil_model_cls=HybridAttentionMIL, mil_feature_scaler_cls=MILFeatureScaler,
        log_feats=window_features, train_fn=run_training_unbiased,
        model_kwargs=model_kwargs, train_kwargs=train_kwargs_hybrid_mil,                       
        n_ensemble=5, test_size=1, save_models=True,
        load_path=get_load_path("hybrid_mil.pt")
    )
    print_performance(hybrid_mil_results, 'Hybrid MIL (Window Features + PC1)')
    results_rows.append(build_results_row("Hybrid MIL", hybrid_mil_results))


    # ==========================================
    # Model 4: Hybrid Mean Pooling (Window + PC1)
    # ==========================================
    train_kwargs_hybrid_mean = {
        'epochs': 600, 'lr': 2e-3, 'train_mae_threshold': 4.2,
        'ablation': 'no_attention', 'patience': 100, 'avg_window': 10,
    }

    print("\nEvaluating Model 4: Hybrid Mean Pooling...")
    hybrid_mean_pool_results = evaluator.evaluate_mil(
        df_global=df_pc1_features, df_windows=df_window_features,
        window_feature_cols=window_features, pca_global_cols=pca_global_features,
        additional_global=[], mil_dataset_cls=AttentionMILDataset,
        mil_model_cls=HybridAttentionMIL, mil_feature_scaler_cls=MILFeatureScaler,
        log_feats=window_features, train_fn=run_training_unbiased,
        model_kwargs=model_kwargs, train_kwargs=train_kwargs_hybrid_mean,                       
        n_ensemble=5, test_size=1, save_models=True,
        load_path=get_load_path("hybrid_mean_pooling.pt")
    )
    print_performance(hybrid_mean_pool_results, 'Hybrid Mean Pooling (Window Features + PC1)')
    results_rows.append(build_results_row("Hybrid Mean Pooling", hybrid_mean_pool_results))

    # ---------------- Save Summary ----------------
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\nAll evaluations complete. Saved summary results to: {output_csv}")


if __name__ == "__main__":
    main()