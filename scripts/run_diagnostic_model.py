"""
Evaluate median aggregated window features alongside global PC1 features.

This script:
1. Loads processed trajectory data and extracts window features.
2. Identifies and handles idle windows, then aggregates the features.
3. Merges aggregated local features with global PC1 features.
4. Evaluates hybrid (PC1 + Window) and local-only models using RidgeCV.
5. Saves summary results and evaluation plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression


from thesis_package.data.loaders import (
    get_eligible_files,
    load_processed_dict,
    load_scores_df,
)
from thesis_package.features.aggregation import aggregate_window_features
from thesis_package.features.local_feature_extractor import WindowFeatureExtractor
from thesis_package.training.evaluation import EvaluationConfig, LOSOEvaluator
from thesis_package.plotting.thesis_figures import plot_two_predicted_vs_true
from thesis_package.utils.script_utils import build_results_row, ensure_output_dirs, print_performance


def main():
    # ---------------- Paths ----------------
    processed_dir = "data/processed/landmark_dataframes2/"
    vid_name_map_path = "data/scores/vid_name_map.csv"
    scores_path = "data/scores/merged_scores.csv"
    pc1_features_path = "data/metrics/pc1_features.csv"

    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    ensure_output_dirs(results_dir, tables_dir, figures_dir)

    output_csv = tables_dir / "diagnostic_results.csv"

    # ---------------- Data Loading & Extraction ----------------
    print("Loading trajectory data and extracting window features...")
    eligible = get_eligible_files(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
    )
    df_dict = load_processed_dict(eligible)

    # Note: load_scores_df is loaded here per your snippet, 
    # though evaluator generally uses the merged targets.
    df_scores = load_scores_df(scores_path)

    extractor = WindowFeatureExtractor(
        hand="Right", 
        window_sec=1.5, 
        step_sec=0.5, 
        log_transform=True, 
        include_bimanual=False, 
        exclude_idle=False
    )
    df_window_features = extractor.extract_features(df_dict)

    # ---------------- Aggregation ----------------
    print("Aggregating window features...")
    df_window_features['is_idle'] = (df_window_features['total_path'] < 45).astype(int)
    df_agg = aggregate_window_features(
        df_window_features, 
        p=90, 
        active_with_idle=False
    )

    # ---------------- Merging ----------------
    print("Loading PC1 features and merging...")
    df_pc1_features = pd.read_csv(pc1_features_path)
    
    # Merge on index as per original snippet
    df_combined = pd.merge(df_pc1_features, df_agg, left_index=True, right_index=True)

    # Identify PC1 features
    pc1_features = [c for c in df_pc1_features.columns if '(R)' in c]
    if not pc1_features:
        print("Warning: No right-hand PC1 features found.")

    # Candidate aggregated features
    candidate_features = [
        'spatial_spread_median', 
        'path_ratio_median', 
        'sparc_median', 
        'palm_area_cv_median'
    ]

    # ---------------- Evaluator Setup ----------------
    evaluator = LOSOEvaluator(EvaluationConfig(
        target_col="QRS_Overal",
        surgeon_col="Participant Number",
        video_col="video_id",
        device="cpu",
        seed=42,
    ))

    results_rows = []

    # ==========================================
    # Model 1: RidgeCV (PC1 + Median Window Features)
    # ==========================================
    print("\nEvaluating Model 1: RidgeCV (PC1 + Median Window Features)...")
    diagnostic_results = evaluator.evaluate_tabular(
        df=df_combined,
        primary_features=pc1_features,
        extra_features=candidate_features,
        pca_components=[0],
        collect_weights=True,
        model=RidgeCV(alphas=np.logspace(-1, 0.5, 20)),
        verbose=False
    )

    print_performance(diagnostic_results, "RidgeCV PC1 + median of selected window features")
    print("Weights:\n", diagnostic_results['weights'])
    results_rows.append(build_results_row("PC1 + Aggregated Windows", diagnostic_results))

    # ==========================================
    # Model 2: RidgeCV (Median Window Features Only)
    # ==========================================
    print("\nEvaluating Model 2: RidgeCV (Selected window features only)...")
    ridge_results = evaluator.evaluate_tabular(
        df=df_combined,
        primary_features=candidate_features,
        collect_weights=True,
        model=RidgeCV(alphas=np.logspace(-1, 0.5, 20)),
        verbose=False
    )

    print_performance(ridge_results, "RidgeCV selected window features only")
    print("Weights:\n", ridge_results['weights'])
    results_rows.append(build_results_row("Aggregated Windows Only", ridge_results))


    # ==========================================
    # Model 3: LinearRegression (PC1 Only)
    # ==========================================
    pc1_only_results = evaluator.evaluate_tabular(
        df=df_combined,
        primary_features=pc1_features,
        extra_features=[],
        pca_components=[0],
        collect_weights=False,
        model=LinearRegression(),
        verbose=False
    )

    # Plot predicted vs true for PC1-only and the diagnostic model side by side
    save_path = figures_dir / "predicted_vs_true_comparison.png"
    _, _, _ = plot_two_predicted_vs_true(
        df_left=pc1_only_results['predictions'],
        df_right=diagnostic_results['predictions'],
        true_col="True",
        pred_col="Pred",
        left_title="PC1 baseline",
        right_title="PC1 + median local descriptors",
        save_path=save_path,
        dpi=400,
        save_pdf=True,
    )
    

    # ---------------- Save Summary ----------------
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"All evaluations complete.")
    print(f"Saved summary results to: {output_csv}")
    print(f"Saved predicted vs true comparison plot to: {save_path} and .pdf")


if __name__ == "__main__":
    main()