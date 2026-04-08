"""
Reproduce the final global-model results reported in the thesis.

This script:
1. Loads eligible tracking files
2. Extracts global kinematic features
3. Merges them with rating metadata
4. Evaluates the final global baselines under LOSO-CV
5. Saves summary results to results/tables/global_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from thesis_package.data.loaders import get_eligible_files, load_scores_df
from thesis_package.features.global_feature_extractor import SurgicalFeatureExtractor
from thesis_package.models.mlp_regressor import PyTorchMLPEnsemble
from thesis_package.training.evaluation import EvaluationConfig, LOSOEvaluator


def print_performance(results: dict, title: str) -> None:
    """Pretty-print summary metrics for one experiment."""
    summary = results["summary"]
    print(f"=== {title} ===")
    print(f"MAE:        {summary['MAE']:.4f} ± {summary['MAE_STD']:.4f}")
    print(f"Spearman ρ: {summary['Spearman_R']:.4f}")
    print(f"R²:         {summary['R2']:.4f}\n")


def extract_global_features(
    processed_dir: str,
    vid_name_map_path: str,
    fps: int = 30,
    min_disp: float = 0,
    vel_threshold: float = 30,
) -> pd.DataFrame:
    """
    Extract one-row global feature summaries for all eligible videos.
    """
    eligible = get_eligible_files(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
    )

    all_metrics = []
    for vid_id, part_id, path in tqdm(eligible, desc="Global feature extraction"):
        df_raw = pd.read_pickle(path)

        extractor = SurgicalFeatureExtractor(
            df_raw,
            fps=fps,
            min_disp=min_disp,
            vel_threshold=vel_threshold,
        )
        feats = extractor.features_df.copy()
        feats["Vid_Name"] = vid_id

        all_metrics.append(feats)

    if not all_metrics:
        raise ValueError("No eligible files found. Check input paths and metadata.")

    return pd.concat(all_metrics, ignore_index=True)


def prepare_full_dataframe(
    processed_dir: str,
    vid_name_map_path: str,
    scores_path: str,
) -> pd.DataFrame:
    """
    Build the full dataframe used for global model evaluation.
    """
    df_features = extract_global_features(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
    )

    df_ratings = load_scores_df(scores_path)
    df_full = pd.merge(df_features, df_ratings, on="Vid_Name")

    # Add one-hot encoded case indicators
    if "Case_Number" in df_full.columns:
        df_case_onehot = pd.get_dummies(df_full["Case_Number"], prefix="Case").astype(int)
        df_full = pd.concat([df_full, df_case_onehot], axis=1)

    return df_full


def build_results_row(model_name: str, results: dict) -> dict:
    """Convert evaluation output into a flat row for CSV export."""
    summary = results["summary"]
    return {
        "Model": model_name,
        "MAE": summary["MAE"],
        "MAE_STD": summary["MAE_STD"],
        "Spearman_R": summary["Spearman_R"],
        "R2": summary["R2"],
    }


def ensure_output_dirs(*paths: Path) -> None:
    """Create output directories if needed."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---------------- Paths ----------------
    processed_dir = "data/processed/landmark_dataframes2/"
    vid_name_map_path = "data/scores/vid_name_map.csv"
    scores_path = "data/scores/merged_scores.csv"

    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    ensure_output_dirs(results_dir, tables_dir)

    output_csv = tables_dir / "global_results.csv"

    # ---------------- Data prep ----------------
    print("Preparing global feature dataframe...")
    df_full = prepare_full_dataframe(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
        scores_path=scores_path,
    )

    # Sanity checks
    required_cols = {"Vid_Name", "Participant Number", "QRS_Overal"}
    missing_required = required_cols.difference(df_full.columns)
    if missing_required:
        raise KeyError(f"Missing required columns in evaluation dataframe: {missing_required}")

    # Some evaluation utilities expect a specific video column
    if "video_id" not in df_full.columns:
        df_full["video_id"] = df_full["Vid_Name"]

    # Select right-hand features
    right_features = [col for col in df_full.columns if "(R)" in col]
    right_features.remove('Total duration (R)')

    if not right_features:
        raise ValueError("No right-hand features found. Check feature extraction output.")

    # Case dummy columns (sorted for deterministic order)
    case_cols = ['Case_1', 'Case_2', 'Case_3']

    # ---------------- Evaluator ----------------
    evaluator = LOSOEvaluator(
        EvaluationConfig(
            target_col="QRS_Overal",
            surgeon_col="Participant Number",
            video_col="video_id",
            device="cpu",
            seed=42,
        )
    )

    results_rows = []

    # ---------------- Ridge: PC1 only ----------------
    ridge_pc1_results = evaluator.evaluate_tabular(
        df=df_full,
        primary_features=right_features,
        model=RidgeCV(alphas=np.logspace(-1, 0.5, 20)),
        pca_components=[0],
        primary_feature_corr_threshold=0.5,
        verbose=False,
    )
    print_performance(ridge_pc1_results, "Linear with PC1 only")
    results_rows.append(build_results_row("PC1 only (Linear)", ridge_pc1_results))

    # ---------------- Ridge: PC1 + case type ----------------
    if case_cols:
        ridge_pc1_case_results = evaluator.evaluate_tabular(
            df=df_full,
            primary_features=right_features,
            extra_features=case_cols,
            model=RidgeCV(alphas=np.logspace(-1, 0.5, 20)),
            pca_components=[0],
            primary_feature_corr_threshold=0.5,
            verbose=False,
        )
        print_performance(ridge_pc1_case_results, "RidgeCV with PC1 + case type")
        results_rows.append(build_results_row("PC1 + Case Type (Ridge)", ridge_pc1_case_results))
    else:
        print("Warning: No case columns found. Skipping PC1 + case type evaluation.\n")

    # ---------------- Ridge: PC1 + velocity correlation ----------------
    velocity_corr_col = "Velocity corr."
    if velocity_corr_col in df_full.columns:
        ridge_pc1_velcorr_results = evaluator.evaluate_tabular(
            df=df_full,
            primary_features=right_features,
            extra_features=[velocity_corr_col],
            model=RidgeCV(alphas=np.logspace(-1, 0.5, 20)),
            pca_components=[0],
            primary_feature_corr_threshold=0.5,
            verbose=False,
        )
        print_performance(ridge_pc1_velcorr_results, "RidgeCV with PC1 + velocity correlation")
        results_rows.append(
            build_results_row("PC1 + Velocity Corr. (Ridge)", ridge_pc1_velcorr_results)
        )
    else:
        print("Warning: 'Velocity corr.' column not found. Skipping velocity-correlation model.\n")

    # ---------------- Linear: Duration only ----------------
    velocity_corr_col = "Velocity corr."
    if velocity_corr_col in df_full.columns:
        ridge_pc1_velcorr_results = evaluator.evaluate_tabular(
            df=df_full,
            primary_features=['Total duration (R)'],
            model=RidgeCV(alphas=np.logspace(-1, 0.5, 20)),
            verbose=False,
        )
        print_performance(ridge_pc1_velcorr_results, "Linear Task Duration ")
        results_rows.append(
            build_results_row("Duration (Linear)", ridge_pc1_velcorr_results)
        )
    else:
        print("Warning: 'Velocity corr.' column not found. Skipping velocity-correlation model.\n")
        
    # ---------------- MLP: PC1 + case type ----------------
    if case_cols:
        print("Evaluating MLP ensemble with PC1 + case type...")
        mlp = PyTorchMLPEnsemble(
            input_dim=1 + len(case_cols),  # PC1 + case dummies
            hidden_dim=16,
            n_hidden=1,
            n_models=5,
            dropout=0.15,
            batch_size=16,
        )

        mlp_pc1_case_results = evaluator.evaluate_tabular(
            df=df_full,
            primary_features=right_features,
            extra_features=case_cols,
            model=mlp,
            pca_components=[0],
            scale_target=True,
            primary_feature_corr_threshold=0.5,
            verbose=False,
            print_fold_metrics=True,
        )
        print_performance(mlp_pc1_case_results, "MLP ensemble with PC1 + case type")
        results_rows.append(build_results_row("PC1 + Case Type (MLP)", mlp_pc1_case_results))

    # ---------------- Save summary table ----------------
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)

    print(f"Saved summary results to: {output_csv}")


if __name__ == "__main__":
    main()