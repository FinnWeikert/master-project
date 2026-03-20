"""
Reproduce the final Bag-of-Words (BoW) model results reported in the thesis.

This script:
1. Loads eligible processed trajectory files
2. Extracts local window-level kinematic features
3. Loads the global feature dataframe used for PC1-based models
4. Evaluates the final BoW configurations under LOSO-CV
5. Saves summary results to results/tables/bow_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from thesis_package.data.loaders import (
    get_eligible_files,
    load_processed_dict,
    load_scores_df,
)
from thesis_package.features.kinematic_vocabulary import KinematicVocabulary
from thesis_package.features.local_feature_extractor import WindowFeatureExtractor
from thesis_package.training.evaluation import EvaluationConfig, LOSOEvaluator


def ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def print_performance(results: dict, title: str) -> None:
    summary = results["summary"]
    print(f"=== {title} ===")
    print(f"MAE:        {summary['MAE']:.4f} ± {summary['MAE_STD']:.4f}")
    print(f"Spearman ρ: {summary['Spearman_R']:.4f}")
    print(f"R²:         {summary['R2']:.4f}\n")


def build_results_row(model_name: str, results: dict) -> dict:
    summary = results["summary"]
    return {
        "Model": model_name,
        "MAE": summary["MAE"],
        "MAE_STD": summary["MAE_STD"],
        "Spearman_R": summary["Spearman_R"],
        "R2": summary["R2"],
    }


def load_window_features(
    processed_dir: str,
    vid_name_map_path: str,
    hand: str = "Right",
    window_sec: float = 1.5,
    step_sec: float = 0.5,
    log_transform: bool = False,
    include_bimanual: bool = False,
) -> pd.DataFrame:
    eligible = get_eligible_files(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
    )
    df_dict = load_processed_dict(eligible)

    extractor = WindowFeatureExtractor(
        hand=hand,
        window_sec=window_sec,
        step_sec=step_sec,
        log_transform=log_transform,
        include_bimanual=include_bimanual,
    )
    return extractor.extract_features(df_dict)


def prepare_pc1_dataframe(
    pc1_features_path: str,
    df_window_features: pd.DataFrame,
) -> pd.DataFrame:
    df_pc1_features = pd.read_csv(pc1_features_path)

    # Ensure a video_id column exists for evaluator compatibility
    if "video_id" not in df_pc1_features.columns:
        if "Vid_Name" in df_pc1_features.columns:
            df_pc1_features["video_id"] = df_pc1_features["Vid_Name"]
        else:
            unique_videos = df_window_features["video_id"].unique()
            if len(unique_videos) != len(df_pc1_features):
                raise ValueError(
                    "Could not infer video_id for pc1 feature dataframe: "
                    "no 'video_id' or 'Vid_Name' column found, and row count "
                    "does not match the number of unique window-feature videos."
                )
            df_pc1_features["video_id"] = unique_videos

    # Add one-hot case indicators if Case_Number is available
    if "Case_Number" in df_pc1_features.columns:
        df_case_onehot = pd.get_dummies(df_pc1_features["Case_Number"], prefix="Case").astype(int)
        df_pc1_features = pd.concat([df_pc1_features, df_case_onehot], axis=1)

    return df_pc1_features


def make_vocab(
    n_clusters: int,
    seed: int,
) -> KinematicVocabulary:
    return KinematicVocabulary(
        n_clusters=n_clusters,
        random_state=seed,
        model_type="kmeans",
        n_init=20,
        feature_cols=["path_ratio", "spatial_spread", "palm_area_cv", "sparc"],
        log_feats=["path_ratio", "spatial_spread", "palm_area_cv", "sparc"],
    )


def main() -> None:
    # ---------------- Paths ----------------
    processed_dir = "data/processed/landmark_dataframes2/"
    vid_name_map_path = "data/scores/vid_name_map.csv"
    pc1_features_path = "data/metrics/pc1_features.csv"

    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    ensure_output_dirs(results_dir, tables_dir)

    output_csv = tables_dir / "bow_results.csv"

    # ---------------- Data loading ----------------

    print("Extracting local window features...")
    df_window_features = load_window_features(
        processed_dir=processed_dir,
        vid_name_map_path=vid_name_map_path,
        hand="Right",
        window_sec=1.5,
        step_sec=0.5,
        log_transform=False,
        include_bimanual=False,
    )

    print("Preparing PC1 feature dataframe...")
    df_pc1_features = prepare_pc1_dataframe(
        pc1_features_path=pc1_features_path,
        df_window_features=df_window_features,
    )

    # ---------------- Idle-window filtering ----------------
    if "total_path" not in df_window_features.columns:
        raise KeyError("Expected column 'total_path' not found in extracted window features.")

    df_window = df_window_features[df_window_features["total_path"] >= 45].copy()
    frac_removed = 1 - len(df_window) / len(df_window_features)
    print(f"Fraction of windows removed due to idle filtering: {frac_removed:.4f}")

    # ---------------- Feature selection ----------------
    pc1_features = [col for col in df_pc1_features.columns if "(R)" in col]
    if not pc1_features:
        raise ValueError("No right-hand PC1 source features found in pc1 feature dataframe.")

    case_cols = ['Case_1', 'Case_2', 'Case_3']
    has_velocity_corr = "Velocity corr." in df_pc1_features.columns

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
    bow_features = ['SurgeMe_' + str(i) for i in range(18)]
    results_rows = []

    # ---------------- Model 1: PC1 + Case Type + SurgeMe (K=18) ----------------
    seed = 42
    vocab_k18 = make_vocab(n_clusters=18, seed=seed)

    bow_hybrid_case_results = evaluator.evaluate_vocabulary(
        df_pc1_features,
        df_window,
        vocab_k18,
        model=LassoCV(alphas=np.logspace(-1, 0.5, 30)),
        primary_features=pc1_features,
        bow_features=bow_features,
        extra_features=case_cols,
        leakage_free=True,
        use_baseline=True,
    )
    print_performance(bow_hybrid_case_results, "PC1 + Case Type + SurgeMe (K=18)")
    results_rows.append(
        build_results_row("PC1 + Case Type + SurgeMe", bow_hybrid_case_results)
    )

    # ---------------- Model 2: PC1 + Velocity Corr. + SurgeMe (K=18) ----------------
    if has_velocity_corr:
        vocab_k18_vel = make_vocab(n_clusters=18, seed=seed)

        bow_hybrid_velcorr_results = evaluator.evaluate_vocabulary(
            df_pc1_features,
            df_window,
            vocab_k18_vel,
            model=LassoCV(alphas=np.logspace(-1, 0.5, 30)),
            primary_features=pc1_features,
            bow_features=bow_features,
            extra_features=case_cols + ["Velocity corr."],
            leakage_free=True,
            use_baseline=True,
        )
        print_performance(
            bow_hybrid_velcorr_results,
            "PC1 + Velocity Corr. + SurgeMe (K=18)",
        )
        results_rows.append(
            build_results_row(
                "PC1 + Velocity Corr. + SurgeMe",
                bow_hybrid_velcorr_results,
            )
        )
    else:
        print("Warning: 'Velocity corr.' column not found. Skipping velocity-correlation BoW model.\n")

    # ---------------- Model 3: SurgeMe only (BoW only, K=12) ----------------
    vocab_k12 = make_vocab(n_clusters=12, seed=seed)

    bow_only_results = evaluator.evaluate_vocabulary(
        df_pc1_features,
        df_window,
        vocab_k12,
        model=LassoCV(alphas=np.logspace(-1, 0.5, 30)),
        primary_features=pc1_features,  # passed for internal API compatibility
        bow_features=bow_features,
        extra_features=[],
        leakage_free=True,
        use_baseline=False,
    )
    print_performance(bow_only_results, "SurgeMe only (K=12)")
    results_rows.append(build_results_row("SurgeMe (BoW only)", bow_only_results))

    # ---------------- Save summary ----------------
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)

    print(f"Saved summary results to: {output_csv}")


if __name__ == "__main__":
    main()