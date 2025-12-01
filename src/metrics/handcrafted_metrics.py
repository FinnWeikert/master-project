# path length, reversals, smoothness
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import inspect
import os


def compute_extended_hand_metrics(df_hand, hand_label="Right",min_disp=1.0, fps=30, velocity_threshold=20.0):
    df = df_hand.copy().sort_values("frame")

    # --- dt ---
    df["frame_diff"] = df["frame"].diff().fillna(1)
    df["dt"] = df["frame_diff"] / fps

    for _, seg in df.groupby("segment_id", sort=False):
        seg = seg.copy()

        seg["disp"] = np.sqrt(np.diff(seg["cx_smooth"], prepend=np.nan)**2 +
                              np.diff(seg["cy_smooth"], prepend=np.nan)**2)

        # apply displacement threshold
        seg["disp_filtered"] = np.where(
            seg["disp"].isna(), 
            np.nan,  # keep NaN as NaN
            np.where(seg["disp"] > min_disp, seg["disp"], 0.0)
        )

        df.loc[seg.index, ["disp_filtered"]] = seg[["disp_filtered"]]

    # --- velocity & acceleration ---
    df["velocity"] = df["disp_filtered"] / df["dt"].replace(0, np.nan)
    df["accel"] = df["velocity"].diff() / df["dt"].replace(0, np.nan)

    # --- total path ---
    total_path = df["disp_filtered"].sum()
    duration = df["dt"].sum()

    # --- curvature measures ---
    dx = np.diff(df["cx_smooth"].ffill())
    dy = np.diff(df["cy_smooth"].ffill())
    path_vec = np.vstack([dx, dy]).T
    disp = df["disp_filtered"].values[1:]

    MIN_MOVE = 5
    valid = disp > MIN_MOVE
    path_vec = path_vec[valid]

    if len(path_vec) > 1:
        angles = (np.arctan2(path_vec[1:,1], path_vec[1:,0]) -
                    np.arctan2(path_vec[:-1,1], path_vec[:-1,0]))
        angles = np.mod(angles + np.pi, 2*np.pi) - np.pi
        mean_abs_angle_change = np.mean(np.abs(angles))
        num_reversals = np.sum(np.diff(np.sign(angles)) != 0)
    else:
        mean_abs_angle_change = 0
        num_reversals = 0

    # --- efficiency & temporal metrics ---
    mean_velocity = df["velocity"].mean(skipna=True)
    rms_accel = np.sqrt((df["accel"]**2).mean(skipna=True))
    efficiency = total_path / duration if duration > 0 else np.nan

    # --- duty cycle ---
    moving_time = (df["velocity"] > velocity_threshold).multiply(df["dt"]).sum()
    duty_cycle = moving_time / duration if duration > 0 else np.nan

    # --- intermittency ---
    moving = df["velocity"] > velocity_threshold
    intermittency_ratio = (moving.astype(int).diff().abs().sum() / len(moving)
                            if moving.any() else 0)

    metrics = pd.DataFrame([{
        f"total_path_{hand_label}": total_path,
        f"total_duration_{hand_label}": duration,
        f"mean_velocity_{hand_label}": mean_velocity,
        f"rms_accel_{hand_label}": rms_accel,
        f"efficiency_{hand_label}": efficiency,
        f"duty_cycle_{hand_label}": duty_cycle,
        f"mean_abs_angle_change_{hand_label}": mean_abs_angle_change,
        f"num_reversals_{hand_label}": num_reversals,
        f"intermittency_ratio_{hand_label}": intermittency_ratio
    }])

    return metrics
    


def analyze_metrics_vs_grs(
    processed_dir: str,
    ratings_csv: str,
    metric_func,
    score_col = "QRS_Overal",
    fps: int = 30,
    correlation = "pearson",
    end = "10fps_processed.pkl"
):
    """
    Compute motion metrics (single-hand or bimanual) for each processed file
    and correlate with GRS ratings.

    Behavior:
    - If metric_func expects (df_hand, hand_label, fps), runs per-hand and merges.
    - If metric_func does NOT include 'hand_label' in its arguments,
    it is assumed to be a bimanual metric function and is called as:
        metric_func(df_left, df_right, fps)
    """

    # Check function signature to determine mode
    params = inspect.signature(metric_func).parameters
    single_hand_mode = "hand_label" in params

    processed_files = sorted([
        f for f in os.listdir(processed_dir)
        if f.endswith(end)
    ])

    df_ratings = pd.read_csv(ratings_csv)
    df_metrics = pd.DataFrame()

    for df_name in tqdm(processed_files):
        if "2024-01-17_17-09-36" in df_name or "2024-01-17_18-24-28" in df_name or "2024-01-17_18-43-42" in df_name:
            continue
        df = pd.read_pickle(os.path.join(processed_dir, df_name))

        df_left = df[df['hand_label'] == 'Left']
        df_right = df[df['hand_label'] == 'Right']

        if single_hand_mode:
            # Expected form: metric_func(df_hand, hand_label, fps)
            metrics_left = metric_func(df_left, hand_label="Left", fps=fps)
            metrics_right = metric_func(df_right, hand_label="Right", fps=fps)
            metrics = pd.concat([metrics_left, metrics_right], axis=1)

        else:
            # Expected form: metric_func(df_left, df_right, fps)
            metrics = metric_func(df_left, df_right, fps=fps)

        metrics["file"] = df_name
        df_metrics = pd.concat([df_metrics, metrics], ignore_index=True)

    # --- Merge Motion Metrics with Ratings ---
    df_full = pd.concat([df_ratings.reset_index(drop=True),
                        df_metrics.reset_index(drop=True)], axis=1)

    # --- Compute correlations vs GRS ---
    correlations = {}
    for col in df_metrics.columns:
        if col == "file":
            continue
        if df_full[col].isna().all():
            continue
        
        if correlation == "pearson":
            r, p = stats.pearsonr(df_full[score_col], df_full[col])
        elif correlation == "spearman":
            r, p = stats.spearmanr(df_full[score_col], df_full[col])
        correlations[col] = {"correlation": r, "p_value": p}

    return df_full, correlations