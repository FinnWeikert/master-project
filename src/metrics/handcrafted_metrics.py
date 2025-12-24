# path length, reversals, smoothness
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import inspect
import os


import scipy.signal as signal

def compute_extended_hand_metrics(df_hand, hand_label="Right", min_disp=1.0, fps=30, velocity_threshold=20.0):
    df = df_hand.copy().sort_values("frame")

    # --- 1. dt and Segmentation ---
    df["frame_diff"] = df["frame"].diff().fillna(1)
    df["dt"] = df["frame_diff"] / fps

    for _, seg in df.groupby("segment_id", sort=False):
        seg = seg.copy()
        # Displacement for center point
        seg["disp"] = np.sqrt(np.diff(seg["cx_smooth"], prepend=np.nan)**2 +
                              np.diff(seg["cy_smooth"], prepend=np.nan)**2)

        seg["disp_filtered"] = np.where(
            seg["disp"].isna(), 
            np.nan, 
            np.where(seg["disp"] > min_disp, seg["disp"], 0.0)
        )
        df.loc[seg.index, ["disp_filtered"]] = seg[["disp_filtered"]]

    # --- 2. Velocity, Acceleration, and Jerk ---
    df["velocity"] = df["disp_filtered"] / df["dt"].replace(0, np.nan)
    df["accel"] = df["velocity"].diff() / df["dt"].replace(0, np.nan)
    # Jerk is the derivative of acceleration
    df["jerk"] = df["accel"].diff() / df["dt"].replace(0, np.nan)

    # --- 3. Total Path & Duration ---
    total_path = df["disp_filtered"].sum()
    duration = df["dt"].sum()

    # --- 4. Curvature & Reversals ---
    dx = np.diff(df["cx_smooth"].ffill())
    dy = np.diff(df["cy_smooth"].ffill())
    path_vec = np.vstack([dx, dy]).T
    disp_vals = df["disp_filtered"].values[1:]

    MIN_MOVE = 5
    valid = disp_vals > MIN_MOVE
    filtered_vec = path_vec[valid]

    if len(filtered_vec) > 1:
        angles = (np.arctan2(filtered_vec[1:,1], filtered_vec[1:,0]) -
                    np.arctan2(filtered_vec[:-1,1], filtered_vec[:-1,0]))
        angles = np.mod(angles + np.pi, 2*np.pi) - np.pi
        mean_abs_angle_change = np.mean(np.abs(angles))
        num_reversals = np.sum(np.diff(np.sign(angles)) != 0)
    else:
        mean_abs_angle_change = 0
        num_reversals = 0

    # --- 5. NEW: Number of Movement Units (NMU) ---
    # Experts have fewer velocity peaks (one smooth motion vs. many stutters)
    # Using a prominence-based peak finding on smoothed velocity
    vel_clean = df["velocity"].fillna(0)
    peaks, _ = signal.find_peaks(vel_clean, prominence=velocity_threshold)
    nmu = len(peaks)

    # --- 6. NEW: Log Dimensionless Jerk (LDLJ) ---
    # Standardized way to measure smoothness independent of time/distance
    # LDLJ = -log(| (Duration^3 / PathLength^2) * Integral(Jerk^2) |)
    jerk_squared_integral = (df["jerk"]**2).sum() * (1.0/fps)
    if total_path > 0 and duration > 0:
        ldlj = -np.log( (duration**3 / total_path**2) * jerk_squared_integral + 1e-6 )
    else:
        ldlj = np.nan

    # --- 7. Temporal & Efficiency Metrics ---
    mean_velocity = df["velocity"].mean(skipna=True)
    rms_accel = np.sqrt((df["accel"]**2).mean(skipna=True))
    efficiency = total_path / duration if duration > 0 else np.nan

    # --- 8. Duty Cycle & Intermittency ---
    moving_time = (df["velocity"] > velocity_threshold).multiply(df["dt"]).sum()
    duty_cycle = moving_time / duration if duration > 0 else np.nan
    
    moving = df["velocity"] > velocity_threshold
    intermittency_ratio = (moving.astype(int).diff().abs().sum() / len(moving)
                            if moving.any() else 0)
    
    # --- 9. Fraction Tracked & Adjustment ---
    # Safe protection against divide by zero or single-frame data
    frame_span = (df["frame"].max() - df["frame"].min())
    if frame_span > 0:
        fraction_tracked = (df["frame"].count() / frame_span) * df['frame_diff'].median()
    else:
        fraction_tracked = 1.0

    adjusted_total_path = total_path / fraction_tracked

    metrics = pd.DataFrame([{
        f"total_path_{hand_label}": total_path,
        f"adjusted_total_path_{hand_label}": adjusted_total_path,
        f"total_duration_{hand_label}": duration,
        f"mean_velocity_{hand_label}": mean_velocity,
        f"rms_accel_{hand_label}": rms_accel,
        f"ldlj_smoothness_{hand_label}": ldlj,
        f"nmu_peaks_{hand_label}": nmu,
        f"efficiency_{hand_label}": efficiency,
        f"duty_cycle_{hand_label}": duty_cycle,
        f"mean_abs_angle_change_{hand_label}": mean_abs_angle_change,
        f"num_reversals_{hand_label}": num_reversals,
        f"intermittency_ratio_{hand_label}": intermittency_ratio,
        f"fraction_tracked_{hand_label}": fraction_tracked
    }])

    return metrics
    
def compute_bimanual_dexterity(df_left, df_right, fps=30, velocity_threshold=20):
    """
    Compute bimanual dexterity metrics:
    - Inter-hand coordination via velocity correlation
    - Stability of inter-hand distance
    - Overlap of movement activity (duty cycle synchrony)

    Returns a single-row DataFrame.
    """
    df_left = df_left.copy().sort_values("frame")
    df_right = df_right.copy().sort_values("frame")
        # --- 1. dt and Segmentation ---
    df_left["frame_diff"] = df_left["frame"].diff().fillna(1)
    df_left["dt"] = df_left["frame_diff"] / fps

    for _, seg in df_left.groupby("segment_id", sort=False):
        seg = seg.copy()
        # Displacement for center point
        seg["disp"] = np.sqrt(np.diff(seg["cx_smooth"], prepend=np.nan)**2 +
                              np.diff(seg["cy_smooth"], prepend=np.nan)**2)

        seg["disp_filtered"] = np.where(
            seg["disp"].isna(), 
            np.nan, 
            np.where(seg["disp"] > 1, seg["disp"], 0.0)
        )
        df_left.loc[seg.index, ["disp_filtered"]] = seg[["disp_filtered"]]
    
    df_right["frame_diff"] = df_right["frame"].diff().fillna(1)
    df_right["dt"] = df_right["frame_diff"] / fps

    for _, seg in df_right.groupby("segment_id", sort=False):
        seg = seg.copy()
        # Displacement for center point
        seg["disp"] = np.sqrt(np.diff(seg["cx_smooth"], prepend=np.nan)**2 +
                              np.diff(seg["cy_smooth"], prepend=np.nan)**2)

        seg["disp_filtered"] = np.where(
            seg["disp"].isna(), 
            np.nan, 
            np.where(seg["disp"] > 1, seg["disp"], 0.0)
        )
        df_right.loc[seg.index, ["disp_filtered"]] = seg[["disp_filtered"]]

    # Require overlapping frames for comparison
    df = pd.merge(
        df_left[["frame", "disp_filtered"]],
        df_right[["frame", "disp_filtered"]],
        on="frame",
        how="inner",
        suffixes=("_L", "_R")
    ).copy()

    if len(df) < 5:
        return pd.DataFrame([{
            "velocity_corr": np.nan,
            "interhand_dist_mean": np.nan,
            "interhand_dist_std": np.nan,
            "movement_overlap_ratio": np.nan,
        }])

    # Compute dt (frame spacing may be irregular)
    df["frame_diff"] = df["frame"].diff()
    df["dt"] = df["frame_diff"] / fps
    df = df[df["dt"] > 0]

    # Compute velocities (px/sec)
    df["vel_L"] = df["disp_filtered_L"] / df["dt"]
    df["vel_R"] = df["disp_filtered_R"] / df["dt"]

    # --- Velocity Correlation (Coordination) ---
    if df["vel_L"].std() > 1e-6 and df["vel_R"].std() > 1e-6:
        velocity_corr = df["vel_L"].corr(df["vel_R"])
    else:
        velocity_corr = np.nan
    
    # --- velocity corrrelation but only when both hands are moving ---
    moving_mask = (df["vel_L"] > velocity_threshold) & (df["vel_R"] > velocity_threshold)
    if moving_mask.sum() >= 5:
        if df.loc[moving_mask, "vel_L"].std() > 1e-6 and df.loc[moving_mask, "vel_R"].std() > 1e-6:
            velocity_corr_moving = df.loc[moving_mask, "vel_L"].corr(df.loc[moving_mask, "vel_R"])
        else:
            velocity_corr_moving = np.nan

    # --- Inter-Hand Distance (Spatial Coordination) ---
    # Need smoothed palm center coordinates
    if "cx_smooth" in df_left.columns:
        merged_xy = pd.merge(
            df_left[["frame", "cx_smooth", "cy_smooth"]],
            df_right[["frame", "cx_smooth", "cy_smooth"]],
            on="frame",
            how="inner",
            suffixes=("_L", "_R")
        )
        merged_xy["dist"] = np.sqrt(
            (merged_xy["cx_smooth_L"] - merged_xy["cx_smooth_R"])**2 +
            (merged_xy["cy_smooth_L"] - merged_xy["cy_smooth_R"])**2
        )
        interhand_dist_mean = merged_xy["dist"].mean()
        interhand_dist_std = merged_xy["dist"].std()
    else:
        interhand_dist_mean = np.nan
        interhand_dist_std = np.nan

    # --- Movement Overlap (Bimanual Duty Cycle Synchrony) ---
    moving_L = df["vel_L"] > velocity_threshold
    moving_R = df["vel_R"] > velocity_threshold
    movement_overlap_ratio = (moving_L & moving_R).mean()

    # Inter-hand distance coefficient of variation
    interhand_dist_cv = interhand_dist_std / interhand_dist_mean if interhand_dist_mean>0 else np.nan

    # RMS of distance change (stability of spacing)
    interhand_dist_change_rms = np.sqrt(np.mean(np.diff(merged_xy["dist"])**2))

    # Velocity ratio
    vel_ratio = df["vel_L"].mean() / df["vel_R"].mean() if df["vel_R"].mean()>0 else np.nan


    # Return in a one-row dataframe
    return pd.DataFrame([{
        "velocity_corr": velocity_corr,
        "velocity_corr_moving": velocity_corr_moving,
        "interhand_dist_mean": interhand_dist_mean,
        "interhand_dist_std": interhand_dist_std,
        "movement_overlap_ratio": movement_overlap_ratio,
        "interhand_dist_cv": interhand_dist_cv,
        "interhand_dist_change_rms": interhand_dist_change_rms,
        "velocity_ratio": vel_ratio
    }])


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




def compute_rotational_metrics(df_hand, hand_label="Right", fps=30):
    """
    Computes global rotational and stability metrics based on palm orientation.
    Requires: lm_0_x_smooth, lm_0_y_smooth, lm_5_x_smooth, lm_5_y_smooth, 
              lm_17_x_smooth, lm_17_y_smooth
    """
    df = df_hand.copy().sort_values("frame")
    
    # --- 1. Vector Construction ---
    # We create two vectors from the wrist to the knuckles to define the hand plane
    # v1: Wrist -> Index MCP
    v1_x = df['lm_5_x_smooth'] - df['lm_0_x_smooth']
    v1_y = df['lm_5_y_smooth'] - df['lm_0_y_smooth']
    
    # v2: Wrist -> Pinky MCP
    v2_x = df['lm_17_x_smooth'] - df['lm_0_x_smooth']
    v2_y = df['lm_17_y_smooth'] - df['lm_0_y_smooth']

    # --- 2. Orientation Calculation ---
    # In 2D tracking, the best proxy for 'rotation' is the angle of the Index vector
    # and the 'Normal' (z-component of cross product) which shows palm open/close/tilt
    hand_angle = np.arctan2(v1_y, v1_x)
    
    # Cross product (2D) represents the signed area of the triangle 0-5-17
    # This changes when the hand tilts or 'pronates/supinates' in 2D
    cross_product = (v1_x * v2_y) - (v1_y * v2_x)
    
    # --- 3. Compute Angular Path (Rotation Economy) ---
    # We wrap the angles to -pi to pi to handle the 180 to -180 jump
    angle_diffs = np.diff(hand_angle)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    
    total_angular_path = np.sum(np.abs(angle_diffs))
    
    # --- 4. Orientation Entropy (Stability Metric) ---
    # Measures how 'random' or 'shaky' the hand orientation is
    def calculate_entropy(series, bins=36):
        # Drop NaNs and bin the angles (-pi to pi)
        counts, _ = np.histogram(series.dropna(), bins=bins, range=(-np.pi, np.pi), density=True)
        counts = counts[counts > 0] # Remove zero bins for log
        return -np.sum(counts * np.log(counts))

    orientation_entropy = calculate_entropy(hand_angle)
    
    # --- 5. Hand Area Variance (Stability of the 'Pose') ---
    # If a surgeon is confident, the area of the palm triangle stays stable.
    # If they are fumbling, the hand shape (projection) fluctuates.
    area_variability = np.nanstd(cross_product) / np.nanmean(np.abs(cross_product)) if len(cross_product) > 0 else 0

    # --- 6. Temporal Normalization ---
    duration = (df["frame"].max() - df["frame"].min()) / fps
    fraction_tracked = df["frame"].count() / (df["frame"].max() - df["frame"].min() + 1)
    
    # Adjusted metrics to account for missing tracking data
    adj_angular_path = total_angular_path / (fraction_tracked if fraction_tracked > 0 else 1)

    metrics = pd.DataFrame([{
        f"total_angular_path_{hand_label}": total_angular_path,
        f"adj_angular_path_{hand_label}": adj_angular_path,
        f"orientation_entropy_{hand_label}": orientation_entropy,
        f"pose_variability_{hand_label}": area_variability,
        f"angular_velocity_mean_{hand_label}": np.nanmean(np.abs(angle_diffs)) * fps,
    }])

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
            np.where(seg["disp"] > 1.0, seg["disp"], 0.0)
        )

        df.loc[seg.index, ["disp_filtered"]] = seg[["disp_filtered"]]

    # --- velocity & acceleration ---
    df["velocity"] = df["disp_filtered"] / df["dt"].replace(0, np.nan)
    df["accel"] = df["velocity"].diff() / df["dt"].replace(0, np.nan)

    # --- total path ---
    total_path = df["disp_filtered"].sum()

    metrics[f"rotation_per_cm_{hand_label}"] = total_angular_path / (total_path + 1e-6)

    return metrics