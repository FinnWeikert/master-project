import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm
import inspect
from scipy.signal import savgol_filter



#################################################
########## Hand Tracking UTILITIES ##############
#################################################

# post processing function

def enforce_hand_label_consistency(df, max_jump_px=250, center_col: str = "palm_center"):
    """
    Ensures each frame has at most one Left and one Right hand,
    keeping the closest to previous known position for each label.
    Removes outliers with too large jumps.
    Enforces consistent labeling based on last known position if multiple hands are detected.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    cleaned = []
    last_pos = {"Left": None, "Right": None}
    last_frame = {"Left": 0, "Right": 0}

    for frame, group in df.groupby("frame"):
        frame_data = []

        for label in ["Left", "Right"]:
            hands = group[group["hand_label"] == label]

            if len(hands) == 0:
                # Hand missing
                if last_pos[label] is not None:
                    px, py = last_pos[label]
                    # If close to frame edge, likely left frame
                    if px < 25 or px > 1920 - 25 or py < 25 or py > 1080 - 25:
                        last_pos[label] = None

                # Clear stale positions if missing for too long
                if frame - last_frame[label] > 6:
                    last_pos[label] = None
                continue

            elif len(hands) == 1:
                row = hands.iloc[0]
                # Check that last known position of other hand is not too close
                other_label = "Right" if label == "Left" else "Left"
                if last_pos[other_label] is not None:
                    ox, oy = last_pos[other_label]
                    hx, hy = row[center_col]
                    if np.hypot(hx - ox, hy - oy) < 30:  # too close
                        last_pos[label] = None
                        continue

            else:
                # Multiple hands detected, pick closest to last known position
                hands = hands.copy()
                if last_pos[label] is not None:
                    px, py = last_pos[label]
                    hands["dist"] = hands[center_col].apply(lambda c: np.hypot(c[0] - px, c[1] - py))
                    row = hands.loc[hands["dist"].idxmin()]
                else:
                    row = hands.iloc[0]

            # Skip frames with unrealistic jumps
            if last_pos[label] is not None:
                dx = row[center_col][0] - last_pos[label][0]
                dy = row[center_col][1] - last_pos[label][1]
                dist = np.hypot(dx, dy)
                if dist > max_jump_px:
                    if frame - last_frame[label] > 6:
                        last_pos[label] = row[center_col] # long gap, resume tracking
                    # Skip this frame but don't update last_pos
                    continue

            frame_data.append(row)
            last_pos[label] = row[center_col]
            last_frame[label] = frame

        cleaned.extend(frame_data)

    return pd.DataFrame(cleaned).reset_index(drop=True).drop(columns=["dist"], errors='ignore')


# --- Functions for distance computation ---
# can exptend later for time, velocity, acceleration, etc.


def generate_segments(df, fps=30, max_gap_sec=0.2):
    # Create an explicit copy to avoid SettingWithCopyWarning
    df = df.copy() 
    
    # --- generate segment ids based on gaps ---
    # fps should be fix at 30 
    max_gap = int(max_gap_sec * fps)

    # Calculate the difference in frame numbers
    df['frame_diff'] = df['frame'].diff()

    # Local segment id: increment whenever frame gap > max_gap
    # The cumsum on a boolean Series (True=1, False=0) creates the segment IDs
    df['segment_id'] = (df['frame_diff'] > max_gap).cumsum()
    
    return df

def smooth_and_compute_distance(
    df: pd.DataFrame,
    center_col: str = "palm_center",
    fps: int = 30,
    window: int = 5,
    poly: int = 2,
    min_disp: float = 2.0,
    max_jump_px: float = 85.0
):
    """
    Smooths hand-tracking coordinates within segments and computes total distance traveled per segment.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must include 'frame', 'segment_id', and a coordinate tuple column (e.g. 'palm_center' or 'bbox_center').
    center_col : str
        Column name containing (x, y) coordinate tuples.
    fps : int
        Frame rate (used only for interpretation / parameter scaling).
    window : int
        Smoothing window size for Savitzky–Golay filter (must be odd).
    poly : int
        Polynomial order for the Savitzky–Golay filter.
    min_disp : float
        Minimum displacement (in pixels) to count as real motion.

    Returns
    -------
    df_proc : pd.DataFrame
        Original dataframe with added smoothed coordinates and displacement columns.
    segment_summary : pd.DataFrame
        Total distance (in pixels) per segment.
    """
    df = df.copy()

    # --- Extract x/y numeric columns ---
    df["cx"] = df[center_col].apply(lambda p: p[0] if isinstance(p, (list, tuple)) and p[0] is not None else np.nan)
    df["cy"] = df[center_col].apply(lambda p: p[1] if isinstance(p, (list, tuple)) and p[1] is not None else np.nan)

    # --- Prepare new columns ---
    df["cx_smooth"], df["cy_smooth"], df["disp"], df["disp_filtered"], df["disp_raw"] = np.nan, np.nan, np.nan, np.nan, np.nan

    # --- Ensure odd window ---
    if window % 2 == 0:
        window += 1

    # --- Process each segment separately ---
    for seg_id, seg in df.groupby("segment_id", sort=False):
        seg = seg.copy()

        # if too few points, skip smoothing
        if len(seg) >= window:
            seg["cx_smooth"] = savgol_filter(seg["cx"], window, poly, mode="interp")
            seg["cy_smooth"] = savgol_filter(seg["cy"], window, poly, mode="interp")
        else:
            seg["cx_smooth"], seg["cy_smooth"] = seg["cx"], seg["cy"]

        # compute displacement *within* segment
        seg["disp_raw"] = np.sqrt(np.diff(seg["cx"], prepend=np.nan)**2 +
                              np.diff(seg["cy"], prepend=np.nan)**2)

        seg["disp"] = np.sqrt(np.diff(seg["cx_smooth"], prepend=np.nan)**2 +
                              np.diff(seg["cy_smooth"], prepend=np.nan)**2)

        # apply displacement threshold
        seg["disp_filtered"] = np.where(
            seg["disp"].isna(), 
            np.nan,  # keep NaN as NaN
            np.where(seg["disp"] > min_disp, seg["disp"], 0.0)
        )


        # write back to main df
        df.loc[seg.index, ["cx_smooth", "cy_smooth", "disp_raw", "disp", "disp_filtered"]] = seg[["cx_smooth", "cy_smooth", "disp_raw", "disp", "disp_filtered"]]

    # --- Summarize per segment ---
    segment_summary = (
        df.groupby("segment_id", dropna=False)[['disp', 'disp_filtered', 'disp_raw']]
        .sum()
        .rename(columns={'disp': 'total_disp', 'disp_filtered': 'total_disp_filtered', 'disp_raw': 'total_disp_raw'})
        .reset_index()
    )

    return df.drop(columns=["cx", "cy"]), segment_summary

def postprocess_and_compute_distance(df: pd.DataFrame,
                     max_gap_sec=0.2,
                     center_col: str = "palm_center",
                     window: int = 5,
                     poly: int = 2,
                     min_disp: float = 2.0,
                     max_jump: int = 250):
    
    # First, ensure hand labels are consistent

    try:
        df["hand_label"] = df["hand_label"].map({"Left": "Right", "Right": "Left"})
    except KeyError:
        return df, pd.DataFrame(columns=["segment_id", "total_disp", "total_disp_filtered", "total_disp_raw"])

    df = enforce_hand_label_consistency(df, max_jump_px=max_jump, center_col=center_col)

    df_hand_left = df[df['hand_label'] == 'Left']
    df_hand_right = df[df['hand_label'] == 'Right']

    df_hand_left = generate_segments(df_hand_left, max_gap_sec=max_gap_sec)
    df_hand_right = generate_segments(df_hand_right, max_gap_sec=max_gap_sec)

    df_hand_left, _ = smooth_and_compute_distance(df_hand_left, window=window, poly=poly, min_disp=min_disp)
    df_hand_right, _ = smooth_and_compute_distance(df_hand_right, window=window, poly=poly, min_disp=min_disp)

    df_combined = pd.concat([df_hand_left, df_hand_right]).sort_values("frame").reset_index(drop=True)

    return df_combined


def draw_hands(df_hand_left, df_hand_right, title):
    plt.figure(figsize=(12, 6))
    
    # blank background
    plt.imshow(np.zeros((1080, 1920, 3), dtype=np.uint8))
    
    # Draw left hand path by segment_id
    for seg_id, seg_data in df_hand_left.groupby('segment_id'):
        plt.plot(seg_data['cx_smooth'], seg_data['cy_smooth'], 
                 color='green', linewidth=0.5)
        plt.scatter(seg_data['cx_smooth'], seg_data['cy_smooth'], 
                    color='green', s=0.5, label='_nolegend_')  # avoid duplicate legend
    
    # Draw right hand path by segment_id
    for seg_id, seg_data in df_hand_right.groupby('segment_id'):
        plt.plot(seg_data['cx_smooth'], seg_data['cy_smooth'], 
                 color='red', linewidth=0.5)
        plt.scatter(seg_data['cx_smooth'], seg_data['cy_smooth'], 
                    color='red', s=0.5, label='_nolegend_')
    
    # coordinate space setup
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    
    plt.title(title)
    # Custom legend
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Left Hand'),
        Line2D([0], [0], color='red', lw=2, label='Right Hand')
    ]

    plt.legend(handles=legend_handles)
    plt.show()

#################################################
############## METRICS UTILITIES ################
#################################################


def compute_motion_metrics(df_hand, hand_label="Left", fps=30):
    df = df_hand.copy()

    # real time delta in seconds
    df["dt"] = df["frame_diff"] / fps

    # Avoid division by zero or invalid dt
    df = df[df["dt"] > 0]

    # velocity (px/s)
    df["velocity"] = df["disp_filtered"] / df["dt"]

    # acceleration (px/s^2)
    df["accel"] = df["velocity"].diff() / df["dt"]

    # jerk (px/s^3)
    df["jerk"] = df["accel"].diff() / df["dt"]

    # RMS values for smoothness
    std_speed = df['velocity'].std(skipna=True)
    mean_speed = df['velocity'].mean(skipna=True)
    rms_accel = np.sqrt((df["accel"]**2).mean())
    rms_jerk = np.sqrt((df["jerk"]**2).mean())

    # tremor approximation (since I sample at 10Hz likely not capturing high freq tremor)
    df["vel_smooth"] = df["velocity"].rolling(7, center=True, min_periods=1).mean()
    tremor_energy = ((df["velocity"] - df["vel_smooth"])**2).mean()

    # duty cycle (was the hand active vs idle?)
    velocity_threshold = 20.0  # px/s
    movement_time = (df['velocity'] > velocity_threshold).multiply(df['dt']).sum()
    total_time = df['dt'].sum()
    duty_cycle = movement_time / total_time


    df_metrics = pd.DataFrame({
        f"total_path_{hand_label.lower()}": [df["disp_filtered"].sum()],
        f"mean_speed_{hand_label.lower()}": [mean_speed],
        f"std_speed_{hand_label.lower()}": [std_speed],
        f"rms_accel_{hand_label.lower()}": [rms_accel],
        f"rms_jerk_{hand_label.lower()}": [rms_jerk],
        f"tremor_energy_{hand_label.lower()}": [tremor_energy],
        f"duty_cycle_{hand_label.lower()}": [duty_cycle],
    })
    
    return df_metrics


def compute_extended_hand_metrics(df_hand, hand_label="Left", fps=30, velocity_threshold=20.0):
    """
    Compute extended surgical skill metrics for a single hand trajectory.
    
    Parameters
    ----------
    df_hand : pd.DataFrame
        Must include: 'frame', 'disp_filtered', 'cx_smooth', 'cy_smooth'
    hand_label : str
        Name of the hand ("Left" or "Right")
    fps : int
        Original video FPS
    velocity_threshold : float
        Threshold (px/sec) for defining active movement
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with extended skill metrics
    """
    df = df_hand.copy().sort_values("frame")

    # --- Compute dt ---
    df['frame_diff'] = df['frame'].diff().fillna(1)
    df['dt'] = df['frame_diff'] / fps

    # --- Velocity & acceleration ---
    df['velocity'] = df['disp_filtered'] / df['dt'].replace(0, np.nan)
    df['accel'] = df['velocity'].diff() / df['dt'].replace(0, np.nan)

    # --- Economy of Motion (path, curvature) ---
    total_path = df['disp_filtered'].sum()

    dx = np.diff(df['cx_smooth'].ffill())
    dy = np.diff(df['cy_smooth'].ffill())
    path_vec = np.vstack([dx, dy]).T

    # Displacement magnitude
    disp = np.sqrt(dx**2 + dy**2)

    # Remove vectors caused by tracking jitter
    MIN_MOVE = 5  # tune between 3–6 depending on dataset
    valid = disp > MIN_MOVE
    path_vec = path_vec[valid]

    if len(path_vec) > 1:
        angles = np.arctan2(path_vec[1:,1], path_vec[1:,0]) - np.arctan2(path_vec[:-1,1], path_vec[:-1,0])
        angles = np.mod(angles + np.pi, 2*np.pi) - np.pi
        mean_abs_angle_change = np.mean(np.abs(angles))
        num_reversals = np.sum(np.diff(np.sign(angles)) != 0)  # rough reversal count
    else:
        mean_abs_angle_change = 0
        num_reversals = 0

    # --- Efficiency ---
    duration = df['dt'].sum()
    mean_velocity = df['velocity'].mean(skipna=True)
    rms_accel = np.sqrt((df['accel']**2).mean(skipna=True))
    efficiency = total_path / duration if duration > 0 else np.nan

    # --- Duty cycle: time fraction moving ---
    movement_time = (df['velocity'] > velocity_threshold).multiply(df['dt']).sum()
    duty_cycle = movement_time / duration if duration > 0 else np.nan

    # --- Tremor / micro-movements ---
    stationary_mask = df['velocity'] < velocity_threshold
    tremor_rms = np.sqrt((df.loc[stationary_mask, 'velocity']**2).mean()) if stationary_mask.any() else 0

    # --- Movement intermittency ratio ---
    moving = df['velocity'] > velocity_threshold
    if moving.any():
        intermittency_ratio = moving.astype(int).diff().abs().sum() / len(moving)
    else:
        intermittency_ratio = 0

    # --- Build single-row DataFrame ---
    metrics = pd.DataFrame([{
        f"total_path_{hand_label.lower()}": total_path,
        f"total_duration_{hand_label.lower()}": duration,
        f"mean_velocity_{hand_label.lower()}": mean_velocity,
        f"rms_accel_{hand_label.lower()}": rms_accel,
        f"efficiency_{hand_label.lower()}": efficiency,
        f"duty_cycle_{hand_label.lower()}": duty_cycle,
        f"tremor_rms_{hand_label.lower()}": tremor_rms,
        f"mean_abs_angle_change_{hand_label.lower()}": mean_abs_angle_change,
        f"num_reversals_{hand_label.lower()}": num_reversals,
        f"intermittency_ratio_{hand_label.lower()}": intermittency_ratio
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
    fps: int = 30,
    correlation = "pearson"
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
        if f.endswith("processed.pkl")
    ])

    df_ratings = pd.read_csv(ratings_csv)
    df_metrics = pd.DataFrame()

    for df_name in tqdm(processed_files):
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
            r, p = stats.pearsonr(df_full["GRS_Total"], df_full[col])
        elif correlation == "spearman":
            r, p = stats.spearmanr(df_full["GRS_Total"], df_full[col])
        correlations[col] = {"correlation": r, "p_value": p}

    return df_full, correlations


def correlated_metrics(df_metrics: pd.DataFrame, correlations: dict, threshold: float = 0.3):
    """
    Compute pairwise correlations between metrics that correlate with GRS above a threshold.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame of motion metrics (columns = metrics, rows = files)
    correlations : dict
        Dictionary of correlations vs GRS, as returned by analyze_metrics_vs_grs
        Format: {metric_name: {'correlation': r, 'p_value': p}}
    threshold : float
        Minimum absolute correlation with GRS to include metric

    Returns
    -------
    df_pairwise_corr : pd.DataFrame
        DataFrame of pairwise Pearson correlations between selected metrics
    selected_metrics : list
        List of metric names included (those with |r| > threshold)
    """

    # --- Step 1: select metrics that correlate with GRS above threshold ---
    selected_metrics = [
        m for m, val in correlations.items()
        if abs(val['correlation']) > threshold
    ]

    if len(selected_metrics) < 2:
        print("Not enough metrics meet the correlation threshold.")
        return pd.DataFrame(), selected_metrics

    # --- Step 2: compute pairwise correlations between selected metrics ---
    df_pairwise_corr = df_metrics[selected_metrics].corr(method='pearson')

    return df_pairwise_corr, selected_metrics


def plot_metric_correlations(correlations_dict, title="Correlation with GRS_Total"):
    """
    Plot correlations as a bar chart with correlation values on top.
    
    Parameters
    ----------
    correlations_dict : dict
        Dictionary of correlations, format: {metric_name: {'correlation': r, 'p_value': p}}
    title : str
        Plot title
    """
    metrics = list(correlations_dict.keys())
    corrs = [v['correlation'] for v in correlations_dict.values()]
    pvals = [v['p_value'] for v in correlations_dict.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, corrs, color='skyblue')
    plt.ylabel("Pearson Correlation with GRS_Total")
    plt.title(title)
    
    # Show correlation and p-value on top of bars
    for i, (r, p) in enumerate(zip(corrs, pvals)):
        plt.text(i, r, f"{r:.2f}\n(p={p:.2g})", ha='center', va='bottom' if r>=0 else 'top', fontsize=8)
    
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.ylim(-0.8, 0.8)
    plt.show()

def boxplot_metrics(df_metrics: pd.DataFrame, metrics: list):
    """
    Create boxplots for specified metrics.
    
    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame containing motion metrics
    metrics : list
        List of metric names to plot
    """
    plt.figure(figsize=(12, 6))
    plt.suptitle("Boxplots of Selected Motion Metrics", fontsize=16)
    for metric in metrics:
        plt.subplot(2, 4, metrics.index(metric)+1)
        plt.boxplot(df_metrics[metric].dropna(), vert=True)
        plt.title(f"{metric}")
        plt.ylabel(metric)
    plt.tight_layout()
    plt.show()