import numpy as np

def aggregate_window_features(df_windows, p=90):
    """
    Aggregates window-level features into video-level descriptors, 
    separating 'Decision Time' (Idle) from 'Execution Quality' (Active).
    """
    
    # --- 1. Pre-Aggregation Feature Engineering ---
    # Convert raw Bimanual Ratio (0-1) into "Hand Bias" (0.0=Balanced, 0.5=One-handed)
    # We do this here because the variance of the raw ratio is ambiguous.
    if 'bimanual_ratio' in df_windows.columns:
        df_windows['bimanual_imbalance'] = (df_windows['bimanual_ratio'] - 0.5).abs()

    # --- 2. Global Economy (The "Hesitation" Signal) ---
    # We use the whole dataframe (Idle + Active) for this
    df_economy = df_windows.groupby('video_id')['is_idle'].agg(
        idle_prop='mean',       # % of time spent waiting
        idle_switch_count='sum' # How many times did they stop and start? (Hesitancy)
    )

    # --- 3. Active Kinematics (The "Skill" Signal) ---
    # Filter for active windows only
    df_active = df_windows[df_windows['is_idle'] == 0].copy()
    
    # Define aggregation strategies based on feature type
    # A. Magnitude Features (Where the 'Average' matters)
    #    Velocity, Spacing, Ang Velocity
    feats_magnitude = [
        'vel_mean', 'vel_p90', 'spatial_spread', 
        'ang_vel_mean', 'bimanual_dist_mean', 'bimanual_dist_std', 'bimanual_sync'
    ]
    
    # B. Error Features (Where the 'Worst Case' matters)
    #    Jerk, Reversals, Path Ratio, Curvature -> We want to catch the spikes
    feats_error = [
        'jerk', 'curvature', 'path_ratio', 'num_reversals', 'palm_area_cv', 'bimanual_imbalance'
    ]
    
    # Filter lists to ensure columns actually exist
    feats_magnitude = [c for c in feats_magnitude if c in df_active.columns]
    feats_error = [c for c in feats_error if c in df_active.columns]

    # helper for clean names
    def p90(x): return np.percentile(x, p)
    def p10(x): return np.percentile(x, 100-p)
    
    aggs = {}
    
    # For Magnitudes: Median (Robust Average), Std (Consistency), P90 (Peak Capacity)
    for f in feats_magnitude:
        aggs[f] = ['median', 'std', p90]

    # For Errors: Median (Typical), P90 (High Error), Max (Catastrophic Error)
    for f in feats_error:
        aggs[f] = ['median', 'std', p90, 'max']

    df_kinematics = df_active.groupby('video_id').agg(aggs)
    
    # --- 4. Flatten MultiIndex Columns Cleanly ---
    # Result: vel_mean_median, jerk_max, etc.
    df_kinematics.columns = [f"{c[0]}_{c[1]}" for c in df_kinematics.columns]

    # --- 5. Merge ---
    df_final = df_economy.join(df_kinematics, how='left')
    
    # Handle Missing Data (Rare case: Surgeon was NEVER active)
    # Filling with 0 is okay for 'jerk' (smooth), but bad for 'path_ratio' (efficiency).
    # A safer bet is filling with the median of the entire dataset, 
    # but for Ridge Regression, 0 (if standardized later) is acceptable.
    df_final = df_final.fillna(0)
    
    return df_final.reset_index()
