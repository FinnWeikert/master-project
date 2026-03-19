import numpy as np

def aggregate_window_features(df_windows, p=90, active_with_idle=True):
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
        idle_prop='mean'      # % of time spent waiting
    )

    # --- 3. Active Kinematics (The "Skill" Signal) ---
    # Filter for active windows only
    if not active_with_idle:
        df_active = df_windows[df_windows['is_idle'] == 0].copy()
    else:
        df_active = df_windows.copy()
    
    
    # Define aggregation strategies based on feature type
    # A. Magnitude Features (Where the 'Average' matters)
    #    Velocity, Spacing, Ang Velocity
    possible_feats = [
        'zvr', 'vel_p90', 'spatial_spread', 
        'jerk', 'curvature', 'path_ratio', 
        'sparc', 'palm_area_cv',
    ]
    
    # Filter lists to ensure columns actually exist
    feats = [c for c in possible_feats if c in df_active.columns]

    # helper for clean names
    def make_percentile_funcs(p):
        def high(x):
            return np.percentile(x, p)
        def low(x):
            return np.percentile(x, 100-p)

        high.__name__ = f"p{p}"
        low.__name__  = f"p{100-p}"

        return high, low


    p_high, p_low = make_percentile_funcs(p)

    
    aggs = {}
    
    for f in feats:
        aggs[f] = ['median', 'std', p_high, p_low]

    df_kinematics = df_active.groupby('video_id').agg(aggs)
    
    # --- 4. Flatten MultiIndex Columns Cleanly ---
    # Result: vel_mean_median, jerk_max, etc.
    df_kinematics.columns = [f"{c[0]}_{c[1]}" for c in df_kinematics.columns]

    # --- 5. Merge ---
    df_final = df_economy.join(df_kinematics, how='left')

    bimanual_feats = ['bimanual_dist_mean', 'bimanual_dist_std', 'bimanual_sync', 'bimanual_imbalance']

    if all(f in df_windows.columns for f in bimanual_feats):
        df_bimanual = df_windows[df_windows['bimanual_dist_mean'].notna() &
                                 df_windows['bimanual_dist_std'].notna() & 
                                 df_windows['bimanual_sync'].notna() & 
                                 df_windows['bimanual_imbalance'].notna()][['video_id']+bimanual_feats]
        
        aggs = {}
    
        for f in bimanual_feats:
            aggs[f] = ['median', 'std', p_high, p_low]

        df_bimanual_agg = df_bimanual.groupby('video_id').agg(aggs)
        df_bimanual_agg.columns = [f"{c[0]}_{c[1]}" for c in df_bimanual_agg.columns]
        df_final = df_final.join(df_bimanual_agg, how='left')
    
    return df_final.reset_index()
