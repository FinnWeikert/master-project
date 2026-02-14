import numpy as np
import pandas as pd
from tqdm import tqdm

class WindowFeatureExtractor:
    def __init__(
        self,
        hand="Right",
        window_sec=1.5,
        step_sec=0.5,
        orig_fps=30.0,
        log_transform=False,
        include_bimanual=False,
        exclude_idle=False
    ):
        self.hand = hand
        self.other_hand = "Left" if hand == "Right" else "Right"
        self.orig_fps = float(orig_fps)
        self.dt = 1.0 / self.orig_fps
        self.window_size = int(window_sec * self.orig_fps)
        self.step_size = int(step_sec * self.orig_fps)
        self.log_transform = log_transform
        self.include_bimanual = include_bimanual
        self.exclude_idle=exclude_idle

    def extract_features(self, df_dict):
        all_window_rows = []

        for video_id, df in tqdm(df_dict.items(), desc=f"Extracting {self.hand} Features"):
            # 1. Separate Hands
            # Assuming df has a 'hand_label' column. 
            if 'hand_label' in df.columns:
                df_primary = df[df["hand_label"] == self.hand].copy()
                df_other = df[df["hand_label"] == self.other_hand].copy()
            else:
                # If structure is different, assume df is already primary hand
                df_primary = df.copy()
                df_other = pd.DataFrame() # Empty

            # Ensure segmentation exists
            if 'segment_id' not in df_primary.columns:
                df_primary['segment_id'] = 0

            # 2. Iterate segments of PRIMARY hand
            for seg_id, df_seg in df_primary.groupby('segment_id'):
                if len(df_seg) < self.window_size:
                    continue

                # 3. Compute Signals (Primary)
                signals_p = self._compute_signals(df_seg)
                
                # 4. Prepare Secondary Signals (for Bimanual)
                signals_o = None
                if self.include_bimanual and not df_other.empty:
                    # Get matching frames for the other hand
                    # We merge on 'frame' to ensure strict time alignment
                    common_frames = df_seg['frame'].values
                    df_seg_other = df_other[df_other['frame'].isin(common_frames)]
                    
                    # Only calculate if we have reasonably overlapping data
                    if len(df_seg_other) > (len(df_seg) * 0.8): 
                        # Note: We need to reindex to match Primary's exact length/order for vector math
                        df_seg_other = df_seg_other.set_index('frame').reindex(common_frames).reset_index()
                        signals_o = self._compute_signals(df_seg_other)

                # 5. Slide Window
                T = len(signals_p['vx'])
                for start in range(1, T - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    
                    # Slice Primary
                    win_p = {k: v[start:end] for k, v in signals_p.items()}
                    
                    # Slice Secondary (if available)
                    win_o = None
                    if signals_o is not None:
                        # Check if secondary data is valid (not NaN from reindex)
                        # We just check a key array like 'vx'
                        if not np.isnan(signals_o['vx'][start:end]).any():
                            win_o = {k: v[start:end] for k, v in signals_o.items()}

                    # 6. Compute Features
                    feats = self._compute_all_metrics(win_p, win_o)
                    
                    feats['video_id'] = video_id
                    feats['window_start_frame'] = df_seg.iloc[start]['frame']

                    if not self.exclude_idle or feats['is_idle'] == 0:
                        all_window_rows.append(feats)

        return pd.DataFrame(all_window_rows)

    def _compute_signals(self, df_seg):
        """ Compute raw time-series vectors from keypoints. """
        # Helper to get numpy arrays
        def get_pts(lm_prefix):
            cols = [f"{lm_prefix}_{ax}_smooth" for ax in ["x", "y"]]
            # Fill NaNs to avoid crashing, though segmenter should have handled this
            return df_seg[cols].values

        pts0 = get_pts("lm_0")   # Wrist
        pts5 = get_pts("lm_5")   # Index Base
        pts17 = get_pts("lm_17") # Pinky Base

        # Velocity & Accel
        v_vec = np.gradient(pts0, axis=0) / self.dt
        a_vec = np.gradient(v_vec, axis=0) / self.dt
        
        # Speed & Path
        # Path distance per frame
        d = np.sqrt(np.sum(np.diff(pts0, axis=0, prepend=pts0[:1])**2, axis=1))
        
        # Orientation (Wrist -> Index)
        v_orient = pts5 - pts0
        angles = np.arctan2(v_orient[:, 1], v_orient[:, 0])
        ang_vel = np.diff(np.unwrap(angles), prepend=angles[0]) / self.dt

        # Palm Area
        v1 = pts5 - pts0
        v2 = pts17 - pts0
        palm_area = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

        return {
            'pts0': pts0, 'd': d, 
            'vx': v_vec[:,0], 'vy': v_vec[:,1], 
            'ax': a_vec[:,0], 'ay': a_vec[:,1],
            'ang_vel': ang_vel, 'palm_area': palm_area
        }

    def _compute_all_metrics(self, win, win_other=None):
        """ Master feature computer that delegates to sub-methods. """
        feats = {}
        
        # Basic Metadata
        feats['total_path'] = np.sum(win['d'])
        
        # idle defined as less the 0.33 movement change per frame => total_path / num_frames (window_size) < 1/3
        feats['is_idle'] = 1.0 if (feats['total_path']/self.window_size) < (1 / 3) else 0.0

        # Feature Groups
        feats.update(self._feat_translation(win))
        feats.update(self._feat_smoothness(win, feats['total_path']))
        feats.update(self._feat_rotation(win))
        feats.update(self._feat_pose(win))
        
        # Bimanual (if enabled and data exists)
        if win_other is not None:
            feats.update(self._feat_bimanual(win, win_other))
        
        return feats

    # --- Feature Modules ---

    def _feat_translation(self, win):
        f = {}
        # Path Ratio
        start_pt, end_pt = win['pts0'][0], win['pts0'][-1]
        euclidean = np.sqrt(np.sum((end_pt - start_pt)**2))
        path_ratio = np.sum(win['d']) / (euclidean + 1)#1e-6
        f['path_ratio'] = np.log1p(path_ratio) if self.log_transform else path_ratio
        if path_ratio > 200:
            d=1

        # Curvature
        v_mag_sq = win['vx']**2 + win['vy']**2
        cross = np.abs(win['vx'] * win['ay'] - win['vy'] * win['ax'])
        # Only calculate curvature where velocity is sufficient (> 10 px/sec)
        valid_mask = v_mag_sq > 20.0 
        if np.sum(valid_mask) > 5:
            k = cross[valid_mask] / (v_mag_sq[valid_mask]**1.5)
            f['curvature'] = np.log1p(np.mean(k)) if self.log_transform else np.mean(k)
        else:
            f['curvature'] = 0.0 # Straight line assumption if stationary

        # Reversals
        #f['num_reversals'] = self._count_reversals(win['vx'], win['vy'])
        
        # Velocity Stats
        v_mag = np.sqrt(v_mag_sq)
        f['vel_mean'] = np.mean(v_mag)
        f['vel_p90'] = np.percentile(v_mag, 90)

        # Calculate the spatial dispersion (Work Envelope)
        # High value = Wandering/Drifting; Low value = Focused/Steady
        pts = win['pts0'] # (N, 2) array of wrist coordinates
        spatial_std = np.sqrt(np.std(pts[:, 0])**2 + np.std(pts[:, 1])**2)
        f['spatial_spread'] = np.log1p(spatial_std) if self.log_transform else spatial_std
        
        return f

    def _feat_smoothness(self, win, total_path):
        # Calculate Square Jerk
        jerk_x = np.diff(win['ax'], prepend=win['ax'][0]) / self.dt
        jerk_y = np.diff(win['ay'], prepend=win['ay'][0]) / self.dt
        int_sq_jerk = np.sum(jerk_x**2 + jerk_y**2) * self.dt
        
        # Duration of movement
        duration = len(win['d']) * self.dt
        
        # Peak Velocity (Robust normalization)
        v_mag = np.sqrt(win['vx']**2 + win['vy']**2)
        v_peak = np.max(v_mag)

        # Prevent divide by zero if hand is stationary
        if v_peak < 1e-6 or duration < 1e-6:
            return {'log_dim_jerk': -10.0} # Log of a small number

        # Standard Formula: (Integrated Jerk * Duration^3) / (PeakVel^2)
        # Note: Powers vary by paper, but this is a common unitless form
        dim_jerk = (int_sq_jerk * (duration**3)) / (v_peak**2)
        dim_jerk = -np.log1p(dim_jerk) if self.log_transform else dim_jerk

        return {'dim_jerk': dim_jerk} # Negative log often used so Higher = Smoother

    def _feat_rotation(self, win):
        return {
            'ang_vel_mean': np.mean(np.abs(win['ang_vel'])),
            'ang_vel_std': np.std(win['ang_vel'])
        }

    def _feat_pose(self, win):
        area_mean = np.mean(win['palm_area'])
        cv = (np.std(win['palm_area']) / area_mean) if area_mean > 1e-6 else 0.0
        return {'palm_area_cv': cv}

    def _feat_bimanual(self, win_p, win_o):
        """ 
        Calculates coordination between Primary (p) and Other (o) hand.
        """
        f = {}
        
        # 1. Inter-hand Distance (Proximity)
        # Are they working close together (Expert) or scared of collision (Novice)?
        dist_vec = win_p['pts0'] - win_o['pts0']
        dist = np.sqrt(np.sum(dist_vec**2, axis=1))
        f['bimanual_dist_mean'] = np.mean(dist)
        f['bimanual_dist_std'] = np.std(dist) # Fixed distance = good holding

        # 2. Velocity Correlation (Synchrony)
        # Do they start/stop together?
        vp_mag = np.sqrt(win_p['vx']**2 + win_p['vy']**2)
        vo_mag = np.sqrt(win_o['vx']**2 + win_o['vy']**2)
        
        # Pearson correlation of speed profiles
        if np.std(vp_mag) > 1e-6 and np.std(vo_mag) > 1e-6:
            f['bimanual_sync'] = np.corrcoef(vp_mag, vo_mag)[0,1]
        else:
            f['bimanual_sync'] = 0.0

        # 3. Dominance Ratio (Activity Balance)
        # 0.5 = Equal usage, 1.0 = Primary only, 0.0 = Other only
        sum_vp = np.sum(vp_mag)
        sum_vo = np.sum(vo_mag)
        f['bimanual_ratio'] = sum_vp / (sum_vp + sum_vo + 1e-6)
        f['bimanual_imbalance'] = np.abs(0.5 - f['bimanual_ratio']) 

        return f

    def _count_reversals(self, vx, vy):
        # (Same robust implementation as before)
        w = int(0.3 * self.orig_fps)
        if len(vx) <= w: return 0.0
        
        v_now_x, v_now_y = vx[:-w], vy[:-w]
        v_fut_x, v_fut_y = vx[w:], vy[w:]
        
        dot = v_now_x * v_fut_x + v_now_y * v_fut_y
        mag_now = np.sqrt(v_now_x**2 + v_now_y**2)
        mag_fut = np.sqrt(v_fut_x**2 + v_fut_y**2)
        
        is_moving = (mag_now > 15.0) & (mag_fut > 15.0)
        
        cosine = np.ones_like(dot)
        valid = is_moving & ((mag_now * mag_fut) > 1e-6)
        cosine[valid] = dot[valid] / (mag_now[valid] * mag_fut[valid])
        
        return float(np.sum(np.diff((cosine < -0.7).astype(int), prepend=0) == 1))