import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
from scipy.spatial import ConvexHull, QhullError

class SurgicalFeatureExtractor:
    """
    A unified class to extract kinematic, rotational, and bimanual features 
    from surgical hand tracking data.
    
    Usage:
        extractor = SurgicalFeatureExtractor(df_tracking, fps=30)
        features_df = extractor.features_df
    """
    
    def __init__(self, df_tracking: pd.DataFrame, fps: int = 30, 
                 min_disp: float = 1.0, vel_threshold: float = 30.0):
        
        self.fps = fps
        self.min_disp = min_disp
        self.vel_thresh = vel_threshold
        
        # 1. Preprocess Data (Compute Derivatives once)
        # We process Left and Right separately and store them as attributes
        self.df_left = self._preprocess_hand(df_tracking, "Left")
        self.df_right = self._preprocess_hand(df_tracking, "Right")
        
        # 2. Compute Features
        results = {}
        
        # Single Hand Features
        results.update(self._compute_hand_metrics(self.df_left, "Left"))
        results.update(self._compute_hand_metrics(self.df_right, "Right"))
        
        # Bimanual Features
        results.update(self._compute_bimanual_metrics())
        
        # 3. Store as DataFrame
        self.features_df = pd.DataFrame([results])

    def _preprocess_hand(self, df_raw, label):
        """
        Filters data for a specific hand, sorts it, and pre-calculates 
        kinematics (velocity, accel, jerk, angles) handling segment gaps.
        """
        df = df_raw[df_raw['hand_label'] == label].copy().sort_values("frame")
        
        if df.empty:
            return df
        
        # --- Temporal derivatives ---
        df["frame_diff"] = df["frame"].diff().fillna(1)
        df["dt"] = df["frame_diff"] / self.fps
        
        # --- Spatial Displacement (respecting segments) ---
        # We calculate displacement only within the same segment_id
        df["disp"] = np.nan
        df["disp_filtered"] = 0.0
        
        for _, seg in df.groupby("segment_id", sort=False):
            dx = np.diff(seg["cx_smooth"], prepend=np.nan)
            dy = np.diff(seg["cy_smooth"], prepend=np.nan)

            # Euclidean distance
            d = np.sqrt(dx**2 + dy**2)
            
            # Filter noise
            d_filtered = np.where(np.isnan(d), np.nan, 
                                  np.where(d > self.min_disp, d, 0.0))
            
            df.loc[seg.index, "disp"] = d
            df.loc[seg.index, "disp_filtered"] = d_filtered

            angles = np.arctan2(dy, dx)
            df.loc[seg.index, "disp_angle"] = angles

        # --- Kinematics ---
        # Replace 0 dt with NaN to avoid Inf
        valid_dt = df["dt"].replace(0, np.nan)
        df["velocity"] = df["disp_filtered"] / valid_dt
        df["accel"] = df["velocity"].diff() / valid_dt
        df["jerk"] = df["accel"].diff() / valid_dt
        
        # --- Rotational Geometry (for Rotational Metrics) ---
        # Vectors: Wrist(0) -> Index(5) and Wrist(0) -> Pinky(17)
        if {'lm_0_x_smooth', 'lm_5_x_smooth', 'lm_17_x_smooth'}.issubset(df.columns):
            v1_x = df['lm_5_x_smooth'] - df['lm_0_x_smooth']
            v1_y = df['lm_5_y_smooth'] - df['lm_0_y_smooth']
            v2_x = df['lm_17_x_smooth'] - df['lm_0_x_smooth']
            v2_y = df['lm_17_y_smooth'] - df['lm_0_y_smooth']
            
            # Hand angle (2D orientation)
            df["hand_angle"] = np.arctan2(v1_y, v1_x)
            # Cross product (Area/Pose stability)
            df["cross_product"] = (v1_x * v2_y) - (v1_y * v2_x)
        else:
            df["hand_angle"] = np.nan
            df["cross_product"] = np.nan

        return df

    def _compute_hand_metrics(self, df, label):
        """Computes both Efficiency/Smoothness and Rotational metrics."""
        if df.empty or len(df) < 2:
            return {f"total_path_{label}": 0.0, f"fraction_tracked_{label}": 0.0}
        
        # --- 1. Basic Efficiency ---
        total_path = df["disp_filtered"].sum()
        duration = df["dt"].sum()
        mean_velocity = df["velocity"].mean()
        rms_accel = np.sqrt((df["accel"]**2).mean())
        efficiency = total_path / duration if duration > 0 else 0
        
        # --- 2. Smoothness (LDLJ & NMU) ---
        # NMU
        vel_clean = df["velocity"].fillna(0)
        peaks, _ = signal.find_peaks(vel_clean, prominence=self.vel_thresh)
        nmu = len(peaks)
        
        # LDLJ
        jerk_sq_sum = (df["jerk"]**2).sum() * (1.0/self.fps)
        if total_path > 0 and duration > 0:
            ldlj = -np.log((duration**3 / total_path**2) * jerk_sq_sum + 1e-6)
        else:
            ldlj = np.nan

        # --- 3. Path Curvature ---
        angles = df["disp_angle"].values[1:]
        # Unwrap and get diff
        angle_change = np.diff(angles)
        angle_change = np.mod(angle_change + np.pi, 2*np.pi) - np.pi # Wrap -pi to pi
        
        # Only count turns when movement is significant
        move_mask = df["velocity"].values[1:] > self.vel_thresh
        # Align masks (diff reduces length by 1)
        valid_changes = angle_change[move_mask[:-1]]
        
        if len(valid_changes) > 0:
            mean_abs_angle = np.mean(np.abs(valid_changes)[~np.isnan(valid_changes)])
            #num_reversals = np.sum(np.diff(np.sign(valid_changes)) != 0)

            signs = np.sign(angle_change)

            valid = move_mask[:-1] & move_mask[1:]

            filtered_signs = signs.copy()
            filtered_signs[~valid] = 0  # or np.nan

            # Now compute diff only where consecutive valid
            reversal_mask = (filtered_signs[1:] != filtered_signs[:-1]) & valid[1:]

            num_reversals = np.sum(reversal_mask)

        else:
            mean_abs_angle = 0
            num_reversals = 0

        # --- 4. Rotational Metrics ---
        if "hand_angle" in df.columns and not df["hand_angle"].isna().all():

            # Entropy
            hist_counts, _ = np.histogram(df["hand_angle"].dropna(), bins=36, 
                                          range=(-np.pi, np.pi), density=True)
            hist_counts = hist_counts[hist_counts > 0]
            entropy = -np.sum(hist_counts * np.log(hist_counts))

            # only keep rotational shift if disp is not nan
            df.loc[df["disp_filtered"].isna(), ["hand_angle"]] = np.nan
            angle_diffs = np.diff(df["hand_angle"])
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

            total_angular_path = np.sum(np.abs(angle_diffs[~np.isnan(angle_diffs)]))
            
            # Pose Variability
            pose_var = (np.nanstd(df["cross_product"]) / 
                        np.nanmean(np.abs(df["cross_product"]))) if len(df) > 0 else 0
            
            rot_over_dist = total_angular_path / (total_path + 1e-6)
        else:
            total_angular_path = entropy = pose_var = rot_over_dist = np.nan

        # --- 5. Tracking Quality ---
        frame_span = df["frame"].max() - df["frame"].min()
        frac_tracked = (len(df) / frame_span) if frame_span > 0 else 1.0

        label = 'L' if label == 'Left' else 'R'
        return {
            f"Total path ({label})": total_path,
            f"Mean vel. ({label})": mean_velocity,
            f"Rms accel. ({label})": rms_accel,
            f"LDLJ smoothness ({label})": ldlj,
            f"Nmu peaks ({label})": nmu,
            f"Path rate ({label})": efficiency,
            f"Mean ang. change ({label})": mean_abs_angle,
            f"Angular switches ({label})": num_reversals,
            f"Total ang. path ({label})": total_angular_path,
            f"Orientation entropy ({label})": entropy,
            f"Pose variability ({label})": pose_var,
            f"Rotation over distance ({label})": rot_over_dist,
            f"Fraction tracked ({label})": frac_tracked
        }

    def _compute_bimanual_metrics(self):
        """Computes coordination metrics using the already-processed hand data."""
        # Merge on Frame to align timelines
        merged = pd.merge(
            self.df_left[["frame", "cx_smooth", "cy_smooth", "velocity", "disp_filtered"]],
            self.df_right[["frame", "cx_smooth", "cy_smooth", "velocity", "disp_filtered"]],
            on="frame", how="inner", suffixes=("_L", "_R")
        )
        
        if len(merged) < 10:
            return {
                "velocity_corr": np.nan, "velocity_corr_moving": np.nan,
                "interhand_dist_mean": np.nan, "movement_overlap_ratio": np.nan
            }

        # 1. Velocity Correlation
        if merged["velocity_L"].std() > 1e-6 and merged["velocity_R"].std() > 1e-6:
            vel_corr = merged["velocity_L"].corr(merged["velocity_R"])
        else:
            vel_corr = 0.0
            
        # 2. movement overlapp (Moving only)
        moving_mask = (merged["velocity_L"] > self.vel_thresh) & \
                      (merged["velocity_R"] > self.vel_thresh)
            
        # 3. Spatial Coordination (Inter-hand Distance)
        dists = np.sqrt((merged["cx_smooth_L"] - merged["cx_smooth_R"])**2 + 
                        (merged["cy_smooth_L"] - merged["cy_smooth_R"])**2)

        # RMS of distance change (stability of spacing)
        interhand_dist_change_rms = np.sqrt(np.mean(np.diff(dists)**2))
        
        # 4. Activity Overlap
        overlap = moving_mask.mean()
        
        return {
            "Velocity corr.": vel_corr,
            "Interhand dist. change RMS": interhand_dist_change_rms,
            "Movement overlap ratio": overlap,
            "Velocity ratio": merged["velocity_L"].mean() / (merged["velocity_R"].mean() + 1e-6)
        }
    
    def _count_reversals_sofisticated(self, df):
        """
        Calculates directional reversals within segments to avoid artifacts 
        at video cuts.
        """
        w = int(0.3 * self.fps) # 300ms window
        total_reversals = 0
        
        # We must iterate by segment to avoid comparing the last frame 
        # of one clip to the first frame of a new clip.
        for _, seg in df.groupby("segment_id"):
            if len(seg) <= w:
                continue
                
            # Get velocities (px/frame is fine for cosine)
            vx = np.diff(seg["cx_smooth"].values)
            vy = np.diff(seg["cy_smooth"].values)
            
            # Align lengths with vx/vy
            v_now_x, v_now_y = vx[:-w], vy[:-w]
            v_fut_x, v_fut_y = vx[w:], vy[w:]
            
            dot = v_now_x * v_fut_x + v_now_y * v_fut_y
            mag_now = np.sqrt(v_now_x**2 + v_now_y**2)
            mag_fut = np.sqrt(v_fut_x**2 + v_fut_y**2)
            
            # Filter: Using your 15.0 px/sec threshold
            # Note: if vx/vy are px/frame, multiply by fps to use px/sec
            is_moving = ((mag_now * self.fps) > 15.0) & ((mag_fut * self.fps) > 15.0)
            
            # Compute Cosine
            denom = mag_now * mag_fut
            # Initialize with 1.0 (no change)
            cosine = np.ones_like(dot)
            mask = is_moving & (denom > 1e-6)
            cosine[mask] = dot[mask] / denom[mask]
            
            # Threshold and Count events
            is_reversing = (cosine < -0.7).astype(int)
            total_reversals += np.sum(np.diff(is_reversing, prepend=0) == 1)
            
        return total_reversals
    