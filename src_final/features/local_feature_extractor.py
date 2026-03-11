import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import periodogram
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

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
                    
                    # Only calculate if we have some overlapping data
                    if len(df_seg_other) > (len(df_seg) * 0.1): 
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
                        seg_ids_window = signals_o['segment_id'][start:end]
                        # Check:
                        # 1) No NaNs
                        # 2) All same segment_id
                        if (not np.isnan(signals_o['vx'][start:end]).any()
                            and len(np.unique(seg_ids_window)) == 1
                        ):
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
            'ang_vel': ang_vel, 'palm_area': palm_area,
            'segment_id': df_seg['segment_id'].values
        }

    def _compute_all_metrics(self, win, win_other=None):
        feats = {}
        feats['total_path'] = np.sum(win['d'])
        feats['is_idle'] = 1.0 if (feats['total_path']/self.window_size) < (1 / 3) else 0.0

        # Feature Group 1: Translation & Workspace (Redundancy Reduced)
        feats.update(self._feat_translation(win))
        
        # Feature Group 2: Robust Smoothness (SPARC)
        feats.update(self._feat_fluidity(win))
        
        # Feature Group 3: Pose Stability
        feats.update(self._feat_pose(win))
        
        if win_other is not None:
            feats.update(self._feat_bimanual(win, win_other))
        
        return feats

    # --- Feature Modules ---
    def _feat_translation(self, win):
        """Focuses on Efficiency and Spatial Economy."""
        f = {}
        # Path Ratio (Efficiency)
        start_pt, end_pt = win['pts0'][0], win['pts0'][-1]
        euclidean = np.sqrt(np.sum((end_pt - start_pt)**2))
        path_ratio = np.sum(win['d']) / (euclidean + 1.0)
        f['path_ratio'] = np.log1p(path_ratio) if self.log_transform else path_ratio

        # Spatial Spread (Economy) - Kept over vel_mean due to lower redundancy
        pts = win['pts0']
        spatial_std = np.sqrt(np.std(pts[:, 0])**2 + np.std(pts[:, 1])**2)
        f['spatial_spread'] = np.log1p(spatial_std) if self.log_transform else spatial_std
        
        return f


    def _feat_fluidity(self, win):        
        """Replaces Jerk with Spectral Arc Length (SPARC). frequency domain smoothnes, works well even after processing smoothing"""
        v_mag = np.sqrt(win['vx']**2 + win['vy']**2)
        
        # 1. SPARC Calculation
        # Pad to next power of 2 for cleaner FFT
        n_fft = max(1024, len(v_mag))
        freqs, psd = periodogram(v_mag, fs=self.orig_fps, nfft=n_fft)
        
        # Normalize amplitude spectrum
        amp = np.sqrt(psd)
        amp /= np.max(amp + 1e-9)
        
        # Cutoff frequency (Surgical motion rarely exceeds 5Hz-10Hz)
        fc = 10.0
        mask = freqs <= fc
        freqs_c = freqs[mask]
        amp_c = amp[mask]
        
        # Calculate Arc Length of the spectrum
        # It's the length of the curve of the normalized amplitude spectrum
        d_freq = freqs_c[1] - freqs_c[0]
        d_amp = np.diff(amp_c)
        arc_length = np.sum(np.sqrt(d_freq**2 + d_amp**2))
        
        # SPARC is the negative arc length (Higher/Less Negative = Smoother)
        f = {'sparc': -arc_length}

        # 2. HF_Ratio (Use Tukey for Spectral Ratio)
        # Taper edges to prevent spectral leakage
        v_tapered = v_mag * tukey(len(v_mag), alpha=0.15)
        freqs, psd = periodogram(v_tapered, fs=self.orig_fps, nfft=1024)
        
        low_band = np.sum(psd[(freqs >= 0.5) & (freqs <= 3.0)]) # 0.5Hz to ignore DC/Offset
        high_band = np.sum(psd[(freqs > 3.0) & (freqs <= 10.0)])
        f['hf_ratio'] = high_band / (low_band + high_band + 1e-9)

        # 3. Speed Peaks (The "Sub-movement" count)
        # Prominence ensures we don't count tiny sensor noise as 'peaks'
        # For surgeons, a 'real' sub-movement usually has some significant amplitude
        peaks, _ = find_peaks(v_mag, prominence=np.std(v_mag) * 0.3)
        f['speed_peaks'] = float(len(peaks))

        # 1. Compute acceleration magnitude at each frame
        # win['ax'] and win['ay'] are already pre-computed in _compute_signals
        a_mag = np.sqrt(win['ax']**2 + win['ay']**2)
        
        # 2. Compute Root Mean Square (RMS)
        # This captures the 'energy' of the acceleration signal
        acc_rms = np.sqrt(np.mean(a_mag**2))
        
        # 3. Optional: Log-transform because acceleration energy 
        # often follows a long-tailed distribution
        if self.log_transform:
            f['acc_rms'] = np.log1p(acc_rms)
        else:
            f['acc_rms'] = acc_rms

        return f

    def _feat_pose(self, win):
        """Palm Area CV - robust 0th order feature."""
        area_mean = np.mean(win['palm_area'])
        cv = (np.std(win['palm_area']) / area_mean) if area_mean > 1e-6 else 0.0
        return {'palm_area_cv': cv}

    def _feat_acceleration_energy(self, win):
        """
        Computes the RMS of acceleration magnitude.
        Higher values indicate 'shaky' or 'staccato' motion (Novice).
        Lower values indicate smooth, constant-velocity motion (Expert).
        """
        # 1. Compute acceleration magnitude at each frame
        # win['ax'] and win['ay'] are already pre-computed in _compute_signals
        a_mag = np.sqrt(win['ax']**2 + win['ay']**2)
        
        # 2. Compute Root Mean Square (RMS)
        # This captures the 'energy' of the acceleration signal
        acc_rms = np.sqrt(np.mean(a_mag**2))
        
        # 3. Optional: Log-transform because acceleration energy 
        # often follows a long-tailed distribution
        if self.log_transform:
            return {'acc_rms': np.log1p(acc_rms)}
        
        return {'acc_rms': acc_rms}

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
        sum_vp = np.sum(vp_mag)
        sum_vo = np.sum(vo_mag)
        f['bimanual_ratio'] = sum_vp / (sum_vp + sum_vo + 1e-6)

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