import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class MotionFeatureDataset(Dataset):
    """
    Dataset producing windows of:
    d_x, d_y, v_x, v_y, a_x, a_y, valid_flag.
    
    Includes logic to invalidate frames immediately following tracking gaps 
    to prevent artificial 'jumps' in derivatives.
    """

    @classmethod
    def compute_scaling_stats(cls, df_dict, hand="Right", orig_fps=30.0, max_gap_seconds=0.11):
        """
        Calculates statistics. Uses strict masking to ensure we don't include 
        'reconnection jumps' in the statistics.
        """
        all_feats = []
        frame_step = int(orig_fps / 10.0) 

        for _, df in df_dict.items():
            dfh = df[df["hand_label"] == hand].copy()
            if dfh.empty: continue

            # --- 1. RECONSTRUCT FULL TIME AXIS ---
            min_frame = dfh['frame'].min()
            max_frame = dfh['frame'].max()
            full_frame_index = pd.RangeIndex(start=min_frame, stop=max_frame + frame_step, step=frame_step)
            
            df_full = dfh.set_index('frame').reindex(full_frame_index).reset_index(names=['frame'])

            # --- 2. EXTRACT TRACKING MASKS (BEFORE FILLING) ---
            T = len(df_full)
            x_raw = df_full["cx_smooth"].values.astype(np.float32)
            
            # Base mask: 1 if this specific frame was tracked, 0 if gap
            is_tracked = (~np.isnan(x_raw)).astype(np.float32)

            # --- 3. CREATE DERIVATIVE MASKS (THE FIX) ---
            # Velocity is valid only if current AND previous frame were tracked
            valid_vel = np.zeros(T, dtype=np.float32)
            valid_vel[1:] = is_tracked[1:] * is_tracked[:-1]

            # Acceleration is valid only if current AND previous velocity were valid
            # (effectively requires 3 sequential tracked frames)
            valid_acc = np.zeros(T, dtype=np.float32)
            valid_acc[1:] = valid_vel[1:] * valid_vel[:-1]

            # --- 4. FILL GAPS FOR CALCULATION SAFETY ---
            x_filled = df_full["cx_smooth"].ffill().fillna(0).values.astype(np.float32)
            y_filled = df_full["cy_smooth"].ffill().fillna(0).values.astype(np.float32)

            # --- 5. DERIVATIVE CALCULATION ---
            dt = np.full(T, 1.0 / orig_fps * frame_step)

            # Displacement
            dx = np.zeros(T, dtype=np.float32); dy = np.zeros(T, dtype=np.float32)
            dx[1:] = x_filled[1:] - x_filled[:-1]
            dy[1:] = y_filled[1:] - y_filled[:-1]
            
            # Velocity (Forward diff relative to index, but using dx[i])
            vx = np.zeros(T, dtype=np.float32); vy = np.zeros(T, dtype=np.float32)
            vx[:-1] = dx[1:] / dt[:-1]
            vy[:-1] = dy[1:] / dt[:-1]

            # Acceleration
            dvx = np.zeros(T, dtype=np.float32); dvy = np.zeros(T, dtype=np.float32)
            dvx[1:] = vx[1:] - vx[:-1]
            dvy[1:] = vy[1:] - vy[:-1]
            ax = np.zeros(T, dtype=np.float32); ay = np.zeros(T, dtype=np.float32)
            ax[:-1] = dvx[1:] / dt[:-1]
            ay[:-1] = dvy[1:] / dt[:-1]

            # --- 6. APPLY STRICT MASKS ---
            # Zero out derivatives where the transition was invalid (jumps)
            dx = dx * valid_vel; dy = dy * valid_vel
            vx = vx * valid_vel; vy = vy * valid_vel
            ax = ax * valid_acc; ay = ay * valid_acc
            
            # --- 7. FILTER FOR STATS ---
            # We use valid_acc as the strictest filter for stats to ensure clean data
            valid_indices = np.where(valid_acc == 1)[0]
            if len(valid_indices) > 0:
                feats = np.stack([dx, dy, vx, vy, ax, ay], axis=1)
                all_feats.append(feats[valid_indices])

        if not all_feats:
            raise ValueError("No valid data found to calculate statistics.")
            
        combined_feats = np.concatenate(all_feats, axis=0)
        return {'mean': combined_feats.mean(axis=0), 'std': combined_feats.std(axis=0)}


    def __init__(
        self,
        df_dict,
        grs_scores,
        hand="Right",
        window_size=100,
        step_size=25,
        orig_fps=30.0,
        max_gap_seconds=0.11,
        device="cpu",
        scaling_stats=None,
        # --- NEW PARAMETER ---
        min_valid_frames_ratio=0.5, # e.g., require at least 50% valid frames
    ):
        assert hand in ("Right", "Left")
        self.hand = hand
        self.window_size = window_size
        self.step_size = step_size
        self.orig_fps = float(orig_fps)
        self.max_gap_seconds = max_gap_seconds
        self.device = device
        self.grs_scores = grs_scores
        self.frame_step = int(self.orig_fps / 10.0) 
        self.min_valid_frames_ratio = min_valid_frames_ratio # New ratio
        self.min_valid_frames_count = int(window_size * min_valid_frames_ratio) # New count

        self.data = {}
        self.index_map = []
        
        self.scaling_stats = scaling_stats
        if scaling_stats is not None:
            mu = scaling_stats['mean']; sigma = scaling_stats['std']
            assert mu.shape == (6,) and sigma.shape == (6,)
            sigma[sigma == 0] = 1.0 
            self.mu = mu.reshape(1, 6)
            self.sigma = sigma.reshape(1, 6)

        with tqdm(total=len(df_dict), desc="Processing Videos") as pbar:
            for sample_key, df in df_dict.items():
                
                sample_name, surgeon_id = sample_key 
                if sample_name not in self.grs_scores:
                    pbar.update(1); continue

                video_grs = self.grs_scores[sample_name]
                dfh = df[df["hand_label"] == hand].copy().reset_index(drop=True)
                if dfh.empty:
                    pbar.update(1); continue

                # --- 1. RECONSTRUCT FULL TIME AXIS ---
                min_frame = dfh['frame'].min()
                max_frame = dfh['frame'].max()
                full_frame_index = pd.RangeIndex(start=min_frame, stop=max_frame + self.frame_step, step=self.frame_step)
                
                df_full = dfh.set_index('frame').reindex(full_frame_index).reset_index(names=['frame'])

                # --- 2. EXTRACT TRACKING MASKS ---
                T = len(df_full)
                x_raw = df_full["cx_smooth"].values.astype(np.float32)
                
                # Base Mask: 1 if tracked, 0 if gap
                is_tracked = (~np.isnan(x_raw)).astype(np.float32)

                # Velocity Mask: Valid only if i AND i-1 were tracked. 
                valid_vel = np.zeros(T, dtype=np.float32)
                valid_vel[1:] = is_tracked[1:] * is_tracked[:-1]

                # Acceleration Mask: Valid only if i AND i-1 had valid velocities.
                valid_acc = np.zeros(T, dtype=np.float32)
                valid_acc[1:] = valid_vel[1:] * valid_vel[:-1]

                # --- 3. FILL & CALCULATE --- (Calculations remain identical)
                x_filled = df_full["cx_smooth"].ffill().fillna(0).values.astype(np.float32)
                y_filled = df_full["cy_smooth"].ffill().fillna(0).values.astype(np.float32)
                
                dt = np.full(T, 1.0 / self.orig_fps * self.frame_step) 

                # Displacement
                dx = np.zeros(T, dtype=np.float32); dy = np.zeros(T, dtype=np.float32)
                dx[1:] = x_filled[1:] - x_filled[:-1]
                dy[1:] = y_filled[1:] - y_filled[:-1]

                # Velocity
                vx = np.zeros(T, dtype=np.float32); vy = np.zeros(T, dtype=np.float32)
                vx[:-1] = dx[1:] / dt[:-1]
                vy[:-1] = dy[1:] / dt[:-1]
                
                # Acceleration
                dvx = np.zeros(T, dtype=np.float32); dvy = np.zeros(T, dtype=np.float32)
                dvx[1:] = vx[1:] - vx[:-1]
                dvy[1:] = vy[1:] - vy[:-1]
                ax = np.zeros(T, dtype=np.float32); ay = np.zeros(T, dtype=np.float32)
                ax[:-1] = dvx[1:] / dt[:-1]
                ay[:-1] = dvy[1:] / dt[:-1]

                # --- 4. APPLY STRICT MASKS --- (Masking remains identical)
                # Apply velocity mask to d, v
                dx = dx * valid_vel; dy = dy * valid_vel
                vx = vx * valid_vel; vy = vy * valid_vel
                
                # Apply acceleration mask to a
                ax = ax * valid_acc; ay = ay * valid_acc
                
                # --- 5. FINALIZE FEATURES ---
                kinematic_feats = np.stack([dx, dy, vx, vy, ax, ay], axis=1) # (T, 6)

                if self.scaling_stats is not None:
                    kinematic_feats = (kinematic_feats - self.mu) / self.sigma
                
                # valid_vel is used as the final mask channel (1 if valid transition, 0 otherwise)
                final_mask = valid_vel.reshape(-1, 1) 
                feats = np.concatenate([
                    kinematic_feats,
                    final_mask
                ], axis=1)

                self.data[sample_key] = feats

                # --- 6. WINDOWING WITH VALIDITY CHECK (NEW LOGIC) ---
                for start in range(0, T - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    
                    # Extract the mask channel for the current window
                    window_mask = final_mask[start:end].squeeze() # (window_size,)
                    
                    # Count the number of valid frames (where mask == 1)
                    valid_count = window_mask.sum()
                    
                    # Only map the window if it meets the minimum threshold
                    if valid_count >= self.min_valid_frames_count:
                        self.index_map.append((sample_key, start, video_grs, surgeon_id))

                pbar.update(1)
                
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        sample_key, start, grs_score, surgeon_id = self.index_map[idx] 
        arr = self.data[sample_key]
        win = arr[start : start + self.window_size]
        
        # Transpose to (Channels, Time) for TCN
        window_tensor = torch.tensor(win.T, dtype=torch.float32, device=self.device)
        score_tensor = torch.tensor(grs_score, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return window_tensor, score_tensor, sample_key