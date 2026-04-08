import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class InvariantWindowDataset(Dataset):
    """
    Extracts translation/rotation invariant features from continuous hand trajectories.
    Channels: [Speed, Accel_Mag, Angular_Velocity, Palm_Area, Direction_Change]
    """
    def __init__(
        self,
        df_dict,
        hand="Right",
        window_size=45, # 1.5 sec @ 30fps
        step_size=15,   # 0.5 sec @ 30fps
        dt=1/30.0,
        device="cpu",
        scaling_stats=None
    ):
        self.hand = hand
        self.window_size = window_size
        self.step_size = step_size
        self.dt = dt
        self.device = device
        self.n_features = 5
        self.scaling_stats = scaling_stats
        
        self.windows = []
        self.metadata = [] # stores (video_id, start_frame)
        
        # Accumulator for robust scaling
        all_valid_frames = []

        for video_id, df in tqdm(df_dict.items(), desc=f"Processing {hand} Hand"):
            dfh = df[df["hand_label"] == hand].copy()
            if dfh.empty:
                continue
                
            # Iterate strictly over continuous segments
            for seg_id, df_seg in dfh.groupby('segment_id'):
                if len(df_seg) < self.window_size:
                    continue
                    
                # 1. Compute Invariant Signals for the segment
                signals = self._compute_invariant_signals(df_seg)
                
                # Stack features: (T, 5)
                feats = np.stack([
                    signals['speed'], 
                    signals['accel_mag'], 
                    # signals['ang_vel'], 
                    signals['palm_area'], 
                    signals['dir_change']
                ], axis=1)
                
                all_valid_frames.append(feats)
                
                # 2. Slice into windows
                T = len(feats)
                for start in range(0, T - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    self.windows.append(feats[start:end])
                    self.metadata.append((video_id, df_seg.iloc[start]['frame']))

        # Convert to numpy array for fast indexing: (N_windows, window_size, 5)
        self.windows = np.array(self.windows, dtype=np.float32)
        
        # 3. Robust Scaling (Median/IQR)
        if self.scaling_stats is None and len(all_valid_frames) > 0:
            concat_feats = np.concatenate(all_valid_frames, axis=0)
            self.scaling_stats = {
                "median": np.median(concat_feats, axis=0).astype(np.float32),
                "iqr": (np.percentile(concat_feats, 75, axis=0) - 
                        np.percentile(concat_feats, 25, axis=0) + 1e-6).astype(np.float32)
            }
            
        if self.scaling_stats is not None:
            med = self.scaling_stats["median"]
            iqr = self.scaling_stats["iqr"]
            # Broadcast over (N_windows, window_size, 5)
            self.windows = (self.windows - med) / iqr
            self.windows = np.clip(self.windows, -5.0, 5.0) # Clip outliers

    def _compute_invariant_signals(self, df_seg):
        # Extract raw coordinates
        p0 = df_seg[["lm_0_x_smooth", "lm_0_y_smooth"]].values
        p5 = df_seg[["lm_5_x_smooth", "lm_5_y_smooth"]].values
        p17 = df_seg[["lm_17_x_smooth", "lm_17_y_smooth"]].values
        T = len(p0)

        # 1. Velocity & Speed
        v = np.gradient(p0, axis=0) / self.dt
        speed = np.linalg.norm(v, axis=1)

        # 2. Acceleration Magnitude
        a = np.gradient(v, axis=0) / self.dt
        accel_mag = np.linalg.norm(a, axis=1)

        # 3. Palm Angular Velocity
        #u = p5 - p0
        #angles = np.arctan2(u[:, 1], u[:, 0])
        #angles_unwrapped = np.unwrap(angles)
        #ang_vel = np.gradient(angles_unwrapped) / self.dt

        # 4. Palm Area (Triangle 0-5-17)
        v1 = p5 - p0
        v2 = p17 - p0
        palm_area = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

        # 5. Directional Change (Cosine of angle between consecutive velocity vectors)
        v_shifted = np.roll(v, shift=1, axis=0)
        v_shifted[0] = v[0] # Fix boundary
        dot_product = np.sum(v * v_shifted, axis=1)
        norms = speed * np.linalg.norm(v_shifted, axis=1)
        # Avoid division by zero when stationary
        dir_change = np.zeros(T)
        valid = norms > 1e-6
        dir_change[valid] = dot_product[valid] / norms[valid]
        dir_change = np.arccos(np.clip(dir_change, -1.0, 1.0)) # Convert to radians

        return {
            'speed': speed.astype(np.float32),
            'accel_mag': accel_mag.astype(np.float32),
            #'ang_vel': ang_vel.astype(np.float32),
            'palm_area': palm_area.astype(np.float32),
            'dir_change': dir_change.astype(np.float32)
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        win = self.windows[idx]
        vid, frame = self.metadata[idx]
        return torch.tensor(win, device=self.device), vid