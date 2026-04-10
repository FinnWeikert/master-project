# Pytorch dataset for windows
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class WindowDataset(Dataset):
    """
    Windowed dataset for a single hand (Left or Right) using central differences.
    
    Expects dataframe columns:
        frame, hand_label, cx_smooth, cy_smooth

    Parameters:
    -----------
    df_dict : dict
        video_id -> dataframe

    hand : str
        "Right" or "Left"

    feature_mode : str
        "pos" → (x, y)
        "pos_vel" → (x, y, vx, vy)
        "pos_vel_acc" → (x, y, vx, vy, ax, ay)

    window_size : int
        number of timesteps per window

    step_size : int
        sliding step between windows

    orig_fps : float
        original video sampling rate (usually 30)

    max_gap_seconds : float
        maximum allowed time gap for derivative validity
        (recommended ~0.11 s → 3 original frames)

    normalize : bool
        per-video min-max normalization for x,y only

    device : torch device
    """

    def __init__(
        self,
        df_dict,
        hand="Right",
        feature_mode="pos_vel",
        window_size=20,
        step_size=5,
        orig_fps=30.0,
        max_gap_seconds=0.11,      # e.g. 3 frames @ 30 fps
        normalize=True,
        device="cpu",
        video_dimensions=(1920, 1080)
    ):
        assert hand in ("Right", "Left")
        assert feature_mode in ("pos", "pos_vel", "pos_vel_acc")

        self.hand = hand
        self.feature_mode = feature_mode
        self.window_size = window_size
        self.step_size = step_size
        self.orig_fps = float(orig_fps)
        self.max_gap_seconds = max_gap_seconds
        self.max_gap_frames = int(round(max_gap_seconds * orig_fps))
        self.normalize = normalize
        self.device = device
        self.video_width, self.video_height = video_dimensions

        self.data = {}
        self.index_map = []

        with tqdm(total=len(df_dict), desc="Processing videos") as pbar:
            for vid, df in df_dict.items():
                # --- Extract & sort single-hand data
                dfh = df[df["hand_label"] == hand].copy()
                dfh = dfh.sort_values("frame").reset_index(drop=True)
                if dfh.empty:
                    pbar.update(1)
                    continue

                frames = dfh["frame"].values.astype(int)
                x = dfh["cx_smooth"].values.astype(np.float32)
                y = dfh["cy_smooth"].values.astype(np.float32)
                T = len(frames)

                # --- Normalize positions
                if self.normalize:
                    x = x / self.video_width
                    y = y / self.video_height

                # --- Compute dt (forward & backward)
                dt_forward = np.zeros(T, dtype=np.float32)
                dt_backward = np.zeros(T, dtype=np.float32)

                dt_forward[:-1] = (frames[1:] - frames[:-1]) / self.orig_fps
                dt_forward[-1] = dt_forward[-2] if T > 1 else 1.0 / self.orig_fps

                dt_backward[1:] = (frames[1:] - frames[:-1]) / self.orig_fps
                dt_backward[0] = dt_backward[1] if T > 1 else 1.0 / self.orig_fps

                # Symmetric dt per frame
                dt = 0.5 * (dt_forward + dt_backward)

                # --- Validity mask (1 good, 0 gap)
                valid = np.ones(T, dtype=np.float32)
                for i in range(1, T):
                    if frames[i] - frames[i - 1] > self.max_gap_frames:
                        valid[i] = 0
                        valid[i - 1] = 0

                # --- Compute velocities with central difference
                vx = np.zeros(T, dtype=np.float32)
                vy = np.zeros(T, dtype=np.float32)

                for i in range(T):
                    if valid[i] == 0:
                        continue

                    if i == 0:
                        # forward difference
                        if dt_forward[i] <= self.max_gap_seconds:
                            vx[i] = (x[i+1] - x[i]) / dt_forward[i]
                            vy[i] = (y[i+1] - y[i]) / dt_forward[i]
                        else:
                            valid[i] = 0
                    elif i == T - 1:
                        # backward difference
                        if dt_backward[i] <= self.max_gap_seconds:
                            vx[i] = (x[i] - x[i-1]) / dt_backward[i]
                            vy[i] = (y[i] - y[i-1]) / dt_backward[i]
                        else:
                            valid[i] = 0
                    else:
                        gap_back = frames[i] - frames[i-1]
                        gap_fwd = frames[i+1] - frames[i]

                        if gap_back <= self.max_gap_frames and gap_fwd <= self.max_gap_frames:
                            dt_b = gap_back / self.orig_fps
                            dt_f = gap_fwd / self.orig_fps
                            denom = dt_b + dt_f
                            if denom > 1e-8:
                                vx[i] = (x[i+1] - x[i-1]) / denom
                                vy[i] = (y[i+1] - y[i-1]) / denom
                        else:
                            valid[i] = 0

                # --- Acceleration if requested
                if feature_mode == "pos_vel_acc":
                    ax = np.zeros(T, dtype=np.float32)
                    ay = np.zeros(T, dtype=np.float32)

                    for i in range(1, T-1):
                        if valid[i] == 0:
                            continue

                        gap_back = frames[i] - frames[i-1]
                        gap_fwd = frames[i+1] - frames[i]
                        if gap_back <= self.max_gap_frames and gap_fwd <= self.max_gap_frames:
                            dt_b = gap_back / self.orig_fps
                            dt_f = gap_fwd / self.orig_fps
                            denom = dt_b + dt_f
                            if denom > 1e-8:
                                ax[i] = 2.0 * ((x[i+1] - x[i]) / dt_f - (x[i] - x[i-1]) / dt_b) / denom
                                ay[i] = 2.0 * ((y[i+1] - y[i]) / dt_f - (y[i] - y[i-1]) / dt_b) / denom
                else:
                    ax = ay = None

                # --- Build feature matrix
                if feature_mode == "pos":
                    feats = np.stack([x, y], axis=1)
                elif feature_mode == "pos_vel":
                    feats = np.stack([x, y, vx, vy], axis=1)
                else:
                    feats = np.stack([x, y, vx, vy, ax, ay], axis=1)

                # Add valid channel, (!! Might want to add dt later too if sampling at 30 fps!!)
                feats = np.concatenate([
                    feats,
                    valid.reshape(-1, 1)
                ], axis=1)

                

                self.data[vid] = feats

                # --- Window indices
                for start in range(0, T - window_size + 1, step_size):
                    self.index_map.append((vid, start))

                pbar.update(1)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid, start = self.index_map[idx]
        arr = self.data[vid]
        win = arr[start : start + self.window_size]
        return torch.tensor(win, dtype=torch.float32, device=self.device), vid



class WindowDatasetTCN(Dataset):
    """
    Windowed dataset for a single hand (Left or Right) designed for Autoencoder training.
    
    Returns:
        X (Tensor): Shape (window_size, n_features). Normalized kinematics.
        meta (tuple): (video_id, start_frame, grs_score, surgeon_id)
    """

    def __init__(
        self,
        df_dict,
        hand="Right",
        feature_mode="pos_vel_acc",
        window_size=90,
        step_size=30,
        fps=10.0,
        orig_fps=30.0,
        device="cpu",
        scaling=True,
        scaling_stats=None,
        min_valid_frames_ratio=0.75,
    ):
        assert hand in ("Right", "Left")
        assert feature_mode in ("pos", "pos_vel", "pos_vel_acc")

        self.hand = hand
        self.window_size = window_size
        self.step_size = step_size
        self.fps = float(fps)
        self.device = device
        self.frame_step = orig_fps / fps 
        self.min_valid_frames_ratio = min_valid_frames_ratio
        self.min_valid_frames_count = int(window_size * min_valid_frames_ratio)
        self.dt = 1.0 / self.fps
        self.scaling = scaling
        self.scaling_stats = scaling_stats

        # Define how many features we take based on mode
        # dx, dy, vx, vy, ax, ay -> Indices 0 to 5
        if feature_mode == "pos":
            self.feat_indices = [0, 1] # dx, dy
        elif feature_mode == "pos_vel":
            self.feat_indices = [0, 1, 2, 3] # dx, dy, vx, vy
        else:
            self.feat_indices = [0, 1, 2, 3, 4, 5] # All
            
        self.n_features = len(self.feat_indices)

        # List to hold the index mapping and dictionary for raw data
        self.index_map = []
        self.data = {}
        
        # Temporary list to collect all valid frames for scaling calculation
        valid_frames_accumulator = []

        with tqdm(total=len(df_dict), desc=f"Processing {hand} Hand Windows") as pbar:
            for sample_key, df in df_dict.items():
                
                # Handle keys (sometimes they are tuples of (video, surgeon))
                if isinstance(sample_key, tuple):
                    sample_name, surgeon_id = sample_key
                else:
                    sample_name = sample_key
                    surgeon_id = "Unknown"
                
                # Filter by hand
                dfh = df[df["hand_label"] == hand].copy()
                if dfh.empty:
                    pbar.update(1); continue

                # 1. Calculate Kinematics (T, 6) and Mask (T, 1)
                T, valid_acc, dx, dy, vx, vy, ax, ay = self._calculate_kinematics(dfh)

                # Stack all potentially useful features first
                full_kinematics = np.stack([dx, dy, vx, vy, ax, ay], axis=1) # (T, 6)
                
                # 2. Select specific features based on mode
                selected_feats = full_kinematics[:, self.feat_indices] # (T, n_features)
                final_mask = valid_acc.reshape(-1, 1)  # (T, 1)
                feats = np.concatenate([
                    selected_feats,
                    final_mask
                ], axis=1)

                # Store in memory (temporarily unscaled)
                # We store the mask in the last channel for internal processing, 
                # but we will strip it out in __getitem__ usually.
                self.data[sample_key] = feats


                # 3. Accumulate valid data for scaling stats (if needed)
                if self.scaling and self.scaling_stats is None:
                    # Only take rows where mask is valid to avoid biasing mean to 0
                    valid_rows = selected_feats[final_mask.flatten() == 1.0]
                    if len(valid_rows) > 0:
                        valid_frames_accumulator.append(valid_rows)

                # 4. Windowing Logic
                # We verify validity here, but we don't store window data yet to save RAM
                for start in range(0, T - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    window_mask = final_mask[start:end].squeeze()
                    
                    if window_mask.sum() >= self.min_valid_frames_count:
                        self.index_map.append((sample_key, start))

                pbar.update(1)

        # --- SCALING LOGIC ---
        if self.scaling:

            # ---------------------------------------------------------
            # 1. Compute scaling stats ONCE (median / IQR)
            # ---------------------------------------------------------
            if self.scaling_stats is None:

                if len(valid_frames_accumulator) > 0:
                    print("Computing robust scaling statistics from training data...")

                    all_valid = np.concatenate(valid_frames_accumulator, axis=0)  # shape (N_valid, n_features)

                    # Median + IQR (robust to outliers)
                    median = np.median(all_valid, axis=0)
                    q1 = np.percentile(all_valid, 25, axis=0)
                    q3 = np.percentile(all_valid, 75, axis=0)
                    iqr = (q3 - q1) + 1e-6   # avoid division by zero

                    # Choose clipping range
                    clip_value = 5.0

                    self.scaling_stats = {
                        "median": median.astype(np.float32),
                        "iqr": iqr.astype(np.float32),
                        "q1": q1.astype(np.float32),
                        "q3": q3.astype(np.float32),
                        "clip": clip_value
                    }

                else:
                    print("Warning: No valid frames found for scaling → using identity transform.")
                    self.scaling_stats = {
                        "median": np.zeros(self.n_features, dtype=np.float32),
                        "iqr": np.ones(self.n_features, dtype=np.float32),
                        "clip": 10.0
                    }

            # ---------------------------------------------------------
            # 2. Apply robust scaling to stored data
            #    This modifies self.data IN PLACE for fast __getitem__
            # ---------------------------------------------------------
            med = self.scaling_stats["median"]
            iqr = self.scaling_stats["iqr"]
            clip = self.scaling_stats["clip"]

            for key in self.data:

                raw = self.data[key][:, :-1]     # features (no mask)
                mask = self.data[key][:, -1]     # mask

                # (X - median) / IQR
                scaled = (raw - med) / iqr

                # Clip extreme values (VERY important for stability)
                scaled = np.clip(scaled, -clip, clip)

                # Reapply mask — invalid frames must be exactly zero
                scaled = scaled * mask[:, None]

                # Reattach mask
                self.data[key] = np.concatenate([scaled, mask[:, None]], axis=1)


    def _calculate_kinematics(self, dfh):
        """Internal method to run kinematic and masking logic."""
        min_frame = dfh['frame'].min()
        max_frame = dfh['frame'].max()
        
        # Ensure strict frame index
        full_frame_index = pd.RangeIndex(start=min_frame, stop=max_frame + self.frame_step, step=self.frame_step)
        df_full = dfh.set_index('frame').reindex(full_frame_index).reset_index(names=['frame'])

        T = len(df_full)
        x_smooth = df_full["cx_smooth"].values.astype(np.float32)
        y_smooth = df_full["cy_smooth"].values.astype(np.float32)
        
        # Fill/Backfill for derivative calculation
        x_filled = pd.Series(x_smooth).ffill().bfill().fillna(0).values.astype(np.float32)
        y_filled = pd.Series(y_smooth).ffill().bfill().fillna(0).values.astype(np.float32)
        
        # Tracking mask
        is_tracked = (~np.isnan(x_smooth)).astype(np.float32)
        
        # Derivative validity masks (using valid pairs/triplets)
        valid_vel = np.zeros(T, dtype=np.float32)
        valid_vel[1:] = is_tracked[1:] * is_tracked[:-1]
        
        valid_acc = np.zeros(T, dtype=np.float32)
        valid_acc[2:] = valid_vel[2:] * valid_vel[1:-1]
        
        # 1. Displacement (dx, dy)
        dx = np.zeros(T, dtype=np.float32); dy = np.zeros(T, dtype=np.float32)
        dx[1:] = x_filled[1:] - x_filled[:-1]
        dy[1:] = y_filled[1:] - y_filled[:-1]
        
        # 2. Velocity
        vx = dx / self.dt; vy = dy / self.dt

        # 3. Acceleration
        dvx = np.zeros(T, dtype=np.float32); dvy = np.zeros(T, dtype=np.float32)
        dvx[1:] = vx[1:] - vx[:-1]; dvy[1:] = vy[1:] - vy[:-1]
        ax = dvx / self.dt; ay = dvy / self.dt
        
        # Apply strict masks (Zero out invalid derivatives)
        dx = dx * valid_vel; dy = dy * valid_vel
        vx = vx * valid_vel; vy = vy * valid_vel
        ax = ax * valid_acc; ay = ay * valid_acc
        
        return T, valid_acc, dx, dy, vx, vy, ax, ay
    
    def get_scaling_stats(self):
        """Returns the calculated or provided scaling stats."""
        return self.scaling_stats

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # Unpack the index map
        vid, start = self.index_map[idx]
        arr = self.data[vid]
        win = arr[start : start + self.window_size]
        return torch.tensor(win, dtype=torch.float32, device=self.device), vid
    
