import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

class WindowMetricDataset(Dataset):

    def __init__(
        self,
        df_dict,
        grs_scores,
        hand="Right",
        window_size=90,  # 3 seconds @ 30 FPS
        step_size=30,    # 1 second step
        orig_fps=30.0,
        device="cpu",
        scaling=True,
        scaling_stats=None,
        min_valid_frames_ratio=0.75, # Require at least 75% valid frames
    ):
        assert hand in ("Right", "Left")
        self.hand = hand
        self.window_size = window_size
        self.step_size = step_size
        self.orig_fps = float(orig_fps)
        self.device = device
        self.grs_scores = grs_scores
        # frame_step is 1 if orig_fps is 30.0
        self.frame_step = 1 if orig_fps == 30.0 else int(self.orig_fps / 10.0) 
        self.min_valid_frames_ratio = min_valid_frames_ratio
        self.min_valid_frames_count = int(window_size * min_valid_frames_ratio)
        self.dt = 1.0 / self.orig_fps

        # List to hold the 5-feature vector for every single valid window
        all_window_features = [] 
        self.index_map = []
        self.data_temp = {} # Temporary storage for raw kinematics

        # --- PASS 1: Calculate Kinematics and 5 Features for ALL windows ---
        with tqdm(total=len(df_dict), desc="Pass 1: Calculating Features") as pbar:
            for sample_key, df in df_dict.items():
                
                sample_name, surgeon_id = sample_key 
                if sample_name not in self.grs_scores:
                    pbar.update(1); continue

                video_grs = self.grs_scores[sample_name]
                dfh = df[df["hand_label"] == hand].copy()
                if dfh.empty:
                    pbar.update(1); continue

                # Calculate kinematics and mask arrays
                T, valid_acc, dx, dy, vx, vy, ax, ay = self._calculate_kinematics(dfh)
                
                # --- Store the required raw data (9 columns) ---
                raw_data = np.stack([dx, dy, vx, vy, ax, ay, valid_acc], axis=1) # (T, 7)
                self.data_temp[sample_key] = raw_data

                # --- Windowing and Feature Calculation ---
                for start in range(0, T - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    win = raw_data[start:end] 
                    
                    # Use valid_acc as the strictest filter for counting valid frames
                    valid_mask = win[:, 6] 
                    valid_count = valid_mask.sum()
                    
                    if valid_count >= self.min_valid_frames_count:
                        # Calculate the 5 features for this window
                        feats = self._compute_window_features(win)
                        all_window_features.append(feats)
                        
                        # Map stores the features directly now, not the raw array
                        self.index_map.append((sample_key, (start, end), feats))

                pbar.update(1)

        # --- PASS 2: SCALING & FINALIZE ---
        
        # Combine all features into one large array
        combined_feats_array = np.array(all_window_features, dtype=np.float32) # (N_windows, 5)

        if scaling and scaling_stats is None:
            print("Calculating scaling statistics for 5 output features...")
            mu = combined_feats_array.mean(axis=0)
            sigma = combined_feats_array.std(axis=0)
            sigma[sigma == 0] = 1e-6 # Avoid division by zero
            scaling_stats = {'mean': mu, 'std': sigma}
        
        self.scaling_stats = scaling_stats
        self.scaling = scaling

        if self.scaling and self.scaling_stats is not None:
            mu = torch.tensor(scaling_stats['mean'], dtype=torch.float32).to(self.device)
            sigma = torch.tensor(scaling_stats['std'], dtype=torch.float32).to(self.device)
            self.mu = mu.reshape(1, 5)
            self.sigma = sigma.reshape(1, 5)

        # --- Apply scaling to the stored features in index_map ---
        self.final_data = []
        for sample_key, frame_range, feats in self.index_map:
            feats_tensor = torch.tensor(feats, dtype=torch.float32).to(self.device).reshape(1, 5)
            if self.scaling:
                feats_tensor = (feats_tensor - self.mu) / self.sigma
            
            self.final_data.append({
                'features': feats_tensor,
                'key': sample_key,
                'frame_range': frame_range,
            })
        
        # Cleanup temporary raw data
        del self.data_temp
                
    def _calculate_kinematics(self, dfh):
        """Internal method to run kinematic and masking logic."""
        min_frame = dfh['frame'].min()
        max_frame = dfh['frame'].max()
        full_frame_index = pd.RangeIndex(start=min_frame, stop=max_frame + self.frame_step, step=self.frame_step)
        
        df_full = dfh.set_index('frame').reindex(full_frame_index).reset_index(names=['frame'])

        T = len(df_full)
        x_smooth = df_full["cx_smooth"].values.astype(np.float32)
        y_smooth = df_full["cy_smooth"].values.astype(np.float32)
        
        
        # Fill/Backfill and use smoothed data
        x_filled = pd.Series(x_smooth).ffill().bfill().fillna(0).values.astype(np.float32)
        y_filled = pd.Series(y_smooth).ffill().bfill().fillna(0).values.astype(np.float32)
        
        is_tracked = (~np.isnan(x_smooth)).astype(np.float32)
        
        # Derivative masks
        valid_vel = np.zeros(T, dtype=np.float32); valid_vel[1:] = is_tracked[1:] * is_tracked[:-1]
        valid_acc = np.zeros(T, dtype=np.float32); valid_acc[2:] = valid_vel[2:] * valid_vel[1:-1]
        
        # Derivatives
        dx = np.zeros(T, dtype=np.float32); dy = np.zeros(T, dtype=np.float32)
        dx[1:] = x_filled[1:] - x_filled[:-1]; dy[1:] = y_filled[1:] - y_filled[:-1]
        
        vx = dx / self.dt; vy = dy / self.dt

        dvx = np.zeros(T, dtype=np.float32); dvy = np.zeros(T, dtype=np.float32)
        dvx[1:] = vx[1:] - vx[:-1]; dvy[1:] = vy[1:] - vy[:-1]
        ax = dvx / self.dt; ay = dvy / self.dt
        
        # Apply strict masks
        dx = dx * valid_vel; dy = dy * valid_vel
        vx = vx * valid_vel; vy = vy * valid_vel
        ax = ax * valid_acc; ay = ay * valid_acc
        
        return T, valid_acc, dx, dy, vx, vy, ax, ay


    def _compute_window_features(self, win):
        """
        Calculates the 5 handcrafted features for a single window array (window_size, 7).
        Input array structure: [dx, dy, vx, vy, ax, ay, valid_acc]
        """
        dx, dy, vx, vy, ax, ay = win[:, 0], win[:, 1], win[:, 2], win[:, 3], win[:, 4], win[:, 5]
        valid_acc = win[:, 6]
        
        # 1. Path Length Ratio
        feat_path_ratio = self._compute_path_length_ratio(dx, dy)
        
        # 2. Dimensionless Squared Jerk
        feat_jerk = self._compute_dimensionless_squared_jerk(dx, dy, ax, ay, self.dt, valid_acc)
        
        # 3. Mean Curvature
        feat_curvature = self._compute_mean_curvature(dx, dy, vx, vy, ax, ay, valid_acc)
        
        # 4. Number of Reversals
        feat_reversals = self._compute_number_of_reversals(vx, vy)

        # 5. STD of Velocity Magnitude
        feat_std_vel = self._compute_std_of_velocity_magnitude(vx, vy, valid_acc)
        
        return np.array([
            feat_path_ratio, feat_jerk, feat_curvature, feat_reversals, feat_std_vel
        ], dtype=np.float32)


    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx):
        item = self.final_data[idx]
        # Features are already calculated, scaled, and converted to tensors in __init__
        return item['features'], item['key'], item['frame_range']
    
    def get_moments_stats(self):
        return self.scaling_stats

    # --- WINDOW METRIC CALCULATION INTERNAL METHODS (Simplified for clarity) ---
    
    @staticmethod
    def _compute_path_length_ratio(dx, dy):
        """1. Path Length Ratio (Tortuosity)"""
        tot_path = np.sum(np.sqrt(dx**2 + dy**2))
        eucl_dist = np.sqrt(np.sum(dx)**2 + np.sum(dy)**2)
        return tot_path / (eucl_dist + 1e-6) # Add epsilon to avoid ZeroDivision

    @staticmethod
    def _compute_dimensionless_squared_jerk(dx, dy, ax, ay, dt, valid_acc):
        """2. Dimensionless Squared Jerk (Normalized Integrated Squared Jerk)"""
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) < 3: return 0.0

        ax_v = ax[valid_indices]; ay_v = ay[valid_indices]
        T_window = len(dx) * dt
        
        tot_path = np.sum(np.sqrt(dx**2 + dy**2))
        if tot_path < 1e-6: return 0.0

        jerk_x = np.diff(ax_v, prepend=ax_v[0]) / dt
        jerk_y = np.diff(ay_v, prepend=ay_v[0]) / dt
        integral_squared_jerk = np.sum(jerk_x**2 + jerk_y**2) * dt

        return (T_window**5 * integral_squared_jerk) / tot_path**2

    @staticmethod
    def _compute_mean_curvature(dx, dy, vx, vy, ax, ay, valid_acc):
        """3. Mean Curvature"""
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) < 2: return 0.0

        vx_v = vx[valid_indices]; vy_v = vy[valid_indices]
        ax_v = ax[valid_indices]; ay_v = ay[valid_indices]
        
        v_mag_sq = vx_v**2 + vy_v**2
        cross_prod_mag = np.abs(vx_v * ay_v - vy_v * ax_v)
        
        curvature = cross_prod_mag / (v_mag_sq**1.5 + 1e-6)
        
        return np.mean(curvature)

    @staticmethod
    def _compute_number_of_reversals(vx, vy):
        """4. Number of Reversals (Zero-crossings in velocity profile)"""
        vel_mag = np.sqrt(vx**2 + vy**2)
        min_vel_threshold = 1.0 # Example: 1mm/s threshold for a 'stop'
        
        # Count rising edges where speed goes below threshold (a stop/pause)
        is_slow = (vel_mag < min_vel_threshold).astype(int)
        
        # We need to ensure we don't count a single long stop as multiple reversals
        # Diff finds changes: 1 is start of stop (0->1), -1 is end of stop (1->0)
        reversals = np.sum(np.diff(is_slow, prepend=0) == 1)
        
        return reversals

    @staticmethod
    def _compute_std_of_velocity_magnitude(vx, vy, valid_acc):
        """5. Standard Deviation of Velocity Magnitude"""
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) < 2: return 0.0

        vx_v = vx[valid_indices]; vy_v = vy[valid_indices]
        vel_mag_v = np.sqrt(vx_v**2 + vy_v**2)
        
        return np.std(vel_mag_v)












import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class MILBagDataset(Dataset):
    """
    MIL Dataset: Group ALL windows from one video into a single Bag.
    Item: (Bag of Windows (N, 5), GRS Score (1))
    """
    def __init__(
        self,
        df_dict,
        grs_scores,
        hand="Right",
        window_size=90,  # 3 seconds @ 30 FPS
        step_size=30,    # 1 second step
        orig_fps=30.0,
        device="cpu",
        scaling=True,
        scaler=None,
        clipping=True,
        min_valid_frames_ratio=0.75,
        log_transform=True,
    ):
        self.hand = hand
        self.window_size = window_size
        self.step_size = step_size
        self.orig_fps = float(orig_fps)
        self.device = device
        self.dt = 1.0 / self.orig_fps
        self.clipping = clipping
        self.log_transform = log_transform
        
        # Hyperparams for validity
        self.min_valid_frames_count = int(window_size * min_valid_frames_ratio)
        self.frame_step = 1 if orig_fps == 30.0 else int(self.orig_fps / 10.0)

        # Storage for Bags
        self.bags = []        # List of tensors: [(N1, 5), (N2, 5), ...]
        self.bag_labels = []  # List of scores: [Score1, Score2, ...]
        self.bag_ids = []     # List of keys: [('Video1', 'Surgeon1'), ...]

        all_window_features_flat = [] # Temp list for calculating global scaling stats

        # --- PASS 1: Generate Bags ---
        with tqdm(total=len(df_dict), desc="Generating MIL Bags") as pbar:
            for sample_key, df in df_dict.items():
                sample_name, surgeon_id = sample_key
                
                # Check for label existence
                if sample_name not in grs_scores:
                    pbar.update(1); continue
                
                label = float(grs_scores[sample_name])
                
                # Filter Hand
                dfh = df[df["hand_label"] == hand].copy()
                if dfh.empty:
                    pbar.update(1); continue

                # Calculate kinematics (Reuse your logic)
                T, valid_acc, dx, dy, vx, vy, ax, ay = self._calculate_kinematics(dfh)
                raw_data = np.stack([dx, dy, vx, vy, ax, ay, valid_acc], axis=1)

                # Collect windows for THIS bag
                bag_windows = []
                
                for start in range(0, T - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    win = raw_data[start:end]
                    
                    valid_mask = win[:, 6]
                    if valid_mask.sum() >= self.min_valid_frames_count:
                        # Calculate features (5,)
                        feats = self._compute_window_features(win)
                        bag_windows.append(feats)
                        all_window_features_flat.append(feats)

                # Only add the bag if it has at least one valid window
                if len(bag_windows) > 0:
                    # Convert list of arrays -> (N, 5) Array
                    bag_array = np.stack(bag_windows) 
                    self.bags.append(bag_array) 
                    self.bag_labels.append(label)
                    self.bag_ids.append(sample_key)

                pbar.update(1)

        # --- PASS 2: Scaling ---
        self.scaling = scaling
        self.scaler = scaler

        if scaling:
            if self.scaler is None:
                # Calculate global mean/std from ALL windows across ALL bags
                flat_data = np.array(all_window_features_flat, dtype=np.float32)
                scaler = RobustScaler()
                scaler.fit(flat_data)
                self.scaler = scaler  # Fit the scaler

                """OLD VERSION:
                mu = flat_data.mean(axis=0)
                sigma = flat_data.std(axis=0)
                sigma[sigma == 0] = 1e-6
                self.scaling_stats = {'mean': mu, 'std': sigma}
                """
                self.all_window_features_flat_scaled = self.scaler.transform(flat_data)
            
            # Store as torch tensors for easy application in __getitem__
            #self.mu = torch.tensor(self.scaling_stats['mean'], dtype=torch.float32).to(device)
            #self.sigma = torch.tensor(self.scaling_stats['std'], dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        """
        Returns:
            bag_feats (Tensor): (N, 5) Scaled features for all windows in the video
            label (Tensor): (1,) The GRS score
            bag_id (Tuple): (VideoName, SurgeonID)
        """
        # Retrieve the bag (N, 5) numpy array
        bag_np = self.bags[idx]
        label = self.bag_labels[idx]
        bag_id = self.bag_ids[idx]

        # Convert to tensor
        #bag_tensor = torch.tensor(bag_np, dtype=torch.float32).to(self.device)

        # Apply Scaling (Broadcasting: (N, 5) - (1, 5))
        if self.scaling:
            bag_tensor = torch.tensor(self.scaler.transform(bag_np), dtype=torch.float32).to(self.device)
        
        if self.clipping:
            bag_tensor = torch.clamp(bag_tensor, -10.0, 10.0)

        # Label to tensor
        label_tensor = torch.tensor([label], dtype=torch.float32).to(self.device)

        return bag_tensor, label_tensor, bag_id

    # ------------------------------------------------------------------
    # --- HELPER METHODS (Identical to your implementation) ---
    # ------------------------------------------------------------------
    
    def _calculate_kinematics(self, dfh):
        # ... (Copy-paste your exact _calculate_kinematics code here) ...
        # (Included for completeness in thought process, but omitted for brevity in final output)
        min_frame = dfh['frame'].min()
        max_frame = dfh['frame'].max()
        full_frame_index = pd.RangeIndex(start=min_frame, stop=max_frame + self.frame_step, step=self.frame_step)
        df_full = dfh.set_index('frame').reindex(full_frame_index).reset_index(names=['frame'])
        T = len(df_full)
        x_smooth = df_full["cx_smooth"].values.astype(np.float32)
        y_smooth = df_full["cy_smooth"].values.astype(np.float32)
        x_filled = pd.Series(x_smooth).ffill().bfill().fillna(0).values.astype(np.float32)
        y_filled = pd.Series(y_smooth).ffill().bfill().fillna(0).values.astype(np.float32)
        is_tracked = (~np.isnan(x_smooth)).astype(np.float32)
        valid_vel = np.zeros(T, dtype=np.float32); valid_vel[1:] = is_tracked[1:] * is_tracked[:-1]
        valid_acc = np.zeros(T, dtype=np.float32); valid_acc[2:] = valid_vel[2:] * valid_vel[1:-1]
        dx = np.zeros(T, dtype=np.float32); dy = np.zeros(T, dtype=np.float32)
        dx[1:] = x_filled[1:] - x_filled[:-1]; dy[1:] = y_filled[1:] - y_filled[:-1]
        vx = dx / self.dt; vy = dy / self.dt
        dvx = np.zeros(T, dtype=np.float32); dvy = np.zeros(T, dtype=np.float32)
        dvx[1:] = vx[1:] - vx[:-1]; dvy[1:] = vy[1:] - vy[:-1]
        ax = dvx / self.dt; ay = dvy / self.dt
        dx = dx * valid_vel; dy = dy * valid_vel
        vx = vx * valid_vel; vy = vy * valid_vel
        ax = ax * valid_acc; ay = ay * valid_acc
        return T, valid_acc, dx, dy, vx, vy, ax, ay

    def _compute_window_features(self, win):
        # ... (Copy-paste your exact _compute_window_features code here) ...
        dx, dy, vx, vy, ax, ay = win[:, 0], win[:, 1], win[:, 2], win[:, 3], win[:, 4], win[:, 5]
        valid_acc = win[:, 6]
        feat_path_ratio = self._compute_path_length_ratio(dx, dy, log_transform=self.log_transform)
        feat_jerk = self._compute_dimensionless_squared_jerk(dx, dy, ax, ay, self.dt, valid_acc, log_transform=self.log_transform)
        feat_curvature = self._compute_mean_curvature(dx, dy, vx, vy, ax, ay, valid_acc)
        feat_reversals = self._compute_reversals_windowed(vx, vy, valid_acc)
        feat_std_vel = self._compute_std_of_velocity_magnitude(vx, vy, valid_acc)

        tot_path = self._get_total_path(dx, dy)
        mean_vel, max_vel, min_vel = self._get_velocity_stats(vx, vy, valid_acc)
        return np.array([feat_path_ratio, feat_jerk, feat_curvature, feat_reversals, feat_std_vel, tot_path, mean_vel], dtype=np.float32)

    # ... (Include your static metric computation methods here) ...
    @staticmethod
    def _compute_path_length_ratio(dx, dy, log_transform=True):
        tot_path = np.sum(np.sqrt(dx**2 + dy**2))
        eucl_dist = np.sqrt(np.sum(dx)**2 + np.sum(dy)**2)
        ratio = tot_path / (eucl_dist + 1e-6)

        # apply log transform for outlier handling
        if log_transform:  
            ratio = np.log1p(ratio)
        return ratio

    @staticmethod
    def _compute_dimensionless_squared_jerk(dx, dy, ax, ay, dt, valid_acc, log_transform=True):
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) < 3: return 0.0

        ax_v = ax[valid_indices]; ay_v = ay[valid_indices]
        
        # 1. Use only valid displacement components for path length
        dx_v = dx[valid_indices]; dy_v = dy[valid_indices]
        tot_path = np.sum(np.sqrt(dx_v**2 + dy_v**2)) # Path only over valid segment
        if tot_path < 1e-6: return 0.0
        
        # 2. Use only the valid duration
        T_valid = len(valid_indices) * dt # Duration of the valid segment
        
        # 3. Jerk calculation (correctly uses ax_v, ay_v)
        jerk_x = np.diff(ax_v, prepend=ax_v[0]) / dt
        jerk_y = np.diff(ay_v, prepend=ay_v[0]) / dt
        integral_squared_jerk = np.sum(jerk_x**2 + jerk_y**2) * dt

        dimensionless_jerk = (T_valid**5 * integral_squared_jerk) / (tot_path**2 + 1e-6)

        # apply log transform for outlier handling
        if log_transform:  
            dimensionless_jerk = np.log1p(dimensionless_jerk)

        return dimensionless_jerk

    @staticmethod
    def _compute_mean_curvature(dx, dy, vx, vy, ax, ay, valid_acc, log_transform=True):
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) < 2: return 0.0
        vx_v = vx[valid_indices]; vy_v = vy[valid_indices]
        ax_v = ax[valid_indices]; ay_v = ay[valid_indices]
        v_mag_sq = vx_v**2 + vy_v**2
        cross_prod_mag = np.abs(vx_v * ay_v - vy_v * ax_v)
        curvature = cross_prod_mag / (v_mag_sq**1.5 + 1e-6)
        mean_curvature = np.mean(curvature)

        # apply log transform for outlier handling
        if log_transform:
            mean_curvature = np.log1p(mean_curvature)
        return mean_curvature

    @staticmethod
    def _compute_reversals_windowed(vx, vy, valid_acc, fps=30, window_sec=0.4):
        """
        Detects reversals by comparing the average velocity vector of a past window
        vs. a future window. Robust to high FPS and sensor noise.
        """
        valid_indices = np.where(valid_acc == 1)[0]
        
        # Extract valid velocities
        vx_v = vx[valid_indices]
        vy_v = vy[valid_indices]
        
        # Define window size in frames (e.g., 0.5s * 30fps = 15 frames)
        w = int(fps * window_sec)
        
        # We need to create two arrays to compare:
        # 1. Past Vectors: velocity at time t (or avg of t-w to t)
        # 2. Future Vectors: velocity at time t+w (or avg of t to t+w)
        
        # Simple Strided Approach (Compare t with t + window)
        # This checks: "Is the direction I am going NOW opposite to where I will be in 0.5s?"
        vec_now_x = vx_v[:-w]
        vec_now_y = vy_v[:-w]
        
        vec_future_x = vx_v[w:]
        vec_future_y = vy_v[w:]
        
        # Dot Product
        dot_product = (vec_now_x * vec_future_x) + (vec_now_y * vec_future_y)
        
        # Magnitudes
        mag_now = np.sqrt(vec_now_x**2 + vec_now_y**2)
        mag_fut = np.sqrt(vec_future_x**2 + vec_future_y**2)
        
        # Filter for significant movement
        # If the object is standing still in either window, it's not a reversal, it's a stop.
        min_speed = 10.0 # Adjust based on your unit scale
        is_moving = (mag_now > min_speed) & (mag_fut > min_speed)
        
        # Cosine Similarity
        # We only compute where moving to avoid div/0
        denom = mag_now * mag_fut
        cosine_sim = np.full_like(dot_product, 1.0) # Default 1.0 (parallel)
        
        np.divide(dot_product, denom, out=cosine_sim, where=(is_moving & (denom!=0)))
    
        # Threshold for Reversal
        # -0.707 approx 135 degrees. -0.5 is 120 degrees.
        # We use a loop/diff to count EVENTS, not frames. 
        # Because a U-turn might trigger "True" for 10 consecutive frames.
        is_reversing = (cosine_sim < -0.7).astype(int)
        
        # Count transitions from 0 to 1 (start of a reversal event)
        num_reversal_events = np.sum(np.diff(is_reversing, prepend=0) == 1)
        
        return float(num_reversal_events)

    @staticmethod
    def _compute_std_of_velocity_magnitude(vx, vy, valid_acc):
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) < 2: return 0.0
        vx_v = vx[valid_indices]; vy_v = vy[valid_indices]
        vel_mag_v = np.sqrt(vx_v**2 + vy_v**2)
        return np.std(vel_mag_v)
    
    @staticmethod
    def _get_total_path(dx, dy):
        return np.sum(np.sqrt(dx**2 + dy**2))
    
    @staticmethod
    def _get_velocity_stats(vx, vy, valid_acc):
        "mean, max, min"
        valid_indices = np.where(valid_acc == 1)[0]
        if len(valid_indices) == 0:
            return 0.0, 0.0, 0.0, 0.0
        vx_v = vx[valid_indices]; vy_v = vy[valid_indices]
        vel_mag_v = np.sqrt(vx_v**2 + vy_v**2)
        return (np.mean(vel_mag_v), np.max(vel_mag_v), np.min(vel_mag_v))