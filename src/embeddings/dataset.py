# Pytorch dataset for windows
import numpy as np
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
        max_gap_seconds=0.11,
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
        self.max_gap_frames = int(round(max_gap_seconds * orig_fps))
        self.normalize = normalize
        self.device = device
        self.video_width, self.video_height = video_dimensions

        self.data = {}
        self.index_map = []

        with tqdm(total=len(df_dict), desc="Processing videos") as pbar:
            for vid, df in df_dict.items():
                dfh = df[df["hand_label"] == hand].copy()
                dfh = dfh.sort_values("frame").reset_index(drop=True)
                if dfh.empty:
                    continue

                frames = dfh["frame"].values.astype(int)
                x = dfh["cx_smooth"].values.astype(np.float32)
                y = dfh["cy_smooth"].values.astype(np.float32)
                T = len(frames)

                # Normalize coordinates
                if self.normalize:
                        x = x / self.video_width
                        y = y / self.video_height

                # Precompute dt
                dt_forward = np.zeros(T, dtype=np.float32)
                dt_backward = np.zeros(T, dtype=np.float32)

                dt_forward[:-1] = (frames[1:] - frames[:-1]) / self.orig_fps
                dt_forward[-1] = dt_forward[-2] if T > 1 else 1.0 / self.orig_fps

                dt_backward[1:] = (frames[1:] - frames[:-1]) / self.orig_fps
                dt_backward[0] = dt_backward[1] if T > 1 else 1.0 / self.orig_fps

                # -------------------------
                # Compute central difference velocity
                # -------------------------
                vx = np.zeros(T, dtype=np.float32)
                vy = np.zeros(T, dtype=np.float32)

                for i in range(T):
                    if i == 0:
                        # forward difference at first frame
                        if dt_forward[i] <= self.max_gap_frames / self.orig_fps:
                            vx[i] = (x[i+1] - x[i]) / dt_forward[i]
                            vy[i] = (y[i+1] - y[i]) / dt_forward[i]
                    elif i == T-1:
                        # backward difference at last frame
                        if dt_backward[i] <= self.max_gap_frames / self.orig_fps:
                            vx[i] = (x[i] - x[i-1]) / dt_backward[i]
                            vy[i] = (y[i] - y[i-1]) / dt_backward[i]
                    else:
                        # central difference
                        gap_back = frames[i] - frames[i-1]
                        gap_fwd = frames[i+1] - frames[i]
                        if gap_back <= self.max_gap_frames and gap_fwd <= self.max_gap_frames:
                            dt_b = gap_back / self.orig_fps
                            dt_f = gap_fwd / self.orig_fps
                            denom = dt_b + dt_f
                            if denom > 1e-8:
                                vx[i] = (x[i+1] - x[i-1]) / denom
                                vy[i] = (y[i+1] - y[i-1]) / denom

                # -------------------------
                # Compute central difference acceleration
                # -------------------------
                ax = np.zeros(T, dtype=np.float32)
                ay = np.zeros(T, dtype=np.float32)

                if feature_mode == "pos_vel_acc":
                    for i in range(1, T-1):
                        gap_back = frames[i] - frames[i-1]
                        gap_fwd = frames[i+1] - frames[i]
                        if gap_back <= self.max_gap_frames and gap_fwd <= self.max_gap_frames:
                            dt_b = gap_back / self.orig_fps
                            dt_f = gap_fwd / self.orig_fps
                            denom = dt_b + dt_f
                            if denom > 1e-8:
                                ax[i] = 2 * ((x[i+1] - x[i]) / dt_f - (x[i] - x[i-1]) / dt_b) / denom
                                ay[i] = 2 * ((y[i+1] - y[i]) / dt_f - (y[i] - y[i-1]) / dt_b) / denom

                # -------------------------
                # Build feature matrix
                # -------------------------
                if feature_mode == "pos":
                    feats = np.stack([x, y], axis=1)
                elif feature_mode == "pos_vel":
                    feats = np.stack([x, y, vx, vy], axis=1)
                else:  # pos_vel_acc
                    feats = np.stack([x, y, vx, vy, ax, ay], axis=1)

                self.data[vid] = feats

                # Build window index map
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




# Notes to self
"""
normalization per sample is not good, need to change 

might want to add validity mask channel in the feature tensor
"""