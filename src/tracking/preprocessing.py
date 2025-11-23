# downsampling, interpolation, missing data handling
# smothing with savitzky-golay filter or other maybe

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class TrajectoryProcessor:
    """
    Cleans and preprocesses hand trajectory data extracted from MediaPipe:
      - optional label swap,
      - hand-label consistency,
      - segmentation based on frame gaps,
      - position smoothing.
    """

    def __init__(
        self,
        fps=30,
        max_gap_sec=0.2,
        smoothing_sec=0.3,
        smoothing_poly=2,
        max_jump_px=100,
        center_col="palm_center",
        enable_label_swap=True,
        max_interp_frames=3,
        frame_size=(1920, 1080),
    ):
        self.fps = fps
        self.max_gap_sec = max_gap_sec
        self.smoothing_window = max(int(smoothing_sec * fps), 5)
        self.smoothing_poly = smoothing_poly
        self.max_jump_px = max_jump_px
        self.center_col = center_col
        self.enable_label_swap = enable_label_swap
        self.max_interp_frames = max_interp_frames
        self.frame_width, self.frame_height = frame_size

    # -------------------------------------------------------------------------
    # --- LABEL OPERATIONS ----------------------------------------------------
    # -------------------------------------------------------------------------

    def swap_labels(self, df):
        """
        Swaps Left <-> Right labels (Mediapipe by default mixes them up as it is trained from POV perspective).
        Wrapped safely so it does nothing if unexpected labels appear.
        """
        if not self.enable_label_swap:
            return df

        df = df.copy()
        mapping = {"Left": "Right", "Right": "Left"}

        if "hand_label" in df.columns:
            df["hand_label"] = df["hand_label"].map(mapping).fillna(df["hand_label"])

        return df

    # -------------------------------------------------------------------------
    # --- HAND LABEL CONSISTENCY ---------------------------------------------
    # -------------------------------------------------------------------------

    def enforce_hand_label_consistency(self, df):
        """
        Ensures each frame contains at most 1 Left and 1 Right hand.
        Chooses the most plausible instance based on proximity to the last known location.
        Removes unrealistic jumps and clears stale hand positions.
        """

        df = df.sort_values("frame").reset_index(drop=True)
        cleaned = []

        last_pos = {"Left": None, "Right": None}
        last_frame = {"Left": 0, "Right": 0}

        for frame, group in df.groupby("frame"):
            frame_entries = []

            for label in ["Left", "Right"]:
                hands = group[group["hand_label"] == label]

                # No hand detected for this label
                if len(hands) == 0:
                    if last_pos[label] is not None:
                        x, y = last_pos[label]
                        # if close to edges -> probably left frame, clear
                        if (
                            x < 25 or
                            x > self.frame_width - 25 or
                            y < 25 or
                            y > self.frame_height - 25
                        ):
                            last_pos[label] = None

                    # Clear if missing too long
                    if frame - last_frame[label] > 6:
                        last_pos[label] = None

                    continue

                # Exactly one hand detected
                elif len(hands) == 1:
                    row = hands.iloc[0]
                    # Check that last known position of other hand is not too close
                    other_label = "Right" if label == "Left" else "Left"
                    if last_pos[other_label] is not None:
                        ox, oy = last_pos[other_label]
                        hx, hy = row[self.center_col]
                        if np.hypot(hx - ox, hy - oy) < 30:  # too close
                            last_pos[label] = None
                            continue

                # Multiple hands detected → choose closest to previous
                else:
                    if last_pos[label] is not None:
                        px, py = last_pos[label]
                        d = hands[self.center_col].apply(
                            lambda c: np.hypot(c[0] - px, c[1] - py)
                        )
                        row = hands.loc[d.idxmin()]
                    else:
                        row = hands.iloc[0]

                # Check for unrealistic jumps
                if last_pos[label] is not None:
                    dx = row[self.center_col][0] - last_pos[label][0]
                    dy = row[self.center_col][1] - last_pos[label][1]
                    dist = np.hypot(dx, dy)

                    frame_gap = (frame - last_frame[label])
                    if dist > self.max_jump_px * frame_gap:
                        # Resume if long gap
                        if frame_gap > 6:
                            last_pos[label] = row[self.center_col]
                        continue

                last_pos[label] = row[self.center_col]
                last_frame[label] = frame
                frame_entries.append(row)

            cleaned.extend(frame_entries)

        return pd.DataFrame(cleaned).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # ------------------- Interpolation ---------------------------------------
    # -------------------------------------------------------------------------

    def interpolate_gaps(self, df, max_gap_sec=0.2):
        """
        Linearly interpolates short gaps in hand trajectories.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'hand_label' and the coordinates in self.center_col (tuple of x,y)
        max_gap_sec : float
            Maximum acceptable gap duration (in seconds) to interpolate.
            Gaps longer than this are left as missing.

        Returns
        -------
        df_interp : pd.DataFrame
            DataFrame with 'cx' and 'cy' interpolated over short gaps.
        """
        df = df.copy()
        df["cx"] = df[self.center_col].apply(lambda p: p[0] if p is not None else np.nan)
        df["cy"] = df[self.center_col].apply(lambda p: p[1] if p is not None else np.nan)

        # Compute max frames to interpolate based on fps
        max_interp_frames = int(max_gap_sec * self.fps)

        interpolated_dfs = []
        for label in ["Left", "Right"]:
            hand_df = df[df["hand_label"] == label].sort_values("frame").copy()
            
            # Linear interpolation over short gaps only
            hand_df["cx"] = hand_df["cx"].interpolate(method="linear", limit=max_interp_frames)
            hand_df["cy"] = hand_df["cy"].interpolate(method="linear", limit=max_interp_frames)

            interpolated_dfs.append(hand_df)

        return pd.concat(interpolated_dfs).sort_values("frame").reset_index(drop=True)

    
    # -------------------------------------------------------------------------
    # --- SEGMENTATION --------------------------------------------------------
    # -------------------------------------------------------------------------

    def generate_segments(self, df):
        """
        Creates segment IDs when temporal gaps exceed max_gap_sec.
        """
        df = df.copy()
        max_gap_frames = int(self.max_gap_sec * 30)

        df["frame_diff"] = df["frame"].diff()
        df["segment_id"] = (df["frame_diff"] > max_gap_frames).cumsum()

        return df

    # -------------------------------------------------------------------------
    # --- SMOOTHING -----------------------------------------------------------
    # -------------------------------------------------------------------------

    def smooth(self, df):
        """
        Applies Savitzky–Golay smoothing to x,y coordinates within each segment.
        """
        df = df.copy()

        df["cx"] = df[self.center_col].apply(lambda p: p[0] if p is not None else np.nan)
        df["cy"] = df[self.center_col].apply(lambda p: p[1] if p is not None else np.nan)

        if self.smoothing_window % 2 == 0:
            window = self.smoothing_window + 1
        else:
            window = self.smoothing_window

        df["cx_smooth"] = np.nan
        df["cy_smooth"] = np.nan

        for seg_id, seg in df.groupby("segment_id"):
            if len(seg) >= window:
                cx_s = savgol_filter(seg["cx"], window, self.smoothing_poly, mode="interp")
                cy_s = savgol_filter(seg["cy"], window, self.smoothing_poly, mode="interp")
            else:
                cx_s = seg["cx"]
                cy_s = seg["cy"]

            df.loc[seg.index, "cx_smooth"] = cx_s
            df.loc[seg.index, "cy_smooth"] = cy_s

        return df.drop(columns=["cx", "cy"])

    # -------------------------------------------------------------------------
    # --- FULL PIPELINE FOR A SINGLE HAND ------------------------------------
    # -------------------------------------------------------------------------

    def process_one_hand(self, df):
        if len(df) == 0:
            return df

        df = self.generate_segments(df)
        df = self.smooth(df)

        return df

    # -------------------------------------------------------------------------
    # --- FULL PIPELINE -------------------------------------------------------
    # -------------------------------------------------------------------------

    def process(self, df):
        """
        Full preprocessing pipeline:
            - optional label swap
            - label consistency
            - process left & right independently
            - return unified dataframe
        """
        df = self.swap_labels(df)
        df = self.enforce_hand_label_consistency(df)
        df = self.interpolate_gaps(df)

        left = df[df["hand_label"] == "Left"]
        right = df[df["hand_label"] == "Right"]

        left = self.process_one_hand(left)
        right = self.process_one_hand(right)

        return pd.concat([left, right]).sort_values("frame").reset_index(drop=True)

