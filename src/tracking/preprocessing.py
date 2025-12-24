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
        max_gap_sec=0.1,
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
        Interpolates only short gaps (<= max_gap_sec) in hand trajectories by:
        1. Identifying small gaps.
        2. Inserting new rows only for frames within those small gaps.
        3. Applying linear interpolation across the entire dataframe.
        
        The resulting dataframe contains the original tracked rows and the newly 
        interpolated rows for short gaps.
        """
        if df.empty:
            return df
            
        df = df.copy() 

        # --- 1. PREP: Identify Coordinates and Frame Limit ---
        
        # Use existing smooth columns if present, otherwise assume raw cx/cy
        coord_cols = ["cx", "cy"] 

        max_interp_frames = int(max_gap_sec * self.fps) # !! might be wrong when working with 10 fps !!
        
        # --- 2. Create and Insert Rows for ONLY Short Gaps ---
        
        df = df.sort_values('frame').reset_index(drop=True)
        df['frame'] = df['frame'] / float(30 / self.fps)  # Normalize frame numbers to seconds for interpolation
        
        # Calculate the frame difference between consecutive tracked points
        frame_diffs = df['frame'].diff()
        
        # Identify the start index of gaps that are *short* (<= max_interp_frames)
        short_gap_indices = df.index[
            (frame_diffs > 1) & (frame_diffs <= max_interp_frames)
        ]
        
        rows_to_insert = []
        
        for start_idx in short_gap_indices:
            # 'frame' of the last tracked point before the gap
            start_frame = df.loc[start_idx - 1, 'frame']
            # 'frame' of the next tracked point after the gap
            end_frame = df.loc[start_idx, 'frame']
            
            # Create new frames between start_frame + 1 and end_frame - 1
            new_frames = pd.RangeIndex(start=start_frame + 1, stop=end_frame, step=30/self.fps)
            
            for f in new_frames:
                # Create a row with only the frame number and hand label, 
                # all coordinate columns will be NaN
                new_row = {'frame': f, 'hand_label': df.loc[start_idx, 'hand_label']}
                rows_to_insert.append(new_row)

        # Convert to DataFrame and concatenate with the original data
        if rows_to_insert:
            df_new_rows = pd.DataFrame(rows_to_insert)
            df = pd.concat([df, df_new_rows], ignore_index=True)
            
            # Sort again to place the new NaN rows correctly in the sequence
            df = df.sort_values('frame').reset_index(drop=True)
            
        # --- 3. Apply Linear Interpolation ---
        
        # Since we only introduced NaN rows in short gaps, a simple linear interpolation 
        # will only fill those specific holes, leaving the large gaps untouched (as they have no 
        # intermediate NaN rows to bridge).
        for col in coord_cols:
            # Note: If the coordinate columns did not exist, they were created as NaN 
            # in step 2 for the new rows. Pandas fillna/interpolate will handle this.
            if col not in df.columns:
                # Fill the coordinate column for the tracked points if they are NaN (unlikely here)
                df[col] = df.apply(lambda row: row[col][0] if isinstance(row[col], list) else row[col], axis=1) # Defensive conversion
                
            df[col] = df[col].interpolate(method='linear')
        
        # Convert frame numbers back to original scale
        df['frame'] = df['frame'] * float(30 / self.fps)
        
        return df
    
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

        df = df.copy()
        df["cx"] = df[self.center_col].apply(lambda p: p[0] if p is not None else np.nan)
        df["cy"] = df[self.center_col].apply(lambda p: p[1] if p is not None else np.nan)
        
        # 1. INTERPOLATION: Fill short gaps with smooth predictions
        df = self.interpolate_gaps(df, max_gap_sec=self.max_gap_sec)
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

        left = df[df["hand_label"] == "Left"]
        right = df[df["hand_label"] == "Right"]

        left = self.process_one_hand(left)
        right = self.process_one_hand(right)

        return pd.concat([left, right]).sort_values("frame").reset_index(drop=True)






# ============================================================================ 
# --- Very similar class but for muliple keypoint processing -----------------
# ============================================================================



class LandmarksProcessor:
    """
    Cleans and preprocesses hand landmarks trajectories extracted from MediaPipe:
      - optional label swap,
      - hand-label consistency,
      - segmentation based on frame gaps,
      - position smoothing.
    """

    def __init__(
        self,
        landmarks={0, 5, 17},  # which landmarks to process
        fps=30,
        max_gap_sec=0.1,
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
        Interpolates short gaps in a 'landmarks' column where each element is a dict:
        {landmark_id: (x, y), ...}.
        
        Strategy:
        1. Explode 'landmarks' dicts into flat columns (0_x, 0_y, 5_x, etc.).
        2. Insert NaN rows for missing frames within short gaps.
        3. Linear Interpolate the flat columns.
        4. Implode the flat columns back into the 'landmarks' dict format.
        """
        if df.empty or 'landmarks' not in df.columns:
            return df
            
        df = df.sort_values('frame').reset_index(drop=True)
        
        # --- 1. PREP: Explode the Dictionary Column ---
        # We convert the list of dicts into a DataFrame of columns like 'lm_0_x', 'lm_0_y'
        # We assume all rows track the same set of landmarks (based on the first valid row)
        
        # Get all unique landmark IDs from the first non-empty row to establish schema
        first_valid_entry = df['landmarks'].dropna().iloc[0]
        tracked_ids = list(first_valid_entry.keys())
        
        # Fast flattening using list comprehension
        flattened_data = []
        for entry in df['landmarks']:
            row_flat = {}
            if isinstance(entry, dict):
                for lid, coords in entry.items():
                    row_flat[f'lm_{lid}_x'] = coords[0]
                    row_flat[f'lm_{lid}_y'] = coords[1]
            flattened_data.append(row_flat)
            
        df_flat = pd.DataFrame(flattened_data)
        
        # Combine with metadata (frame, hand_label, etc.)
        # We drop the original complex 'landmarks' column for now
        df_merged = pd.concat([df.drop(columns=['landmarks']), df_flat], axis=1)

        # --- 2. Identify Gaps and Insert Rows ---
        
        max_gap_frames = int(max_gap_sec * self.fps)
        
        # Calculate difference between consecutive frame numbers
        frame_diffs = df_merged['frame'].diff()
        
        # Find gaps that are missing data (>1) but are small enough (<= max)
        short_gap_mask = (frame_diffs > 1) & (frame_diffs <= max_gap_frames)
        short_gap_indices = df_merged.index[short_gap_mask]
        
        rows_to_insert = []
        
        for idx in short_gap_indices:
            start_frame = int(df_merged.loc[idx - 1, 'frame'])
            end_frame = int(df_merged.loc[idx, 'frame'])
            
            # Determine the constant columns (e.g., hand_label) from the PREVIOUS valid frame
            # to propagate them into the gap
            base_row = df_merged.loc[idx - 1, ['hand_label']].to_dict()
            
            # Create missing frames
            # range(start + 1, end) creates integers strictly between start and end
            for f in range(start_frame + 1, end_frame):
                new_row = base_row.copy()
                new_row['frame'] = f
                # All 'lm_...' columns will implicitly be NaN
                rows_to_insert.append(new_row)
                
        if rows_to_insert:
            df_new = pd.DataFrame(rows_to_insert)
            df_merged = pd.concat([df_merged, df_new], ignore_index=True)
            df_merged = df_merged.sort_values('frame').reset_index(drop=True)

        # --- 3. Apply Linear Interpolation ---
        
        # Identify the coordinate columns we generated earlier
        coord_cols = [c for c in df_merged.columns if c.startswith('lm_')] + ['cx', 'cy']
        
        # Interpolate only the coordinate columns
        df_merged[coord_cols] = df_merged[coord_cols].interpolate(method='linear', limit_direction='both')

        # --- 4. REPACK: Implode back to Dictionary ---
        """
        def repack_row(row):
            d = {}
            for lid in tracked_ids:
                col_x = f'lm_{lid}_x'
                col_y = f'lm_{lid}_y'
                
                # Only add if interpolation resulted in valid numbers
                if pd.notna(row.get(col_x)) and pd.notna(row.get(col_y)):
                    d[lid] = (row[col_x], row[col_y])
            return d

        # Recreate the landmarks column
        df_merged['landmarks'] = df_merged.apply(repack_row, axis=1)
        
        # Clean up: keep only original columns + interpolated landmarks
        final_cols = [c for c in df.columns if c in df_merged.columns]"""""
        return df_merged#[final_cols]
    
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
        Applies Savitzky–Golay smoothing to all landmark coordinate columns 
        within each segment.
        """
        df = df.copy()

        # Determine window size (must be odd)
        window = self.smoothing_window
        if window % 2 == 0:
            window += 1

        # 1. Identify all coordinate columns (e.g., lm_0_x, lm_5_y...)
        coord_cols = [c for c in df.columns if c.startswith('lm_')] + ['cx', 'cy']
        
        if not coord_cols:
            return df

        # Prepare a list to store smoothed values to avoid setting with copy warnings
        # and to keep the original columns until the end
        smoothed_data = {f"{col}_smooth": np.full(len(df), np.nan) for col in coord_cols}

        # 2. Iterate through segments
        for seg_id, seg in df.groupby("segment_id"):
            # We need enough data points in the segment to apply the filter
            if len(seg) >= window:
                for col in coord_cols:
                    smoothed_values = savgol_filter(
                        seg[col], 
                        window_length=window, 
                        polyorder=self.smoothing_poly, 
                        mode="interp"
                    )
                    smoothed_data[f"{col}_smooth"][seg.index] = smoothed_values
            else:
                # If segment is too short, just copy the original data
                for col in coord_cols:
                    smoothed_data[f"{col}_smooth"][seg.index] = seg[col].values

        # 3. Update the DataFrame
        for col_name, values in smoothed_data.items():
            df[col_name] = values

        # 4. Cleanup: Drop the original unsmoothed columns
        return df.drop(columns=coord_cols)

    # -------------------------------------------------------------------------
    # --- FULL PIPELINE FOR A SINGLE HAND ------------------------------------
    # -------------------------------------------------------------------------

    def process_one_hand(self, df):
        if len(df) == 0:
            return df

        df = df.copy()
        df["cx"] = df[self.center_col].apply(lambda p: p[0] if p is not None else np.nan)
        df["cy"] = df[self.center_col].apply(lambda p: p[1] if p is not None else np.nan)
        
        # Fill short gaps and smooth
        df = self.interpolate_gaps(df, max_gap_sec=self.max_gap_sec)
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
        if 'bbox_center' in df.columns:
            df.drop(labels=['bbox_center'], axis=1, inplace=True)

        landmarks={0, 5, 17}

        # use only selected landmarks
        df['landmarks'] = df['landmarks'].apply(
            lambda lms: {lm['id']: (round(lm['coord'][0] * 1920, 1), round(lm['coord'][1] * 1080, 1)) for lm in lms if lm['id'] in landmarks} if lms is not None else None
        )

        df = self.swap_labels(df)
        df = self.enforce_hand_label_consistency(df)

        left = df[df["hand_label"] == "Left"]
        right = df[df["hand_label"] == "Right"]

        left = self.process_one_hand(left)
        right = self.process_one_hand(right)

        return pd.concat([left, right]).sort_values("frame").reset_index(drop=True)

