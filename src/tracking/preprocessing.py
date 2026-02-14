# downsampling, interpolation, missing data handling
# smothing with savitzky-golay filter or other maybe

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter


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

        # State memory
        last_pos = {"Left": None, "Right": None}
        last_frame = {"Left": 0, "Right": 0}

        for frame, group in df.groupby("frame"):
            frame_entries = []

            for label in ["Left", "Right"]:
                other_label = "Right" if label == "Left" else "Left"

                # --- STEP 1: Collect detections for this label ---
                hands = group[group["hand_label"] == label]

                # --- CASE A: NO HAND DETECTED ---
                if len(hands) == 0:

                    # Edge-based clearing
                    if last_pos[label] is not None:
                        x, y = last_pos[label]
                        if (
                            x < 25 or
                            x > self.frame_width - 25 or
                            y < 25 or
                            y > self.frame_height - 25
                        ):
                            last_pos[label] = None

                    # Timeout clearing
                    if frame - last_frame[label] > 6:
                        last_pos[label] = None

                    continue

                # --- CASE B: EXACTLY ONE HAND ---
                elif len(hands) == 1:
                    row = hands.iloc[0]

                    # Collision check with other hand
                    if last_pos[other_label] is not None:
                        ox, oy = last_pos[other_label]
                        hx, hy = row[self.center_col]

                        if np.hypot(hx - ox, hy - oy) < 40:
                            if frame - last_frame[other_label] > 10:
                                last_pos[other_label] = None
                            continue

                # --- CASE C: MULTIPLE HANDS ---
                else:
                    # Score dominance check
                    sorted_scores = hands["hand_score"].sort_values(ascending=False).values

                    if sorted_scores[0] - sorted_scores[1] > 0.1:
                        row = hands.loc[hands["hand_score"].idxmax()]

                    # Otherwise use spatial continuity
                    elif last_pos[label] is not None:
                        px, py = last_pos[label]
                        d = hands[self.center_col].apply(
                            lambda c: np.hypot(c[0] - px, c[1] - py)
                        )
                        row = hands.loc[d.idxmin()]

                    # Fallback
                    else:
                        row = hands.iloc[0]

                # --- STEP 2: Jump Consistency Check ---
                if last_pos[label] is not None:
                    dx = row[self.center_col][0] - last_pos[label][0]
                    dy = row[self.center_col][1] - last_pos[label][1]
                    dist = np.hypot(dx, dy)

                    frame_gap = frame - last_frame[label]

                    if dist > self.max_jump_px * frame_gap:
                        # Allow re-init after long gap
                        if frame_gap > 15:
                            last_pos[label] = row[self.center_col]
                        continue

                # --- STEP 3: ACCEPTANCE ---
                last_pos[label] = row[self.center_col]
                last_frame[label] = frame
                frame_entries.append(row)

            cleaned.extend(frame_entries)

        return pd.DataFrame(cleaned).reset_index(drop=True)

    

    def enforce_hand_label_consistency_conservative(self, df):
        """
        Conservative approach: Prioritizes dropping doubtful frames over 
        interpolating false jumps. Strictly rejects 'Ghost' detections 
        that appear on top of the other active hand.
        """
        # --- CONFIGURATION ---
        MIN_CONFIDENCE = 0.5       # Drop weak detections immediately
        COLLISION_RADIUS = 40      # If Left/Right are this close, reject the weaker one
        REENTRY_COLLISION_GUARD = 50 # Stricter radius when a hand 'reappears' after a gap

        df = df.sort_values("frame").reset_index(drop=True)
        cleaned = []

        last_pos = {"Left": None, "Right": None}
        last_frame = {"Left": 0, "Right": 0}

        for frame, group in df.groupby("frame"):
            frame_entries = []

            # We iterate labels, but checking 'other_label' dynamically 
            # allows us to compare against the *freshest* data available.
            for label in ["Left", "Right"]:
                other_label = "Right" if label == "Left" else "Left"
                
                # 1. Filter by Confidence first (Conservative Step)
                hands = group[(group["hand_label"] == label) & (group["hand_score"] > MIN_CONFIDENCE)]

                # --- CASE A: NO HAND DETECTED ---
                if len(hands) == 0:
                    # Check if we should clear history due to edge exit or timeout
                    if last_pos[label] is not None:
                        x, y = last_pos[label]
                        # Edge exit logic
                        if (x < 25 or x > self.frame_width - 25 or 
                            y < 25 or y > self.frame_height - 25):
                            last_pos[label] = None
                    
                    # Timeout logic (short memory prevents connecting across huge gaps)
                    if frame - last_frame[label] > 6:
                        last_pos[label] = None
                    continue

                # --- CASE B: CANDIDATE SELECTION ---
                # Determine the best candidate row
                if len(hands) == 1:
                    row = hands.iloc[0]
                else:
                    # Multiple candidates: 
                    # If we have history, pick the one closest to last known pos
                    if last_pos[label] is not None:
                        px, py = last_pos[label]
                        d = hands[self.center_col].apply(lambda c: np.hypot(c[0] - px, c[1] - py))
                        row = hands.loc[d.idxmin()]
                    else:
                        # No history? Pick the highest confidence
                        row = hands.loc[hands['hand_score'].idxmax()]

                # --- CASE C: CONSERVATIVE VALIDATION (The Fix) ---
                
                current_pos = row[self.center_col]
                valid_candidate = True

                # 1. Check Collision with OTHER Hand (Prevent "Hand Stealing")
                # We check against the other hand's last known position
                if last_pos[other_label] is not None:
                    ox, oy = last_pos[other_label]
                    dist_to_other = np.hypot(current_pos[0] - ox, current_pos[1] - oy)
                    
                    # If we are too close to the other hand, this is suspicious.
                    if dist_to_other < COLLISION_RADIUS:
                        # CONSERVATIVE: If we are clashing, drop this current detection.
                        # It is better to have a gap than a swapped label.
                        valid_candidate = False

                if not valid_candidate:
                    continue

                # 2. Check Jumps / Continuity
                if last_pos[label] is not None:
                    dx = current_pos[0] - last_pos[label][0]
                    dy = current_pos[1] - last_pos[label][1]
                    dist_jump = np.hypot(dx, dy)
                    frame_gap = frame - last_frame[label]

                    # Threshold: Allow larger jumps only if frame gap is small
                    max_allowed = 0.5 * self.max_jump_px * (1 + frame_gap)
                    
                    if dist_jump > max_allowed:
                        # --- RE-ENTRY LOGIC ---
                        # If the gap was long (>15), this might be a valid re-entry.
                        # BUT, we must ensure it's not a "Ghost" of the other hand.
                        if frame_gap > 15:
                            # Re-entry Guard: Only accept if far from other hand
                            if last_pos[other_label] is not None:
                                ox, oy = last_pos[other_label]
                                if np.hypot(current_pos[0]-ox, current_pos[1]-oy) < REENTRY_COLLISION_GUARD:
                                    # It reappeared right on top of the other hand -> REJECT
                                    continue
                            
                            # If passed guard, accept as new position (teleport allowed on re-entry)
                            pass 
                        else:
                            # Short gap + Huge jump = Noise/Teleport -> REJECT
                            continue

                # --- ACCEPTANCE ---
                last_pos[label] = current_pos
                last_frame[label] = frame
                frame_entries.append(row)

            cleaned.extend(frame_entries)

        return pd.DataFrame(cleaned).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # ------------------- Interpolation ---------------------------------------
    # -------------------------------------------------------------------------


    def interpolate_gaps(self, df, max_gap_sec=0.2):
        """
        Interpolates short gaps ONLY if the spatial jump is physically plausible.
        """
        if df.empty or 'landmarks' not in df.columns:
            return df
            
        df = df.sort_values('frame').reset_index(drop=True)
        
        # --- 1. PREP: Explode the Dictionary Column ---
        first_valid_entry = df['landmarks'].dropna().iloc[0]
        tracked_ids = list(first_valid_entry.keys())
        
        flattened_data = []
        for entry in df['landmarks']:
            row_flat = {}
            if isinstance(entry, dict):
                for lid, coords in entry.items():
                    row_flat[f'lm_{lid}_x'] = coords[0]
                    row_flat[f'lm_{lid}_y'] = coords[1]
            flattened_data.append(row_flat)
            
        df_flat = pd.DataFrame(flattened_data)
        df_merged = pd.concat([df.drop(columns=['landmarks']), df_flat], axis=1)

        # --- 2. Identify Gaps and Insert Rows (with Spatial Check) ---
        max_gap_frames = int(max_gap_sec * self.fps)
        frame_diffs = df_merged['frame'].diff()
        
        # We look for temporal gaps that are small enough to potentially interpolate
        short_gap_mask = (frame_diffs > 1) & (frame_diffs <= max_gap_frames)
        short_gap_indices = df_merged.index[short_gap_mask]
        
        rows_to_insert = []
        
        for idx in short_gap_indices:
            # Get endpoints of the gap
            prev_row = df_merged.loc[idx - 1]
            curr_row = df_merged.loc[idx]
            
            # Calculate Euclidean distance of the jump (using palm center cx, cy)
            dist = np.hypot(curr_row['cx'] - prev_row['cx'], 
                            curr_row['cy'] - prev_row['cy'])
            
            # SPATIAL GUARDRAIL:
            # If the hand 'teleported' faster than max_jump_px * time_elapsed, 
            # we do NOT insert rows to interpolate. We let it remain a gap.
            frame_gap = curr_row['frame'] - prev_row['frame']
            if dist > (self.max_jump_px * frame_gap):
                continue # Skip interpolation for this specific gap

            # If it passed the check, create the missing frame rows
            base_row = prev_row[['hand_label']].to_dict()
            for f in range(int(prev_row['frame']) + 1, int(curr_row['frame'])):
                new_row = base_row.copy()
                new_row['frame'] = f
                rows_to_insert.append(new_row)
                
        if rows_to_insert:
            df_new = pd.DataFrame(rows_to_insert)
            df_merged = pd.concat([df_merged, df_new], ignore_index=True)
            df_merged = df_merged.sort_values('frame').reset_index(drop=True)

        # --- 3. Apply Linear Interpolation ---
        coord_cols = [c for c in df_merged.columns if c.startswith('lm_')] + ['cx', 'cy']
        
        # limit=max_gap_frames ensures we don't accidentally fill massive holes
        df_merged[coord_cols] = df_merged[coord_cols].interpolate(
            method='linear', 
            limit=max_gap_frames,
            limit_area='inside'
        )

        return df_merged
    
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


    def smooth(self, df, median_kernel=3):
        """
        Applies a hybrid filter: 
        1. Median filter to remove impulsive outliers (MediaPipe glitches).
        2. Savitzky–Golay smoothing to reduce high-frequency jitter.
        """
        df = df.copy()

        # Determine Savgol window size (must be odd)
        window = self.smoothing_window
        if window % 2 == 0:
            window += 1

        # Identify all coordinate columns
        coord_cols = [c for c in df.columns if c.startswith('lm_')] + ['cx', 'cy']
        
        if not coord_cols:
            return df

        # Prepare a list to store smoothed values
        smoothed_data = {f"{col}_smooth": np.full(len(df), np.nan) for col in coord_cols}

        # Iterate through segments
        for seg_id, seg in df.groupby("segment_id"):
            if len(seg) >= window:
                for col in coord_cols:
                    # --- STEP 1: Median Filtering ---
                    # This removes the "188.0" outliers before they can ruin the Savgol fit
                    interim_signal = medfilt(seg[col], kernel_size=median_kernel)

                    # --- STEP 2: Savitzky-Golay Smoothing ---
                    smoothed_values = savgol_filter(
                        interim_signal, 
                        window_length=window, 
                        polyorder=self.smoothing_poly, 
                        mode="interp"
                    )
                    smoothed_data[f"{col}_smooth"][seg.index] = smoothed_values
            else:
                # If segment is too short, just copy the original data
                for col in coord_cols:
                    smoothed_data[f"{col}_smooth"][seg.index] = seg[col].values

        # Update the DataFrame
        for col_name, values in smoothed_data.items():
            df[col_name] = values

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
            lambda lms: {lm['id']: (round(lm['coord'][0] * 1920, 2), round(lm['coord'][1] * 1080, 2)) for lm in lms if lm['id'] in landmarks} if lms is not None else None
        )

        df = self.swap_labels(df)
        df = self.enforce_hand_label_consistency(df)

        left = df[df["hand_label"] == "Left"]
        right = df[df["hand_label"] == "Right"]

        left = self.process_one_hand(left)
        right = self.process_one_hand(right)

        return pd.concat([left, right]).sort_values("frame").reset_index(drop=True)

