# path overlays on video

# visualization.py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw_hands(df_hand_left, df_hand_right, title="Hand Trajectories", background=None, save_path=None):
    """
    Draws processed hand positions in the video plane (1920x1080).
    
    Parameters
    ----------
    df_hand_left : DataFrame
        Left hand trajectory with columns cx_smooth, cy_smooth, segment_id
    df_hand_right : DataFrame
        Right hand trajectory with columns cx_smooth, cy_smooth, segment_id
    title : str
        Plot title
    background : str or ndarray, optional
        Path to background image or image array
    """

    plt.figure(figsize=(12, 6))

    # Handle background image
    if background is None:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    else:
        if isinstance(background, str):
            img = plt.imread(background)
        else:
            img = background

    plt.imshow(img)

    # Draw left hand path by segment
    for _, seg_data in df_hand_left.groupby('segment_id'):
        plt.plot(seg_data['cx_smooth'], seg_data['cy_smooth'],
                 color='dodgerblue', linewidth=1.8, alpha=0.5)

    # Draw right hand path by segment
    for _, seg_data in df_hand_right.groupby('segment_id'):
        plt.plot(seg_data['cx_smooth'], seg_data['cy_smooth'],
                 color='orangered', linewidth=1.8, alpha=0.5)

    # Match video coordinate system
    height, width = img.shape[:2]
    print(f"Image dimensions: width={width}, height={height}")
    plt.xlim(0, width)
    plt.ylim(height, 0)

    # no axis ticks
    plt.axis("off")

    plt.margins(0)

    #plt.title(title)

    legend_handles = [
        Line2D([0], [0], color='dodgerblue', lw=1.8, label='Left Hand', alpha=0.6),
        Line2D([0], [0], color='orangered', lw=1.8, label='Right Hand', alpha=0.6)
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.8, 0.98))#, loc='upper left')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300
        )
    
    plt.show()


def draw_hand_trajectories(video_path, df_processed=None, fps=30, tail_length=30):
    """
    Annotates a video with trajectories. 
    The 'head' dot is only drawn if tracking data exists for the EXACT current frame.
    """
    # --- 1. Load Data ---
    base_name = os.path.basename(video_path).replace('.mp4', '')
    if df_processed is None:
        pkl_path = f'data/processed/landmark_dataframes/hand_tracking_{base_name}_{fps}fps_processed.pkl'
        df = pd.read_pickle(pkl_path)
    else:
        df = df_processed.copy()

    # --- 2. Setup Landmarks & Colors ---
    # Find all smoothed coordinate columns
    x_cols = [c for c in df.columns if c.endswith('_x_smooth')]
    landmark_ids = [c.split('_')[1] for c in x_cols]

    landmark_colors = {
        '0': (0, 255, 0),    # Wrist: Green
        '5': (255, 255, 0),  # Index MCP: Cyan
        '17': (0, 255, 255), # Pinky MCP: Yellow
    }
    default_palette = [(255, 0, 0), (255, 0, 255), (0, 165, 255)]

    # --- 3. Setup Video IO ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = f'data/experiments/{base_name}_trajectories_{fps}fps.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))

    # --- 4. Process Video ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Drawing trajectories for: {base_name}")

    with tqdm(total=total_frames, unit="frames") as pbar:
        for current_frame in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            for hand_label in ['Right', 'Left']:
                # Filter data for the "tail" window up to current frame
                tail_start = max(0, current_frame - tail_length)
                
                # We filter <= current_frame. If current_frame is a gap, 
                # path_data will end at the most recent VALID frame before the gap.
                path_data = df[(df['hand_label'] == hand_label) & 
                               (df['frame'] >= tail_start) & 
                               (df['frame'] <= current_frame)].sort_values('frame')

                if len(path_data) < 2:
                    continue

                for idx, lm_id in enumerate(landmark_ids):
                    x_col = f'lm_{lm_id}_x_smooth'
                    y_col = f'lm_{lm_id}_y_smooth'
                    color = landmark_colors.get(lm_id, default_palette[idx % len(default_palette)])

                    # Draw the segments of the tail
                    for i in range(1, len(path_data)):
                        # Break line if it belongs to a different tracking segment
                        if path_data['segment_id'].iloc[i] != path_data['segment_id'].iloc[i-1]:
                            continue
                            
                        p1 = (int(path_data[x_col].iloc[i-1]), int(path_data[y_col].iloc[i-1]))
                        p2 = (int(path_data[x_col].iloc[i]), int(path_data[y_col].iloc[i]))
                        
                        # Draw line
                        # Make it thicker if it is the very last segment available
                        thickness = 2 if i == len(path_data) - 1 else 1
                        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
                        
                        # --- MODIFICATION START ---
                        # Only draw the "head" dot if this last point corresponds 
                        # exactly to the current video frame.
                        if i == len(path_data) - 1:
                            # Check if the frame of this point is the current frame
                            point_frame = int(path_data['frame'].iloc[i])
                            
                            if point_frame == current_frame:
                                cv2.circle(frame, p2, 4, color, -1, cv2.LINE_AA)
                        # --- MODIFICATION END ---

            # Overlay Frame Number
            cv2.putText(frame, f'Frame: {current_frame}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_path}")
    return output_path


def draw_raw_vs_processed_trajectories(
    video_path,
    raw_df,
    processed_df,
    output_path=None,
    tail_length=30,
    show_raw=True,
    show_processed=True,
    show_landmarks=("palm_center", "0", "5", "17"),
):

    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import os

    base_name = os.path.basename(video_path).replace(".mp4", "")
    if output_path is None:
        output_path = f"{base_name}_raw_vs_processed_bothhands.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_width, frame_height)
    )

    raw_df = raw_df.copy()
    processed_df = processed_df.copy()

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    def _to_pixel_xy(coord):
        if coord is None:
            return None
        if not isinstance(coord, (tuple, list, np.ndarray)) or len(coord) < 2:
            return None

        x, y = coord[0], coord[1]

        if x is None or y is None:
            return None

        x = float(x)
        y = float(y)

        if 0 <= x <= 1.5 and 0 <= y <= 1.5:
            x *= frame_width
            y *= frame_height

        return int(x), int(y)

    def _extract_raw_landmark(row, lm_id):
        lms = row["landmarks"]
        if not isinstance(lms, list):
            return None

        for item in lms:
            if item["id"] == lm_id:
                return _to_pixel_xy(item["coord"])

        return None

    def _processed_point(row, key):
        if key == "palm_center":
            x = row.get("cx_smooth")
            y = row.get("cy_smooth")
        else:
            x = row.get(f"lm_{key}_x_smooth")
            y = row.get(f"lm_{key}_y_smooth")

        if pd.isna(x) or pd.isna(y):
            return None

        return int(x), int(y)

    # ---------------------------------------------------------
    # Color scheme per hand
    # ---------------------------------------------------------

    colors = {
        "Right": {
            "palm_center": (0, 0, 255),
            "0": (0, 255, 0),
            "5": (255, 255, 0),
            "17": (0, 255, 255),
        },
        "Left": {
            "palm_center": (0, 0, 255),
            "0": (0, 255, 0),
            "5": (255, 255, 0),
            "17": (0, 255, 255),
        }
    }

    def darken(c):
        return tuple(int(v * 0.5) for v in c)

    # ---------------------------------------------------------
    # Drawing loop
    # ---------------------------------------------------------

    print("Drawing trajectories for both hands...")

    with tqdm(total=total_frames) as pbar:

        for current_frame in range(total_frames):

            ret, frame = cap.read()
            if not ret:
                break

            tail_start = max(0, current_frame - tail_length)

            for hand in ["Right", "Left"]:
                opposite_hand = "Left" if hand == "Right" else "Right"

                # the raw tracked data has swaped hand labels
                raw_tail = raw_df[
                    (raw_df["hand_label"] == opposite_hand) &
                    (raw_df["frame"] >= tail_start) &
                    (raw_df["frame"] <= current_frame)
                ].sort_values("frame")

                proc_tail = processed_df[
                    (processed_df["hand_label"] == hand) &
                    (processed_df["frame"] >= tail_start) &
                    (processed_df["frame"] <= current_frame)
                ].sort_values("frame")

                for key in show_landmarks:

                    base_color = colors[hand].get(key, (255, 255, 255))
                    raw_color = darken(base_color)

                    # ---- RAW TRAJECTORY ----
                    if show_raw and len(raw_tail) >= 2:

                        pts = []
                        frames = []

                        for _, row in raw_tail.iterrows():

                            if key == "palm_center":
                                pt = _to_pixel_xy(row["palm_center"])
                            else:
                                pt = _extract_raw_landmark(row, int(key))

                            if pt is not None:
                                pts.append(pt)
                                frames.append(row["frame"])

                        for i in range(1, len(pts)):
                            if frames[i] - frames[i - 1] > 1:
                                continue
                            cv2.line(frame, pts[i - 1], pts[i], raw_color, 1)

                        if len(pts) > 0 and frames[-1] == current_frame:
                            cv2.circle(frame, pts[-1], 3, raw_color, -1)

                    # ---- PROCESSED TRAJECTORY ----
                    if show_processed and len(proc_tail) >= 2:

                        pts = []
                        frames = []
                        segs = []

                        for _, row in proc_tail.iterrows():

                            pt = _processed_point(row, key)

                            if pt is not None:
                                pts.append(pt)
                                frames.append(row["frame"])
                                segs.append(row["segment_id"])

                        for i in range(1, len(pts)):
                            if segs[i] != segs[i - 1]:
                                continue
                            cv2.line(frame, pts[i - 1], pts[i], base_color, 2)

                        if len(pts) > 0 and frames[-1] == current_frame:
                            cv2.circle(frame, pts[-1], 5, base_color, -1)

            # frame index
            cv2.putText(
                frame,
                f"Frame {current_frame}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()

    print(f"Video saved to {output_path}")

    return output_path