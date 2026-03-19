# MediapipeTracker class

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm


import os
import cv2
import mediapipe as mp
import pandas as pd
from google.protobuf.json_format import ParseDict
from mediapipe.framework.formats import landmark_pb2
import time


class MediapipeHandTracker:
    """
    Wrapper for Mediapipe hand tracking:
    - tracks left + right hand
    - extracts bbox centers, palm centers, all landmarks
    - supports FPS downsampling
    - saves results as .pkl DataFrames
    """

    def __init__(
        self,
        target_fps=30,
        min_detection_conf=0.5,
        min_tracking_conf=0.5,
        max_num_hands=2
    ):
        self.target_fps = target_fps
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.max_num_hands = max_num_hands

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    # ---------------------------------------------------------
    # Core API call
    # ---------------------------------------------------------
    def track_video(self, video_path, output_dir="../data/raw/output_dataframes", overwrite=False, verbose=True):
        """
        Tracks hands in a single video and returns dataframe.
        """

        os.makedirs(output_dir, exist_ok=True)
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            output_dir,
            f"hand_tracking_{vid_name}_{self.target_fps}fps.pkl"
        )

        if os.path.exists(output_path) and not overwrite and verbose:
            print(f"⏭️ Skipping {vid_name} (already tracked). to apply hand tracking again, set overwrite=True")
            return pd.read_pickle(output_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open {video_path}")
            return None

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(round(fps_video / self.target_fps)) if fps_video > self.target_fps else 1

        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_tracking_conf
        )

        all_data = []
        frame_idx = 0
        prev_time = time.time()

        print(f"▶️ Tracking hands in {vid_name} ...")
        with tqdm(total=num_frames // frame_step, desc="Processing frames", ncols=100) as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                frame_data = self._process_frame(frame, frame_idx, hands)
                all_data.extend(frame_data)

                curr_time = time.time()
                fps_display = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time
                pbar.set_postfix({"FPS": f"{fps_display:.2f}"})
                pbar.update(1)

                frame_idx += 1

        cap.release()
        hands.close()

        df = pd.DataFrame(all_data)
        df.to_pickle(output_path)
        print(f"✅ Saved DataFrame: {output_path} ({df.shape[0]} rows)")

        return df

    # ---------------------------------------------------------
    # Helper: process a single frame
    # ---------------------------------------------------------
    def _process_frame(self, frame, frame_idx, hands_model):
        """
        Run Mediapipe on one frame and extract data for both hands.
        Returns a list of dicts (one per detected hand or one empty row).
        """

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)
        h, w, _ = frame.shape

        frame_output = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                score = handedness.classification[0].score

                xs = [lm.x for lm in hand_lm.landmark]
                ys = [lm.y for lm in hand_lm.landmark]

                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)

                bbox_center = (round((x_min + x_max) / 2), round((y_min + y_max) / 2))

                palm_indices = [0, 1, 5, 9, 13, 17]
                palm_cx = np.mean([xs[i] for i in palm_indices]) * w
                palm_cy = np.mean([ys[i] for i in palm_indices]) * h
                palm_center = (round(palm_cx), round(palm_cy))

                landmarks_array = [
                    {"id": i, "coord": (lm.x, lm.y, lm.z)}
                    for i, lm in enumerate(hand_lm.landmark)
                ]

                frame_output.append({
                    "frame": frame_idx,
                    "hand_label": label,
                    "hand_score": score,
                    "bbox_center": bbox_center,
                    "palm_center": palm_center,
                    "landmarks": landmarks_array
                })
        else:
            frame_output.append({
                "frame": frame_idx,
                "hand_label": None,
                "hand_score": None,
                "bbox_center": (None, None),
                "palm_center": (None, None),
                "landmarks": None
            })

        return frame_output

    # ---------------------------------------------------------
    # Helper: rebuild MediaPipe landmark object from dataframe row
    # ---------------------------------------------------------
    def _landmarks_from_row(self, landmarks_list):
        """
        Convert stored landmark list
        [{'id': i, 'coord': (x, y, z)}, ...]
        back into a MediaPipe NormalizedLandmarkList.
        """
        if not isinstance(landmarks_list, list):
            return None

        lm_dict = {}
        for item in landmarks_list:
            if not isinstance(item, dict):
                continue
            lm_id = item.get("id")
            coord = item.get("coord")
            if lm_id is None or coord is None or len(coord) < 3:
                continue
            lm_dict[int(lm_id)] = coord

        if len(lm_dict) == 0:
            return None

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(21):
            if i in lm_dict:
                x, y, z = lm_dict[i]
            else:
                x, y, z = 0.0, 0.0, 0.0

            landmark = landmark_list.landmark.add()
            landmark.x = float(x)
            landmark.y = float(y)
            landmark.z = float(z)

        return landmark_list

    # ---------------------------------------------------------
    # Overlay tracked landmarks from dataframe onto video
    # ---------------------------------------------------------
    def render_tracking_overlay_video(
        self,
        video_path,
        tracking_df,
        output_path=None,
        show_label=True,
        show_score=True,
        show_frame_idx=False,
        overwrite=False
    ):
        """
        Render a video with MediaPipe-style hand overlay using an existing tracking dataframe.

        Parameters
        ----------
        video_path : str
            Path to original input video.
        tracking_df : pd.DataFrame
            DataFrame produced by track_video().
        output_path : str or None
            Path to save rendered video. If None, saves next to input video.
        show_label : bool
            Whether to draw Left/Right label.
        show_score : bool
            Whether to draw hand confidence score.
        show_frame_idx : bool
            Whether to draw frame index.
        overwrite : bool
            Whether to overwrite existing file.

        Returns
        -------
        str
            Path to saved overlay video.
        """
        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), "output_videos")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"tracked_{base}.mp4")

        if os.path.exists(output_path) and not overwrite:
            print(f"⏭️ Output already exists: {output_path}")
            return output_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

        # keep only valid rows with a detected hand
        df = tracking_df.copy()
        df = df[df["hand_label"].notna()].copy()

        # group rows by frame for fast lookup
        rows_by_frame = {frame: grp for frame, grp in df.groupby("frame")}

        frame_idx = 0
        print("▶️ Rendering tracking overlay video...")

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx in rows_by_frame:
                frame_rows = rows_by_frame[frame_idx]

                for _, row in frame_rows.iterrows():
                    hand_landmarks = self._landmarks_from_row(row["landmarks"])
                    if hand_landmarks is None:
                        continue
                    
                    # --- Draw normal skeleton (neutral color) ---
                    landmark_style = self.mp_drawing.DrawingSpec(
                        color=(200, 200, 200),  # light gray
                        thickness=2,
                        circle_radius=2
                    )

                    connection_style = self.mp_drawing.DrawingSpec(
                        color=(200, 200, 200),
                        thickness=2
                    )

                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_style,
                        connection_style,
                    )

                    # --- Highlight palm landmarks ---
                    palm_indices = [0, 1, 5, 9, 13, 17]

                    h, w, _ = frame.shape

                    for idx in palm_indices:
                        lm = hand_landmarks.landmark[idx]

                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        cv2.circle(
                            frame,
                            (x, y),
                            6,                # larger radius
                            (0, 0, 255),      # red highlight
                            -1
                        )
                    
                    # --- Draw the palm center ---
                    palm_center = row["palm_center"]
                    if (
                        isinstance(palm_center, (tuple, list)) and
                        len(palm_center) == 2 and
                        pd.notna(palm_center[0]) and
                        pd.notna(palm_center[1])
                    ):
                        cx = int(round(float(palm_center[0])))
                        cy = int(round(float(palm_center[1])))
                        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

                    # label position: use top-left landmark extent
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(xs) * width)
                    y_min = int(min(ys) * height)

                    text_parts = []
                    if show_label:
                        text_parts.append(str(row["hand_label"]))
                    if show_score and pd.notna(row.get("hand_score", None)):
                        text_parts.append(f"{float(row['hand_score']):.2f}")

                    if text_parts:
                        label_text = " | ".join(text_parts)
                        cv2.putText(
                            frame,
                            label_text,
                            (x_min, max(25, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

            if show_frame_idx:
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Video saved successfully: {output_path}")

        return output_path

    # ---------------------------------------------------------
    # Batch processing wrapper
    # ---------------------------------------------------------
    def process_directory(
        self,
        input_dir,
        output_dir="../data/raw/output_dataframes",
        batch_size=5,
        overwrite=False
    ):
        """
        Process up to batch_size unprocessed videos in a directory.
        """

        os.makedirs(output_dir, exist_ok=True)

        videos = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(".mp4")
        ]

        processed_files = {
            f.split("_", 2)[1]: f
            for f in os.listdir(output_dir)
            if f.endswith(f"{self.target_fps}fps.pkl")
        }

        unprocessed = [
            v for v in videos
            if os.path.splitext(os.path.basename(v))[0] not in processed_files
        ]

        if not unprocessed:
            print("✅ All videos already processed.")
            return []

        print(f"🔹 Found {len(unprocessed)} unprocessed videos, processing up to {batch_size}...")

        results = []
        for v in unprocessed[:batch_size]:
            out_path = self.track_video(v, output_dir=output_dir, overwrite=overwrite)
            if out_path is not None:
                results.append(out_path)

        return results
