# MediapipeTracker class

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm


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

    # ---------------------------------------------------------
    # Core API call
    # ---------------------------------------------------------
    def track_video(self, video_path, output_dir="data/raw/output_dataframes", overwrite=False):
        """
        Tracks hands in a single video and returns dataframe.
        """

        os.makedirs(output_dir, exist_ok=True)
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            output_dir,
            f"hand_tracking_{vid_name}_{self.target_fps}fps.pkl"
        )

        if os.path.exists(output_path) and not overwrite:
            print(f"⏭️ Skipping {vid_name} (already processed)")
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

                # downsample frames
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                frame_data = self._process_frame(frame, frame_idx, hands)
                all_data.extend(frame_data)

                # progress bar & fps display
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

                # palm = average of 0,1,5,9,13,17 as in your code
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
            # no hands detected
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
    # Batch processing wrapper
    # ---------------------------------------------------------
    def process_directory(
        self,
        input_dir,
        output_dir="output_dataframes",
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
            if out_path:
                results.append(out_path)

        return results
