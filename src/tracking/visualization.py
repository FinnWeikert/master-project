# path overlays on video

# visualization.py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Static hand trajectory plot
# -----------------------------
def draw_hands(df_hand_left, df_hand_right, title="Hand Trajectories"):
    """
    Draws processed hand positions in the video plane (1920x1080) with connected dots.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(np.zeros((1080, 1920, 3), dtype=np.uint8))
    
    # Draw left hand path by segment
    for _, seg_data in df_hand_left.groupby('segment_id'):
        plt.plot(seg_data['cx_smooth'], seg_data['cy_smooth'], color='green', linewidth=0.5)
        plt.scatter(seg_data['cx_smooth'], seg_data['cy_smooth'], color='green', s=0.5, label='_nolegend_')

    # Draw right hand path by segment
    for _, seg_data in df_hand_right.groupby('segment_id'):
        plt.plot(seg_data['cx_smooth'], seg_data['cy_smooth'], color='red', linewidth=0.5)
        plt.scatter(seg_data['cx_smooth'], seg_data['cy_smooth'], color='red', s=0.5, label='_nolegend_')
    
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.title(title)

    # Legend
    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='Left Hand'),
        Line2D([0], [0], color='red', lw=2, label='Right Hand')
    ]
    plt.legend(handles=legend_handles)
    plt.show()


# -----------------------------
# Video annotation with trajectories
# -----------------------------
def draw_hand_trajectories(video_path, df_processed=None, fps=30, tail_length=30, velocity_overlay=False):
    """
    Annotates a video with hand trajectories (tail) and velocity overlay.

    Parameters
    ----------
    video_path : str
        Path to input video (.mp4)
    fps : int
        Which processed dataframe fps to use (10 or 30)
    tail_length : int
        Number of frames to show behind current frame
    velocity_overlay : bool                                 
        !!!!! For now not compatible at this stage in this form !!!!!
        If True, color lines by velocity
    """
    # --- Load processed dataframe ---
    base_name = os.path.basename(video_path).replace('.mp4', '')
    if df_processed is None:
        processed_dir = 'data/processed'
        pkl_path = os.path.join(processed_dir, f'hand_tracking_{base_name}_{fps}fps_processed.pkl')

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Processed dataframe not found: {pkl_path}")
        
        df = pd.read_pickle(pkl_path)
    else:
        df = df_processed

    # Compute velocity
    #dt = 1.0 / fps
    #df['velocity'] = df['disp_filtered'] / dt
    #df['velocity'] = df['velocity'].replace([np.inf, -np.inf], np.nan)

    # Setup video IO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path = 'data/experiments/' + os.path.basename(video_path).replace('.mp4', f'_annotated_path_{fps}fps.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))

    # Velocity color mapping
    default_color = (0, 255, 0)
    if velocity_overlay:
        valid_vel = df['velocity'].dropna()
        max_vel = valid_vel.abs().quantile(0.95) if not valid_vel.empty else 1.0

        def get_color_from_velocity(v):
            norm_speed = np.clip(abs(v)/max_vel, 0, 1)
            return (int(255*(1-norm_speed)), 0, int(255*norm_speed))  # BGR

    # Process video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Annotating video: {video_path}")

    with tqdm(total=total_frames, unit="frames", desc="Processing Frames") as pbar:
        for current_frame in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            for hand_label in ['Right', 'Left']:
                tail_start = max(0, current_frame - tail_length)
                path_data = df[(df['hand_label'] == hand_label) & 
                               (df['frame'] >= tail_start) & 
                               (df['frame'] <= current_frame)].sort_values('frame')

                if len(path_data) > 1:
                    for i in range(1, len(path_data)):
                        if path_data['segment_id'].iloc[i] != path_data['segment_id'].iloc[i-1]:
                            continue
                        p1 = (int(path_data['cx_smooth'].iloc[i-1]), int(path_data['cy_smooth'].iloc[i-1]))
                        p2 = (int(path_data['cx_smooth'].iloc[i]), int(path_data['cy_smooth'].iloc[i]))
                        color = get_color_from_velocity(path_data['velocity'].iloc[i]) if velocity_overlay else default_color
                        cv2.line(frame, p1, p2, color, 2)
                        cv2.circle(frame, p2, 3, color, -1)

            # Frame counter overlay
            cv2.putText(frame, f'Frame: {current_frame}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Saved annotated video: {output_video_path}")
    return output_video_path


def compare_trajectories(video_path, df_30_fps=None, df_10_fps=None, tail_length=30):
    """
    Overlays and compares hand trajectories from both 10 fps and 30 fps processed dataframes 
    on the original video.

    Args:
        video_path (str): Path to the input video file (e.g., 'example.mp4').
        tail_length (int): Number of previous frames to trace for the hand's path (tail).

    Returns:
        str: Path to the saved comparison video file.
    """
    
    # --- 1. Load and Prepare Dataframes ---
    
    data_sources = {}
    
    # 10 fps Data
    if df_10_fps is not None:
        data_sources['10fps'] = {'df': df_10_fps, 'color': (255, 0, 0)} # Blue
    else:
        pkl_path_10 = os.path.join('data/processed', 'hand_tracking_' + video_path.split('/')[-1].replace('.mp4', '_10fps_processed.pkl'))
        if os.path.exists(pkl_path_10):
            df_10 = pd.read_pickle(pkl_path_10)
            data_sources['10fps'] = {'df': df_10, 'color': (255, 0, 0)} # Blue
        else:
            print(f"Warning: 10 fps dataframe not found at: {pkl_path_10}")

    # 30 fps Data
    if df_30_fps is not None:
        data_sources['30fps'] = {'df': df_30_fps, 'color': (0, 0, 255)} # Red
    pkl_path_30 = os.path.join('data/processed', 'hand_tracking_' + video_path.split('/')[-1].replace('.mp4', '_30fps_processed.pkl'))
    if os.path.exists(pkl_path_30):
        df_30 = pd.read_pickle(pkl_path_30)
        data_sources['30fps'] = {'df': df_30, 'color': (0, 0, 255)} # Red
    else:
        print(f"Warning: 30 fps dataframe not found at: {pkl_path_30}")

    if not data_sources:
        raise FileNotFoundError("Neither 10 fps nor 30 fps dataframes were found. Cannot compare.")

    # --- 2. Setup Video I/O ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path = 'data/experiments/' + os.path.basename(video_path).replace('.mp4', '_comparison.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    # --- 3. Process Video Frame by Frame (with TQDM) ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting comparison annotation for video: {video_path}")
    
    # Define a single color function for the velocity-less comparison
    # We use fixed colors (Blue for 10fps, Red for 30fps) instead of velocity coding
    # to clearly distinguish the two tracking rates.
    
    with tqdm(total=total_frames, unit="frames", desc="Processing Frames") as pbar: 
        current_video_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw trajectories for EACH processed rate (10fps and 30fps)
            for label, data in data_sources.items():
                df_rate = data['df']
                line_color = data['color'] # Blue or Red
                
                for hand_label in ['Right', 'Left']:
                    tail_start_frame = max(0, current_video_frame - tail_length)
                    
                    # Query data for the current rate and hand within the tail range
                    path_data = df_rate[
                        (df_rate['hand_label'] == hand_label) & 
                        (df_rate['frame'] >= tail_start_frame) & 
                        (df_rate['frame'] <= current_video_frame)
                    ].sort_values(by='frame')

                    if len(path_data) > 1:
                        # Draw the Tail Trace
                        for i in range(1, len(path_data)):
                            if path_data['segment_id'].iloc[i] != path_data['segment_id'].iloc[i-1]:
                                continue
                                
                            p1 = (int(path_data['cx_smooth'].iloc[i-1]), int(path_data['cy_smooth'].iloc[i-1]))
                            p2 = (int(path_data['cx_smooth'].iloc[i]), int(path_data['cy_smooth'].iloc[i]))
                            
                            # Draw the line using the fixed color for the rate
                            cv2.line(frame, p1, p2, line_color, 2)
                            cv2.circle(frame, p2, 3, line_color, -1)
                            
                            # Add a label near the point (optional, but helpful for debugging)
                            # if i == len(path_data) - 1:
                            #     cv2.putText(frame, f'{label[0]}', p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)

            # --- 4. Add Legend (for clarity) ---
            # Define legend positions
            cv2.line(frame, (20, 30), (50, 30), data_sources.get('10fps', {}).get('color', (100, 100, 100)), 3)
            cv2.putText(frame, '10 fps', (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.line(frame, (20, 60), (50, 60), data_sources.get('30fps', {}).get('color', (100, 100, 100)), 3)
            cv2.putText(frame, '30 fps', (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # ---- Frame Display ----
            cv2.putText(frame, f'Frame: {current_video_frame}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # Write the annotated frame
            out.write(frame)
            current_video_frame += 1
            pbar.update(1)

    # --- 5. Cleanup and Return ---
    cap.release()
    out.release()
    
    return output_video_path


def draw_hand_trajectories_new(video_path, df_processed=None, fps=30, tail_length=30):
    """
    Annotates a video with trajectories for ALL smoothed landmarks found in the dataframe.
    """
    # --- 1. Load Data ---
    base_name = os.path.basename(video_path).replace('.mp4', '')
    if df_processed is None:
        pkl_path = f'data/processed/landmark_dataframes/hand_tracking_{base_name}_{fps}fps_processed.pkl'
        df = pd.read_pickle(pkl_path)
    else:
        df = df_processed.copy()

    # --- 2. Setup Landmarks & Colors ---
    # Find all smoothed coordinate columns (e.g., lm_0_x_smooth)
    x_cols = [c for c in df.columns if c.endswith('_x_smooth')]
    landmark_ids = [c.split('_')[1] for c in x_cols]

    # Assign a fixed color palette for specific MediaPipe IDs
    # BGR format
    landmark_colors = {
        '0': (0, 255, 0),    # Wrist: Green
        '5': (255, 255, 0),  # Index MCP: Cyan
        '17': (0, 255, 255), # Pinky MCP: Yellow
    }
    # Fallback for any other landmarks
    default_palette = [(255, 0, 0), (255, 0, 255), (0, 165, 255)] # Blue, Magenta, Orange

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
                # Filter data for the "tail" window
                tail_start = max(0, current_frame - tail_length)
                path_data = df[(df['hand_label'] == hand_label) & 
                               (df['frame'] >= tail_start) & 
                               (df['frame'] <= current_frame)].sort_values('frame')

                if len(path_data) < 2:
                    continue

                for idx, lm_id in enumerate(landmark_ids):
                    x_col = f'lm_{lm_id}_x_smooth'
                    y_col = f'lm_{lm_id}_y_smooth'
                    
                    # Select color based on ID or index
                    color = landmark_colors.get(lm_id, default_palette[idx % len(default_palette)])

                    # Draw the segments of the tail
                    for i in range(1, len(path_data)):
                        # Break line if it belongs to a different tracking segment
                        if path_data['segment_id'].iloc[i] != path_data['segment_id'].iloc[i-1]:
                            continue
                            
                        p1 = (int(path_data[x_col].iloc[i-1]), int(path_data[y_col].iloc[i-1]))
                        p2 = (int(path_data[x_col].iloc[i]), int(path_data[y_col].iloc[i]))
                        
                        # Draw line (thicker as it reaches the current frame)
                        thickness = 1 if i < len(path_data) - 1 else 2
                        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
                        
                        # Draw a circle at the very head of the trajectory
                        if i == len(path_data) - 1:
                            cv2.circle(frame, p2, 4, color, -1, cv2.LINE_AA)

            # Overlay Frame Number
            cv2.putText(frame, f'Frame: {current_frame}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_path}")
    return output_path



def compare_trajectories_new(video_path, df_30_fps=None, df_10_fps=None, tail_length=30):
    """
    Overlays all smoothed landmark trajectories found in the dataframes onto the video.
    """
    # --- 1. Load Data (Keeping your existing logic) ---
    data_sources = {}
    if df_10_fps is not None:
        data_sources['10fps'] = {'df': df_10_fps, 'base_color': (255, 0, 0)} # Blue tones
    else:
        pkl_path_10 = os.path.join('data/processed', video_path.split('/')[-1].replace('.mp4', '_10fps_processed.pkl'))
        if os.path.exists(pkl_path_10):
            df_10 = pd.read_pickle(pkl_path_10)
            data_sources['10fps'] = {'df': df_10, 'base_color': (255, 0, 0)} # Blue
        else:
            print(f"Warning: 10 fps dataframe not found at: {pkl_path_10}")

    if df_30_fps is not None:
        data_sources['30fps'] = {'df': df_30_fps, 'base_color': (0, 0, 255)} # Red tones
    else:
        pkl_path_30 = os.path.join('data/processed', video_path.split('/')[-1].replace('.mp4', '_30fps_processed.pkl'))
        if os.path.exists(pkl_path_30):
            df_30 = pd.read_pickle(pkl_path_30)
            data_sources['30fps'] = {'df': df_30, 'base_color': (0, 0, 255)} # Red
        else:
            print(f"Warning: 30 fps dataframe not found at: {pkl_path_30}")
    if not data_sources:
        raise FileNotFoundError("Neither 10 fps nor 30 fps dataframes were found. Cannot compare.")

    # --- 2. Setup Video I/O ---
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = 'data/experiments/' + os.path.basename(video_path).replace('.mp4', '_comparison.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))

    # --- 3. Process Video ---
    with tqdm(total=total_frames, unit="frames", desc="Annotating All Landmarks") as pbar:
        current_video_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            for label, data in data_sources.items():
                df_rate = data['df']
                base_col = data['base_color']
                
                # Identify all pairs of smoothed coordinate columns
                # Finds 'lm_0_x_smooth' and assumes a corresponding 'lm_0_y_smooth'
                x_cols = [c for c in df_rate.columns if c.endswith('_x_smooth')]
                
                for x_col in x_cols:
                    y_col = x_col.replace('_x_smooth', '_y_smooth')
                    lm_id = x_col.split('_')[1] # Extracts the ID for specific coloring
                    
                    # Create a unique color for this specific landmark to avoid a "mess"
                    # We slightly shift the base color based on the landmark ID
                    lm_color = (
                        max(0, min(255, base_col[0] + int(lm_id) * 10)),
                        max(0, min(255, base_col[1] + int(lm_id) * 5)),
                        max(0, min(255, base_col[2] - int(lm_id) * 10))
                    )

                    for hand_label in ['Right', 'Left']:
                        tail_start = max(0, current_video_frame - tail_length)
                        path_data = df_rate[
                            (df_rate['hand_label'] == hand_label) & 
                            (df_rate['frame'] >= tail_start) & 
                            (df_rate['frame'] <= current_video_frame)
                        ].sort_values(by='frame')

                        if len(path_data) > 1:
                            # Draw segments
                            for i in range(1, len(path_data)):
                                # Ensure we don't draw lines across different segments (gaps)
                                if path_data['segment_id'].iloc[i] != path_data['segment_id'].iloc[i-1]:
                                    continue
                                
                                p1 = (int(path_data[x_col].iloc[i-1]), int(path_data[y_col].iloc[i-1]))
                                p2 = (int(path_data[x_col].iloc[i]), int(path_data[y_col].iloc[i]))
                                
                                # Dim the tail, brighten the head
                                alpha = i / len(path_data)
                                current_color = tuple([int(c * alpha) for c in lm_color])
                                
                                cv2.line(frame, p1, p2, current_color, 2)
                                if i == len(path_data) - 1: # Current position
                                    cv2.circle(frame, p2, 4, lm_color, -1)

            # --- 4. Legend & Info ---
            cv2.putText(frame, f'Frame: {current_video_frame}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Blue shades: 10fps | Red shades: 30fps', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            out.write(frame)
            current_video_frame += 1
            pbar.update(1)

    cap.release()
    out.release()
    return output_video_path