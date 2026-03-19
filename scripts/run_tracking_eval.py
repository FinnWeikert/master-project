"""
Tracking Evaluation Script for Thesis Reproduction.
Calculates coverage, trajectory continuity, and interpolation metrics 
for raw and processed hand-tracking data.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_filtered_files(path, fps_str='30fps', year_str='2024'):
    """Fetch and filter pkl files based on thesis exclusion criteria."""
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist.")
        return []
        
    files = [f for f in os.listdir(path) 
             if fps_str in f and year_str in f and f.endswith('.pkl')]
    
    # Thesis Exclusions: Remove specific sessions with known hardware/sync issues
    exclude_sessions = [
        '2024-01-17_17-09-36', 
        '2024-01-17_18-24-28', 
        '2024-01-17_18-43-42'
    ]
    
    for sess in exclude_sessions:
        files = [f for f in files if sess not in f]
    return sorted(files)

def run_evaluation(raw_dir, proc_dir, fps=30, min_seg_sec=1.5):
    """Core logic to compute tracking metrics."""
    raw_files = get_filtered_files(raw_dir)
    proc_files = get_filtered_files(proc_dir)
    
    min_seg_frames = int(min_seg_sec * fps)
    
    # 1. Raw Data Analysis
    raw_cov_l, raw_cov_r = [], []
    print(f"Analyzing {len(raw_files)} raw files...")
    for f in tqdm(raw_files):
        df = pd.read_pickle(os.path.join(raw_dir, f))
        total_f = df['frame'].max() - df['frame'].min() + 1
        
        # Note: Swapped labels in raw data to correct MediaPipe handedness error
        raw_cov_r.append(df[df['hand_label'] == 'Left']['frame'].nunique() / total_f)
        raw_cov_l.append(df[df['hand_label'] == 'Right']['frame'].nunique() / total_f)

    # 2. Processed Data Analysis
    stats = {k: [] for k in ['cov_l', 'cov_r', 'traj_l', 'traj_r', 'intp_l', 'intp_r']}
    
    print(f"Analyzing {len(proc_files)} processed files...")
    for f in tqdm(proc_files):
        df = pd.read_pickle(os.path.join(proc_dir, f))
        total_f = df['frame'].max() - df['frame'].min() + 1
        
        for side, key in [('Left', 'l'), ('Right', 'r')]:
            df_s = df[df['hand_label'] == side]
            if len(df_s) == 0: continue
            
            # Coverage
            stats[f'cov_{key}'].append(df_s['frame'].nunique() / total_f)
            
            # Trajectory Continuity (> 1.5s)
            segs = df_s.groupby('segment_id').size()
            stats[f'traj_{key}'].append(segs[segs >= min_seg_frames].sum() / df_s['frame'].nunique())
            
            # Interpolation Rate
            stats[f'intp_{key}'].append(df_s['hand_score'].isna().sum() / len(df_s))

    # 3. Aggregate Results
    metrics = [
        ("Raw Coverage (L)", raw_cov_l), ("Raw Coverage (R)", raw_cov_r),
        ("Proc. Coverage (L)", stats['cov_l']), ("Proc. Coverage (R)", stats['cov_r']),
        ("Cont. Traj. (L)", stats['traj_l']), ("Cont. Traj. (R)", stats['traj_r']),
        ("Interp. Rate (L)", stats['intp_l']), ("Interp. Rate (R)", stats['intp_r'])
    ]
    
    print("\n" + "="*65)
    print(f"{'Metric':<25} | {'Mean':<12} | {'Std Dev':<12}")
    print("-" * 65)
    for name, data in metrics:
        if data:
            print(f"{name:<25} | {np.mean(data):>10.2%} | {np.std(data):>10.2%}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hand tracking quality.")
    parser.add_argument("--raw_path", type=str, default="data/raw/output_dataframes/")
    parser.add_argument("--proc_path", type=str, default="data/processed/landmark_dataframes/")
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    run_evaluation(args.raw_path, args.proc_path, fps=args.fps)