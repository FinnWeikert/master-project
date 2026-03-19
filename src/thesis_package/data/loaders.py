import os
import pandas as pd
from tqdm import tqdm

import os
import pandas as pd
from tqdm import tqdm

import os
import pandas as pd
from tqdm import tqdm

def get_vid_id(filename, suffix="_30fps_processed.pkl"):
    return filename.replace('hand_tracking_', '').replace(suffix, '')

def get_eligible_files(
    processed_dir="data/processed/landmark_dataframes2/",
    vid_name_map_path="data/scores/vid_name_map.csv",
    exclude_participants=[8],
    suffix="_30fps_processed.pkl"
):
    """
    Returns a list of (vid_id, participant_id, full_path) for files 
    that pass the exclusion filters. 
    Does NOT load the data into RAM yet.
    """
    df_map = pd.read_csv(vid_name_map_path, index_col=0)
    all_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(suffix)])
    
    eligible = []
    for f in all_files:
        vid_id = get_vid_id(f, suffix)
        if vid_id in df_map.index:
            part_id = int(df_map.loc[vid_id]['Participant Number'])
            if part_id not in exclude_participants:
                eligible.append((vid_id, part_id, os.path.join(processed_dir, f)))
    return eligible

def load_processed_dict(eligible_files):
    """Bulk loads a dictionary. Use only if RAM allows."""
    return { (vid, pid): pd.read_pickle(path) for vid, pid, path in tqdm(eligible_files, desc="Bulk Loading") }

def load_scores_df(scores_csv="data/scores/merged_scores.csv"):
    return pd.read_csv(scores_csv)