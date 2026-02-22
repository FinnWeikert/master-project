import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class AttentionMILDataset(Dataset):
    def __init__(self, df_window_features, df_global_features, feature_cols, global_feature_cols, label_col='QRS_Overal'):
        """
        df_features: DataFrame with window-level features and labels
        feature_cols: List of columns to use as local features (windows)
        global_feature_cols: List of columns to use as global features (per video)
        label_col: Column name for the target label
        """
        self.df_window_features = df_window_features
        self.df_global_features = df_global_features
        self.feature_cols = feature_cols
        self.global_feature_cols = global_feature_cols
        self.label_col = label_col
        
        # Group by video_id to create bags
        self.video_ids = df_window_features['video_id'].unique()
        self.bags = []
        self.global_features = []
        self.labels = []
        
        for vid in tqdm(self.video_ids, desc="Preparing Bags"):
            df_vid = df_window_features[df_window_features['video_id'] == vid]
            bag = df_vid[feature_cols].values.astype(np.float32)
            global_feat = df_global_features[df_global_features['video_id'] == vid][global_feature_cols].iloc[0].values.astype(np.float32)
            label = df_global_features[df_global_features['video_id'] == vid][label_col].iloc[0].astype(np.float32)
            
            self.bags.append(bag)
            self.global_features.append(global_feat)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        global_feat = self.global_features[idx]
        label = self.labels[idx]
        vid_id = self.video_ids[idx]

        return torch.from_numpy(bag), torch.from_numpy(global_feat), torch.tensor(label), vid_id
    
    def mil_collate_fn(self,batch):
        """
        Custom collate function to handle variable-sized bags.
        Batch is a list of tuples: [(bag, global_feat, label, vid_id), ...]
        """
        bags = [item[0] for item in batch]        # List of Tensors (variable size)
        global_feats = torch.stack([item[1] for item in batch]) # Stacked Tensor (Batch, D_global)
        labels = torch.stack([item[2] for item in batch])  # Stacked Tensor (Batch, )
        vid_ids = [item[3] for item in batch]     # List of IDs
        
        return bags, global_feats, labels, vid_ids