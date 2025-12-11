import torch
import numpy as np

###########################################################################
################ helper functions for MIL dataset creation ################
###########################################################################

def mil_collate_fn(batch):
    """
    Custom collate function to handle the unique sample_key tuple.
    
    Input: batch (list of tuples: [(window_tensor, score_tensor, sample_key), ...])
    Output: (collated_windows, collated_scores, sample_keys_list)
    """
    # 1. Separate the components
    windows, scores, keys = zip(*batch)
    
    # 2. Collate the tensors (default behavior)
    windows = torch.stack(windows, 0)
    scores = torch.stack(scores, 0)
    
    # 3. CRUCIAL: Keep the keys as a simple list of tuples
    # 'keys' is already a list of tuples due to the zip(*batch) operation.
    keys_list = list(keys)
    
    return windows, scores, keys_list

def generate_mil_bags_cpu(data_loader, encoder):
    """
    Processes the window-level DataLoader to generate video-level bag embeddings (CPU-Only).
    
    Expected DataLoader return: (window, score, sample_key)
    (sample_key is the tuple: (video_id, surgeon_id))
    """
    grouped_data = {}
    encoder.to("cpu") 

    with torch.no_grad():
        # Unpack the batch to get windows, scores, AND the unique keys
        # Note the loop now unpacks 3 items: (windows, scores, batch_keys)
        for windows, scores, batch_keys in data_loader:
            
            # Ensure input tensors are on CPU
            windows = windows.to("cpu")
            
            # Generate the window embeddings (Instance Features)
            # embeddings_batch shape: (B, 32)
            embeddings_batch = encoder(windows)

            # --- Grouping Logic (Using the sample_key tuple as the dictionary key) ---
            for j in range(embeddings_batch.size(0)):
                # batch_keys[j] is the tuple (video_id, surgeon_id)
                key = batch_keys[j]
                
                score = scores[j].item()
                embedding = embeddings_batch[j].cpu().numpy()
                
                if key not in grouped_data:
                    # Initialize storage for the new video bag
                    grouped_data[key] = {
                        'embeddings': [],
                        'score': score,
                    }
                
                grouped_data[key]['embeddings'].append(embedding)

    # --- Final List Construction (The extracted_data list) ---
    extracted_data = []
    
    # Iterate over the unique video/surgeon keys
    for key, data in grouped_data.items():
        # Stack all window embeddings for a video into a single Bag tensor
        # Bag tensor shape: (N_windows, 32)
        bag_tensor = torch.tensor(np.stack(data['embeddings'], axis=0), dtype=torch.float32)
        score_tensor = torch.tensor(data['score'], dtype=torch.float32).unsqueeze(0)
        
        # The output tuple for the MILBagDataset: (Bag, Score, Unique_Key)
        extracted_data.append((bag_tensor, score_tensor, key))
        
    return extracted_data