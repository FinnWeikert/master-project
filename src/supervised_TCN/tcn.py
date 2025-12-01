import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
#                     TCN Block
# ---------------------------------------------------

class TemporalBlock(nn.Module):
    """
    Standard TCN block with:
    - Dilated Conv1D
    - ReLU
    - Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout=0.2):
        super().__init__()

        # First dilated convolution
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1×1 residual projection if needed
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) \
            if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.downsample]:
            if conv is not None:
                nn.init.normal_(conv.weight, mean=0.0, std=0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu2(out + res)  # Residual merge


# ---------------------------------------------------
#                Masked TCN Model
# ---------------------------------------------------

# ---------------------------------------------------
#                Masked TCN Model
# ---------------------------------------------------

class SurgicalTCN(nn.Module):
    """
    TCN for surgical skill assessment (scalar regression).
    
    Input:
        x: (B, C_total, T) where C_total = C_kinematic + 1 (mask)
        
    Output:
        score : (B, 1)
    """
    def __init__(self, num_inputs=6, num_channels=[32, 32, 32], kernel_size=3, dropout=0.4):
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            # "Same" padding for dilated convolution
            padding = (kernel_size - 1) * dilation // 2

            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

        # Regression head: Kaiming initialization is not strictly necessary here, 
        # but standard initialization for regression/classification is common.
        self.linear = nn.Linear(num_channels[-1], 1)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def masked_global_avg_pool(self, y, mask):
        """
        Calculates the average over the time dimension (T) only for non-masked entries.
        y:    (B, C, T)
        mask: (B, 1, T)
        """
        # Expand mask to match channel dimensions for element-wise multiplication
        mask_expanded = mask.float().expand_as(y)

        # Weighted sum over time (numerator)
        numerator = (y * mask_expanded).sum(dim=2)            
        
        # Count of valid frames (denominator)
        denominator = mask.sum(dim=2) + 1e-6 # (B, 1) to avoid divide-by-zero

        return numerator / denominator # (B, C)

    def forward(self, x):
        """
        x: (B, C_total, T)
           Assumes the LAST channel (index -1) is the binary mask.
        """
        # 1. Split Kinematics and Mask (Doctrine: handle input complexity internally)
        kinematics = x[:, :-1, :] # (B, C_kinematic, T)
        mask = x[:, -1:, :]      # (B, 1, T)
        
        # if the sum of the mask is zero along any batch item break
        if (mask.sum(dim=2) < mask.size(2)/2).any():
            d = 1

        # 2. Pass Kinematics through TCN
        y = self.network(kinematics)  # (B, hidden, T)

        # 3. Apply Masked Pooling
        pooled = self.masked_global_avg_pool(y, mask)
        
        # 4. Regression
        score = self.linear(pooled)  # (B, 1)
        
        return score




class SurgicalTCNEncoder(nn.Module):
    """
    Feature Extractor version of SurgicalTCN.
    
    Input:
        x: (B, C_total, T)
        
    Output:
        embedding: (B, Output_Channel_Dim) 
                   A single vector representation for each window.
    """
    def __init__(self, num_inputs=6, num_channels=[32, 32, 32], kernel_size=3, dropout=0.4):
        super().__init__()
        
        # Expose output dim for the MIL model to know
        self.output_dim = num_channels[-1]

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            padding = (kernel_size - 1) * dilation // 2

            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def masked_global_avg_pool(self, y, mask):
        """
        Same pooling logic as original.
        """
        mask_expanded = mask.float().expand_as(y)
        numerator = (y * mask_expanded).sum(dim=2)            
        denominator = mask.sum(dim=2) + 1e-6 
        return numerator / denominator 

    def forward(self, x):
        """
        x: (B, C_total, T)
        """
        # 1. Split Kinematics and Mask
        kinematics = x[:, :-1, :] 
        mask = x[:, -1:, :]      
        
        # 2. Pass Kinematics through TCN
        # Output shape: (B, num_channels[-1], T)
        y = self.network(kinematics)  

        # 3. Apply Masked Pooling to get Window Embedding
        # Output shape: (B, num_channels[-1])
        embedding = self.masked_global_avg_pool(y, mask)
        
        return embedding

    def load_supervised_weights(self, supervised_model_path, device="cpu"):
        """
        Helper to load weights from the trained SurgicalTCN.
        It uses strict=False to ignore the missing 'linear' layer.
        """
        # Load the checkpoint
        state_dict = torch.load(supervised_model_path, map_location=device)
        
        # If saved as a full dictionary (e.g. {'model_state': ..., 'optimizer': ...})
        # extract just the model weights. Adjust key as needed.
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Load weights
        # strict=False is KEY here. It tells PyTorch:
        # "Load the matching layers (network.*) and ignore that I don't have 'linear.*'"
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        print(f"Weights loaded. \nMissing keys (expected): {missing} \nUnexpected keys: {unexpected}")