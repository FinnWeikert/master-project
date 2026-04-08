from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
import pandas as pd
import numpy as np

class MILFeatureScaler:
    def __init__(self, feature_cols, log_features=None, method='robust'):
        """
        feature_cols: List of all features to be scaled.
        log_features: List of specific features to apply log(1+x) transformation to.
        method: 'robust', 'quantile', or 'standard'.
        """
        self.feature_cols = feature_cols
        self.log_features = log_features if log_features is not None else []
        self.method = method
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif method == 'standard':
            self.scaler = StandardScaler()
            
    def _apply_log(self, df):
        """Internal helper to apply log transformation."""
        df_log = df.copy()
        for col in self.log_features:
            # np.log1p computes ln(1 + x) to handle zeros safely
            df_log[col] = np.log1p(df_log[col].abs())
        return df_log

    def fit(self, df_all_windows, verbose=True):
        """Fit the scaler on training windows after log-transforming subsets."""
        # 1. Apply log transform to the subset
        df_transformed = self._apply_log(df_all_windows[self.feature_cols])
        
        # 2. Extract values and fit the scaler
        X = df_transformed.values
        self.scaler.fit(X)
        if verbose:
            print(f"Scaler ({self.method}) fitted on {len(X)} windows with {len(self.log_features)} log-transformed features.")
        
    def transform(self, df_windows):
        """Transform windows for training or inference."""
        # 1. Apply log transform to the subset
        df_transformed = self._apply_log(df_windows[self.feature_cols])
        
        # 2. Transform the features
        X = df_transformed.values
        X_scaled = self.scaler.transform(X)
        
        # 3. Clip for numerical stability in the Neural Net
        X_scaled = np.clip(X_scaled, -7.5, 7.5)
        
        # Create a copy of the original to return, but updated with scaled values
        df_out = df_windows.copy()
        df_out[self.feature_cols] = X_scaled
        return df_out