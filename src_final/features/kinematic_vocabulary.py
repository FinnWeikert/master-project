import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans 
from sklearn.preprocessing import RobustScaler
from scipy.special import softmax

class KinematicVocabulary_old:
    def __init__(self, n_clusters=16, transform_method='log', clip=5.0, random_state=42, model_type='gmm', n_init=5,
                 feature_cols=None, log_feats=['total_path', 'path_ratio', 'spatial_spread', 'sparc']):
        self.n_clusters = n_clusters
        self.transform_method = transform_method
        self.clip = clip
        self.random_state = random_state
        self.model_type = model_type
        
        self.scaler = RobustScaler()
        
        if model_type == 'gmm':
            self.model = GaussianMixture(
                n_components=n_clusters, 
                covariance_type='diag', 
                random_state=random_state,
                n_init=n_init
            )
        elif model_type == 'kmeans':
            self.model = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=random_state, 
                n_init=n_init,
                batch_size=1024
            )
        else:
            raise ValueError("model_type must be 'gmm' or 'kmeans'")
        
        self.feature_cols = feature_cols if feature_cols is not None else ['total_path', 'path_ratio', 'spatial_spread', 'sparc', 'palm_area_cv', 'zvr']
        self.log_feats = log_feats

    def _preprocess(self, df, fit=False):
        X = df[self.feature_cols].copy()
        if 'sparc' in X.columns:
            X['sparc'] = -X['sparc']
            
        if self.transform_method == 'log':
            for col in self.log_feats:
                X[col] = np.log1p(np.abs(X[col]))
                
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return np.clip(X_scaled, -self.clip, self.clip)

    def fit(self, df_train):
        X_processed = self._preprocess(df_train, fit=True)
        self.model.fit(X_processed)
        return self

    def transform(self, df):
        video_features = []
        vids = df['video_id'].unique()
        
        for vid in vids:
            df_vid = df[df['video_id'] == vid]
            X_vid = self._preprocess(df_vid, fit=False)
            
            if self.model_type == 'gmm':
                # Natural posterior probabilities
                probs = self.model.predict_proba(X_vid)
            
            else:  # K-Means Soft Assignment
                # .transform() returns Euclidean distance to each centroid
                distances = self.model.transform(X_vid) 
                
                # Convert distance to "probability" using a Softmax over negative distances
                # We use a small sigma (temperature) to control how 'soft' the assignment is.
                # A high sigma makes the BoW more uniform; a low sigma makes it like hard K-means.
                sigma = 1.0 
                probs = softmax(-distances / sigma, axis=1)
            
            # Aggregate: Mean of window-wise SurgeMe probabilities = Video-level SurgeMe Profile
            bow_vector = np.mean(probs, axis=0)
            video_features.append(bow_vector)
            
        cols = [f'SurgeMe_{i}' for i in range(self.n_clusters)]
        return pd.DataFrame(video_features, columns=cols, index=vids)


import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import RobustScaler, PowerTransformer
from scipy.special import softmax


class KinematicVocabulary:
    def __init__(
        self,
        n_clusters=16,
        clip=5.0,
        random_state=42,
        model_type='gmm',
        n_init=5,
        feature_cols=None,
        power_feats=None,
        log_feats=None  # Ignored if transform_method is not 'log'
    ):
        self.n_clusters = n_clusters
        self.clip = clip
        self.random_state = random_state
        self.model_type = model_type
        
        self.scaler = RobustScaler()
        
        # Power transformer (only used if requested)
        self.power_feats = power_feats
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)

        # Log transformer (only used if requested)
        self.log_feats = log_feats

        if model_type == 'gmm':
            self.model = GaussianMixture(
                n_components=n_clusters,
                covariance_type='diag',
                random_state=random_state,
                n_init=n_init
            )
        elif model_type == 'kmeans':
            self.model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                batch_size=1024
            )
        else:
            raise ValueError("model_type must be 'gmm' or 'kmeans'")
        
        self.feature_cols = (
            feature_cols
            if feature_cols is not None
            else ['total_path', 'path_ratio', 'spatial_spread',
                  'sparc', 'palm_area_cv', 'zvr']
        )

    def _preprocess(self, df, fit=False):
        X = df[self.feature_cols].copy()

        # Optional sign flip
        if 'sparc' in X.columns:
            X['sparc'] = -X['sparc']

        # --- Yeo-Johnson transform ---
        if self.power_feats is not None:
            pow_cols = [c for c in self.power_feats if c in X.columns]

            if len(pow_cols) > 0:
                if fit:
                    X[pow_cols] = self.power_transformer.fit_transform(
                        X[pow_cols]
                    )
                else:
                    X[pow_cols] = self.power_transformer.transform(
                        X[pow_cols]
                    )
        
        # --- Log transform ---
        if self.log_feats is not None:
            log_cols = [c for c in self.log_feats if c in X.columns]

            for col in log_cols:
                X[col] = np.log1p(np.abs(X[col]))

        # --- Scaling ---
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return np.clip(X_scaled, -self.clip, self.clip)

    def fit(self, df_train):
        X_processed = self._preprocess(df_train, fit=True)
        self.model.fit(X_processed)
        return self

    def transform(self, df):
        video_features = []
        vids = df['video_id'].unique()
        
        for vid in vids:
            df_vid = df[df['video_id'] == vid]
            X_vid = self._preprocess(df_vid, fit=False)
            
            if self.model_type == 'gmm':
                probs = self.model.predict_proba(X_vid)
            else:
                distances = self.model.transform(X_vid)
                sigma = 1.0
                probs = softmax(-distances / sigma, axis=1)
            
            bow_vector = np.mean(probs, axis=0)
            video_features.append(bow_vector)
            
        cols = [f'SurgeMe_{i}' for i in range(self.n_clusters)]
        return pd.DataFrame(video_features, columns=cols, index=vids)