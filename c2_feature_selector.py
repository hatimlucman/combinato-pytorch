"""
C2 - FeatureSelector
=====================
Selects the most informative wavelet features for clustering.
"""

import torch
import numpy as np
from scipy import stats
from block import Block

FEATURE_FACTOR = 3
N_FEATURES_OUT = 10


class FeatureSelector(Block):
    def __init__(self, feature_factor=FEATURE_FACTOR, n_features_out=N_FEATURES_OUT):
        super().__init__()
        self.feature_factor = feature_factor
        self.n_features_out = n_features_out

    def compute_scores(self, features_np):
        N, F      = features_np.shape
        scores    = np.zeros(F)
        feat_std  = self.feature_factor * features_np.std(0)
        feat_mean = features_np.mean(0)
        feat_up   = feat_mean + feat_std
        feat_down = feat_mean - feat_std

        for i in range(F):
            idx = ((features_np[:, i] > feat_down[i]) &
                   (features_np[:, i] < feat_up[i]))
            if idx.any():
                good = features_np[idx, i]
                good = good - good.mean()
                std  = good.std()
                if std > 0:
                    good = good / std
                    scores[i] = stats.kstest(good, 'norm')[1]

        sorted_scores = np.sort(scores)
        border        = sorted_scores[self.n_features_out]
        indices       = (scores <= border).nonzero()[0][:self.n_features_out]
        return scores, indices

    def forward(self, features):
        features_np     = features.cpu().numpy()
        scores, indices = self.compute_scores(features_np)
        selected_np     = features_np[:, indices]
        selected = torch.tensor(selected_np, dtype=torch.float32, device=features.device)
        idx_t    = torch.tensor(indices,     dtype=torch.long,    device=features.device)
        return selected, idx_t
