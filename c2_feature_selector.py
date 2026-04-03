"""
C2 - FeatureSelector
=====================
Selects the most informative wavelet features for clustering.

Replaces select_features.py:
    def select_features(features):
        factor = options['feature_factor']        # 3
        num_features_out = options['nFeatures']   # 10
        feat_std  = factor * features.std(0)
        feat_mean = features.mean(0)
        feat_up   = feat_mean + feat_std
        feat_down = feat_mean - feat_std
        scores = np.zeros(num_features)
        for i in range(num_features):
            idx = (features[:,i] > feat_down[i]) & (features[:,i] < feat_up[i])
            if idx.any():
                good = features[idx,i]
                good = (good - good.mean()) / good.std()
                scores[i] = stats.kstest(good, 'norm')[1]
        border = np.sort(scores)[num_features_out]
        ret = (scores <= border).nonzero()[0][:num_features_out]
        return ret

Logic:
    1. For each feature, take only values within mean ± 3*std (remove outliers)
    2. Normalize those values to zero mean, unit std
    3. Run KS test against standard normal — p-value is the score
    4. LOW p-value = non-Gaussian = more informative for clustering
    5. Keep the 10 features with lowest p-values

Input:
    features: (N, 73) tensor — from C1

Output:
    selected: (N, 10) tensor — most informative features
    indices:  (10,)   tensor — which feature columns were selected
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats


# mirrors options in combinato __init__.py
FEATURE_FACTOR   = 3
N_FEATURES_OUT   = 10


class FeatureSelector(nn.Module):
    """
    C2: Select most informative wavelet features via KS normality test.

    Features that deviate most from Gaussian distribution are most
    discriminative for clustering — neurons create non-Gaussian
    structure in feature space.

    Args:
        feature_factor (int): std multiplier for outlier removal. Default 3.
        n_features_out (int): number of features to keep. Default 10.
    """

    def __init__(self, feature_factor=FEATURE_FACTOR, n_features_out=N_FEATURES_OUT):
        super().__init__()
        self.feature_factor = feature_factor
        self.n_features_out = n_features_out

    def compute_scores(self, features_np):
        """
        Compute KS test score for each feature.
        Low p-value = non-Gaussian = more informative.

        Args:
            features_np: (N, F) numpy array

        Returns:
            scores: (F,) numpy array of p-values
            indices: (n_features_out,) selected feature indices
        """
        N, F   = features_np.shape
        scores = np.zeros(F)

        feat_std  = self.feature_factor * features_np.std(0)   # (F,)
        feat_mean = features_np.mean(0)                         # (F,)
        feat_up   = feat_mean + feat_std
        feat_down = feat_mean - feat_std

        for i in range(F):
            # keep only values within mean ± factor*std
            idx = ((features_np[:, i] > feat_down[i]) &
                   (features_np[:, i] < feat_up[i]))
            if idx.any():
                good = features_np[idx, i]
                good = good - good.mean()
                std  = good.std()
                if std > 0:
                    good = good / std
                    scores[i] = stats.kstest(good, 'norm')[1]

        # sort scores, pick border at n_features_out-th value
        sorted_scores = np.sort(scores)
        border        = sorted_scores[self.n_features_out]
        indices       = (scores <= border).nonzero()[0][:self.n_features_out]
        return scores, indices

    def forward(self, features):
        """
        Args:
            features: (N, F) tensor — from C1

        Returns:
            selected: (N, 10) tensor
            indices:  (10,)   tensor
        """
        features_np       = features.cpu().numpy()
        scores, indices   = self.compute_scores(features_np)
        selected_np       = features_np[:, indices]

        selected = torch.tensor(selected_np, dtype=torch.float32,
                                device=features.device)
        idx_t    = torch.tensor(indices, dtype=torch.long,
                                device=features.device)
        return selected, idx_t


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("C2 FEATURE SELECTOR — VALIDATION")
    print("=" * 60)

    np.random.seed(42)
    N = 500; F = 73
    # simulate features with some non-Gaussian columns
    features_np = np.random.randn(N, F).astype(np.float32)
    features_np[:, [5, 15, 25]] += np.random.exponential(2, (N, 3)).astype(np.float32)

    # --- Original ---
    print("\n[ORIGINAL] select_features...")
    factor = 3; n_out = 10
    feat_std  = factor * features_np.std(0)
    feat_mean = features_np.mean(0)
    scores_orig = np.zeros(F)
    for i in range(F):
        idx = ((features_np[:, i] > feat_mean[i] - feat_std[i]) &
               (features_np[:, i] < feat_mean[i] + feat_std[i]))
        if idx.any():
            good = features_np[idx, i]
            good = (good - good.mean())
            if good.std() > 0:
                good = good / good.std()
                scores_orig[i] = stats.kstest(good, 'norm')[1]
    sorted_s = np.sort(scores_orig)
    border   = sorted_s[n_out]
    orig_idx = (scores_orig <= border).nonzero()[0][:n_out]
    orig_sel = features_np[:, orig_idx]
    print(f"  Selected indices : {orig_idx}")
    print(f"  Output shape     : {orig_sel.shape}")

    # --- PyTorch C2 ---
    print("\n[PYTORCH] C2 FeatureSelector...")
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selector = FeatureSelector().to(device)
    x        = torch.tensor(features_np).to(device)

    with torch.no_grad():
        pt_sel, pt_idx = selector(x)

    pt_sel_np = pt_sel.cpu().numpy()
    pt_idx_np = pt_idx.cpu().numpy()
    print(f"  Selected indices : {pt_idx_np}")
    print(f"  Output shape     : {pt_sel_np.shape}")

    # --- Validate ---
    print("\n[VALIDATION]")
    idx_match   = np.array_equal(pt_idx_np, orig_idx)
    shape_match = pt_sel_np.shape == orig_sel.shape
    print(f"  Indices match : {'✓' if idx_match else '✗'}")
    print(f"  Shape match   : {orig_sel.shape} vs {pt_sel_np.shape}  {'✓' if shape_match else '✗'}")
    if shape_match:
        max_diff = np.max(np.abs(pt_sel_np - orig_sel))
        print(f"  Max diff      : {max_diff:.2e}  {'✓' if max_diff < 1e-5 else '✗'}")

    print(f"  Output cols   : {pt_sel_np.shape[1]} (expected {n_out})  {'✓' if pt_sel_np.shape[1] == n_out else '✗'}")

    print("\n" + "="*60)
    all_pass = idx_match and shape_match and max_diff < 1e-5
    print("C2 PASSED ✓  — Ready to build C3" if all_pass else "C2 FAILED ✗")
    print("="*60)
