"""
C5 - TemplateMatcher
======================
Assigns unmatched spikes to nearest cluster template via euclidean distance.

Exact port of template_match() + get_means() + distances_euclidean() from dist.py:

    def template_match(spikes, sort_idx, match_idx, factor):
        num_samples = spikes.shape[1]
        unmatched_idx = sort_idx == CLID_UNMATCHED
        class_ids = np.unique(sort_idx[~unmatched_idx])
        if not len(class_ids): return
        ids, mean_array, stds = get_means(sort_idx, spikes)
        if options['ExcludeVariableClustersMatch']:
            median_std = np.median(stds)
            std_too_high_idx = stds > 3 * median_std
            mean_array = mean_array[~std_too_high_idx]
            ids = ids[~std_too_high_idx]
            stds = stds[~std_too_high_idx]
        all_distances = distances_euclidean(spikes[unmatched_idx], mean_array)
        all_distances[all_distances > factor * stds] = np.inf
        minimizers_idx = all_distances.argmin(1)
        minimizers = ids[minimizers_idx]
        minima = all_distances.min(1)
        minimizers[minima >= options['FirstMatchMaxDist'] * num_samples] = CLID_UNMATCHED
        sort_idx[unmatched_idx] = minimizers
        match_idx[unmatched_idx] = minimizers

Key differences from C5b (TotalMatcher):
    - Conservative: uses FirstMatchFactor (default 3.0) and FirstMatchMaxDist (default 1.4)
    - Called inside each iteration of iterative_sorter
    - match_idx is boolean in original (any nonzero assignment = True)
    - Supports ExcludeVariableClustersMatch option

Input:
    spikes:    (N, 64) tensor — all spike waveforms
    sort_idx:  (N,)    numpy array — cluster assignments (0 = unassigned)
    match_idx: (N,)    numpy bool array — which spikes were template-matched
    factor:    float — distance threshold multiplier

Output:
    Modifies sort_idx and match_idx in-place (returned for convenience)
"""

import torch
import torch.nn as nn
import numpy as np

# mirrors options in combinato/__init__.py
CLID_UNMATCHED              = 0
FIRST_MATCH_FACTOR          = 0.75
FIRST_MATCH_MAX_DIST        = 4
EXCLUDE_VARIABLE_CLUSTERS   = True


class TemplateMatcher(nn.Module):
    """
    C5: Assign unmatched spikes to nearest cluster template.

    Exact port of template_match() from dist.py.
    Vectorized euclidean distance computation on GPU.

    Args:
        first_match_factor (float): default distance factor. Default 3.0.
        first_match_max_dist (float): absolute max distance multiplier. Default 1.4.
        exclude_variable_clusters (bool): exclude high-variance clusters. Default False.
        clid_unmatched (int): label for unmatched spikes. Default 0.
    """

    def __init__(self,
                 first_match_factor=FIRST_MATCH_FACTOR,
                 first_match_max_dist=FIRST_MATCH_MAX_DIST,
                 exclude_variable_clusters=EXCLUDE_VARIABLE_CLUSTERS,
                 clid_unmatched=CLID_UNMATCHED):
        super().__init__()
        self.first_match_factor         = first_match_factor
        self.first_match_max_dist       = first_match_max_dist
        self.exclude_variable_clusters  = exclude_variable_clusters
        self.clid_unmatched             = clid_unmatched

    def get_means(self, sort_idx, spikes):
        """
        Compute mean waveform and std for each cluster.
        Mirrors get_means() in dist.py exactly.

        Returns:
            ids:        (K,) array of cluster ids
            mean_array: (K, 64) array of mean waveforms
            std_array:  (K,) array of per-cluster stds
        """
        ids   = []
        means = []
        stds  = []

        for clid in np.unique(sort_idx):
            if clid == self.clid_unmatched:
                continue
            meandata = spikes[sort_idx == clid]
            if meandata.shape[0]:
                ids.append(clid)
                means.append(meandata.mean(0))
                stds.append(np.sqrt(meandata.var(0).sum()))

        if not len(means):
            return np.array([]), np.array([]), np.array([])

        return (np.array(ids),
                np.vstack(means),
                np.array(stds))

    def distances_euclidean(self, spikes_t, templates_t):
        """
        Vectorized euclidean distance — fully on GPU if available.
        Mirrors distances_euclidean() in dist.py but vectorized.

        Args:
            spikes_t:    (M, 64) tensor — unmatched spikes
            templates_t: (K, 64) tensor — cluster templates

        Returns:
            dists: (M, K) tensor
        """
        # (M, 1, 64) - (1, K, 64) -> (M, K, 64) -> (M, K)
        diff  = spikes_t.unsqueeze(1) - templates_t.unsqueeze(0)
        dists = torch.sqrt((diff ** 2).sum(dim=2))
        return dists

    def forward(self, spikes, sort_idx, match_idx, factor=None):
        """
        Mirrors template_match(spikes, sort_idx, match_idx, factor) exactly.

        The original modifies sort_idx and match_idx in-place.
        This version also modifies them in-place AND returns them.

        Args:
            spikes:    (N, 64) tensor or numpy array
            sort_idx:  (N,) numpy uint16 array — modified in-place
            match_idx: (N,) numpy bool array — modified in-place
            factor:    float — distance multiplier. Default self.first_match_factor.

        Returns:
            None (modifies in-place, like the original)
        """
        if factor is None:
            factor = self.first_match_factor

        if isinstance(spikes, torch.Tensor):
            spikes_np = spikes.cpu().numpy()
            device    = spikes.device
        else:
            spikes_np = spikes
            device    = torch.device('cpu')

        num_samples   = spikes_np.shape[1]
        unmatched_idx = sort_idx == self.clid_unmatched

        # original: class_ids = np.unique(sort_idx[~unmatched_idx])
        # if not len(class_ids): return
        class_ids = np.unique(sort_idx[~unmatched_idx])
        if not len(class_ids):
            return

        # get cluster templates
        ids, mean_array, stds = self.get_means(sort_idx, spikes_np)
        if not len(ids):
            return

        # original: ExcludeVariableClustersMatch
        if self.exclude_variable_clusters:
            median_std = np.median(stds)
            std_too_high_idx = stds > 3 * median_std
            mean_array = mean_array[~std_too_high_idx]
            ids        = ids[~std_too_high_idx]
            stds       = stds[~std_too_high_idx]

        if not len(ids):
            return

        # move to device for vectorized distance computation
        unmatched_spikes_t = torch.tensor(
            spikes_np[unmatched_idx], dtype=torch.float32).to(device)
        templates_t = torch.tensor(
            mean_array, dtype=torch.float32).to(device)
        stds_t = torch.tensor(stds, dtype=torch.float32).to(device)

        # vectorized euclidean distances: (M, K)
        all_distances = self.distances_euclidean(unmatched_spikes_t, templates_t)

        # mask distances beyond factor * std
        all_distances[all_distances > factor * stds_t.unsqueeze(0)] = float('inf')

        # find nearest cluster
        min_vals, minimizers_idx = all_distances.min(dim=1)
        minimizers = ids[minimizers_idx.cpu().numpy()]

        # reject if absolute distance too large
        minima   = min_vals.cpu().numpy()
        minimizers[minima >= self.first_match_max_dist * num_samples] = \
            self.clid_unmatched

        # modify in-place — exactly as original
        sort_idx[unmatched_idx]  = minimizers
        match_idx[unmatched_idx] = minimizers


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("C5 TEMPLATE MATCHER — VALIDATION")
    print("=" * 60)

    np.random.seed(42)
    N  = 500
    L  = 64

    # simulate 3 clusters + unmatched spikes
    spikes_np = np.zeros((N, L), dtype=np.float32)
    sort_idx  = np.zeros(N, dtype=np.uint16)

    # cluster 1: peak at index 10
    spikes_np[:150, 8:12]  = np.hanning(4) * 50
    sort_idx[:150]         = 1

    # cluster 2: peak at index 30
    spikes_np[150:300, 28:32] = np.hanning(4) * 40
    sort_idx[150:300]         = 2

    # cluster 3: peak at index 50
    spikes_np[300:400, 48:52] = np.hanning(4) * 45
    sort_idx[300:400]         = 3

    # unmatched: similar to cluster 1
    spikes_np[400:, 8:12] = np.hanning(4) * 48
    spikes_np += np.random.randn(N, L).astype(np.float32) * 2
    # sort_idx[400:] stays 0 = unmatched

    print(f"\nInput: {N} spikes, {(sort_idx==0).sum()} unmatched")

    # --- Original dist.py logic ---
    print("\n[ORIGINAL] template_match...")

    def orig_match(spikes, sort_idx, factor=3.0, max_dist=1.4):
        sort_idx  = sort_idx.copy()
        match_idx = np.zeros(len(sort_idx), dtype=bool)
        unmatched = sort_idx == 0
        ids, means, stds = [], [], []
        for clid in np.unique(sort_idx[~unmatched]):
            mask = sort_idx == clid
            ids.append(clid)
            means.append(spikes[mask].mean(0))
            stds.append(np.sqrt(spikes[mask].var(0).sum()))
        ids   = np.array(ids)
        means = np.vstack(means)
        stds  = np.array(stds)
        # distances_euclidean loop
        dists = np.empty((spikes[unmatched].shape[0], means.shape[0]))
        for i, template in enumerate(means):
            dists[:, i] = np.sqrt(((spikes[unmatched] - template)**2).sum(1))
        dists[dists > factor * stds] = np.inf
        min_idx  = dists.argmin(1)
        assigned = ids[min_idx]
        min_vals = dists.min(1)
        assigned[min_vals >= max_dist * spikes.shape[1]] = 0
        sort_idx[unmatched]  = assigned
        match_idx[unmatched] = assigned  # nonzero -> True
        return sort_idx, match_idx

    orig_sort_idx = sort_idx.copy()
    orig_sort, orig_match = orig_match(spikes_np, orig_sort_idx)
    print(f"  Unmatched after : {(orig_sort==0).sum()}")
    print(f"  Matched (bool)  : {orig_match.sum()}")

    # --- PyTorch C5 ---
    print("\n[PYTORCH] C5 TemplateMatcher...")
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = TemplateMatcher().to(device)
    spikes_t = torch.tensor(spikes_np).to(device)

    pt_sort_idx  = sort_idx.copy()
    pt_match_idx = np.zeros(N, dtype=bool)

    with torch.no_grad():
        matcher(spikes_t, pt_sort_idx, pt_match_idx, factor=3.0)

    print(f"  Unmatched after : {(pt_sort_idx==0).sum()}")
    print(f"  Matched (bool)  : {pt_match_idx.sum()}")

    # --- Validate ---
    print("\n[VALIDATION]")
    sort_match  = np.array_equal(orig_sort, pt_sort_idx)
    # compare boolean match: original sets match_idx[unmatched] = minimizers,
    # which on a bool array becomes True for any nonzero
    match_match = np.array_equal(orig_match, pt_match_idx)
    print(f"  sort_idx match  : {'✓' if sort_match else '✗'}")
    print(f"  match_idx match : {'✓' if match_match else '✗'}")

    print("\n" + "="*60)
    print("C5 PASSED ✓" if sort_match and match_match else "C5 FAILED ✗")
    print("="*60)
