"""
C5b - TotalMatcher
====================
Second template matching pass — more aggressive than C5.

Exact port of total_match() from concatenate.py:

    def total_match(fid, all_spikes):
        classes = fid.root.classes[:]
        ids, mean_array, stds = get_means(classes, all_spikes)
        unmatched_idx = (classes == CLID_UNMATCHED).nonzero()[0]
        blocksize = 50*1000
        n_unmatched = unmatched_idx.shape[0]
        starts = np.arange(0, n_unmatched, blocksize)
        if not len(starts):
            starts = np.array([0])
            stops = np.array([n_unmatched])
        else:
            stops = starts + blocksize
            stops[-1] = n_unmatched
        for start, stop in zip(starts, stops):
            this_idx = unmatched_idx[start:stop]
            all_dists = distances_euclidean(all_spikes[this_idx], mean_array)
            all_dists[all_dists > SecondMatchFactor * stds] = np.inf
            minimizers_idx = all_dists.argmin(1)
            minimizers = ids[minimizers_idx]
            minima = all_dists.min(1)
            minimizers[minima >= SecondMatchMaxDist * num_samples] = 0
            fid.root.classes[this_idx] = minimizers
            fid.root.matches[this_idx] = SPIKE_MATCHED_2   # ALL processed, not just successful
            fid.root.distance[this_idx] = minima

Differences from C5 (first match):
    - Larger max distance (1.9 vs 1.4) — more aggressive
    - Processes in batches of 50,000 for memory efficiency
    - Sets match type to SPIKE_MATCHED_2 = 2 for ALL processed spikes
    - Stores distance for all processed spikes

Input:
    spikes:    (N, 64) tensor
    sort_idx:  (N,) numpy array — current assignments (0 = unmatched)
    match_idx: (N,) numpy int8 array — current match types (0=SPC, 1=C5)

Output:
    sort_idx:  (N,) numpy array — updated, fewer zeros
    match_idx: (N,) numpy int8 array — 2 for ALL processed spikes
    distance:  (N,) numpy float32 array — distances for processed spikes
"""

import torch
import torch.nn as nn
import numpy as np

# mirrors combinato/__init__.py
CLID_UNMATCHED        = 0
SPIKE_MATCHED_2       = 2
SECOND_MATCH_FACTOR   = 3
SECOND_MATCH_MAX_DIST = 20
BLOCKSIZE             = 50000


class TotalMatcher(nn.Module):
    """
    C5b: Second template matching pass from concatenate.py.

    Exact port of total_match(). More aggressive than C5 — uses
    larger distance threshold and processes in batches.

    IMPORTANT: The original sets matches=SPIKE_MATCHED_2 for ALL
    processed spikes, even ones that remain unmatched (classes=0).
    This is replicated exactly here.

    Args:
        second_match_factor (float): max dist = factor * std. Default 3.0.
        second_match_max_dist (float): absolute max dist. Default 1.9.
        blocksize (int): batch size for processing. Default 50000.
        clid_unmatched (int): unmatched label. Default 0.
    """

    def __init__(self,
                 second_match_factor=SECOND_MATCH_FACTOR,
                 second_match_max_dist=SECOND_MATCH_MAX_DIST,
                 blocksize=BLOCKSIZE,
                 clid_unmatched=CLID_UNMATCHED):
        super().__init__()
        self.second_match_factor   = second_match_factor
        self.second_match_max_dist = second_match_max_dist
        self.blocksize             = blocksize
        self.clid_unmatched        = clid_unmatched

    def get_means(self, sort_idx, spikes_np):
        """
        Compute mean waveform and std per cluster.
        Mirrors get_means() in dist.py exactly.
        """
        ids   = []
        means = []
        stds  = []
        for clid in np.unique(sort_idx):
            if clid == self.clid_unmatched:
                continue
            meandata = spikes_np[sort_idx == clid]
            if meandata.shape[0]:
                ids.append(clid)
                means.append(meandata.mean(0))
                stds.append(np.sqrt(meandata.var(0).sum()))
        if not len(means):
            return np.array([]), np.array([]), np.array([])
        return np.array(ids), np.vstack(means), np.array(stds)

    def distances_euclidean_batch(self, spikes_t, templates_t):
        """
        Vectorized euclidean distance on GPU.
        (M, 64) vs (K, 64) -> (M, K)
        """
        diff = spikes_t.unsqueeze(1) - templates_t.unsqueeze(0)
        return torch.sqrt((diff ** 2).sum(dim=2))

    def forward(self, spikes, sort_idx, match_idx, distance=None):
        """
        Mirrors total_match(fid, all_spikes) from concatenate.py exactly.

        Args:
            spikes:    (N, 64) tensor
            sort_idx:  (N,) numpy array — modified in-place
            match_idx: (N,) numpy int8 array — modified in-place
            distance:  (N,) numpy float32 array — modified in-place (optional)

        Returns:
            sort_idx:  (N,) numpy array — updated
            match_idx: (N,) numpy int8 array — 2 for all processed spikes
            distance:  (N,) numpy float32 array — distances
        """
        if isinstance(spikes, torch.Tensor):
            spikes_np = spikes.cpu().numpy()
            device    = spikes.device
        else:
            spikes_np = spikes
            device    = torch.device('cpu')

        N = len(sort_idx)
        if distance is None:
            distance = np.zeros(N, dtype=np.float32)

        # get cluster templates ONCE before the loop (like original)
        ids, mean_array, stds = self.get_means(sort_idx, spikes_np)
        if not len(ids):
            return sort_idx, match_idx, distance

        unmatched_idx = (sort_idx == self.clid_unmatched).nonzero()[0]

        # mirrors the original's start/stop logic exactly
        n_unmatched = unmatched_idx.shape[0]
        starts = np.arange(0, n_unmatched, self.blocksize)
        if not len(starts):
            starts = np.array([0])
            stops  = np.array([n_unmatched])
        else:
            stops = starts + self.blocksize
            stops[-1] = n_unmatched

        templates_t = torch.tensor(mean_array, dtype=torch.float32).to(device)
        stds_t      = torch.tensor(stds,       dtype=torch.float32).to(device)
        num_samples = spikes_np.shape[1]

        for start, stop in zip(starts, stops):
            this_idx       = unmatched_idx[start:stop]
            block_spikes_t = torch.tensor(
                spikes_np[this_idx], dtype=torch.float32).to(device)

            # vectorized distances: (block, K)
            all_dists = self.distances_euclidean_batch(block_spikes_t, templates_t)

            # mask distances beyond factor * std
            all_dists[all_dists > self.second_match_factor * stds_t.unsqueeze(0)] = float('inf')

            # find nearest cluster
            min_vals, minimizers_idx = all_dists.min(dim=1)
            minimizers = ids[minimizers_idx.cpu().numpy()]
            minima     = min_vals.cpu().numpy()

            # reject if too far
            minimizers[minima >= self.second_match_max_dist * num_samples] = \
                self.clid_unmatched

            # update — matches original exactly:
            # fid.root.classes[this_idx] = minimizers
            # fid.root.matches[this_idx] = SPIKE_MATCHED_2   (ALL processed, not just successful)
            # fid.root.distance[this_idx] = minima
            sort_idx[this_idx]  = minimizers
            match_idx[this_idx] = SPIKE_MATCHED_2   # ALL processed spikes, even rejected
            distance[this_idx]  = minima

        still_unmatched = (sort_idx == self.clid_unmatched).sum()
        print(f"    TotalMatcher: {still_unmatched} still unmatched after second pass")

        return sort_idx, match_idx, distance


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("C5b TOTAL MATCHER — VALIDATION")
    print("=" * 60)

    np.random.seed(42)
    N = 500; L = 64

    spikes_np = np.zeros((N, L), dtype=np.float32)
    sort_idx  = np.zeros(N, dtype=np.uint16)
    match_idx = np.zeros(N, dtype=np.int8)

    for cid, center, n in [(1, 10, 150), (2, 30, 150), (3, 50, 100)]:
        s = sum([150, 150, 100][:cid-1])
        e = s + [150, 150, 100][cid-1]
        spikes_np[s:e, center-5:center+5] = np.hanning(10) * (40 + cid*5)
        sort_idx[s:e] = cid

    spikes_np += np.random.randn(N, L).astype(np.float32) * 2

    # leave 100 unmatched — similar to cluster 1
    sort_idx[-100:] = 0
    spikes_np[-100:, 8:15] = np.hanning(7) * 42
    spikes_np[-100:] += np.random.randn(100, L).astype(np.float32) * 2

    print(f"\nBefore: unmatched={(sort_idx==0).sum()}")

    device  = torch.device('cpu')
    matcher = TotalMatcher()
    spikes_t = torch.tensor(spikes_np).to(device)

    with torch.no_grad():
        new_sort, new_match, new_dist = matcher(spikes_t, sort_idx, match_idx)

    print(f"After : unmatched={(new_sort==0).sum()}")
    print(f"match=2 count: {(new_match == 2).sum()}")
    print(f"Nonzero distances: {(new_dist > 0).sum()}")

    print("\n[VALIDATION]")
    fewer_unmatched = (new_sort == 0).sum() < 100
    match2_all      = (new_match == 2).sum() == 100  # ALL 100 processed
    has_distances   = (new_dist > 0).sum() > 0
    print(f"  Fewer unmatched     : {'✓' if fewer_unmatched else '✗'}")
    print(f"  match=2 for all 100 : {'✓' if match2_all else '✗'}")
    print(f"  Distances stored    : {'✓' if has_distances else '✗'}")

    print("\n" + "=" * 60)
    all_pass = fewer_unmatched and match2_all and has_distances
    print("C5b PASSED ✓" if all_pass else "C5b FAILED ✗")
    print("=" * 60)
