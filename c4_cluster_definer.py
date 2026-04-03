"""
C4 - ClusterDefiner
=====================
Reads SPC temperature tree and extracts final cluster assignments.

Replaces define_clusters.py:
    def find_relevant_tree_points(tree, min_spikes):
        for shift in range(max_clusters_per_temp):
            col = tree[:, 5+shift]
            rise = (col[1:] > col[:-1]).nonzero()[0] + 1
            fall = (col[:-1] >= col[1:]).nonzero()[0]
            peaks = set(rise) & set(fall)
            if 1 in fall: peaks.add(1)
            for peak in peaks:
                if tree[peak, col_idx] >= min_spikes:
                    ret.append((peak, nspk, shift+1))

    def define_clusters(clu, tree):
        relevant_rows = find_relevant_tree_points(tree, min_spikes)
        idx = np.zeros(num_features, dtype=np.uint8)
        for row, _, col in relevant_rows:
            row_idx = (clu[row, 2:] == col) & (idx == 0)
            if row_idx.any():
                idx[row_idx] = current_id
                current_id += 1
        # assign remaining to biggest cluster at highest used temperature
        row_idx = clu[max_row, 2:] == 0
        idx[row_idx] = 1
        return idx, tree, used_points

Input:
    clu:  numpy array — from C3, cluster labels at each temperature
    tree: numpy array — from C3, temperature tree

Output:
    idx:         (N,) numpy array — cluster assignment per spike
    tree:        numpy array — passed through
    used_points: list — temperature points used for selection
"""

import numpy as np
import torch
import torch.nn as nn

# mirrors options in combinato/__init__.py
MAX_CLUSTERS_PER_TEMP           = 5
MIN_SPIKES_PER_CLUSTER          = 15
FRACTION_OF_BIGGEST_CLUSTER     = 0.1


class ClusterDefiner(nn.Module):
    """
    C4: Extract cluster assignments from SPC temperature tree.

    Scans the tree for stable cluster peaks across temperatures,
    assigns spikes to clusters, remaining spikes go to cluster 1
    (biggest cluster at highest used temperature).

    Args:
        max_clusters_per_temp (int): max clusters to look for per temp. Default 5.
        min_spikes (int): minimum spikes for a valid cluster. Default 15.
    """

    def __init__(self,
                 max_clusters_per_temp=MAX_CLUSTERS_PER_TEMP,
                 min_spikes=MIN_SPIKES_PER_CLUSTER):
        super().__init__()
        self.max_clusters_per_temp = max_clusters_per_temp
        self.min_spikes            = min_spikes

    def find_relevant_tree_points(self, tree):
        """
        Find temperature points where stable clusters appear.
        Mirrors find_relevant_tree_points() exactly.

        Returns list of (row, nspikes, col_shift) tuples.
        """
        ret = []
        for shift in range(self.max_clusters_per_temp):
            col_idx = 5 + shift
            if col_idx >= tree.shape[1]:
                break
            col  = tree[:, col_idx]

            # peaks: bigger than predecessor AND at least as big as successor
            rise  = (col[1:] > col[:-1]).nonzero()[0] + 1
            fall  = (col[:-1] >= col[1:]).nonzero()[0]
            peaks = set(rise.tolist()) & set(fall.tolist())

            # special case: falling line at beginning
            if 1 in fall.tolist():
                peaks.add(1)

            for peak in peaks:
                nspk = tree[peak, col_idx]
                if nspk >= self.min_spikes:
                    ret.append((peak, nspk, shift + 1))

        return ret

    def forward(self, clu, tree):
        """
        Args:
            clu:  numpy array (T, N+2) — from C3
            tree: numpy array          — from C3

        Returns:
            idx:         (N,) numpy array — cluster index per spike (0 = unassigned)
            tree:        numpy array — passed through
            used_points: list of (row, col, color) for plotting
        """
        relevant_rows = self.find_relevant_tree_points(tree)
        num_features  = clu.shape[1] - 2
        idx           = np.zeros(num_features, dtype=np.uint8)

        used_points = []
        current_id  = 2
        max_row     = 0

        for row, _, col in relevant_rows:
            row_idx = (clu[row, 2:] == col) & (idx == 0)
            if row_idx.any():
                idx[row_idx] = current_id
                current_id  += 1
                p_type       = 'k'
                max_row      = max(max_row, row)
            else:
                p_type = 'r'
            used_points.append((row, col + 4, p_type))

        # assign remaining spikes to biggest cluster at highest used temperature
        if len(used_points):
            row_idx = clu[max_row, 2:] == 0
            used_points.append((max_row, 4, 'm'))
        else:
            row_idx = clu[1, 2:] == 0
            used_points.append((1, 4, 'c'))

        idx[row_idx] = 1

        return idx, tree, used_points


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("C4 CLUSTER DEFINER — VALIDATION (synthetic tree)")
    print("=" * 60)

    # simulate SPC output tree and clu
    np.random.seed(42)
    T = 21   # temperatures
    N = 200  # spikes

    # synthetic tree: [temp_idx, temp_val, nspk_total, ?, ?, col1, col2, ...]
    tree = np.zeros((T, 10))
    tree[:, 1] = np.linspace(0, 0.2, T)
    tree[:, 4] = N  # total spikes
    # simulate a cluster appearing at temp 5-10 in col 5
    tree[5:10, 5] = 50
    tree[3:5, 5]  = 30

    # synthetic clu: rows=temperatures, cols=[?, ?, spike_labels...]
    clu = np.zeros((T, N + 2))
    # at temp 5, 50 spikes are in cluster 1
    for t in range(3, 10):
        clu[t, 2:52] = 1

    print("\n[PYTORCH] C4 ClusterDefiner...")
    definer  = ClusterDefiner(min_spikes=15)
    pt_idx, _, pt_pts = definer(clu, tree)
    print(f"  Unique classes : {np.unique(pt_idx)}")
    print(f"  Used points    : {len(pt_pts)}")

    print("\n" + "="*60)
    print("C4 DONE")
    print("="*60)
