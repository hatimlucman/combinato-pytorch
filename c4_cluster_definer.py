"""
C4 - ClusterDefiner
=====================
Reads SPC temperature tree and extracts final cluster assignments.
"""

import numpy as np
import torch
from block import Block

MAX_CLUSTERS_PER_TEMP = 5
MIN_SPIKES_PER_CLUSTER = 15


class ClusterDefiner(Block):
    def __init__(self,
                 max_clusters_per_temp=MAX_CLUSTERS_PER_TEMP,
                 min_spikes=MIN_SPIKES_PER_CLUSTER):
        super().__init__()
        self.max_clusters_per_temp = max_clusters_per_temp
        self.min_spikes            = min_spikes

    def find_relevant_tree_points(self, tree):
        ret = []
        for shift in range(self.max_clusters_per_temp):
            col_idx = 5 + shift
            if col_idx >= tree.shape[1]:
                break
            col   = tree[:, col_idx]
            rise  = (col[1:] > col[:-1]).nonzero()[0] + 1
            fall  = (col[:-1] >= col[1:]).nonzero()[0]
            peaks = set(rise.tolist()) & set(fall.tolist())
            if 1 in fall.tolist():
                peaks.add(1)
            for peak in peaks:
                nspk = tree[peak, col_idx]
                if nspk >= self.min_spikes:
                    ret.append((peak, nspk, shift + 1))
        return ret

    def forward(self, clu, tree):
        relevant_rows = self.find_relevant_tree_points(tree)
        num_features  = clu.shape[1] - 2
        idx           = np.zeros(num_features, dtype=np.uint8)
        used_points   = []
        current_id    = 2
        max_row       = 0

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

        if len(used_points):
            row_idx = clu[max_row, 2:] == 0
            used_points.append((max_row, 4, 'm'))
        else:
            row_idx = clu[1, 2:] == 0
            used_points.append((1, 4, 'c'))

        idx[row_idx] = 1
        return idx, tree, used_points
