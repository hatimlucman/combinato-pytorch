"""
C5b - TotalMatcher
====================
Second template matching pass — more aggressive than C5.
"""

import torch
import numpy as np
from block import Block

CLID_UNMATCHED      = 0
SPIKE_MATCHED_2     = 2
SECOND_MATCH_FACTOR = 3
SECOND_MATCH_MAX_DIST = 20
BLOCKSIZE           = 50000


class TotalMatcher(Block):
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
        ids, means, stds = [], [], []
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
        diff = spikes_t.unsqueeze(1) - templates_t.unsqueeze(0)
        return torch.sqrt((diff ** 2).sum(dim=2))

    def forward(self, spikes, sort_idx, match_idx, distance=None):
        if isinstance(spikes, torch.Tensor):
            spikes_np = spikes.cpu().numpy()
            device    = spikes.device
        else:
            spikes_np = spikes
            device    = torch.device('cpu')

        N = len(sort_idx)
        if distance is None:
            distance = np.zeros(N, dtype=np.float32)

        ids, mean_array, stds = self.get_means(sort_idx, spikes_np)
        if not len(ids):
            return sort_idx, match_idx, distance

        unmatched_idx = (sort_idx == self.clid_unmatched).nonzero()[0]
        n_unmatched   = unmatched_idx.shape[0]
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
            block_spikes_t = torch.tensor(spikes_np[this_idx], dtype=torch.float32).to(device)

            all_dists = self.distances_euclidean_batch(block_spikes_t, templates_t)
            all_dists[all_dists > self.second_match_factor * stds_t.unsqueeze(0)] = float('inf')

            min_vals, minimizers_idx = all_dists.min(dim=1)
            minimizers = ids[minimizers_idx.cpu().numpy()]
            minima     = min_vals.cpu().numpy()
            minimizers[minima >= self.second_match_max_dist * num_samples] = self.clid_unmatched

            sort_idx[this_idx]  = minimizers
            match_idx[this_idx] = SPIKE_MATCHED_2
            distance[this_idx]  = minima

        still_unmatched = (sort_idx == self.clid_unmatched).sum()
        print(f"    TotalMatcher: {still_unmatched} still unmatched after second pass")

        return sort_idx, match_idx, distance
