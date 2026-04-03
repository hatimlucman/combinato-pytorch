"""
C5 - TemplateMatcher
======================
Assigns unmatched spikes to nearest cluster template via euclidean distance.
"""

import torch
import numpy as np
from block import Block

CLID_UNMATCHED            = 0
FIRST_MATCH_FACTOR        = 0.75
FIRST_MATCH_MAX_DIST      = 4
EXCLUDE_VARIABLE_CLUSTERS = True


class TemplateMatcher(Block):
    def __init__(self,
                 first_match_factor=FIRST_MATCH_FACTOR,
                 first_match_max_dist=FIRST_MATCH_MAX_DIST,
                 exclude_variable_clusters=EXCLUDE_VARIABLE_CLUSTERS,
                 clid_unmatched=CLID_UNMATCHED):
        super().__init__()
        self.first_match_factor        = first_match_factor
        self.first_match_max_dist      = first_match_max_dist
        self.exclude_variable_clusters = exclude_variable_clusters
        self.clid_unmatched            = clid_unmatched

    def get_means(self, sort_idx, spikes):
        ids, means, stds = [], [], []
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
        return np.array(ids), np.vstack(means), np.array(stds)

    def distances_euclidean(self, spikes_t, templates_t):
        diff  = spikes_t.unsqueeze(1) - templates_t.unsqueeze(0)
        return torch.sqrt((diff ** 2).sum(dim=2))

    def forward(self, spikes, sort_idx, match_idx, factor=None):
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
        class_ids     = np.unique(sort_idx[~unmatched_idx])
        if not len(class_ids):
            return

        ids, mean_array, stds = self.get_means(sort_idx, spikes_np)
        if not len(ids):
            return

        if self.exclude_variable_clusters:
            median_std       = np.median(stds)
            std_too_high_idx = stds > 3 * median_std
            mean_array = mean_array[~std_too_high_idx]
            ids        = ids[~std_too_high_idx]
            stds       = stds[~std_too_high_idx]

        if not len(ids):
            return

        unmatched_spikes_t = torch.tensor(spikes_np[unmatched_idx], dtype=torch.float32).to(device)
        templates_t        = torch.tensor(mean_array, dtype=torch.float32).to(device)
        stds_t             = torch.tensor(stds,       dtype=torch.float32).to(device)

        all_distances = self.distances_euclidean(unmatched_spikes_t, templates_t)
        all_distances[all_distances > factor * stds_t.unsqueeze(0)] = float('inf')

        min_vals, minimizers_idx = all_distances.min(dim=1)
        minimizers = ids[minimizers_idx.cpu().numpy()]
        minima     = min_vals.cpu().numpy()
        minimizers[minima >= self.first_match_max_dist * num_samples] = self.clid_unmatched

        sort_idx[unmatched_idx]  = minimizers
        match_idx[unmatched_idx] = minimizers
