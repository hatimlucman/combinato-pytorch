"""
C7 - CombinatorSorter
=======================
Full clustering pipeline — C1 through C6 + C5b wired together.
"""

import os
import time
import torch
import numpy as np
from block import Block

from c1_wavelet_features  import WaveletFeatureExtractor
from c2_feature_selector  import FeatureSelector
from c3_spc_clusterer     import SPCClusterer
from c4_cluster_definer   import ClusterDefiner
from c5_template_matcher  import TemplateMatcher
from c5b_total_matcher    import TotalMatcher
from c6_artifact_detector import ArtifactDetector

MIN_INPUT_SIZE           = 15
MIN_INPUT_SIZE_RECLUSTER = 2000
RECLUSTER_CLUSTERS       = True
RECURSIVE_DEPTH          = 1
FIRST_MATCH_FACTOR       = 0.75
MARK_ARTIFACT_CLASSES    = True
DEFAULT_SEED             = 12345.0


class CombinatorSorter(Block):
    def __init__(self,
                 cluster_path,
                 n_iterations=RECURSIVE_DEPTH,
                 recluster=RECLUSTER_CLUSTERS,
                 min_input_size=MIN_INPUT_SIZE,
                 min_input_size_recluster=MIN_INPUT_SIZE_RECLUSTER,
                 first_match_factor=FIRST_MATCH_FACTOR,
                 mark_artifacts=MARK_ARTIFACT_CLASSES,
                 seed=DEFAULT_SEED):
        super().__init__()
        self.n_iterations             = n_iterations
        self.recluster                = recluster
        self.min_input_size           = min_input_size
        self.min_input_size_recluster = min_input_size_recluster
        self.first_match_factor       = first_match_factor
        self.mark_artifacts           = mark_artifacts
        self.seed                     = seed

        self.c1  = WaveletFeatureExtractor()
        self.c2  = FeatureSelector()
        self.c3  = SPCClusterer(cluster_path=cluster_path)
        self.c4  = ClusterDefiner()
        self.c5  = TemplateMatcher()
        self.c5b = TotalMatcher()
        self.c6  = ArtifactDetector()

    def _cluster_step(self, features, folder, sub_name):
        with torch.no_grad():
            sel_features, feat_idx = self.c2(features)
        clu, tree = self.c3(sel_features, folder, sub_name, self.seed)
        res_idx, tree, used_points = self.c4(clu, tree)
        return res_idx

    def _iterative_sorter(self, all_features, spikes, folder):
        N         = all_features.shape[0]
        idx       = np.zeros(N, dtype=np.uint16)
        match_idx = np.zeros(N, dtype=bool)

        for i in range(self.n_iterations):
            sub_idx  = idx == 0
            sub_name = 'sort_' + str(i)

            if sub_idx.sum() < self.min_input_size:
                print(f"    Stopping iteration, {sub_idx.sum()} spikes left")
                break

            print(f"    Iteration {i}: clustering {sub_idx.sum()} spikes...")

            t0      = time.time()
            res_idx = self._cluster_step(all_features[sub_idx], folder, sub_name)
            print(f"      cluster_step done: {time.time()-t0:.1f}s  "
                  f"clusters={len(np.unique(res_idx[res_idx > 0]))}  "
                  f"unclustered={(res_idx==0).sum()}")

            clustered_idx = res_idx > 0
            prev_idx_max  = idx.max()
            res_idx[clustered_idx] += prev_idx_max
            idx[sub_idx] = res_idx

            if self.recluster:
                clids = np.unique(res_idx[clustered_idx])
                for clid in clids:
                    recluster_mask = idx == clid
                    cluster_size   = recluster_mask.sum()
                    if cluster_size < self.min_input_size_recluster:
                        continue
                    print(f"      Reclustering cluster {clid} ({cluster_size} spikes)")
                    sub_sub_name      = '{}_{:02d}'.format(sub_name, clid)
                    recluster_res_idx = self._cluster_step(
                        all_features[recluster_mask], folder, sub_sub_name)
                    biggest_clid = idx.max()
                    recluster_res_idx[recluster_res_idx != 0] += biggest_clid
                    idx[recluster_mask] = recluster_res_idx

            with torch.no_grad():
                self.c5(spikes, idx, match_idx, factor=self.first_match_factor)

            print(f"      After template match: unmatched={(idx==0).sum()}")

        return idx, match_idx

    def forward(self, spikes, folder, sign='pos'):
        N      = spikes.shape[0]
        device = spikes.device

        print(f"\n  [C7] Sorting {N} {sign} spikes...")

        if N < self.min_input_size:
            print(f"  [C7] Too few spikes ({N}), skipping")
            return (np.zeros(N, dtype=np.uint16),
                    np.zeros(N, dtype=np.int8),
                    np.zeros(N, dtype=np.float32),
                    [])

        t0 = time.time()
        with torch.no_grad():
            all_features = self.c1(spikes)
        print(f"  C1  features   : {tuple(all_features.shape)}  {time.time()-t0:.2f}s")

        t0 = time.time()
        sort_idx, bool_match_idx = self._iterative_sorter(all_features, spikes, folder)
        print(f"  Iterative sort : {time.time()-t0:.1f}s  unmatched={(sort_idx==0).sum()}")

        match_idx = bool_match_idx.astype(np.int8)

        if self.mark_artifacts:
            with torch.no_grad():
                art_scores, artifact_ids = self.c6(spikes, sort_idx, sign=sign)
            print(f"  C6  artifacts  : {len(artifact_ids)}/{len(art_scores)} flagged")
        else:
            artifact_ids = []

        distance = np.zeros(N, dtype=np.float32)
        with torch.no_grad():
            sort_idx, match_idx, distance = self.c5b(
                spikes, sort_idx, match_idx, distance)
        print(f"  C5b total match: unmatched={(sort_idx==0).sum()}")

        return sort_idx, match_idx, distance, artifact_ids
