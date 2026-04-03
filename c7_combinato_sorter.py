"""
C7 - CombinatorSorter
=======================
Full clustering pipeline — C1 through C6 + C5b wired together.

Mirrors the full css-simple-clustering pipeline:

    1. sort_spikes() from cluster.py:
        all_features = wavelet_features(spikes)                     # C1
        iterative_sorter(all_features, spikes, ...):
            for each iteration:
                feat_idx = select_features(features[subset])        # C2 per subset
                cluster_features(features[subset, feat_idx])        # C3
                define_clusters(clu, tree)                          # C4
                recluster large clusters                            # C3+C4 again
                template_match(spikes, idx, match_idx, factor)      # C5
        find_artifacts(spikes, sorted_idx, ...)                     # C6

    2. total_match() from concatenate.py:                           # C5b
        second pass template matching with larger threshold

Output matches sort_cat.h5 format:
    sort_idx:  (N,) uint16 — cluster assignment per spike
    match_idx: (N,) int8   — 0=SPC, 1=C5 first match, 2=C5b total match
    distance:  (N,) float32 — template match distance (0 for SPC-assigned)
    artifact_ids: list of artifact cluster ids

Input:
    spikes:  (N, 64) tensor
    folder:  str — working directory
    sign:    'pos' or 'neg'
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

from c1_wavelet_features import WaveletFeatureExtractor
from c2_feature_selector import FeatureSelector
from c3_spc_clusterer import SPCClusterer
from c4_cluster_definer import ClusterDefiner
from c5_template_matcher import TemplateMatcher
from c5b_total_matcher import TotalMatcher
from c6_artifact_detector import ArtifactDetector

# mirrors combinato options
MIN_INPUT_SIZE           = 15       # options['MinInputSize']
MIN_INPUT_SIZE_RECLUSTER = 2000     # options['MinInputSizeRecluster']
RECLUSTER_CLUSTERS       = True     # options['ReclusterClusters']
RECURSIVE_DEPTH          = 1        # options['RecursiveDepth']
FIRST_MATCH_FACTOR       = 0.75    # options['FirstMatchFactor']
MARK_ARTIFACT_CLASSES    = True     # options['MarkArtifactClasses']
DEFAULT_SEED             = 12345.0


class CombinatorSorter(nn.Module):
    """
    C7: Full spike sorting pipeline C1 -> C6 + C5b.

    Matches the full css-simple-clustering output (sort_cat.h5):
        Step 1 (sort_spikes):  C1 -> C2 -> C3+C4 -> recluster -> C5 -> C6
        Step 2 (total_match):  C5b

    Args:
        cluster_path (str): path to SPC binary.
        n_iterations (int): recursive depth. Default 1.
        recluster (bool): recluster large clusters. Default True.
        min_input_size (int): minimum spikes to cluster. Default 20.
        min_input_size_recluster (int): minimum for reclustering. Default 100.
        first_match_factor (float): template match factor. Default 3.0.
        mark_artifacts (bool): run artifact detection. Default True.
        seed (float): SPC random seed. Default 12345.0.
    """

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
        self.n_iterations              = n_iterations
        self.recluster                 = recluster
        self.min_input_size            = min_input_size
        self.min_input_size_recluster  = min_input_size_recluster
        self.first_match_factor        = first_match_factor
        self.mark_artifacts            = mark_artifacts
        self.seed                      = seed

        self.c1  = WaveletFeatureExtractor()
        self.c2  = FeatureSelector()
        self.c3  = SPCClusterer(cluster_path=cluster_path)
        self.c4  = ClusterDefiner()
        self.c5  = TemplateMatcher()
        self.c5b = TotalMatcher()
        self.c6  = ArtifactDetector()

    def _cluster_step(self, features, folder, sub_name):
        """
        Mirrors cluster_step() -> features_to_index() from cluster.py.

        Takes the FULL wavelet features for a subset of spikes,
        runs C2 (feature selection) on that subset, then C3 + C4.

        Args:
            features: (M, 73) tensor — full wavelet features for subset
            folder:   str — working directory
            sub_name: str — name for temp files

        Returns:
            res_idx: (M,) numpy uint8 array — cluster assignment per spike
        """
        # C2: select features on THIS subset
        with torch.no_grad():
            sel_features, feat_idx = self.c2(features)

        # C3: SPC clustering on selected features
        clu, tree = self.c3(sel_features, folder, sub_name, self.seed)

        # C4: define clusters from temperature tree
        res_idx, tree, used_points = self.c4(clu, tree)

        return res_idx

    def _iterative_sorter(self, all_features, spikes, folder):
        """
        Exact port of iterative_sorter() from cluster.py.

        match_idx is bool here (like original). It gets converted to
        int8 (0/1) after this function returns.

        Args:
            all_features: (N, 73) tensor — wavelet features from C1
            spikes:       (N, 64) tensor — original spike waveforms
            folder:       str — working directory

        Returns:
            idx:       (N,) numpy uint16 array — cluster assignments
            match_idx: (N,) numpy bool array — True where C5 template-matched
        """
        N = all_features.shape[0]
        idx       = np.zeros(N, dtype=np.uint16)
        match_idx = np.zeros(N, dtype=bool)

        for i in range(self.n_iterations):
            # input to clustering: spikes that have no index so far
            sub_idx  = idx == 0
            sub_name = 'sort_' + str(i)

            if sub_idx.sum() < self.min_input_size:
                print(f"    Stopping iteration, {sub_idx.sum()} spikes left")
                break

            print(f"    Iteration {i}: clustering {sub_idx.sum()} spikes...")

            # cluster_step on subset of features
            t0      = time.time()
            res_idx = self._cluster_step(all_features[sub_idx], folder, sub_name)
            print(f"      cluster_step done: {time.time()-t0:.1f}s  "
                  f"clusters={len(np.unique(res_idx[res_idx > 0]))}  "
                  f"unclustered={(res_idx==0).sum()}")

            # offset cluster IDs to avoid collisions with previous iterations
            clustered_idx = res_idx > 0
            prev_idx_max  = idx.max()
            res_idx[clustered_idx] += prev_idx_max
            idx[sub_idx] = res_idx

            # --- Recluster large clusters to reduce under-clustering ---
            if self.recluster:
                clids = np.unique(res_idx[clustered_idx])

                for clid in clids:
                    recluster_mask = idx == clid
                    cluster_size   = recluster_mask.sum()

                    if cluster_size < self.min_input_size_recluster:
                        continue

                    print(f"      Reclustering cluster {clid} ({cluster_size} spikes)")

                    sub_sub_name = '{}_{:02d}'.format(sub_name, clid)

                    recluster_res_idx = self._cluster_step(
                        all_features[recluster_mask], folder, sub_sub_name)

                    # make sure to increase the cluster numbers enough
                    biggest_clid = idx.max()
                    recluster_res_idx[recluster_res_idx != 0] += biggest_clid
                    idx[recluster_mask] = recluster_res_idx

            # --- Conservative template matching INSIDE each iteration ---
            with torch.no_grad():
                self.c5(spikes, idx, match_idx, factor=self.first_match_factor)

            print(f"      After template match: unmatched={(idx==0).sum()}")

        return idx, match_idx

    def forward(self, spikes, folder, sign='pos'):
        """
        Mirrors the full css-simple-clustering pipeline.

        Step 1: sort_spikes() — C1, iterative_sorter(C2+C3+C4+recluster+C5), C6
        Step 2: total_match() — C5b

        Args:
            spikes: (N, 64) tensor
            folder: str — working directory
            sign:   'pos' or 'neg'

        Returns:
            sort_idx:     (N,) numpy uint16 array — cluster per spike
            match_idx:    (N,) numpy int8 array — 0=SPC, 1=C5, 2=C5b
            distance:     (N,) numpy float32 array — match distances
            artifact_ids: list of artifact cluster ids
        """
        N      = spikes.shape[0]
        device = spikes.device

        print(f"\n  [C7] Sorting {N} {sign} spikes...")

        if N < self.min_input_size:
            print(f"  [C7] Too few spikes ({N}), skipping")
            return (np.zeros(N, dtype=np.uint16),
                    np.zeros(N, dtype=np.int8),
                    np.zeros(N, dtype=np.float32),
                    [])

        # ==================================================================
        # STEP 1: sort_spikes() from cluster.py
        # ==================================================================

        # ---- C1: wavelet features (computed once, like original) ----
        t0 = time.time()
        with torch.no_grad():
            all_features = self.c1(spikes)
        print(f"  C1  features   : {tuple(all_features.shape)}  {time.time()-t0:.2f}s")

        # ---- Iterative sorter: C2+C3+C4 per subset, recluster, C5 per iter ----
        t0 = time.time()
        sort_idx, bool_match_idx = self._iterative_sorter(
            all_features, spikes, folder)
        print(f"  Iterative sort : {time.time()-t0:.1f}s  "
              f"unmatched={(sort_idx==0).sum()}")

        # Convert bool match_idx to int8 (False=0=SPC, True=1=C5 matched)
        # This mirrors how session.update_sorting_data saves it as int8
        match_idx = bool_match_idx.astype(np.int8)

        # ---- C6: artifact detection ----
        if self.mark_artifacts:
            with torch.no_grad():
                art_scores, artifact_ids = self.c6(spikes, sort_idx, sign=sign)
            print(f"  C6  artifacts  : {len(artifact_ids)}/{len(art_scores)} flagged")
        else:
            artifact_ids = []

        # ==================================================================
        # STEP 2: total_match() from concatenate.py
        # ==================================================================
        t0 = time.time()
        distance = np.zeros(N, dtype=np.float32)
        with torch.no_grad():
            sort_idx, match_idx, distance = self.c5b(
                spikes, sort_idx, match_idx, distance)
        print(f"  C5b total match: unmatched={(sort_idx==0).sum()}  "
              f"{time.time()-t0:.2f}s")

        return sort_idx, match_idx, distance, artifact_ids


if __name__ == '__main__':
    print("C7 CombinatorSorter defined.")
    print("Run test_full_clustering.py to validate against sort_cat.h5.")
