"""
save_original_intermediates.py
================================
Runs the original Combinato pipeline stage by stage and saves
intermediate outputs for comparison against PyTorch reimplementation.

Usage:
    python save_original_intermediates.py --datafile simulation_5/data_simulation_5.h5

Run this from your Combinato environment where 'from combinato.cluster...' works.
Saves all intermediates to 'intermediates/' folder as .npy files.
"""

import os
import numpy as np
import h5py
from argparse import ArgumentParser

# Original Combinato imports
from combinato.cluster.wave_features import wavelet_features
from combinato.cluster.select_features import select_features
from combinato.cluster.cluster_features import cluster_features, read_results
from combinato.cluster.define_clusters import define_clusters
from combinato.cluster.dist import template_match, get_means, distances_euclidean
from combinato.cluster.artifacts import find_artifacts, artifact_score
from combinato.cluster.handle_random_seed import handle_random_seed
from combinato import options


def main():
    parser = ArgumentParser(description='Save intermediate outputs from original pipeline')
    parser.add_argument('--datafile', required=True, help='Path to data_simulation_5.h5')
    parser.add_argument('--outdir', default='intermediates', help='Output directory')
    parser.add_argument('--seed', type=float, default=12345.0, help='Random seed')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Load spikes
    print("=" * 60)
    print("LOADING SPIKES")
    print("=" * 60)
    with h5py.File(args.datafile, 'r') as f:
        spikes = f['pos']['spikes'][:]
    print(f"Spikes shape: {spikes.shape}")
    np.save(os.path.join(outdir, 'input_spikes.npy'), spikes)

    # ---- C1: Wavelet Features ----
    print("\n" + "=" * 60)
    print("C1: WAVELET FEATURES")
    print("=" * 60)
    all_features = wavelet_features(spikes)
    print(f"Features shape: {all_features.shape}")
    print(f"Features dtype: {all_features.dtype}")
    print(f"Features mean: {all_features.mean():.6f}")
    print(f"Features std:  {all_features.std():.6f}")
    np.save(os.path.join(outdir, 'c1_features.npy'), all_features)

    # ---- C2: Feature Selection ----
    print("\n" + "=" * 60)
    print("C2: FEATURE SELECTION")
    print("=" * 60)
    feat_idx = select_features(all_features)
    print(f"Selected indices: {feat_idx}")
    print(f"Selected features shape: {all_features[:, feat_idx].shape}")
    np.save(os.path.join(outdir, 'c2_feat_idx.npy'), feat_idx)
    np.save(os.path.join(outdir, 'c2_selected_features.npy'), all_features[:, feat_idx])

    # ---- C3: SPC Clustering ----
    print("\n" + "=" * 60)
    print("C3: SPC CLUSTERING")
    print("=" * 60)
    spc_folder = os.path.join(outdir, 'spc_temp')
    os.makedirs(spc_folder, exist_ok=True)
    spc_name = 'sort_0'
    seed = handle_random_seed(args.seed)
    
    cluster_features(all_features[:, feat_idx], spc_folder, spc_name, seed)
    clu, tree = read_results(spc_folder, spc_name)
    print(f"Clu shape: {clu.shape}")
    print(f"Tree shape: {tree.shape}")
    np.save(os.path.join(outdir, 'c3_clu.npy'), clu)
    np.save(os.path.join(outdir, 'c3_tree.npy'), tree)

    # ---- C4: Define Clusters ----
    print("\n" + "=" * 60)
    print("C4: DEFINE CLUSTERS")
    print("=" * 60)
    res_idx, tree_out, used_points = define_clusters(clu, tree)
    print(f"Cluster assignments shape: {res_idx.shape}")
    print(f"Unique clusters: {np.unique(res_idx)}")
    for cid in np.unique(res_idx):
        print(f"  Cluster {cid}: {(res_idx == cid).sum()} spikes")
    np.save(os.path.join(outdir, 'c4_res_idx.npy'), res_idx)
    np.save(os.path.join(outdir, 'c4_used_points.npy'), np.array(used_points, dtype=object), allow_pickle=True)

    # ---- C5: Template Match ----
    print("\n" + "=" * 60)
    print("C5: TEMPLATE MATCH (first pass)")
    print("=" * 60)
    sort_idx = res_idx.astype(np.uint16)
    # offset like iterative_sorter does
    clustered = sort_idx > 0
    # sort_idx[clustered] += 0  # no offset needed for first iteration
    
    match_idx = np.zeros(spikes.shape[0], dtype=bool)
    
    print(f"Before C5: unmatched={(sort_idx == 0).sum()}")
    
    # Save pre-C5 state
    np.save(os.path.join(outdir, 'c5_pre_sort_idx.npy'), sort_idx.copy())
    
    first_match_factor = options['FirstMatchFactor']
    print(f"FirstMatchFactor: {first_match_factor}")
    print(f"FirstMatchMaxDist: {options['FirstMatchMaxDist']}")
    
    template_match(spikes, sort_idx, match_idx, first_match_factor)
    
    print(f"After C5:  unmatched={(sort_idx == 0).sum()}")
    print(f"Matched by C5: {match_idx.sum()}")
    np.save(os.path.join(outdir, 'c5_sort_idx.npy'), sort_idx)
    np.save(os.path.join(outdir, 'c5_match_idx.npy'), match_idx)

    # ---- C6: Artifact Detection ----
    print("\n" + "=" * 60)
    print("C6: ARTIFACT DETECTION")
    print("=" * 60)
    class_ids = np.unique(sort_idx)
    artifact_idx, artifact_ids = find_artifacts(spikes, sort_idx, class_ids, invert=False)
    print(f"Artifact clusters: {artifact_ids}")
    print(f"Total clusters: {len(class_ids[class_ids != 0])}")
    np.save(os.path.join(outdir, 'c6_artifact_ids.npy'), np.array(artifact_ids))
    np.save(os.path.join(outdir, 'c6_artifact_idx.npy'), artifact_idx)

    # ---- Print options for reference ----
    print("\n" + "=" * 60)
    print("COMBINATO OPTIONS")
    print("=" * 60)
    important_keys = [
        'FirstMatchFactor', 'FirstMatchMaxDist',
        'SecondMatchFactor', 'SecondMatchMaxDist',
        'MinInputSize', 'MinInputSizeRecluster',
        'ReclusterClusters', 'RecursiveDepth',
        'MaxClustersPerTemp', 'MinSpikesPerClusterMultiSelect',
        'TempStep', 'Wavelet', 'nFeatures', 'feature_factor',
        'ExcludeVariableClustersMatch', 'MarkArtifactClasses',
    ]
    for key in important_keys:
        if key in options:
            print(f"  {key}: {options[key]}")

    # Save options
    opts_to_save = {k: options[k] for k in important_keys if k in options}
    np.save(os.path.join(outdir, 'options.npy'), opts_to_save, allow_pickle=True)

    print("\n" + "=" * 60)
    print(f"All intermediates saved to '{outdir}/'")
    print("=" * 60)


if __name__ == '__main__':
    main()
