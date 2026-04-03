"""
test_full_pipeline.py
======================
Runs C7 on simulation_5 data and compares output against sort_cat.h5.

Usage:
    python test_full_pipeline.py \
        --datafile simulation_5/data_simulation_5.h5 \
        --groundtruth sort_cat.h5 \
        --spc-path path/to/cluster/binary

Make sure c1 through c7 .py files are in the same directory.
"""

import argparse
import numpy as np
import h5py
import torch
import time

from c7_combinato_sorter import CombinatorSorter


def load_spikes(datafile, sign='pos'):
    """Load spikes from extraction H5 file."""
    with h5py.File(datafile, 'r') as f:
        spikes = f[sign]['spikes'][:]
    print(f"Loaded {spikes.shape[0]} {sign} spikes, shape={spikes.shape}")
    return spikes


def load_ground_truth(gt_file):
    """Load ground truth from sort_cat.h5."""
    with h5py.File(gt_file, 'r') as f:
        gt = {
            'classes':   f['classes'][:],
            'matches':   f['matches'][:],
            'distance':  f['distance'][:],
            'artifacts': f['artifacts'][:],
        }
    print(f"Loaded ground truth: {gt['classes'].shape[0]} spikes, "
          f"{len(np.unique(gt['classes']))} clusters")
    return gt


def compare_results(sort_idx, match_idx, distance, artifact_ids, gt):
    """Compare C7 output against ground truth from sort_cat.h5."""

    print("\n" + "=" * 70)
    print("COMPARISON: PyTorch C7 vs Original sort_cat.h5")
    print("=" * 70)

    N = len(sort_idx)
    gt_classes  = gt['classes']
    gt_matches  = gt['matches']
    gt_distance = gt['distance']
    gt_artifacts = gt['artifacts']

    # ------------------------------------------------------------------
    # 1. Basic shape check
    # ------------------------------------------------------------------
    print("\n--- 1. SHAPE CHECK ---")
    print(f"  Spikes          : {N} vs {len(gt_classes)}  "
          f"{'PASS' if N == len(gt_classes) else 'FAIL'}")

    # ------------------------------------------------------------------
    # 2. Cluster count
    # ------------------------------------------------------------------
    print("\n--- 2. CLUSTER COUNT ---")
    our_clusters = np.unique(sort_idx)
    gt_clusters  = np.unique(gt_classes)
    print(f"  Our clusters    : {our_clusters}")
    print(f"  GT  clusters    : {gt_clusters}")
    print(f"  Count           : {len(our_clusters)} vs {len(gt_clusters)}  "
          f"{'PASS' if len(our_clusters) == len(gt_clusters) else 'DIFF'}")

    # ------------------------------------------------------------------
    # 3. Unassigned spikes
    # ------------------------------------------------------------------
    print("\n--- 3. UNASSIGNED SPIKES ---")
    our_unassigned = (sort_idx == 0).sum()
    gt_unassigned  = (gt_classes == 0).sum()
    print(f"  Ours            : {our_unassigned}")
    print(f"  GT              : {gt_unassigned}")
    print(f"  Match           : {'PASS' if our_unassigned == gt_unassigned else 'DIFF'}")

    # ------------------------------------------------------------------
    # 4. Match type breakdown
    # ------------------------------------------------------------------
    print("\n--- 4. MATCH TYPE BREAKDOWN ---")
    for mtype, label in [(0, 'SPC clustered'), (1, 'C5 first match'), (2, 'C5b total match')]:
        ours = (match_idx == mtype).sum()
        theirs = (gt_matches == mtype).sum()
        status = 'PASS' if ours == theirs else 'DIFF'
        print(f"  match={mtype} ({label:15s}): {ours:6d} vs {theirs:6d}  {status}")

    # ------------------------------------------------------------------
    # 5. Exact class match (cluster IDs may differ, so check structure)
    # ------------------------------------------------------------------
    print("\n--- 5. EXACT CLASS MATCH ---")
    exact = np.array_equal(sort_idx.astype(np.uint16), gt_classes.astype(np.uint16))
    print(f"  Exact match     : {'PASS' if exact else 'FAIL'}")

    if not exact:
        # Check if it's just a relabeling — same grouping, different IDs
        from collections import defaultdict

        def cluster_sets(labels):
            d = defaultdict(set)
            for i, l in enumerate(labels):
                d[l].add(i)
            return set(frozenset(v) for v in d.values())

        our_sets = cluster_sets(sort_idx)
        gt_sets  = cluster_sets(gt_classes)
        structural = our_sets == gt_sets
        print(f"  Structural match: {'PASS' if structural else 'FAIL'}")
        print(f"  (same spike groupings, possibly different cluster IDs)")

        if not structural:
            # Per-cluster size comparison
            print("\n  Per-cluster sizes:")
            print(f"  {'Cluster':>8s}  {'Ours':>6s}  {'GT':>6s}")
            all_ids = sorted(set(our_clusters) | set(gt_clusters))
            for cid in all_ids:
                ours = (sort_idx == cid).sum()
                theirs = (gt_classes == cid).sum()
                flag = '' if ours == theirs else ' <--'
                print(f"  {cid:8d}  {ours:6d}  {theirs:6d}{flag}")

    # ------------------------------------------------------------------
    # 6. Cluster sizes (regardless of IDs)
    # ------------------------------------------------------------------
    print("\n--- 6. CLUSTER SIZE COMPARISON ---")
    our_sizes = sorted([(sort_idx == c).sum() for c in our_clusters if c != 0], reverse=True)
    gt_sizes  = sorted([(gt_classes == c).sum() for c in gt_clusters if c != 0], reverse=True)
    print(f"  Our sizes (desc): {our_sizes}")
    print(f"  GT  sizes (desc): {gt_sizes}")
    sizes_match = our_sizes == gt_sizes
    print(f"  Match           : {'PASS' if sizes_match else 'DIFF'}")

    # ------------------------------------------------------------------
    # 7. Match type array
    # ------------------------------------------------------------------
    print("\n--- 7. MATCH TYPE EXACT ---")
    match_exact = np.array_equal(match_idx, gt_matches)
    print(f"  Exact match     : {'PASS' if match_exact else 'FAIL'}")
    if not match_exact:
        diffs = (match_idx != gt_matches).sum()
        print(f"  Differences     : {diffs}/{N} spikes")

    # ------------------------------------------------------------------
    # 8. Distance array
    # ------------------------------------------------------------------
    print("\n--- 8. DISTANCE CHECK ---")
    # distances should match for C5b-matched spikes
    dist_nonzero_ours = (distance > 0).sum()
    dist_nonzero_gt   = (gt_distance > 0).sum()
    print(f"  Nonzero dists   : {dist_nonzero_ours} vs {dist_nonzero_gt}")
    if dist_nonzero_ours == dist_nonzero_gt and dist_nonzero_ours > 0:
        max_diff = np.max(np.abs(distance - gt_distance))
        print(f"  Max diff        : {max_diff:.6f}  "
              f"{'PASS' if max_diff < 1.0 else 'DIFF'}")

    # ------------------------------------------------------------------
    # 9. Artifacts
    # ------------------------------------------------------------------
    print("\n--- 9. ARTIFACTS ---")
    gt_art_ids = sorted(gt_artifacts[gt_artifacts[:, 1] > 0, 0].tolist())
    print(f"  Our artifacts   : {sorted(artifact_ids)}")
    print(f"  GT  artifacts   : {gt_art_ids}")
    art_match = sorted(artifact_ids) == gt_art_ids
    print(f"  Match           : {'PASS' if art_match else 'DIFF'}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    all_pass = exact and match_exact and art_match
    if all_pass:
        print("ALL TESTS PASSED — PyTorch C7 matches original exactly!")
    else:
        checks = [
            ("Cluster count", len(our_clusters) == len(gt_clusters)),
            ("Unassigned", our_unassigned == gt_unassigned),
            ("Exact classes", exact),
            ("Match types", match_exact),
            ("Cluster sizes", sizes_match),
            ("Artifacts", art_match),
        ]
        passed = sum(1 for _, v in checks if v)
        print(f"RESULTS: {passed}/{len(checks)} checks passed")
        for name, val in checks:
            print(f"  {'PASS' if val else 'FAIL'}: {name}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Test C7 pipeline against sort_cat.h5 ground truth')
    parser.add_argument('--datafile', required=True,
                        help='Path to data_simulation_5.h5')
    parser.add_argument('--groundtruth', required=True,
                        help='Path to sort_cat.h5')
    parser.add_argument('--spc-path', required=True,
                        help='Path to SPC cluster binary')
    parser.add_argument('--work-dir', default='test_pytorch_output',
                        help='Working directory for temp files')
    parser.add_argument('--seed', type=float, default=12345.0,
                        help='Random seed (default: 12345.0)')
    args = parser.parse_args()

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    spikes_np = load_spikes(args.datafile, sign='pos')
    gt        = load_ground_truth(args.groundtruth)

    # Run C7
    print("\n" + "=" * 70)
    print("RUNNING C7 PIPELINE")
    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sorter = CombinatorSorter(
        cluster_path=args.spc_path,
        seed=args.seed
    ).to(device)

    spikes_t = torch.tensor(spikes_np, dtype=torch.float32).to(device)

    t0 = time.time()
    with torch.no_grad():
        sort_idx, match_idx, distance, artifact_ids = sorter(
            spikes_t, args.work_dir, sign='pos')
    total_time = time.time() - t0
    print(f"\nTotal C7 time: {total_time:.1f}s")

    # Compare
    compare_results(sort_idx, match_idx, distance, artifact_ids, gt)

    # Save results to H5 (same format as sort_cat.h5 for MATLAB)
    import os
    output_file = 'pytorch_sort_cat.h5'
    unique_clusters = np.unique(sort_idx)
    n_clusters = len(unique_clusters)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('classes', data=sort_idx.astype(np.uint16))
        f.create_dataset('matches', data=match_idx.astype(np.int8))
        f.create_dataset('distance', data=distance.astype(np.float32))
        f.create_dataset('index', data=np.arange(len(sort_idx), dtype=np.uint32))
        
        # artifacts: Nx2 [cluster_id, score]
        artifacts = np.zeros((n_clusters, 2), dtype=np.int64)
        artifacts[:, 0] = unique_clusters
        for i, c in enumerate(unique_clusters):
            if c in artifact_ids:
                artifacts[i, 1] = 1
        f.create_dataset('artifacts', data=artifacts)
        f.create_dataset('artifacts_prematch', data=artifacts)
        
        # groups: Nx2 [cluster_id, group_id] (1:1 mapping if no manual merging)
        groups = np.column_stack([unique_clusters, unique_clusters]).astype(np.int16)
        f.create_dataset('groups', data=groups)
        f.create_dataset('groups_orig', data=groups)
        
        # types: Nx2 [cluster_id, type] (0=unassigned, 1=MU, 2=SU, -1=artifact)
        types_arr = np.zeros((n_clusters, 2), dtype=np.int16)
        types_arr[:, 0] = unique_clusters
        types_arr[:, 1] = 1  # default to MU
        for i, c in enumerate(unique_clusters):
            if c in artifact_ids:
                types_arr[i, 1] = -1
        f.create_dataset('types', data=types_arr)
        f.create_dataset('types_orig', data=types_arr)

    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
