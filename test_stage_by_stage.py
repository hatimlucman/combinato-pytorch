"""
test_stage_by_stage.py
========================
Compares each PyTorch module (C1-C6) against saved intermediates
from the original Combinato pipeline.

Usage:
    python test_stage_by_stage.py --intermediates intermediates/

Run save_original_intermediates.py first to generate the intermediates.
"""

import os
import numpy as np
import torch
from argparse import ArgumentParser


def test_c1(intermediates_dir):
    """Test C1: Wavelet Features"""
    print("\n" + "=" * 60)
    print("C1: WAVELET FEATURES")
    print("=" * 60)

    from c1_wavelet_features import WaveletFeatureExtractor

    spikes = np.load(os.path.join(intermediates_dir, 'input_spikes.npy'))
    gt     = np.load(os.path.join(intermediates_dir, 'c1_features.npy'))

    extractor = WaveletFeatureExtractor()
    spikes_t  = torch.tensor(spikes, dtype=torch.float32)

    with torch.no_grad():
        ours = extractor(spikes_t).cpu().numpy()

    print(f"  Our shape : {ours.shape}")
    print(f"  GT  shape : {gt.shape}")

    if ours.shape != gt.shape:
        print(f"  FAIL — shape mismatch!")
        return False

    max_diff = np.max(np.abs(ours - gt))
    print(f"  Max diff  : {max_diff:.2e}")
    passed = max_diff < 1e-5
    print(f"  Result    : {'PASS' if passed else 'FAIL'}")
    return passed


def test_c2(intermediates_dir):
    """Test C2: Feature Selection"""
    print("\n" + "=" * 60)
    print("C2: FEATURE SELECTION")
    print("=" * 60)

    from c2_feature_selector import FeatureSelector

    features = np.load(os.path.join(intermediates_dir, 'c1_features.npy'))
    gt_idx   = np.load(os.path.join(intermediates_dir, 'c2_feat_idx.npy'))
    gt_sel   = np.load(os.path.join(intermediates_dir, 'c2_selected_features.npy'))

    selector   = FeatureSelector()
    features_t = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        our_sel, our_idx = selector(features_t)

    our_idx_np = our_idx.cpu().numpy()
    our_sel_np = our_sel.cpu().numpy()

    print(f"  Our indices : {our_idx_np}")
    print(f"  GT  indices : {gt_idx}")

    idx_match = np.array_equal(our_idx_np, gt_idx)
    print(f"  Indices match: {'PASS' if idx_match else 'FAIL'}")

    if idx_match:
        max_diff = np.max(np.abs(our_sel_np - gt_sel))
        print(f"  Max diff     : {max_diff:.2e}")
        val_pass = max_diff < 1e-5
        print(f"  Values match : {'PASS' if val_pass else 'FAIL'}")
        return val_pass
    return False


def test_c4(intermediates_dir):
    """Test C4: Define Clusters (using saved SPC output)"""
    print("\n" + "=" * 60)
    print("C4: DEFINE CLUSTERS")
    print("=" * 60)

    from c4_cluster_definer import ClusterDefiner

    clu    = np.load(os.path.join(intermediates_dir, 'c3_clu.npy'))
    tree   = np.load(os.path.join(intermediates_dir, 'c3_tree.npy'))
    gt_idx = np.load(os.path.join(intermediates_dir, 'c4_res_idx.npy'))

    definer = ClusterDefiner()
    our_idx, _, _ = definer(clu, tree)

    print(f"  Our clusters : {np.unique(our_idx)}")
    print(f"  GT  clusters : {np.unique(gt_idx)}")

    for cid in sorted(set(np.unique(our_idx)) | set(np.unique(gt_idx))):
        ours   = (our_idx == cid).sum()
        theirs = (gt_idx == cid).sum()
        flag   = '' if ours == theirs else ' <--'
        print(f"    Cluster {cid:2d}: {ours:5d} vs {theirs:5d}{flag}")

    match = np.array_equal(our_idx, gt_idx)
    print(f"  Exact match  : {'PASS' if match else 'FAIL'}")
    return match


def test_c5(intermediates_dir):
    """Test C5: Template Match"""
    print("\n" + "=" * 60)
    print("C5: TEMPLATE MATCH")
    print("=" * 60)

    from c5_template_matcher import TemplateMatcher

    spikes      = np.load(os.path.join(intermediates_dir, 'input_spikes.npy'))
    pre_sort    = np.load(os.path.join(intermediates_dir, 'c5_pre_sort_idx.npy'))
    gt_sort     = np.load(os.path.join(intermediates_dir, 'c5_sort_idx.npy'))
    gt_match    = np.load(os.path.join(intermediates_dir, 'c5_match_idx.npy'))

    matcher   = TemplateMatcher()
    spikes_t  = torch.tensor(spikes, dtype=torch.float32)

    our_sort  = pre_sort.copy()
    our_match = np.zeros(len(our_sort), dtype=bool)

    with torch.no_grad():
        matcher(spikes_t, our_sort, our_match)

    print(f"  Before: unmatched = {(pre_sort == 0).sum()}")
    print(f"  Our:    unmatched = {(our_sort == 0).sum()}")
    print(f"  GT:     unmatched = {(gt_sort == 0).sum()}")
    print(f"  Our:    matched   = {our_match.sum()}")
    print(f"  GT:     matched   = {gt_match.sum()}")

    sort_match  = np.array_equal(our_sort, gt_sort)
    match_match = np.array_equal(our_match, gt_match)
    print(f"  sort_idx match  : {'PASS' if sort_match else 'FAIL'}")
    print(f"  match_idx match : {'PASS' if match_match else 'FAIL'}")

    if not sort_match:
        diffs = (our_sort != gt_sort).sum()
        print(f"  Differences     : {diffs}/{len(our_sort)} spikes")

    return sort_match and match_match


def test_c6(intermediates_dir):
    """Test C6: Artifact Detection"""
    print("\n" + "=" * 60)
    print("C6: ARTIFACT DETECTION")
    print("=" * 60)

    from c6_artifact_detector import ArtifactDetector

    spikes   = np.load(os.path.join(intermediates_dir, 'input_spikes.npy'))
    sort_idx = np.load(os.path.join(intermediates_dir, 'c5_sort_idx.npy'))
    gt_arts  = np.load(os.path.join(intermediates_dir, 'c6_artifact_ids.npy'))

    detector = ArtifactDetector()
    spikes_t = torch.tensor(spikes, dtype=torch.float32)

    with torch.no_grad():
        art_scores, art_ids = detector(spikes_t, sort_idx, sign='pos')

    print(f"  Our artifacts : {sorted(art_ids)}")
    print(f"  GT  artifacts : {sorted(gt_arts.tolist())}")

    match = sorted(art_ids) == sorted(gt_arts.tolist())
    print(f"  Match         : {'PASS' if match else 'FAIL'}")
    return match


def main():
    parser = ArgumentParser(description='Stage-by-stage comparison')
    parser.add_argument('--intermediates', default='intermediates',
                        help='Directory with saved intermediates')
    args = parser.parse_args()

    results = {}
    results['C1'] = test_c1(args.intermediates)
    results['C2'] = test_c2(args.intermediates)
    # C3 is the SPC binary — same binary = same output, skip
    results['C4'] = test_c4(args.intermediates)
    results['C5'] = test_c5(args.intermediates)
    results['C6'] = test_c6(args.intermediates)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for stage, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        if not passed:
            all_pass = False
        print(f"  {stage}: {status}")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL STAGES PASSED — each module matches the original exactly")
    else:
        print("SOME STAGES FAILED — fix the first failing stage before moving on")
    print("=" * 60)


if __name__ == '__main__':
    main()
