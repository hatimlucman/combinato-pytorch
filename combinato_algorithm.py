"""
combinato_algorithm.py
========================
Single-file PyTorch reimplementation of the full Combinato pipeline.

Chains extraction (M1-M6) and sorting (C1-C7) into one nn.Module.

Usage:
    from combinato_algorithm import CombinatoAlgorithm

    model = CombinatoAlgorithm(spc_path='/path/to/cluster/binary')
    model = model.to('cuda')  # or 'cpu'

    results = model(signal_tensor, times_tensor)

    # Extraction outputs
    results['pos_spikes']    # (P, 64) tensor
    results['pos_times']     # (P,)    tensor  — ms
    results['neg_spikes']    # (Q, 64) tensor
    results['neg_times']     # (Q,)    tensor  — ms
    results['threshold']     # scalar tensor

    # Sorting outputs — positive polarity
    results['pos_sort_idx']      # (P,) numpy uint16  — cluster ID per spike
    results['pos_match_idx']     # (P,) numpy int8    — 0=SPC, 1=C5, 2=C5b
    results['pos_distance']      # (P,) numpy float32 — template match distance
    results['pos_artifact_ids']  # list of artifact cluster IDs

    # Sorting outputs — negative polarity
    results['neg_sort_idx']
    results['neg_match_idx']
    results['neg_distance']
    results['neg_artifact_ids']

    # Per-stage runtimes (seconds)
    results['timings']  # dict: {'M1': 0.12, 'M2': 0.01, ..., 'C7_pos': 4.2, ...}

Requirements:
    torch, torchaudio, numpy, scipy, pywt (PyWavelets), h5py, tables
    SPC binary (external C program, not GPU-accelerated)

All M and C modules must be importable from the same directory,
OR you can paste them all into this file (see INLINE MODE below).
"""

import os
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# IMPORTS — M1-M6 and C1-C7 modules
# ---------------------------------------------------------------------------
# If running from the directory containing all module files:
from m1_preprocessor      import Preprocessor
from m2_threshold_detector import ThresholdDetector
from m3_waveform_extractor import WaveformExtractor
from m4_cubic_upsampler    import CubicUpsampler
from m5_peak_aligner       import PeakAligner
from m6_pipeline           import CombinatoExtractor, save_to_h5

from c1_wavelet_features   import WaveletFeatureExtractor
from c2_feature_selector   import FeatureSelector
from c3_spc_clusterer      import SPCClusterer
from c4_cluster_definer    import ClusterDefiner
from c5_template_matcher   import TemplateMatcher
from c5b_total_matcher     import TotalMatcher
from c6_artifact_detector  import ArtifactDetector
from c7_combinato_sorter   import CombinatorSorter


# ---------------------------------------------------------------------------
# CombinatoAlgorithm
# ---------------------------------------------------------------------------

class CombinatoAlgorithm(nn.Module):
    """
    Full Combinato pipeline: raw signal → spike waveforms → sorted clusters.

    Combines CombinatoExtractor (M1-M6) and CombinatorSorter (C1-C7)
    into a single nn.Module. Both polarities (pos + neg) are processed
    in one forward() call.

    Args:
        spc_path (str):
            Path to the SPC binary (e.g. '/path/to/cluster' or
            'C:/combinato/cluster.exe'). Required for sorting.

        sample_rate (int):
            Recording sample rate in Hz. Default 24000.

        work_dir (str or None):
            Working directory for SPC temp files.
            If None, a temp directory is auto-created and cleaned up
            after each forward() call.

        seed (float):
            Random seed passed to SPC for reproducibility. Default 12345.0.

        run_sorting (bool):
            If False, only extraction (M1-M6) is run. Useful for fast
            extraction-only benchmarks. Default True.

        Remaining kwargs are forwarded to CombinatorSorter —
        see c7_combinato_sorter.py for full list.
    """

    def __init__(self,
                 spc_path,
                 sample_rate=24000,
                 work_dir=None,
                 seed=12345.0,
                 run_sorting=True,
                 # CombinatorSorter kwargs
                 n_iterations=1,
                 recluster=True,
                 min_input_size=15,
                 min_input_size_recluster=2000,
                 first_match_factor=0.75,
                 mark_artifacts=True):
        super().__init__()

        self.work_dir    = work_dir
        self._auto_dir   = (work_dir is None)
        self.run_sorting = run_sorting

        # ---- Extraction pipeline (M1-M6) ----
        self.extractor = CombinatoExtractor(sample_rate=sample_rate)

        # ---- Sorting pipeline (C1-C7) ----
        if run_sorting:
            self.sorter = CombinatorSorter(
                cluster_path=spc_path,
                seed=seed,
                n_iterations=n_iterations,
                recluster=recluster,
                min_input_size=min_input_size,
                min_input_size_recluster=min_input_size_recluster,
                first_match_factor=first_match_factor,
                mark_artifacts=mark_artifacts,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _timed(self, label, fn, timings, *args, **kwargs):
        """Call fn(*args, **kwargs), record wall time under label."""
        t0     = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[label] = time.perf_counter() - t0
        return result

    def _get_work_dir(self):
        """Return (path, should_cleanup) for SPC temp files."""
        if self._auto_dir:
            return tempfile.mkdtemp(prefix='combinato_'), True
        os.makedirs(self.work_dir, exist_ok=True)
        return self.work_dir, False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, signal, atimes):
        """
        Run the full pipeline on a raw signal.

        Args:
            signal: (N,) float64 tensor — raw scaled signal
            atimes: (N,) float64 tensor — timestamp per sample in ms

        Returns:
            dict with keys described in module docstring.
        """
        timings = {}
        device  = signal.device

        # ==============================================================
        # STAGE 1: Extraction  M1-M6
        # ==============================================================
        print("\n[CombinatoAlgorithm] Stage 1: Extraction (M1-M6)")

        with torch.no_grad():
            extraction = self._timed(
                'extraction', self.extractor, timings, signal, atimes)

        pos_spikes = extraction['pos_spikes']
        neg_spikes = extraction['neg_spikes']
        pos_times  = extraction['pos_times']
        neg_times  = extraction['neg_times']
        threshold  = extraction['threshold']

        print(f"  Threshold  : {threshold.item():.4f}")
        print(f"  Pos spikes : {pos_spikes.shape[0]}")
        print(f"  Neg spikes : {neg_spikes.shape[0]}")
        print(f"  Extraction : {timings['extraction']:.2f}s")

        # Build result dict — populated incrementally
        result = {
            'pos_spikes'  : pos_spikes,
            'pos_times'   : pos_times,
            'neg_spikes'  : neg_spikes,
            'neg_times'   : neg_times,
            'threshold'   : threshold,
            'timings'     : timings,
        }

        # ==============================================================
        # STAGE 2: Sorting  C1-C7  (both polarities)
        # ==============================================================
        if not self.run_sorting:
            # Fill sorting fields with empty arrays so callers don't
            # need to guard against missing keys
            for sign in ('pos', 'neg'):
                n = pos_spikes.shape[0] if sign == 'pos' else neg_spikes.shape[0]
                result[f'{sign}_sort_idx']     = np.zeros(n, dtype=np.uint16)
                result[f'{sign}_match_idx']    = np.zeros(n, dtype=np.int8)
                result[f'{sign}_distance']     = np.zeros(n, dtype=np.float32)
                result[f'{sign}_artifact_ids'] = []
            return result

        print("\n[CombinatoAlgorithm] Stage 2: Sorting (C1-C7)")

        work_dir, cleanup = self._get_work_dir()

        try:
            for sign, spikes_t in [('pos', pos_spikes), ('neg', neg_spikes)]:
                label = f'sorting_{sign}'
                print(f"\n  --- {sign.upper()} polarity ---")

                t0 = time.perf_counter()
                with torch.no_grad():
                    sort_idx, match_idx, distance, artifact_ids = self.sorter(
                        spikes_t,
                        os.path.join(work_dir, sign),
                        sign=sign,
                    )
                timings[label] = time.perf_counter() - t0

                result[f'{sign}_sort_idx']     = sort_idx
                result[f'{sign}_match_idx']    = match_idx
                result[f'{sign}_distance']     = distance
                result[f'{sign}_artifact_ids'] = artifact_ids

                n_clusters = len(np.unique(sort_idx[sort_idx > 0]))
                print(f"  Clusters   : {n_clusters}")
                print(f"  Unmatched  : {(sort_idx == 0).sum()}")
                print(f"  Artifacts  : {len(artifact_ids)}")
                print(f"  Time       : {timings[label]:.1f}s")

        finally:
            if cleanup:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)

        return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_extraction(result, gt_h5_path):
    """
    Compare extraction output against a ground-truth H5 file.

    Args:
        result:      dict returned by CombinatoAlgorithm.forward()
        gt_h5_path:  path to ground-truth H5 file
                     (must have /pos/spikes, /pos/times, /neg/spikes, /neg/times)

    Returns:
        passed (bool), report (str)
    """
    import tables
    import numpy as np

    h5 = tables.open_file(gt_h5_path, 'r')
    gt = {
        'pos_spikes': h5.root.pos.spikes[:],
        'pos_times' : h5.root.pos.times[:],
        'neg_spikes': h5.root.neg.spikes[:],
        'neg_times' : h5.root.neg.times[:],
    }
    h5.close()

    lines  = ["\n" + "="*60, "EXTRACTION VALIDATION", "="*60]
    checks = {}

    for sign in ('pos', 'neg'):
        our_spikes = result[f'{sign}_spikes'].cpu().numpy()
        our_times  = result[f'{sign}_times'].cpu().numpy()
        gt_spikes  = gt[f'{sign}_spikes']
        gt_times   = gt[f'{sign}_times']

        count_pct = abs(our_spikes.shape[0] - gt_spikes.shape[0]) \
                    / max(gt_spikes.shape[0], 1) * 100
        corr      = float(np.corrcoef(
            our_spikes.mean(0), gt_spikes.mean(0))[0, 1]) \
            if our_spikes.shape[0] > 0 else 0.0
        amp_ratio = (np.abs(our_spikes).max() / np.abs(gt_spikes).max()) \
            if gt_spikes.size > 0 else 1.0

        count_ok = count_pct < 5
        corr_ok  = corr > 0.99
        amp_ok   = 0.8 < amp_ratio < 1.2

        lines.append(f"\n  [{sign.upper()}]")
        lines.append(f"    Count : {our_spikes.shape[0]} vs {gt_spikes.shape[0]}"
                     f"  ({count_pct:.2f}%)  {'✓' if count_ok else '✗'}")
        lines.append(f"    Corr  : {corr:.4f}           {'✓' if corr_ok else '✗'}")
        lines.append(f"    Amp   : {amp_ratio:.4f}          {'✓' if amp_ok else '✗'}")

        checks[f'{sign}_count'] = count_ok
        checks[f'{sign}_corr']  = corr_ok
        checks[f'{sign}_amp']   = amp_ok

    passed = all(checks.values())
    lines += ["", "EXTRACTION: PASSED ✓" if passed else "EXTRACTION: FAILED ✗", "="*60]
    return passed, '\n'.join(lines)


def validate_sorting(result, sort_cat_h5_path):
    """
    Compare sorting output against sort_cat.h5 ground truth.

    Args:
        result:            dict returned by CombinatoAlgorithm.forward()
        sort_cat_h5_path:  path to sort_cat.h5

    Returns:
        passed (bool), report (str)
    """
    import h5py
    import numpy as np

    with h5py.File(sort_cat_h5_path, 'r') as f:
        gt = {
            'classes'  : f['classes'][:],
            'matches'  : f['matches'][:],
            'distance' : f['distance'][:],
            'artifacts': f['artifacts'][:],
        }

    # sort_cat.h5 stores pos+neg concatenated; here we validate pos only
    # (extend if your ground truth separates them)
    sort_idx     = result['pos_sort_idx']
    match_idx    = result['pos_match_idx']
    distance     = result['pos_distance']
    artifact_ids = result['pos_artifact_ids']

    lines  = ["\n" + "="*60, "SORTING VALIDATION (pos)", "="*60]
    checks = {}

    # cluster count
    our_n = len(np.unique(sort_idx[sort_idx > 0]))
    gt_n  = len(np.unique(gt['classes'][gt['classes'] > 0]))
    checks['cluster_count'] = our_n == gt_n
    lines.append(f"\n  Clusters : {our_n} vs {gt_n}  "
                 f"{'✓' if checks['cluster_count'] else 'DIFF'}")

    # unassigned
    our_u = (sort_idx == 0).sum()
    gt_u  = (gt['classes'] == 0).sum()
    checks['unassigned'] = our_u == gt_u
    lines.append(f"  Unassigned: {our_u} vs {gt_u}  "
                 f"{'✓' if checks['unassigned'] else 'DIFF'}")

    # exact class match
    if len(sort_idx) == len(gt['classes']):
        exact = np.array_equal(
            sort_idx.astype(np.uint16), gt['classes'].astype(np.uint16))
        checks['exact_classes'] = exact
        lines.append(f"  Exact classes: {'✓' if exact else 'DIFF'}")
    else:
        checks['exact_classes'] = False
        lines.append(f"  Length mismatch: {len(sort_idx)} vs {len(gt['classes'])}")

    # artifacts
    gt_art = sorted(gt['artifacts'][gt['artifacts'][:, 1] > 0, 0].tolist())
    checks['artifacts'] = sorted(artifact_ids) == gt_art
    lines.append(f"  Artifacts: {sorted(artifact_ids)} vs {gt_art}  "
                 f"{'✓' if checks['artifacts'] else 'DIFF'}")

    passed = all(checks.values())
    lines += ["", "SORTING: PASSED ✓" if passed else "SORTING: FAILED ✗", "="*60]
    return passed, '\n'.join(lines)


# ---------------------------------------------------------------------------
# Benchmarking helper
# ---------------------------------------------------------------------------

def benchmark_timings(result, original_timings=None):
    """
    Print and optionally plot a bar chart of per-stage timings.

    Args:
        result:             dict from CombinatoAlgorithm.forward()
        original_timings:   dict of {stage: seconds} from original Combinato.
                            If provided, a grouped bar chart is shown.

    Returns:
        fig (matplotlib Figure) or None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    our = result['timings']

    if original_timings is None:
        print("\nPer-stage timings (PyTorch):")
        for k, v in our.items():
            print(f"  {k:25s}: {v:.3f}s")
        return None

    # --- grouped bar chart ---
    stages = sorted(set(our) | set(original_timings))
    x      = np.arange(len(stages))
    w      = 0.35

    our_vals  = [our.get(s, 0)              for s in stages]
    orig_vals = [original_timings.get(s, 0) for s in stages]

    fig, ax = plt.subplots(figsize=(max(8, len(stages)*1.2), 5))
    ax.bar(x - w/2, orig_vals, w, label='Original Combinato', color='steelblue')
    ax.bar(x + w/2, our_vals,  w, label='PyTorch Combinato',  color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_title('Per-stage runtime: Original vs PyTorch Combinato')
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_results(result, output_h5_path):
    """
    Save extraction + sorting results to HDF5.

    Extraction part matches Combinato's data_*.h5 format.
    Sorting part matches sort_cat.h5 format.

    Args:
        result:          dict from CombinatoAlgorithm.forward()
        output_h5_path:  path to write
    """
    # save extraction
    save_to_h5(result, output_h5_path.replace('.h5', '_extraction.h5'))

    # save sorting
    import h5py
    with h5py.File(output_h5_path, 'w') as f:
        for sign in ('pos', 'neg'):
            grp      = f.create_group(sign)
            sort_idx = result[f'{sign}_sort_idx']
            grp.create_dataset('classes',  data=sort_idx.astype(np.uint16))
            grp.create_dataset('matches',  data=result[f'{sign}_match_idx'].astype(np.int8))
            grp.create_dataset('distance', data=result[f'{sign}_distance'].astype(np.float32))

            unique_c = np.unique(sort_idx)
            art_ids  = set(result[f'{sign}_artifact_ids'])
            artifacts = np.zeros((len(unique_c), 2), dtype=np.int64)
            artifacts[:, 0] = unique_c
            for i, c in enumerate(unique_c):
                if c in art_ids:
                    artifacts[i, 1] = 1
            grp.create_dataset('artifacts', data=artifacts)

    print(f"Saved results to {output_h5_path}")


# ---------------------------------------------------------------------------
# CLI — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run CombinatoAlgorithm on a .mat file')
    parser.add_argument('--matfile',   required=True)
    parser.add_argument('--spc-path',  required=True)
    parser.add_argument('--scale',     type=float, default=100.0)
    parser.add_argument('--device',    default='auto')
    parser.add_argument('--output',    default='combinato_output.h5')
    parser.add_argument('--gt-extraction', default=None,
                        help='Ground-truth extraction H5 for validation')
    parser.add_argument('--gt-sorting', default=None,
                        help='sort_cat.h5 for sorting validation')
    args = parser.parse_args()

    from scipy.io import loadmat

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    mat    = loadmat(args.matfile)
    fdata  = mat['data'].ravel().astype(np.float64) * args.scale
    sr     = float(mat['sr'].ravel()[0]) if 'sr' in mat else 24000.0
    atimes = np.linspace(0, fdata.shape[0] / (sr / 1000), fdata.shape[0])

    signal_t = torch.tensor(fdata,  dtype=torch.float64).to(device)
    times_t  = torch.tensor(atimes, dtype=torch.float64).to(device)

    model  = CombinatoAlgorithm(spc_path=args.spc_path).to(device)

    t0     = time.perf_counter()
    with torch.no_grad():
        result = model(signal_t, times_t)
    total  = time.perf_counter() - t0

    print(f"\nTotal time: {total:.1f}s  |  device: {device}")

    if args.gt_extraction:
        passed, report = validate_extraction(result, args.gt_extraction)
        print(report)

    if args.gt_sorting:
        passed, report = validate_sorting(result, args.gt_sorting)
        print(report)

    save_results(result, args.output)
    benchmark_timings(result)
