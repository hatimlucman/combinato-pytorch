"""
combinato_algorithm.py
========================
Single entry point for the full PyTorch Combinato pipeline.

Usage in a notebook or script:
    from combinato_algorithm import CombinatoAlgorithm

    model   = CombinatoAlgorithm(spc_path='/path/to/spc')
    results = model(signal_tensor, times_tensor)

    results['pos_spikes']        # (P, 64) tensor
    results['pos_times']         # (P,)    tensor — ms
    results['neg_spikes']        # (Q, 64) tensor
    results['neg_times']         # (Q,)    tensor — ms
    results['threshold']         # scalar
    results['pos_sort_idx']      # (P,) numpy uint16
    results['pos_match_idx']     # (P,) numpy int8  — 0=SPC,1=C5,2=C5b
    results['pos_distance']      # (P,) numpy float32
    results['pos_artifact_ids']  # list
    results['neg_sort_idx']
    results['neg_match_idx']
    results['neg_distance']
    results['neg_artifact_ids']
    results['timings']           # dict of per-block runtimes in seconds
"""

import os
import tempfile
import shutil
import numpy as np
import torch

from block import Block
from m6_pipeline        import CombinatoExtractor, save_to_h5
from c7_combinato_sorter import CombinatorSorter


class CombinatoAlgorithm(Block):
    """
    Full Combinato pipeline: raw signal -> spike waveforms -> sorted clusters.

    Args:
        spc_path (str):      Path to SPC binary.
        sample_rate (int):   Recording sample rate in Hz. Default 24000.
        work_dir (str):      Working dir for SPC temp files.
                             Auto-created and cleaned up if None.
        seed (float):        SPC random seed. Default 12345.0.
        run_sorting (bool):  Set False to run extraction only. Default True.
    """

    def __init__(self,
                 spc_path,
                 sample_rate=24000,
                 work_dir=None,
                 seed=12345.0,
                 run_sorting=True,
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

        self.extractor = CombinatoExtractor(sample_rate=sample_rate)

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

    def _get_work_dir(self):
        if self._auto_dir:
            return tempfile.mkdtemp(prefix='combinato_'), True
        os.makedirs(self.work_dir, exist_ok=True)
        return self.work_dir, False

    def _collect_timings(self):
        """Read .runtime_measure() from every sub-block after a forward pass."""
        timings = {}
        ext = self.extractor
        timings['M1_filter']     = ext.pre.runtime_measure()
        timings['M2_detect']     = ext.detector.runtime_measure()
        timings['M3_extract']    = ext.extractor.runtime_measure()
        timings['M4_upsample']   = ext.upsampler.runtime_measure()
        timings['M5_align']      = ext.aligner.runtime_measure()
        timings['M6_extraction'] = self.extractor.runtime_measure()

        if self.run_sorting:
            s = self.sorter
            timings['C1_wavelet']  = s.c1.runtime_measure()
            timings['C2_features'] = s.c2.runtime_measure()
            timings['C3_spc']      = s.c3.runtime_measure()
            timings['C4_clusters'] = s.c4.runtime_measure()
            timings['C5_match']    = s.c5.runtime_measure()
            timings['C5b_match']   = s.c5b.runtime_measure()
            timings['C6_artifact'] = s.c6.runtime_measure()

        return {k: (v if v is not None else 0.0) for k, v in timings.items()}

    def forward(self, signal, atimes):
        """
        Run the full pipeline.

        Args:
            signal: (N,) float64 tensor - raw scaled signal
            atimes: (N,) float64 tensor - ms timestamp per sample

        Returns:
            dict - see module docstring for all keys
        """
        print("\n[CombinatoAlgorithm] Stage 1: Extraction (M1-M6)")

        with torch.no_grad():
            extraction = self.extractor(signal, atimes)

        pos_spikes = extraction['pos_spikes']
        neg_spikes = extraction['neg_spikes']
        pos_times  = extraction['pos_times']
        neg_times  = extraction['neg_times']
        threshold  = extraction['threshold']

        print(f"  Threshold  : {threshold.item():.4f}")
        print(f"  Pos spikes : {pos_spikes.shape[0]}")
        print(f"  Neg spikes : {neg_spikes.shape[0]}")

        result = {
            'pos_spikes': pos_spikes,
            'pos_times':  pos_times,
            'neg_spikes': neg_spikes,
            'neg_times':  neg_times,
            'threshold':  threshold,
        }

        if not self.run_sorting:
            for sign in ('pos', 'neg'):
                n = pos_spikes.shape[0] if sign == 'pos' else neg_spikes.shape[0]
                result[f'{sign}_sort_idx']     = np.zeros(n, dtype=np.uint16)
                result[f'{sign}_match_idx']    = np.zeros(n, dtype=np.int8)
                result[f'{sign}_distance']     = np.zeros(n, dtype=np.float32)
                result[f'{sign}_artifact_ids'] = []
            result['timings'] = self._collect_timings()
            return result

        print("\n[CombinatoAlgorithm] Stage 2: Sorting (C1-C7)")

        work_dir, cleanup = self._get_work_dir()

        try:
            for sign, spikes_t in [('pos', pos_spikes), ('neg', neg_spikes)]:
                print(f"\n  --- {sign.upper()} polarity ---")
                with torch.no_grad():
                    sort_idx, match_idx, distance, artifact_ids = self.sorter(
                        spikes_t,
                        os.path.join(work_dir, sign),
                        sign=sign,
                    )
                result[f'{sign}_sort_idx']     = sort_idx
                result[f'{sign}_match_idx']    = match_idx
                result[f'{sign}_distance']     = distance
                result[f'{sign}_artifact_ids'] = artifact_ids

                n_clusters = len(np.unique(sort_idx[sort_idx > 0]))
                print(f"  Clusters   : {n_clusters}")
                print(f"  Unmatched  : {(sort_idx == 0).sum()}")
                print(f"  Artifacts  : {len(artifact_ids)}")

        finally:
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

        result['timings'] = self._collect_timings()
        return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_extraction(result, gt_h5_path):
    """Compare extraction output against ground-truth H5 file."""
    import tables
    h5 = tables.open_file(gt_h5_path, 'r')
    gt = {
        'pos_spikes': h5.root.pos.spikes[:],
        'neg_spikes': h5.root.neg.spikes[:],
    }
    h5.close()

    lines  = ["\n" + "="*60, "EXTRACTION VALIDATION", "="*60]
    checks = {}

    for sign in ('pos', 'neg'):
        our  = result[f'{sign}_spikes'].cpu().numpy()
        gt_s = gt[f'{sign}_spikes']
        pct  = abs(our.shape[0] - gt_s.shape[0]) / max(gt_s.shape[0], 1) * 100
        corr = float(np.corrcoef(our.mean(0), gt_s.mean(0))[0, 1]) if our.shape[0] > 0 else 0.0
        amp  = np.abs(our).max() / np.abs(gt_s).max() if gt_s.size > 0 else 1.0

        checks[f'{sign}_count'] = pct  < 5
        checks[f'{sign}_corr']  = corr > 0.99
        checks[f'{sign}_amp']   = 0.8  < amp < 1.2

        lines.append(f"\n  [{sign.upper()}]")
        lines.append(f"    Count : {our.shape[0]} vs {gt_s.shape[0]}  ({pct:.2f}%)  {'OK' if checks[f'{sign}_count'] else 'FAIL'}")
        lines.append(f"    Corr  : {corr:.4f}  {'OK' if checks[f'{sign}_corr'] else 'FAIL'}")
        lines.append(f"    Amp   : {amp:.4f}   {'OK' if checks[f'{sign}_amp'] else 'FAIL'}")

    passed = all(checks.values())
    lines += ["", "EXTRACTION PASSED" if passed else "EXTRACTION FAILED", "="*60]
    return passed, '\n'.join(lines)


def validate_sorting(result, sort_cat_h5_path):
    """Compare sorting output against sort_cat.h5."""
    import h5py
    with h5py.File(sort_cat_h5_path, 'r') as f:
        gt = {'classes': f['classes'][:], 'artifacts': f['artifacts'][:]}

    sort_idx     = result['pos_sort_idx']
    artifact_ids = result['pos_artifact_ids']

    lines  = ["\n" + "="*60, "SORTING VALIDATION (pos)", "="*60]
    checks = {}

    our_n = len(np.unique(sort_idx[sort_idx > 0]))
    gt_n  = len(np.unique(gt['classes'][gt['classes'] > 0]))
    checks['cluster_count'] = our_n == gt_n
    lines.append(f"\n  Clusters  : {our_n} vs {gt_n}  {'OK' if checks['cluster_count'] else 'DIFF'}")

    our_u = (sort_idx == 0).sum()
    gt_u  = (gt['classes'] == 0).sum()
    checks['unassigned'] = our_u == gt_u
    lines.append(f"  Unassigned: {our_u} vs {gt_u}  {'OK' if checks['unassigned'] else 'DIFF'}")

    if len(sort_idx) == len(gt['classes']):
        exact = np.array_equal(sort_idx.astype(np.uint16), gt['classes'].astype(np.uint16))
        checks['exact'] = exact
        lines.append(f"  Exact match: {'OK' if exact else 'DIFF'}")

    gt_art = sorted(gt['artifacts'][gt['artifacts'][:, 1] > 0, 0].tolist())
    checks['artifacts'] = sorted(artifact_ids) == gt_art
    lines.append(f"  Artifacts : {sorted(artifact_ids)} vs {gt_art}  {'OK' if checks['artifacts'] else 'DIFF'}")

    passed = all(checks.values())
    lines += ["", "SORTING PASSED" if passed else "SORTING FAILED", "="*60]
    return passed, '\n'.join(lines)


def plot_timings(our_timings, original_timings=None):
    """Bar chart of per-block runtimes."""
    import matplotlib.pyplot as plt

    stages   = list(our_timings.keys())
    our_vals = [our_timings[s] for s in stages]

    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.2), 5))

    if original_timings:
        x         = np.arange(len(stages))
        w         = 0.35
        orig_vals = [original_timings.get(s, 0) for s in stages]
        ax.bar(x - w/2, orig_vals, w, label='Original Combinato', color='steelblue')
        ax.bar(x + w/2, our_vals,  w, label='PyTorch Combinato',  color='tomato')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=30, ha='right')
        ax.legend()
    else:
        ax.bar(range(len(stages)), our_vals, color='tomato')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=30, ha='right')

    ax.set_ylabel('Time (s)')
    ax.set_title('Per-block runtime — PyTorch Combinato')
    fig.tight_layout()
    return fig


def save_results(result, output_h5_path):
    """Save extraction + sorting results to HDF5."""
    import h5py
    save_to_h5(result, output_h5_path.replace('.h5', '_extraction.h5'))
    with h5py.File(output_h5_path, 'w') as f:
        for sign in ('pos', 'neg'):
            grp      = f.create_group(sign)
            sort_idx = result[f'{sign}_sort_idx']
            grp.create_dataset('classes',  data=sort_idx.astype(np.uint16))
            grp.create_dataset('matches',  data=result[f'{sign}_match_idx'].astype(np.int8))
            grp.create_dataset('distance', data=result[f'{sign}_distance'].astype(np.float32))
            unique_c  = np.unique(sort_idx)
            art_ids   = set(result[f'{sign}_artifact_ids'])
            artifacts = np.zeros((len(unique_c), 2), dtype=np.int64)
            artifacts[:, 0] = unique_c
            for i, c in enumerate(unique_c):
                if c in art_ids:
                    artifacts[i, 1] = 1
            grp.create_dataset('artifacts', data=artifacts)
    print(f"Saved results to {output_h5_path}")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from scipy.io import loadmat

    parser = argparse.ArgumentParser()
    parser.add_argument('--matfile',       required=True)
    parser.add_argument('--spc-path',      required=True)
    parser.add_argument('--scale',         type=float, default=100.0)
    parser.add_argument('--output',        default='combinato_output.h5')
    parser.add_argument('--gt-extraction', default=None)
    parser.add_argument('--gt-sorting',    default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat    = loadmat(args.matfile)
    fdata  = mat['data'].ravel().astype(np.float64) * args.scale
    sr     = float(mat['sr'].ravel()[0]) if 'sr' in mat else 24000.0
    atimes = np.linspace(0, fdata.shape[0] / (sr / 1000), fdata.shape[0])

    signal_t = torch.tensor(fdata,  dtype=torch.float64).to(device)
    times_t  = torch.tensor(atimes, dtype=torch.float64).to(device)

    model = CombinatoAlgorithm(spc_path=args.spc_path).to(device)
    with torch.no_grad():
        result = model(signal_t, times_t)

    print("\nTimings:")
    for k, v in result['timings'].items():
        print(f"  {k:20s}: {v:.3f}s")

    if args.gt_extraction:
        passed, report = validate_extraction(result, args.gt_extraction)
        print(report)

    if args.gt_sorting:
        passed, report = validate_sorting(result, args.gt_sorting)
        print(report)

    save_results(result, args.output)
    plot_timings(result['timings']).savefig('timings.png')
    print("Saved timings.png")
