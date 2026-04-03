"""
Full Pipeline Test — M1 through M6
=====================================
Run this locally where simulation_5.mat lives.

Usage:
    python test_full_pipeline.py --matfile simulation_5.mat --scale 100 --h5file simulation_5\data_simulation_5.h5

What this tests:
    1. Full pipeline produces correct spike counts
    2. Waveform shapes match H5 ground truth (correlation > 0.99)
    3. Timestamps are in the right range
    4. Output H5 file is written correctly
    5. Total pipeline timing
"""

import argparse
import time
import numpy as np
import torch
import tables
from scipy.io import loadmat

from m6_pipeline import CombinatoExtractor, save_to_h5

GT_POS_SPIKES = 8968
GT_NEG_SPIKES = 5436


def load_mat(path, scale_factor):
    mat    = loadmat(path)
    fdata  = mat['data'].ravel().astype(np.float64)
    sr     = float(mat['sr'].ravel()[0]) if 'sr' in mat else 24000.0
    fdata *= scale_factor
    atimes = np.linspace(0, fdata.shape[0] / (sr / 1000), fdata.shape[0])
    print(f"  Loaded  : {path}")
    print(f"  Samples : {fdata.shape[0]:,}")
    print(f"  SR      : {sr} Hz")
    print(f"  Scaled  : x{scale_factor}")
    return fdata, atimes, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matfile', required=True)
    parser.add_argument('--h5file',  required=True)
    parser.add_argument('--scale',   type=float, default=100.0)
    parser.add_argument('--device',  default='auto')
    parser.add_argument('--save',    default='pytorch_output.h5',
                        help='path to save output H5 file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)

    print("\n" + "="*60)
    print("FULL PIPELINE TEST — M1 through M6")
    print("="*60)

    # ---- Load ----
    print("\n[1] Loading data...")
    fdata, atimes, sr = load_mat(args.matfile, args.scale)

    # ---- Load ground truth ----
    print("\n[2] Loading H5 ground truth...")
    h5 = tables.open_file(args.h5file, 'r')
    gt_pos_spikes = h5.root.pos.spikes[:]
    gt_neg_spikes = h5.root.neg.spikes[:]
    gt_pos_times  = h5.root.pos.times[:]
    gt_neg_times  = h5.root.neg.times[:]
    h5.close()
    print(f"  GT pos : {gt_pos_spikes.shape}  time range: {gt_pos_times.min():.1f} - {gt_pos_times.max():.1f} ms")
    print(f"  GT neg : {gt_neg_spikes.shape}  time range: {gt_neg_times.min():.1f} - {gt_neg_times.max():.1f} ms")

    # ---- Build pipeline ----
    print(f"\n[3] Building CombinatoExtractor on {device}...")
    pipeline = CombinatoExtractor(sample_rate=int(sr)).to(device)
    signal   = torch.tensor(fdata,  dtype=torch.float64).to(device)
    times    = torch.tensor(atimes, dtype=torch.float64).to(device)

    # ---- Run ----
    print("\n[4] Running full pipeline...")
    t0 = time.time()
    with torch.no_grad():
        result = pipeline(signal, times)
    total_time = time.time() - t0

    pos_spikes = result['pos_spikes'].cpu().numpy()
    neg_spikes = result['neg_spikes'].cpu().numpy()
    pos_times  = result['pos_times'].cpu().numpy()
    neg_times  = result['neg_times'].cpu().numpy()
    threshold  = result['threshold'].item()

    print(f"  Total time  : {total_time:.2f}s")
    print(f"  Threshold   : {threshold:.4f}")
    print(f"  Pos spikes  : {pos_spikes.shape}")
    print(f"  Neg spikes  : {neg_spikes.shape}")

    # ---- Validation ----
    print("\n[5] Validation results...")

    # spike counts
    pos_pct = abs(pos_spikes.shape[0] - GT_POS_SPIKES) / GT_POS_SPIKES * 100
    neg_pct = abs(neg_spikes.shape[0] - GT_NEG_SPIKES) / GT_NEG_SPIKES * 100
    print(f"\n  Pos count {pos_spikes.shape[0]:5d} vs GT {GT_POS_SPIKES} : {pos_pct:.2f}%  {'✓' if pos_pct < 5 else '✗'}")
    print(f"  Neg count {neg_spikes.shape[0]:5d} vs GT {GT_NEG_SPIKES} : {neg_pct:.2f}%  {'✓' if neg_pct < 5 else '✗'}")

    # waveform shape correlation
    pos_corr = np.corrcoef(pos_spikes.mean(0), gt_pos_spikes.mean(0))[0, 1]
    neg_corr = np.corrcoef(neg_spikes.mean(0), gt_neg_spikes.mean(0))[0, 1]
    print(f"\n  Pos mean waveform corr : {pos_corr:.4f}  {'✓' if pos_corr > 0.99 else '✗'}")
    print(f"  Neg mean waveform corr : {neg_corr:.4f}  {'✓' if neg_corr > 0.99 else '✗'}")

    # amplitude
    pos_amp_ratio = np.abs(pos_spikes).max() / np.abs(gt_pos_spikes).max()
    neg_amp_ratio = np.abs(neg_spikes).max() / np.abs(gt_neg_spikes).max()
    print(f"\n  Pos amplitude ratio    : {pos_amp_ratio:.4f}  {'✓' if 0.8 < pos_amp_ratio < 1.2 else '✗'}")
    print(f"  Neg amplitude ratio    : {neg_amp_ratio:.4f}  {'✓' if 0.8 < neg_amp_ratio < 1.2 else '✗'}")

    # timestamps
    pos_time_ok = (pos_times.min() >= 0 and
                   pos_times.max() <= atimes.max() + 1)
    neg_time_ok = (neg_times.min() >= 0 and
                   neg_times.max() <= atimes.max() + 1)
    print(f"\n  Pos times range : {pos_times.min():.1f} - {pos_times.max():.1f} ms  {'✓' if pos_time_ok else '✗'}")
    print(f"  Neg times range : {neg_times.min():.1f} - {neg_times.max():.1f} ms  {'✓' if neg_time_ok else '✗'}")

    # ---- Save output ----
    print(f"\n[6] Saving output to {args.save}...")
    save_to_h5(result, args.save)

    # verify saved file
    h5_out = tables.open_file(args.save, 'r')
    saved_pos = h5_out.root.pos.spikes[:]
    saved_neg = h5_out.root.neg.spikes[:]
    h5_out.close()
    save_ok = (saved_pos.shape == pos_spikes.shape and
               saved_neg.shape == neg_spikes.shape)
    print(f"  Saved pos : {saved_pos.shape}  {'✓' if save_ok else '✗'}")
    print(f"  Saved neg : {saved_neg.shape}  {'✓' if save_ok else '✗'}")

    # ---- Timing summary ----
    print(f"\n[7] Timing summary...")
    print(f"  Total pipeline : {total_time:.2f}s  ({14_400_000/total_time/1000:.1f}k samples/sec)")
    print(f"  Device         : {device}")

    # ---- Final summary ----
    print("\n" + "="*60)
    all_pass = (pos_pct < 5 and neg_pct < 5 and
                pos_corr > 0.99 and neg_corr > 0.99 and
                pos_time_ok and neg_time_ok and save_ok)
    if all_pass:
        print("FULL PIPELINE PASSED ✓")
        print("PyTorch Combinato extraction matches original.")
    else:
        print("FAILED ✗  — Check output above")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
