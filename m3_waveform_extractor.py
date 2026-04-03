"""
M3 - WaveformExtractor Module
==============================
Takes spike peak indices from M2 and cuts out 64-sample waveforms
from the filtered signal.

Replaces this logic in extract_spikes.py:
    data_extract = filt.filter_extract(data)
    extract_indices = [
        range(maxima[i] - pre_indices - 5, maxima[i] + post_indices + 5)
        for i in range(len(maxima))]
    spikes = np.zeros((len(extract_indices), indices_per_spike + 10))
    for i, spike in enumerate(extract_indices):
        spikes[i] = data_extract[extract_indices[i]]

Input:
    data_denoised:  (N,) tensor  — from M1 forward() step 1
    peak_indices:   (K,) tensor  — from M2, one polarity at a time
    sign:           'pos' or 'neg'

Output:
    spikes: (K, 64) tensor — waveform for each spike
    valid:  (K,)   bool tensor — False for spikes too close to signal border
"""

import torch
import torch.nn as nn
import numpy as np

# --- Constants (mirrors extract_spikes.py options) ---
INDICES_PER_SPIKE = 64   # final waveform length
INDEX_MAXIMUM     = 19   # peak position within waveform (0-indexed)
BORDER_PAD        = 5    # extra samples on each side before upsampling


class WaveformExtractor(nn.Module):
    """
    M3: Extract 64-sample waveforms around each detected spike.

    Steps:
        1. Apply filter_extract to denoised signal (wider 300-3000 Hz bandpass)
        2. Build index matrix — one row per spike, 74 columns (64 + 5 pad each side)
        3. Gather all waveforms simultaneously (no loop)
        4. Remove spikes too close to signal borders
        5. Flip negative spikes (multiply by -1) so all peaks point up

    Args:
        preprocessor: the M1 Preprocessor instance (needed for filter_extract)
        indices_per_spike (int): waveform length. Default 64.
        index_maximum (int): expected peak position. Default 19.
        border_pad (int): extra samples around window. Default 5.
    """

    def __init__(self,
                 preprocessor,
                 indices_per_spike=64,
                 index_maximum=19,
                 border_pad=5):
        super().__init__()
        self.preprocessor      = preprocessor
        self.indices_per_spike = indices_per_spike
        self.index_maximum     = index_maximum
        self.border_pad        = border_pad
        # total window width extracted before downsampling
        self.window_size = indices_per_spike + 2 * border_pad  # 74

    def forward(self, data_denoised, peak_indices, sign):
        """
        Args:
            data_denoised: (N,) tensor — from M1
            peak_indices:  (K,) tensor — spike peak sample indices from M2
            sign:          'pos' or 'neg'

        Returns:
            spikes: (M, 74) tensor — raw extracted windows (M <= K, valid only)
            valid:  (K,)    bool tensor — which spikes passed border check
            data_extract: (N,) tensor — the filtered signal (needed by M4/M5)
        """
        N  = data_denoised.shape[0]
        K  = peak_indices.shape[0]
        pre  = self.index_maximum + self.border_pad   # 24 samples before peak
        post = self.indices_per_spike - self.index_maximum + self.border_pad  # 50 samples after

        # Step 1: apply extract filter (300-3000 Hz) to denoised signal
        data_extract = self.preprocessor.filter_extract(data_denoised)  # (N,)

        # Step 2: border check
        # spikes too close to start or end of signal can't have full 74-sample window
        valid = (peak_indices >= pre) & (peak_indices <= N - post - 1)  # (K,)
        peak_indices = peak_indices[valid]                               # (M,)
        M = peak_indices.shape[0]

        if M == 0:
            empty = torch.zeros((0, self.window_size),
                                dtype=data_extract.dtype,
                                device=data_extract.device)
            return empty, valid, data_extract

        # Step 3: build index matrix (M, 74) — vectorized, no loop
        # each row contains the absolute signal indices for one spike window
        offsets = torch.arange(-pre, post,
                               device=data_extract.device)          # (74,)
        indices = peak_indices.unsqueeze(1) + offsets.unsqueeze(0) # (M, 74)

        # Step 4: gather all waveforms at once
        spikes = data_extract[indices]                              # (M, 74)

        # Step 5: flip negative spikes so peaks point upward
        # (makes alignment in M5 consistent for both polarities)
        if sign == 'neg':
            spikes = spikes * -1

        return spikes, valid, data_extract


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("M3 WAVEFORM EXTRACTOR — VALIDATION (synthetic data)")
    print("=" * 60)

    import numpy as np
    from scipy.signal import ellip, filtfilt as scipy_filtfilt

    SAMPLE_RATE = 24000
    N = 500_000
    np.random.seed(42)

    # Simulate denoised signal with planted spikes
    baseline = np.random.randn(N).astype(np.float64) * 7.0
    spike_times = [50000, 150000, 250000, 350000, 450000]
    for t in spike_times:
        baseline[t:t+20] += np.hanning(20) * 60

    # --- Original numpy extraction ---
    print("\n[ORIGINAL] numpy waveform extraction...")
    ts = 1.0 / SAMPLE_RATE
    b_e, a_e = ellip(2, 0.1, 40, (2*ts*300, 2*ts*3000), 'bandpass')
    data_extract_orig = scipy_filtfilt(b_e, a_e, baseline)

    # simulate peak indices (as M2 would give)
    peak_indices_np = np.array(spike_times) + 10  # peak near center of hanning

    PRE  = 19 + 5   # 24
    POST = 64 - 19 + 5  # 50

    orig_spikes = []
    for idx in peak_indices_np:
        if idx >= PRE and idx <= N - POST - 1:
            orig_spikes.append(data_extract_orig[idx - PRE: idx + POST])
    orig_spikes = np.array(orig_spikes)
    print(f"  Extracted: {orig_spikes.shape}")

    # --- PyTorch M3 ---
    print("\n[PYTORCH] M3 WaveformExtractor...")
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from m1_preprocessor import Preprocessor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    pre  = Preprocessor(SAMPLE_RATE).to(device)
    extr = WaveformExtractor(pre).to(device)

    x_denoised    = torch.tensor(baseline, dtype=torch.float64).to(device)
    peak_idx_tens = torch.tensor(peak_indices_np, dtype=torch.long).to(device)

    with torch.no_grad():
        pt_spikes, valid, data_extract_pt = extr(x_denoised, peak_idx_tens, 'pos')

    print(f"  Extracted: {pt_spikes.shape}")

    # --- Compare ---
    print("\n[VALIDATION]")
    shape_match = pt_spikes.shape == orig_spikes.shape
    print(f"  Shape match : {orig_spikes.shape} vs {pt_spikes.shape}  {'✓' if shape_match else '✗'}")

    if shape_match:
        max_diff = np.max(np.abs(pt_spikes.cpu().numpy() - orig_spikes))
        print(f"  Max waveform diff : {max_diff:.6f}  {'✓' if max_diff < 1.0 else '✗'}")

    valid_count = valid.sum().item()
    print(f"  Valid spikes : {valid_count} / {len(peak_indices_np)}  {'✓' if valid_count == len(orig_spikes) else '✗'}")

    print("\n" + "="*60)
    all_pass = shape_match and max_diff < 1.0
    print("M3 PASSED ✓  — Ready to build M4" if all_pass else "M3 FAILED ✗")
    print("="*60)
