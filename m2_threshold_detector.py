"""
M2 - ThresholdDetector Module
==============================
Takes data_detected from M1 and finds spike locations.

Input:
    data_detected: (N,) tensor — output of M1 filter_detect

Output:
    pos_indices: (P,) tensor — sample indices of positive spike peaks
    neg_indices: (Q,) tensor — sample indices of negative spike peaks
    threshold:   scalar tensor
"""

import torch
import torch.nn as nn
import numpy as np

THRESHOLD_FACTOR   = 5
MAX_SPIKE_DURATION = 0.0015   # 1.5ms


class ThresholdDetector(nn.Module):
    def __init__(self, sample_rate=24000, threshold_factor=5, max_spike_duration=0.0015):
        super().__init__()
        self.sample_rate       = sample_rate
        self.threshold_factor  = threshold_factor
        self.max_spike_samples = int(max_spike_duration * sample_rate)  # 36 samples

    def compute_threshold(self, data_detected):
        """
        Step 1: robust noise estimate → threshold.
        median(|x|) / 0.6745 = robust std estimate for Gaussian noise.
        Spikes don't affect median because they are sparse.
        """
        noise_level = torch.median(torch.abs(data_detected)) / 0.6745
        return self.threshold_factor * noise_level

    def find_crossings(self, data_detected, threshold, sign):
        """
        Step 2: find where signal crosses threshold.

        torch.diff() on boolean mask gives:
            +1 where False→True  = signal enters threshold region (entry)
            -1 where True→False  = signal leaves threshold region (exit)

        Each spike = one entry + one exit = one (start, end) pair.
        Returns (K, 2) tensor of [start, end] pairs, or None if no crossings.
        """
        mask = (data_detected > threshold) if sign == 'pos' else (data_detected < -threshold)
        diff    = torch.diff(mask.to(torch.int32))
        entries = (diff ==  1).nonzero(as_tuple=True)[0]
        exits   = (diff == -1).nonzero(as_tuple=True)[0]

        n = min(len(entries), len(exits))
        if n == 0:
            return None

        return torch.stack([entries[:n], exits[:n]], dim=1)  # (K, 2)

    def filter_duration(self, borders):
        """
        Step 3: remove crossings longer than max_spike_samples (36).
        Anything longer is an artifact, not a real spike.
        """
        durations = borders[:, 1] - borders[:, 0]
        return borders[durations <= self.max_spike_samples]

    def find_peaks(self, data_detected, borders, sign):
        """
        Step 4: find exact peak sample within each crossing window.

        Vectorized — no loop over spikes:
            1. Build index matrix (M, W) where each row = window indices
            2. Gather all signal values at once
            3. argmax/argmin across each row simultaneously
            4. Convert local position → absolute signal index
        """
        if borders is None or len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)

        W       = self.max_spike_samples
        N       = data_detected.shape[0]
        starts  = borders[:, 0]                                         # (M,)
        offsets = torch.arange(W, device=data_detected.device)         # (W,)
        indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N-1)  # (M, W)

        windows     = data_detected[indices]                            # (M, W)
        local_peaks = windows.argmax(dim=1) if sign == 'pos' else windows.argmin(dim=1)
        return starts + local_peaks                                     # (M,) absolute indices

    def _detect_one(self, data_detected, threshold, sign):
        borders = self.find_crossings(data_detected, threshold, sign)
        if borders is None:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        borders = self.filter_duration(borders)
        if len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        return self.find_peaks(data_detected, borders, sign)

    def forward(self, data_detected):
        threshold   = self.compute_threshold(data_detected)
        pos_indices = self._detect_one(data_detected, threshold, 'pos')
        neg_indices = self._detect_one(data_detected, threshold, 'neg')
        return pos_indices, neg_indices, threshold


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("M2 THRESHOLD DETECTOR — VALIDATION (synthetic data)")
    print("=" * 60)

    SAMPLE_RATE = 24000
    N = 500_000
    np.random.seed(42)

    # Simulate post-filter signal: low noise baseline + planted spikes
    baseline = np.random.randn(N).astype(np.float64) * 7.0

    true_pos_times = [50000, 150000, 250000, 350000, 450000]
    true_neg_times = [100000, 200000, 300000, 400000]

    for t in true_pos_times:
        baseline[t:t+20] += np.hanning(20) * 60
    for t in true_neg_times:
        baseline[t:t+20] -= np.hanning(20) * 60

    # --- Original numpy logic ---
    print("\n[ORIGINAL] numpy detection...")
    noise_orig = np.median(np.abs(baseline)) / 0.6745
    thr_orig   = 5 * noise_orig
    print(f"  Threshold: {thr_orig:.4f}")

    def get_peaks_numpy(data, thr, find_max):
        mask    = data > thr if find_max else data < -thr
        diff    = np.diff(mask.astype(int))
        entries = np.where(diff ==  1)[0]
        exits   = np.where(diff == -1)[0]
        n       = min(len(entries), len(exits))
        peaks   = []
        for i in range(n):
            if (exits[i] - entries[i]) > 36:
                continue
            seg   = data[entries[i]:exits[i]]
            local = np.argmax(seg) if find_max else np.argmin(seg)
            peaks.append(entries[i] + local)
        return np.array(peaks)

    orig_pos = get_peaks_numpy(baseline, thr_orig, find_max=True)
    orig_neg = get_peaks_numpy(baseline, thr_orig, find_max=False)
    print(f"  Positive spikes: {len(orig_pos)}")
    print(f"  Negative spikes: {len(orig_neg)}")

    # --- PyTorch M2 ---
    print("\n[PYTORCH] M2 ThresholdDetector...")
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    detector = ThresholdDetector(SAMPLE_RATE).to(device)
    x        = torch.tensor(baseline, dtype=torch.float64).to(device)

    with torch.no_grad():
        pt_pos, pt_neg, pt_thr = detector(x)

    print(f"  Threshold: {pt_thr.item():.4f}")
    print(f"  Positive spikes: {len(pt_pos)}")
    print(f"  Negative spikes: {len(pt_neg)}")

    # --- Results ---
    print("\n[VALIDATION]")
    thr_diff  = abs(pt_thr.item() - thr_orig) / thr_orig * 100
    pos_match = len(pt_pos) == len(orig_pos)
    neg_match = len(pt_neg) == len(orig_neg)

    print(f"  Threshold diff  : {thr_diff:.6f}%  {'✓' if thr_diff < 0.01 else '✗'}")
    print(f"  Pos spike count : {len(orig_pos)} orig  vs  {len(pt_pos)} pytorch  {'✓' if pos_match else '✗'}")
    print(f"  Neg spike count : {len(orig_neg)} orig  vs  {len(pt_neg)} pytorch  {'✓' if neg_match else '✗'}")

    if pos_match and len(orig_pos) > 0:
        diff = np.max(np.abs(pt_pos.cpu().numpy() - orig_pos))
        print(f"  Pos peak idx diff : {diff} samples  {'✓' if diff <= 1 else '✗'}")
    if neg_match and len(orig_neg) > 0:
        diff = np.max(np.abs(pt_neg.cpu().numpy() - orig_neg))
        print(f"  Neg peak idx diff : {diff} samples  {'✓' if diff <= 1 else '✗'}")

    print("\n" + "="*60)
    all_pass = pos_match and neg_match and thr_diff < 0.01
    print("M2 PASSED ✓  — Ready to build M3" if all_pass else "M2 FAILED ✗")
    print("="*60)
