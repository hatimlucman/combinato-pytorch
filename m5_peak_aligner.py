"""
M5 - PeakAligner Module
=========================
Aligns all upsampled waveforms so peaks land at the same index,
removes outliers, then downsamples back to 64 samples.

Exactly mirrors interpolate.py: align() → clean() → downsample()

Input:
    spikes_up: (K, 220) tensor — from M4

Output:
    spikes_final: (M, 64) tensor — aligned, cleaned, downsampled
    removed_mask: (K,)    bool tensor — True where spike was removed
"""

import torch
import torch.nn as nn
import numpy as np


class PeakAligner(nn.Module):
    """
    M5: Align waveform peaks, remove outliers, downsample to 64 samples.

    Mirrors interpolate.py exactly:

    align():
        center = (index_maximum + border_pad) * factor = 72
        search: [center - width*low, center + width*high] = [57, 87]
        shift each waveform so found peak lands at new_center = center - width*low = 57
        output length = 220 - width*low - width*high = 190

    clean():
        after alignment, remove any waveform whose argmax != new_center (57)

    downsample():
        pick every factor-th sample: indices = arange(64) * 3
        output shape: (M, 64), peaks at index 19

    Args:
        factor (int): upsampling factor. Default 3.
        indices_per_spike (int): final waveform length. Default 64.
        index_maximum (int): expected peak in final waveform. Default 19.
        border_pad (int): padding added in M3. Default 5.
    """

    def __init__(self,
                 factor=3,
                 indices_per_spike=64,
                 index_maximum=19,
                 border_pad=5):
        super().__init__()
        self.factor            = factor
        self.indices_per_spike = indices_per_spike
        self.index_maximum     = index_maximum
        self.border_pad        = border_pad
        self.width             = border_pad          # width=5 in interpolate.py

        # mirrors interpolate.py constants exactly
        self.center    = (index_maximum + border_pad) * factor  # 72
        self.low       = factor                                  # 3
        self.high      = factor                                  # 3
        self.lo        = self.center - self.width * self.low    # 57
        self.hi        = self.center + self.width * self.high   # 87
        self.new_center = self.center - self.width * self.low   # 57
        # aligned output length = 220 - 15 - 15 = 190
        self.aligned_len = (indices_per_spike + 2 * border_pad) * factor \
                           - self.width * self.low - self.width * self.high

    def forward(self, spikes_up):
        """
        Args:
            spikes_up: (K, 220) tensor

        Returns:
            spikes_final: (M, 64) tensor
            removed_mask: (K,)    bool tensor
        """
        K, L   = spikes_up.shape
        device = spikes_up.device

        # ---- Step 1: align() ----
        # find peak in search window for each waveform
        hi = min(self.hi, L)
        window     = spikes_up[:, self.lo:hi]          # (K, 30)
        local_peaks = window.argmax(dim=1)             # (K,) position within window
        index_max  = local_peaks + self.lo             # (K,) absolute position

        # shift each waveform so peak lands at new_center (57)
        # start = index_max - center + width*low = index_max - 72 + 15 = index_max - 57
        starts  = index_max - self.center + self.width * self.low   # (K,)
        ends    = starts + self.aligned_len                         # (K,)

        # bounds check for extraction
        in_bounds = (starts >= 0) & (ends <= L)

        # vectorized extraction: build index matrix (K, 190)
        offsets = torch.arange(self.aligned_len, device=device)        # (190,)
        safe_starts = starts.clamp(0, L - self.aligned_len)
        all_indices = safe_starts.unsqueeze(1) + offsets.unsqueeze(0)  # (K, 190)
        all_indices = all_indices.clamp(0, L - 1)
        row_idx     = torch.arange(K, device=device).unsqueeze(1).expand(K, self.aligned_len)
        aligned     = spikes_up[row_idx, all_indices]                  # (K, 190)

        # ---- Step 2: clean() ----
        # after alignment, peak should be at new_center (57) in the 190-sample waveform
        # mirrors: index_max = data.argmax(1); removed = (index_max != center)
        index_max_aligned = aligned.argmax(dim=1)                      # (K,)
        at_center         = (index_max_aligned == self.new_center)     # (K,)
        valid             = at_center & in_bounds                      # (K,)
        removed_mask      = ~valid                                     # (K,)

        if valid.sum() == 0:
            empty = torch.zeros((0, self.indices_per_spike),
                                dtype=spikes_up.dtype, device=device)
            return empty, removed_mask

        cleaned = aligned[valid]                                       # (M, 190)

        # ---- Step 3: downsample() ----
        # index = arange(64) * 3 = [0, 3, 6, ..., 189]
        ds_idx       = torch.arange(self.indices_per_spike, device=device) * self.factor
        spikes_final = cleaned[:, ds_idx]                             # (M, 64)

        return spikes_final, removed_mask


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    from scipy.interpolate import make_interp_spline
    import numpy as np

    print("=" * 60)
    print("M5 PEAK ALIGNER — VALIDATION")
    print("=" * 60)

    np.random.seed(42)
    K = 500; L = 74; factor = 3

    spikes_np = np.zeros((K, L), dtype=np.float64)
    for i in range(K):
        offset = np.random.randint(-5, 6)
        center = np.clip(24 + offset, 10, 60)
        spikes_np[i, center-8:center+8] = np.hanning(16) * (50 + np.random.randn()*3)
        spikes_np[i] += np.random.randn(L) * 0.5

    # scipy upsample
    target_len = (L - 1) * factor + 1
    axis    = np.arange(0, target_len, factor)
    splines = make_interp_spline(axis, spikes_np.T)
    up_np   = splines(np.arange(target_len)).T

    # --- original pipeline ---
    center_o = 72; width_o = 5; low_o = 3; high_o = 3
    lo_o = center_o - width_o*low_o; hi_o = center_o + width_o*high_o
    new_center_o = center_o - width_o*low_o  # 57
    num_vpe = up_np.shape[1]; new_len = num_vpe - width_o*low_o - width_o*high_o

    idx_max = up_np[:, lo_o:hi_o].argmax(1) + lo_o
    aligned_o = np.zeros((K, new_len))
    for i in range(K):
        s = idx_max[i] - center_o + width_o*low_o
        e = idx_max[i] - center_o + num_vpe - width_o*high_o
        if 0 <= s and e <= num_vpe:
            aligned_o[i] = up_np[i, s:e]

    idx_al = aligned_o.argmax(1)
    removed_o = (idx_al != new_center_o)
    cleaned_o = aligned_o[~removed_o]
    ds_idx_o  = np.arange(64) * factor
    final_o   = cleaned_o[:, ds_idx_o]
    print(f"\n[ORIGINAL]  kept={final_o.shape[0]}  removed={removed_o.sum()}")

    # --- pytorch M5 ---
    device  = torch.device('cpu')
    aligner = PeakAligner().to(device)
    x_up    = torch.tensor(up_np, dtype=torch.float64).to(device)

    with torch.no_grad():
        pt_final, pt_removed = aligner(x_up)

    pt_np = pt_final.cpu().numpy()
    print(f"[PYTORCH]   kept={pt_np.shape[0]}  removed={pt_removed.sum().item()}")

    print(f"\n[VALIDATION]")
    count_diff = abs(pt_np.shape[0] - final_o.shape[0])
    print(f"  Count diff     : {count_diff}  {'✓' if count_diff == 0 else '✗'}")
    if count_diff == 0 and pt_np.shape[0] > 0:
        max_diff = np.abs(pt_np - final_o).max()
        print(f"  Max value diff : {max_diff:.6f}  {'✓' if max_diff < 0.01 else '✗'}")
    pt_peaks = pt_np.argmax(1)
    peak_ok  = (np.abs(pt_peaks - 19) <= 1).mean()
    print(f"  Peaks at 19    : {peak_ok*100:.1f}%  {'✓' if peak_ok > 0.95 else '✗'}")

    print("\n" + "="*60)
    print("M5 PASSED ✓" if count_diff == 0 and peak_ok > 0.95 else "M5 FAILED ✗")
    print("="*60)
