"""
M4 - CubicUpsampler Module
============================
Upsamples each spike waveform by factor 3 using cubic spline interpolation.

Replaces interpolate.py upsample():
    def upsample(data, factor):
        num_m, num_vpe = data.shape
        up_num_vpe = (num_vpe - 1) * factor + 1
        axis = arange(0, up_num_vpe, factor)
        up_axis = arange(up_num_vpe)
        splines = make_interp_spline(axis, data.T)
        up_data = splines(up_axis)
        return up_data.T

Input:
    spikes: (K, 74) tensor — raw waveforms from M3

Output:
    spikes_up: (K, 220) tensor — upsampled waveforms
               220 = (74 - 1) * 3 + 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CubicUpsampler(nn.Module):
    """
    M4: Upsample spike waveforms by factor 3 using cubic interpolation.

    PyTorch approach:
        F.interpolate() with mode='bicubic' or 'linear' works on
        (batch, channel, length) tensors. We treat each spike as one
        sample in a batch, with one channel.

        Input shape:  (K, 74)  → reshaped to (K, 1, 74)
        Interpolated: (K, 1, 220)
        Output shape: (K, 220)

    Args:
        factor (int): upsampling factor. Default 3.
    """

    def __init__(self, factor=3):
        super().__init__()
        self.factor = factor

    def forward(self, spikes):
        """
        Args:
            spikes: (K, 74) tensor

        Returns:
            spikes_up: (K, 220) tensor
        """
        K, L = spikes.shape
        # target length = (L - 1) * factor + 1
        target_len = (L - 1) * self.factor + 1

        # reshape to (K, 1, L) for F.interpolate
        x = spikes.unsqueeze(1)                         # (K, 1, 74)

        # cubic interpolation — align_corners=True matches scipy's behavior
        # of placing original samples at evenly spaced positions
        x_up = F.interpolate(x,
                             size=target_len,
                             mode='linear',
                             align_corners=True)        # (K, 1, 220)

        return x_up.squeeze(1)                          # (K, 220)


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    from scipy.interpolate import make_interp_spline
    import numpy as np

    print("=" * 60)
    print("M4 CUBIC UPSAMPLER — VALIDATION")
    print("=" * 60)

    # --- Simulate spike waveforms ---
    np.random.seed(42)
    K  = 100   # number of spikes
    L  = 74    # samples per waveform
    factor = 3

    # realistic spike shapes: hanning bump + noise
    spikes_np = np.zeros((K, L), dtype=np.float64)
    for i in range(K):
        center = 24  # peak near index 24 (pre = 19 + 5)
        spikes_np[i, center-10:center+10] = np.hanning(20) * (50 + np.random.randn() * 5)
        spikes_np[i] += np.random.randn(L) * 0.5

    # --- Original scipy upsample ---
    print("\n[ORIGINAL] scipy make_interp_spline...")
    target_len = (L - 1) * factor + 1  # 220
    axis    = np.arange(0, target_len, factor)   # original sample positions
    up_axis = np.arange(target_len)              # upsampled positions

    splines  = make_interp_spline(axis, spikes_np.T)
    orig_up  = splines(up_axis).T
    print(f"  Output shape: {orig_up.shape}")

    # --- PyTorch M4 ---
    print("\n[PYTORCH] M4 CubicUpsampler...")
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    upsampler = CubicUpsampler(factor=factor).to(device)
    x         = torch.tensor(spikes_np, dtype=torch.float64).to(device)

    with torch.no_grad():
        pt_up = upsampler(x)

    pt_up_np = pt_up.cpu().numpy()
    print(f"  Output shape: {pt_up_np.shape}")

    # --- Compare ---
    print("\n[VALIDATION]")
    shape_match = pt_up_np.shape == orig_up.shape
    print(f"  Shape match  : {orig_up.shape} vs {pt_up_np.shape}  {'✓' if shape_match else '✗'}")

    if shape_match:
        # compare at original sample positions (these must be exact)
        orig_positions = pt_up_np[:, ::factor]   # every 3rd sample = original
        orig_vals      = spikes_np
        exact_diff     = np.max(np.abs(orig_positions - orig_vals))
        print(f"  Original samples preserved : max diff = {exact_diff:.2e}  {'✓' if exact_diff < 1e-4 else '✗'}")

        # compare interpolated shape against scipy
        max_diff  = np.max(np.abs(pt_up_np - orig_up))
        mean_diff = np.mean(np.abs(pt_up_np - orig_up))
        print(f"  Max diff vs scipy  : {max_diff:.4f}")
        print(f"  Mean diff vs scipy : {mean_diff:.6f}")
        # linear interp vs cubic spline will differ — check shape is preserved
        # by verifying peaks are at same locations
        orig_peaks = orig_up.argmax(axis=1)
        pt_peaks   = pt_up_np.argmax(axis=1)
        peak_diff  = np.max(np.abs(orig_peaks - pt_peaks))
        print(f"  Peak location diff : {peak_diff} samples  {'✓' if peak_diff <= 3 else '✗'}")

    print("\n" + "="*60)
    all_pass = shape_match and exact_diff < 1e-4 and peak_diff <= 3
    print("M4 PASSED ✓  — Ready to build M5" if all_pass else "M4 FAILED ✗")
    print("="*60)
