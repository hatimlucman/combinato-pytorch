"""
M5 - PeakAligner Module
=========================
Aligns upsampled waveforms so peaks land at the same index,
removes outliers, then downsamples back to 64 samples.
"""

import torch
import numpy as np
from block import Block


class PeakAligner(Block):
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
        self.width             = border_pad

        self.center     = (index_maximum + border_pad) * factor  # 72
        self.low        = factor                                  # 3
        self.high       = factor                                  # 3
        self.lo         = self.center - self.width * self.low    # 57
        self.hi         = self.center + self.width * self.high   # 87
        self.new_center = self.center - self.width * self.low    # 57
        self.aligned_len = (indices_per_spike + 2 * border_pad) * factor \
                           - self.width * self.low - self.width * self.high

    def forward(self, spikes_up):
        K, L   = spikes_up.shape
        device = spikes_up.device

        hi = min(self.hi, L)
        window      = spikes_up[:, self.lo:hi]
        local_peaks = window.argmax(dim=1)
        index_max   = local_peaks + self.lo

        starts  = index_max - self.center + self.width * self.low
        ends    = starts + self.aligned_len
        in_bounds = (starts >= 0) & (ends <= L)

        offsets     = torch.arange(self.aligned_len, device=device)
        safe_starts = starts.clamp(0, L - self.aligned_len)
        all_indices = safe_starts.unsqueeze(1) + offsets.unsqueeze(0)
        all_indices = all_indices.clamp(0, L - 1)
        row_idx     = torch.arange(K, device=device).unsqueeze(1).expand(K, self.aligned_len)
        aligned     = spikes_up[row_idx, all_indices]

        index_max_aligned = aligned.argmax(dim=1)
        at_center         = (index_max_aligned == self.new_center)
        valid             = at_center & in_bounds
        removed_mask      = ~valid

        if valid.sum() == 0:
            empty = torch.zeros((0, self.indices_per_spike),
                                dtype=spikes_up.dtype, device=device)
            return empty, removed_mask

        cleaned  = aligned[valid]
        ds_idx   = torch.arange(self.indices_per_spike, device=device) * self.factor
        spikes_final = cleaned[:, ds_idx]

        return spikes_final, removed_mask
