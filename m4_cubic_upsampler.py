"""
M4 - CubicUpsampler Module
============================
Upsamples each spike waveform by factor 3 using linear interpolation.
"""

import torch
import torch.nn.functional as F
from block import Block


class CubicUpsampler(Block):
    def __init__(self, factor=3):
        super().__init__()
        self.factor = factor

    def forward(self, spikes):
        K, L = spikes.shape
        target_len = (L - 1) * self.factor + 1
        x    = spikes.unsqueeze(1)
        x_up = F.interpolate(x, size=target_len, mode='linear', align_corners=True)
        return x_up.squeeze(1)
