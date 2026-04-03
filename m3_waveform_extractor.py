"""
M3 - WaveformExtractor Module
==============================
Takes spike peak indices from M2 and cuts out waveforms from the filtered signal.
"""

import torch
import numpy as np
from block import Block

INDICES_PER_SPIKE = 64
INDEX_MAXIMUM     = 19
BORDER_PAD        = 5


class WaveformExtractor(Block):
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
        self.window_size = indices_per_spike + 2 * border_pad

    def forward(self, data_denoised, peak_indices, sign):
        N  = data_denoised.shape[0]
        K  = peak_indices.shape[0]
        pre  = self.index_maximum + self.border_pad
        post = self.indices_per_spike - self.index_maximum + self.border_pad

        data_extract = self.preprocessor.filter_extract(data_denoised)

        valid = (peak_indices >= pre) & (peak_indices <= N - post - 1)
        peak_indices = peak_indices[valid]
        M = peak_indices.shape[0]

        if M == 0:
            empty = torch.zeros((0, self.window_size),
                                dtype=data_extract.dtype,
                                device=data_extract.device)
            return empty, valid, data_extract

        offsets = torch.arange(-pre, post, device=data_extract.device)
        indices = peak_indices.unsqueeze(1) + offsets.unsqueeze(0)
        spikes  = data_extract[indices]

        if sign == 'neg':
            spikes = spikes * -1

        return spikes, valid, data_extract
