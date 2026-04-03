"""
C1 - WaveletFeatureExtractor
==============================
Transforms spike waveforms into wavelet feature vectors.
"""

import torch
import numpy as np
import pywt
from block import Block

WAVELET = 'haar'
LEVEL   = 4


class WaveletFeatureExtractor(Block):
    def __init__(self, wavelet=WAVELET, level=LEVEL):
        super().__init__()
        self.wavelet_name = wavelet
        self.level        = level
        self.wavelet      = pywt.Wavelet(wavelet)

        dummy             = np.zeros(64, dtype=np.float32)
        coeffs            = pywt.wavedec(dummy, self.wavelet, level=self.level)
        self.feature_size = sum(c.shape[0] for c in coeffs)
        self.coeff_lengths = [c.shape[0] for c in coeffs]

    def forward(self, spikes):
        spikes_np = spikes.cpu().numpy().astype(np.float32)
        N         = spikes_np.shape[0]
        output    = np.empty((N, self.feature_size), dtype=np.float32)
        for i in range(N):
            coeffs    = pywt.wavedec(spikes_np[i], self.wavelet, level=self.level)
            output[i] = np.hstack(coeffs)
        return torch.tensor(output, dtype=torch.float32, device=spikes.device)
