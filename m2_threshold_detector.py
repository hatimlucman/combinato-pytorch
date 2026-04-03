"""
M2 - ThresholdDetector Module
==============================
Takes data_detected from M1 and finds spike locations.
"""

import torch
import numpy as np
from block import Block

THRESHOLD_FACTOR   = 5
MAX_SPIKE_DURATION = 0.0015


class ThresholdDetector(Block):
    def __init__(self, sample_rate=24000, threshold_factor=5, max_spike_duration=0.0015):
        super().__init__()
        self.sample_rate       = sample_rate
        self.threshold_factor  = threshold_factor
        self.max_spike_samples = int(max_spike_duration * sample_rate)

    def compute_threshold(self, data_detected):
        noise_level = torch.median(torch.abs(data_detected)) / 0.6745
        return self.threshold_factor * noise_level

    def find_crossings(self, data_detected, threshold, sign):
        mask = (data_detected > threshold) if sign == 'pos' else (data_detected < -threshold)
        diff    = torch.diff(mask.to(torch.int32))
        entries = (diff ==  1).nonzero(as_tuple=True)[0]
        exits   = (diff == -1).nonzero(as_tuple=True)[0]
        n = min(len(entries), len(exits))
        if n == 0:
            return None
        return torch.stack([entries[:n], exits[:n]], dim=1)

    def filter_duration(self, borders):
        durations = borders[:, 1] - borders[:, 0]
        return borders[durations <= self.max_spike_samples]

    def find_peaks(self, data_detected, borders, sign):
        if borders is None or len(borders) == 0:
            return torch.zeros(0, dtype=torch.long, device=data_detected.device)
        W       = self.max_spike_samples
        N       = data_detected.shape[0]
        starts  = borders[:, 0]
        offsets = torch.arange(W, device=data_detected.device)
        indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, N-1)
        windows     = data_detected[indices]
        local_peaks = windows.argmax(dim=1) if sign == 'pos' else windows.argmin(dim=1)
        return starts + local_peaks

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
