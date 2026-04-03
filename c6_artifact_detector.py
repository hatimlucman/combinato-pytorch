"""
C6 - ArtifactDetector
=======================
Scores each cluster for artifact-like waveform properties.
"""

import torch
import numpy as np
from block import Block

ARTIFACT_CRITERIA = {
    'maxima':           5,
    'maxima_1_2_ratio': 2,
    'max_min_ratio':    1.5,
    'sem':              4,
    'ptp':              1,
}
TOLERANCE = 10


class ArtifactDetector(Block):
    def __init__(self, criteria=None, tolerance=TOLERANCE):
        super().__init__()
        self.criteria  = criteria or ARTIFACT_CRITERIA
        self.tolerance = tolerance

    def find_maxima_ratio(self, data):
        up    = (data[1:] > data[:-1]).nonzero()[0] + 1
        down  = (data[:-1] > data[1:]).nonzero()[0]
        peaks = np.intersect1d(up, down)
        peaks = np.append(peaks, len(data))
        idx   = np.diff(peaks) >= self.tolerance
        num   = idx.sum()
        if num > 1:
            vals  = np.sort(data[peaks[idx.nonzero()[0]]])
            ratio = np.abs(vals[-1] / vals[-2])
        else:
            ratio = np.inf
        return num, ratio

    def max_min_ratio(self, data):
        return np.abs(data.max() / data.min())

    def std_err_mean(self, data):
        return data.std(0).mean() / np.sqrt(data.shape[0])

    def peak_to_peak(self, data):
        cut = int(data.shape[0] / 2)
        return np.ptp(data[cut:] - data[0]) / data.max()

    def artifact_score(self, data):
        mean    = data.mean(0)
        score   = 0
        reasons = []

        num_peaks, peak_ratio = self.find_maxima_ratio(mean)
        ratio   = self.max_min_ratio(mean)
        std_err = self.std_err_mean(data)
        ptp     = self.peak_to_peak(mean)

        if num_peaks > self.criteria['maxima']:
            score += 1; reasons.append('maxima')
        if peak_ratio < self.criteria['maxima_1_2_ratio']:
            score += 1; reasons.append('maxima_1_2_ratio')
        if ratio < self.criteria['max_min_ratio']:
            score += 1; reasons.append('max_min_ratio')
        if std_err > self.criteria['sem']:
            score += 1; reasons.append('sem')
        if ptp > self.criteria['ptp']:
            score += 1; reasons.append('ptp')

        return score, reasons, mean

    def forward(self, spikes, sort_idx, sign='pos'):
        if isinstance(spikes, torch.Tensor):
            spikes_np = spikes.cpu().numpy()
        else:
            spikes_np = spikes.copy()

        invert          = (sign == 'neg')
        class_ids       = np.unique(sort_idx)
        artifact_scores = {}
        artifact_ids    = []

        for class_id in class_ids:
            if class_id == 0:
                continue
            class_idx    = sort_idx == class_id
            class_spikes = spikes_np[class_idx]
            if invert:
                class_spikes = -class_spikes
            score, reasons, _ = self.artifact_score(class_spikes)
            artifact_scores[int(class_id)] = score
            if score:
                artifact_ids.append(int(class_id))

        return artifact_scores, artifact_ids
