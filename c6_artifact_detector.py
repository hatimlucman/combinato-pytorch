"""
C6 - ArtifactDetector
=======================
Scores each cluster for artifact-like waveform properties.

Exact port of artifacts.py:
    find_maxima_ratio(data, tolerance) → num peaks + peak ratio
    max_min_ratio(data)                → abs(max/min) of mean spike
    std_err_mean(data)                 → within-cluster variability
    peak_to_peak(data)                 → second-half waveform shape
    artifact_score(data)               → combines all four into a score
    find_artifacts(spikes, sorted_idx, class_ids, invert) → score each cluster

Input:
    spikes:   (N, 64) tensor — all spike waveforms
    sort_idx: (N,) numpy array — cluster assignments
    sign:     'pos' or 'neg'

Output:
    artifact_scores: dict {cluster_id: score}
    artifact_ids:    list of cluster ids with score > 0
"""

import torch
import torch.nn as nn
import numpy as np

# mirrors artifact_criteria from combinato/__init__.py
ARTIFACT_CRITERIA = {
    'maxima':           5,
    'maxima_1_2_ratio': 2,
    'max_min_ratio':    1.5,
    'sem':              4,
    'ptp':              1,
}
TOLERANCE = 10


class ArtifactDetector(nn.Module):
    """
    C6: Score clusters for artifact-like waveform properties.

    Exact port of artifacts.py. Five criteria, each failure
    adds 1 to score. Score > 0 = artifact cluster.

    Args:
        criteria (dict): artifact thresholds.
        tolerance (int): min sample gap between peaks. Default 10.
    """

    def __init__(self, criteria=None, tolerance=TOLERANCE):
        super().__init__()
        self.criteria  = criteria or ARTIFACT_CRITERIA
        self.tolerance = tolerance

    def find_maxima_ratio(self, data):
        """
        Mirrors find_maxima_ratio(data, tolerance) in artifacts.py exactly.
        data: mean spike (1D array).
        """
        up    = (data[1:] > data[:-1]).nonzero()[0] + 1
        down  = (data[:-1] > data[1:]).nonzero()[0]
        peaks = np.intersect1d(up, down)
        peaks = np.append(peaks, len(data))

        # exclude nearby peaks
        idx   = np.diff(peaks) >= self.tolerance
        num   = idx.sum()

        if num > 1:
            vals  = np.sort(data[peaks[idx.nonzero()[0]]])
            ratio = np.abs(vals[-1] / vals[-2])
        else:
            ratio = np.inf

        return num, ratio

    def max_min_ratio(self, data):
        """
        Mirrors max_min_ratio(data) in artifacts.py exactly.
        Original: return np.abs(data.max()/data.min())
        No zero guard — matches original behavior.
        """
        return np.abs(data.max() / data.min())

    def std_err_mean(self, data):
        """
        Mirrors std_err_mean(data) in artifacts.py exactly.
        data: all spikes in cluster (2D array).
        """
        return data.std(0).mean() / np.sqrt(data.shape[0])

    def peak_to_peak(self, data):
        """
        Mirrors peak_to_peak(data) in artifacts.py exactly.
        Original: cut = int(data.shape[0]/2)
                  return (data[cut:] - data[0]).ptp()/data.max()
        data: mean spike (1D array).
        No zero guard — matches original behavior.
        """
        cut = int(data.shape[0] / 2)
        return np.ptp(data[cut:] - data[0]) / data.max()

    def artifact_score(self, data):
        """
        Mirrors artifact_score(data) in artifacts.py exactly.
        data: all spikes in cluster (2D array).
        """
        mean    = data.mean(0)
        score   = 0
        reasons = []

        num_peaks, peak_ratio = self.find_maxima_ratio(mean)
        ratio   = self.max_min_ratio(mean)
        std_err = self.std_err_mean(data)
        ptp     = self.peak_to_peak(mean)

        if num_peaks > self.criteria['maxima']:
            score += 1
            reasons.append('maxima')

        if peak_ratio < self.criteria['maxima_1_2_ratio']:
            score += 1
            reasons.append('maxima_1_2_ratio')

        if ratio < self.criteria['max_min_ratio']:
            score += 1
            reasons.append('max_min_ratio')

        if std_err > self.criteria['sem']:
            score += 1
            reasons.append('sem')

        if ptp > self.criteria['ptp']:
            score += 1
            reasons.append('ptp')

        return score, reasons, mean

    def forward(self, spikes, sort_idx, sign='pos'):
        """
        Mirrors find_artifacts(spikes, sorted_idx, class_ids, invert) exactly.

        Args:
            spikes:   (N, 64) tensor or numpy array
            sort_idx: (N,) numpy array
            sign:     'pos' or 'neg'

        Returns:
            artifact_scores: dict {cluster_id: score}
            artifact_ids:    list of flagged cluster ids
        """
        if isinstance(spikes, torch.Tensor):
            spikes_np = spikes.cpu().numpy()
        else:
            spikes_np = spikes.copy()

        # invert negative spikes — mirrors find_artifacts(invert=True if sign=='neg')
        invert = True if sign == 'neg' else False

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


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("C6 ARTIFACT DETECTOR — VALIDATION")
    print("=" * 60)

    # use the test data from the original artifacts.py testit()
    data = np.array([[0, 1, 14, 5, 5, 5, 5, 20, 30, 0, 11, 0],
                     [-1, 3, 12, 3, 4, 7, 7, 20, 30, 0, 11, 1]], float)

    detector = ArtifactDetector()
    score, reasons, mean = detector.artifact_score(data)
    print(f"  Score   : {score}")
    print(f"  Reasons : {reasons}")
    print(f"  Mean    : {mean}")

    # compare with original
    from combinato.spike_sort.artifacts import artifact_score as orig_artifact_score
    orig_score, orig_reasons, orig_mean = orig_artifact_score(data)
    print(f"\n  Original score   : {orig_score}")
    print(f"  Original reasons : {orig_reasons}")

    match = (score == orig_score) and (reasons == orig_reasons)
    print(f"\n  Match : {'✓' if match else '✗'}")

    print("\n" + "=" * 60)
    print("C6 PASSED ✓" if match else "C6 FAILED ✗")
    print("=" * 60)
