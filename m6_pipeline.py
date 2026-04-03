"""
M6 - Full Extraction Pipeline
===============================
Wires M1 through M5 into a single module.
Handles timestamp conversion and negative spike polarity restoration.

Replaces the full extract_spikes() function in extract_spikes.py.

Input:
    signal:    (N,) tensor — raw scaled signal
    atimes:    (N,) tensor — timestamp in ms for each sample
    sign:      run for 'pos', 'neg', or 'both'

Output:
    pos_spikes: (P, 64) tensor
    pos_times:  (P,)    tensor — ms timestamps
    neg_spikes: (Q, 64) tensor
    neg_times:  (Q,)    tensor — ms timestamps
    threshold:  scalar tensor
"""

import torch
import torch.nn as nn
import numpy as np

from m1_preprocessor import Preprocessor
from m2_threshold_detector import ThresholdDetector
from m3_waveform_extractor import WaveformExtractor
from m4_cubic_upsampler import CubicUpsampler
from m5_peak_aligner import PeakAligner


class CombinatoExtractor(nn.Module):
    """
    M6: Full spike extraction pipeline — M1 through M5 wired together.

    Args:
        sample_rate (int): Hz. Default 24000.
        threshold_factor (float): noise multiplier. Default 5.
        max_spike_duration (float): seconds. Default 0.0015.
        upsampling_factor (int): Default 3.
        indices_per_spike (int): final waveform length. Default 64.
        index_maximum (int): peak position in final waveform. Default 19.
        border_pad (int): extraction padding. Default 5.
    """

    def __init__(self,
                 sample_rate=24000,
                 threshold_factor=5,
                 max_spike_duration=0.0015,
                 upsampling_factor=3,
                 indices_per_spike=64,
                 index_maximum=19,
                 border_pad=5):
        super().__init__()

        self.pre       = Preprocessor(sample_rate=sample_rate)
        self.detector  = ThresholdDetector(sample_rate=sample_rate,
                                           threshold_factor=threshold_factor,
                                           max_spike_duration=max_spike_duration)
        self.extractor = WaveformExtractor(self.pre,
                                           indices_per_spike=indices_per_spike,
                                           index_maximum=index_maximum,
                                           border_pad=border_pad)
        self.upsampler = CubicUpsampler(factor=upsampling_factor)
        self.aligner   = PeakAligner(factor=upsampling_factor,
                                     indices_per_spike=indices_per_spike,
                                     index_maximum=index_maximum,
                                     border_pad=border_pad)

    def _process_one(self, data_denoised, peak_indices, atimes, sign):
        """
        Run M3 → M4 → M5 for one polarity and return waveforms + timestamps.

        Args:
            data_denoised: (N,) — from M1
            peak_indices:  (K,) — from M2
            atimes:        (N,) — ms timestamps
            sign:          'pos' or 'neg'

        Returns:
            spikes: (M, 64) — in original polarity (neg flipped back down)
            times:  (M,)    — ms timestamps
        """
        if len(peak_indices) == 0:
            device = data_denoised.device
            return (torch.zeros((0, self.aligner.indices_per_spike),
                                dtype=data_denoised.dtype, device=device),
                    torch.zeros(0, dtype=atimes.dtype, device=device))

        # M3: extract waveforms (neg spikes flipped up for alignment)
        spikes, valid, _ = self.extractor(data_denoised, peak_indices, sign)

        # timestamps for valid spikes only
        valid_indices = peak_indices[valid]
        times = atimes[valid_indices]

        if len(spikes) == 0:
            device = data_denoised.device
            return (torch.zeros((0, self.aligner.indices_per_spike),
                                dtype=data_denoised.dtype, device=device),
                    torch.zeros(0, dtype=atimes.dtype, device=device))

        # M4: upsample
        spikes_up = self.upsampler(spikes)

        # M5: align, clean, downsample
        spikes_final, removed = self.aligner(spikes_up)
        times_final = times[~removed]

        # restore original polarity for negative spikes
        # M3 flipped them up for alignment — flip back down for storage
        if sign == 'neg':
            spikes_final = spikes_final * -1

        return spikes_final, times_final

    def forward(self, signal, atimes):
        """
        Full extraction pipeline.

        Args:
            signal: (N,) tensor — raw scaled signal (float64)
            atimes: (N,) tensor — ms timestamp per sample (float64)

        Returns dict with keys:
            pos_spikes: (P, 64)
            pos_times:  (P,)
            neg_spikes: (Q, 64)
            neg_times:  (Q,)
            threshold:  scalar
        """
        # M1: filter
        data_denoised, data_detected = self.pre(signal)

        # M2: detect
        pos_indices, neg_indices, threshold = self.detector(data_detected)

        # M3 → M5: extract, upsample, align for each polarity
        pos_spikes, pos_times = self._process_one(
            data_denoised, pos_indices, atimes, 'pos')
        neg_spikes, neg_times = self._process_one(
            data_denoised, neg_indices, atimes, 'neg')

        return {
            'pos_spikes': pos_spikes,
            'pos_times':  pos_times,
            'neg_spikes': neg_spikes,
            'neg_times':  neg_times,
            'threshold':  threshold,
        }


def save_to_h5(result, output_path):
    """
    Save pipeline output to HDF5 matching Combinato's format exactly.

    Creates:
        /pos/spikes  (P, 64)
        /pos/times   (P,)
        /neg/spikes  (Q, 64)
        /neg/times   (Q,)
        /thr         (1, 3)  [0, threshold_scaled, noise_level]
    """
    import tables

    pos_spikes = result['pos_spikes'].cpu().numpy().astype(np.float32)
    pos_times  = result['pos_times'].cpu().numpy()
    neg_spikes = result['neg_spikes'].cpu().numpy().astype(np.float32)
    neg_times  = result['neg_times'].cpu().numpy()
    threshold  = result['threshold'].item()

    f = tables.open_file(output_path, 'w')
    f.create_group('/', 'pos', 'positive spikes')
    f.create_group('/', 'neg', 'negative spikes')

    for sign in ('pos', 'neg'):
        f.create_earray('/' + sign, 'spikes', tables.Float32Atom(), (0, 64))
        f.create_earray('/' + sign, 'times',  tables.FloatAtom(),   (0,))

    f.create_earray('/', 'thr', tables.FloatAtom(), (0, 3))

    r = f.root
    if len(pos_spikes): r.pos.spikes.append(pos_spikes)
    if len(pos_times):  r.pos.times.append(pos_times)
    if len(neg_spikes): r.neg.spikes.append(neg_spikes)
    if len(neg_times):  r.neg.times.append(neg_times)

    # store threshold in same format as Combinato
    r.thr.append(np.array([[0, threshold * 100, threshold / 5 * 0.6745]]))
    f.flush()
    f.close()
    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    print("M6 defined. Run test_full_pipeline.py to validate.")
