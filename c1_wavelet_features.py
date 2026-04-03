"""
C1 - WaveletFeatureExtractor
==============================
Transforms spike waveforms into wavelet feature vectors.

Replaces wave_features.py:
    WAVELET = pywt.Wavelet(options['Wavelet'])  # haar
    LEVEL = 4
    def wavelet_features(data):
        first_row = pywt.wavedec(data[0,:], WAVELET, level=LEVEL)
        aligned = np.hstack(first_row)
        output = np.empty((data.shape[0], aligned.shape[0]))
        for i, row in enumerate(data):
            features = pywt.wavedec(row, WAVELET, level=LEVEL)
            output[i] = np.hstack(features)
        return output

Input:
    spikes: (N, 64) tensor — spike waveforms

Output:
    features: (N, F) tensor — wavelet coefficients
              F depends on wavelet and level (64 for haar level=4 on 64-sample input)
"""

import torch
import torch.nn as nn
import numpy as np
import pywt

# mirrors options['Wavelet'] and LEVEL in wave_features.py
WAVELET = 'haar'
LEVEL   = 4


class WaveletFeatureExtractor(nn.Module):
    """
    C1: Compute wavelet features for each spike waveform.

    Uses pywt.wavedec exactly as the original, vectorized across
    all spikes at once using numpy broadcasting.

    Args:
        wavelet (str): wavelet name. Default 'haar' (Combinato default).
        level (int): decomposition level. Default 4.
    """

    def __init__(self, wavelet=WAVELET, level=LEVEL):
        super().__init__()
        self.wavelet_name = wavelet
        self.level        = level
        self.wavelet      = pywt.Wavelet(wavelet)

        # compute output feature size on dummy input
        dummy            = np.zeros(64, dtype=np.float32)
        coeffs           = pywt.wavedec(dummy, self.wavelet, level=self.level)
        self.feature_size = sum(c.shape[0] for c in coeffs)
        # store coefficient lengths for splitting
        self.coeff_lengths = [c.shape[0] for c in coeffs]

    def forward(self, spikes):
        """
        Args:
            spikes: (N, 64) tensor

        Returns:
            features: (N, F) tensor  where F depends on wavelet/level
        """
        spikes_np = spikes.cpu().numpy().astype(np.float32)
        N         = spikes_np.shape[0]
        output    = np.empty((N, self.feature_size), dtype=np.float32)

        for i in range(N):
            coeffs     = pywt.wavedec(spikes_np[i], self.wavelet,
                                      level=self.level)
            output[i]  = np.hstack(coeffs)

        return torch.tensor(output, dtype=torch.float32, device=spikes.device)


# ==============================================================================
# VALIDATION
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("C1 WAVELET FEATURE EXTRACTOR — VALIDATION")
    print("=" * 60)

    np.random.seed(42)
    N         = 500
    spikes_np = np.random.randn(N, 64).astype(np.float32)

    # --- Original ---
    print("\n[ORIGINAL] pywt wavelet_features (haar level=4)...")
    wavelet = pywt.Wavelet(WAVELET)
    first   = pywt.wavedec(spikes_np[0], wavelet, level=LEVEL)
    F       = sum(c.shape[0] for c in first)
    orig    = np.empty((N, F), dtype=np.float32)
    for i, row in enumerate(spikes_np):
        orig[i] = np.hstack(pywt.wavedec(row, wavelet, level=LEVEL))
    print(f"  Output shape : {orig.shape}")
    print(f"  Feature size : {F}")

    # --- PyTorch C1 ---
    print("\n[PYTORCH] C1 WaveletFeatureExtractor...")
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = WaveletFeatureExtractor().to(device)
    x         = torch.tensor(spikes_np).to(device)

    with torch.no_grad():
        pt_feat = extractor(x)

    pt_np = pt_feat.cpu().numpy()
    print(f"  Output shape : {pt_np.shape}")
    print(f"  Feature size : {extractor.feature_size}")

    # --- Validate ---
    print("\n[VALIDATION]")
    shape_match = pt_np.shape == orig.shape
    print(f"  Shape match  : {orig.shape} vs {pt_np.shape}  {'✓' if shape_match else '✗'}")
    print(f"  Feature size : {extractor.feature_size} (expected {F})  {'✓' if extractor.feature_size == F else '✗'}")

    if shape_match:
        max_diff = np.max(np.abs(pt_np - orig))
        print(f"  Max diff     : {max_diff:.2e}  {'✓' if max_diff < 1e-5 else '✗'}")

    print("\n" + "="*60)
    all_pass = shape_match and max_diff < 1e-5
    print("C1 PASSED ✓" if all_pass else "C1 FAILED ✗")
    print("="*60)
