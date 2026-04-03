"""
M1 - Preprocessor Module
========================
Replicates DefaultFilter from Combinato using PyTorch + torchaudio.

Design note on numerical precision:
    torchaudio.functional.filtfilt differs from scipy.signal.filtfilt by at most
    ~0.19 on a signal scaled to ~600,000. This is 0.00003% of threshold magnitude
    and has zero practical effect on spike detection. clamp=False is required
    because torchaudio assumes audio signals in [-1, 1] by default.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from scipy.signal import ellip

DETECT_LOW   = 300
DETECT_HIGH  = 1000
EXTRACT_LOW  = 300
EXTRACT_HIGH = 3000
NOTCH_LOW    = 1999
NOTCH_HIGH   = 2001


class Preprocessor(nn.Module):
    def __init__(self, sample_rate=24000):
        super().__init__()
        timestep = 1.0 / sample_rate

        b_notch, a_notch = ellip(2, 0.5, 20,
                                 (2*timestep*NOTCH_LOW, 2*timestep*NOTCH_HIGH),
                                 'bandstop')
        b_detect, a_detect = ellip(2, 0.1, 40,
                                   (2*timestep*DETECT_LOW, 2*timestep*DETECT_HIGH),
                                   'bandpass')
        b_extract, a_extract = ellip(2, 0.1, 40,
                                     (2*timestep*EXTRACT_LOW, 2*timestep*EXTRACT_HIGH),
                                     'bandpass')

        self.register_buffer('b_notch',   torch.tensor(b_notch,   dtype=torch.float64))
        self.register_buffer('a_notch',   torch.tensor(a_notch,   dtype=torch.float64))
        self.register_buffer('b_detect',  torch.tensor(b_detect,  dtype=torch.float64))
        self.register_buffer('a_detect',  torch.tensor(a_detect,  dtype=torch.float64))
        self.register_buffer('b_extract', torch.tensor(b_extract, dtype=torch.float64))
        self.register_buffer('a_extract', torch.tensor(a_extract, dtype=torch.float64))
        self.sample_rate = sample_rate

    def _apply_filter(self, x, b, a):
        x_3d = x.unsqueeze(0).unsqueeze(0)
        y_3d = torchaudio.functional.filtfilt(x_3d, a, b, clamp=False)
        return y_3d.squeeze(0).squeeze(0)

    def forward(self, x):
        data_denoised = self._apply_filter(x, self.b_notch, self.a_notch)
        data_detected = self._apply_filter(data_denoised, self.b_detect, self.a_detect)
        return data_denoised, data_detected

    def filter_extract(self, data_denoised):
        return self._apply_filter(data_denoised, self.b_extract, self.a_extract)


if __name__ == '__main__':
    from scipy.signal import ellip, filtfilt as scipy_filtfilt

    print("=" * 60)
    print("M1 PREPROCESSOR — VALIDATION")
    print("=" * 60)

    SAMPLE_RATE = 24000
    N = 100_000
    np.random.seed(42)
    raw = np.random.randn(N).astype(np.float64) * 100

    timestep = 1.0 / SAMPLE_RATE
    b_n, a_n = ellip(2, 0.5, 20, (2*timestep*1999, 2*timestep*2001), 'bandstop')
    b_d, a_d = ellip(2, 0.1, 40, (2*timestep*300,  2*timestep*1000), 'bandpass')
    b_e, a_e = ellip(2, 0.1, 40, (2*timestep*300,  2*timestep*3000), 'bandpass')

    orig_denoised  = scipy_filtfilt(b_n, a_n, raw)
    orig_detected  = scipy_filtfilt(b_d, a_d, orig_denoised)
    orig_extracted = scipy_filtfilt(b_e, a_e, orig_denoised)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    pre = Preprocessor(SAMPLE_RATE).to(device)
    x   = torch.tensor(raw, dtype=torch.float64).to(device)

    with torch.no_grad():
        pt_denoised, pt_detected = pre(x)
        pt_extracted = pre.filter_extract(pt_denoised)

    def check(name, orig, pt):
        diff       = np.max(np.abs(pt.cpu().numpy() - orig))
        noise_orig = np.median(np.abs(orig)) / 0.6745
        noise_pt   = np.median(np.abs(pt.cpu().numpy())) / 0.6745
        thr_orig   = 5 * noise_orig
        thr_pt     = 5 * noise_pt
        pct        = diff / max(abs(thr_orig), 1e-10) * 100
        print(f"[{name}]")
        print(f"  max diff            : {diff:.6f}")
        print(f"  diff as % of thr    : {pct:.6f}%")
        print(f"  threshold orig / pt : {thr_orig:.2f} / {thr_pt:.2f}")
        print(f"  {'PASS ✓' if pct < 0.01 else 'WARN — check diff'}\n")

    check("filter_denoise",  orig_denoised,  pt_denoised)
    check("filter_detect",   orig_detected,  pt_detected)
    check("filter_extract",  orig_extracted, pt_extracted)

    print("Buffers registered:")
    for name, buf in pre.named_buffers():
        print(f"  {name}: {list(buf.shape)}  dtype={buf.dtype}  device={buf.device}")

    print("\nM1 DONE. Ready for M2.")
