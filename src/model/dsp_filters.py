"""
Conventional DSP denoising baselines for auscultation audio.

Each filter is a stateless nn.Module with the standard denoiser contract:
    forward(audio: Tensor[B, 1, T], **batch) -> {"audio": Tensor[B, 1, T]}

Filtering is done with scipy on CPU using zero-phase `filtfilt` / `sosfiltfilt`
(no phase distortion, which matters for diagnostic morphology of breath sounds).
The tensor -> numpy -> tensor round-trip is cheap at the batch sizes used for
inference; these are baselines, not the hot path.
"""

import numpy as np
import torch
from scipy.signal import butter, iirnotch, istft, sosfiltfilt, stft, tf2sos
from torch import nn


def _to_numpy(audio):
    """[B, 1, T] (any device) -> (np.ndarray[B, T] float64, original device)."""
    device = audio.device
    arr = audio.squeeze(1).detach().cpu().numpy().astype(np.float64)
    return arr, device


def _to_tensor(arr, device):
    """np.ndarray[B, T] -> Tensor[B, 1, T] float32 on device."""
    return torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(1).to(device)


def _apply_sos(arr, sos):
    """Zero-phase apply a second-order-sections filter row-wise over [B, T]."""
    # sosfiltfilt operates along the last axis; arr is [B, T].
    return sosfiltfilt(sos, arr, axis=-1)


class NotchFilter(nn.Module):
    """
    IIR notch at `freq` Hz and its harmonics (power-line interference).

    Args:
        sample_rate (int): audio SR. Must match the rate the audio is at.
        freq (float): fundamental line frequency (50 EU/RU, 60 US).
        n_harmonics (int): how many harmonics to notch (1 = fundamental only).
            Harmonics above Nyquist are skipped automatically.
        quality (float): Q factor of each notch (higher = narrower).
    """

    def __init__(self, sample_rate, freq=50.0, n_harmonics=4, quality=30.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq = freq
        self.n_harmonics = n_harmonics
        self.quality = quality

        nyquist = sample_rate / 2.0
        sos_list = []
        for k in range(1, n_harmonics + 1):
            f0 = freq * k
            if f0 >= nyquist:
                break
            b, a = iirnotch(w0=f0, Q=quality, fs=sample_rate)
            sos_list.append(tf2sos(b, a))
        # stack all harmonic notches into one SOS cascade (may be empty if the
        # fundamental is already >= Nyquist, in which case forward is a no-op)
        self._sos = np.concatenate(sos_list, axis=0) if sos_list else None

    def forward(self, audio, **batch):
        if self._sos is None:
            return {"audio": audio}
        arr, device = _to_numpy(audio)
        filtered = _apply_sos(arr, self._sos)
        return {"audio": _to_tensor(filtered, device)}

    def __str__(self):
        return (
            f"{type(self).__name__}(freq={self.freq}, "
            f"n_harmonics={self.n_harmonics}, Q={self.quality}, "
            f"sample_rate={self.sample_rate})"
        )


class BandpassFilter(nn.Module):
    """
    Zero-phase Butterworth bandpass restricting signal to the respiratory band.

    Args:
        sample_rate (int): audio SR.
        low_hz (float): lower cutoff (default 100 Hz).
        high_hz (float): upper cutoff (default 2000 Hz). Clamped below Nyquist.
        order (int): Butterworth order (per direction; filtfilt doubles it).
    """

    def __init__(self, sample_rate, low_hz=100.0, high_hz=2000.0, order=4):
        super().__init__()
        self.sample_rate = sample_rate
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

        nyquist = sample_rate / 2.0
        high = min(high_hz, nyquist * 0.999)  # keep strictly below Nyquist
        assert low_hz < high, (
            f"Bandpass low_hz ({low_hz}) must be < effective high "
            f"({high}); sample_rate={sample_rate}"
        )
        self._sos = butter(
            order, [low_hz, high], btype="bandpass", fs=sample_rate, output="sos"
        )

    def forward(self, audio, **batch):
        arr, device = _to_numpy(audio)
        filtered = _apply_sos(arr, self._sos)
        return {"audio": _to_tensor(filtered, device)}

    def __str__(self):
        return (
            f"{type(self).__name__}(low_hz={self.low_hz}, high_hz={self.high_hz}, "
            f"order={self.order}, sample_rate={self.sample_rate})"
        )


class WienerFilter(nn.Module):
    """
    Spectral Wiener filter with the noise spectrum estimated from the signal
    itself (no external noise reference) — the speech-enhancement Wiener of
    Lim & Oppenheim (1979).

    Method: STFT -> estimate a per-frequency-bin noise PSD as a low percentile
    of |S|^2 across time frames (robust when the desired signal is mostly
    present, as breath sounds are) -> Wiener gain H = SNR / (1 + SNR) per bin,
    floored so bins are attenuated rather than nulled (reduces musical noise)
    -> apply gain -> iSTFT.

    Each waveform's noise floor is estimated independently, so processing is
    per-item over the batch.

    Args:
        sample_rate (int): audio SR (passed through STFT for correct framing).
        n_fft (int): STFT window length.
        hop_length (int): STFT hop. Defaults to n_fft // 4.
        noise_percentile (float): percentile (0-100) of |S|^2 over time used as
            the per-bin noise PSD estimate. Lower = assume less noise.
        gain_floor (float): minimum Wiener gain in [0, 1]; bins never attenuate
            below this (limits artifacts). 0.0 allows full suppression.
    """

    def __init__(
        self,
        sample_rate,
        n_fft=1024,
        hop_length=None,
        noise_percentile=10.0,
        gain_floor=0.05,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.noise_percentile = noise_percentile
        self.gain_floor = gain_floor

    def _wiener_one(self, x):
        """Apply spectral Wiener to a single 1-D signal x (np.ndarray)."""
        noverlap = self.n_fft - self.hop_length
        f, t, S = stft(
            x,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=noverlap,
            boundary="zeros",
        )
        power = np.abs(S) ** 2
        # per-bin noise PSD: low percentile of power across time frames
        noise_psd = np.percentile(power, self.noise_percentile, axis=-1, keepdims=True)
        eps = np.finfo(power.dtype).eps
        snr = power / (noise_psd + eps)
        gain = snr / (1.0 + snr)
        gain = np.maximum(gain, self.gain_floor)
        _, x_rec = istft(
            gain * S,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=noverlap,
            boundary=True,
        )
        # istft length can differ from input by a few samples; align to input
        if x_rec.shape[-1] >= x.shape[-1]:
            return x_rec[: x.shape[-1]]
        out = np.zeros_like(x)
        out[: x_rec.shape[-1]] = x_rec
        return out

    def forward(self, audio, **batch):
        arr, device = _to_numpy(audio)  # [B, T]
        out = np.stack([self._wiener_one(row) for row in arr], axis=0)
        return {"audio": _to_tensor(out, device)}

    def __str__(self):
        return (
            f"{type(self).__name__}(n_fft={self.n_fft}, "
            f"hop_length={self.hop_length}, "
            f"noise_percentile={self.noise_percentile}, "
            f"gain_floor={self.gain_floor}, sample_rate={self.sample_rate})"
        )


class DSPPipeline(nn.Module):
    """
    Chains DSP filters in order. Default Phase-2 baseline: notch -> bandpass.

    Pass already-instantiated filter modules (Hydra builds them from config).
    Each must follow the {"audio": ...} contract.
    """

    def __init__(self, filters):
        super().__init__()
        self.filters = nn.ModuleList(filters)

    def forward(self, audio, **batch):
        out = audio
        for f in self.filters:
            out = f(audio=out)["audio"]
        return {"audio": out}

    def __str__(self):
        inner = " -> ".join(str(f) for f in self.filters)
        return f"{type(self).__name__}({inner})"
