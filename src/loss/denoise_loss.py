"""
Denoising loss for waveform regression: L1 (time domain) + multi-resolution STFT.

The MR-STFT term preserves spectral detail (wheeze / crepitus texture) that plain
L1 tends to over-smooth — important since that texture is the diagnostic content.
Standard recipe (Yamamoto et al. 2020 / parallel-WaveGAN): sum of spectral
convergence + log-magnitude L1 over several FFT resolutions, added to a waveform L1.

Returns a dict with the aggregate under "loss" (the Trainer sums via this key)
plus components for logging.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _stft_mag(x, n_fft, hop, win_length, window):
    # x: [B, T] -> magnitude [B, F, frames]
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win_length,
        window=window, return_complex=True, center=True,
    )
    return spec.abs().clamp(min=1e-7)


class _SingleResSTFTLoss(nn.Module):
    def __init__(self, n_fft, hop, win_length):
        super().__init__()
        self.n_fft, self.hop, self.win_length = n_fft, hop, win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, pred, target):
        p = _stft_mag(pred, self.n_fft, self.hop, self.win_length, self.window)
        t = _stft_mag(target, self.n_fft, self.hop, self.win_length, self.window)
        # spectral convergence
        sc = torch.norm(t - p, p="fro") / (torch.norm(t, p="fro") + 1e-7)
        # log-magnitude L1
        mag = F.l1_loss(torch.log(p), torch.log(t))
        return sc + mag


class DenoiseLoss(nn.Module):
    """
    L1 + multi-resolution STFT loss for noisy→clean waveform regression.

    Args:
        l1_weight (float): weight on time-domain L1.
        stft_weight (float): weight on the (summed) multi-res STFT loss.
        fft_sizes / hop_sizes / win_lengths (list[int]): the STFT resolutions.
            Defaults are the common 3-resolution set, scaled fine enough for
            our 22.05 kHz / short segments.
        sisdr_weight (float): weight on the SI-SDR loss term (loss = -SI-SDR, so
            minimizing it MAXIMIZES SI-SDR). Set > 0 to optimize the metric we
            actually evaluate on — the L1/STFT-only loss does NOT (it ignores
            phase+scale and is won by smooth/quiet output, which tanks SI-SDR;
            wave_unet_v1 scored BELOW the noisy floor because of this).
    """

    def __init__(
        self,
        l1_weight=1.0,
        stft_weight=1.0,
        sisdr_weight=0.0,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_lengths=(512, 1024, 2048),
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.sisdr_weight = sisdr_weight
        self.stft_losses = nn.ModuleList(
            _SingleResSTFTLoss(n, h, w)
            for n, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        )

    @staticmethod
    def _si_sdr(pred, tgt, eps=1e-8):
        """SI-SDR (dB) per item, mean over batch. Scale-invariant: project pred
        onto tgt, ratio of target-energy to residual-energy."""
        # zero-mean
        pred = pred - pred.mean(dim=-1, keepdim=True)
        tgt = tgt - tgt.mean(dim=-1, keepdim=True)
        alpha = (pred * tgt).sum(-1, keepdim=True) / (tgt.pow(2).sum(-1, keepdim=True) + eps)
        proj = alpha * tgt
        noise = pred - proj
        ratio = proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps)
        return (10 * torch.log10(ratio + eps)).mean()

    def forward(self, audio, target, **batch):
        # audio = model output (denoised), target = clean. Both [B, 1, T].
        pred = audio.squeeze(1)
        tgt = target.squeeze(1)
        m = min(pred.shape[-1], tgt.shape[-1])
        pred, tgt = pred[..., :m], tgt[..., :m]

        l1 = F.l1_loss(pred, tgt)
        stft = sum(f(pred, tgt) for f in self.stft_losses) / len(self.stft_losses)
        sisdr = self._si_sdr(pred, tgt)  # dB (higher = better)

        loss = self.l1_weight * l1 + self.stft_weight * stft
        if self.sisdr_weight:
            loss = loss - self.sisdr_weight * sisdr  # minimize loss => maximize SI-SDR
        return {
            "loss": loss,
            "l1_loss": l1.detach(),
            "stft_loss": stft.detach(),
            "sisdr": sisdr.detach(),
        }
