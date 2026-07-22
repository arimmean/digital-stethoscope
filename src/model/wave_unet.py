"""
Wave-U-Net: a 1-D convolutional U-Net for waveform-domain denoising
(noisy → clean regression). Deterministic — chosen over generative diffusion
specifically so it CANNOT hallucinate diagnostic detail (see
.agent_memory/phase3_learned_method.md).

Compact, from-scratch implementation (à la Macartney & Weyde speech-enhancement
Wave-U-Net), sized SMALL by default because the clean-target pool is thin
(~0.33h) — start under-parameterized to resist overfitting, grow if it underfits.

Contract (same as every denoiser here):
    forward(audio: Tensor[B, 1, T], **batch) -> {"audio": Tensor[B, 1, T]}

Handles arbitrary T at inference by padding to a multiple of 2**n_levels and
trimming back. Training typically feeds fixed-length crops (NoisyMixDataset
crop_len), but variable lengths work too.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _DownBlock(nn.Module):
    """Conv (+ act) then downsample by 2 (strided conv). Returns (skip, down)."""

    def __init__(self, in_ch, out_ch, kernel=15):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.act = nn.LeakyReLU(0.1)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.act(self.conv(x))   # [B, out_ch, T]
        down = self.act(self.down(skip))  # [B, out_ch, T/2]
        return skip, down


class _UpBlock(nn.Module):
    """Upsample by 2, concat skip, conv (+ act)."""

    def __init__(self, in_ch, skip_ch, out_ch, kernel=15):
        super().__init__()
        pad = kernel // 2
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel, padding=pad)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, skip):
        x = self.up(x)
        # align length to skip (ConvTranspose can be off by 1)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff)) if diff > 0 else x[..., : skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.act(self.conv(x))


class WaveUNet(nn.Module):
    """
    Args:
        n_levels (int): encoder/decoder depth. Input T is padded to a multiple
            of 2**n_levels. Default 5.
        base_channels (int): channels at the first level; doubles each level
            down. Default 24 (small, for thin data).
        kernel (int): conv kernel size. Default 15 (large receptive field per
            layer, common for Wave-U-Net).
        residual (bool): if True, model predicts the clean signal as
            noisy + delta (learn the residual). Often easier/stabler for
            denoising. Default True.
    """

    def __init__(self, n_levels=5, base_channels=24, kernel=15, residual=True):
        super().__init__()
        self.n_levels = n_levels
        self.residual = residual

        downs = []
        chs = [1]
        for i in range(n_levels):
            out_ch = base_channels * (2 ** i)
            downs.append(_DownBlock(chs[-1], out_ch, kernel))
            chs.append(out_ch)
        self.downs = nn.ModuleList(downs)

        bott = base_channels * (2 ** n_levels)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(chs[-1], bott, kernel, padding=kernel // 2),
            nn.LeakyReLU(0.1),
        )

        ups = []
        in_ch = bott
        for i in reversed(range(n_levels)):
            skip_ch = base_channels * (2 ** i)
            out_ch = skip_ch
            ups.append(_UpBlock(in_ch, skip_ch, out_ch, kernel))
            in_ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, audio, **batch):
        x = audio  # [B, 1, T]
        T = x.shape[-1]
        mult = 2 ** self.n_levels
        pad = (mult - T % mult) % mult
        if pad:
            x = F.pad(x, (0, pad))

        h = x
        skips = []
        for d in self.downs:
            skip, h = d(h)
            skips.append(skip)
        h = self.bottleneck(h)
        for u, skip in zip(self.ups, reversed(skips)):
            h = u(h, skip)
        delta = self.out_conv(h)

        out = x + delta if self.residual else delta
        out = out[..., :T]  # trim padding back to original length
        return {"audio": out}

    def __str__(self):
        n = sum(p.numel() for p in self.parameters())
        return (
            f"{type(self).__name__}(n_levels={self.n_levels}, "
            f"residual={self.residual}, params={n/1e6:.2f}M)"
        )
