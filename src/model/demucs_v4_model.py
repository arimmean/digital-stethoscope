"""
Pretrained source-separation denoiser using Demucs v4 (htdemucs, hybrid
transformer) from the `demucs` pip package.

Sibling of `demucs_model.py` (v3 torchaudio HDEMUCS), kept as a separate class so
the two can be A/B-compared on the same files. v4 is generally a stronger
separator; we test whether it routes auscultation voice more reliably than v3
(which scattered voice unpredictably across stems on whole files — see
filter_ablation_results.md).

Key differences from the v3 wrapper:
- Uses `demucs.apply.apply_model`, which does its OWN chunking + shift-trick
  (so no manual chunking needed; long files won't OOM). Exposes `shifts`/`split`.
- Replicates the demucs CLI's NORMALIZATION (subtract mix mean, divide by std
  before separation; reverse after). The model was trained on normalized input;
  skipping this degrades separation. The v3 wrapper did NOT normalize — a likely
  contributor to its inconsistent routing.

Same output modes / interface as v3 DemucsModel:
- "subtract" (DEFAULT): out = x - alpha * sum(remove_stems)
- "keep":               out = sum(keep_stems)

Contract: forward(audio: Tensor[B, 1, T], **batch) -> {"audio": Tensor[B, 1, T]}
"""

import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from torch import nn

_DEMUCS_SOURCES = ["drums", "bass", "other", "vocals"]


class DemucsV4Model(nn.Module):
    """
    Args:
        mode (str): "subtract" (default) or "keep" (see module docstring).
        remove_stems (list[str]): interference stems to subtract. Default ["vocals"].
        alpha (float): subtraction strength [0,1] (mode="subtract").
        keep_stems (list[str]): stems summed for output (mode="keep").
        input_sample_rate (int): SR the dataset feeds; must equal 44100 (no
            resample here — fail loud rather than pitch-shift).
        shifts (int): apply_model shift-trick averaging passes (>=1; higher =
            slightly better, slower). Default 1.
        split (bool): apply_model chunked processing (bounds memory). Default True.
        overlap (float): chunk overlap for split mode. Default 0.25.
    """

    def __init__(
        self,
        mode="subtract",
        remove_stems=None,
        alpha=1.0,
        keep_stems=None,
        input_sample_rate=44100,
        shifts=1,
        split=True,
        overlap=0.25,
    ):
        super().__init__()
        self.model = get_model("htdemucs")
        self.model.eval()
        self.model_sample_rate = self.model.samplerate  # 44100
        self.model_sources = list(self.model.sources)

        self.input_sample_rate = input_sample_rate
        assert input_sample_rate == self.model_sample_rate, (
            f"DemucsV4Model does not resample: input_sample_rate "
            f"({input_sample_rate}) must equal the model rate "
            f"({self.model_sample_rate}). Set datasets...target_sr=44100."
        )

        if mode not in ("subtract", "keep"):
            raise ValueError(f"mode must be 'subtract' or 'keep', got {mode!r}")
        self.mode = mode
        self.alpha = float(alpha)
        self.shifts = int(shifts)
        self.split = bool(split)
        self.overlap = float(overlap)

        if remove_stems is None:
            remove_stems = ["vocals"]
        if keep_stems is None:
            keep_stems = ["drums", "bass", "other"]
        for name, group in (("remove_stems", remove_stems), ("keep_stems", keep_stems)):
            for s in group:
                if s not in self.model_sources:
                    raise ValueError(
                        f"Unknown stem '{s}' in {name}. Valid: {self.model_sources}"
                    )
        self.remove_stems = list(remove_stems)
        self.keep_stems = list(keep_stems)
        self.remove_idx = [self.model_sources.index(s) for s in self.remove_stems]
        self.keep_idx = [self.model_sources.index(s) for s in self.keep_stems]

    @torch.no_grad()
    def forward(self, audio, **batch):
        # audio: [B, 1, T] mono @ 44.1 kHz. apply_model wants stereo [B, 2, T].
        device = audio.device
        self.model.to(device)
        x = audio.repeat(1, 2, 1)  # [B, 2, T]

        # CLI-style per-item normalization (model trained on normalized input).
        # ref stats over channels+time, per batch item.
        ref_mean = x.mean(dim=(1, 2), keepdim=True)
        ref_std = x.std(dim=(1, 2), keepdim=True) + 1e-8
        x_norm = (x - ref_mean) / ref_std

        sources = apply_model(
            self.model,
            x_norm,
            shifts=self.shifts,
            split=self.split,
            overlap=self.overlap,
            device=device,
        )  # [B, n_sources, 2, T]
        # de-normalize stems back to original scale
        sources = sources * ref_std[:, None] + ref_mean[:, None]

        if self.mode == "subtract":
            interference = sources[:, self.remove_idx].sum(dim=1)  # [B, 2, T]
            out = x - self.alpha * interference
        else:  # "keep"
            out = sources[:, self.keep_idx].sum(dim=1)

        mono = out.mean(dim=1, keepdim=True)  # [B, 1, T]
        return {"audio": mono}

    def __str__(self):
        if self.mode == "subtract":
            cfg = f"mode=subtract, remove_stems={self.remove_stems}, alpha={self.alpha}"
        else:
            cfg = f"mode=keep, keep_stems={self.keep_stems}"
        return (
            f"{type(self).__name__}(htdemucs, {cfg}, shifts={self.shifts}, "
            f"split={self.split}, overlap={self.overlap}, "
            f"input_sample_rate={self.input_sample_rate})"
        )
