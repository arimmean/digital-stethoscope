"""
Pretrained source-separation denoiser using torchaudio's bundled Hybrid Demucs
(HDEMUCS_HIGH_MUSDB_PLUS).

Hypothesis (project brief): a music source separator's vocal channel captures
speech / crying / talk interference; the rest is the breathing we want to keep.

EMPIRICAL FINDINGS (notebook 04, see filter_ablation_results.md):
- Demucs separates voice CONDITIONALLY — quality tracks the voice's own SNR.
  Clean/clear voice (high-pitched, loud crying) → vocals grabs 60-95%, sounds
  right. Noise-corrupted/unintelligible voice → collapses into `other`, no
  separation.
- It does NOT damage clean breathing/pathology segments (KEEP ≈ 99-102% of input,
  vocals ≈ empty there) — safe to apply broadly.
- Open risk: breathing co-occurring with a clean strong voice is under-tested —
  the breathing might ride along in the vocals stem and be over-removed.

Two output modes (mode arg):
- "subtract" (DEFAULT): out = x - alpha * sum(remove_stems). Subtracting the
  estimated interference rather than reconstructing from kept stems. With
  alpha < 1 this is gentle precisely when vocals is confident (~95%), so any
  breathing that rode along is partially preserved — de-risks the overlap case.
  alpha is a single global, defensible hyperparameter (not a per-file gate).
- "keep": out = sum(keep_stems). The original hard-reconstruction (drop vocals
  entirely). Kept for comparison / ablation.

Contract (same as the other denoisers):
    forward(audio: Tensor[B, 1, T], **batch) -> {"audio": Tensor[B, 1, T]}
"""

import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torch import nn

# Stem order produced by HDEMUCS, fixed by the model.
_DEMUCS_SOURCES = ["drums", "bass", "other", "vocals"]


class DemucsModel(nn.Module):
    """
    Args:
        mode (str): "subtract" (default) -> out = x - alpha*sum(remove_stems);
            "keep" -> out = sum(keep_stems).
        remove_stems (list[str]): stems treated as interference to subtract
            (mode="subtract"). Default ["vocals"].
        alpha (float): subtraction strength in [0, 1] (mode="subtract"). 1.0 =
            full removal (≈ the old hard-drop); < 1.0 keeps some interference
            back, preserving breathing that rode along in a confident vocals
            stem. Global, tunable by ear.
        keep_stems (list[str]): stems summed for the output (mode="keep").
            Default the three non-vocal stems.
        input_sample_rate (int): SR the dataset feeds (datasets...target_sr).
            HDEMUCS runs at 44100; this model does NOT resample, so it asserts
            input_sample_rate == 44100 to avoid silently pitch-shifting.

    NOTE on resampling: deliberately not done here (model SR contract = 44.1 kHz,
    fed natively). A mismatch fails the assert loud rather than pitch-shifting —
    the bug class we hit with save_sample_rate. If a different working rate is
    ever needed, add an explicit librosa resample and update the assert.

    NOTE on memory: whole-signal forward (no chunking). Fine for short segments;
    long files may OOM. TODO: overlapping-chunk application if that happens.
    """

    def __init__(
        self,
        mode="subtract",
        remove_stems=None,
        alpha=1.0,
        keep_stems=None,
        input_sample_rate=44100,
    ):
        super().__init__()
        self.model = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
        self.model.eval()
        self.model_sample_rate = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate  # 44100
        self.input_sample_rate = input_sample_rate
        assert input_sample_rate == self.model_sample_rate, (
            f"DemucsModel does not resample: input_sample_rate "
            f"({input_sample_rate}) must equal the model rate "
            f"({self.model_sample_rate}). Set datasets...target_sr=44100."
        )

        if mode not in ("subtract", "keep"):
            raise ValueError(f"mode must be 'subtract' or 'keep', got {mode!r}")
        self.mode = mode
        self.alpha = float(alpha)

        if remove_stems is None:
            remove_stems = ["vocals"]
        if keep_stems is None:
            keep_stems = ["drums", "bass", "other"]
        for name, group in (("remove_stems", remove_stems), ("keep_stems", keep_stems)):
            for s in group:
                if s not in _DEMUCS_SOURCES:
                    raise ValueError(
                        f"Unknown stem '{s}' in {name}. Valid: {_DEMUCS_SOURCES}"
                    )
        self.remove_stems = list(remove_stems)
        self.keep_stems = list(keep_stems)
        self.remove_idx = [_DEMUCS_SOURCES.index(s) for s in self.remove_stems]
        self.keep_idx = [_DEMUCS_SOURCES.index(s) for s in self.keep_stems]

    @torch.no_grad()
    def forward(self, audio, **batch):
        # audio: [B, 1, T] mono at 44.1 kHz. HDEMUCS wants stereo [B, 2, T].
        device = audio.device
        self.model.to(device)
        x = audio.repeat(1, 2, 1)
        sources = self.model(x)  # [B, n_sources, 2, T]
        # print(sources)
        if self.mode == "subtract":
            interference = sources[:, self.remove_idx].sum(dim=1)  # [B, 2, T]
            out = x - self.alpha * interference
        else:  # "keep"
            out = sources[:, self.keep_idx].sum(dim=1)  # [B, 2, T]

        mono = out.mean(dim=1, keepdim=True)  # [B, 1, T]
        return {"audio": mono}

    def __str__(self):
        if self.mode == "subtract":
            cfg = f"mode=subtract, remove_stems={self.remove_stems}, alpha={self.alpha}"
        else:
            cfg = f"mode=keep, keep_stems={self.keep_stems}"
        return (
            f"{type(self).__name__}({cfg}, "
            f"input_sample_rate={self.input_sample_rate}, "
            f"model_sample_rate={self.model_sample_rate})"
        )
