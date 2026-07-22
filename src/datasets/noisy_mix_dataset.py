"""
On-the-fly synthetic denoising-pair dataset for supervised training (Wave-U-Net).

Each item: a clean breathing/pathology SEGMENT (the target) + the same segment
mixed with a sampled noise donor at a sampled SNR (the noisy input). Fresh
donor+SNR every access → strong augmentation from a thin clean pool (~0.33h).

Design (see .agent_memory/phase3_learned_method.md):
- clean targets and noise donors are pre-split by uid (leak-free) via
  src.datasets.markup; pass the split's clean df + donor df.
- working SR = 22.05 kHz (resample with librosa — quality, [[library-choices]]).
- SNR ~ uniform[snr_min, snr_max] dB (default [-5, 15]); explicit SNR, not the
  low/mod/high labels.
- returns {"audio": noisy[1,T], "target": clean[1,T], "audio_path": str, ...},
  matching the collate (which pads `target` too).

For TRAIN use shuffle/random mixing. For frozen val/test use a separate
precomputed generator (build_frozen_pairs) so metrics are comparable — do NOT
use this dataset's random mixing for reported eval.
"""

import logging
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def crop_or_tile(x, n):
    """Make noise length match clean length n: truncate, or tile if too short."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def mix_at_snr(clean, noise, snr_db):
    """Mix clean + noise at a target SNR (dB). Peak-normalize to avoid clipping.

    Returns (mixed, scaled_noise) both length len(clean). If clean or noise is
    silent, returns clean unchanged. (Ported from nb02 cell 23.)
    """
    noise = crop_or_tile(noise, len(clean))
    c_rms, n_rms = rms(clean), rms(noise)
    if c_rms == 0 or n_rms == 0:
        return clean.astype(np.float32), noise.astype(np.float32)
    target_n_rms = c_rms / (10 ** (snr_db / 20))
    scaled = noise * (target_n_rms / n_rms)
    mixed = clean + scaled
    peak = max(np.max(np.abs(clean)), np.max(np.abs(scaled)), np.max(np.abs(mixed)), 1e-8)
    return (mixed / peak * 0.95).astype(np.float32), (scaled / peak * 0.95).astype(np.float32)


def load_segment(row, target_sr):
    """Read one annotated segment (start/length crop) as mono @ target_sr (librosa)."""
    info = sf.info(row["wav_path"])
    sr = info.samplerate
    start_f = int(round(float(row["start"]) * sr))
    n_f = max(1, int(round(float(row["length"]) * sr)))
    audio, sr = sf.read(
        row["wav_path"], start=start_f, frames=n_f, dtype="float32", always_2d=False
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


class NoisyMixDataset(Dataset):
    """
    On-the-fly (noisy, clean) pairs for training.

    Args:
        clean_df (pd.DataFrame): clean-target segments for THIS split
            (from markup.clean_target_segments + split_by_file).
        donor_df (pd.DataFrame): noise-donor segments for THIS split
            (from markup.noise_donor_segments + split_by_file). Must be the
            SAME split as clean_df (train-clean ↔ train-noise) to stay leak-free.
        target_sr (int): working sample rate (22050).
        snr_range (tuple): (min_db, max_db) uniform SNR sampling.
        seed (int): base RNG seed (combined with index for per-item determinism).
        limit (int | None): cap number of clean targets (debugging).
    """

    def __init__(
        self,
        clean_df,
        donor_df,
        target_sr=22050,
        snr_range=(-5.0, 15.0),
        seed=42,
        limit=None,
        crop_len=None,
    ):
        self.clean = clean_df.reset_index(drop=True)
        self.donor = donor_df.reset_index(drop=True)
        if limit is not None:
            self.clean = self.clean.iloc[:limit].reset_index(drop=True)
        assert len(self.clean) > 0, "no clean target segments"
        assert len(self.donor) > 0, "no noise donor segments"

        self.target_sr = target_sr
        self.snr_range = snr_range
        self.seed = seed
        # fixed-length training crops (samples). None = variable (whole segments).
        # Required for batching > 1; e.g. 1.0s @22.05k -> 22050.
        self.crop_len = crop_len

    def _fit_len(self, x, rng):
        """Random-crop or right-pad x to self.crop_len (no-op if crop_len None)."""
        if self.crop_len is None:
            return x
        n = self.crop_len
        if len(x) > n:
            start = rng.randrange(0, len(x) - n + 1)
            return x[start : start + n]
        if len(x) < n:
            return np.pad(x, (0, n - len(x)))
        return x

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, ind):
        # per-item RNG so a given (seed, ind) is reproducible across epochs/workers
        rng = random.Random(self.seed * 1_000_003 + ind)

        clean_row = self.clean.iloc[ind]
        clean = load_segment(clean_row, self.target_sr)
        clean = self._fit_len(clean, rng)  # fixed crop for batching (if set)

        # draw a donor from a DIFFERENT file than the clean target (don't let the
        # "noise" carry the same recording's acoustic signature). Retry a few
        # times; fall back to any donor if the split is tiny.
        clean_uid = clean_row["uid"]
        donor_row = self.donor.iloc[rng.randrange(len(self.donor))]
        for _ in range(8):
            if donor_row["uid"] != clean_uid:
                break
            donor_row = self.donor.iloc[rng.randrange(len(self.donor))]
        noise = load_segment(donor_row, self.target_sr)

        snr = rng.uniform(*self.snr_range)
        noisy, _ = mix_at_snr(clean, noise, snr)

        return {
            "audio": torch.from_numpy(noisy).unsqueeze(0),       # [1, T]
            "target": torch.from_numpy(clean).unsqueeze(0),      # [1, T]
            "audio_path": clean_row["segment_id"],
            "snr_db": snr,
            "donor_category": donor_row.get("donor_category"),
        }


def build_synth_dataset(
    split,
    json_path,
    wav_dir,
    target_sr=22050,
    snr_range=(-5.0, 15.0),
    crop_len=None,
    ratios=(0.7, 0.15, 0.15),
    split_seed=42,
    mix_seed=42,
    donor_categories=None,
    limit=None,
):
    """Hydra-friendly factory: build the on-the-fly NoisyMixDataset for one
    split ('train'/'val'/'test'). Does the uid-level clean/donor split internally
    so the config only needs paths + params, not DataFrames.
    """
    from src.datasets.markup import build_clean_donor_splits

    s = build_clean_donor_splits(
        json_path, wav_dir, ratios=tuple(ratios), seed=split_seed,
        donor_categories=donor_categories,
    )
    return NoisyMixDataset(
        s["clean"][split], s["donor"][split],
        target_sr=target_sr, snr_range=tuple(snr_range),
        seed=mix_seed, limit=limit, crop_len=crop_len,
    )


def build_frozen_pairs(clean_df, donor_df, out_dir, target_sr=22050,
                       snr_range=(-5.0, 15.0), seed=1234):
    """Materialize ONE deterministic (noisy, clean) pair per clean segment to disk.

    For frozen val/test eval — fixed donor + SNR per item so metrics (incl.
    full-reference PESQ/STOI/SI-SDR, which need the clean ref) are comparable
    across model checkpoints. Writes:
        out_dir/noisy/<segment_id>.wav
        out_dir/clean/<segment_id>.wav
        out_dir/manifest.csv   (segment_id, snr_db, donor_category, donor_segment)
    Returns the manifest as a DataFrame.

    Uses the same NoisyMixDataset logic (same-file donor guard, mix_at_snr) but a
    distinct default seed so frozen pairs differ from any training draw.
    """
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    (out_dir / "noisy").mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)

    ds = NoisyMixDataset(clean_df, donor_df, target_sr=target_sr,
                         snr_range=snr_range, seed=seed)
    manifest = []
    for i in range(len(ds)):
        item = ds[i]
        sid = item["audio_path"].replace("/", "_").replace(":", "_")
        noisy = item["audio"].squeeze(0).numpy()
        clean = item["target"].squeeze(0).numpy()
        sf.write(out_dir / "noisy" / f"{sid}.wav", noisy, target_sr)
        sf.write(out_dir / "clean" / f"{sid}.wav", clean, target_sr)
        manifest.append({
            "segment_id": item["audio_path"],
            "snr_db": round(float(item["snr_db"]), 3),
            "donor_category": item["donor_category"],
        })

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id", "snr_db", "donor_category"])
        w.writeheader()
        w.writerows(manifest)

    import pandas as pd
    return pd.DataFrame(manifest)
