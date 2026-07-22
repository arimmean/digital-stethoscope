"""
Full-reference evaluation of a denoiser on the FROZEN synthetic pairs
(data/synth_pairs/{split}).

Metrics:
- SI-SDR  (lead — reference-based, SR-agnostic, measures real signal distortion)
- PESQ, STOI (secondary — 16 kHz SPEECH metrics; resampled to 16k for scoring;
  proxies for breathing, cross-check by ear)

Scores a trained WaveUNet checkpoint AND fixed baselines (noisy-as-is = the floor;
optionally other methods) on the SAME pairs, so the model has the ceiling to beat.
Aggregates overall + per-SNR-bin + per-donor-category (from the pair manifest).

Usage:
  python eval_fullref.py --ckpt saved/<run>/model_best.pth --split test
  python eval_fullref.py --baseline noisy --split test      # the floor
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
)

from src.utils.io_utils import ROOT_PATH

PAIRS = ROOT_PATH / "data" / "synth_pairs"
PESQ_FS = 16000


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = OmegaConf.create(ckpt["config"]) if not OmegaConf.is_config(
        ckpt["config"]
    ) else ckpt["config"]
    model = instantiate(cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def denoise(model, noisy, sr, device, model_sr=None):
    """Run model on a 1-D numpy waveform → 1-D numpy.

    If model_sr is given and != sr, resample in→model_sr (librosa), run, resample
    out→sr so the estimate aligns with the (sr) clean reference for scoring.
    Used for the pretrained baselines that require their own rate (Demucs 44.1k).
    """
    x_np = noisy.astype("float32")
    if model_sr is not None and model_sr != sr:
        x_np = librosa.resample(x_np, orig_sr=sr, target_sr=model_sr)
    x = torch.from_numpy(x_np)[None, None, :].to(device)
    with torch.no_grad():
        out = model(audio=x)["audio"][0, 0].cpu().numpy()
    if model_sr is not None and model_sr != sr:
        out = librosa.resample(out, orig_sr=model_sr, target_sr=sr)
    return out


def load_method_model(method, device):
    """Instantiate a baseline/method model from src/configs/model/<method>.yaml.

    Returns (model, model_sr): model_sr is the rate the model must run at (its
    input_sample_rate if it has one, else None = runs at the pair SR). Resolves
    ${datasets.test.target_sr} so configs that interpolate it still build.
    """
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="src/configs"):
        # set target_sr=44100 so Demucs-based models (which assert input_sr==44100
        # at construction) build; we feed them 44.1k-resampled audio in denoise().
        cfg = compose(
            config_name="inference",
            overrides=[f"model={method}", "datasets.test.target_sr=44100"],
        )
    model = instantiate(cfg.model).to(device)
    model.eval()
    # the rate the model was BUILT for. DemucsV4Model exposes input_sample_rate;
    # DSPPipeline/combined don't (their filters carry it) → fall back to the
    # target_sr we composed with. DSP-only methods are SR-agnostic but we built
    # their filters at 44100 too, so run everything at the composed rate.
    model_sr = getattr(model, "input_sample_rate", None)
    if model_sr is None:
        model_sr = int(cfg.datasets.test.target_sr)  # = 44100
    return model, model_sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pth (a WaveUNet run)")
    ap.add_argument(
        "--baseline", default=None, choices=["noisy"],
        help="score a baseline instead of a model: 'noisy' = identity floor",
    )
    ap.add_argument(
        "--method", default=None,
        help="model config name (src/configs/model/<name>.yaml) to apply, e.g. "
             "dsp_baseline / demucs_v4 / combined_wiener — for the SI-SDR ceiling.",
    )
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pair_dir = PAIRS / args.split
    manifest = pd.read_csv(pair_dir / "manifest.csv")
    # manifest segment_id has ':'/'/' → filenames replaced them with '_'
    manifest["fname"] = (
        manifest["segment_id"].str.replace("/", "_").str.replace(":", "_") + ".wav"
    )

    si_sdr = ScaleInvariantSignalDistortionRatio()
    pesq = PerceptualEvaluationSpeechQuality(PESQ_FS, "wb")
    stoi = ShortTimeObjectiveIntelligibility(PESQ_FS)

    model = None
    model_sr = None
    if args.ckpt:
        model = load_model_from_ckpt(args.ckpt, args.device)
        tag = f"ckpt:{Path(args.ckpt).parent.name}"
    elif args.method:
        model, model_sr = load_method_model(args.method, args.device)
        tag = f"method:{args.method}"
    else:
        tag = f"baseline:{args.baseline}"

    rows = []
    for _, m in manifest.iterrows():
        clean, sr = sf.read(pair_dir / "clean" / m["fname"], dtype="float32")
        noisy, _ = sf.read(pair_dir / "noisy" / m["fname"], dtype="float32")

        if model is not None:
            est = denoise(model, noisy, sr, args.device, model_sr=model_sr)
        else:  # baseline 'noisy' = identity (the floor every method must beat)
            est = noisy

        n = min(len(est), len(clean))
        est_t = torch.from_numpy(est[:n].astype("float32"))
        clean_t = torch.from_numpy(clean[:n].astype("float32"))

        sisdr_v = float(si_sdr(est_t, clean_t))
        # PESQ/STOI at 16k
        est16 = librosa.resample(est[:n], orig_sr=sr, target_sr=PESQ_FS)
        cln16 = librosa.resample(clean[:n], orig_sr=sr, target_sr=PESQ_FS)
        e16 = torch.from_numpy(est16.astype("float32"))
        c16 = torch.from_numpy(cln16.astype("float32"))
        try:
            pesq_v = float(pesq(e16, c16))
        except Exception:
            pesq_v = float("nan")  # PESQ throws on silent/degenerate frames
        stoi_v = float(stoi(e16, c16))

        rows.append({
            "snr_db": m["snr_db"], "donor_category": m["donor_category"],
            "si_sdr": sisdr_v, "pesq": pesq_v, "stoi": stoi_v,
        })

    df = pd.DataFrame(rows)
    print(f"\n=== {tag} | split={args.split} | n={len(df)} ===")
    print("OVERALL (mean):")
    print(df[["si_sdr", "pesq", "stoi"]].mean().round(3).to_string())
    print("\nby donor_category (si_sdr / pesq / stoi):")
    print(df.groupby("donor_category")[["si_sdr", "pesq", "stoi"]].mean().round(3))
    # SNR bins
    df["snr_bin"] = pd.cut(df["snr_db"], [-6, 0, 6, 16],
                           labels=["[-5,0)", "[0,6)", "[6,15]"])
    print("\nby SNR bin:")
    print(df.groupby("snr_bin", observed=True)[["si_sdr", "pesq", "stoi"]].mean().round(3))


if __name__ == "__main__":
    main()
