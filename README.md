# Auscultation audio denoising with pytorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#denoising-inference">Inference</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Denoising digital stethoscope recordings (bachelor's thesis, HSE). The recordings are contaminated with speech, crying, heartbeat, power-line hum etc., which makes some data unusable both for automated pathology classification and for doctors annotating it. The goal is to clean it up.

What's inside:

- **DSP baselines** — notch, Butterworth bandpass, spectral Wiener.
- **Demucs** - pretrained source separation, v3 and v4; subtracting the `vocals` stem turns out to work as a voice remover on auscultation audio.
- **Combined cascade** — Demucs → notch → bandpass (→ Wiener). Best non-trained pipeline and a baseline.
- **Wave-U-Net** — small 1-D U-Net trained on synthetic (noisy, clean) pairs. 

The data itself is private medical recordings and is not in the repo.

## Installation

Follow these steps to install the project:

0. Clone the repository:
   ```bash
   git clone https://github.com/arimmean/digital-stethoscope.git
   cd digital-stethoscope
   ```

1. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n digital-stethoscope python=3.10

   # activate env
   conda activate digital-stethoscope
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/3.10/bin/python3 -m venv digital-stethoscope

   # alternatively, using default python version
   python3 -m venv digital-stethoscope

   # activate env
   source digital-stethoscope/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Denoising (inference)

Point it at a directory of wavs, pick a model, get denoised wavs in `data/saved/<save_path>/`:

```bash
python inference.py model=dsp_baseline \
    datasets.test.audio_dir=path/to/wavs \
    inferencer.save_path=my_run
```

Available models (`src/configs/model/`): `identity`, `notch`, `bandpass`, `wiener`, `dsp_baseline` (notch→bandpass), `dsp_baseline_wiener`, `demucs`, `demucs_v4`, `combined` (demucs_v4→notch→bandpass), `combined_wiener`, `wave_unet`.

Demucs-based models must run at their native rate, so add `datasets.test.target_sr=44100` for `demucs*` / `combined*`. First run downloads the pretrained weights.

For a trained Wave-U-Net checkpoint:

```bash
python inference.py model=wave_unet \
    inferencer.from_pretrained=saved/<run>/model_best.pth \
    datasets.test.audio_dir=path/to/wavs \
    inferencer.save_path=my_run
```

## Training

Training needs the synthetic pairs (clean pathology segments + noise donor segments mixed on the fly at random SNR — see `src/datasets/`), which in turn need the actual dataset, so this part is not reproducible without the data.

```bash
python train.py -cn=denoise
```

Config is `src/configs/denoise.yaml` (loss weights, crop length, model size — all overridable from the CLI). Logging goes to CometML, set `COMET_API_KEY` in the environment.

## Evaluation

Two axes:

- **Full-reference** on the frozen synthetic val/test pairs:

  ```bash
  python eval_fullref.py --baseline noisy --split test          # the do-nothing floor
  python eval_fullref.py --method combined_wiener --split test  # any model config
  python eval_fullref.py --ckpt saved/<run>/model_best.pth      # a trained checkpoint
  ```

- **DNSMOS** (no-reference) on real noisy recordings — `notebooks/05_dnsmos_eval.ipynb`. Speech-trained metric on breathing audio, so we prefer the BAK (background noise) score per noise category.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
