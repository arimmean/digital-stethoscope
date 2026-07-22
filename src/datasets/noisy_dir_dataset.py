from pathlib import Path

from src.datasets.base_dataset import BaseDataset


_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


class NoisyDirDataset(BaseDataset):
    """
    Structure-agnostic denoising-inference dataset: walks a directory and
    picks up every audio file (no labels, no metadata schema assumed).

    The goal is "run a denoiser over a folder of recordings
    and write the cleaned versions out." Works for our stethoscope data, for
    dnd outputs, for downloaded benchmark dirs — anything.

    Args:
        audio_dir (str | Path): directory to scan.
        recursive (bool): if True, walk subdirectories too.
        target_dir (str | Path | None): optional parallel directory of clean
            references (matched by stem). Useful later for full-reference
            metric runs; in the no-reference Phase 2 inference path, leave None.
    """

    def __init__(
        self,
        audio_dir,
        recursive=False,
        target_dir=None,
        *args,
        **kwargs,
    ):
        audio_dir = Path(audio_dir)
        target_dir = Path(target_dir) if target_dir is not None else None

        iterator = audio_dir.rglob("*") if recursive else audio_dir.iterdir()

        index = []
        for path in iterator:
            if path.suffix.lower() not in _AUDIO_EXTS:
                continue
            entry = {"path": str(path)}
            if target_dir is not None:
                candidate = target_dir / (path.stem + path.suffix)
                if candidate.exists():
                    entry["target_path"] = str(candidate)
            index.append(entry)

        index.sort(key=lambda e: e["path"])  # deterministic order
        super().__init__(index, *args, **kwargs)
