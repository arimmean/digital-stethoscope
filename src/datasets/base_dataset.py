import logging
import random

import numpy as np
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for denoising datasets.

    An item is an audio waveform (noisy input) and, optionally, a clean target
    waveform.

    Subclasses build an `index` — a list of dicts. Each dict must contain at
    least:
        - "path": path to the (noisy) input audio file.
    Optionally:
        - "target_path": path to a clean reference (for full-reference metrics
          or supervised training pairs in later phases).
        - "audio_len": duration in seconds, used by the length filter.

    Convention for `__getitem__` keys (kept consistent across dataset, collate,
    model.forward, and the inferencer):
        - "audio":        Tensor [1, T], the noisy input
        - "audio_path":   str, source path (used for naming saved outputs)
        - "target":       Tensor [1, T], optional clean reference
    """

    def __init__(
        self,
        index,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        min_sample_rate=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            min_sample_rate (int | None): if set, drop files whose NATIVE sample
                rate is below this threshold instead of upsampling them. Use in
                training configs (e.g. min_sample_rate: ${...target_sr}) so the
                few low-rate files don't get fed in as interpolated fakes. Leave
                None for inference, where we want to denoise every recording.
        """
        self._assert_index_is_valid(index)

        # Always read each file's native sample rate (header-only, cheap) and
        # stash it in the entry as "native_sr", so it's inspectable downstream
        # and available to the optional min_sample_rate filter below.
        index = self._populate_native_sr(index)

        index = self._filter_by_sample_rate(index, min_sample_rate)
        index = self._filter_records_from_dataset(index, max_audio_length)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)

        instance_data = {
            "audio": audio,
            "audio_path": audio_path,
        }

        target_path = data_dict.get("target_path")
        if target_path is not None:
            instance_data["target"] = self.load_audio(target_path)

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # mono: keep first channel
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        return audio_tensor

    def preprocess_data(self, instance_data):
        """
        Apply instance transforms keyed by tensor name (same pattern as the
        ASR reference: each transform's key matches the dict key it acts on).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(index, max_audio_length):
        if max_audio_length is None:
            return index
        if not all("audio_len" in el for el in index):
            logger.info(
                "max_audio_length set but some index entries lack 'audio_len'; "
                "skipping length filter."
            )
            return index

        initial_size = len(index)
        exceeds = np.array([el["audio_len"] for el in index]) >= max_audio_length
        n_excluded = int(exceeds.sum())
        if n_excluded == 0:
            return index
        logger.info(
            f"{n_excluded} ({n_excluded / initial_size:.1%}) records are longer than "
            f"{max_audio_length} seconds. Excluding them."
        )
        return [el for el, drop in zip(index, exceeds) if not drop]

    @staticmethod
    def _populate_native_sr(index):
        """
        Read each file's native sample rate (header-only via libsndfile) and
        store it as entry["native_sr"]. Files whose header can't be read at all
        are skipped with a warning (per the corrupt-RIFF gotcha in
        dataset_findings) rather than killing construction.
        """
        kept = []
        for entry in index:
            try:
                entry["native_sr"] = torchaudio.info(entry["path"]).sample_rate
                kept.append(entry)
            except Exception as e:  # unreadable header -> drop, don't crash
                logger.warning(f"Skipping unreadable file {entry['path']}: {e}")
        return kept

    @staticmethod
    def _filter_by_sample_rate(index, min_sample_rate):
        """
        Optionally drop files whose native SR is below min_sample_rate, to avoid
        feeding upsampled (interpolated) audio into training. No-op when
        min_sample_rate is None. Relies on entry["native_sr"] populated above.
        """
        if min_sample_rate is None:
            return index
        initial_size = len(index)
        below = np.array([el["native_sr"] < min_sample_rate for el in index])
        n_excluded = int(below.sum())
        if n_excluded == 0:
            return index
        logger.info(
            f"{n_excluded} ({n_excluded / initial_size:.1%}) records have native "
            f"sample rate below {min_sample_rate} Hz. Excluding them "
            f"(would otherwise be upsampled)."
        )
        return [el for el, drop in zip(index, below) if not drop]

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, (
                "Each dataset item must include field 'path' - path to the "
                "(noisy) input audio file."
            )

    @staticmethod
    def _sort_index(index):
        if not all("audio_len" in el for el in index):
            return index
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(1)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
