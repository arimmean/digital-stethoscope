import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Denoising inferencer. Runs a model over each evaluation partition and
    writes per-utterance denoised wavs to disk. Metrics are optional and
    intentionally not part of the Phase 2 path — outputs are meant to be
    listened to first.

    Saved file naming: <save_path>/<part>/<original_stem>.wav
    Each saved wav is trimmed back to its original (pre-padding) length via
    `audio_lengths` from the collate.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        # SR written to saved wavs. The config sets this to the denoiser's
        # working rate (datasets.test.target_sr) via Hydra ${...} interpolation,
        # so saved audio is never silently relabeled. The inferencer does NOT
        # resample; downstream-classifier rate conversion is a separate step.
        self.save_sample_rate = int(self.cfg_trainer.save_sample_rate)

        # metrics are optional in Phase 2
        self.metrics = metrics
        if self.metrics is not None and self.metrics.get("inference"):
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        # checkpoint loading: only required if from_pretrained is set.
        # Stateless DSP "models" have no checkpoint — leave from_pretrained
        # null in their configs.
        pretrained_path = config.inferencer.get("from_pretrained")
        if not skip_model_load and pretrained_path is not None:
            self._from_pretrained(pretrained_path)

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        if self.batch_transforms is not None:
            batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        if isinstance(outputs, torch.Tensor):
            outputs = {"audio": outputs}
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        self._save_outputs(batch, part)
        return batch

    def _save_outputs(self, batch, part):
        if self.save_path is None:
            return
        denoised = batch["audio"]  # [B, 1, T_max] after model
        lengths = batch["audio_lengths"]  # [B]
        paths = batch["audio_paths"]  # list[str]

        out_dir = self.save_path / part
        for i in range(denoised.shape[0]):
            length = int(lengths[i])
            wav = denoised[i, :, :length].detach().cpu()
            if wav.dtype != torch.float32:
                wav = wav.to(torch.float32)
            stem = paths[i].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            torchaudio.save(
                str(out_dir / f"{stem}.wav"),
                wav,
                self.save_sample_rate,
            )

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is None:
            return {}
        return self.evaluation_metrics.result()
