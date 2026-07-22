"""
Trainer for waveform denoising (Wave-U-Net). Reuses BaseTrainer's train/eval
loop; overrides process_batch (model→loss→step, no text/CTC) and _log_batch
(logs audio examples, not transcriptions).
"""

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class DenoiseTrainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        if self.batch_transforms is not None:
            batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        # log a denoised/clean/noisy audio example from the batch
        if self.writer is None:
            return
        sr = self.config.writer.get("audio_sample_rate", 22050)
        if "audio" in batch:
            self.writer.add_audio("denoised", batch["audio"][0].detach().cpu(), sr)
        if "target" in batch:
            self.writer.add_audio("clean", batch["target"][0].detach().cpu(), sr)
