from torch import nn


class IdentityModel(nn.Module):
    """
    Pass-through "denoiser": returns the input audio unchanged.

    Useful as a smoke test for the inference pipeline (dataset → collate →
    model → save) and as a true control condition for downstream-task
    evaluation later: any real denoiser should beat the identity baseline.
    """

    def forward(self, audio, **batch):
        return {"audio": audio}

    def __str__(self):
        return f"{type(self).__name__}(identity passthrough, 0 params)"
