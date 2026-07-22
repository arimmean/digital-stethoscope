import torch


def collate_fn(dataset_items: list[dict]):
    """
    Pad variable-length audio into a batch.

    Each item from a denoising dataset is expected to contain:
        - "audio":      Tensor [1, T_i], the noisy input
        - "audio_path": str
        - "target":     Tensor [1, T_i] (optional, clean reference)

    Returns a dict with:
        - "audio":         Tensor [B, 1, T_max]   zero-padded on the right
        - "audio_lengths": Tensor [B]             original lengths in samples
        - "audio_paths":   list[str]              for naming saved outputs
        - "target":        Tensor [B, 1, T_max]   if every item had a target
        - "target_lengths":Tensor [B]             if every item had a target

    Right-padding with zeros is the standard convention for variable-length
    audio and lets the inferencer trim each saved wav back to its original
    length via `audio_lengths`.
    """
    audios = [item["audio"] for item in dataset_items]
    lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
    max_len = int(lengths.max())

    padded = torch.zeros(len(audios), audios[0].shape[0], max_len)
    for i, a in enumerate(audios):
        padded[i, :, : a.shape[-1]] = a

    batch = {
        "audio": padded,
        "audio_lengths": lengths,
        "audio_paths": [item["audio_path"] for item in dataset_items],
    }

    if all("target" in item for item in dataset_items):
        targets = [item["target"] for item in dataset_items]
        t_lengths = torch.tensor([t.shape[-1] for t in targets], dtype=torch.long)
        t_max = int(t_lengths.max())
        t_padded = torch.zeros(len(targets), targets[0].shape[0], t_max)
        for i, t in enumerate(targets):
            t_padded[i, :, : t.shape[-1]] = t
        batch["target"] = t_padded
        batch["target_lengths"] = t_lengths

    return batch
