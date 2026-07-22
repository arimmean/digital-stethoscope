from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device. Batch_transforms is a nested dict 
    keyed by partition then tensor name.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each dataset partition. Also resolves
    batch_transforms (optional) and moves them to device.
    """
    if config.get("transforms") is not None and config.transforms.get(
        "batch_transforms"
    ) is not None:
        batch_transforms = instantiate(config.transforms.batch_transforms)
        move_batch_transforms_to_device(batch_transforms, device)
    else:
        batch_transforms = None

    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = instantiate(config.datasets[dataset_partition])

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
