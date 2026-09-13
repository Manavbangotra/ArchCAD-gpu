"""
Dataset construction helpers.

`build_dataset` / `build_dataloader` are the entry points used by
tools/train.py and tools/test.py.
"""

from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from svgnet.util import worker_init_fn

from .multi import MultiSourceDataset, SourceBatchSampler
from .svg import (BG_SEMANTIC_ID, NUM_CLASSES, SVG_CATEGORIES, SVGDataset)

__all__ = [
    "SVGDataset", "SVG_CATEGORIES", "NUM_CLASSES", "BG_SEMANTIC_ID",
    "MultiSourceDataset", "SourceBatchSampler",
    "build_dataset", "build_dataloader",
]


def build_dataset(data_cfg, logger):
    assert "type" in data_cfg
    _data_cfg = dict(data_cfg).copy()
    _data_cfg["logger"] = logger
    data_type = _data_cfg.pop("type")

    # Not every config key is a dataset argument; drop the ones that aren't.
    _data_cfg.pop("split_path", None)

    if data_type == "svg":
        return SVGDataset(**_data_cfg)
    if data_type == "multi":
        return MultiSourceDataset(**_data_cfg)
    raise ValueError(f"Unknown dataset type: {data_type}")


def build_dataloader(args, dataset, batch_size=1, num_workers=1, training=True, dist=False):
    shuffle = training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist else None
    if sampler is not None:
        shuffle = False

    seed = getattr(args, "seed", None)

    if training and isinstance(dataset, MultiSourceDataset):
        if dist:
            raise NotImplementedError("multi-source training is single-GPU for now")
        return DataLoader(
            dataset,
            batch_sampler=dataset.batch_sampler(batch_size, seed=seed or 0),
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=partial(worker_init_fn, seed=seed),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=training,
        pin_memory=True,
        # Windows spawns workers; re-spawning them every epoch and every
        # validation pass re-imports torch each time.
        persistent_workers=num_workers > 0,
        worker_init_fn=partial(worker_init_fn, seed=seed),
    )
