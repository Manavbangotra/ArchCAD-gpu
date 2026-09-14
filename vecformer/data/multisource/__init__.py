"""
Joint training over several line-JSON corpora (FloorPlanCAD, US plan windows,
CubiCasa5K), each a FloorPlanCAD-format dataset.

    dataset_name: multisource
    dataset_args:
        common:                     # passed to every source's FloorPlanCAD dataset
            use_text: true
            train_transform_args: {...}
            eval_transform_args: {...}
        sources:
            - name: fpcad           # a key of dataset/taxonomy.py ANNOTATED
              root_dir: datasets/FloorPlanCAD-lines
              weight: 0.35          # share of training samples
              background_remap: [35, 43]
            - name: us
              root_dir: ../dataset/us_plans/lines
              weight: 0.45
              splits: {val: test}   # split directory per role (default: same name)
        source_order: []            # must equal the model config's `sources`; empty = ANNOTATED order

Training draws samples with replacement so each source contributes its `weight`
share regardless of corpus size (the US corpus has ~5x the windows of
FloorPlanCAD). An epoch is the summed size of the corpora. Evaluation keeps the
sources apart: val/test are dicts {name: dataset}, so the trainer reports
eval_fpcad_PQ, eval_us_PQ, ... and a pooled number cannot hide a regression on
one source.
"""
import os.path as osp
import sys
from typing import Dict, List

import torch
from torch.utils.data import ConcatDataset

from data import DatasetSplits, register_dataset
from data.floorplancad.floorplancad import FloorPlanCAD

_DATASET_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", "dataset"))


def default_source_order() -> List[str]:
    if _DATASET_DIR not in sys.path:
        sys.path.insert(0, _DATASET_DIR)
    import taxonomy
    return list(taxonomy.ANNOTATED)


class WeightedConcat(ConcatDataset):
    """ConcatDataset with per-sample sampling weights (read by VecFormerTrainer)."""

    def __init__(self, datasets, weights, names):
        super().__init__(datasets)
        self.names = list(names)
        total = float(sum(weights))
        per = []
        for ds, w in zip(datasets, weights):
            per.append(torch.full((len(ds),), (w / total) / max(len(ds), 1), dtype=torch.double))
        self.sample_weights = torch.cat(per) if per else torch.zeros(0, dtype=torch.double)


def build_sources(dataset_args: dict):
    common = dict(dataset_args.get("common", {}))
    order = list(dataset_args.get("source_order") or default_source_order())
    train, val, test, weights, names = [], {}, {}, [], []
    for src in dataset_args["sources"]:
        name = src["name"]
        if name not in order:
            raise ValueError(f"source {name!r} is not in source_order {order}")
        splits = {"train": "train", "val": "val", "test": "test", **src.get("splits", {})}
        kwargs = dict(common, root_dir=src["root_dir"], source_id=order.index(name))
        for key in ("background_remap", "use_text", "max_texts", "train_transform_args", "eval_transform_args"):
            if key in src:
                kwargs[key] = src[key]

        def make(role):
            ds = FloorPlanCAD(**kwargs, split=splits[role])
            # the split directory may differ from the role (US val = test); the
            # transform must follow the role, never augmenting evaluation data
            ds.split = "train" if role == "train" else "val"
            return ds

        if src.get("weight", 1.0) > 0:
            train.append(make("train"))
            weights.append(float(src.get("weight", 1.0)))
            names.append(name)
        val[name] = make("val")
        test[name] = make("test")
    return WeightedConcat(train, weights, names), val, test


@register_dataset("multisource")
def build(dataset_args: dict):
    train, val, test = build_sources(dataset_args)
    return DatasetSplits(train=train, val=val, test=test), FloorPlanCAD.collate_fn
