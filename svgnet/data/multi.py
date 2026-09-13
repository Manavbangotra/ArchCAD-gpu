"""Several SVG datasets trained as one: FloorPlanCAD, CubiCasa5K and the US corpus.

Config shape (see configs/svg/svg_pointT_joint43_3060.yaml):

    data:
      train:
        type: multi
        split: train
        samples_per_epoch: 6000     # an "epoch" is this many drawings, drawn by weight
        img_size: 700               # every key outside `sources` is shared...
        num_classes: 43
        sources:
          - name: fpcad
            data_root: dataset/FloorplanCAD/json
            weight: 0.35
            annotated: fpcad          # ...and each source may override any of them
          - name: us
            data_root: dataset/us_plans/json5
            weight: 0.45
            coarse_policy: keep
            annotated: us

Two things a plain ConcatDataset gets wrong, and this does not:

* Instance ids are offset by the sample index to stay unique within a batch.
  ConcatDataset hands each sub-dataset its *local* index, so sample 5 of one
  source and sample 5 of another collide; here every source gets an
  `index_base`.
* Sources differ in what they label, so a batch must come from one source for
  its `annotated` mask to mean anything. SourceBatchSampler guarantees that, and
  draws sources by weight rather than by size -- FloorPlanCAD would otherwise
  be drowned by 12k US tiles that label no furniture at all.
"""

import bisect

import torch
from torch.utils.data import Dataset, Sampler

from .svg import SVGDataset

# Keys that describe how a source is sampled, not how it is loaded.
_SAMPLING_KEYS = ("name", "weight")


class MultiSourceDataset(Dataset):
    def __init__(self, sources, split, logger=None, samples_per_epoch=0, **common):
        if not sources:
            raise ValueError("data type 'multi' needs at least one entry in `sources`")
        self.split = split
        self.samples_per_epoch = int(samples_per_epoch or 0)
        self.datasets, self.names, self.weights, self.starts = [], [], [], []
        base = 0
        for src in sources:
            cfg = dict(common)
            cfg.update(dict(src))
            name = cfg.pop("name")
            weight = float(cfg.pop("weight", 1.0))
            cfg.setdefault("source", name)
            ds = SVGDataset(split=split, logger=logger, index_base=base, **cfg)
            if len(ds) == 0:
                if logger is not None:
                    logger.warning(f"source '{name}' has no drawings for split '{split}' -- skipped")
                continue
            self.starts.append(base)
            self.datasets.append(ds)
            self.names.append(name)
            self.weights.append(weight)
            base += len(ds)
        if not self.datasets:
            raise FileNotFoundError(f"no source has drawings for split '{split}'")
        self.total = base
        if logger is not None:
            parts = ", ".join(f"{n} {len(d)} (w {w:g})"
                              for n, d, w in zip(self.names, self.datasets, self.weights))
            logger.info(f"Multi-source {split}: {parts}")

    def __len__(self):
        return self.total

    def locate(self, index):
        k = bisect.bisect_right(self.starts, index) - 1
        return k, index - self.starts[k]

    def __getitem__(self, index):
        k, local = self.locate(index)
        return self.datasets[k][local]

    def collate_fn(self, batch):
        return self.datasets[0].collate_fn(batch)

    def ranges(self):
        return [(s, s + len(d)) for s, d in zip(self.starts, self.datasets)]

    def batch_sampler(self, batch_size, seed=0):
        n = self.samples_per_epoch or self.total
        return SourceBatchSampler(self.ranges(), self.weights, batch_size,
                                  num_batches=n // batch_size, seed=seed)


class SourceBatchSampler(Sampler):
    """Batches drawn from one source at a time, sources chosen by weight.

    Each call to __iter__ is a new epoch with a new, reproducible permutation:
    sampling is without replacement within a source until it is exhausted, then
    that source reshuffles, so a small source is cycled rather than repeated at
    random.
    """

    def __init__(self, ranges, weights, batch_size, num_batches, seed=0):
        self.ranges = list(ranges)
        w = torch.tensor(weights, dtype=torch.double)
        self.probs = w / w.sum()
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed or 0)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch)
        self.epoch += 1
        pools = [[] for _ in self.ranges]
        picks = torch.multinomial(self.probs, self.num_batches, replacement=True, generator=g)
        for k in picks.tolist():
            lo, hi = self.ranges[k]
            batch = []
            while len(batch) < self.batch_size:
                if not pools[k]:
                    pools[k] = (torch.randperm(hi - lo, generator=g) + lo).tolist()
                batch.append(pools[k].pop())
            yield batch
