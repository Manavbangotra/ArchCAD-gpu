"""
What a target id means to the classifier head, as a set of head columns.

Ported from the repository's svgnet/model/label_space.py for VecFormer's losses.
The head has one column per fine class plus background. Two kinds of target do
not fit that when sources are trained jointly:

* Coarse ids (Arch-43: door-any 51, furniture-any 52, fixture-any 53,
  appliance-any 54). A US CAD layer says "plumbing fixture", not which one. Such a
  target is scored against the sum of its members' probabilities (the marginal):
  the model learns where a fixture is without being told which fixture.

* Classes a source never labels. FloorPlanCAD has no electrical class; the US
  corpus has no toilet class. A background label on such a source means
  "background or something this source does not annotate", so it is scored
  against background plus the unannotated columns. VecFormer's instance class
  loss only sees matched queries, so this matters for its dense semantic loss.

Both reduce to one rule: a target is a boolean row over the head's columns and
the loss is -log(sum of softmax over that row). A one-hot row is ordinary
cross-entropy, so a model without a label space trains exactly as upstream (the
criteria keep the upstream code path when `LabelSpace.plain`).

Ignore ids (Arch-43: 55) and ids outside the table have an empty row and are left
out of every loss and of instance targets.
"""
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence

import torch

_DATASET_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", "..", "dataset"))


def _taxonomy():
    if _DATASET_DIR not in sys.path:
        sys.path.insert(0, _DATASET_DIR)
    import taxonomy
    return taxonomy


class LabelSpace:
    def __init__(self, num_classes: int, coarse_groups: Optional[Dict[int, Sequence[int]]] = None,
                 annotated: Optional[Dict[str, Optional[Sequence[int]]]] = None,
                 sources: Sequence[str] = ()):
        """`annotated`: source name -> class ids that source labels wherever they occur
        (None = all). `sources`: the order of source ids in batches; source id -1, or
        a source missing from `annotated`, means fully annotated."""
        self.num_classes = C = num_classes
        groups = {int(k): [int(m) for m in v] for k, v in (coarse_groups or {}).items()}
        k = max([C] + list(groups)) + 1
        rows = torch.zeros(k, C + 1, dtype=torch.bool)
        for c in range(C + 1):
            rows[c, c] = True
        for cid, members in groups.items():
            for m in members:
                if 0 <= m < C:
                    rows[cid, m] = True
        self.rows = rows
        self.groups = groups
        self.sources = list(sources)
        ann = torch.ones(len(self.sources) + 1, C + 1, dtype=torch.bool)   # last row: source -1
        for i, name in enumerate(self.sources):
            ids = (annotated or {}).get(name)
            if ids is not None:
                ann[i] = False
                ann[i, [int(c) for c in ids if 0 <= int(c) < C]] = True
                ann[i, C] = True                 # every source says what is not an object
        self.annotated = ann
        self.plain = not groups and bool(ann.all())

    @classmethod
    def build(cls, name: Optional[str], num_classes: int, sources: Sequence[str] = ()):
        if not name:
            return None
        if name == "arch43":
            tx = _taxonomy()
            assert num_classes == tx.ARCH_NUM_CLASSES, f"arch43 needs num_semantic_classes={tx.ARCH_NUM_CLASSES}"
            return cls(num_classes, tx.COARSE_GROUPS, tx.ANNOTATED, sources or tuple(tx.ANNOTATED))
        raise ValueError(f"unknown label space {name!r}")

    # ------------------------------------------------------------------ #
    def _source_row(self, source: int) -> int:
        return source if 0 <= source < len(self.sources) else len(self.sources)

    def target_rows(self, labels: torch.Tensor, source: int = -1) -> torch.Tensor:
        """Bool (n, C+1) head columns each target covers; empty for ignore/unknown ids."""
        rows = self.rows.to(labels.device)
        labels = labels.long()
        known = (labels >= 0) & (labels < rows.shape[0])
        out = torch.zeros(labels.shape[0], self.num_classes + 1, dtype=torch.bool, device=labels.device)
        out[known] = rows[labels[known]]
        bg = labels == self.num_classes
        if bg.any():
            out[bg] |= ~self.annotated[self._source_row(source)].to(labels.device)
        return out

    def is_instance_target(self, sem_id: int) -> bool:
        """Fine class or coarse id with members; not background, ignore or unknown."""
        if 0 <= sem_id < self.num_classes:
            return True
        return self.num_classes < sem_id < self.rows.shape[0] and bool(self.rows[sem_id].any())

    @staticmethod
    def marginal_nll(logits: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """-log sum_{c in row} softmax(logits)_c, per row. Rows must be non-empty."""
        masked = logits.float().masked_fill(~rows, float("-inf"))
        return torch.logsumexp(logits.float(), dim=-1) - torch.logsumexp(masked, dim=-1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def resolve(self, target_labels: torch.Tensor, pred_labels: torch.Tensor) -> torch.Tensor:
        """Evaluation labels: a coarse target takes the predicted class when that is one
        of its members (a toilet predicted on a fixture-any object is right), otherwise
        its first member (then it counts as wrong, as it should). Ignore and unknown
        ids become background. Fine ids and background are unchanged."""
        out = target_labels.clone().long()
        pred = pred_labels.long()
        C = self.num_classes
        for cid, members in self.groups.items():
            m = out == cid
            if not m.any():
                continue
            if members:
                ok = torch.zeros(C + 1, dtype=torch.bool, device=out.device)
                ok[members] = True
                p = pred[m].clamp(0, C)
                out[m] = torch.where(ok[p], p, torch.full_like(p, members[0]))
            else:
                out[m] = C
        out[(out < 0) | (out > C)] = C
        return out

    def annotated_classes(self, source: int) -> torch.Tensor:
        return self.annotated[self._source_row(source)]
