"""What each target id means to the classifier head, as a set of head columns.

The head has one column per fine class plus background. Two kinds of target do
not fit that:

* **Coarse ids** (Arch-43 band C: door-any 51, furniture-any 52, fixture-any 53,
  appliance-any 54). A US CAD layer says "this is a plumbing fixture" but not
  which one. Such a target is scored against the *sum* of its members'
  probabilities -- the marginal -- so the model is told where a fixture is
  without being told a lie about which fixture.

* **Classes a source never labels.** FloorPlanCAD has no electrical class; the
  US corpus has no toilet class. An unmatched query is normally pushed to
  background, which on a US tile teaches "a toilet is nothing". With a
  per-source `annotated` mask, an unmatched query is instead scored against
  background *or any class this source does not annotate*.

Both reduce to the same thing: every target is a boolean row over the head's
columns, and the loss is -log(sum of softmax over that row). A one-hot row is
exactly the ordinary cross-entropy, so a fully annotated source with no coarse
ids trains identically to before.
"""

import os.path as osp
import sys

import torch

_DATASET_DIR = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
                        "dataset")


def _coarse_groups(num_classes):
    """Coarse id -> member list, for the taxonomy that defines them (Arch-43)."""
    if num_classes != 43:
        return {}
    if _DATASET_DIR not in sys.path:
        sys.path.insert(0, _DATASET_DIR)
    from taxonomy import COARSE_GROUPS
    return {int(k): [int(m) for m in v] for k, v in COARSE_GROUPS.items()}


def target_rows(num_classes):
    """Bool [K, num_classes + 1]: row t is the set of head columns target id t covers.

    Rows exist for every fine id, background, and every coarse id. A coarse id
    with no members (ignore) gets an empty row; `usable` says which ids may be
    targets at all.
    """
    groups = _coarse_groups(num_classes)
    k = max([num_classes] + list(groups)) + 1
    rows = torch.zeros(k, num_classes + 1, dtype=torch.bool)
    for c in range(num_classes + 1):
        rows[c, c] = True
    for cid, members in groups.items():
        for m in members:
            if 0 <= m < num_classes:
                rows[cid, m] = True
    return rows


def usable(rows, sem_id, num_classes):
    """True if `sem_id` can be a foreground target: a fine class or a coarse id
    with members. Background, ignore and unknown ids are not."""
    if 0 <= sem_id < num_classes:
        return True
    return num_classes < sem_id < rows.shape[0] and bool(rows[sem_id].any())


def annotated_mask(annotated, num_classes):
    """Bool [num_classes + 1] from a class-id list (None = everything). Background
    is always annotated: every source says what is not an object."""
    m = torch.ones(num_classes + 1, dtype=torch.bool)
    if annotated is not None:
        m[:] = False
        for c in annotated:
            if 0 <= int(c) < num_classes:
                m[int(c)] = True
        m[num_classes] = True
    return m
