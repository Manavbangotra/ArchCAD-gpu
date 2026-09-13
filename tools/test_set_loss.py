#!/usr/bin/env python3
"""Self-check for the classification loss over target rows. CPU only.

    python tools/test_set_loss.py

Pins four things:
  1. one-hot rows reproduce the weighted cross-entropy the model trained with
     before, so fully labelled data is unaffected;
  2. the vectorised prepare_targets builds the same targets as the original loop;
  3. a query matched to a coarse id (fixture-any) is scored on its members' sum;
  4. with an `annotated` mask, an unmatched query that predicts a class the
     source never labels is not pushed to background.
"""

import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "dataset"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from munch import Munch  # noqa: E402

import taxonomy as tx  # noqa: E402
from svgnet.model.criterion import SetCriterion  # noqa: E402
from svgnet.model.label_space import annotated_mask, target_rows  # noqa: E402
from svgnet.model.svgnet import SVGNet  # noqa: E402

FAILURES = []
C = 43
BG = 43


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def criterion(class_weights=-1):
    cfg = yaml.safe_load(open(osp.join(ROOT, "configs/svg/svg_pointT_fpcad43_3060.yaml"),
                              encoding="utf-8"))
    crit = Munch.fromDict(cfg["criterion"])
    crit.class_weights = class_weights
    return SetCriterion(None, {"loss_ce": 1.0}, crit)


class _Targets:
    """Just enough of SVGNet to call its target builders without the network."""
    num_classes = C
    target_rows = target_rows(C)
    prepare_targets = SVGNet.prepare_targets
    _prepare_targets_reference = SVGNet._prepare_targets_reference


def matched(labels, query_ids):
    rows = target_rows(C)
    t = {"labels": torch.tensor(labels), "label_rows": rows[torch.tensor(labels)]}
    J = torch.arange(len(labels))
    return [t], [(torch.tensor(query_ids), J)]


def nll(crit, logits, targets, indices):
    idx = crit._get_src_permutation_idx(indices)
    tc = torch.full(logits.shape[:2], C, dtype=torch.int64)
    tc[idx] = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    return crit._set_nll(logits, tc, idx, indices, targets), tc, idx


def test_one_hot_is_cross_entropy():
    g = torch.Generator().manual_seed(0)
    weights = (torch.rand(C, generator=g) + 0.5).tolist()
    for cw in (-1, weights):
        crit = criterion(cw)
        logits = torch.randn(1, 50, C + 1, generator=g)
        targets, indices = matched([3, 32, 26, 7], [5, 9, 11, 40])
        got, tc, _ = nll(crit, logits, targets, indices)
        want = F.cross_entropy(logits.transpose(1, 2), tc, crit.empty_weight)
        check(torch.allclose(got, want, atol=1e-5),
              f"one-hot rows != cross_entropy (weights={'set' if cw != -1 else -1}): "
              f"{got.item():.6f} vs {want.item():.6f}")


def test_vectorised_targets():
    g = torch.Generator().manual_seed(1)
    m = _Targets()
    for trial in range(20):
        n = 300
        sem = torch.randint(0, C + 1, (n,), generator=g)
        ins = torch.randint(-1, 12, (n,), generator=g)
        lab = torch.stack([sem, ins], 1)
        new = m.prepare_targets(lab)[0]
        ref = m._prepare_targets_reference(lab)[0]
        same = (torch.equal(new["labels"], ref["labels"])
                and torch.equal(new["masks"], ref["masks"]))
        check(same, f"prepare_targets differs from the reference loop (trial {trial})")
        if not same:
            break
    empty = torch.stack([torch.full((10,), BG), torch.full((10,), -1)], 1)
    t = m.prepare_targets(empty)[0]
    check(t["labels"].tolist() == [BG] and t["masks"].sum() == 0,
          "an all-background drawing still gets the dummy background target")


def test_coarse_targets_kept():
    m = _Targets()
    sem = torch.tensor([tx.FIXTURE_ANY] * 4 + [tx.IGNORE] * 3 + [BG] * 3)
    ins = torch.tensor([7] * 4 + [8] * 3 + [-1] * 3)
    t = m.prepare_targets(torch.stack([sem, ins], 1))[0]
    check(t["labels"].tolist() == [tx.FIXTURE_ANY],
          f"coarse fixture kept as one target, ignore dropped: got {t['labels'].tolist()}")
    members = set(tx.COARSE_GROUPS[tx.FIXTURE_ANY])
    got = set(torch.nonzero(t["label_rows"][0]).flatten().tolist())
    check(got == members, f"fixture-any row covers its members: {sorted(got)}")


def peaked(cls, q=4):
    logits = torch.full((1, q, C + 1), -10.0)
    logits[0, :, cls] = 10.0
    return logits


def test_marginal():
    crit = criterion()
    toilet = tx.COARSE_GROUPS[tx.FIXTURE_ANY][0]
    sofa = 10
    targets, indices = matched([tx.FIXTURE_ANY], [0])
    on_member, _, _ = nll(crit, _one(toilet), targets, indices)
    on_other, _, _ = nll(crit, _one(sofa), targets, indices)
    check(on_member.item() < 0.1, f"fixture-any matched to a toilet costs ~0: {on_member.item():.3f}")
    check(on_other.item() > 1.0, f"fixture-any matched to a sofa is penalised: {on_other.item():.3f}")


def _one(cls):
    """Query 0 predicts `cls`, every other query predicts background."""
    logits = peaked(BG)
    logits[0, 0, :] = -10.0
    logits[0, 0, cls] = 10.0
    return logits


def test_partial_annotation():
    crit = criterion()
    toilet = 26
    logits = _one(toilet)                      # query 0 says toilet, nothing matched it
    targets, indices = matched([32], [3])      # one wall, matched to query 3
    logits[0, 3, :] = -10.0
    logits[0, 3, 32] = 10.0

    # FloorPlanCAD: everything annotated. No-object queries weigh eos_coef (0.05)
    # against 1 for the matched wall, so one wrong query costs ~20 x 0.05 / 1.15.
    fp, _, _ = nll(crit, logits, targets, indices)

    us = [c for c in range(C) if c != toilet]                 # a source that never labels toilets
    targets[0]["annotated"] = annotated_mask(us, C)
    part, _, _ = nll(crit, logits, targets, indices)

    check(fp.item() > 0.5, f"fully annotated: an unmatched toilet is penalised ({fp.item():.3f})")
    check(part.item() < 0.1, f"toilet unannotated: the same prediction costs ~0 ({part.item():.3f})")

    # The mask must not excuse a class the source does label.
    logits2 = _one(32)
    logits2[0, 3, :] = -10.0
    logits2[0, 3, 32] = 10.0
    part2, _, _ = nll(crit, logits2, targets, indices)
    check(part2.item() > 0.5, f"an unmatched wall is still penalised ({part2.item():.3f})")


def main():
    tests = [test_one_hot_is_cross_entropy, test_vectorised_targets, test_coarse_targets_kept,
             test_marginal, test_partial_annotation]
    for fn in tests:
        fn()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"ok - {len(tests)} groups, set loss over target rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
