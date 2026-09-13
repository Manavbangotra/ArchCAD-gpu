"""Evaluation kept separately per training source.

One pooled mIoU over FloorPlanCAD, CubiCasa and the US corpus hides the thing
joint training can break: the US tiles outnumber the others, so FloorPlanCAD
furniture accuracy could collapse without the pooled number moving. Each source
gets its own PointWiseEval and InstanceEval, and "best" is the macro mean.

PQ and mIoU only look at primitives with a ground-truth object, so neither can
see a model that paints objects onto background. `bg_fp` is the share of
ground-truth background primitives covered by a predicted instance. On a fully
annotated source (FloorPlanCAD) that is a hallucination rate; on a partially
annotated one it also counts real but unlabelled objects, so read it as an
upper bound there.
"""

import numpy as np

from .point_wise_eval import InstanceEval, PointWiseEval


def batch_source(batch):
    """Source name of a loader batch, or 'all' for single-source batches."""
    if len(batch) == 10 and isinstance(batch[-2], dict):
        names = {n for n in batch[-2].get("source") or [] if n}
        if len(names) == 1:
            return names.pop()
        if len(names) > 1:
            return "mixed"
    return "all"


class SourceEvals:
    def __init__(self, num_classes, gpu_num=1):
        self.num_classes = num_classes
        self.gpu_num = gpu_num
        self.sem, self.ins, self.bg = {}, {}, {}

    def _get(self, name):
        if name not in self.sem:
            self.sem[name] = PointWiseEval(num_classes=self.num_classes,
                                           ignore_label=self.num_classes, gpu_num=self.gpu_num)
            self.ins[name] = InstanceEval(num_classes=self.num_classes,
                                          ignore_label=self.num_classes, gpu_num=self.gpu_num)
            self.bg[name] = [0, 0]
        return self.sem[name], self.ins[name], self.bg[name]

    def update(self, name, res):
        sem, ins, bg = self._get(name)
        sem_preds = res["semantic_scores"].argmax(dim=1).cpu().numpy()
        sem_gts = res["semantic_labels"].cpu().numpy()
        sem.update(sem_preds, sem_gts)
        ins.update(res["instances"], res["targets"], res["lengths"])

        real = res["lengths"].cpu().numpy() > 0
        is_bg = (sem_gts == self.num_classes) & real[:len(sem_gts)]
        covered = np.zeros(len(sem_gts), dtype=bool)
        for inst in res["instances"]:
            m = np.asarray(inst["masks"])[:len(sem_gts)]
            covered[:len(m)] |= m
        bg[0] += int((covered & is_bg).sum())
        bg[1] += int(is_bg.sum())

    def report(self, logger, metric="miou"):
        """Log per source; return (macro score, {source: {miou, pq, bg_fp}})."""
        out = {}
        for name in sorted(self.sem):
            logger.info(f"==== source: {name} ====")
            miou, _ = self.sem[name].get_eval(logger)
            pq, _, _ = self.ins[name].get_eval(logger)
            fp, n = self.bg[name]
            bg_fp = 100.0 * fp / max(n, 1)
            logger.info(f"[{name}] mIoU {miou:.2f}  PQ {pq:.2f}  background covered by "
                        f"predictions {bg_fp:.2f}% of {n} primitives")
            out[name] = {"miou": float(miou), "pq": float(pq), "bg_fp": bg_fp}
        key = "pq" if str(metric).lower() == "pq" else "miou"
        if not out:
            return 0.0, out
        macro = float(np.mean([v[key] for v in out.values()]))
        if len(out) > 1:
            logger.info("macro over sources: " + ", ".join(
                f"{k} {np.mean([v[k] for v in out.values()]):.2f}" for k in ("miou", "pq", "bg_fp")))
        return macro, out
