"""The trained VecFormer / TextCAD model inside the takeoff.

A parsed sheet is cut into the real-size, half-overlapping windows the model was
trained on (dataset/to_lines_us.py convert_page, with page indices kept), every
window is predicted with the evaluation transforms, and the windows are
recombined by sliding-window aggregation (takeoff/swa.py) into page objects for
the rest of the takeoff (openings, rooms, document).

    model = WindowModel("vecformer/configs/model/product_arch43.yaml", "outputs/product/checkpoint-best")
    objects, n_windows = page_objects_swa(data, lines, scales.at, viewport_of, model)
"""
import os.path as osp
import sys
from typing import List

import numpy as np

from .swa import WindowPrediction, aggregate

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
VEC = osp.join(ROOT, "vecformer")
EVAL_TRANSFORM = dict(random_vertical_flip=0.0, random_horizontal_flip=0.0, random_rotate=False,
                      random_scale=[1.0, 1.0], random_translation=[0.0, 0.0])


def _vecformer_imports(cpu_kernels=False):
    for p in (VEC, osp.join(ROOT, "dataset")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if cpu_kernels:
        sys.path.insert(0, osp.join(VEC, "checks"))
        import cpu_kernels as ck
        ck.install()


class WindowModel:
    def __init__(self, model_config, checkpoint=None, device=None, cpu_kernels=False, batch_size=4,
                 model_args_override=None):
        _vecformer_imports(cpu_kernels)
        import torch
        import yaml
        from data.floorplancad.floorplancad import FloorPlanCAD
        from model.vecformer import VecFormer, VecFormerConfig, load_matching

        self.torch = torch
        with open(model_config, encoding="utf-8") as f:
            args = dict(yaml.safe_load(f)["model_args"])
        args.pop("init_checkpoint", None)
        args.update(model_args_override or {})
        self.model = VecFormer(VecFormerConfig(**args))
        if checkpoint:
            report = load_matching(self.model, checkpoint)
            if report["missing"]:
                raise ValueError(f"checkpoint {checkpoint} does not match {model_config}: "
                                 f"{len(report['missing'])} tensors missing, e.g. {report['missing'][:3]}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.model.set_inference_mode(True)
        cfg = self.model.config
        self.num_classes = cfg.num_semantic_classes
        self.stuff_classes = list(cfg.stuff_class_idxs)
        self.batch_size = batch_size
        self.dataset = FloorPlanCAD.for_inference(EVAL_TRANSFORM, use_text=bool(cfg.text_config["enabled"]),
                                                  use_layer_names=bool(cfg.layer_name_config["enabled"]))
        self.collate = FloorPlanCAD.collate_fn

    def predict(self, records: List[dict]) -> List[dict]:
        """Window records -> per window dict(sem_probs (n, C+1), inst_masks (m, n), inst_labels, inst_scores)."""
        torch = self.torch
        out = []
        for start in range(0, len(records), self.batch_size):
            items = [self.dataset.item(r) for r in records[start:start + self.batch_size]]
            batch = self.collate(items)
            for key in ("sem_ids", "inst_ids", "data_paths"):
                batch.pop(key, None)
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                res = self.model(**batch)
            sem, pan = res.dict_pred_sem_segs, res.dict_pred_panop_segs
            for b in range(len(items)):
                out.append(dict(sem_probs=sem["list_pred_probs"][b].float().cpu().numpy(),
                                inst_masks=pan["list_pred_masks"][b].cpu().numpy(),
                                inst_labels=pan["list_pred_labels"][b].cpu().numpy(),
                                inst_scores=pan["list_pred_scores"][b].float().cpu().numpy()))
        return out


def primitive_positions(rec):
    """Centre of each window primitive, normalised to [-0.5, 0.5] of the window."""
    coords = np.asarray(rec["coords"], dtype=np.float64).reshape(-1, 4)
    pid = np.asarray(rec["primitive_ids"], dtype=np.int64)
    n = len(rec["semantic_ids"])
    mid = np.stack([(coords[:, 0] + coords[:, 2]) / 2, (coords[:, 1] + coords[:, 3]) / 2], 1)
    sums = np.zeros((n, 2))
    np.add.at(sums, pid, mid)
    counts = np.bincount(pid, minlength=n).clip(min=1)[:, None]
    size = float(rec["viewBox"][2])
    return sums / counts / size - 0.5


def page_objects_swa(data, text_lines, scale_at, viewport_of, model, window_m=10.0, step_ratio=0.5,
                     sampling_ratio=0.01, **aggregate_kwargs):
    """PageObjects for one parsed page from the model over sliding windows -> (objects, n_windows)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    sys.path.insert(0, osp.join(ROOT, "dataset"))
    from to_lines_us import convert_page

    wins = convert_page(data, text_lines, scale_at, viewport_of, window_m, sampling_ratio, min_fg=0,
                        max_segments=10 ** 9, meta_base={}, step_ratio=step_ratio, keep_idxs=True)
    if not wins:
        return [], 0
    preds = model.predict([rec for _, rec in wins])
    windows = [WindowPrediction(suffix, np.asarray(rec["idxs"]), primitive_positions(rec), p["sem_probs"],
                                p["inst_masks"], p["inst_labels"], p["inst_scores"])
               for (suffix, rec), p in zip(wins, preds)]
    objects, _, _ = aggregate(len(data["args"]), windows, model.num_classes, stuff_classes=model.stuff_classes,
                              **aggregate_kwargs)
    return objects, len(wins)
