from typing import Mapping, Dict, List
from dataclasses import dataclass
import os
import json

import torch
from transformers.trainer import EvalPrediction

@dataclass
class EvaluatorConfig:
    num_classes: int
    ignore_label: int
    iou_threshold: float
    output_dir: str


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.num_classes = config.num_classes
        self.ignore_label = config.ignore_label
        self.iou_threshold = config.iou_threshold
        self.output_dir = config.output_dir
        
    def __call__(self, preds, targets):
        return self.eval_panoptic_quality(preds, targets), self.eval_semantic_quality(preds["pred_sem_segs"], targets["sem_labels"], targets["prim_lens"])

    def eval_panoptic_quality(self, preds, targets):
        """
        Calculate panoptic quality metrics: PQ, SQ, RQ

        NOTE:a list of tensors means a batch of data, each tensor represents a sample in the batch

        Args:

            `preds` (`Dict[str, List[torch.Tensor]]`): In panoptic quality, the preds should have the following keys:

                \\- `pred_masks` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions, num_primitives)
                    each line represents a predicted instance, each column represents a primitive,
                    each value is 0 or 1, 1 means the primitive is part of the instance

                \\- `pred_labels` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions,)
                    each value represents the class of the predicted instance

            `targets` (`Dict[str, List[torch.Tensor]]`): In panoptic quality, the targets should have the following keys:

                \\- `target_masks` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_ground_truths, num_primitives)
                    each line represents a ground truth instance, each column represents a primitive,
                    each value is 0 or 1, 1 means the primitive is part of the instance

                \\- `target_labels` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_ground_truths,)
                    each value represents the class of the ground truth instance

                \\- `prim_lens` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_primitives,)
                    each value represents the length of the primitive

        Returns:

            `metric_states` (`Dict[str, torch.Tensor]`): Dictionary containing intermediate states for metric calculation

                \\- `tp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of true positive instances for each class

                \\- `fp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false positive instances for each class

                \\- `fn_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false negative instances for each class

                \\- `tp_iou_score_per_class` (`torch.Tensor`, shape is (num_classes,)): Summary of True positive IoU score for each class
        """
        # Initialize the metric states
        tp_per_class = torch.zeros(self.num_classes, dtype=torch.int32, device=preds["pred_masks"][0].device)
        fp_per_class = torch.zeros(self.num_classes, dtype=torch.int32, device=preds["pred_masks"][0].device)
        fn_per_class = torch.zeros(self.num_classes, dtype=torch.int32, device=preds["pred_masks"][0].device)
        tp_iou_score_per_class = torch.zeros(self.num_classes, dtype=torch.float32, device=preds["pred_masks"][0].device)
        # Log lengths to degrade the influence of lines with very large span. (Follow the FloorPlanCAD paper)
        log_prim_lens = [torch.log(1 + prim_len) for prim_len in targets["prim_lens"]]
        # Iterate over all batches
        for batch_idx in range(len(preds["pred_masks"])):
            # Iterate over all ground truth instances
            for target_idx in range(len(targets["target_masks"][batch_idx])):
                target_mask = targets["target_masks"][batch_idx][target_idx]
                target_label = targets["target_labels"][batch_idx][target_idx]
                # Ignore the ground truth instance with ignore label
                # In FloorPlanCAD dataset, the ignore label means the background
                if target_label == self.ignore_label:
                    continue
                # Iterate over all predictions to find the matching prediction for current ground truth instance
                found_match = False
                for pred_idx in range(len(preds["pred_masks"][batch_idx])):
                    pred_mask = preds["pred_masks"][batch_idx][pred_idx]
                    pred_label = preds["pred_labels"][batch_idx][pred_idx]
                    # Ignore the predicted instance with ignore label
                    # In FloorPlanCAD dataset, the ignore label means the background
                    if pred_label == self.ignore_label: # ignore the background
                        continue
                    # Calculate the IoU between the predicted instance and the ground truth instance
                    iou = self._calculate_primitive_iou(
                        pred_mask, target_mask,
                        log_prim_lens[batch_idx])
                    if iou > self.iou_threshold:
                        found_match = True
                        if pred_label == target_label:
                            tp_per_class[pred_label] += 1
                            tp_iou_score_per_class[pred_label] += iou
                        else:
                            fp_per_class[pred_label] += 1
                if not found_match:
                    fn_per_class[target_label] += 1
        return dict(
            tp_per_class=tp_per_class,
            fp_per_class=fp_per_class,
            fn_per_class=fn_per_class,
            tp_iou_score_per_class=tp_iou_score_per_class
        )

    def eval_semantic_quality(self, list_pred_sem_labels, list_target_sem_labels, list_primitive_lens):
        """
        Calculate semantic symbol spotting metrics: F1, wF1

        Args:
            `list_pred_sem_labels` (`List[torch.Tensor]`):
                a list of tensors, each tensor shape is (num_primitives,)
                each value represents the predicted class of primitive
            `list_target_sem_labels` (`List[torch.Tensor]`):
                a list of tensors, each tensor shape is (num_primitives,)
                each value represents the ground truth class of primitive
            `list_primitive_lens` (`List[torch.Tensor]`):
                a list of tensors, each tensor shape is (num_primitives,)
                each value represents the length of primitive
        Returns:
            `Dict[str, float]`: Dictionary containing semantic symbol spotting metrics
                `tp_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of true positive instances for each class
                `pred_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of predicted instances for each class
                `gt_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of ground truth instances for each class
                `w_tp_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of true positive instances for each class
                `w_pred_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of predicted instances for each class
                `w_gt_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of ground truth instances for each class
        """
        tp_per_class = torch.zeros(self.num_classes + 1, dtype=torch.int32, device=list_pred_sem_labels[0].device)
        pred_per_class = torch.zeros(self.num_classes + 1, dtype=torch.int32, device=list_pred_sem_labels[0].device)
        gt_per_class = torch.zeros(self.num_classes + 1, dtype=torch.int32, device=list_pred_sem_labels[0].device)
        
        w_tp_per_class = torch.zeros(self.num_classes + 1, dtype=torch.float32, device=list_pred_sem_labels[0].device)
        w_pred_per_class = torch.zeros(self.num_classes + 1, dtype=torch.float32, device=list_pred_sem_labels[0].device)
        w_gt_per_class = torch.zeros(self.num_classes + 1, dtype=torch.float32, device=list_pred_sem_labels[0].device)
        
        for pred_sem_labels, target_sem_labels, primitive_lens in zip(list_pred_sem_labels, list_target_sem_labels, list_primitive_lens):
            for i in range(pred_sem_labels.shape[0]):
                pred_sem_label = pred_sem_labels[i]
                target_sem_label = target_sem_labels[i]
                primitive_length = primitive_lens[i]
                
                pred_per_class[pred_sem_label] += 1
                gt_per_class[target_sem_label] += 1
                w_pred_per_class[pred_sem_label] += primitive_length
                w_gt_per_class[target_sem_label] += primitive_length

                if pred_sem_label == target_sem_label:
                    tp_per_class[pred_sem_label] += 1
                    w_tp_per_class[pred_sem_label] += primitive_length
                
        return dict(
            tp_per_class=tp_per_class,
            pred_per_class=pred_per_class,
            gt_per_class=gt_per_class,
            w_tp_per_class=w_tp_per_class,
            w_pred_per_class=w_pred_per_class,
            w_gt_per_class=w_gt_per_class
        )

    def eval_instance_quality(self, preds, data_paths):
        """
        `preds` (`Dict[str, List[torch.Tensor]]`): In panoptic quality, the preds should have the following keys:

                \\- `pred_masks` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions, num_primitives)
                    each line represents a predicted instance, each column represents a primitive,
                    each value is 0 or 1, 1 means the primitive is part of the instance

                \\- `pred_labels` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions,)
                    each value represents the class of the predicted instance
        """
        for batch_idx in range(len(preds["pred_masks"])):
            pred_labels = preds["pred_labels"][batch_idx]
            pred_masks = preds["pred_masks"][batch_idx]
            data_path = data_paths[batch_idx]
            data_split, data_name = data_path.split("/")[-2:]
            output_path = os.path.join(self.output_dir, data_split, data_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            """
            output json format:
            {
                "pred_instances": [
                    {
                       "primitive_ids": [1, 2, 3, ...],
                       "label": 1, 
                       "score": 0.8,
                    },
                    ...
                ]
            }
            """
            output = {"pred_instances": []}
            for idx in range(len(pred_masks)):
                pred_label = pred_labels[idx].item()
                pred_mask = pred_masks[idx]
                # 获得pred_mask中1的值的索引
                primitive_ids = pred_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                output["pred_instances"].append({
                    "primitive_ids": primitive_ids,
                    "label": pred_label,
                    "score": 1.0
                })
            with open(output_path, "w") as f:
                json.dump(output, f)
                


    def _calculate_primitive_iou(self, pred_mask, target_mask, primitive_length):
        """
        Calculate the IoU between the predicted instance and the ground truth instance

        Args:

            `pred_mask` (`torch.Tensor`, shape is (num_primitives,)):
                each value is 0 or 1, 1 means the primitive is part of the instance

            `target_mask` (`torch.Tensor`, shape is (num_primitives,)):
                each value is 0 or 1, 1 means the primitive is part of the instance

            `primitive_length` (`torch.Tensor`, shape is (num_primitives,)):
                each value represents the length of the primitive

        Returns:

            `torch.Tensor`: IoU between the predicted instance and the ground truth instance
        """
        inter_area = torch.sum(primitive_length[torch.logical_and(pred_mask, target_mask)])
        union_area = torch.sum(primitive_length[torch.logical_or(pred_mask, target_mask)])
        iou = inter_area / (union_area + torch.finfo(union_area.dtype).eps)
        return iou


@dataclass
class MetricsComputerConfig:
    num_classes: int
    thing_class_idxs: List[int]
    stuff_class_idxs: List[int]


class MetricsComputer:

    def __init__(self, config: MetricsComputerConfig) -> None:
        self.num_classes = config.num_classes
        self.thing_class_idxs = config.thing_class_idxs
        self.stuff_class_idxs = config.stuff_class_idxs
        self.dict_sublosses = {}
        self.metric_states = {}
        self.f1_states = {}

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        outputs, labels = eval_pred
        dict_sublosses, metric_states, f1_states = outputs

        self._update_dict_sublosses(dict_sublosses)
        self._update_metric_states(metric_states)
        self._update_f1_states(f1_states)
        if not compute_result:
            return
        metrics = {}
        metrics.update(self._get_dict_sublosses())
        metrics.update(self._compute_panoptic_quality())
        metrics.update(self._compute_f1_scores())
        return metrics
    
    def _update_f1_states(self, f1_states: Dict[str, torch.Tensor]) -> None:
        for key, value in f1_states.items():
            if key not in self.f1_states:
                self.f1_states[key] = \
                    torch.tensor(value).reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs) + 1).sum(dim=0)
            else:
                self.f1_states[key] += \
                    torch.tensor(value).reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs) + 1).sum(dim=0)
    
    def _compute_f1_scores(self) -> Dict[str, float]:
        results = {}
        eps = torch.finfo(torch.float32).eps
        # calculate total F1 and wF1
        
        tp = self.f1_states["tp_per_class"].sum()
        pred = self.f1_states["pred_per_class"].sum()
        gt = self.f1_states["gt_per_class"].sum()
        precision = tp / (pred + eps)
        recall = tp / (gt + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        results["F1"] = f1.item()
        
        w_tp = self.f1_states["w_tp_per_class"].sum()
        w_pred = self.f1_states["w_pred_per_class"].sum()
        w_gt = self.f1_states["w_gt_per_class"].sum()
        w_precision = w_tp / (w_pred + eps)
        w_recall = w_tp / (w_gt + eps)
        w_f1 = 2 * w_precision * w_recall / (w_precision + w_recall + eps)
        results["wF1"] = w_f1.item()
        
        # calculate each class F1 and wF1
        for i in range(self.num_classes):
            precision_i = self.f1_states["tp_per_class"][i] / (self.f1_states["pred_per_class"][i] + eps)
            recall_i = self.f1_states["tp_per_class"][i] / (self.f1_states["gt_per_class"][i] + eps)
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + eps)
            results[f"class_{i + 1}_F1"] = f1_i.item()
            w_precision_i = self.f1_states["w_tp_per_class"][i] / (self.f1_states["w_pred_per_class"][i] + eps)
            w_recall_i = self.f1_states["w_tp_per_class"][i] / (self.f1_states["w_gt_per_class"][i] + eps)
            w_f1_i = 2 * w_precision_i * w_recall_i / (w_precision_i + w_recall_i + eps)
            results[f"class_{i + 1}_wF1"] = w_f1_i.item()
        precision_bg = self.f1_states["tp_per_class"][-1] / (self.f1_states["pred_per_class"][-1] + eps)
        recall_bg = self.f1_states["tp_per_class"][-1] / (self.f1_states["gt_per_class"][-1] + eps)
        f1_bg = 2 * precision_bg * recall_bg / (precision_bg + recall_bg + eps)
        results["class_bg_F1"] = f1_bg.item()
        w_precision_bg = self.f1_states["w_tp_per_class"][-1] / (self.f1_states["w_pred_per_class"][-1] + eps)
        w_recall_bg = self.f1_states["w_tp_per_class"][-1] / (self.f1_states["w_gt_per_class"][-1] + eps)
        w_f1_bg = 2 * w_precision_bg * w_recall_bg / (w_precision_bg + w_recall_bg + eps)
        results["class_bg_wF1"] = w_f1_bg.item()
        
        self.f1_states.clear()
        return results

    def _update_dict_sublosses(self, dict_sublosses: Dict[str, float]) -> None:
        for key, value in dict_sublosses.items():
            if key not in self.dict_sublosses:
                self.dict_sublosses[key] = {
                    "count": len(value),
                    "sum": value.sum().item()
                }
            else:
                self.dict_sublosses[key]["count"] += len(value)
                self.dict_sublosses[key]["sum"] += value.sum().item()

    def _get_dict_sublosses(self) -> Dict[str, float]:
        dict_sublosses = {}
        for key, value in self.dict_sublosses.items():
            dict_sublosses[key] = value["sum"] / value["count"]
        self.dict_sublosses.clear()
        return dict_sublosses

    def _update_metric_states(self, metric_states: Dict[str, torch.Tensor]) -> None:
        for key, value in metric_states.items():
            if key not in self.metric_states:
                self.metric_states[key] = \
                    torch.tensor(value).reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs)).sum(dim=0)
            else:
                self.metric_states[key] += \
                    torch.tensor(value).reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs)).sum(dim=0)

    def _compute_panoptic_quality(self):
        """
        Compute panoptic quality metrics: PQ, SQ, RQ

        Args:

            `metric_states` (`Dict[str, torch.Tensor]`): Dictionary containing intermediate states for metric calculation

                \\- `tp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of true positive instances for each class

                \\- `fp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false positive instances for each class

                \\- `fn_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false negative instances for each class

                \\- `tp_iou_score_per_class` (`torch.Tensor`, shape is (num_classes,)): Summary of True positive IoU score for each class

            `thing_class_idxs` (`List[int]`): List of thing class indices

            `stuff_class_idxs` (`List[int]`): List of stuff class indices

        Returns:

            `Dict[str, float]`: Dictionary containing panoptic quality metrics
                `PQ` (`float`): Panoptic quality
                `SQ` (`float`): Semantic quality
                `RQ` (`float`): Instance quality
                `thing_PQ` (`float`): Thing panoptic quality
                `thing_SQ` (`float`): Thing semantic quality
                `thing_RQ` (`float`): Thing instance quality
                `stuff_PQ` (`float`): Stuff panoptic quality
                `stuff_SQ` (`float`): Stuff semantic quality
                `stuff_RQ` (`float`): Stuff instance quality
                `class_{id}_PQ` (`float`): Panoptic quality for class `id`
        """
        metric_states = self.metric_states
        thing_class_idxs = self.thing_class_idxs
        stuff_class_idxs = self.stuff_class_idxs
        tp_per_class = metric_states["tp_per_class"].to(torch.float32)
        fp_per_class = metric_states["fp_per_class"].to(torch.float32)
        fn_per_class = metric_states["fn_per_class"].to(torch.float32)
        tp_iou_score_per_class = metric_states["tp_iou_score_per_class"].to(torch.float32)
        eps = torch.finfo(torch.float32).eps

        def cal_scores(tp, fp, fn, tp_iou_score):
            rq = tp / (tp + 0.5 * fn + 0.5 * fp + eps)
            sq = tp_iou_score / (tp + eps)
            pq = rq * sq
            return pq, sq, rq

        pq_per_class, sq_per_class, rq_per_class = cal_scores(
            tp_per_class, fp_per_class, fn_per_class, tp_iou_score_per_class)

        class_metrics = {f"class_{id+1}_PQ": pq.item()*100 for id, pq in enumerate(pq_per_class)}

        thing_pq, thing_sq, thing_rq = cal_scores(
            tp_per_class[thing_class_idxs].sum(),
            fp_per_class[thing_class_idxs].sum(),
            fn_per_class[thing_class_idxs].sum(),
            tp_iou_score_per_class[thing_class_idxs].sum())

        stuff_pq, stuff_sq, stuff_rq = cal_scores(
            tp_per_class[stuff_class_idxs].sum(),
            fp_per_class[stuff_class_idxs].sum(),
            fn_per_class[stuff_class_idxs].sum(),
            tp_iou_score_per_class[stuff_class_idxs].sum())

        pq, sq, rq = cal_scores(tp_per_class.sum(), fp_per_class.sum(),
                                fn_per_class.sum(),
                                tp_iou_score_per_class.sum())

        self.metric_states.clear()

        metrics = {
            "PQ": pq.item()*100,
            "SQ": sq.item()*100,
            "RQ": rq.item()*100,
            "thing_PQ": thing_pq.item()*100,
            "thing_SQ": thing_sq.item()*100,
            "thing_RQ": thing_rq.item()*100,
            "stuff_PQ": stuff_pq.item()*100,
            "stuff_SQ": stuff_sq.item()*100,
            "stuff_RQ": stuff_rq.item()*100,
        }
        metrics.update(class_metrics)

        return metrics
