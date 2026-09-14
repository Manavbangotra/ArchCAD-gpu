"""
VecFormer model
"""
from dataclasses import dataclass

import torch
from torch_scatter import scatter

from transformers import PreTrainedModel
from transformers.utils import ModelOutput, logging
from typing import Optional, Dict, List, Tuple

from .configuration_vecformer import VecFormerConfig
from .point_transformer_v3 import PointTransformerV3
from .cad_decoder import CADDecoder
from .criterion import Criterion
from .modules import FusionLayerFeatsModule
from .evaluator import Evaluator, EvaluatorConfig
logger = logging.get_logger("transformers")


@dataclass
class VecFormerOutput(ModelOutput):
    """
    Class defining the output of Vecformer model

    Attributes:
        `list_pred_sem_segs` and `dict_pred_inst_segs` will only be returned when `is_inference_mode` is `True`

        `loss` (`torch.Tensor`): Total loss of the model

        `dict_sublosses` (`Dict[str, torch.Tensor]`): Dictionary containing sub-losses from multi-task learning

        `metric_states` (`Dict[str, torch.Tensor]`): Dictionary containing intermediate states for metric calculation

            \\- `tp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of true positive instances for each class

            \\- `fp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false positive instances for each class

            \\- `fn_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false negative instances for each class

            \\- `tp_iou_score_per_class` (`torch.Tensor`, shape is (num_classes,)): Summary of True positive IoU score for each class

        `f1_states` (`Dict[str, torch.Tensor]`): Dictionary of tp, fp, fn, w_tp, w_fp, w_fn for each class and total

        `dict_pred_sem_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of semantic segmentations predictions, contains

            \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_primitives,)): Predicted labels

            \\- `list_pred_scores` (`List[torch.Tensor]`, each tensor shape is (num_primitives,)): Predicted scores

            see `VecFormerOutput` for more details

        `dict_pred_inst_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of instance segmentations predictions, contains

            \\- `list_pred_masks` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances, num_primitives)): Predicted masks,
                each tensor is output of a batch, and indicates the predicted mask for each instance, e.g.
                ```
                pred_masks[i][j] = True
                ```
                means the j-th primitive is predicted to be the i-th instance

            \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted labels, only contains instance classes
                each tensor is output of a batch, and indicates the predicted label for each instance, e.g.
                ```
                pred_labels[i] = 1
                ```
                means the i-th instance is predicted to be the label `1` or the `1-th` class

            \\- `list_pred_scores` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted scores,
                each tensor is output of a batch, and indicates the predicted score for each instance, e.g.
                ```
                pred_scores[i] = 0.9
                ```
                means the i-th instance is predicted to have a score of `0.9`

        `dict_pred_panop_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of panoptic segmentations predictions,
            similar to `dict_pred_inst_segs`, but without scores, only contains masks and labels, and labels will contain all classes

            \\- `list_pred_masks` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances, num_primitives)): Predicted masks

            \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted labels

        `data_paths` (`List[str]`): List of data paths, each path is the path of the data in the batch
    """
    loss: Optional[torch.Tensor] = None
    dict_sublosses: Optional[Dict] = None
    metric_states: Optional[Dict[str, torch.Tensor]] = None
    f1_states: Optional[Dict[str, torch.Tensor]] = None
    dict_pred_sem_segs: Optional[Dict[str, List[torch.Tensor]]] = None
    dict_pred_inst_segs: Optional[Dict[str, List[torch.Tensor]]] = None
    dict_pred_panop_segs: Optional[Dict[str, List[torch.Tensor]]] = None
    data_paths: Optional[List[str]] = None

class VecFormer(PreTrainedModel):
    config_class = VecFormerConfig
    base_model_prefix = "vecformer"
    supports_gradient_checkpointing = True

    def __init__(self, config: VecFormerConfig):
        super().__init__(config)
        self.config = config
        self.is_inference_mode = False
        self.num_instance_classes = config.num_instance_classes
        self.num_semantic_classes = config.num_semantic_classes
        self.thing_class_idxs = config.thing_class_idxs
        self.stuff_class_idxs = config.stuff_class_idxs

        self.backbone = PointTransformerV3(**config.backbone_config)

        if config.use_layer_fusion:
            self.lfe = FusionLayerFeatsModule(
                self.config.cad_decoder_config["input_dim"],
                self.config.cad_decoder_config["embed_dim"])
        else:
            self.lfe = lambda x, y, z: x

        self.cad_decoder = CADDecoder(**config.cad_decoder_config)

        self.criterion = Criterion(
            instance_criterion_config=config.instance_criterion_config,
            semantic_criterion_config=config.semantic_criterion_config,
        )

        self.evaluator = Evaluator(EvaluatorConfig(**config.evaluator_config))

    def set_inference_mode(self, is_inference_mode: bool=True):
        self.is_inference_mode = is_inference_mode

    @torch.no_grad()
    def prepare_targets(self, semantic_id, instance_id, primitive_length, cu_numprims):
        """
        Prepare the targets for the model

        Args:

            `semantic_id` (`torch.Tensor`, shape is (N1+N2+...,)): The semantic label of each primitive
                N1, N2, ... are the number of primitives in each sequence in the batch

            `instance_id` (`torch.Tensor`, shape is (N1+N2+...,)): The instance label of each primitive
                N1, N2, ... are the number of primitives in each sequence in the batch

            `primitive_length` (`torch.Tensor`, shape is (N1+N2+...,)): The length of each primitive
                N1, N2, ... are the number of primitives in each sequence in the batch

            `cu_numprims` (`torch.Tensor`, shape is (B+1,)): The cumulative number of primitives in each sequence in the batch,
                the first element is 0, the last element is the total number of primitives in the batch

        Returns:

            `targets` (`Dict`):

                \\- `list_target_inst_labels` (`List[torch.Tensor]`, each tensor shape is (M, )):
                    Ground truth label of each instance, M is the number of instances

                \\- `list_target_inst_masks` (`List[torch.Tensor]`, each tensor shape is (M, N)):
                    Ground truth mask of each instance, M is the number of instances, N is the number of primitives

                \\- `list_target_prim_lens` (`List[torch.Tensor]`, each tensor shape is (N, )):
                    Ground truth length of each primitive, N is the number of primitives

                \\- `list_target_sem_labels` (`List[torch.Tensor]`, each tensor shape is (N, )):
                    Ground truth semantic label of each primitive, N is the number of primitives

                \\- `list_target_panop_labels` (`List[torch.Tensor]`, each tensor shape is (M, )):
                    Ground truth label of each instance, M is the number of instances, including stuff classes

                \\- `list_target_panop_masks` (`List[torch.Tensor]`, each tensor shape is (M, N)):
                    Ground truth mask of each instance, M is the number of instances, including stuff classes, N is the number of primitives
        """
        if semantic_id is None or instance_id is None:
            return None

        targets_device = cu_numprims.device

        list_target_inst_labels = []
        list_target_inst_masks = []
        list_target_prim_lens = []
        list_target_sem_labels = []
        list_target_panop_labels = []
        list_target_panop_masks = []
        for batch_idx in range(len(cu_numprims) - 1):
            idx_start, idx_end = cu_numprims[batch_idx], cu_numprims[batch_idx + 1]
            batch_semantic_id = semantic_id[idx_start:idx_end]
            batch_instance_id = instance_id[idx_start:idx_end]
            batch_prim_lens = primitive_length[idx_start:idx_end]
            n_primitives = idx_end - idx_start

            # ------- get instance masks and labels ------ #
            unique_sem_inst_pairs = torch.unique(
                torch.stack((batch_semantic_id, batch_instance_id), dim=0),
                dim=1
            ).T     # shape: (num_unique_pairs, 2)
            valid_mask = ~(
                (unique_sem_inst_pairs[:, 0] == self.num_semantic_classes)
                & (unique_sem_inst_pairs[:, 1] == -1)
            )  # remove background
            unique_sem_inst_pairs = unique_sem_inst_pairs[valid_mask]

            target_inst_labels = []
            target_inst_masks = []
            target_panop_labels = []
            target_panop_masks = []
            for (cur_pair_sem_id, cur_pair_inst_id) in unique_sem_inst_pairs:
                cur_pair_sem_mask = batch_semantic_id == cur_pair_sem_id
                cur_pair_inst_mask = batch_instance_id == cur_pair_inst_id
                # get the intersection of semantic and instance masks
                cur_pair_inter_mask = (cur_pair_sem_mask & cur_pair_inst_mask).bool()

                target_inst_labels.append(cur_pair_sem_id)
                target_inst_masks.append(cur_pair_inter_mask)
                # append both instance and semantic to panoptic
                target_panop_labels.append(cur_pair_sem_id)
                target_panop_masks.append(cur_pair_inter_mask)


            # post process
            # instance
            if target_inst_labels and target_inst_masks:
                target_inst_labels = torch.tensor(target_inst_labels, dtype=torch.int32, device=targets_device)
                target_inst_masks = torch.cat(target_inst_masks).view(-1, n_primitives)
            else:
                target_inst_labels = torch.tensor([self.num_instance_classes], dtype=torch.int32, device=targets_device)
                target_inst_masks = torch.zeros(1, n_primitives, dtype=torch.bool, device=targets_device)
            # panoptic
            if target_panop_labels and target_panop_masks:
                target_panop_labels = torch.tensor(target_panop_labels, dtype=torch.int32, device=targets_device)
                target_panop_masks = torch.cat(target_panop_masks).view(-1, n_primitives)
            else:
                target_panop_labels = torch.tensor([self.num_semantic_classes], dtype=torch.int32, device=targets_device)
                target_panop_masks = torch.zeros(1, n_primitives, dtype=torch.bool, device=targets_device)

            # ------------ get semantic labels ------------ #
            target_sem_labels = batch_semantic_id

            # --------------- get prim len --------------- #
            target_prim_lens = batch_prim_lens
            # -------------------------------------------- #
            list_target_inst_labels.append(target_inst_labels)
            list_target_inst_masks.append(target_inst_masks)
            list_target_prim_lens.append(target_prim_lens)
            list_target_sem_labels.append(target_sem_labels)
            list_target_panop_labels.append(target_panop_labels)
            list_target_panop_masks.append(target_panop_masks)

        return dict(list_target_inst_labels=list_target_inst_labels,
                    list_target_inst_masks=list_target_inst_masks,
                    list_target_prim_lens=list_target_prim_lens,
                    list_target_sem_labels=list_target_sem_labels,
                    list_target_panop_labels=list_target_panop_labels,
                    list_target_panop_masks=list_target_panop_masks)

    @torch.no_grad()
    def predict(self, list_pred_sem_labels, list_pred_inst_masks, list_pred_inst_labels, list_pred_inst_scores, prim_lengths):
        """
        Predict the semantic, instance and panoptic segmentations

        Args:

            `list_pred_sem_labels` (`List[torch.Tensor]`, each tensor shape is (N, num_semantic_classes + 1)):
                Predicted label logits of each primitive, N is the number of primitives

            `list_pred_inst_labels` (`List[torch.Tensor]`, each tensor shape is (Q, num_instance_classes + 1)):
                Predicted label logits of each query, Q is the number of queries

            `list_pred_inst_scores` (`Optional[List[torch.Tensor]]`, each tensor shape is (Q, 1)):
                Predicted confidence score of each query, Q is the number of queries

            `list_pred_inst_masks` (`List[torch.Tensor]`, each tensor shape is (Q, N)):
                Predicted mask of each query, Q is the number of queries, N is the number of primitives

            `prim_lengths` (`List[torch.Tensor]`, each tensor shape is (N, )):
                Length of each primitive, N is the number of primitives

        Returns:

            Tuple of three elements:

            `dict_pred_sem_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of semantic segmentations predictions, contains

                \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_primitives,)): Predicted labels

                \\- `list_pred_scores` (`List[torch.Tensor]`, each tensor shape is (num_primitives,)): Predicted scores

                see `VecFormerOutput` for more details

            `dict_pred_inst_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of instance segmentations predictions, contains

                \\- `list_pred_masks` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances, num_primitives)): Predicted masks

                \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted labels

                \\- `list_pred_scores` (`Optional[List[torch.Tensor]]`, each tensor shape is (num_predicted_instances,)): Predicted scores

            `dict_pred_panop_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of panoptic segmentations predictions, contains

                \\- `list_pred_masks` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances, num_primitives)): Predicted masks

                \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted labels

                see `VecFormerOutput` for more details
        """
        dict_pred_sem_segs = self.predict_semantic(pred_labels=list_pred_sem_labels)
        dict_pred_inst_segs = self.predict_instance(
            pred_masks=list_pred_inst_masks,
            pred_labels=list_pred_inst_labels,
            pred_scores=list_pred_inst_scores)
        dict_pred_panop_segs = self.predict_panoptic(
            sem_segs=dict_pred_sem_segs,
            inst_segs=dict_pred_inst_segs,
            prim_lengths=prim_lengths)
        return dict_pred_sem_segs, dict_pred_inst_segs, dict_pred_panop_segs

    @torch.no_grad()
    def predict_semantic(self, pred_labels):
        """
        Predict the semantic segmentations

        Args:

            `pred_labels` (`List[Tensor]`, each tensor shape is (num_primitives, num_semantic_classes + 1)): Predicted label logits of each primitive

        Returns:

            `dict_pred_sem_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of semantic segmentations predictions, contains

                \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_primitives,)): Predicted labels

                \\- `list_pred_scores` (`List[torch.Tensor]`, each tensor shape is (num_primitives,)): Predicted scores

                see `VecFormerOutput` for more details
        """
        list_pred_sem_scores = []
        list_pred_sem_labels = []
        for pred_label in pred_labels:
            pred_score, pred_label = pred_label.softmax(-1).max(-1)
            list_pred_sem_scores.append(pred_score)
            list_pred_sem_labels.append(pred_label)
        return dict(list_pred_labels=list_pred_sem_labels,
                    list_pred_scores=list_pred_sem_scores)

    @torch.no_grad()
    def predict_instance(self, pred_masks, pred_labels, pred_scores):
        """
        Predict the instance segmentations

        Args:

            `pred_masks` (`List[Tensor]`, each tensor shape is (num_queries, num_primitives)): Predicted mask of each query

            `pred_labels` (`List[Tensor]`, each tensor shape is (num_queries, num_instance_classes + 1)): Predicted label logits of each query

            `pred_scores` (`List[Tensor]`, each tensor shape is (num_queries, 1)): Predicted confidence score of each query

        Returns:
            `dict_pred_inst_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of instance segmentations predictions, contains

                \\- `list_pred_masks` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances, num_primitives)): Predicted masks

                \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted labels

                \\- `list_pred_scores` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted scores

                see `VecFormerOutput` for more details
        """
        list_pred_masks, list_pred_labels, list_pred_scores = [], [], []
        for i in range(len(pred_masks)):
            pred_mask = pred_masks[i]
            pred_label = pred_labels[i]
            # get label probabilities
            pred_score, pred_label = pred_label.softmax(dim=-1).max(dim=-1)
            # get predicted scores: we use label probabilities to enhance the predicted scores
            if pred_scores is not None:
                pred_score *= pred_scores[i]
            # get topk preds
            num_topk_preds = min(self.config.num_topk_preds, pred_score.shape[0])
            pred_score, topk_idx = pred_score.topk(num_topk_preds, sorted=False)
            # get predicted labels
            pred_label = pred_label[topk_idx]
            # get predicted masks
            pred_mask = pred_mask[topk_idx]
            pred_mask_sigmoid = pred_mask.sigmoid()
            # object normalization
            if self.config.use_obj_normalization:
                pred_mask_thr = pred_mask_sigmoid > \
                    self.config.obj_normalization_thr
                mask_scores = (pred_mask_sigmoid * pred_mask_thr).sum(1) / \
                    (pred_mask_thr.sum(1) + 1e-6)
                pred_score = pred_score * mask_scores
            # get 0/1 mask
            pred_mask = pred_mask_sigmoid > self.config.mask_logit_thr
            # filter out low-confidence preds
            probs_filter_mask = pred_score > self.config.pred_score_thr
            pred_mask = pred_mask[probs_filter_mask]
            pred_label = pred_label[probs_filter_mask]
            pred_score = pred_score[probs_filter_mask]
            # filter out too empty preds
            pred_n_primitives = pred_mask.sum(1)
            nprimitive_mask = pred_n_primitives > self.config.n_primitives_thr
            pred_mask = pred_mask[nprimitive_mask]
            pred_label = pred_label[nprimitive_mask]
            pred_score = pred_score[nprimitive_mask]
            # -------------------------------------------- #
            list_pred_masks.append(pred_mask)
            list_pred_labels.append(pred_label)
            list_pred_scores.append(pred_score)

        dict_pred_inst_segs = dict(list_pred_masks=list_pred_masks,
                                   list_pred_labels=list_pred_labels,
                                   list_pred_scores=list_pred_scores)

        return dict_pred_inst_segs

    def predict_panoptic(self, sem_segs: Dict[str, List[torch.Tensor]],
                         inst_segs: Dict[str, List[torch.Tensor]],
                         prim_lengths: List[torch.Tensor]):
        """
        Predict the panoptic segmentations

        Args:

            `sem_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of semantic segmentations predictions

            `inst_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of instance segmentations predictions

            `prim_lengths` (`List[torch.Tensor]`, each tensor shape is (N, )):
                Length of each primitive, N is the number of primitives

            see `Returns` of `predict_semantic` and `predict_instance` for more details

        Returns:

            `dict_pred_panop_segs` (`Dict[str, List[torch.Tensor]]`): Dictionary of panoptic segmentations predictions, contains

                \\- `list_pred_masks` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances, num_primitives)): Predicted masks

                \\- `list_pred_labels` (`List[torch.Tensor]`, each tensor shape is (num_predicted_instances,)): Predicted labels

                see `VecFormerOutput` for more details
        """
        list_pred_masks = []
        list_pred_labels = []
        for sem_labels, sem_scores, inst_masks, inst_labels, inst_scores, prim_length in zip(
                sem_segs['list_pred_labels'], sem_segs['list_pred_scores'],
                inst_segs["list_pred_masks"], inst_segs["list_pred_labels"],
                inst_segs["list_pred_scores"], prim_lengths):
            stuff_masks, stuff_labels = self.convert_sem_labels_to_panop_stuff_segs(sem_labels)
            # vote for panoptic labels
            inst_masks, inst_labels, inst_scores = self.voting(inst_masks, inst_labels, inst_scores,
                                      sem_labels, sem_scores, prim_length)
            # remove wrong mask by instance labels
            inst_masks = self.remasking(inst_masks, inst_labels, sem_labels)
            # get panoptic labels
            panop_labels = torch.cat([inst_labels, stuff_labels])
            panop_masks = torch.cat([inst_masks, stuff_masks])
            list_pred_masks.append(panop_masks)
            list_pred_labels.append(panop_labels)
        return dict(list_pred_masks=list_pred_masks,
                    list_pred_labels=list_pred_labels)

    @torch.no_grad()
    def convert_sem_labels_to_panop_stuff_segs(self, sem_labels):
        """
        Convert semantic labels to panoptic stuff segments

        Args:

            `sem_labels` (`torch.Tensor`, shape is (num_primitives,)): Semantic labels

        Returns:

            `stuff_masks` (`torch.Tensor`, shape is (num_preds, num_primitives)): Stuff masks

            `stuff_labels` (`torch.Tensor`, shape is (num_preds,)): Stuff labels
        """
        stuff_labels = torch.tensor(self.stuff_class_idxs, device=sem_labels.device)
        stuff_masks = torch.zeros(self.num_semantic_classes + 1,
                                  sem_labels.shape[0],
                                  device=sem_labels.device,
                                  dtype=torch.bool)
        stuff_masks[sem_labels, torch.arange(sem_labels.shape[0])] = True
        stuff_masks = stuff_masks[stuff_labels]
        return stuff_masks, stuff_labels

    @torch.no_grad()
    def voting(self, inst_masks: torch.Tensor, inst_labels: torch.Tensor,
               inst_scores: torch.Tensor, sem_labels: torch.Tensor,
               sem_scores: torch.Tensor,
               prim_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # update semantic labels with instance labels
        if inst_scores.numel() > 0:
            prim_scores = inst_scores.unsqueeze(-1) * inst_masks # (num_queries, num_primitives)
            prim_scores, prim_idxs = prim_scores.max(0) # (num_primitives,)
            prim_labels = inst_labels[prim_idxs]
            update_mask = prim_scores > sem_scores
            sem_labels = torch.where(update_mask, prim_labels, sem_labels)
            sem_scores = torch.where(update_mask, prim_scores, sem_scores)
        n_preds, n_prims = inst_masks.shape
        n_classes = self.num_semantic_classes + 1
        sem_one_hot = torch.zeros((n_prims, n_classes),
                                  dtype=torch.int32,
                                  device=inst_masks.device)
        sem_one_hot[torch.arange(n_prims), sem_labels] = 1
        votes = inst_masks.float() @ sem_one_hot.float()
        inst_labels = votes.argmax(dim=-1)
        not_empty_masks = inst_masks.sum(dim=-1) != 0
        inst_masks  = inst_masks[not_empty_masks]
        inst_labels = inst_labels[not_empty_masks]
        inst_scores = inst_scores[not_empty_masks]
        return inst_masks, inst_labels, inst_scores

    @torch.no_grad()
    def remasking(self, inst_masks: torch.Tensor, inst_labels: torch.Tensor,
                  sem_labels: torch.Tensor) -> torch.Tensor:
        sem_labels_expanded = sem_labels.unsqueeze(0).expand(inst_masks.shape[0], -1)
        inst_labels_expanded = inst_labels.unsqueeze(1).expand(-1, inst_masks.shape[1])
        mismatch_mask = sem_labels_expanded != inst_labels_expanded
        inst_masks = inst_masks & ~(inst_masks & mismatch_mask)
        return inst_masks

    @torch.no_grad()
    def _init_queries(self, feats, cu_seqlens, targets, query_thr: float = 0.5):
        """Init queries for decoder.

        Args:

            `feats` (`torch.Tensor`, shape is (N1+N2+..., feats_embed_dim)): Features of primitives
                N1, N2, ... are the number of primitives in each batch

            `cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of primitives
                The first element is 0, and the last element is the total number of primitives in all batches

            `targets` (`Dict`):

                \\- `list_target_inst_masks` (`List[torch.Tensor]`, each tensor shape is (M, N)):
                    Ground truth mask of each instance, M is the number of instances, N is the number of primitives

        Returns:

            Tuple of three elements:

                `queries` (`torch.Tensor`, shape is (Q1+Q2+..., embed_dim)): Queries of all batches
                    Q1, Q2, ... are the number of queries in each batch

                `query_cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of queries
                    The first element is 0, and the last element is the total number of queries in all batches

                `targets` (`Dict`): Include raw inputs, and updated with `query_masks`

                    \\- `list_target_selected_idxs` (`List[torch.Tensor]`, each tensor shape is (Q,)):
                        indicates which primitives are selected as queries, Q is the number of queries
        """
        queries = []
        list_target_selected_idxs = []
        for batch_idx in range(len(cu_seqlens) - 1):
            idx_start, idx_end = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
            batch_feats_len = len(feats[idx_start:idx_end])
            if query_thr < 1:
                # select n feats to init as queries
                n_queries = (1 - query_thr) * torch.rand(1) + query_thr
                n_queries = (n_queries * batch_feats_len).int().item()
                if self.config.max_num_queries > 0:
                    n_queries = min(n_queries, self.config.max_num_queries)
                selected_idxs = torch.randperm(batch_feats_len)[:n_queries].to(feats.device)
                queries.append(feats[idx_start:idx_end][selected_idxs])
                list_target_selected_idxs.append(selected_idxs)
            else:
                queries.append(feats[idx_start:idx_end])
                list_target_selected_idxs.append(torch.arange(batch_feats_len))

        query_seq_lens = torch.tensor(
            [len(query) for query in queries],
            dtype=torch.int32,
            device=cu_seqlens.device)
        query_cu_seqlens = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=cu_seqlens.device),
            torch.cumsum(query_seq_lens, dim=0, dtype=torch.int32)
        ])

        targets["list_target_selected_idxs"] = list_target_selected_idxs

        return torch.cat(queries, dim=0), query_cu_seqlens, targets

    @torch.no_grad()
    def _get_data_dict(self, coords, feats, cu_seqlens, grid_size=0.01, prim_ids=None, layer_ids=None, sample_mode="line"):
        if sample_mode == 'point':
            z_coords = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
            coords = torch.cat([coords, z_coords], dim=-1)
        elif sample_mode == 'line':
            layer_ids_feats = torch.zeros((coords.shape[0], 1), dtype=feats.dtype, device=feats.device)
            for batch_idx in range(len(cu_seqlens) - 1):
                eps = torch.finfo(torch.float32).eps
                idx_start, idx_end = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
                layer_ids_batch = layer_ids[idx_start:idx_end] # type: ignore
                layer_id_min = layer_ids_batch.min()
                layer_id_max = layer_ids_batch.max()
                layer_ids_feats[idx_start:idx_end, 0] = (layer_ids_batch - layer_id_min) / (layer_id_max - layer_id_min + eps) - 0.5
            coords = torch.cat([coords, layer_ids_feats], dim=-1)
        else:
            raise ValueError(f"Invalid sample mode: {sample_mode}")
        return dict(feat=feats,
                    coord=coords,
                    grid_size=grid_size,
                    offset=cu_seqlens[1:])

    @torch.no_grad()
    def prepare_primitive_layerid(self, prim_ids, layer_ids, cu_seqlens):
        """
        Return the correspondence of primitive and layer id, used for layer feature fusion after backbone feature aggregation
        """
        fusion_layerids = []
        for batch_idx in range(len(cu_seqlens) - 1):
            idx_start, idx_end = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
            batch_prim_ids = prim_ids[idx_start:idx_end]
            batch_layerids = layer_ids[idx_start:idx_end]
            # Get the layer id corresponding to each primitive
            fusion_layerids.append(scatter(batch_layerids, batch_prim_ids.long(), dim=0, reduce="max"))
        return torch.cat(fusion_layerids, dim=0)

    def forward(self,
                coords,
                feats,
                prim_ids,
                layer_ids,
                cu_seqlens,
                sem_ids=None,
                inst_ids=None,
                prim_lengths=None,
                cu_numprims=None,
                data_paths=None):
        # prepare targets
        targets = None
        if sem_ids is not None and inst_ids is not None and prim_lengths is not None and cu_numprims is not None:
            targets = self.prepare_targets(sem_ids, inst_ids, prim_lengths, cu_numprims)
        # vecformer backbone forward
        data_dict = self._get_data_dict(coords, feats, cu_seqlens, grid_size=0.01, prim_ids=prim_ids, layer_ids=layer_ids, sample_mode=self.config.sample_mode)
        fusion_layer_ids = self.prepare_primitive_layerid(prim_ids, layer_ids, cu_seqlens)
        feats, cu_seqlens = self.backbone(data_dict, cu_seqlens, prim_ids)
        feats = self.lfe(feats, cu_seqlens, fusion_layer_ids)
        # init queries
        if self.training:
            queries, query_cu_seqlens, targets = self._init_queries(feats, cu_seqlens, targets, query_thr=self.config.query_thr)
        else:
            # when evaluating, queries=feats, query_cu_seqlens=cu_seqlens
            queries, query_cu_seqlens, targets = self._init_queries(feats, cu_seqlens, targets, query_thr=1.0)
        # cad decoder forward
        outputs = self.cad_decoder(feats, cu_seqlens, queries, query_cu_seqlens)

        # ------------- get vecformer output ------------- #
        loss, dict_sublosses, metric_states, f1_states = None, None, None, None
        # ---------------- calculate loss ---------------- #
        if targets is not None and self.training:
            # calculate panoptic symbol spotting loss
            loss, dict_sublosses = self.criterion(outputs, targets)
        # -------------- get vecformer preds ------------- #
        if targets is not None and not self.training:
            # get the last layer's outputs
            last_outputs = outputs[-1]
            dict_pred_sem_segs, dict_pred_inst_segs, dict_pred_panop_segs = self.predict(
                last_outputs["list_pred_sem_labels"],
                last_outputs["list_pred_inst_masks"],
                last_outputs["list_pred_inst_labels"],
                last_outputs["list_pred_inst_scores"],
                targets["list_target_prim_lens"]
            )
            preds = dict(
                pred_masks=dict_pred_panop_segs["list_pred_masks"],
                pred_labels=dict_pred_panop_segs["list_pred_labels"],
                pred_sem_segs=dict_pred_sem_segs["list_pred_labels"])
            targets = dict(
                target_masks=targets["list_target_panop_masks"],
                target_labels=targets["list_target_panop_labels"],
                prim_lens=targets["list_target_prim_lens"],
                sem_labels=targets["list_target_sem_labels"])
            metric_states, f1_states = self.evaluator(preds, targets)
            if self.config.whether_output_instance:
                self.evaluator.eval_instance_quality(preds, data_paths)
        return VecFormerOutput(loss=loss if loss is not None else torch.tensor(0.0, device=feats.device),
                               dict_sublosses=dict_sublosses if dict_sublosses is not None else {},
                               metric_states=metric_states,
                               f1_states=f1_states,
                               dict_pred_sem_segs=dict_pred_sem_segs # type: ignore
                               if self.is_inference_mode else None,
                               dict_pred_inst_segs=dict_pred_inst_segs # type: ignore
                               if self.is_inference_mode else None,
                               dict_pred_panop_segs=dict_pred_panop_segs # type: ignore
                               if self.is_inference_mode else None,
                               data_paths=data_paths
                               if self.is_inference_mode else None)
