from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class SparseMatcher:
    """
    Sparse matcher

    Args:

        `topk` (`int`): Limit topk matches per query

        `class_loss_weight` (`float`): Weight for query classification loss

        `bce_loss_weight` (`float`): Weight for BCE loss

        `dice_loss_weight` (`float`): Weight for dice loss
    """
    def __init__(self,
            topk: int,
            class_cost_weight: float = 0.5,
            bce_cost_weight: float = 1.0,
            dice_cost_weight: float = 1.0,
        ):
        self.topk = topk
        self.inf = 1e8
        self.class_cost_weight = class_cost_weight
        self.bce_cost_weight = bce_cost_weight
        self.dice_cost_weight = dice_cost_weight

    @torch.no_grad()
    def __call__(self,
            list_query_labels, list_query_masks,
            list_target_labels, list_target_masks,
            list_target_selected_idxs
        ):
        """
        Get match indices

        Args:

            `list_query_labels` (`List[torch.Tensor]`, each tensor shape is (n_queries, n_instance_classes + 1)): Predicted labels of each batch

            `list_query_masks` (`List[torch.Tensor]`, each tensor shape is (n_queries, n_primitives)): Predicted masks of each batch

            `list_target_labels` (`List[torch.Tensor]`, each tensor shape is (n_targets,)): Target labels of each batch

            `list_target_masks` (`List[torch.Tensor]`, each tensor shape is (n_targets, n_primitives)): Target masks of each batch

            `list_target_selected_idxs` (`List[torch.Tensor]`, each tensor shape is (n_queries,)): selected primitive idxs of each batch

        Returns:

            `list_match_indices` (`List[torch.Tensor]`, each tensor shape is (2, n_matched)): Match indices of each batch
                in each matched pair, the first element is the query index and the second element is the target index
        """
        list_match_indices = []
        for query_labels, query_masks, target_labels, target_masks, target_selected_idxs in zip(
            list_query_labels, list_query_masks, list_target_labels, list_target_masks, list_target_selected_idxs
        ):
            costs = self._get_costs(query_labels, query_masks, target_labels, target_masks) # (n_queries, n_targets)
            costs_cloned = costs.clone()
            # Get sparse match
            costs = torch.where(target_masks[:, target_selected_idxs].T, costs, self.inf)
            values = torch.topk(costs, self.topk + 1, dim=0, sorted=True, largest=False).values[-1:, :]
            ids = torch.argwhere(costs < values)
            if ids.numel() == 0:
                # if no match, use hungarian algorithm to find the best match
                query_ids, object_ids = linear_sum_assignment(costs_cloned.cpu().numpy())
                query_ids = torch.from_numpy(query_ids).to(query_labels.device)
                object_ids = torch.from_numpy(object_ids).to(query_labels.device)
                list_match_indices.append(torch.stack([query_ids, object_ids]))
            else:
                list_match_indices.append(ids.T)
        return list_match_indices

    @torch.no_grad()
    def _get_costs(self, query_labels, query_masks, target_labels, target_masks):
        class_cost = self._get_class_cost(query_labels, target_labels)
        bce_cost = self._get_bce_cost(query_masks, target_masks)
        dice_cost = self._get_dice_cost(query_masks, target_masks)
        # Normalize costs
        eps = torch.finfo(torch.float32).eps
        class_cost = class_cost / (class_cost.detach().mean() + eps)
        bce_cost = bce_cost / (bce_cost.detach().mean() + eps)
        dice_cost = dice_cost / (dice_cost.detach().mean() + eps)
        # Weight costs
        class_cost *= self.class_cost_weight
        bce_cost *= self.bce_cost_weight
        dice_cost *= self.dice_cost_weight
        return class_cost + bce_cost + dice_cost

    @torch.no_grad()
    def _get_class_cost(self, query_labels, target_labels):
        """
        Get class cost

        Args:

            `query_labels` (`torch.Tensor`): Predicted labels of shape (n_queries, n_classes + 1)

            `target_labels` (`torch.Tensor`): Target labels of shape (n_targets,)

        Returns:

            `class_cost` (`torch.Tensor`, shape is (n_queries, n_targets)): Class cost
        """
        class_probs = query_labels.softmax(-1)
        class_cost = 1 - class_probs[:, target_labels]
        return class_cost

    @torch.no_grad()
    def _get_bce_cost(self, query_masks, target_masks):
        """
        Get BCE cost

        Args:

            `query_masks` (`torch.Tensor`): Predicted masks of shape (n_queries, n_primitives)

            `target_masks` (`torch.Tensor`): Target masks of shape (n_targets, n_primitives)

        Returns:

            `bce_cost` (`torch.Tensor`, shape is (n_queries, n_targets)): BCE cost
        """
        pos = F.binary_cross_entropy_with_logits(
            query_masks, torch.ones_like(query_masks), reduction='none')
        pos_loss = torch.einsum('nc,mc->nm', pos, target_masks.float())

        neg = F.binary_cross_entropy_with_logits(
            query_masks, torch.zeros_like(query_masks), reduction='none')
        neg_loss = torch.einsum('nc,mc->nm', neg, (~target_masks).float())

        bce_cost = (pos_loss + neg_loss) / query_masks.shape[-1]
        return bce_cost

    @torch.no_grad()
    def _get_dice_cost(self, query_masks, target_masks):
        """
        Get dice cost

        Args:

            `query_masks` (`torch.Tensor`): Predicted masks of shape (n_queries, n_primitives)

            `target_masks` (`torch.Tensor`): Target masks of shape (n_targets, n_primitives)

        Returns:

            `dice_cost` (`torch.Tensor`, shape is (n_queries, n_targets)): Dice cost
        """
        query_masks = query_masks.sigmoid()
        numerator = 2 * torch.einsum('nc,mc->nm', query_masks, target_masks.float())
        denominator = query_masks.sum(-1)[:, None] + target_masks.sum(-1)[None, :]
        dice_cost = 1 - (numerator + 1) / (denominator + 1)
        return dice_cost


class InstanceCriterion(nn.Module):
    """
    Instance criterion

    Args:

        `num_instance_classes` (`int`): Number of instance classes

        `class_loss_weight` (`float`): Weight for query classification loss

        `ce_non_object_weight` (`float`): Non-object weight in query classification, used in `F.cross_entropy` as arg `weight`

        `bce_loss_weight` (`float`): Weight for BCE loss

        `dice_loss_weight` (`float`): Weight for dice loss

        `score_loss_weight` (`float`): Weight for score loss

        `topk_matches` (`int`): Limit topk matches per query

        `iter_matcher` (`bool`): If set `True`,
            use different match indices for each layer,
            else use the last layer match indices for all layers

        `label_smoothing` (`float`): Label smoothing value, see `F.cross_entropy`

        `use_mean_batch_loss` (`bool`): If set `True`,
            use `mean` loss to reduce loss of each `batch`,
            else use `sum` loss to reduce loss of each `batch`
    """

    def __init__(self,
                 num_instance_classes: int,
                 class_loss_weight: float = 0.5,
                 ce_non_object_weight: float = 0.01,
                 bce_loss_weight: float = 1.0,
                 dice_loss_weight: float = 1.0,
                 score_loss_weight: float = 0.5,
                 topk_matches: int = 1,
                 iter_matcher: bool = True,
                 use_mean_batch_loss: bool = True,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.class_loss_weight = class_loss_weight
        self.ce_non_object_weight = ce_non_object_weight
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.score_loss_weight = score_loss_weight
        self.num_instance_classes = num_instance_classes
        self.iter_matcher = iter_matcher
        self.use_mean_batch_loss = use_mean_batch_loss
        self.label_smoothing = label_smoothing

        self.matcher = SparseMatcher(
            topk=topk_matches,
            class_cost_weight=class_loss_weight,
            bce_cost_weight=bce_loss_weight,
            dice_cost_weight=dice_loss_weight
        )

    def forward(self, preds, targets):
        """
        Calculate the instance loss.

        Args:

            `preds` (`List[Dict[str, List[torch.Tensor]]]`): each dict in the list contains the predicted labels, scores and masks of a decoder block, contains

                \\- `list_query_labels` (`List[torch.Tensor]`, each tensor shape is (n_queries, n_classes + 1)): Predicted labels of each batch

                \\- `list_query_scores` (`List[torch.Tensor]`, each tensor shape is (n_queries, 1)): Predicted scores of each batch

                \\- `list_query_masks` (`List[torch.Tensor]`, each tensor shape is (n_queries, n_primitives)): Predicted masks of each batch

            `targets` (`Dict[str, List[torch.Tensor]]`): the dict contains

                \\- `list_target_labels` (`List[torch.Tensor]`, each tensor shape is (n_targets,)): Target labels of each batch

                \\- `list_target_masks` (`List[torch.Tensor]`, each tensor shape is (n_targets, n_primitives)): Target masks of each batch

                \\- `list_target_selected_idxs` (`List[torch.Tensor]`, each tensor shape is (n_queries,)): selected primitive idxs of each batch

                \\- `list_target_prim_lens` (`List[torch.Tensor]`, each tensor shape is (n_targets,)): Target primitive lengths of each batch

        Returns:

            `inst_loss` (`Dict[str, torch.Tensor]`): A dictionary of instance loss
        """
        list_target_labels = targets['list_target_labels']
        list_target_masks = targets['list_target_masks']
        list_target_selected_idxs = targets['list_target_selected_idxs']
        list_target_prim_lens = targets['list_target_prim_lens']

        # get match
        with torch.no_grad():
            block_match_indices = []
            preds_for_match = preds if self.iter_matcher else preds[-1]
            for pred in preds_for_match:
                list_query_labels = pred['list_query_labels']
                list_query_masks = pred['list_query_masks']
                block_match_indices.append(
                    self.matcher(
                        list_query_labels, list_query_masks,
                        list_target_labels, list_target_masks,
                        list_target_selected_idxs
                    )
                )

        # get loss
        blocks_losses = []
        for block_id, pred in enumerate(preds):
            list_query_labels = pred['list_query_labels']
            list_query_scores = pred['list_query_scores']
            list_query_masks = pred['list_query_masks']
            list_match_indices = block_match_indices[block_id] \
                if self.iter_matcher else block_match_indices[-1]
            blocks_losses.append(self._get_loss(
                list_query_labels, list_query_scores,
                list_query_masks, list_target_labels,
                list_target_masks, list_target_prim_lens,
                list_match_indices
            ))

        # post process
        inst_loss = {}
        if len(blocks_losses) == 1:
            inst_loss["inst_class_loss"] = blocks_losses[0]["class_loss"]
            inst_loss["inst_bce_loss"] = blocks_losses[0]["bce_loss"]
            inst_loss["inst_dice_loss"] = blocks_losses[0]["dice_loss"]
            if blocks_losses[0]["score_loss"] > 0:
                inst_loss["inst_score_loss"] = blocks_losses[0]["score_loss"]
        else:
            for block_id, block_loss in enumerate(blocks_losses):
                inst_loss[f"inst_block_{block_id}_class_loss"] = block_loss["class_loss"]
                inst_loss[f"inst_block_{block_id}_bce_loss"] = block_loss["bce_loss"]
                inst_loss[f"inst_block_{block_id}_dice_loss"] = block_loss["dice_loss"]
                if block_loss["score_loss"] > 0:
                    inst_loss[f"inst_block_{block_id}_score_loss"] = block_loss["score_loss"]
            inst_loss["inst_class_loss"] = torch.stack([block_loss["class_loss"] for block_loss in blocks_losses]).sum()
            inst_loss["inst_bce_loss"] = torch.stack([block_loss["bce_loss"] for block_loss in blocks_losses]).sum()
            inst_loss["inst_dice_loss"] = torch.stack([block_loss["dice_loss"] for block_loss in blocks_losses]).sum()
            inst_score_loss = torch.stack([block_loss["score_loss"] for block_loss in blocks_losses]).sum()
            if inst_score_loss > 0:
                inst_loss["inst_score_loss"] = inst_score_loss
        inst_loss["instance_loss"] = inst_loss["inst_class_loss"] + \
                                     inst_loss["inst_bce_loss"] + \
                                     inst_loss["inst_dice_loss"] + \
                                     (inst_loss["inst_score_loss"]
                                      if "inst_score_loss" in inst_loss else 0)

        return inst_loss

    def _get_loss(self,
                  list_query_labels: List[torch.Tensor],
                  list_query_scores: List[torch.Tensor],
                  list_query_masks: List[torch.Tensor],
                  list_target_labels: List[torch.Tensor],
                  list_target_masks: List[torch.Tensor],
                  list_target_prim_lens: List[torch.Tensor],
                  list_match_indices: List[torch.Tensor]):
        class_losses = []
        bce_losses = []
        dice_losses = []
        score_losses = []
        for batch_idx in range(len(list_query_labels)):
            # get data from batch
            query_labels = list_query_labels[batch_idx]
            query_scores = list_query_scores[batch_idx] if list_query_scores is not None else None
            query_masks = list_query_masks[batch_idx]
            target_labels = list_target_labels[batch_idx]
            target_masks = list_target_masks[batch_idx]
            target_prim_lens = list_target_prim_lens[batch_idx]
            match_indices = list_match_indices[batch_idx]
            # get matched data
            query_ids, object_ids = match_indices[0], match_indices[1]
            query_labels_matched = query_labels[query_ids]
            query_scores_matched = query_scores[query_ids] if query_scores is not None else None
            query_masks_matched = query_masks[query_ids]
            target_labels_matched = target_labels[object_ids]
            target_masks_matched = target_masks[object_ids]
            # get loss
            class_loss = self._get_class_loss(query_labels_matched, target_labels_matched)
            bce_loss = self._get_bce_loss(query_masks_matched, target_masks_matched)
            dice_loss = self._get_dice_loss(query_masks_matched, target_masks_matched)
            if query_scores_matched is not None:
                score_loss = self._get_score_loss(query_scores_matched,
                                                  query_masks_matched,
                                                  target_masks_matched,
                                                  target_prim_lens)
            else:
                score_loss = None
            # append loss
            class_losses.append(class_loss)
            bce_losses.append(bce_loss)
            dice_losses.append(dice_loss)
            if score_loss is not None:
                score_losses.append(score_loss)

        # reduce batch loss
        class_loss = torch.stack(class_losses).mean() if self.use_mean_batch_loss else torch.stack(class_losses).sum()
        bce_loss = torch.stack(bce_losses).mean() if self.use_mean_batch_loss else torch.stack(bce_losses).sum()
        dice_loss = torch.stack(dice_losses).mean() if self.use_mean_batch_loss else torch.stack(dice_losses).sum()
        if len(score_losses) > 0:
            score_loss = torch.stack(score_losses).mean() if self.use_mean_batch_loss else torch.stack(score_losses).sum()
        else:
            score_loss = torch.tensor(0.0, device=class_loss.device)

        # weight loss
        class_loss = self.class_loss_weight * class_loss
        bce_loss = self.bce_loss_weight * bce_loss
        dice_loss = self.dice_loss_weight * dice_loss
        score_loss = self.score_loss_weight * score_loss

        return {
            "class_loss": class_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "score_loss": score_loss,
        }

    def _get_class_loss(self, query_labels, target_labels):
        loss = F.cross_entropy(
            query_labels,
            target_labels.long(),
            weight=self._get_ce_weight(target_labels),
            label_smoothing=self.label_smoothing)
        return loss

    @torch.no_grad()
    def _get_ce_weight(self, target_label):
        ce_weight = torch.tensor([1.0] * self.num_instance_classes + [self.ce_non_object_weight], device=target_label.device)
        return ce_weight

    def _get_bce_loss(self, preds, targets):
        loss = F.binary_cross_entropy_with_logits(preds,
                                                  targets.float())
        return loss

    def _get_dice_loss(self, preds, targets):
        preds = preds.sigmoid()
        numerator = 2 * (preds * targets).sum(-1)
        denominator = preds.sum(-1) + targets.sum(-1)
        loss = 1 - ((numerator + 1) / (denominator + 1))
        return loss.mean()

    def _get_score_loss(self, pred_scores, pred_masks, target_masks, target_prim_lens):
        target_scores = self._cal_iou(pred_masks, target_masks, target_prim_lens)
        filter_id = torch.where(target_scores > 0.5)[0]
        if filter_id.numel():
            pred_scores = pred_scores[filter_id].reshape(-1)
            target_scores = target_scores[filter_id]
            score_loss = F.mse_loss(pred_scores, target_scores)
        else:
            score_loss = None
        return score_loss

    @torch.no_grad()
    def _cal_iou(self, preds, targets, prim_lens):
        preds = preds.sigmoid()
        preds = preds > 0.5
        targets = targets > 0.5
        prim_lens = torch.log(1 + prim_lens)
        inter = torch.sum(prim_lens * torch.logical_and(preds, targets), dim=-1)
        union = torch.sum(prim_lens * torch.logical_or(preds, targets), dim=-1)
        score = inter / (union + torch.finfo(torch.float32).eps)
        return score
