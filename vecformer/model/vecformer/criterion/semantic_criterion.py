import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticCriterion(nn.Module):
    """
    Semantic criterion

    Args:

        `num_semantic_classes` (`int`): Number of semantic classes

        `ce_loss_weight` (`float`): Weight for cross entropy loss

        `ce_unlabeled_weight` (`float`): Weight for unlabeled class in cross entropy loss

        `label_smoothing` (`float`): Label smoothing value. see `nn.CrossEntropyLoss`.

        `use_mean_batch_loss` (`bool`): If set `True`,
            use `mean` loss to reduce loss of each `batch`,
            else use `sum` loss to reduce loss of each `batch`
    """
    def __init__(self,
                 num_semantic_classes: int,
                 ce_loss_weight: float = 0.2,
                 ce_unlabeled_weight: float = 0.01,
                 label_smoothing: float = 0.1,
                 use_mean_batch_loss: bool = True):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.ce_loss_weight = ce_loss_weight
        self.ce_unlabeled_weight = ce_unlabeled_weight
        self.label_smoothing = label_smoothing
        self.use_mean_batch_loss = use_mean_batch_loss

    label_space = None      # set by Criterion when the model has a label space

    def forward(self, pred_labels, target_labels, target_sources=None):
        """
        Calculate the semantic loss

        Args:

            `pred_labels` (`List[List[torch.Tensor]]`, each tensor shape is (n_queries, num_semantic_classes + 1)): Predicted labels
                the first list is the predicted labels of each decoder block, the second list is the predicted labels of each batch

            `target_labels` (`List[torch.Tensor]`, each tensor shape is (n_queries,)): Target labels
                the list is the target labels of each batch

        Returns:

            `sem_loss` (`Dict[str, torch.Tensor]`): semantic loss
        """
        blocks_losses = []

        # get loss
        for block_pred_labels in pred_labels:
            blocks_losses.append(self._get_loss(block_pred_labels, target_labels, target_sources))

        # post process
        sem_loss = {}
        if len(blocks_losses) == 1:
            sem_loss["sem_ce_loss"] = blocks_losses[0]["ce_loss"]
        else:
            for block_id, block_loss in enumerate(blocks_losses):
                sem_loss[f"sem_block_{block_id}_ce_loss"] = block_loss["ce_loss"]
            sem_loss["sem_ce_loss"] = torch.stack([block_loss["ce_loss"] for block_loss in blocks_losses]).sum()
        sem_loss["semantic_loss"] = sem_loss["sem_ce_loss"]

        return sem_loss

    def _get_loss(self, pred_labels, target_labels, target_sources=None):
        loss = []
        ls = self.label_space
        for i, (pred_label, target_label) in enumerate(zip(pred_labels, target_labels)):
            if ls is None or (ls.plain and bool((target_label <= self.num_semantic_classes).all())):
                loss.append(F.cross_entropy(pred_label, target_label.long()))
                continue
            # marginal cross-entropy over each target's head columns (label_space.py)
            source = int(target_sources[i]) if target_sources is not None else -1
            rows = ls.target_rows(target_label, source)
            valid = rows.any(-1)
            if valid.any():
                loss.append(ls.marginal_nll(pred_label[valid], rows[valid]).mean().to(pred_label.dtype))
            else:
                loss.append(pred_label.sum() * 0.0)
        loss = torch.stack(loss).mean() if self.use_mean_batch_loss else torch.stack(loss).sum()
        return dict(ce_loss=self.ce_loss_weight * loss)

    @torch.no_grad()
    def _get_ce_weight(self, target_label):
        target_label = torch.cat([target_label, torch.arange(0, self.num_semantic_classes + 1, device=target_label.device)], dim=-1)
        label_cnts = target_label.bincount()
        max_cnt = label_cnts.max()
        ce_weight = max_cnt / label_cnts
        ce_weight[self.num_semantic_classes] = self.ce_unlabeled_weight * ce_weight[self.num_semantic_classes]
        return ce_weight
