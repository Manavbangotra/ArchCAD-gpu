import torch
import torch.nn as nn

from .instance_criterion import InstanceCriterion
from .semantic_criterion import SemanticCriterion


class Criterion(nn.Module):
    def __init__(self,
                 instance_criterion_config: dict,
                 semantic_criterion_config: dict):
        super().__init__()
        self.instance_criterion = InstanceCriterion(**instance_criterion_config)
        self.semantic_criterion = SemanticCriterion(**semantic_criterion_config)

    def forward(self, preds, targets):
        """

        Args:

            `preds` (`List[Dict]`): A list of dictionaries, each dictionary contains outputs of a layer
                if length of `preds` is 1, the list contains only the last layer's outputs, the dictionary contains

                \\- `list_pred_sem_labels` (`List[torch.Tensor]`, each tensor shape is (Q, num_semantic_classes + 1)):
                    Predicted semantic label logits of each query, Q is the number of queries

                \\- `list_pred_inst_masks` (`List[torch.Tensor]`, each tensor shape is (Q, N)):
                    Predicted instance mask of each query, Q is the number of queries, N is the number of primitives

                \\- `list_pred_inst_labels` (`List[torch.Tensor]`, each tensor shape is (Q, num_instance_classes + 1)):
                    Predicted instance label logits of each query, Q is the number of queries

                \\- `list_pred_inst_scores` (`Optional[List[torch.Tensor]]`, each tensor shape is (Q, 1)):
                    Predicted instance confidence score of each query, Q is the number of queries

            `targets` (`Dict`):

                \\- `list_target_inst_labels` (`List[torch.Tensor]`, each tensor shape is (M, )):
                    Ground truth label of each instance, M is the number of instances

                \\- `list_target_inst_masks` (`List[torch.Tensor]`, each tensor shape is (M, N)):
                    Ground truth mask of each instance, M is the number of instances, N is the number of primitives

                \\- `list_target_prim_lens` (`List[torch.Tensor]`, each tensor shape is (N, )):
                    Ground truth length of each primitive, N is the number of primitives

                \\- `list_target_sem_labels` (`List[torch.Tensor]`, each tensor shape is (N, )):
                    Ground truth semantic label of each primitive, N is the number of primitives

                \\- `list_target_selected_idxs` (`List[torch.Tensor]`, each tensor shape is (Q,)):
                    indicates which primitives are selected as queries, Q is the number of queries

        Returns:

            `loss` (`torch.Tensor`): Total loss

            `dict_sublosses` (`Dict[str, torch.Tensor]`): A dictionary of loss, contains instance loss, semantic loss and total loss

                \\- `instance_loss` (`torch.Tensor`): Instance loss

                \\- `semantic_loss` (`torch.Tensor`): Semantic loss

                ... other auxiliary losses
        """
        dict_sublosses = {}
        # Calculate instance loss
        inst_preds, inst_targets = self._prepare_instance_preds_targets(preds, targets)
        inst_loss = self.instance_criterion(inst_preds, inst_targets)
        dict_sublosses.update(inst_loss)
        # Calculate semantic loss
        sem_preds, sem_targets = self._prepare_semantic_preds_targets(preds, targets)
        sem_loss = self.semantic_criterion(sem_preds, sem_targets)
        dict_sublosses.update(sem_loss)
        # calculate total loss
        loss = dict_sublosses['instance_loss'] + dict_sublosses['semantic_loss']
        # detach all sublosses for logging
        for key in dict_sublosses:
            dict_sublosses[key] = dict_sublosses[key].detach().clone()

        return loss, dict_sublosses

    @torch.no_grad()
    def _prepare_instance_preds_targets(self, preds, targets):
        inst_preds = []
        for block_preds in preds:
            inst_preds.append({
                "list_query_labels": block_preds['list_pred_inst_labels'],
                "list_query_scores": block_preds['list_pred_inst_scores'],
                "list_query_masks": block_preds['list_pred_inst_masks']
            })

        inst_targets = {
            "list_target_labels": targets['list_target_inst_labels'],
            "list_target_masks": targets['list_target_inst_masks'],
            "list_target_selected_idxs": targets['list_target_selected_idxs'],
            "list_target_prim_lens": targets['list_target_prim_lens']
        }

        return inst_preds, inst_targets

    @torch.no_grad()
    def _prepare_semantic_preds_targets(self, preds, targets):
        sem_preds = []
        for block_preds in preds:
            if block_preds['list_pred_sem_labels'] is not None:
                sem_preds.append(block_preds['list_pred_sem_labels'])

        sem_targets = []
        for batch_idxs in range(len(targets['list_target_sem_labels'])):
            # select `target sem labels of queries(num_queries, )` from `targets sem labels of prims(num_primitives, )`
            sem_targets.append(
                targets['list_target_sem_labels'][batch_idxs][targets['list_target_selected_idxs'][batch_idxs]]
            )

        return sem_preds, sem_targets