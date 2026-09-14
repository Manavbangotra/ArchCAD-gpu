from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn

from ..modules import (
    VarlenSelfAttention,
    VarlenCrossAttention,
    VarlenCrossAttentionWithMask,
    FFN,
    AddNorm,
    Projection,
    cat_tensor_to_list_tensor
)

CADDecoderOutput = List[Dict[str, Optional[List[torch.Tensor]]]]

class CADDecoderBlock(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, attn_drop: float,
                 dropout: float, activation: str, use_attn_mask: bool = False):
        super().__init__()
        self.self_attn = VarlenSelfAttention(embed_dim, n_heads, attn_drop,
                                             dropout)
        if use_attn_mask:
            self.cross_attn = VarlenCrossAttentionWithMask(embed_dim, n_heads, dropout)
        else:
            self.cross_attn = VarlenCrossAttention(embed_dim, n_heads, attn_drop, dropout)
        self.ffn = FFN(embed_dim, activation, dropout)
        self.self_attn_norm = AddNorm(embed_dim, dropout)
        self.cross_attn_norm = AddNorm(embed_dim, dropout)
        self.ffn_norm = AddNorm(embed_dim, dropout)

    def forward(self,
                inputs,
                cu_seqlens_inputs,
                queries,
                cu_seqlens_queries,
                attn_masks=None):
        queries = self.self_attn_norm(
            self.self_attn(queries, cu_seqlens_queries), queries)
        if attn_masks is not None:
            output = self.cross_attn_norm(
                self.cross_attn(inputs, cu_seqlens_inputs, queries,
                                cu_seqlens_queries, attn_masks=attn_masks), queries)
        else:
            output = self.cross_attn_norm(
                self.cross_attn(inputs, cu_seqlens_inputs, queries,
                                cu_seqlens_queries), queries)
        output = self.ffn_norm(self.ffn(output), output)
        return output


class CADDecoder(nn.Module):

    def __init__(self,
                 num_instance_classes: int,
                 num_semantic_classes: int,
                 input_dim: int,
                 embed_dim: int,
                 activation: str = "GELU",
                 dropout: float = 0.1,
                 n_heads: int = 8,
                 n_blocks: int = 3,
                 attn_drop: float = 0.1,
                 objectiveness_flag: bool = True,
                 iter_pred: bool = True,
                 only_last_block_sem: bool = True,
                 use_attn_mask: bool = True,
                 ):
        super().__init__()
        self.num_instance_classes = num_instance_classes
        self.num_semantic_classes = num_semantic_classes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.activation = activation
        self.dropout = dropout
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attn_drop = attn_drop
        self.objectiveness_flag = objectiveness_flag
        self.iter_pred = iter_pred
        self.only_last_block_sem = only_last_block_sem
        self.use_attn_mask = use_attn_mask

        # project input features to embed_dim
        self.input_proj = Projection(self.input_dim, self.embed_dim,
                                     self.activation, self.dropout)

        # project input features to mask features
        self.mask_proj = Projection(self.input_dim, self.embed_dim,
                                    self.activation, self.dropout,
                                    use_norm=False)

        # project queries features to embed_dim
        self.query_proj = Projection(self.input_dim, self.embed_dim,
                                     self.activation, self.dropout,
                                     use_norm=False)

        # decoder blocks
        self.blocks = nn.ModuleList([
            CADDecoderBlock(self.embed_dim, self.n_heads, self.attn_drop,
                            self.dropout, self.activation, self.use_attn_mask)
            for _ in range(self.n_blocks)
        ])

        # normalize queries features
        self.query_norm = nn.LayerNorm(self.embed_dim)

        # project queries features to semantic class logits
        self.sem_class_head = Projection(self.embed_dim,
                                         self.num_semantic_classes + 1,
                                         self.activation, self.dropout,
                                         use_norm=False)

        # project queries features to instance class logits
        self.inst_class_head = Projection(self.embed_dim,
                                          self.num_instance_classes + 1,
                                          self.activation, self.dropout,
                                          use_norm=False)

        # project queries features to predicted score
        if self.objectiveness_flag:
            self.score_head = Projection(self.embed_dim,
                                         1,
                                         self.activation,
                                         self.dropout,
                                         use_norm=False,
                                         use_sigmoid=True)

    def forward(self,
            feats: torch.Tensor,
            cu_seqlens: torch.Tensor,
            queries: torch.Tensor,
            query_cu_seqlens: torch.Tensor,
        ) -> CADDecoderOutput:
        """
        Args:

            `feats` (`torch.Tensor`, shape is (N1+N2+..., feats_embed_dim)): Features of primitives
                N1, N2, ... are the number of primitives in each batch

            `cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of primitives
                The first element is 0, and the last element is the total number of primitives in all batches

            `queries` (`torch.Tensor`, shape is (Q1+Q2+..., embed_dim)): Queries of all batches
                Q1, Q2, ... are the number of queries in each batch

            `query_cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of queries
                The first element is 0, and the last element is the total number of queries in all batches

            `layer_ids` (`torch.Tensor`, shape is (N1+N2+..., 1)): Layer ids of primitives
                N1, N2, ... are the number of primitives in each batch

        Returns:

            `outputs` (`CADDecoderOutput`): A list of dictionaries, each dictionary contains outputs of a layer
                if `iter_pred` is `False`, the list contains only the last layer's outputs, the dictionary contains

                \\- `list_pred_sem_labels` (`List[torch.Tensor]`, each tensor shape is (Q, num_semantic_classes + 1)):
                    Predicted semantic label logits of each query, Q is the number of queries

                \\- `list_pred_inst_masks` (`List[torch.Tensor]`, each tensor shape is (Q, N)):
                    Predicted instance mask of each query, Q is the number of queries, N is the number of primitives

                \\- `list_pred_inst_labels` (`List[torch.Tensor]`, each tensor shape is (Q, num_instance_classes + 1)):
                    Predicted instance label logits of each query, Q is the number of queries

                \\- `list_pred_inst_scores` (`Optional[List[torch.Tensor]]`, each tensor shape is (Q, 1)):
                    Predicted instance confidence score of each query, Q is the number of queries
        """
        outputs = []

        # project input features to embed_dim
        input_feats = self.input_proj(feats)
        # get mask features
        mask_feats = self.mask_proj(feats)
        # project queries features to embed_dim
        queries = self.query_proj(queries)
        # decoder blocks
        if self.iter_pred:
            # first forward
            head_outputs = self._forward_head(queries, query_cu_seqlens, mask_feats, cu_seqlens, is_last_block=False)
            attn_masks = head_outputs[4]
            for i, block in enumerate(self.blocks):
                queries = block(input_feats, cu_seqlens, queries, query_cu_seqlens, attn_masks)
                head_outputs = self._forward_head(queries, query_cu_seqlens, mask_feats, cu_seqlens, is_last_block=(i == self.n_blocks - 1))
                outputs.append(dict(
                    list_pred_sem_labels=head_outputs[0],
                    list_pred_inst_masks=head_outputs[1],
                    list_pred_inst_labels=head_outputs[2],
                    list_pred_inst_scores=head_outputs[3]
                ))
                attn_masks = head_outputs[4]
        else:
            for block in self.blocks:
                queries = block(input_feats, cu_seqlens, queries, query_cu_seqlens)
            head_outputs = self._forward_head(queries, query_cu_seqlens, mask_feats, cu_seqlens)
            outputs.append(dict(
                list_pred_sem_labels=head_outputs[0],
                list_pred_inst_masks=head_outputs[1],
                list_pred_inst_labels=head_outputs[2],
                list_pred_inst_scores=head_outputs[3]
            ))

        return outputs

    def _forward_head(
        self,
        queries: torch.Tensor,
        query_cu_seqlens: torch.Tensor,
        mask_feats: torch.Tensor,
        mask_cu_seqlens: torch.Tensor,
        is_last_block: bool = True
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]],
               List[torch.Tensor], Optional[List[torch.Tensor]],
               Optional[List[torch.Tensor]]]:
        """
        Prediction head forward

        Args:

            `queries` (`torch.Tensor`, shape is (Q1+Q2+..., embed_dim)): Queries of all batches
                Q1, Q2, ... are the number of queries in each batch

            `query_cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of queries
                The first element is 0, and the last element is the total number of queries in all batches

            `mask_feats` (`torch.Tensor`, shape is (N1+N2+..., embed_dim)): Mask features of all batches
                N1, N2, ... are the number of primitives in each batch

            `mask_cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of mask features
                The first element is 0, and the last element is the total number of mask features in all batches

        Returns:

            `list_pred_sem_labels` (`Optional[List[torch.Tensor]]`, each tensor shape is (Q, num_semantic_classes + 1)):
                Predicted semantic label logits of each query, Q is the number of queries

            `list_pred_inst_masks` (`List[torch.Tensor]`, each tensor shape is (Q, N)):
                Predicted instance mask of each query, Q is the number of queries, N is the number of primitives

            `list_pred_inst_labels` (`List[torch.Tensor]`, each tensor shape is (Q, num_instance_classes + 1)):
                Predicted instance label logits of each query, Q is the number of queries

            `list_pred_inst_scores` (`Optional[List[torch.Tensor]]`, each tensor shape is (Q, 1)):
                Predicted instance confidence score of each query, Q is the number of queries

            `attn_masks` (`Optional[List[torch.Tensor]]`, each tensor shape is (Q, N)):
                Attention masks, Q is the number of queries, N is the number of primitives
        """
        queries = self.query_norm(queries)

        # predict semantic labels
        if self.only_last_block_sem and not is_last_block:
            list_pred_sem_labels = None
        else:
            sem_labels = self.sem_class_head(queries)
            list_pred_sem_labels = cat_tensor_to_list_tensor(sem_labels, query_cu_seqlens)

        # predict instance labels
        inst_labels = self.inst_class_head(queries)
        list_pred_inst_labels = cat_tensor_to_list_tensor(inst_labels, query_cu_seqlens)

        # predict instance scores
        if self.objectiveness_flag:
            inst_scores = self.score_head(queries)
            list_pred_inst_scores = cat_tensor_to_list_tensor(inst_scores, query_cu_seqlens)
        else:
            list_pred_inst_scores = None

        # predict query masks
        list_pred_inst_masks = []
        for i in range(len(query_cu_seqlens) - 1):
            query_idx_start = query_cu_seqlens[i]
            query_idx_end = query_cu_seqlens[i + 1]
            mask_idx_start = mask_cu_seqlens[i]
            mask_idx_end = mask_cu_seqlens[i + 1]
            inst_mask = torch.einsum('qe,ne->qn',
                                      queries[query_idx_start:query_idx_end],
                                      mask_feats[mask_idx_start:mask_idx_end])
            list_pred_inst_masks.append(inst_mask)

        # get attention masks
        attn_masks = self._get_attn_masks(list_pred_inst_masks) if self.use_attn_mask else None

        return list_pred_sem_labels, list_pred_inst_masks, list_pred_inst_labels, list_pred_inst_scores, attn_masks

    @torch.no_grad()
    def _get_attn_masks(self, list_pred_inst_masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Get attention masks

        Args:

            `list_pred_inst_masks` (`List[torch.Tensor]`, each tensor shape is (Q, N)):
                Predicted instance mask of each query, Q is the number of queries, N is the number of primitives

        Returns:

            `attn_masks` (`List[torch.Tensor]`, each tensor shape is (Q, N)):
                Attention masks, Q is the number of queries, N is the number of primitives
        """
        attn_masks = []
        for inst_mask in list_pred_inst_masks:
            attn_mask = (inst_mask.sigmoid() < 0.5).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_masks.append(attn_mask)
        return attn_masks
