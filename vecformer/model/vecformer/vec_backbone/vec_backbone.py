from typing import List, Tuple

import torch
import torch.nn as nn

from .vec_encoder import VecEncoder
from .vec_decoder import VecDecoder
from ..modules import (
    Projection,
    GroupFeatFusion,
    AbsolutePosEmbedding,
    list_tensor_to_cat_tensor,
)


class VecBackbone(nn.Module):
    def __init__(self,
            feats_input_dim: int = 10,  # (`int`): raw dimension of features
            embed_dim: int = 128,  # (`int`): embedding dimension of features
            output_dim: int = 128, # (`int`): output dimension of backbone
            n_blocks_enc: List[int] = [2, 3, 4, 6, 3], # (`list[int]`): number of blocks in each encoder layer
            prim_fusion_idx_enc: List[int] = [0, 1, 2, 3, 4], # (`list[int]`): indicates which layer will use primitive fusion
            layer_fusion_idx_enc: List[int] = [4], # (`list[int]`): indicates which layer will use layer fusion
            n_blocks_dec: List[int] = [2, 2, 2, 2, 2], # (`list[int]`): number of blocks in each decoder layer
            prim_fusion_idx_dec: List[int] = [0, 1, 2, 3, 4], # (`list[int]`): indicates which layer will use primitive fusion
            layer_fusion_idx_dec: List[int] = [4], # (`list[int]`): indicates which layer will use layer fusion
            n_heads: int = 8, # (`int`): number of attention heads
            attn_drop: float = 0.1, # (`float`): attention drop rate
            dropout: float = 0.1, # (`float`): dropout rate
            activation: str = "GELU", # (`str`): activation function
            ape_dim: int = 4, # (`int`): absolute position embedding dimension
            ape_theta: float = 10000.0, # (`float`): absolute position embedding theta
            ape_learnable: bool = True, # (`bool`): whether to learn absolute position embedding
            rope_dim: int = 4, # (`int`): RoPE dimension
            rope_theta: float = 10000.0, # (`float`): RoPE theta
            rope_learnable: bool = True, # (`bool`): whether to learn RoPE
        ) -> None:
        super().__init__()
        self.num_encoders = len(n_blocks_enc)
        self.num_decoders = len(n_blocks_dec)
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.ape = AbsolutePosEmbedding(
            embed_dim,
            coords_dim=ape_dim,
            theta=ape_theta,
            learnable=ape_learnable
        )
        self.feats_up_proj = Projection(
            feats_input_dim,
            embed_dim,
            activation,
            dropout
        )
        self.encoders = nn.ModuleList([
            VecEncoder(
                id=i,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_blocks=n_blocks_enc[i],
                attn_drop=attn_drop,
                dropout=dropout,
                activation=activation,
                rope_dim=rope_dim,
                rope_theta=rope_theta,
                rope_learnable=rope_learnable,
                use_prim_fusion=True if i in prim_fusion_idx_enc else False,
                use_layer_fusion=True if i in layer_fusion_idx_enc else False
            ) for i in range(self.num_encoders)
        ])
        self.decoders = nn.ModuleList([
            VecDecoder(
                id=i,
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_blocks=n_blocks_dec[i],
                attn_drop=attn_drop,
                dropout=dropout,
                activation=activation,
                rope_dim=rope_dim,
                rope_theta=rope_theta,
                rope_learnable=rope_learnable,
                use_prim_fusion=True if i in prim_fusion_idx_dec else False,
                use_layer_fusion=True if i in layer_fusion_idx_dec else False
            ) for i in range(self.num_decoders)
        ])
        self.output_proj = Projection(
            embed_dim,
            output_dim,
            activation,
            dropout
        )


    def forward(self, coords, feats, prim_id_map, layer_id_map, cu_seqlens):
        # prepare coords and feats
        coords = coords * 2000 # scale from [-0.5, 0.5] to [-1000, 1000]
        feats = self.feats_up_proj(feats)
        # absolute position embedding
        feats = self.ape(feats, coords)
        # encoder forward
        encoder_outputs = []
        for encoder in self.encoders:
            feats = encoder(coords, feats, prim_id_map, layer_id_map,
                            cu_seqlens)
            if encoder.id != self.num_encoders - 1:
                encoder_outputs.append(feats)
        # decoder forward
        for decoder in self.decoders:
            if decoder.id == 0:
                feats = decoder(coords, feats, prim_id_map, layer_id_map,
                                cu_seqlens)
            else:
                residual_feats = encoder_outputs[-decoder.id]
                feats = decoder(coords, feats, prim_id_map, layer_id_map,
                                cu_seqlens, residual_feats)
        feats, cu_seqlens = self._pooling_feats(feats, cu_seqlens, prim_id_map)
        feats = self.output_proj(feats)
        return feats, cu_seqlens


    def _pooling_feats(self,
            line_feats: torch.Tensor,
            line_cu_seqlens: torch.Tensor,
            primitive_id_map: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pooling line segments features to primitive features

        Args:

            `line_feats` (`torch.Tensor`, shape is (N1+N2+..., feats_embed_dim)): Features of line segments
                N1, N2, ... are the number of line segments in each batch

            `line_cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of line segments
                The first element is 0, and the last element is the total number of line segments in all batches

            `primitive_id_map` (`torch.Tensor`, shape is (N1+N2+...,)): Primitive id map
                Indicates the primitive id of each line segment, e.g.
                ```
                primitive_id_map[0] = 0
                ```
                means the 0-th line segment belongs to the 0-th primitive

        Returns:

            `prim_feats` (`torch.Tensor`, shape is (P1+P2+..., feats_embed_dim)): Features of primitives
                P1, P2, ... are the number of primitives in each batch

            `prim_cu_seqlens` (`torch.Tensor`, shape is (batch_size + 1,)): Cumulative sequence lengths of primitives
                The first element is 0, and the last element is the total number of primitives in all batches
        """
        prim_feats = []
        for batch_idx in range(len(line_cu_seqlens) - 1):
            idx_start = line_cu_seqlens[batch_idx]
            idx_end = line_cu_seqlens[batch_idx + 1]
            line_feats_batch = line_feats[idx_start:idx_end]
            primitive_id_map_batch = primitive_id_map[idx_start:idx_end]
            pooled_feats = GroupFeatFusion._group_pooling(
                line_feats_batch, primitive_id_map_batch)
            prim_feats.append(pooled_feats)

        prim_feats, prim_cu_seqlens = list_tensor_to_cat_tensor(prim_feats)

        return prim_feats, prim_cu_seqlens
