import torch.nn as nn

from ..modules import (
    TransformerBlock,
    GroupFeatFusion,
)


class VecEncoder(nn.Module):

    def __init__(self,
            id: int,
            embed_dim: int,
            n_heads: int,
            n_blocks: int,
            attn_drop: float = 0.1,
            activation: str = "GELU",
            dropout: float = 0.1,
            rope_dim: int = 4,
            rope_theta: float = 10000.0,
            rope_learnable: bool = True,
            use_prim_fusion: bool = True,
            use_layer_fusion: bool = True
        ) -> None:
        super().__init__()
        self.id = id

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                attn_drop=attn_drop,
                activation=activation,
                dropout=dropout,
                rope_dim=rope_dim,
                rope_theta=rope_theta,
                rope_learnable=rope_learnable
            ) for _ in range(n_blocks)
        ])
        if use_prim_fusion:
            self.group_feat_fusion_by_prim = GroupFeatFusion(embed_dim, dropout)
        else:
            self.group_feat_fusion_by_prim = lambda x, y, z: x
        if use_layer_fusion:
            self.group_feat_fusion_by_layer = GroupFeatFusion(embed_dim, dropout)
        else:
            self.group_feat_fusion_by_layer = lambda x, y, z: x

    def forward(self, coords, feats, prim_id_map, layer_id_map, cu_seqlens):
        for block in self.blocks:
            feats = block(coords, feats, cu_seqlens)
        feats = self.group_feat_fusion_by_prim(feats, prim_id_map, cu_seqlens)
        feats = self.group_feat_fusion_by_layer(feats, layer_id_map, cu_seqlens)
        return feats
