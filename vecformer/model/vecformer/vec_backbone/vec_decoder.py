from .vec_encoder import VecEncoder
from ..modules import AddNorm


class VecDecoder(VecEncoder):
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
        super().__init__(
            id=id,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            attn_drop=attn_drop,
            activation=activation,
            dropout=dropout,
            rope_dim=rope_dim,
            rope_theta=rope_theta,
            rope_learnable=rope_learnable,
            use_prim_fusion=use_prim_fusion,
            use_layer_fusion=use_layer_fusion
        )

        self.add_norm = AddNorm(embed_dim, dropout)

    def forward(self, coords, feats, prim_id_map, layer_id_map, cu_seqlens, residual_feats=None):
        if residual_feats is not None:
            feats = self.add_norm(feats, residual_feats)
        feats = super().forward(coords, feats, prim_id_map, layer_id_map, cu_seqlens)
        return feats
