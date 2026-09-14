import logging
import os.path as osp

import torch

from model import register_model
from .modeling_vecformer import VecFormer
from .configuration_vecformer import VecFormerConfig
from .vecformer_trainer import VecFormerTrainer

logger = logging.getLogger("transformers")


def load_matching(model: torch.nn.Module, checkpoint: str):
    """Warm start: load every tensor whose name and shape match; report the rest.

    `checkpoint` is a Trainer checkpoint directory (model.safetensors or
    pytorch_model.bin) or one of those files. Used when the next stage changes
    the head (35-class TextCAD -> Arch-43 product) or adds modules (text fusion).
    """
    path = checkpoint
    if osp.isdir(checkpoint):
        for name in ("model.safetensors", "pytorch_model.bin"):
            if osp.isfile(osp.join(checkpoint, name)):
                path = osp.join(checkpoint, name)
                break
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu")
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    shape_mismatch = sorted(k for k, v in state.items() if k in own and own[k].shape != v.shape)
    missing = sorted(k for k in own if k not in matched)
    model.load_state_dict(matched, strict=False)
    logger.info(f"init_checkpoint {path}: loaded {len(matched)}/{len(own)} tensors; "
                f"{len(shape_mismatch)} shape mismatches (e.g. {shape_mismatch[:3]}); "
                f"{len(missing)} left at initialisation")
    return dict(loaded=len(matched), shape_mismatch=shape_mismatch, missing=missing)


@register_model("vecformer")
def build(model_args: dict):
    model_args = dict(model_args)
    init_checkpoint = model_args.pop("init_checkpoint", None)
    model = VecFormer(VecFormerConfig(**model_args))
    if init_checkpoint:
        load_matching(model, init_checkpoint)
    return model, VecFormerTrainer
