from model import register_model
from .modeling_vecformer import VecFormer
from .configuration_vecformer import VecFormerConfig
from .vecformer_trainer import VecFormerTrainer


@register_model("vecformer")
def build(model_args: dict):
    return VecFormer(VecFormerConfig(**model_args)), VecFormerTrainer