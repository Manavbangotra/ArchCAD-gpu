__all__ = ["build_model", "register_model"]

import json
import yaml
import logging
import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from transformers import Trainer, PreTrainedModel

logger = logging.getLogger("transformers")

class ModelRegistry:
    """Model registry for automatic model registration and building"""
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new model builder"""
        def wrapper(build_fn: Callable):
            if name in cls._registry:
                logger.warning(f"Model {name} already registered, overwriting...")
            cls._registry[name] = build_fn
            return build_fn
        return wrapper

    @classmethod
    def get_build_fn(cls, name: str) -> Callable:
        """get the build function of the registered model"""
        if name not in cls._registry:
            raise ValueError(f"Model {name} not found in registry")
        return cls._registry[name]

register_model = ModelRegistry.register

@dataclass
class ModelArguments:
    model_name: str = field(default="")
    model_args: dict = field(default_factory=dict)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

    def __repr__(self):
        return self.__str__()

def build_model(model_args_path: str) -> Tuple[PreTrainedModel, Trainer]:
    with open(model_args_path, "r") as f:
        model_args = yaml.safe_load(f)
    model_args = ModelArguments(**model_args)

    try:
        importlib.import_module("model." + model_args.model_name)
        build_fn = ModelRegistry.get_build_fn(model_args.model_name)
        model, trainer = build_fn(model_args.model_args)
        logger.info(f"Model Config: {model.config}")
        return model, trainer
    except ImportError as e:
        raise ImportError(f"Failed to import model module: {e}")
