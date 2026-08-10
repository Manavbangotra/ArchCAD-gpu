"""
Minimal stand-ins for the handful of detectron2 symbols this project uses.

Upstream pulls in the whole of detectron2 for three small things. detectron2
has to be compiled from source and (in practice) wants CUDA, which makes it a
heavy dependency for what it provides here. These shims are used only when
detectron2 is not importable, so a real install still takes precedence.

Covers:
  * get_world_size          -> svgnet/model/criterion.py
  * maybe_add_gradient_clipping, CfgNode -> svgnet/util/optim.py
"""

import torch
import torch.distributed as dist


def get_world_size() -> int:
    """Number of processes in the default process group (1 if not distributed)."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


class CfgNode(dict):
    """Attribute-accessible dict, standing in for detectron2's CfgNode.

    optim.py only uses it as a scratch namespace to record clipping settings,
    so plain attribute get/set is enough.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def maybe_add_gradient_clipping(cfg, optimizer):
    """Wrap `optimizer` so it clips gradients on each step, per `cfg`.

    Mirrors detectron2's helper: reads cfg.SOLVER.CLIP_GRADIENTS and returns the
    optimizer untouched when clipping is disabled.
    """
    solver = getattr(cfg, "SOLVER", None)
    clip = getattr(solver, "CLIP_GRADIENTS", None) if solver is not None else None
    if clip is None or not getattr(clip, "ENABLED", False):
        return optimizer

    clip_value = getattr(clip, "CLIP_VALUE", 1.0)
    norm_type = getattr(clip, "NORM_TYPE", 2.0)

    class _GradientClippingOptimizer(type(optimizer)):
        def step(self, closure=None):
            for group in self.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], clip_value, norm_type)
            super().step(closure=closure)

    optimizer.__class__ = _GradientClippingOptimizer
    return optimizer
