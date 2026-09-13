import torch
from torch import distributed as dist

import functools
import os
from collections import OrderedDict
from math import cos, pi
from .dist import get_dist_info, master_only
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np

def get_device():
    """Preferred compute device: CUDA when present, otherwise CPU.

    Upstream hardcodes `.cuda()` throughout. Routing every one of those through
    this helper lets the same code run unmodified on a CPU-only machine.
    Set ARCHCAD_DEVICE (e.g. "cpu") to override the automatic choice.
    """
    override = os.environ.get("ARCHCAD_DEVICE")
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Setting of Global Seed

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=None):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)



class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, apply_dist_reduce=False):
        self.apply_dist_reduce = apply_dist_reduce
        self.reset() #

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def dist_reduce(self, val):
        rank, world_size = get_dist_info()
        if world_size == 1:
            return val
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device="cuda")
        dist.all_reduce(val)
        return val.item() / world_size

    def get_val(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.val)
        else:
            return self.val

    def get_avg(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.avg)
        else:
            return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def fixed_lr(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    """
    固定的学习率
    """
    lr = base_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def custimized_lr(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    """
    定制的学习率计划, 注意已训练过的ep

    注意: lr_list记录断点ep开始的后的lr计划, 同时也要注意和batch_size统一
    step_epoch要记录为断点的step
    """
    lr_list = [0.0002, 0.00015, 0.0001, 0.0001, 0.0001, 0.0001]
    try:
        lr = lr_list[epoch - 1 - step_epoch]
    except:
        lr = base_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def cosine_lr_after_step_a(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6, accumulation=1):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    lr = lr * accumulation
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def get_scheduler(cfg, optimizer):
    if cfg.type == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay)
    else:
        raise Exception('Not impl. such scheduler')
    return scheduler

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


@master_only
def _atomic_save(obj, path):
    """Save without ever leaving `path` missing or half-written.

    The previous implementation deleted latest.pth and then wrote the new one.
    Losing power inside that window destroyed the only rolling checkpoint —
    exactly the failure mode on a machine with scheduled outages. Writing to a
    temporary file and renaming makes the swap atomic, and the previous
    checkpoint is kept as a second line of defence.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)

    # Force the bytes to disk before the rename, so a crash cannot leave a
    # renamed-but-empty file behind.
    # Writable handle: Windows refuses fsync on a read-only one (EBADF).
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())

    if os.path.exists(path):
        prev = path.replace(".pth", "_prev.pth")
        try:
            os.replace(path, prev)
        except OSError:
            pass

    os.replace(tmp, path)


def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16, best=False,
                    rolling=False, best_metric=None):
    if hasattr(model, "module"):
        model = model.module

    if best:
        checkpoint = {"net": weights_to_cpu(model.state_dict()), "epoch": epoch,
                      "best_metric": best_metric}
        _atomic_save(checkpoint, os.path.join(work_dir, "best.pth"))
        return

    # best_metric rides along so that resuming does not restart the comparison
    # from zero and overwrite best.pth with a worse checkpoint.
    checkpoint = {"net": weights_to_cpu(model.state_dict()), "optimizer": optimizer.state_dict(),
                  "epoch": epoch, "best_metric": best_metric}

    # `rolling` writes only latest.pth — used for frequent mid-epoch saves so an
    # unexpected shutdown costs minutes instead of a whole epoch.
    if not rolling:
        # Weights only (~0.4 GB instead of ~1.2 GB with AdamW state): the kept
        # epoch files are for evaluation and rollback; resuming uses latest.pth.
        # A 50-epoch run kept ~16 GB of these before.
        torch.save({k: v for k, v in checkpoint.items() if k != "optimizer"},
                   os.path.join(work_dir, f"epoch_{epoch}.pth"))

    _atomic_save(checkpoint, os.path.join(work_dir, "latest.pth"))

    if rolling:
        return

    # remove previous checkpoints unless they are a power of 2 or a multiple of save_freq
    epoch = epoch - 1
    f = os.path.join(work_dir, f"epoch_{epoch}.pth")
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


# Buffers that are configuration, not learned state. The criterion's
# `empty_weight` holds the run's class weights; restoring it from a checkpoint
# silently replaced the weights the current config asked for.
CONFIG_STATE_PREFIXES = ("criterion.",)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False,
                    skip_prefixes=CONFIG_STATE_PREFIXES):
    if hasattr(model, "module"):
        model = model.module
    # Load onto CPU first, then let the caller move the model; this keeps
    # CUDA-serialised checkpoints usable on a CPU-only machine.
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    src_state_dict = state_dict["net"]
    target_state_dict = model.state_dict()
    for k in [k for k in src_state_dict if k.startswith(tuple(skip_prefixes))]:
        del src_state_dict[k]
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        # A warning, not info: a head that fails to transfer is easy to miss.
        logger.warning(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    missing_keys = [k for k in missing_keys if not k.startswith(tuple(skip_prefixes))]
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

    # load optimizer
    if optimizer is not None:
        assert "optimizer" in state_dict, (
            f"{checkpoint} holds weights only; resume from latest.pth, or load it as "
            "`pretrain` to start a new schedule from these weights")
        optimizer.load_state_dict(state_dict["optimizer"])

    if "epoch" in state_dict:
        epoch = state_dict["epoch"]
    else:
        epoch = 0
    return epoch + 1


def get_max_memory():
    if not torch.cuda.is_available():
        # No CUDA allocator to query; report host RSS so the training log
        # still carries a useful memory figure on CPU.
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)], dtype=torch.int, device="cuda")
    _, world_size = get_dist_info()
    if world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()


def _to_device(x, device):
    # Recurse into lists and tuples: the collate function hands images and their
    # centres over as lists of tensors. Moving only bare tensors left those on the
    # CPU, ImgEmbed raised a device mismatch on every batch, and the training loop
    # caught it and skipped the batch -- GPU training ran for whole epochs
    # without a single backward pass.
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)) and any(isinstance(v, (torch.Tensor, list, tuple)) for v in x):
        return type(x)(_to_device(v, device) for v in x)
    return x


def cuda_cast(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        device = get_device()
        new_args = [_to_device(x, device) for x in args]
        new_kwargs = {k: _to_device(v, device) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapper
