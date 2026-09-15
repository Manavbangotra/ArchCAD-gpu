import json
import logging
import os
import warnings

# ------------ apply patches at very beginning ----------- #
from utils import apply_patches

apply_patches()

# ---- CUDA kernels: pure-torch replacements where flash-attn/spconv/torch_scatter are missing
# (Windows / Python 3.12 has no wheels). On the Linux image the real kernels import and
# nothing is replaced.
from utils import torch_kernels

_replaced = torch_kernels.install()

# -------------- import installed modules ------------- #
from transformers import TrainingArguments
from transformers.utils.logging import (set_verbosity_info,
                                        enable_default_handler,
                                        enable_explicit_format)
import torch.distributed as dist

# ----------------- import custom modules ---------------- #
from model import build_model
from data import build_dataset
from utils import get_args

# --------------------- setup logging -------------------- #
enable_default_handler()
enable_explicit_format()

warnings.filterwarnings('ignore')
logger = logging.getLogger("transformers")


# ----------------------- main func ---------------------- #
def main():
    # --------------------- parse args -------------------- #
    training_args = get_args(TrainingArguments)[0]
    # ------------------- setup logging ------------------ #
    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        set_verbosity_info()

    logger.info(
        f"Training Arguments: {json.dumps(training_args.to_dict(), indent=4)}")
    if _replaced:
        logger.warning(f"CUDA kernels not installed, using pure-torch replacements for: {_replaced}")
    # ----------------------- model ---------------------- #
    model, ModelTrainer = build_model(training_args.model_args_path)
    # ---------------------- dataset --------------------- #
    dataset_splits, data_collator = build_dataset(training_args.data_args_path)
    # ---------------------- trainer --------------------- #
    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits.train,
        eval_dataset=dataset_splits.test
        if training_args.launch_mode == "test" else dataset_splits.val,
        data_collator=data_collator) # type: ignore
    # ----------------------- train ---------------------- #
    if training_args.launch_mode == "train":
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", train_result.metrics)
    # ------------------ continue train ------------------ #
    if training_args.launch_mode == "continue":
        if training_args.resume_from_checkpoint is None:
            raise ValueError(
                "resume_from_checkpoint is required for continue mode")
        logger.info(
            f"Continuing from checkpoint: {training_args.resume_from_checkpoint}"
        )
        trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
    # ----------------------- test ----------------------- #
    if training_args.launch_mode == "test":
        if training_args.resume_from_checkpoint is None:
            raise ValueError(
                "resume_from_checkpoint is required for test mode")
        logger.info(
            f"Testing from checkpoint: {training_args.resume_from_checkpoint}")
        trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("test", metrics)       # <output_dir>/test_results.json, read by tools/ablation_report.py
    # ---------------------------------------------------- #
    if training_args.launch_mode not in ["train", "continue", "test"]:
        raise ValueError(f"Invalid launch mode: {training_args.launch_mode}")


if __name__ == "__main__":
    # torchrun sets these; a plain `python launch.py` runs as a single process
    for key, value in dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1",
                           MASTER_PORT="29531").items():
        os.environ.setdefault(key, value)
    if os.name == "nt":
        os.environ.setdefault("USE_LIBUV", "0")       # Windows builds of torch have no libuv store
    import torch
    backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend)
    main()
    dist.destroy_process_group()
