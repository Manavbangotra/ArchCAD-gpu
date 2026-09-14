import json
import logging
import warnings

# ------------ apply patches at very beginning ----------- #
from utils import apply_patches

apply_patches()

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
    # ---------------------------------------------------- #
    if training_args.launch_mode not in ["train", "continue", "test"]:
        raise ValueError(f"Invalid launch mode: {training_args.launch_mode}")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
    dist.destroy_process_group()
