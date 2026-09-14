"""
Hack callback to log more messages
Modify from:
https://github.com/modelscope/ms-swift/blob/main/swift/trainers/callback.py
"""
import os
import time
import json
import logging

from transformers import trainer
from transformers.trainer_callback import PrinterCallback
from utils.os_util import safe_symlink

logger = logging.getLogger("transformers")


def format_time(seconds):
    days = int(seconds // (24 * 3600))
    hours = int((seconds % (24 * 3600)) // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if days > 0:
        time_str = f'{days}d {hours}h {minutes}m {seconds}s'
    elif hours > 0:
        time_str = f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        time_str = f'{minutes}m {seconds}s'
    else:
        time_str = f'{seconds}s'

    return time_str


def add_train_message(logs, state, start_time) -> None:
    logs['global_step/max_steps'] = f'{state.global_step}/{state.max_steps}'
    train_percentage = state.global_step / state.max_steps if state.max_steps else 0.
    logs['percentage'] = f'{train_percentage * 100:.2f}%'
    elapsed = time.time() - start_time
    logs['elapsed_time'] = format_time(elapsed)
    if train_percentage != 0:
        logs['remaining_time'] = format_time(elapsed / train_percentage -
                                             elapsed)
    for k, v in logs.items():
        if isinstance(v, float):
            logs[k] = round(logs[k], 8)


def append_to_jsonl(jsonl_path, logs):
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(logs) + '\n')


class PatchedPrinterCallback(PrinterCallback):

    def on_init_end(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return super().on_init_end(args, state, control, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return super().on_train_begin(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        add_train_message(logs, state, self.start_time)
        if state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)

        _ = logs.pop('total_flos', None)
        if state.is_local_process_zero:
            logger.info(logs)

    def on_save(self, args, state, control, **kwargs):
        # create an easy-to-find soft link to the latest checkpoint
        if state.is_world_process_zero:
            safe_symlink(
                args.output_dir,
                os.path.join(os.path.dirname(args.output_dir), "latest"),
            )
            if state.best_model_checkpoint:
                safe_symlink(
                    state.best_model_checkpoint,
                    os.path.join(args.output_dir, "checkpoint-best"),
                )
        return super().on_save(args, state, control, **kwargs)

def patch_printer_callback():
    """
    Patch the PrinterCallback to log more messages
    """
    trainer.PrinterCallback = PatchedPrinterCallback
