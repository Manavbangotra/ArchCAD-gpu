from typing import Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer

from .configuration_vecformer import VecFormerConfig
from .evaluator import MetricsComputer, MetricsComputerConfig


class VecFormerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         compute_metrics=MetricsComputer(MetricsComputerConfig(**VecFormerConfig().metrics_computer_config)),
                         **kwargs)
        self.label_names = ["sem_ids", "inst_ids", "prim_lengths", "cu_numprims", "data_paths"]
        self.custom_logs: Dict[str, torch.Tensor] = {}
        self.custom_logs_accumulated_step: Dict[str, int] = {}
        self.custom_logs_is_training: bool = False

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # ----------- hack to log multiple loss ---------- #
        if self.custom_logs_is_training:
            if self.custom_logs:
                stacked_values = torch.stack(list(self.custom_logs.values()))
                gathered_values = self._nested_gather(stacked_values).view(dist.get_world_size(), -1)
                mean_values = gathered_values.mean(dim=0)
                for key, value in zip(self.custom_logs.keys(), mean_values):
                    logs[key] = round(value.item() / (self.custom_logs_accumulated_step[key]), 4)
                self.custom_logs.clear()
        # ------------------------------------------------ #
        super().log(logs, start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        # ----------- hack to log multiple loss ---------- #
        is_training = return_outputs is False # in huggingface trainer, return_outputs is True only when evaluating
        self.custom_logs_is_training = is_training is True
        if is_training:
            dict_sublosses = outputs["dict_sublosses"]
            for key, value in dict_sublosses.items():
                if key in self.custom_logs:
                    self.custom_logs[key] += value
                    self.custom_logs_accumulated_step[key] += 1
                else:
                    self.custom_logs[key] = value
                    self.custom_logs_accumulated_step[key] = 1
        # ------------------------------------------------ #
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        # hack to handle `nested_detach` stuck in huggingface trainer
        losses, logits, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return losses, logits, torch.tensor(0.0, dtype=torch.float32, device=losses.device)