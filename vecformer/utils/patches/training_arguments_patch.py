from typing import Optional
from dataclasses import dataclass, field

import transformers

@dataclass
class CustomTrainingArguments(transformers.TrainingArguments):
    config_path: Optional[str] = field(default=None)
    model_args_path: Optional[str] = field(default=None)
    data_args_path: Optional[str] = field(default=None)
    launch_mode: Optional[str] = field(default="train")


def patch_training_arguments():
    """
    Patch the TrainingArguments to add config_path as an argument.
    """
    transformers.TrainingArguments = CustomTrainingArguments
