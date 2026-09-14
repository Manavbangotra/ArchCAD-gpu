__all__ = ["apply_patches", "patch_logging", "patch_printer_callback", "patch_training_arguments"]

from .logging_patch import patch_logging
from .printer_callback_patch import patch_printer_callback
from .training_arguments_patch import patch_training_arguments

def apply_patches():
    """
    Apply all patches.
    """
    patch_logging()
    patch_printer_callback()
    patch_training_arguments()
