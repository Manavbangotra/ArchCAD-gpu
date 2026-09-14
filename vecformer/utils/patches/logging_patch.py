from transformers import logging


def custom_enable_explicit_format(
    format_str:
    str = "[%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s] -> %(message)s"
) -> None:
    """
    Custom format for HuggingFace Transformers's logger with monkey patch.
    """
    handlers = logging._get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.logging.Formatter(format_str)
        handler.setFormatter(formatter)

def patch_logging():
    """
    Patch the logging to use the custom format.
    """
    logging.enable_explicit_format = custom_enable_explicit_format
