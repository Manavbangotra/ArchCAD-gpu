from dataclasses import fields
from transformers import HfArgumentParser

__all__ = ["get_args"]

def override_args(base_args, override_args, default_args):
    """
    Override base args with override args
    if an arg in override_args is different from default value,
    then override this arg in base_args with the value in override_args
    """
    for field in fields(override_args):
        arg_name = field.name
        arg_default = getattr(default_args, arg_name)
        arg_value = getattr(override_args, arg_name)
        if arg_value != arg_default:
            setattr(base_args, arg_name, arg_value)

def get_args(*args_dataclasses):
    """
    Get arguments from command line and yaml config
    use yaml config as default values,
    and override with command line args
    Args:
        args_dataclasses: tuple of dataclass types
    Returns:
        tuple of parsed dataclass objects
    """
    parser = HfArgumentParser(args_dataclasses)
    tuple_args_from_cmd = parser.parse_args_into_dataclasses()
    config_path = None
    for args in tuple_args_from_cmd:
        config_path = getattr(args, "config_path", None)
        if config_path:
            break
    if config_path is None:
        raise ValueError("config_path is not set")
    tuple_args_from_yaml = parser.parse_yaml_file(config_path)
    for args_from_yaml, args_from_cmd, args_dataclass in zip(tuple_args_from_yaml, tuple_args_from_cmd, args_dataclasses):
        override_args(args_from_yaml, args_from_cmd, args_dataclass())
    return tuple_args_from_yaml
