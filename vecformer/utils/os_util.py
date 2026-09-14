import os
import logging

def safe_symlink(target_path: str, link_path: str, logger: logging.Logger = None):
    """
    Create a soft link pointing to the target file or directory.
    If the link already exists, remove it before creating a new one.

    Args:
        base (str): The base directory for the soft link.
        target (str): The path to the target file or directory.
        name (str): The name of the soft link.

    Raises:
        OSError: If the soft link creation fails.
    """
    target_path = os.path.abspath(target_path)
    link_path = os.path.abspath(link_path)
    link_dir = os.path.dirname(link_path)
    link_name = os.path.basename(link_path)

    tmp_name = os.path.join(link_dir, f"{link_name}.{os.getpid()}")

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target path does not exist: {target_path}")

    try:
        target_path = os.path.relpath(target_path, link_dir)
        os.symlink(target_path, tmp_name)
        if os.path.islink(link_path):
            os.unlink(link_path)
        os.rename(tmp_name, link_path)
        if logger:
            logger.info(f"Soft link created: {link_path} -> {target_path}")
        else:
            print(f"Soft link created: {link_path} -> {target_path}")
    except OSError as e:
        if logger:
            logger.error(f"Failed to create soft link: {e}")
        else:
            print(f"Failed to create soft link: {e}")
