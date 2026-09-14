"""
This module provides utility functions for preprocess data.

It includes functions for:
- `scan_dir`: Scanning directories for files with specific extensions
- `get_namespace`: Get namespace from element
- `add_ns`: Add namespace to tag
- `del_ns`: Delete namespace from tag
- `primitive2str`: Convert primitive to string
"""

import os
import glob
import xml.etree.ElementTree as ET

import numpy as np

def scan_dir(dir_path: str, suffix: str, recursive: bool = True) -> list[str]:
    """
    Scan directory for files with specific suffix
    Args:
        dir_path: directory path
        suffix: suffix of the file, e.g. "png, svg, jpg, etc."
        recursive: whether to scan recursively
    Returns:
        list[str]: List of file paths relative to dir_path, containing only files with the specified suffix
    """
    if suffix.startswith("."):
        suffix = suffix[1:]
    if recursive:
        files = glob.glob(os.path.join(dir_path, "**", f"*.{suffix}"),
                          recursive=True)
    else:
        files = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
    return [os.path.relpath(f, dir_path) for f in files]


def get_namespace(element: ET.Element) -> str:
    """
    Get namespace from element
    """
    tag = element.tag
    if '}' in tag:
        return tag.split('}')[0].strip('{')
    return ''


def del_ns(tag: str, namespace: str) -> str:
    """
    Delete namespace from tag
    Args:
        tag: tag with namespace, e.g. "{http://www.w3.org/2000/svg}path"
        namespace: namespace string
    Returns:
        str: tag name without namespace, e.g. "path"
    """
    if namespace and tag.startswith(f"{{{namespace}}}"):
        return tag[len(namespace) + 2:]
    return tag


def add_ns(tag: str, namespace: str) -> str:
    """
    Add namespace to tag
    Args:
        tag: tag without namespace, e.g. "path"
        namespace: namespace string
    Returns:
        str: tag with namespace, e.g. "{http://www.w3.org/2000/svg}path"
    """
    if namespace:
        return f"{{{namespace}}}{tag}"
    return tag


def primitive2str(primitive: ET.Element) -> str:
    """
    Convert primitive to string
    Args:
        primitive: primitive element
    Returns:
        str: primitive string
    """
    pri_str = "<"
    pri_str += f"{del_ns(primitive.tag, get_namespace(primitive))}"
    for key, value in primitive.attrib.items():
        pri_str += f" {key}=\"{value}\""
    pri_str += "/>"
    return pri_str


def get_t_values(sample_points: int) -> list[float]:
    """
    Get t values
    Args:
        sample_points: number of sample points
    Returns:
        list[float]: list of t values
    """
    t_values = np.linspace(0, 1, sample_points)
    return t_values.tolist()