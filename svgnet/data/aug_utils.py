"""
Geometric augmentations for SVG primitive point sets.

Coordinates are expected to already be normalised to [0, 1], so the default
width/height of 1 corresponds to the full drawing extent.
"""

import math
import random

import numpy as np


def RandomHorizonFilp(args, width=1):
    """Mirror x coordinates about the vertical centre line."""
    args[:, 0::2] = width - args[:, 0::2]
    return args


def RandomVerticalFilp(args, Hight=1):
    """Mirror y coordinates about the horizontal centre line."""
    args[:, 1::2] = Hight - args[:, 1::2]
    return args


def rotate_xy(args, width, height, angle):
    """Rotate points by `angle` degrees about the drawing centre."""
    pi_angle = angle * math.pi / 180.0
    a, b = width / 2, height / 2
    x0, y0 = args[:, ::2], args[:, 1::2]
    x_rot = (x0 - a) * math.cos(pi_angle) - (y0 - b) * math.sin(pi_angle) + a
    y_rot = (x0 - a) * math.sin(pi_angle) + (y0 - b) * math.cos(pi_angle) + b
    return np.concatenate([x_rot, y_rot], axis=1)


def random_rotate(points, width, height):
    """Rotate points by a uniformly random angle about the drawing centre."""
    angle = random.uniform(-180, 180)
    centroid = np.array([width / 2, height / 2])
    centered = points - centroid

    rad = np.deg2rad(angle)
    rotation = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)],
    ])
    return np.dot(centered, rotation.T) + centroid
