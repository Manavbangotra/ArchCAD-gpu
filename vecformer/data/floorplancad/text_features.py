"""
Text annotations of one drawing -> tensors for the TextCAD-style text encoder.

Types and attributes come from the repository's dataset/text_types.py (regex
typing written from the text measured in FloorPlanCAD-V2, CubiCasa5K and the US
plan sets). Geometry per annotation: sin and cos of its reading angle, and its size
relative to the drawing (TextCAD embeds each annotation's own geometry).

The dataset moves each annotation through augmentation as a short line from its
anchor along its reading direction (`text_vec`, as long as the text is tall), and
recomputes the geometry afterwards with `geometry_after_transform`, so rotation,
flips and scaling change angle and size exactly as they change the drawing.
"""
import math
import os
import sys

import torch

_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset"))
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)

import text_types  # noqa: E402

NUM_TYPES = len(text_types.TYPE_NAMES)
NUM_GRADES = len(text_types.GRADES)


def encode_texts(texts, viewbox, max_texts=512):
    """Returns a dict of tensors (T,...) and the (T, 2) raw positions, or None if no text.

    Annotations typed as padding (empty) are dropped; beyond `max_texts` the rest
    are dropped too (drawings rarely carry more; the median is 17 on FloorPlanCAD-V2).
    """
    if not texts:
        return None
    extent = max(float(viewbox[2]), float(viewbox[3]), 1e-6)
    types, attrs, grades, masks, geo, pos, vec, ratio = [], [], [], [], [], [], [], []
    for t in texts:
        tok = text_types.parse(t.get("text", ""))
        if tok.type == text_types.TYPE_ID["pad"]:
            continue
        types.append(tok.type)
        attrs.append(tok.attrs)
        grades.append(tok.grade)
        masks.append(tok.mask)
        a = math.radians(float(t.get("angle", 0.0)))
        size = float(t.get("size", 0.0))
        geo.append([math.sin(a), math.cos(a), size / extent * 100.0])
        pos.append([float(t.get("x", 0.0)), float(t.get("y", 0.0))])
        length = max(size, extent * 1e-3)                 # a zero-size text still needs a direction
        vec.append([math.cos(a) * length, math.sin(a) * length])
        ratio.append(size / length)
        if len(types) >= max_texts:
            break
    if not types:
        return None
    return dict(
        text_types=torch.tensor(types, dtype=torch.long),
        text_attrs=torch.tensor(attrs, dtype=torch.float32),
        text_grades=torch.tensor(grades, dtype=torch.long),
        text_masks=torch.tensor(masks, dtype=torch.bool),
        text_geo=torch.tensor(geo, dtype=torch.float32),
        text_vec=torch.tensor(vec, dtype=torch.float32),
        text_size_ratio=torch.tensor(ratio, dtype=torch.float32),
    ), torch.tensor(pos, dtype=torch.float32)


def geometry_after_transform(vec, size_ratio):
    """(T, 2) direction vectors in normalised coordinates (drawing extent = 1) after
    augmentation -> text_geo (T, 3): sin, cos of the reading angle and size in percent
    of the drawing, as encode_texts defines them."""
    length = vec.norm(dim=-1).clamp(min=1e-9)
    return torch.stack([vec[:, 1] / length, vec[:, 0] / length, length * size_ratio * 100.0], dim=-1)
