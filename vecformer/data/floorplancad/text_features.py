"""
Text annotations of one drawing -> tensors for the TextCAD-style text encoder.

Types and attributes come from the repository's dataset/text_types.py (regex
typing written from the text measured in FloorPlanCAD-V2, CubiCasa5K and the US
plan sets). Geometry per annotation: sin and cos of its angle, and its size
relative to the drawing (TextCAD embeds each annotation's own geometry).
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
    are dropped too (drawings rarely carry more; the median is 14 on FloorPlanCAD-V2).
    """
    if not texts:
        return None
    extent = max(float(viewbox[2]), float(viewbox[3]), 1e-6)
    types, attrs, grades, masks, geo, pos = [], [], [], [], [], []
    for t in texts:
        tok = text_types.parse(t.get("text", ""))
        if tok.type == text_types.TYPE_ID["pad"]:
            continue
        types.append(tok.type)
        attrs.append(tok.attrs)
        grades.append(tok.grade)
        masks.append(tok.mask)
        a = math.radians(float(t.get("angle", 0.0)))
        geo.append([math.sin(a), math.cos(a), float(t.get("size", 0.0)) / extent * 100.0])
        pos.append([float(t.get("x", 0.0)), float(t.get("y", 0.0))])
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
    ), torch.tensor(pos, dtype=torch.float32)
