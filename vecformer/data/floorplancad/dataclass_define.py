from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ProcessArgs:
    file_path: str
    input_dir: str
    output_dir: str
    save_type: str
    connect_lines: bool
    line_t_values: list[float]
    curve_t_values: list[float]
    dynamic_sampling: bool
    dynamic_sampling_ratio: float


@dataclass
class SVGData:
    viewBox: list[float]
    coords: list[list[float]]
    primitive_ids: list[int]
    layer_ids: list[int]
    semantic_ids: list[int]
    instance_ids: list[int]
    primitive_lengths: list[float]
    # Optional: the US and CubiCasa converters leave out per-segment colours and
    # widths (constant there, and the model drops colour features anyway), which
    # keeps a 35k-window corpus several times smaller. Missing = black, width 1.
    colors: list[list[int]] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    # Text annotations (FloorPlanCAD-V2 <text> elements), kept for text-aware
    # models. Each: {"text", "x", "y", "size", "angle", "layer_id"}. Optional, so
    # JSONs written before this field existed still load.
    texts: list[dict] = field(default_factory=list)
    # US plan windows (dataset/to_lines_us.py): the CAD layer name behind each
    # layer id, and provenance (source, doc, page, scale, viewport). Optional.
    layer_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class SVGDataTensor:
    # N: number of sampled points or lines
    viewBox: torch.Tensor # shape is (4,), [minx, miny, width, height]
    coords: torch.Tensor # point mode: shape is (N, 2), [N * [x, y]]; line mode: shape is (N, 4), [N * [x1, y1, x2, y2]]
    colors: torch.Tensor # shape is (N, 3), [N * [r, g, b]]
    widths: torch.Tensor # shape is (N,), [N * [width]]
    primitive_ids: torch.Tensor # shape is (N,), [N * [primitive_id]]
    layer_ids: torch.Tensor # shape is (N,), [N * [layer_id]]
    semantic_ids: torch.Tensor # shape is (N,), [N * [semantic_id]]
    instance_ids: torch.Tensor # shape is (N,), [N * [instance_id]]
    primitive_lengths: torch.Tensor # shape is (N,), [N * [length]]


@dataclass
class VecData:
    # metadata
    data_path: str # file path for debug
    # --------------------- features --------------------- #
    # N: number of points or lines
    coords: torch.Tensor
    # coordinates of points or lines
    # point mode: shape is (N, 2), [N * [x, y]]
    # line mode: shape is (N, 4), [N * [x1, y1, x2, y2]]
    feats: torch.Tensor
    # features of points or lines
    # point mode: shape is (N, 7), [N * [cx, cy, pcx, pcy, color_r, color_g, color_b]]
    # line mode: shape is (N, 10), [N * [length, |dx|, |dy|, cx, cy, pcx, pcy, color_r, color_g, color_b]]
    prim_ids: torch.Tensor # shape is (N,), [N * [primitive_id]]
    layer_ids: torch.Tensor # shape is (N,), [N * [layer_id]]
    # ---------------------- labels ---------------------- #
    sem_ids: torch.Tensor # shape is (N,), [N * [semantic_id]]
    inst_ids: torch.Tensor # shape is (N,), [N * [instance_id]]
    prim_lengths: torch.Tensor # shape is (N,), [N * [length]]
    # ------------- text annotations (optional) ------------- #
    # T annotations; see data/floorplancad/text_features.py. None when text is off.
    text: Optional[dict] = None       # text_types (T,), text_attrs (T,3), text_grades (T,), text_masks (T,4), text_geo (T,3)
    text_pos: Optional[torch.Tensor] = None   # (T, 2), normalised and augmented with the lines


@dataclass
class VecDataTransformArgs:
    norm_range: tuple[float, float] = (-0.5, 0.5)     # range of normalized coordinates
    random_vertical_flip: float = 0.5                       # probability of vertical flip
    random_horizontal_flip: float = 0.5                     # probability of horizontal flip
    random_rotate: bool = True                              # whether to rotate
    random_scale: tuple[float, float] = (0.8, 1.2)          # scale range: (min_scale_ratio, max_scale_ratio)
    random_translation: tuple[float, float] = (0.1, 0.1)    # translation range: (x_translation_ratio, y_translation_ratio)
