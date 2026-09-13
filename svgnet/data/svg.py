"""
Dataset for panoptic symbol spotting on vector CAD drawings.

This module was missing from the published DPSS release; it is reconstructed
from the contract the rest of the codebase expects:

  * svgnet/model/svgnet.py:43 unpacks a 9-tuple batch:
        coords, feats, semantic_labels, offsets, lengths, layerIds,
        imgs, centers, json_file
  * tools/inference.py:88 calls the classmethod
        SVGDataset.load(data_root=..., file_name=..., idx=..., min_points=...)
    and expects 8 return values (see `load` below).
  * configs/svg/*.yaml sets model.in_channels: 10, so `feat` has 10 columns.

Each drawing is stored as a JSON file produced by dataset/parse_FpCAD_svg.py,
accompanied by a PNG render used by the image pathway. See `load` for the
exact schema.
"""

import json
import math
import os.path as osp
import random
import sys
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .aug_utils import (RandomHorizonFilp, RandomVerticalFilp, random_rotate,
                        rotate_xy)

# FloorPlanCAD's 35 annotated classes plus a background entry. Indices used by
# the model are 0-based (id - 1), so "single door" is 0 and background is 35.
SVG_CATEGORIES = [
    # 1-6: doors
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
    # 7-10: windows
    {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    # 11-27: furniture
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    # 28: stairs
    {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
    # 29-30: equipment
    {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},
    # 31-35: uncountable ("stuff") classes
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
    # background sentinel — excluded from the class list used for evaluation
    {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
]

# Number of real classes; also the semantic id reserved for background.
NUM_CLASSES = len(SVG_CATEGORIES) - 1  # 35
BG_SEMANTIC_ID = NUM_CLASSES           # 35

# Narrow taxonomy for door/window spotting on US construction documents. Kept
# here alongside SVG_CATEGORIES so evaluation can name classes for either
# dataset; dataset/taxonomy.py holds the parser-side mapping onto these indices.
# Wall is a context class, not a deliverable: openings are defined by the walls
# they sit in, so labelling walls helps the model find doors and windows.
US4_CATEGORIES = [
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "door"},
    {"color": [96, 78, 245], "isthing": 1, "id": 2, "name": "window"},
    {"color": [167, 92, 32], "isthing": 0, "id": 3, "name": "wall"},
    {"color": [0, 0, 0], "isthing": 0, "id": 4, "name": "bg"},
]

# The 43-class Arch taxonomy: FloorPlanCAD's 35 with their ids untouched, plus
# eight US construction-document families FloorPlanCAD has no class for. Defined
# in dataset/taxonomy.py because the parsers must be able to read it without
# importing torch, and duplicating it here is how the editor ended up painting
# "background" with the id for "folding door".
_DATASET_DIR = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
                        "dataset")
if _DATASET_DIR not in sys.path:
    sys.path.insert(0, _DATASET_DIR)
from taxonomy import ARCH_CATEGORIES  # noqa: E402

# Keyed by class count (excluding background), which is what the configs set as
# model.semantic_classes.
TAXONOMIES = {
    len(ARCH_CATEGORIES) - 1: ARCH_CATEGORIES,  # 43 — Arch-43 (FpCAD + US)
    len(SVG_CATEGORIES) - 1: SVG_CATEGORIES,   # 35 — FloorPlanCAD / ArchCAD
    len(US4_CATEGORIES) - 1: US4_CATEGORIES,   #  3 — door / window / wall
}


def get_categories(num_classes):
    """Category list for a given class count, background entry included.

    Always returns exactly num_classes + 1 entries. The old version returned the
    full 35-name list for any count it did not recognise, which silently
    misaligned every per-class metric for configs/svg/svg_pointT.yaml (30
    classes): point_wise_eval.py zips 35 names against a 30-slot confusion
    matrix, so each class was reported under a different class's name.
    """
    n = int(num_classes)
    known = TAXONOMIES.get(n)
    if known is not None:
        return known
    # Unknown count: take the first n classes of the widest taxonomy and give
    # them a background entry, so names and slots still line up.
    head = ARCH_CATEGORIES[:n]
    bg = dict(ARCH_CATEGORIES[-1])
    bg["id"] = n + 1
    return head + [bg]

# Per-primitive feature width. The point backbone concatenates the 3 xyz
# coordinates in front of this, so configs must set in_channels = 3 + FEAT_DIM.
FEAT_DIM = 7


def angles_with_horizontal(coords):
    """Angle of each primitive's chord relative to the x axis, in radians.

    `coords` is (N, 4, 2): four control points per primitive. The chord runs
    from the first to the last control point.
    """
    coords = np.array(coords).reshape(-1, 4, 2)
    start_points, end_points = coords[:, 0, :], coords[:, -1, :]
    x1, y1 = start_points[:, 0], start_points[:, 1]
    x2, y2 = end_points[:, 0], end_points[:, 1]

    slopes = (y2 - y1) / (x2 - x1 + 1e-8)
    slopes[x2 - x1 == 0] = np.inf
    return np.arctan(slopes)


class SVGDataset(Dataset):
    """Vector CAD drawings as point sets, paired with a raster render.

    Every SVG primitive becomes one "point": its position is the centroid of
    the primitive's control points, and its feature vector encodes orientation,
    length, colour, command type and stroke width.
    """

    CLASSES = tuple([x["name"] for x in SVG_CATEGORIES])

    def __init__(self, data_root, split, data_norm, aug, img_size=980,
                 repeat=1, split_path=None, num_classes=NUM_CLASSES, logger=None,
                 use_corrections=True, coarse_policy="bg", source=None, annotated=None,
                 stuff_classes=None, index_base=0, max_samples=0):
        self.data_root = data_root
        self.split = split
        self.data_norm = data_norm
        self.aug = aug
        self.img_size = img_size
        self.repeat = repeat
        # Background id follows the class count, so the same loader serves the
        # 35-class FloorPlanCAD data and the 3-class door/window taxonomy.
        self.num_classes = int(num_classes)
        # Set data.*.use_corrections: False to train against the raw
        # layer-derived labels, which is the ablation for 'did hand
        # correction help'.
        self.use_corrections = bool(use_corrections)
        self.coarse_policy = str(coarse_policy)

        self.data_list = sorted(glob(osp.join(data_root, split, "*_s2.json")))
        if not self.data_list:  # tolerate a flat directory with no split subdir
            self.data_list = sorted(glob(osp.join(data_root, "*_s2.json")))

        if logger is not None:
            logger.info(f"Load {split} dataset: {len(self.data_list)} svg")
        elif not self.data_list:
            raise FileNotFoundError(
                f"No '*_s2.json' files found under {data_root} (split={split}). "
                "Run dataset/parse_FpCAD_svg.py first."
            )

        # A fixed, evenly spaced subset -- for per-epoch validation on a sample
        # rather than the full test set. Deterministic, so epochs compare.
        if max_samples and len(self.data_list) > int(max_samples):
            keep = np.linspace(0, len(self.data_list) - 1, int(max_samples)).round().astype(int)
            self.data_list = [self.data_list[i] for i in keep]
            if logger is not None:
                logger.info(f"  {source or data_root}: evaluating a fixed subset of "
                            f"{len(self.data_list)} drawings")

        # Multi-source training (svgnet/data/multi.py). `index_base` keeps the
        # per-sample instance/layer id offsets unique across sources; `annotated`
        # is a class-id list or a preset name from dataset/taxonomy.py ANNOTATED;
        # `stuff_classes` ("default" = taxonomy.STUFF_CLASSES) forces one mask per
        # class, FloorPlanCAD's convention for walls and railings.
        self.source = source
        self.index_base = int(index_base)
        if isinstance(annotated, str):
            from taxonomy import ANNOTATED
            annotated = ANNOTATED[annotated]
        if isinstance(stuff_classes, str):
            from taxonomy import STUFF_CLASSES
            stuff_classes = STUFF_CLASSES
        self.stuff_classes = list(stuff_classes) if stuff_classes else None
        self.annotated = None
        if annotated is not None:
            m = torch.zeros(self.num_classes + 1, dtype=torch.bool)
            for c in annotated:
                if 0 <= int(c) < self.num_classes:
                    m[int(c)] = True
            m[self.num_classes] = True
            self.annotated = m

        self.data_idx = np.arange(len(self.data_list))
        self.instance_queues = []

    def __len__(self):
        return len(self.data_list) * self.repeat

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _canon_dir(json_file):
        """Directory holding the durable copy of this tile's sidecars.

        dataset/split_us_plans.py rebuilds train/ and test/ with rmtree, and on
        this checkout it falls back to copyfile rather than os.link, so anything
        written beside a split copy is both a private duplicate and one re-split
        away from deletion. all/ is the copy that survives, so corrections are
        read from there whenever the corpus has one.
        """
        split_dir = osp.dirname(osp.abspath(json_file))
        alld = osp.join(osp.dirname(split_dir), "all")
        return alld if osp.isdir(alld) else split_dir

    @staticmethod
    def _sidecar(json_file, kind):
        """Path to a per-tile sidecar, preferring the durable copy under all/."""
        base = osp.basename(json_file).replace("_s2.json", f"_s2.{kind}.json")
        canon = osp.join(SVGDataset._canon_dir(json_file), base)
        if osp.exists(canon):
            return canon
        return osp.join(osp.dirname(osp.abspath(json_file)), base)

    @staticmethod
    def _apply_corrections(json_file, data, sem, num):
        """Fold human corrections over the parser's layer-derived labels.

        Two kinds, weakest first, because the brush is the exception handler for
        the rule:

          1. layermap  {layer name: class}   -- "everything on A-FLOR-PFIX is a
             toilet". Keyed by CAD layer name, not primitive index, so one
             decision covers every tile of a plan set and survives a re-parse
             that renumbers primitives. A project-wide file applies first, then
             the tile's own.
          2. override  {primitive index: class} -- a single painted primitive.

        Missing or malformed sidecars are ignored: a corrupt correction must not
        take a training run down.
        """
        names = data.get("layerNames") or []
        if names:
            rules = {}
            proj = osp.join(SVGDataset._canon_dir(json_file),
                            "_project_layermap.json")
            for path in (proj, SVGDataset._sidecar(json_file, "layermap")):
                try:
                    with open(path) as fh:
                        rules.update(json.load(fh).get("rules", {}))
                except (OSError, ValueError):
                    pass
            if rules:
                lut = np.array([rules.get(n, -1) for n in names], dtype=np.int64)
                lids = np.asarray(data["layerIds"], dtype=np.int64)[:num]
                # Guard: a tile whose layerIds outrun its layerNames is
                # malformed; leave it alone rather than index out of bounds.
                if lids.size and int(lids.max()) < lut.size:
                    hit = lut[lids] >= 0
                    sem = np.where(hit, lut[lids], sem)

        try:
            with open(SVGDataset._sidecar(json_file, "override")) as fh:
                over = json.load(fh)
        except (OSError, ValueError):
            return sem
        for k, v in over.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= i < num:
                sem[i] = int(v)
        return sem

    @staticmethod
    def load(data_root=None, file_name=None, idx=0, min_points=2048,
             json_file=None, img_size=None, num_classes=NUM_CLASSES,
             use_corrections=True, coarse_policy="bg", stuff_classes=None):
        """Read one drawing.

        Accepts either an explicit `json_file`, or `data_root` + `file_name`
        (the form tools/inference.py uses, where file_name has no "_s2" suffix).

        Expected JSON keys:
            width, height   drawing extent in SVG units
            args            (N, 8) four control points per primitive
            lengths         (N,)   arc length of each primitive
            commands        (N,)   command type index in [0, 4)
            widths          (N,)   stroke width
            rgb             (N, 3) stroke colour, 0-255
            semanticIds     (N,)   class index, background = 35
            instanceIds     (N,)   instance index, -1 for stuff/background
            layerIds        (N,)   CAD layer index

        Returns:
            coord      (P, 3) float, xy normalised to [0, 1], z = 0
            feat       (P, 10) float
            label      (P, 2) int, columns are (semantic, instance)
            lengths    (P,) float
            layerIds   (P,) int
            img        PIL.Image of the drawing
            bound      (width, height) of the source drawing
            json_file  path that was read
        P = max(N, min_points); rows beyond N are zero-padded background.
        """
        if json_file is None:
            if data_root is None or file_name is None:
                raise ValueError("provide either json_file, or data_root and file_name")
            json_file = osp.join(data_root, f"{file_name}_s2.json")

        with open(json_file) as f:
            data = json.load(f)

        width, height = data["width"], data["height"]
        raw_coords = data["args"]
        num = len(raw_coords)
        if num == 0:
            raise ValueError(f"{json_file} contains no primitives")

        arcs = angles_with_horizontal(raw_coords)

        coords = np.array(raw_coords, dtype=np.float64).reshape(-1, 8)
        coords[:, 0::2] = coords[:, 0::2] / width      # x -> [0, 1]
        coords[:, 1::2] = coords[:, 1::2] / height     # y -> [0, 1]

        # Pad up to min_points so downsampling stages always have enough points.
        max_num = max(num, min_points)

        coord = np.zeros((max_num, 3))
        coord[:num, 0] = np.mean(coords[:, 0::2], axis=1)   # centroid x
        coord[:num, 1] = np.mean(coords[:, 1::2], axis=1)   # centroid y
        # coord[:, 2] stays 0 — drawings are planar

        lengths = np.zeros(max_num)
        lengths[:num] = np.array(data["lengths"])

        # Feature layout — 7 channels:
        #   0     chord angle
        #   1     normalised arc length
        #   2:6   one-hot command type
        #   6     normalised stroke width
        #
        # Note this is 7, not model.in_channels (10). The backbone prepends the
        # 3 xyz coordinates itself (svgnet/model/pointtransformer.py:60), so
        # in_channels = 3 + FEAT_DIM. Adding the rgb channels parsed below would
        # mean bumping in_channels to 13.
        feat = np.zeros((max_num, FEAT_DIM))
        max_length = max(width, height)
        norm_lens = np.array(data["lengths"]).clip(0, max_length) / max_length
        ctype = np.eye(4)[np.asarray(data["commands"]).astype(int).clip(0, 3)]
        widths = np.array(data["widths"], dtype=np.float64)
        width_denom = np.max(widths) if np.max(widths) > 0 else 1.0

        feat[:num, 0] = arcs
        feat[:num, 1] = norm_lens
        feat[:num, 2:6] = ctype
        feat[:num, 6] = widths / width_denom

        # Padded rows are background: semantic = num_classes, instance = -1.
        semanticIds = np.full(max_num, int(num_classes))
        semanticIds[:num] = np.array(data["semanticIds"])
        semanticIds = semanticIds.astype(np.int64)

        # Human corrections from tools/label_editor.py. Until this existed,
        # nothing read them: the editor wrote <stem>_s2.override.json and every
        # training run loaded the raw semanticIds straight past it, so hand
        # correction had literally no effect on the model.
        if use_corrections:
            semanticIds[:num] = SVGDataset._apply_corrections(
                json_file, data, semanticIds[:num], num)

        # Coarse labels (band C: door-any, furniture-any, ...) say which family a
        # primitive belongs to but not which member. They live above background
        # by construction, so every `>= num_classes` guard in the model already
        # treats them as background -- but "already treats them as" is not a
        # contract, so make it one here and let the config say which it wants:
        #
        #   "bg"        collapse to background. The default, and the only
        #               correct choice until the marginal loss lands: the
        #               classifier head has no column for a coarse id.
        #   "canonical" replace with the group's modal member. Cheap, and a lie
        #               for every non-modal member.
        #   "keep"      pass through, for the marginal loss.
        if coarse_policy != "keep":
            over = semanticIds > int(num_classes)
            if over.any():
                if coarse_policy == "canonical":
                    from taxonomy import canonical
                    for i in np.flatnonzero(over):
                        c = canonical(int(semanticIds[i]))
                        semanticIds[i] = c if c < int(num_classes) else int(num_classes)
                else:
                    semanticIds[over] = int(num_classes)

        instanceIds = np.full(max_num, -1)
        ins = np.array(data["instanceIds"]).astype(np.int64)
        if stuff_classes:
            ins[np.isin(semanticIds[:num], stuff_classes)] = -1
        # Offset instance ids per sample so they stay unique after batching.
        valid = ins != -1
        ins[valid] += idx * min_points
        instanceIds[:num] = ins
        instanceIds = instanceIds.astype(np.int64)

        label = np.concatenate([semanticIds[:, None], instanceIds[:, None]], axis=1)

        layerIds = np.array(data["layerIds"]).astype(np.int64)
        bg_layerId = int(np.max(layerIds)) + 1 if layerIds.size else 0
        pad_layerIds = np.full(max_num, bg_layerId)
        pad_layerIds[:num] = layerIds
        pad_layerIds = pad_layerIds.astype(np.int64) + idx * min_points

        img = SVGDataset._load_image(json_file, data, img_size)

        return coord, feat, label, lengths, pad_layerIds, img, (width, height), json_file

    @staticmethod
    def _load_image(json_file, data, img_size=None):
        """Companion PNG render for the image pathway.

        Falls back to a blank white canvas when the render is absent, so the
        vector pathway can still be exercised on its own.
        """
        candidates = []
        if data.get("image"):
            candidates.append(osp.join(osp.dirname(json_file), data["image"]))
        stem = osp.splitext(json_file)[0]
        candidates.append(stem + ".png")
        candidates.append(stem.replace("_s2", "") + ".png")

        for path in candidates:
            if osp.isfile(path):
                return Image.open(path).convert("RGB")

        size = img_size or 980
        return Image.new("RGB", (size, size), (255, 255, 255))

    # ------------------------------------------------------------------ #
    # sample assembly
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        json_file = self.data_list[data_idx]
        coord, feat, label, lengths, layerIds, img, _, _ = SVGDataset.load(
            json_file=json_file, idx=idx + getattr(self, "index_base", 0), img_size=self.img_size,
            num_classes=self.num_classes,
            use_corrections=self.use_corrections,
            coarse_policy=self.coarse_policy,
            stuff_classes=getattr(self, "stuff_classes", None),
        )

        if self.split == "train" and self.aug:
            out = self.transform_train(coord, feat, label, lengths, layerIds, img, json_file)
        else:
            out = self.transform_test(coord, feat, label, lengths, layerIds, img, json_file)
        meta = {"source": getattr(self, "source", None),
                "annotated": getattr(self, "annotated", None)}
        return out[:-1] + (meta, out[-1])

    def _finalize(self, coord, feat, label, lengths, layerIds, img, json_file):
        """Shared tail: derive image sampling centres, then normalise coords.

        `centers` must be computed *before* re-centring, because grid_sample
        expects [-1, 1] coordinates relative to the rendered image.
        """
        centers = coord[:, :2].copy() * 2 - 1

        if self.data_norm == "mean":
            coord = coord - np.mean(coord, 0)
        elif self.data_norm == "min":
            coord = coord - np.min(coord, 0)

        img_tensor = self.img_transform(img)

        return (
            torch.FloatTensor(coord),
            torch.FloatTensor(feat),
            torch.LongTensor(label),
            torch.FloatTensor(lengths),
            torch.LongTensor(layerIds),
            img_tensor,
            torch.FloatTensor(centers),
            json_file,
        )

    @property
    def img_transform(self):
        # Built lazily so the dataset stays picklable across dataloader workers.
        if not hasattr(self, "_img_transform"):
            import torchvision.transforms as T
            self._img_transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return self._img_transform

    def transform_train(self, coord, feat, label, lengths, layerIds, img, json_file):
        aug = self.aug

        if aug.hflip and np.random.rand() < aug.aug_prob:
            coord[:, :2] = RandomHorizonFilp(coord[:, :2], width=1)

        if aug.vflip and np.random.rand() < aug.aug_prob:
            coord[:, :2] = RandomVerticalFilp(coord[:, :2], Hight=1)

        if aug.rotate.enable and np.random.rand() < aug.aug_prob:
            _min, _max = aug.rotate.angle
            coord[:, :2] = rotate_xy(coord[:, :2], width=1, height=1,
                                     angle=random.uniform(_min, _max))

        if aug.rotate2 and np.random.rand() < aug.aug_prob:
            coord[:, :2] = random_rotate(coord[:, :2], width=1, height=1)

        if aug.shift.enable and np.random.rand() < aug.aug_prob:
            _min, _max = aug.shift.scale
            shift = np.random.uniform(_min, _max, 3)
            shift[2] = 0
            coord = coord + shift

        if aug.scale.enable and np.random.rand() < aug.aug_prob:
            _min, _max = aug.scale.ratio
            scale = np.random.uniform(_min, _max, 1)
            coord = coord * scale
            feat[:, 1] = feat[:, 1] * scale

        mix_coord, mix_feat = [coord], [feat]
        mix_label, mix_layerid, mix_lengths = [label], [layerIds], [lengths]

        # CutMix: keep a rolling queue of instances and paste them in at a
        # random offset. Only "thing" instances are queued.
        if aug.cutmix.enable and np.random.rand() < aug.aug_prob:
            for sem, ins in np.unique(label, axis=0):
                if sem >= self.num_classes:
                    continue
                valid = np.logical_and(label[:, 0] == sem, label[:, 1] == ins)
                if len(self.instance_queues) <= aug.cutmix.queueK:
                    self.instance_queues.insert(0, {
                        "coord": coord[valid],
                        "feat": feat[valid],
                        "label": label[valid],
                        "layerid": layerIds[valid],
                        "lengths": lengths[valid],
                    })
                else:
                    self.instance_queues.pop()

            _min, _max = aug.cutmix.relative_shift
            rand_pos = np.random.uniform(_min, _max, 3)
            rand_pos[2] = 0
            for instance in self.instance_queues:
                mix_coord.append(instance["coord"] + rand_pos)
                mix_feat.append(instance["feat"])
                mix_label.append(instance["label"])
                mix_layerid.append(instance["layerid"])
                mix_lengths.append(instance["lengths"])

        coord = np.concatenate(mix_coord, axis=0)
        feat = np.concatenate(mix_feat, axis=0)
        label = np.concatenate(mix_label, axis=0)
        layerIds = np.concatenate(mix_layerid)
        lengths = np.concatenate(mix_lengths)

        # Shuffle so the point order carries no information.
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]
        label, layerIds, lengths = label[shuf_idx], layerIds[shuf_idx], lengths[shuf_idx]

        return self._finalize(coord, feat, label, lengths, layerIds, img, json_file)

    def transform_test(self, coord, feat, label, lengths, layerIds, img, json_file):
        return self._finalize(coord, feat, label, lengths, layerIds, img, json_file)

    # ------------------------------------------------------------------ #
    # batching
    # ------------------------------------------------------------------ #
    def collate_fn(self, batch):
        """Assemble the 10-tuple that SVGNet.forward unpacks (meta before names).

        Point tensors are concatenated into one flat array and delimited by
        `offsets`; images and centres stay as per-sample lists because the
        image pathway indexes them individually.
        """
        coord, feat, label, lengths, layerIds, imgs, centers, metas, json_file = list(zip(*batch))

        # Batch metadata for the loss. A class counts as annotated only if every
        # drawing in the batch annotates it: a query can land on any of them.
        masks = [m["annotated"] for m in metas if m["annotated"] is not None]
        annotated = None
        if masks:
            annotated = masks[0].clone()
            for m in masks[1:]:
                annotated &= m
        meta = {"source": [m["source"] for m in metas], "annotated": annotated}

        offset, count = [], 0
        for item in coord:
            count += item.shape[0]
            offset.append(count)

        return (
            torch.cat(coord),
            torch.cat(feat),
            torch.cat(label),
            torch.IntTensor(offset),
            torch.cat(lengths),
            torch.cat(layerIds),
            list(imgs),
            list(centers),
            meta,
            list(json_file),
        )
