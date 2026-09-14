import os
import json
from typing import Dict, Any

import torch
from torch.utils.data import Dataset

from utils.svg_util import scan_dir
from .dataclass_define import (
    SVGData,
    VecData,
    VecDataTransformArgs
)
from .text_features import encode_texts
from .transform_utils import (
    to_tensor,
    norm_coords,
    augment_line_args,
    to_vec_data
)


class FloorPlanCAD(Dataset):

    def __init__(self, root_dir: str, split: str,
                 train_transform_args: Dict[str, Any],
                 eval_transform_args: Dict[str, Any],
                 use_text: bool = False,
                 max_texts: int = 512,
                 source_id: int = -1,
                 background_remap: list = None):
        self.root_dir = root_dir
        self.split = split
        # Text annotations for the TextCAD-style model. Off by default, which keeps
        # the upstream VecFormer data path unchanged.
        self.use_text = use_text
        self.max_texts = max_texts
        # Joint training: which source this corpus is (index into the model config's
        # sources; -1 = not given, and batches carry no source_ids), and an optional
        # [from, to] background id remap, e.g. [35, 43] to put FloorPlanCAD's
        # 35-class files into the Arch-43 label space (whose id 35 is a real class).
        self.source_id = source_id
        self.background_remap = background_remap
        self.train_transform_args = train_transform_args
        self.eval_transform_args = eval_transform_args
        self.data_dir = os.path.join(root_dir, split)
        self.data_paths = scan_dir(self.data_dir, suffix=".json")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # ------------- load origin json data ------------ #
        data_path = os.path.join(self.data_dir, self.data_paths[idx])
        json_data = json.load(open(data_path, "r"))
        # ------ dict to data class for easy access ------ #
        svg_data = SVGData(**json_data)
        # -------- transform to line data (tensor) ------- #
        transform_args = self._get_transform_args()
        vec_data = self._transform(svg_data,
                                   VecDataTransformArgs(**transform_args))
        vec_data.data_path = data_path
        vec_data.source_id = self.source_id
        if self.background_remap:
            src, dst = self.background_remap
            vec_data.sem_ids = torch.where(vec_data.sem_ids == src, torch.full_like(vec_data.sem_ids, dst), vec_data.sem_ids)
        return vec_data

    def _get_transform_args(self) -> Dict[str, Any]:
        if self.split == "train":
            return self.train_transform_args
        elif self.split == "val" or self.split == "test":
            return self.eval_transform_args
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _transform(self, data: SVGData, transform_args: VecDataTransformArgs) -> VecData:
        # -------------- transform to tensor ------------- #
        svg_data_tensor = to_tensor(data)
        # transform all line coords from [N * [x1, y1, x2, y2]] to [N * 2 * [x, y]] for easier processing
        if svg_data_tensor.coords.shape[-1] == 4:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 2, 2)
        # Text positions ride along as zero-length lines, so normalisation and every
        # random flip / rotation / scale / translation hits them exactly as it hits
        # the geometry; they are split off again before features are computed.
        text, n_lines = None, svg_data_tensor.coords.shape[0]
        if self.use_text and svg_data_tensor.coords.dim() == 3:
            encoded = encode_texts(data.texts, data.viewBox, self.max_texts)
            if encoded is not None:
                text, text_pos = encoded
                svg_data_tensor.coords = torch.cat(
                    [svg_data_tensor.coords, text_pos.unsqueeze(1).repeat(1, 2, 1)], dim=0)
        # ---------------- normalize lines --------------- #
        svg_data_tensor.coords = norm_coords(
            coords=svg_data_tensor.coords,
            bbox=svg_data_tensor.viewBox,
            min_val=transform_args.norm_range[0],
            max_val=transform_args.norm_range[1])
        # ----------------- augment lines ---------------- #
        svg_data_tensor.coords = augment_line_args(
            coords=svg_data_tensor.coords,
            min_val=transform_args.norm_range[0],
            max_val=transform_args.norm_range[1],
            transform_args=transform_args)
        text_pos = None
        if text is not None:
            text_pos = svg_data_tensor.coords[n_lines:, 0, :].clone()
            svg_data_tensor.coords = svg_data_tensor.coords[:n_lines]
        # ------------ transform to line data ------------ #
        # transform all line coords back to [N * [x1, y1, x2, y2]]
        if len(svg_data_tensor.coords.shape) == 3:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 4)
        vec_data = to_vec_data(svg_data_tensor)
        vec_data.text = text
        vec_data.text_pos = text_pos
        return vec_data


    @staticmethod
    def collate_fn(
        batch: list[VecData]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for FloorPlanCAD dataset, concatenate variable length sequences in the batch.

        Args:

            `batch`: List of `VecData` objects with variable length sequences

        Returns:
            Dict containing concatenated tensors:
                # Inputs
                `coords`: (N1+N2+..., coords_dim) - Concatenated coordinates
                `feats`: (N1+N2+..., feats_dim) - Concatenated features
                `prim_ids`: (N1+N2+...,) - Concatenated primitive IDs
                `layer_ids`: (N1+N2+...,) - Concatenated layer IDs
                `cu_seqlens`: (B+1,) - The cumulative sequence lengths of the sequences in the batch
                # Labels
                `sem_ids`: (N1+N2+...,) - Concatenated semantic IDs
                `inst_ids`: (N1+N2+...,) - Concatenated instance IDs
                `prim_lengths`: (N1+N2+...,) - Concatenated primitive lengths
                `cu_numprims`: (B+1,) - The cumulative number of primitives in each sequence
            B is the batch size,
            N1, N2, ... are lengths of sequences of each element in the batch.

        Raises:
            ValueError: If batch is empty or contains invalid data
        """
        if not batch:
            raise ValueError("Batch cannot be empty")

        # Define fields to extract and pad
        fields = ['coords', 'feats', 'prim_ids', 'layer_ids', 'sem_ids', 'inst_ids', 'prim_lengths']

        # Extract sequences for each field
        sequences = {
            field: [getattr(item, field) for item in batch]
            for field in fields
        }

        # Calculate the cumulative sequence lengths and the number of primitives
        seq_lens = torch.tensor([len(seq) for seq in sequences['coords']], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)])
        num_prim = torch.tensor([len(seq) for seq in sequences['sem_ids']], dtype=torch.int32)
        cu_numprims = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(num_prim, dim=0, dtype=torch.int32)])

        # Concatenate all sequences
        concat_data = {
            # Inputs
            'coords': torch.cat(sequences['coords'], dim=0),
            'feats': torch.cat(sequences['feats'], dim=0),
            'prim_ids': torch.cat(sequences['prim_ids'], dim=0),
            'layer_ids': torch.cat(sequences['layer_ids'], dim=0),
            'cu_seqlens': cu_seqlens,
            # Labels
            'sem_ids': torch.cat(sequences['sem_ids'], dim=0),
            'inst_ids': torch.cat(sequences['inst_ids'], dim=0),
            'prim_lengths': torch.cat(sequences['prim_lengths'], dim=0),
            'cu_numprims': cu_numprims,
            # Data paths
            'data_paths': [item.data_path for item in batch]
        }

        if any(item.source_id >= 0 for item in batch):
            concat_data["source_ids"] = torch.tensor([item.source_id for item in batch], dtype=torch.long)

        # Text annotations, only when the dataset was built with use_text. Every
        # drawing contributes zero or more rows; text_cu_seqlens delimits them.
        if any(item.text is not None for item in batch):
            keys = ("text_types", "text_attrs", "text_grades", "text_masks", "text_geo")
            empty = dict(text_types=torch.zeros(0, dtype=torch.long), text_attrs=torch.zeros(0, 3),
                         text_grades=torch.zeros(0, dtype=torch.long), text_masks=torch.zeros(0, 4, dtype=torch.bool),
                         text_geo=torch.zeros(0, 3))
            parts = [item.text if item.text is not None else empty for item in batch]
            for k in keys:
                concat_data[k] = torch.cat([p[k] for p in parts], dim=0)
            concat_data["text_pos"] = torch.cat(
                [item.text_pos if item.text_pos is not None else torch.zeros(0, 2) for item in batch], dim=0)
            n_text = torch.tensor([len(p["text_types"]) for p in parts], dtype=torch.int32)
            concat_data["text_cu_seqlens"] = torch.cat(
                [torch.tensor([0], dtype=torch.int32), torch.cumsum(n_text, dim=0, dtype=torch.int32)])

        return concat_data
