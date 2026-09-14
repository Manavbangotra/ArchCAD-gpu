#!/usr/bin/env python3
"""Plan-set PDF -> takeoff JSON.

    # with a trained model
    python tools/takeoff.py --pdf plans.pdf --config configs/svg/svg_pointT_joint43_3060.yaml \
        --checkpoint work_dirs/joint43/best.pth --out plans.takeoff.json

    # with the line model (VecFormer/TextCAD) over 10 m sliding windows
    python tools/takeoff.py --pdf plans.pdf --labels vecformer         --model_config vecformer/configs/model/product_arch43.yaml         --checkpoint vecformer/outputs/product/checkpoint-best --out plans.takeoff.json

    # without one: the CAD layers' own labels (a baseline, and a pipeline check)
    python tools/takeoff.py --pdf plans.pdf --labels layers --out plans.layers.json

    # add door/window sizes from Schedule-detection's output for the same set
    ... --schedules ../Schedule-detection/out/plansets/project_461_aaa865c6.json

Per page: parse the vector primitives, cut the same overlapping tiles training
used, predict each tile, stitch the tiles back into one object per thing on
the page (takeoff/stitch.py), then measure: scale per viewport, openings with
type marks, rooms from walls and room tags. Totals count plan viewports only.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
import time

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "dataset"))

import numpy as np  # noqa: E402

from takeoff import document, openings as openings_mod, rooms as rooms_mod  # noqa: E402
from takeoff.scale import page_scales  # noqa: E402
from takeoff.stitch import objects_from_labels, split_connected, stitch  # noqa: E402


def text_layer(doc_page, box, origin):
    """Words and lines as (text, x, y) / (text, bbox) in page-local PDF points, y up.

    PyMuPDF reports text in the unrotated page frame, y down, relative to the
    box corner; parse_page's primitives are PDF user space shifted by the
    drawing's origin. See classify_viewports.to_pdf_space.
    """
    ox, oy = origin

    def xy(x, y):
        return x + box[0] - ox, box[3] - y - oy

    words = []
    for x0, y0, x1, y1, w, *_ in doc_page.get_text("words"):
        cx, cy = xy((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        words.append((w, cx, cy))
    lines = []
    for block in doc_page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line.get("spans", [])).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            ax, ay = xy(x0, y1)
            bx, by = xy(x1, y0)
            lines.append((text, (ax, ay, bx, by)))
    return words, lines


def sheet_number(doc_page):
    """The sheet's own number ("A4.2"), printed largest in the title block."""
    from classify_viewports import SHEET_NO
    best = (0.0, "")
    for block in doc_page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(sp["text"] for sp in spans).strip()
            if spans and SHEET_NO.match(text):
                size = max(sp["size"] for sp in spans)
                if size > best[0]:
                    best = (size, text)
    return best[1]


class TileModel:
    """The trained network, fed tiles exactly the way training fed them."""

    def __init__(self, config, checkpoint, device=None):
        import torch
        import yaml
        from munch import Munch

        from svgnet.model.svgnet import SVGNet
        from svgnet.util import get_device, get_root_logger, load_checkpoint

        self.torch = torch
        if device:
            # cuda_cast moves every model input to get_device(), which reads this.
            os.environ["ARCHCAD_DEVICE"] = str(device)
        with open(config, encoding="utf-8") as f:
            self.cfg = Munch.fromDict(yaml.safe_load(f))
        self.device = torch.device(device) if device else get_device()
        self.model = SVGNet(self.cfg.model, criterion=None)
        load_checkpoint(checkpoint, get_root_logger(), self.model)
        self.model.to(self.device).eval()
        test = self.cfg.data.get("test") or {}
        self.img_size = int(test.get("img_size", 700))
        self.num_classes = int(self.cfg.model.semantic_classes)
        import torchvision.transforms as T
        self.tf = T.Compose([T.Resize((self.img_size, self.img_size)), T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def predict(self, json_file):
        from svgnet.data.svg import SVGDataset
        torch = self.torch
        coord, feat, label, lengths, layer_ids, img, _, _ = SVGDataset.load(
            json_file=json_file, idx=0, img_size=self.img_size, num_classes=self.num_classes,
            use_corrections=False, coarse_policy="bg")
        centers = coord[:, :2].copy() * 2 - 1
        coord = coord - np.mean(coord, 0)
        batch = (torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label),
                 torch.IntTensor([coord.shape[0]]), torch.FloatTensor(lengths),
                 torch.LongTensor(layer_ids), [self.tf(img)], [torch.FloatTensor(centers)],
                 json_file)
        fp16 = bool(self.cfg.get("fp16")) and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(self.device.type, enabled=fp16):
            res = self.model(batch, return_loss=False)
        return res["instances"]


def page_objects(data, pdf_path, page_no, model, tmp, skip_full=True, img_size=980):
    """PageObjects for one parsed page, from the model over overlapping tiles."""
    from parse_pdf_plans import crop_tile_image, render_page, tile_sheet

    png = render_page(pdf_path, page_no, osp.join(tmp, f"page{page_no}.png"), img_size * 3)
    preds, n_tiles = [], 0
    tiles = list(tile_sheet(data, 6000, overlap=0.15, emit_page_max=40000, taxonomy="arch"))
    if skip_full and len(tiles) > 1:
        tiles = [(s, t) for s, t in tiles if s != "_full"]
    for suffix, tile in tiles:
        idxs = tile.pop("idxs")
        stem = osp.join(tmp, f"p{page_no}{suffix or '_page'}_s2")
        if png:
            if crop_tile_image(png, tile, stem + ".png", img_size):
                tile["image"] = osp.basename(stem + ".png")
        for k in ("origin", "page_box"):
            tile.pop(k, None)
        with open(stem + ".json", "w") as f:
            json.dump(tile, f)
        preds.append((suffix, idxs, model.predict(stem + ".json")))
        n_tiles += 1
    objects, _ = stitch(len(data["args"]), preds)
    return objects, n_tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", choices=("model", "layers", "vecformer"), default="model",
                    help="model: the SVGNet tile model; vecformer: the line model over real-size sliding "
                         "windows (takeoff/vecformer_model.py); layers: CAD-layer labels, no model")
    ap.add_argument("--config")
    ap.add_argument("--model_config", help="vecformer model YAML, e.g. vecformer/configs/model/product_arch43.yaml")
    ap.add_argument("--window_m", type=float, default=10.0, help="vecformer window side in metres")
    ap.add_argument("--cpu_kernels", action="store_true",
                    help="vecformer without flash-attn/spconv (pure torch; slow, for machines without CUDA)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--pages", default="", help="e.g. 3-5,9 (1-based); default all")
    ap.add_argument("--schedules", help="Schedule-detection planset JSON for this PDF")
    # Overall floor plans of a multi-storey building run to 300k+ primitives;
    # they are exactly the sheets a takeoff needs, and parse in seconds now.
    ap.add_argument("--max_page_prims", type=int, default=800000)
    ap.add_argument("--page_time_budget", type=float, default=120.0,
                    help="seconds to spend parsing one sheet's vectors before skipping it")
    ap.add_argument("--device")
    ap.add_argument("--overlay_dir", help="write a QA picture per page here")
    a = ap.parse_args()

    import pikepdf
    import pymupdf
    from classify_viewports import boilerplate, nearest_title, page_box, page_titles
    from parse_pdf_plans import parse_page

    model, model_info = None, {"labels": a.labels}
    if a.labels == "model":
        if not (a.config and a.checkpoint):
            ap.error("--labels model needs --config and --checkpoint (or use --labels layers)")
        model = TileModel(a.config, a.checkpoint, a.device)
        model_info.update(config=a.config, checkpoint=a.checkpoint)
    elif a.labels == "vecformer":
        if not (a.model_config and a.checkpoint):
            ap.error("--labels vecformer needs --model_config and --checkpoint")
        from takeoff.vecformer_model import WindowModel
        model = WindowModel(a.model_config, a.checkpoint, a.device, cpu_kernels=a.cpu_kernels)
        model_info.update(model_config=a.model_config, checkpoint=a.checkpoint, window_m=a.window_m)
    schedule = openings_mod.load_schedule(a.schedules) if a.schedules else None

    wanted = set()
    for part in filter(None, a.pages.split(",")):
        lo, _, hi = part.partition("-")
        wanted.update(range(int(lo), int(hi or lo) + 1))

    pdf = pikepdf.open(a.pdf)
    doc = pymupdf.open(a.pdf)
    skip = boilerplate(doc)
    tmp = tempfile.mkdtemp(prefix="takeoff_")
    pages, warnings = [], []
    t_start = time.time()
    try:
        for i, page in enumerate(list(pdf.pages), start=1):
            if wanted and i not in wanted:
                continue
            try:
                data = parse_page(pdf, page, i, taxonomy="arch", time_budget=a.page_time_budget)
            except Exception as exc:
                warnings.append(f"page {i}: not parsed ({exc})")
                continue
            n = len(data["args"])
            if a.max_page_prims and n > a.max_page_prims:
                warnings.append(f"page {i}: {n} primitives, over --max_page_prims; skipped")
                continue

            box = page_box(page)
            origin = data["origin"]
            words, lines = text_layer(doc[i - 1], box, origin)
            titles = [(k, x - origin[0], y - origin[1], t)
                      for k, x, y, t in page_titles(doc[i - 1], box, skip)]

            def viewport_of(x, y, titles=titles):
                hit = nearest_title(x, y, titles) if titles else None
                return hit if hit else ("untitled", "")

            scales = page_scales([(t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for t, b in lines], words)

            if model is None:
                objects = objects_from_labels(data["semanticIds"], data["instanceIds"], bg_id=43)
                n_tiles = 0
            elif a.labels == "vecformer":
                from takeoff.vecformer_model import page_objects_swa
                objects, n_tiles = page_objects_swa(data, lines, scales.at, viewport_of, model, window_m=a.window_m)
            else:
                objects, n_tiles = page_objects(data, a.pdf, i, model, tmp)

            args = data["args"]
            # Same tolerance the parser clusters layer instances with.
            objects = split_connected(objects, args, document.BARRIER_CLASSES - document.WINDOW_CLASSES,
                                      0.004 * max(data["width"], data["height"]))
            ops = openings_mod.build_openings(objects, args, document.OPENING_KINDS, scales.at,
                                              words, schedule)

            # Rooms per drawing scale: the door-width seal is in real millimetres,
            # so drawings at different scales need their own raster. Grouping by
            # title instead split one plan's walls between neighbouring titles.
            groups, door_boxes = {}, []
            for ob in objects:
                if ob.label in document.DOOR_CLASSES:
                    door_boxes.append(openings_mod.bbox_of(args, ob.prims))
                if ob.label not in document.BARRIER_CLASSES:
                    continue
                pts = np.asarray([args[j] for j in ob.prims], dtype=np.float64).reshape(-1, 8)
                cx, cy = float(pts[:, 0::2].mean()), float(pts[:, 1::2].mean())
                if viewport_of(cx, cy)[0] == "drop":
                    continue
                mm_per_pt = round(scales.at(cx, cy).mm_per_pt, 6)
                groups.setdefault(mm_per_pt, []).append(ob.prims)
            rooms = []
            for mm_per_pt, prims in groups.items():
                rooms += rooms_mod.build_rooms(args, np.concatenate(prims), mm_per_pt, lines,
                                               door_boxes=door_boxes)

            entry = document.page_entry(i, {"sheet_no": sheet_number(doc[i - 1])}, objects, args, data["lengths"], ops, rooms,
                                        viewport_of, scales.at, pdf_origin=origin)
            if a.overlay_dir and any(entry[c] for c in ("doors", "windows", "rooms", "fixtures")):
                from takeoff.overlay import draw_page
                os.makedirs(a.overlay_dir, exist_ok=True)
                draw_page(osp.join(a.overlay_dir, f"page{i:03d}.png"), args, entry, origin)
            entry["tiles"] = n_tiles
            if scales.fallback.source == "default" and not scales.anchors:
                warnings.append(f"page {i}: no scale found; 1/4\"=1'-0\" assumed")
            pages.append(entry)
            print(f"[page {i}] {len(entry['doors'])} doors, {len(entry['windows'])} windows, "
                  f"{len(entry['rooms'])} rooms, {sum(len(entry[c]) for c in ('fixtures', 'appliances', 'furniture'))} "
                  f"fixtures/appliances/furniture ({time.time() - t_start:.0f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        pdf.close()

    out = document.document(a.pdf, len(doc), pages, model_info, warnings)
    doc.close()
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    t = out["totals"]
    print(f"\n[done] {len(pages)} pages -> {a.out}")
    print(f"  doors {t['doors']['count']}, windows {t['windows']['count']}, rooms {t['rooms']['count']} "
          f"({t['rooms']['area_m2']} m2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
