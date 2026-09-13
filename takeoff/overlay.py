"""A picture of what the takeoff found on one page, for checking it by eye.

Drawn from the parsed vectors themselves (no PDF renderer needed): linework in
grey, rooms filled in their group colour with name and area, doors and windows
boxed with their tag, fixtures / appliances / furniture boxed by category.
Objects outside counted viewports are drawn faded, so what the totals include
is visible at a glance.
"""

import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "dataset"))
import room_names  # noqa: E402

CAT_BGR = {"doors": (40, 40, 220), "windows": (220, 120, 20), "fixtures": (160, 60, 160),
           "appliances": (20, 150, 200), "furniture": (40, 140, 40),
           "vertical_circulation": (90, 90, 90)}


def _hex_bgr(h):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def draw_page(path, args, entry, pdf_origin, max_side=4000):
    """Write a PNG for one page entry of the takeoff document.

    args        page-local primitive control points (PDF points, y up)
    entry       the page dict from takeoff.document.page_entry
    pdf_origin  the page's origin, to bring pdf_bbox / pdf_polygon back to page-local
    """
    pts = np.asarray(args, dtype=np.float64).reshape(-1, 8)
    if not len(pts):
        return None
    ox, oy = pdf_origin
    x0, y0 = pts[:, 0::2].min(), pts[:, 1::2].min()
    x1, y1 = pts[:, 0::2].max(), pts[:, 1::2].max()
    s = max_side / max(x1 - x0, y1 - y0, 1e-6)
    W, H = int((x1 - x0) * s) + 1, int((y1 - y0) * s) + 1
    img = np.full((H, W, 3), 255, np.uint8)

    def px(x, y):
        """PDF user space point -> image pixel."""
        return int((x - ox - x0) * s), int((y1 - (y - oy)) * s)

    lines = np.stack([(pts[:, 0] - x0) * s, (y1 - pts[:, 1]) * s,
                      (pts[:, 6] - x0) * s, (y1 - pts[:, 7]) * s], 1).astype(np.int32)
    for a, b, c, d in lines:
        cv2.line(img, (int(a), int(b)), (int(c), int(d)), (185, 185, 185), 1)

    def faded(col):
        return tuple(int(0.35 * v + 0.65 * 255) for v in col)

    overlay = img.copy()
    for rm in entry.get("rooms", []):
        arr = np.array([px(*q) for q in rm["pdf_polygon"]], np.int32)
        gid = [k for k, v in room_names.GROUP_NAMES.items() if v == rm["group"]]
        col = _hex_bgr(room_names.GROUP_HEX[gid[0]]) if gid else (200, 200, 200)
        cv2.fillPoly(overlay, [arr], col if rm["counted"] else faded(col))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    for rm in entry.get("rooms", []):
        c = np.mean([px(*q) for q in rm["pdf_polygon"]], axis=0).astype(int)
        label = f"{rm['name'] or rm['group']} {rm['area_sqft']:.0f}sf"
        cv2.putText(img, label, (int(c[0]) - 40, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (30, 30, 30), 1, cv2.LINE_AA)

    for cat, col in CAT_BGR.items():
        for it in entry.get(cat, []):
            bx0, by0, bx1, by1 = it["pdf_bbox"]
            c = col if it["counted"] else faded(col)
            cv2.rectangle(img, px(bx0, by1), px(bx1, by0), c, 2)
            if it.get("tag"):
                a, b = px(bx1, by1)
                cv2.putText(img, it["tag"], (a + 2, b - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1,
                            cv2.LINE_AA)

    cv2.imwrite(path, img)
    return path
