#!/usr/bin/env python3
"""Browser editor for inspecting and correcting tile labels.

The labels come from CAD layer names, which is fast and free but not always
right: a firm may draw a door on a wall layer, or use a layer name the taxonomy
does not recognise. This lets you look at any tile, see the labels on the sheet
they came from, and fix what is wrong.

Corrections are written to `<stem>_s2.override.json` next to the tile -- the
original file is never modified, so a correction survives a corpus rebuild only
if you keep the override files, and can always be thrown away.

    python tools/label_editor.py --root dataset/us_plans/json4 --port 8900

Then open http://localhost:8900/
"""

import argparse
import json
import os
import os.path as osp
import re
import struct
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

def _autofit(json_path, png_path, margin=0.10):
    """Fit the labels onto the render by cross-correlation.

    crop_tile_image() clamps its crop to the page raster, so a tile at a sheet
    edge shows slightly less than the vector extent covers; resizing that short
    crop to a square then leaves the labels a few percent out. The source PDFs
    are gone, so the crop cannot be redone -- instead recover the scale and
    offset that actually line the two up. Returns None unless it clearly helps,
    so a tile that is already right is never disturbed.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    d = json.load(open(json_path))
    if not d.get("args"):
        return None

    def ink(N):
        im = Image.open(png_path).convert("L").resize((N, N), Image.BILINEAR)
        return (np.array(im) < 210).astype(np.float32)

    def vec(N, fx, fy, dx=0.0, dy=0.0):
        sx, sy = N / d["width"] * fx, N / d["height"] * fy
        v = Image.new("L", (N, N), 0)
        dr = ImageDraw.Draw(v)
        for a in d["args"]:
            dr.line([(a[i] * sx + dx, a[i + 1] * sy + dy) for i in range(0, 8, 2)],
                    fill=255, width=2)
        return np.array(v) > 0

    def corr(I, V, lim):
        F = np.fft.fftshift(np.fft.irfft2(
            np.fft.rfft2(I) * np.conj(np.fft.rfft2(V.astype(np.float32))), s=I.shape))
        c = np.array(F.shape) // 2
        w = F[c[0] - lim:c[0] + lim + 1, c[1] - lim:c[1] + lim + 1]
        dy, dx = np.unravel_index(np.argmax(w), w.shape)
        return w.max(), dy - lim, dx - lim

    def cover(I, N, fx, fy, dx, dy):
        V = vec(N, fx, fy, dx * N / 980.0, dy * N / 980.0)
        return float((I.astype(bool) & V).sum()) / max(V.sum(), 1)

    N = 256
    I = ink(N)
    best = None
    for fx in np.arange(0.90, 1.1251, 0.025):
        for fy in np.arange(0.90, 1.1251, 0.025):
            p_, _, _ = corr(I, vec(N, fx, fy), 32)
            if best is None or p_ > best[0]:
                best = (p_, fx, fy)
    _, fx, fy = best

    N2 = 512
    I2 = ink(N2)
    b2 = None
    for gx in np.arange(fx - 0.02, fx + 0.0201, 0.01):
        for gy in np.arange(fy - 0.02, fy + 0.0201, 0.01):
            p_, dy, dx = corr(I2, vec(N2, gx, gy), 64)
            if b2 is None or p_ > b2[0]:
                b2 = (p_, gx, gy, dy, dx)
    _, gx, gy, dy, dx = b2
    DX, DY = dx * 980.0 / N2, dy * 980.0 / N2

    NV = 490
    IV = ink(NV)
    plain = cover(IV, NV, 1, 1, 0, 0)
    fit = cover(IV, NV, gx, gy, DX, DY)
    if fit <= plain * (1 + margin):
        return None
    return {"dx": float(DX), "dy": float(DY), "sx": float(gx), "sy": float(gy)}


def auto_align(tile_path):
    """`_autofit` once per tile, remembered on disk -- the fit costs seconds."""
    cache = tile_path.replace("_s2.json", "_s2.autoalign.json")
    if osp.exists(cache):
        try:
            return json.load(open(cache)) or None
        except Exception:
            pass
    png = tile_path.replace("_s2.json", "_s2.png")
    if not osp.exists(png):
        return None
    try:
        r = _autofit(tile_path, png)
    except Exception:
        return None          # numpy/PIL absent, or an unreadable render
    try:
        json.dump(r or {}, open(cache, "w"))
    except Exception:
        pass
    return r


def _side(tile_path, kind):
    """Path for a sidecar file, following symlinks to the real tile.

    The plans and rejected views are symlink trees over one corpus, so a verdict
    saved in one has to be visible in the other; writing beside the link would
    give each view its own private answer.
    """
    return osp.realpath(tile_path).replace("_s2.json", f"_s2.{kind}.json")


def _relink(split, name, verdict):
    """Move a tile's links so the two views agree with the verdict.

    The kept and rejected views are symlink trees over one corpus, so deciding a
    tile belongs elsewhere means unlinking it here and linking it there. The
    tile itself never moves -- only which view lists it.
    """
    if not (KEEP_ROOT and DROP_ROOT and verdict):
        return
    real = osp.realpath(osp.join(ROOT, split, name))
    dest = KEEP_ROOT if verdict == "keep" else DROP_ROOT
    other = DROP_ROOT if verdict == "keep" else KEEP_ROOT
    for ext in (".json", ".png"):
        target = real.replace("_s2.json", "_s2" + ext)
        base = osp.basename(target)
        gone = osp.join(other, split, base)
        if osp.lexists(gone):
            os.remove(gone)
        here = osp.join(dest, split, base)
        if osp.exists(target) and not osp.lexists(here):
            os.makedirs(osp.dirname(here), exist_ok=True)
            os.symlink(target, here)


def _read_side(tile_path, kind, default=None):
    try:
        return json.load(open(_side(tile_path, kind)))
    except Exception:
        return default


def _png_size(path):
    """Pixel size straight from the PNG header, so the editor stays stdlib-only."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        w, h = struct.unpack(">II", head[16:24])
        return int(w), int(h)
    except Exception:
        return None


def _norm_align(a):
    """Alignment as {dx, dy, sx, sy}.

    Older files stored a single uniform `scale`; a plot that is stretched in one
    axis needs the axes separately, so `scale` is read as both.
    """
    a = a or {}
    k = float(a.get("scale", 1.0) or 1.0)
    return {"dx": float(a.get("dx", 0) or 0), "dy": float(a.get("dy", 0) or 0),
            "sx": float(a.get("sx", k) or k), "sy": float(a.get("sy", k) or k)}


CLASSES = [("door", "#E11D48"), ("window", "#2563EB"),
           ("wall", "#C2761E"), ("background", "#B9C0C8")]
PROJ = re.compile(r"^(project_\d+)_")
ROOT = None
AUTOALIGN = True
KEEP_ROOT = None
DROP_ROOT = None
VIEW = ""


def tiles():
    """Every tile under root, newest split first, with a cheap summary."""
    out = []
    for split in ("train", "test"):
        d = osp.join(ROOT, split)
        if not osp.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith("_s2.json"):
                out.append((split, f))
    if not out:                      # corpus not split yet
        d = osp.join(ROOT, "all")
        if osp.isdir(d):
            out = [("all", f) for f in sorted(os.listdir(d)) if f.endswith("_s2.json")]
    return out


INDEX = None
INDEX_PATH = None


def build_index(force=False):
    """Summarise every tile once and cache it to disk.

    Counting classes means parsing each tile, and a tile can be megabytes, so
    doing it per request made the list take minutes. Cache is keyed on the tile
    count, so adding tiles rebuilds it.
    """
    global INDEX
    names = tiles()
    if not force and INDEX_PATH and osp.exists(INDEX_PATH):
        try:
            cached = json.load(open(INDEX_PATH))
            if cached.get("key_len") == len(names):
                INDEX = cached["rows"]
                return INDEX
        except Exception:
            pass
    rows = []
    for i, (sp, n) in enumerate(names, 1):
        r = summary(sp, n)
        if r:
            rows.append(r)
        if i % 250 == 0:
            print(f"  indexed {i}/{len(names)}", flush=True)
    INDEX = rows
    if INDEX_PATH:
        try:
            json.dump({"key_len": len(names), "rows": rows}, open(INDEX_PATH, "w"))
        except Exception:
            pass
    return INDEX


def summary(split, name):
    p = osp.join(ROOT, split, name)
    try:
        d = json.load(open(p))
    except Exception:
        return None
    c = Counter(d["semanticIds"])
    ov = p.replace("_s2.json", "_s2.override.json")
    return {"split": split, "name": name,
            "project": (PROJ.match(name).group(1) if PROJ.match(name) else "?"),
            "n": len(d["semanticIds"]),
            "door": c.get(0, 0), "window": c.get(1, 0), "wall": c.get(2, 0),
            "edited": osp.exists(ov)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        # The page is generated from this file; a cached copy silently keeps
        # running yesterday's JS, which looks exactly like "the fix didn't work".
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if u.path == "/api/tiles":
            rows = INDEX if INDEX is not None else build_index()
            # Tiles are linked in and out of this tree by the other view, so a
            # count that no longer matches means the cached index is stale.
            if len(tiles()) != len(rows):
                rows = build_index(force=True)
            proj = q.get("project", [None])[0]
            if proj:
                rows = [r for r in rows if r["project"] == proj]
            # Read verdicts at request time rather than caching them into the
            # index: the index is keyed on tile count, so a verdict changed
            # after it was built would never show up.
            out = []
            for r in rows:
                v = _read_side(osp.join(ROOT, r["split"], r["name"]), "verdict")
                out.append(dict(r, verdict=(v or {}).get("verdict", "")))
            return self._send(200, {"tiles": out, "view": VIEW,
                                    "projects": sorted({r["project"] for r in rows})})

        if u.path == "/api/tile":
            split, name = q["split"][0], unquote(q["name"][0])
            p = osp.join(ROOT, split, name)
            d = json.load(open(p))
            ovp = p.replace("_s2.json", "_s2.override.json")
            over = json.load(open(ovp)) if osp.exists(ovp) else {}
            alp = p.replace("_s2.json", "_s2.align.json")
            align = json.load(open(alp)) if osp.exists(alp) else None
            if align is None:
                pj = osp.join(ROOT, "_project_align.json")
                allp = json.load(open(pj)) if osp.exists(pj) else {}
                m = PROJ.match(name)
                align = allp.get(m.group(1)) if m else None
            auto = False
            if align is None and AUTOALIGN:
                fitted = auto_align(p)
                if fitted:
                    align, auto = fitted, True
            # crop_tile_image() squashes every crop into a square, so a vector
            # unit is worth a *different* number of pixels on each axis. That
            # ratio is the base transform; a saved align is a nudge on top of
            # it, which is why it stays meaningful across tiles of differing
            # size when applied to a whole project.
            size = _png_size(osp.join(ROOT, split,
                                      name.replace("_s2.json", "_s2.png")))
            base = ({"sx": size[0] / d["width"], "sy": size[1] / d["height"]}
                    if size else {"sx": 1.0, "sy": 1.0})
            # only what the editor needs -- args are the bulk of the file
            return self._send(200, {
                "width": d["width"], "height": d["height"],
                "base": base,
                "args": d["args"], "semanticIds": d["semanticIds"],
                "commands": d.get("commands", []),
                "layerIds": d.get("layerIds", []),
                "image": f"/img?split={split}&name={name}",
                "overrides": over,
                "verdict": (_read_side(p, "verdict") or {}).get("verdict", ""),
                "regions": _read_side(p, "regions", []),
                "auto": auto,
                "align": _norm_align(align)})

        if u.path == "/img":
            split, name = q["split"][0], unquote(q["name"][0])
            p = osp.join(ROOT, split, name.replace("_s2.json", "_s2.png"))
            if not osp.exists(p):
                return self._send(404, {"error": "no image"})
            data = open(p, "rb").read()
            return self._send(200, data, "image/png")

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")

        if u.path == "/api/verdict":
            # Whether this tile belongs in the training set at all. Kept apart
            # from the class overrides: one is "what is this primitive", the
            # other is "should this drawing be here".
            p = osp.join(ROOT, payload["split"], payload["name"])
            v = payload.get("verdict", "")
            f = _side(p, "verdict")
            if v:
                json.dump({"verdict": v}, open(f, "w"))
            elif osp.exists(f):
                os.remove(f)
            # Deliberately does not re-file the tile. Relinking each mark meant
            # a reindex and a list reload per keystroke, which made marking a
            # few hundred tiles unusable; the moves are applied in one pass by
            # /api/migrate instead.
            return self._send(200, {"verdict": v})

        if u.path == "/api/regions":
            # Rectangles in [0,1] of the tile, each keep or drop, for the common
            # case of one sheet holding a plan next to elevations: the tile as a
            # whole is neither, so the verdict has to be per area.
            p = osp.join(ROOT, payload["split"], payload["name"])
            regs = payload.get("regions") or []
            f = _side(p, "regions")
            if regs:
                json.dump(regs, open(f, "w"))
            elif osp.exists(f):
                os.remove(f)
            return self._send(200, {"n": len(regs)})

        if u.path == "/api/migrate":
            # Apply every mark in this view that sends a tile to the other one.
            moved = []
            for split, name in tiles():
                v = (_read_side(osp.join(ROOT, split, name), "verdict") or {}).get("verdict")
                if v and v != VIEW:
                    _relink(split, name, v)
                    moved.append(name)
            if moved:
                build_index(force=True)
            return self._send(200, {"moved": len(moved)})

        if u.path == "/api/align":
            # Alignment is stored per tile, or for a whole project at once --
            # a plot-scale mismatch is a property of how the set was exported,
            # so it is almost never worth fixing tile by tile.
            al = _norm_align(payload)
            ident = (abs(al["dx"]) < 1e-9 and abs(al["dy"]) < 1e-9
                     and abs(al["sx"] - 1) < 1e-9 and abs(al["sy"] - 1) < 1e-9)
            if payload.get("project"):
                pj = osp.join(ROOT, "_project_align.json")
                allp = json.load(open(pj)) if osp.exists(pj) else {}
                if ident:
                    allp.pop(payload["project"], None)
                else:
                    allp[payload["project"]] = al
                json.dump(allp, open(pj, "w"))
                return self._send(200, {"scope": "project", "project": payload["project"], **al})
            p = osp.join(ROOT, payload["split"], payload["name"])
            alp = p.replace("_s2.json", "_s2.align.json")
            if ident:
                if osp.exists(alp):
                    os.remove(alp)
            else:
                json.dump(al, open(alp, "w"))
            return self._send(200, {"scope": "tile", **al})

        if u.path == "/api/save":
            p = osp.join(ROOT, payload["split"], payload["name"])
            ovp = p.replace("_s2.json", "_s2.override.json")
            over = {str(k): int(v) for k, v in payload.get("overrides", {}).items()}
            if over:
                json.dump(over, open(ovp, "w"))
            elif osp.exists(ovp):
                os.remove(ovp)          # cleared every correction -> drop the file
            if INDEX is not None:
                for r in INDEX:
                    if r["split"] == payload["split"] and r["name"] == payload["name"]:
                        r["edited"] = bool(over)
                        break
            return self._send(200, {"saved": len(over), "path": ovp})

        return self._send(404, {"error": "not found"})


PAGE = r"""<!doctype html><meta charset=utf-8>
<title>Label editor</title>
<style>
:root{--bg:#EBEDF0;--panel:#fff;--ink:#12161B;--rule:#D2D8DF;--muted:#69747F;--accent:#2F5D8A}
@media(prefers-color-scheme:dark){:root{--bg:#0E1216;--panel:#171B21;--ink:#E7ECF2;--rule:#262D35;--muted:#7D8895;--accent:#7FB0DC}}
*{box-sizing:border-box}
body{margin:0;height:100vh;display:grid;grid-template-columns:290px 1fr 210px;
     background:var(--bg);color:var(--ink);font:14px/1.5 "Ubuntu Sans",system-ui,sans-serif}
aside{background:var(--panel);border-right:1px solid var(--rule);overflow-y:auto}
aside.r{border-right:none;border-left:1px solid var(--rule);padding:14px}
h2{font:600 11px/1 ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;
   color:var(--muted);margin:16px 12px 8px}
select,button{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--rule);
  border-radius:6px;padding:6px 9px}
button{cursor:pointer}button:hover{border-color:var(--accent)}
.row{padding:7px 12px;border-bottom:1px solid var(--rule);cursor:pointer;font:12px ui-monospace,monospace}
.row:hover{background:var(--bg)}.row[aria-selected=true]{background:var(--accent);color:#fff}
.row b{display:block;font-weight:600}
.row s{text-decoration:none;opacity:.7;font-size:11px}
.row .e{float:right;color:#2C7A4B;font-weight:700}
main{position:relative;overflow:hidden;background:var(--bg)}
#stage{position:absolute;inset:0;cursor:crosshair}
#stage img{position:absolute;left:0;top:0;transform-origin:0 0;image-rendering:crisp-edges}
#svg{position:absolute;left:0;top:0;transform-origin:0 0}
#regs{position:absolute;left:0;top:0;transform-origin:0 0;pointer-events:none}
#regs rect{stroke-width:3;vector-effect:non-scaling-stroke}
#vkeep.on{background:#0d7a33;color:#fff}
#vdrop.on{background:#b0203a;color:#fff}
#migrate.on{background:#1d4ed8;color:#fff;font-weight:600}
#rkeep.on{background:#0d7a33;color:#fff}
#rdrop.on{background:#b0203a;color:#fff}
.row b .v{font-size:9px;padding:1px 4px;border-radius:3px;margin-left:5px}
#svg line{stroke-width:2.4;vector-effect:non-scaling-stroke;cursor:pointer}
#svg line.sel{stroke-width:6}
.modes{display:inline-flex;border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.modes button{border:none;border-right:1px solid var(--rule);border-radius:0;padding:6px 11px}
.modes button:last-child{border-right:none}
.modes button[aria-pressed=true]{background:var(--accent);color:#fff}
.bar{position:absolute;top:0;left:0;right:0;padding:8px 12px;background:var(--panel);
     border-bottom:1px solid var(--rule);display:flex;gap:8px;align-items:center;z-index:5;
     font:12px ui-monospace,monospace}
.cls{display:block;width:100%;text-align:left;margin-bottom:6px;padding:8px 10px;border-radius:6px}
.cls[aria-pressed=true]{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}
.cls i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:8px;vertical-align:-1px}
.k{font:11px/1.7 ui-monospace,monospace;color:var(--muted);margin-top:14px}
.stat{font:12px/1.7 ui-monospace,monospace}
#toast{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:var(--ink);
  color:var(--bg);padding:8px 14px;border-radius:6px;opacity:0;transition:.2s;font:12px ui-monospace,monospace}
#toast.on{opacity:1}
</style>

<aside>
  <h2>Project</h2>
  <div style="padding:0 12px"><select id=proj style="width:100%"></select></div>
  <h2>Tiles</h2>
  <div id=list></div>
</aside>

<main>
  <div class=bar>
    <span class=modes>
      <button class=mode data-m=paint aria-pressed=true title="drag to paint (P)">&#9998; paint</button>
      <button class=mode data-m=pan   aria-pressed=false title="drag to pan (H)">&#9995; pan</button>
      <button class=mode data-m=align aria-pressed=false title="drag to move labels (A)">&#10021; align</button>
      <button class=mode data-m=region aria-pressed=false title="drag a box over part of the sheet (R)">&#9744; region</button>
    </span>
    <span style="width:10px"></span>
    <button id=vkeep title="whole tile belongs in the dataset (K)">&#10003; keep tile</button>
    <button id=vdrop title="whole tile does not belong (D)">&#10007; drop tile</button>
    <span style="width:10px"></span>
    <button id=fit>Fit</button><button id=zin>+</button><button id=zout>&minus;</button>
    <span id=info style="color:var(--muted)"></span>
    <span style="flex:1"></span>
    <label><input type=checkbox id=showbg checked> show background</label>
    <label style="font-size:11px"><input type=checkbox id=autonext checked> auto-advance</label>
    <button id=migrate disabled>Migrate</button>
    <button id=save>Save corrections</button>
  </div>
  <div id=stage><img id=sheet><svg id=svg xmlns="http://www.w3.org/2000/svg"></svg><svg id=regs xmlns="http://www.w3.org/2000/svg"></svg></div>
  <div id=toast></div>
</main>

<aside class=r>
  <h2 style="margin:0 0 8px">Paint as</h2>
  <div id=palette></div>
  <div class=stat id=counts></div>
  <h2 style="margin:16px 0 8px">Regions</h2>
  <div style="font-size:11px;color:var(--muted);margin-bottom:6px">
    box part of a sheet when the tile is partly plan and partly not.<br>
    click a box to select &middot; double-click or Del to delete
  </div>
  <div style="display:flex;gap:6px;margin-bottom:6px">
    <button id=rkeep class=on style="flex:1" onclick="setRpaint('keep')">&#10003; keep area</button>
    <button id=rdrop style="flex:1" onclick="setRpaint('drop')">&#10007; drop area</button>
  </div>
  <div style="font-size:11px;color:var(--muted);margin-bottom:6px">
    regions: <span id=rinfo>none yet</span>
  </div>
  <div style="display:flex;gap:6px;margin-bottom:4px">
    <button style="flex:1" id=rdel disabled onclick="deleteRegion(selReg)">delete selected</button>
    <button style="flex:1" onclick="undoRegion()">undo last</button>
  </div>
  <div style="display:flex;gap:6px;margin-bottom:4px">
    <button style="flex:1" onclick="clearRegions()">clear all</button>
  </div>

  <h2 style="margin:16px 0 8px">Align labels</h2>
  <div class=stat id=alv style="margin-bottom:6px">dx 0 dy 0 ×1.000</div>
  <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px">
    <button onclick="nudge(-1,0)">&larr;</button><button onclick="nudge(1,0)">&rarr;</button>
    <button onclick="nudge(0,-1)">&uarr;</button><button onclick="nudge(0,1)">&darr;</button>
  </div>
  <div class=k style="margin:2px 0 4px">stretch wider / narrower</div>
  <div style="display:flex;gap:4px;margin-bottom:6px">
    <button style="flex:1" onclick="nudge(0,0,1.01,1)">wider &#8596;</button>
    <button style="flex:1" onclick="nudge(0,0,1/1.01,1)">narrower</button>
  </div>
  <div class=k style="margin:2px 0 4px">stretch taller / shorter</div>
  <div style="display:flex;gap:4px;margin-bottom:6px">
    <button style="flex:1" onclick="nudge(0,0,1,1.01)">taller &#8597;</button>
    <button style="flex:1" onclick="nudge(0,0,1,1/1.01)">shorter</button>
  </div>
  <div style="display:flex;gap:4px;margin-bottom:6px">
    <button style="flex:1" onclick="nudge(0,0,1.01,1.01)">both +</button>
    <button style="flex:1" onclick="nudge(0,0,1/1.01,1/1.01)">both &minus;</button>
    <button onclick="resetAlign()">reset</button>
  </div>
  <button style="width:100%;margin-bottom:4px" onclick="saveAlign('tile')">save for this tile</button>
  <button style="width:100%" onclick="saveAlign('project')">apply to whole project</button>
  <div class=k>
    <b>p</b> paint &middot; <b>h</b> pan &middot; <b>a</b> align<br>
    or hold shift (pan) / alt (align)<br>
    in align mode, <b>shift+drag stretches</b><br>
    click &mdash; paint one<br>
    drag &mdash; paint many<br>
    scroll &mdash; zoom<br>
    1&ndash;4 &mdash; pick class<br>
    ctrl+z &mdash; undo<br>
    ctrl+s &mdash; save
  </div>
</aside>

<script>
const CLASSES=[["door","#E11D48"],["window","#2563EB"],["wall","#C2761E"],["background","#B9C0C8"]];
let TILES=[], cur=null, data=null, over={}, undo=[], paint=0, showbg=true;
let al={dx:0,dy:0,sx:1,sy:1};
let base={sx:1,sy:1}, auto=false;
let verdict='', regions=[], rpaint='keep', drawing=null, selReg=-1;
let VIEW="";
const regsvg=document.getElementById('regs');
let view={x:0,y:0,k:1};
const $=id=>document.getElementById(id);
const svg=$("svg"), sheet=$("sheet"), stage=$("stage");

function toast(m){const t=$("toast");t.textContent=m;t.classList.add("on");setTimeout(()=>t.classList.remove("on"),1400);}
function cls(i){return (i in over)?over[i]:data.semanticIds[i];}

async function loadList(project){
  const r=await fetch("/api/tiles"+(project?`?project=${project}`:""));
  const j=await r.json(); TILES=j.tiles; VIEW=j.view||"";
  if(!$("proj").options.length){
    $("proj").innerHTML='<option value="">all projects</option>'+
      j.projects.map(p=>`<option>${p}</option>`).join("");
  }
  showPending();
  $("list").innerHTML=TILES.map((t,i)=>`<div class=row data-i=${i}>
     <b>${t.name.replace("_s2.json","").slice(-34)}${t.edited?'<span class=e>edited</span>':''}${t.verdict?`<span class=v style="background:${t.verdict==="keep"?"#0d7a33":"#b0203a"};color:#fff">${t.verdict}</span>`:''}</b>
     <s>${t.split} &middot; ${t.n} prims &middot; d${t.door} w${t.window} W${t.wall}</s></div>`).join("");
}

async function open_(i){
  const t=TILES[i]; cur=t; undo=[];
  document.querySelectorAll(".row").forEach((r,j)=>r.setAttribute("aria-selected",j===i));
  const r=await fetch(`/api/tile?split=${t.split}&name=${encodeURIComponent(t.name)}`);
  data=await r.json(); over=Object.fromEntries(Object.entries(data.overrides||{}).map(([k,v])=>[+k,+v]));
  al=Object.assign({dx:0,dy:0,sx:1,sy:1}, data.align||{});
  base=Object.assign({sx:1,sy:1}, data.base||{}); auto=!!data.auto; showAlign();
  verdict=data.verdict||''; regions=(data.regions||[]).slice(); showVerdict();
  sheet.src=data.image;
  svg.setAttribute("viewBox",`0 0 ${data.width} ${data.height}`);
  svg.setAttribute("width",data.width); svg.setAttribute("height",data.height);
  regsvg.setAttribute("viewBox",`0 0 ${data.width} ${data.height}`);
  regsvg.setAttribute("width",data.width); regsvg.setAttribute("height",data.height);
  draw(); fit(); $("info").textContent=`${t.project} · ${data.args.length} primitives`;
}

function draw(){
  const f=document.createDocumentFragment();
  data.args.forEach((a,i)=>{
    const c=cls(i);
    if(c===3 && !showbg) return;
    const l=document.createElementNS("http://www.w3.org/2000/svg","line");
    l.setAttribute("x1",a[0]);l.setAttribute("y1",a[1]);
    l.setAttribute("x2",a[6]);l.setAttribute("y2",a[7]);
    l.setAttribute("stroke",CLASSES[c][1]);
    l.dataset.i=i; if(i in over) l.classList.add("sel");
    f.appendChild(l);
  });
  svg.replaceChildren(f); counts();
}
function counts(){
  const c=[0,0,0,0]; data.args.forEach((_,i)=>c[cls(i)]++);
  $("counts").innerHTML=CLASSES.map((x,k)=>
    `<div><span style="color:${x[1]}">&#9632;</span> ${x[0]} ${c[k]}</div>`).join("")
    +`<div style="margin-top:6px;color:var(--muted)">${Object.keys(over).length} corrected</div>`;
}

function apply(){
  const s=`translate(${view.x}px,${view.y}px) scale(${view.k})`;
  sheet.style.transform=s;
  // labels carry an extra nudge so a mis-plotted sheet can be lined up by hand
  svg.style.transform=`${s} translate(${al.dx}px,${al.dy}px) scale(${base.sx*al.sx},${base.sy*al.sy})`;
  regsvg.style.transform=`${s} scale(${base.sx},${base.sy})`;
}
function showAlign(){
  $("alv").innerHTML=`dx ${al.dx.toFixed(0)} &nbsp; dy ${al.dy.toFixed(0)}<br>`+
    `wide &times;${al.sx.toFixed(3)} &nbsp; tall &times;${al.sy.toFixed(3)}`;
  const off=al.dx||al.dy||al.sx!==1||al.sy!==1;
  $("alv").innerHTML+=`<br><span style="opacity:.65">fit &times;${base.sx.toFixed(3)}, `+
    `&times;${base.sy.toFixed(3)}${auto?" &middot; auto-fitted":""}</span>`;
  $("alv").style.color=off?"var(--accent)":"var(--muted)";
}
function markRow(i){
  const row=document.querySelector(`.row[data-i="${i}"] b`);
  if(!row) return;
  const old=row.querySelector(".v"); if(old) old.remove();
  const v=TILES[i].verdict;
  if(!v) return;
  const t=document.createElement("span"); t.className="v";
  t.style.background = v==="keep" ? "#0d7a33" : "#b0203a";
  t.style.color="#fff"; t.textContent=v; row.appendChild(t);
}
function pendingCount(){
  return TILES.filter(t=>t.verdict && t.verdict!==VIEW).length;
}
function showPending(){
  const n=pendingCount();
  const b=$("migrate");
  b.textContent = n ? `Migrate ${n} \u2192` : "Migrate";
  b.disabled = !n;
  b.classList.toggle("on", n>0);
}
async function migrate(){
  const n=pendingCount();
  if(!n) return;
  $("migrate").disabled=true; $("migrate").textContent="Migrating\u2026";
  const r=await fetch("/api/migrate",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
  const j=await r.json().catch(()=>({moved:0}));
  await loadList($("proj").value||"");
  if(TILES.length) open_(0);
  toast(`migrated ${j.moved} tile(s) to the other set`);
}
function showVerdict(){
  $("vkeep").classList.toggle("on", verdict==="keep");
  $("vdrop").classList.toggle("on", verdict==="drop");
  drawRegions();
}
async function setVerdict(v){
  verdict = (verdict===v) ? "" : v;        // clicking the active one clears it
  showVerdict();
  const i=TILES.indexOf(cur);
  cur.verdict=verdict;
  markRow(i);
  showPending();
  fetch("/api/verdict",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({split:cur.split,name:cur.name,verdict})});
  if(verdict && verdict!==VIEW && AUTONEXT && i+1<TILES.length) open_(i+1);
}
function regionAt(x,y){
  for(let i=regions.length-1;i>=0;i--){
    const r=regions[i];
    if(x>=r.x0&&x<=r.x1&&y>=r.y0&&y<=r.y1) return i;
  }
  return -1;
}
function deleteRegion(i){
  if(i<0||i>=regions.length) return;
  regions.splice(i,1); selReg=-1; drawRegions(); saveRegions();
  toast("region deleted");
}
function drawRegions(){
  const f=document.createDocumentFragment();
  regions.forEach((r,i)=>{
    const el=document.createElementNS("http://www.w3.org/2000/svg","rect");
    el.setAttribute("x",r.x0*data.width); el.setAttribute("y",r.y0*data.height);
    el.setAttribute("width",(r.x1-r.x0)*data.width);
    el.setAttribute("height",(r.y1-r.y0)*data.height);
    const c = r.verdict==="keep" ? "#0d7a33" : "#b0203a";
    el.setAttribute("stroke",c); el.setAttribute("fill",c);
    el.setAttribute("fill-opacity", i===selReg ? ".34" : ".16");
    if(i===selReg) el.setAttribute("stroke-width","7");
    f.appendChild(el);
    if(i===selReg){
      const g=document.createElementNS("http://www.w3.org/2000/svg","text");
      g.setAttribute("x",r.x1*data.width-6); g.setAttribute("y",r.y0*data.height+22);
      g.setAttribute("text-anchor","end"); g.setAttribute("font-size","26");
      g.setAttribute("fill",c); g.textContent="\u2715";
      f.appendChild(g);
    }
  });
  if(drawing){
    const el=document.createElementNS("http://www.w3.org/2000/svg","rect");
    el.setAttribute("x",Math.min(drawing.x0,drawing.x1)*data.width);
    el.setAttribute("y",Math.min(drawing.y0,drawing.y1)*data.height);
    el.setAttribute("width",Math.abs(drawing.x1-drawing.x0)*data.width);
    el.setAttribute("height",Math.abs(drawing.y1-drawing.y0)*data.height);
    const c = rpaint==="keep" ? "#0d7a33" : "#b0203a";
    el.setAttribute("stroke",c); el.setAttribute("fill",c); el.setAttribute("fill-opacity",".1");
    el.setAttribute("stroke-dasharray","6 4");
    f.appendChild(el);
  }
  regsvg.replaceChildren(f);
  $("rdel").disabled = selReg<0;
  $("rinfo").textContent = regions.length
    ? `${regions.filter(r=>r.verdict==="keep").length} keep / ${regions.filter(r=>r.verdict==="drop").length} drop`
    : "none yet";
}
async function saveRegions(){
  await fetch("/api/regions",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({split:cur.split,name:cur.name,regions})});
  toast(`${regions.length} region(s) saved`);
}
function setRpaint(v){ rpaint=v;
  $("rkeep").classList.toggle("on",v==="keep"); $("rdrop").classList.toggle("on",v==="drop"); }
function undoRegion(){ regions.pop(); drawRegions(); saveRegions(); }
function clearRegions(){ regions=[]; drawRegions(); saveRegions(); }

function nudge(dx,dy,fx,fy){al.dx+=dx;al.dy+=dy;al.sx*=(fx||1);al.sy*=(fy||1);showAlign();apply();}
function resetAlign(){al={dx:0,dy:0,sx:1,sy:1};showAlign();apply();}
async function saveAlign(scope){
  const body=scope==="project"
    ? {project:cur.project,...al}
    : {split:cur.split,name:cur.name,...al};
  const r=await fetch("/api/align",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)});
  const j=await r.json(); toast(`alignment saved for ${j.scope}`);
}
function fit(){const r=stage.getBoundingClientRect();
  // content is the rendered sheet, i.e. the vector extent through `base`
  const cw=data.width*base.sx, ch=data.height*base.sy;
  view.k=Math.min(r.width/cw,(r.height-40)/ch)*0.94;
  view.x=(r.width-cw*view.k)/2; view.y=40+(r.height-40-ch*view.k)/2; apply();}
$("fit").onclick=fit;
$("vkeep").onclick=()=>setVerdict("keep");
$("vdrop").onclick=()=>setVerdict("drop");
$("migrate").onclick=migrate;
let AUTONEXT=true;
$("autonext").onchange=e=>{AUTONEXT=e.target.checked;};
$("zin").onclick=()=>{view.k*=1.25;apply()};
$("zout").onclick=()=>{view.k/=1.25;apply()};
stage.addEventListener("wheel",e=>{e.preventDefault();
  const r=stage.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const f=e.deltaY>0?1/1.15:1.15;
  view.x=mx-(mx-view.x)*f; view.y=my-(my-view.y)*f; view.k*=f; apply();},{passive:false});

let panning=null, painting=false;
let aligning=null, mode='paint';
stage.addEventListener("pointerdown",e=>{
  // A held modifier still works, but the toolbar mode means you can just drag.
  // Precedence matters: while align mode is active, shift means "stretch",
  // so it must not be grabbed by the pan shortcut first.
  const m = e.altKey ? "align"
          : (mode === "align") ? "align"
          : (mode === "region") ? "region"
          : e.shiftKey ? "pan" : mode;
  if(m==="region"){
    const q=stageToTile(e);
    const hit=regionAt(q.x,q.y);
    if(hit>=0 && !e.shiftKey){
      // clicking a box selects it; double-click removes it outright
      if(selReg===hit && e.detail>=2){ deleteRegion(hit); return; }
      selReg=hit; drawRegions(); return;
    }
    selReg=-1;
    drawing={x0:q.x,y0:q.y,x1:q.x,y1:q.y};
    stage.setPointerCapture(e.pointerId); drawRegions(); return;}
  if(m==="align"){
    // dx/dy are image pixels and apply() multiplies them by view.k, so the
    // screen delta has to be divided by it or the labels do not track the
    // cursor at any zoom but 1. Also remember the user-space point grabbed,
    // so a stretch can pivot about it instead of the top-left corner.
    const r0=stage.getBoundingClientRect();
    const ux=((e.clientX-r0.left-view.x)/view.k-al.dx)/(base.sx*al.sx);
    const uy=((e.clientY-r0.top -view.y)/view.k-al.dy)/(base.sy*al.sy);
    aligning={x:e.clientX-al.dx*view.k, y:e.clientY-al.dy*view.k,
              stretch:(mode==="align" && e.shiftKey),
              px:e.clientX, py:e.clientY, sx0:al.sx, sy0:al.sy,
              dx0:al.dx, dy0:al.dy, ux:ux, uy:uy};
    stage.setPointerCapture(e.pointerId); return;}
  if(m==="pan"){panning={x:e.clientX-view.x,y:e.clientY-view.y};stage.setPointerCapture(e.pointerId);return;}
  painting=true; hit(e);
});
function stageToTile(e){
  // screen -> tile fraction, through the same transform the sheet is drawn with
  const r=stage.getBoundingClientRect();
  const x=((e.clientX-r.left-view.x)/view.k)/(base.sx*data.width);
  const y=((e.clientY-r.top -view.y)/view.k)/(base.sy*data.height);
  return {x:Math.max(0,Math.min(1,x)), y:Math.max(0,Math.min(1,y))};
}
stage.addEventListener("pointermove",e=>{
  if(drawing){ const q=stageToTile(e); drawing.x1=q.x; drawing.y1=q.y; drawRegions(); return;}
  if(aligning){
    if(aligning.stretch){
      // shift while aligning stretches instead of moving: drag right to widen,
      // down to lengthen, each axis independent.
      const f=v=>Math.max(0.2,Math.min(5,1+v/400));
      al.sx=aligning.sx0*f(e.clientX-aligning.px);
      al.sy=aligning.sy0*f(e.clientY-aligning.py);
      // hold the grabbed point still, otherwise widening also slides the sheet
      al.dx=aligning.dx0+base.sx*aligning.ux*(aligning.sx0-al.sx);
      al.dy=aligning.dy0+base.sy*aligning.uy*(aligning.sy0-al.sy);
    }else{
      al.dx=(e.clientX-aligning.x)/view.k; al.dy=(e.clientY-aligning.y)/view.k;
    }
    showAlign(); apply(); return;}
  if(panning){view.x=e.clientX-panning.x;view.y=e.clientY-panning.y;apply();return;}
  if(painting) hit(e);
});
addEventListener("pointerup",()=>{
  if(drawing){
    const r={x0:Math.min(drawing.x0,drawing.x1), y0:Math.min(drawing.y0,drawing.y1),
             x1:Math.max(drawing.x0,drawing.x1), y1:Math.max(drawing.y0,drawing.y1),
             verdict:rpaint};
    drawing=null;
    if((r.x1-r.x0)>0.01 && (r.y1-r.y0)>0.01){ regions.push(r); saveRegions(); }
    drawRegions();
  }
  panning=null;painting=false;aligning=null;});

function hit(e){
  const el=document.elementFromPoint(e.clientX,e.clientY);
  if(!el||el.tagName!=="line") return;
  const i=+el.dataset.i;
  if(cls(i)===paint) return;
  undo.push({i,prev:(i in over)?over[i]:undefined});
  over[i]=paint;
  el.setAttribute("stroke",CLASSES[paint][1]); el.classList.add("sel");
  if(paint===3&&!showbg) el.remove();
  counts();
}

$("palette").innerHTML=CLASSES.map((c,k)=>
  `<button class=cls data-k=${k} aria-pressed=${k===0}><i style="background:${c[1]}"></i>${c[0]} <span style="float:right;opacity:.5">${k+1}</span></button>`).join("");
$("palette").onclick=e=>{const b=e.target.closest(".cls"); if(!b)return;
  paint=+b.dataset.k;
  document.querySelectorAll(".cls").forEach(x=>x.setAttribute("aria-pressed",x===b));};

function setMode(m){
  mode=m;
  document.querySelectorAll(".mode").forEach(b=>b.setAttribute("aria-pressed",b.dataset.m===m));
  stage.style.cursor = m==="pan" ? "grab" : m==="align" ? "move" : "crosshair";
}
document.querySelector(".modes").onclick=e=>{const b=e.target.closest(".mode"); if(b) setMode(b.dataset.m);};

$("showbg").onchange=e=>{showbg=e.target.checked; draw();};
$("list").onclick=e=>{const r=e.target.closest(".row"); if(r) open_(+r.dataset.i);};
$("proj").onchange=e=>loadList(e.target.value);

async function save(){
  if(!cur) return;
  const r=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({split:cur.split,name:cur.name,overrides:over})});
  const j=await r.json(); toast(`saved ${j.saved} corrections`);
  const row=document.querySelector(`.row[data-i="${TILES.indexOf(cur)}"] b`);
  if(row && j.saved && !row.querySelector(".e")) row.insertAdjacentHTML("beforeend",'<span class=e>edited</span>');
}
$("save").onclick=save;

addEventListener("keydown",e=>{
  if(e.key>="1"&&e.key<="4"){document.querySelector(`.cls[data-k="${+e.key-1}"]`).click();}
  if(e.key==="p"||e.key==="P") setMode("paint");
  if(e.key==="h"||e.key==="H") setMode("pan");
  if(e.key==="a"||e.key==="A") setMode("align");
  if(e.key==="r"||e.key==="R") setMode("region");
  if((e.key==="Delete"||e.key==="Backspace") && selReg>=0){e.preventDefault(); deleteRegion(selReg);}
  if(e.key==="Escape"){ selReg=-1; if(data) drawRegions(); }
  if(e.key==="k"||e.key==="K") setVerdict("keep");
  if(e.key==="d"||e.key==="D") setVerdict("drop");
  if(e.ctrlKey&&e.key==="s"){e.preventDefault();save();}
  if(e.ctrlKey&&e.key==="z"){e.preventDefault();
    const u=undo.pop(); if(!u)return;
    if(u.prev===undefined) delete over[u.i]; else over[u.i]=u.prev;
    draw();}
});

loadList().then(()=>{ if(TILES.length) open_(0); });
</script>
"""


def _warm_one(t):
    auto_align(t)
    return t


def _warm_all():
    """Fit every tile in the background so opening one is instant."""
    import multiprocessing as mp
    paths = [osp.join(ROOT, sp, nm) for sp, nm in tiles()]
    todo = [t for t in paths
            if not osp.exists(t.replace("_s2.json", "_s2.autoalign.json"))]
    if not todo:
        return
    print(f"fitting {len(todo)} tiles in the background ...", flush=True)
    try:
        with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            for i, _ in enumerate(pool.imap_unordered(_warm_one, todo, 8), 1):
                if i % 200 == 0:
                    print(f"  ... fitted {i}/{len(todo)}", flush=True)
    except Exception as e:
        print(f"background fit stopped: {e}", flush=True)
    else:
        print("background fit done", flush=True)


def main():
    global ROOT, INDEX_PATH, AUTOALIGN, KEEP_ROOT, DROP_ROOT, VIEW
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="corpus dir containing train/ test/ or all/")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--reindex", action="store_true", help="rebuild the cached index")
    ap.add_argument("--no-autoalign", action="store_true",
                    help="do not fit the labels onto each render")
    ap.add_argument("--view", choices=("keep", "drop"), default="",
                    help="which set this instance shows, so a tile re-filed to "
                         "the other one drops out of the list")
    ap.add_argument("--keep_root", help="tree to link a tile into when kept")
    ap.add_argument("--drop_root", help="tree to link a tile into when dropped")
    ap.add_argument("--warm", action="store_true",
                    help="precompute every tile's fit up front instead of on open")
    a = ap.parse_args()
    ROOT = osp.abspath(a.root)
    KEEP_ROOT = osp.abspath(a.keep_root) if a.keep_root else None
    DROP_ROOT = osp.abspath(a.drop_root) if a.drop_root else None
    VIEW = a.view
    INDEX_PATH = osp.join(ROOT, ".editor_index.json")
    print(f"indexing {len(tiles())} tiles from {ROOT} ...", flush=True)
    build_index(force=a.reindex)
    AUTOALIGN = not a.no_autoalign
    if AUTOALIGN and a.warm:
        # Fitting the whole corpus takes far too long to make anyone wait for
        # it, and a tile opened before its turn just fits itself on demand.
        import threading
        threading.Thread(target=_warm_all, daemon=True).start()
    print(f"ready -- {len(INDEX)} tiles"
          + ("" if AUTOALIGN else "  (auto-align off)"), flush=True)
    print(f"open http://localhost:{a.port}/")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
