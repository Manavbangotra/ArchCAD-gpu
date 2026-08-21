"""SVG geometry helpers shared by the dataset converters.

Transform composition, path-data flattening, and refitting flattened runs back
into straight lines and circular arcs. Kept here rather than in one converter
because both the CubiCasa and the PDF paths need it: a symbol's curvature is one
of the few features that separates a door from a wall, so whichever source the
drawing comes from, arcs have to survive as arcs (CMD_ARC) rather than being
smashed into a chain of short lines.
"""

import math
import re

NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|"
                   r"([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


# --------------------------------------------------------------------------- #
# affine transforms, composed in SVG order
# --------------------------------------------------------------------------- #
def mat_mul(m, n):
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1*a2 + c1*b2, b1*a2 + d1*b2,
            a1*c2 + c1*d2, b1*c2 + d1*d2,
            a1*e2 + c1*f2 + e1, b1*e2 + d1*f2 + f1)


IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def parse_transform(s):
    m = IDENTITY
    for name, body in re.findall(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in NUM.findall(body)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate" and v:
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale" and v:
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate" and v:
            r = math.radians(v[0])
            t = (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)
        else:
            continue
        m = mat_mul(m, t)
    return m


def apply(m, x, y):
    a, b, c, d, e, f = m
    return (a*x + c*y + e, b*x + d*y + f)


# --------------------------------------------------------------------------- #
# path data -> point runs
# --------------------------------------------------------------------------- #
def path_segments(d, bez_steps=6):
    """Flatten a `d` attribute into [(x0,y0,x1,y1), ...] in local coordinates.

    Beziers and elliptical arcs are sampled; `rejoin` puts the circular ones
    back together afterwards.
    """
    toks, cmd = [], None
    for m in TOKEN.finditer(d or ""):
        toks.append(m.group(1) if m.group(1) else float(m.group(2)))

    segs, i = [], 0
    cx = cy = sx = sy = 0.0
    while i < len(toks):
        t = toks[i]
        if isinstance(t, str):
            cmd = t
            i += 1
            if cmd.upper() == "Z":
                if (cx, cy) != (sx, sy):
                    segs.append((cx, cy, sx, sy))
                cx, cy = sx, sy
                continue
        elif cmd is None:
            break
        rel = cmd.islower()
        k = cmd.upper()

        def take(n):
            nonlocal i
            vals = toks[i:i+n]
            i += n
            return vals if len(vals) == n and all(isinstance(v, float) for v in vals) else None

        if k == "M":
            v = take(2)
            if not v:
                break
            cx, cy = (cx+v[0], cy+v[1]) if rel else (v[0], v[1])
            sx, sy = cx, cy
            cmd = "l" if rel else "L"          # implicit lineto after moveto
        elif k == "L":
            v = take(2)
            if not v:
                break
            nx, ny = (cx+v[0], cy+v[1]) if rel else (v[0], v[1])
            segs.append((cx, cy, nx, ny)); cx, cy = nx, ny
        elif k == "H":
            v = take(1)
            if not v:
                break
            nx = cx+v[0] if rel else v[0]
            segs.append((cx, cy, nx, cy)); cx = nx
        elif k == "V":
            v = take(1)
            if not v:
                break
            ny = cy+v[0] if rel else v[0]
            segs.append((cx, cy, cx, ny)); cy = ny
        elif k in ("C", "S", "Q", "T", "A"):
            n = {"C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}[k]
            v = take(n)
            if not v:
                break
            if k in ("C", "Q"):
                if k == "C":
                    p1 = (cx+v[0], cy+v[1]) if rel else (v[0], v[1])
                    p2 = (cx+v[2], cy+v[3]) if rel else (v[2], v[3])
                    p3 = (cx+v[4], cy+v[5]) if rel else (v[4], v[5])
                else:
                    q1 = (cx+v[0], cy+v[1]) if rel else (v[0], v[1])
                    p3 = (cx+v[2], cy+v[3]) if rel else (v[2], v[3])
                    # raise the quadratic to a cubic
                    p1 = (cx + 2/3*(q1[0]-cx), cy + 2/3*(q1[1]-cy))
                    p2 = (p3[0] + 2/3*(q1[0]-p3[0]), p3[1] + 2/3*(q1[1]-p3[1]))
                px, py = cx, cy
                for s in range(1, bez_steps+1):
                    u = s/bez_steps; w = 1-u
                    bx = w**3*cx + 3*w*w*u*p1[0] + 3*w*u*u*p2[0] + u**3*p3[0]
                    by = w**3*cy + 3*w*w*u*p1[1] + 3*w*u*u*p2[1] + u**3*p3[1]
                    segs.append((px, py, bx, by)); px, py = bx, by
                cx, cy = p3
            else:  # S/T/A: approximate by the chord to the endpoint
                nx, ny = (cx+v[-2], cy+v[-1]) if rel else (v[-2], v[-1])
                segs.append((cx, cy, nx, ny)); cx, cy = nx, ny
        else:
            break
    return segs


def segments_to_runs(segs):
    """Chain consecutive segments into point runs: [(points, closed), ...]."""
    runs, cur = [], []
    for (x0, y0, x1, y1) in segs:
        if cur and abs(cur[-1][0]-x0) < 1e-9 and abs(cur[-1][1]-y0) < 1e-9:
            cur.append((x1, y1))
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = [(x0, y0), (x1, y1)]
    if len(cur) > 1:
        runs.append(cur)
    out = []
    for r in runs:
        closed = math.dist(r[0], r[-1]) < 1e-7 and len(r) > 3
        out.append((r[:-1] if closed else r, closed))
    return out


# --------------------------------------------------------------------------- #
# refit runs into lines and arcs
# --------------------------------------------------------------------------- #
def _turn(a, b, c):
    v1 = math.atan2(b[1]-a[1], b[0]-a[0])
    v2 = math.atan2(c[1]-b[1], c[0]-b[0])
    t = v2 - v1
    while t > math.pi:
        t -= 2*math.pi
    while t < -math.pi:
        t += 2*math.pi
    return t


def circle_through(p, q, r):
    """Circumcentre and radius of three points, or None if near-collinear."""
    ax, ay = p; bx, by = q; cx, cy = r
    d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by))/d
    uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax))/d
    return (ux, uy), math.dist((ux, uy), p)


MAX_ARC_RADIUS_RATIO = 50.0   # beyond this, a "circle" is really a straight line


# A sampled curve turns by a little at every step; a polygon corner turns by a
# lot at one step. Without this guard the three 90-degree corners of a wall quad
# trivially "fit a circle" and the wall is emitted as an arc.
CORNER_TURN = math.radians(35)


def rejoin(pts, closed, straight_tol=math.radians(4), arc_min=4, arc_tol=0.08):
    """Rebuild a sampled run into ("line", p0, p1) and ("arc", p0, p1, r) parts."""
    n = len(pts)
    if n < 2:
        return []
    seq = pts + [pts[0]] if closed else pts
    m = len(seq)
    turns = [_turn(seq[i-1], seq[i], seq[i+1]) for i in range(1, m-1)]

    out, i = [], 0
    span = max(math.dist(seq[0], p) for p in seq) or 1.0
    while i < m-1:
        j = i+1
        while j < m-1:
            t = turns[j-1]
            if abs(t) >= CORNER_TURN:
                break                      # a corner ends the run, never an arc
            if abs(t) < straight_tol:
                if any(abs(turns[k-1]) >= straight_tol for k in range(i+1, j+1)):
                    break
            else:
                first = next((turns[k-1] for k in range(i+1, m-1)
                              if straight_tol <= abs(turns[k-1]) < CORNER_TURN), None)
                if first is None or t*first < 0 or abs(abs(t)-abs(first)) > abs(first)*0.6:
                    break
            j += 1

        run = seq[i:j+1]
        curved = any(abs(turns[k-1]) >= straight_tol for k in range(i+1, j))
        fit = circle_through(run[0], run[len(run)//2], run[-1]) if (curved and len(run) >= arc_min) else None
        # A huge radius means the fit is really a straight line; emitting it as an
        # arc would put nonsense in the curve feature.
        if (fit and 0 < fit[1] < span*MAX_ARC_RADIUS_RATIO
                and all(abs(math.dist(fit[0], p)-fit[1]) <= arc_tol*fit[1] for p in run)):
            out.append(("arc", run[0], run[-1], fit[1]))
        elif not curved:
            out.append(("line", run[0], run[-1]))
        else:
            out.extend(("line", run[k], run[k+1]) for k in range(len(run)-1))
        i = j
    return out


def polyline_length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
