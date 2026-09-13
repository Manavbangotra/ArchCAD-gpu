"""Drawing scale: real millimetres per PDF point, per viewport.

Every length and area in a takeoff is multiplied by this number, so it is never
guessed silently. A sheet carries several drawings at different scales (a plan
at 1/4" next to details at 1-1/2"), so each printed scale string is an anchor
at its position and an object takes the nearest one below it -- the same
"titles sit under their drawing" rule classify_viewports uses.

Order, per object: the nearest printed scale string; else the page's dimension
chains, snapped to a real architectural scale; else 1/4" = 1'-0", reported as
`default` so nothing downstream mistakes it for a measurement.

Method ported from bim-ai extraction/dimension_parser.py. Measured there on 595
pages: a scale string on 59%, and where both sources exist they agree on
101 of 106 (the rest are whole-step 1.5x/2x errors).
"""

import re
from dataclasses import dataclass, field

MM_PER_INCH = 25.4
MM_PER_FOOT = 304.8
PAPER_MM_PER_PT = MM_PER_INCH / 72.0

# 12'-6", 12'-6 1/2", 4'-0"
_FT_IN = re.compile(r"^(\d{1,3})'\s*[-–]?\s*(\d{1,2})?(?:\s+(\d{1,2})/(\d{1,2}))?\"?$")
# bare inches: 36", 5 1/2"
_IN_ONLY = re.compile(r"^(\d{1,3})(?:\s+(\d{1,2})/(\d{1,2}))?\"$")

# 1/4" = 1'-0"  |  3/16"=1'-0"  |  1/8" = 1'  |  1 1/2" = 1'-0"
_SCALE_IMPERIAL = re.compile(r"(?:(\d+)\s*[-\s]\s*)?(\d+)\s*/\s*(\d+)\s*\"\s*=\s*1'(?:\s*[-–]?\s*0\"?)?")
_SCALE_IMPERIAL_WHOLE = re.compile(r"(?<![/\d])(\d+)\s*\"\s*=\s*1'(?:\s*[-–]?\s*0\"?)?")
_SCALE_RATIO = re.compile(r"\b1\s*:\s*(\d{2,4})\b")
_NTS = re.compile(r"\bN\.?\s*T\.?\s*S\.?\b|NOT\s+TO\s+SCALE", re.I)

# Every scale an architectural sheet is actually drawn at. A measured factor is
# snapped onto this list; refusing to snap is how junk measurements show up.
SCALE_CODEBOOK = {
    4.0: '3" = 1\'-0"', 8.0: '1-1/2" = 1\'-0"', 12.0: '1" = 1\'-0"',
    16.0: '3/4" = 1\'-0"', 24.0: '1/2" = 1\'-0"', 32.0: '3/8" = 1\'-0"',
    48.0: '1/4" = 1\'-0"', 64.0: '3/16" = 1\'-0"', 96.0: '1/8" = 1\'-0"',
    128.0: '3/32" = 1\'-0"', 192.0: '1/16" = 1\'-0"', 384.0: '1/32" = 1\'-0"',
    20.0: "1:20", 25.0: "1:25", 50.0: "1:50", 100.0: "1:100", 200.0: "1:200",
}
SNAP_TOL = 0.06          # adjacent codebook entries are ~33% apart
MIN_DIM_VOTES = 3
DEFAULT_FACTOR = 48.0    # 1/4" = 1'-0", the commonest US plan scale


def _norm(text):
    return (text or "").replace("”", '"').replace("″", '"').replace("’", "'").replace("′", "'")


def parse_dim_text(s):
    """One dimension string -> millimetres, or None."""
    s = _norm(s).strip()
    m = _FT_IN.match(s)
    if m:
        inch = int(m.group(2) or 0)
        if m.group(3):
            inch += int(m.group(3)) / int(m.group(4))
        return int(m.group(1)) * MM_PER_FOOT + inch * MM_PER_INCH
    m = _IN_ONLY.match(s)
    if m:
        inch = int(m.group(1))
        if m.group(2):
            inch += int(m.group(2)) / int(m.group(3))
        return inch * MM_PER_INCH
    return None


def scale_from_text(text):
    """Scale factor (real length per paper length) from one string, or None.
    1/4" = 1'-0" -> 48; 1-1/2" = 1'-0" -> 8; 1:100 -> 100."""
    t = _norm(text)
    m = _SCALE_IMPERIAL.search(t)
    if m:
        paper_in = int(m.group(2)) / int(m.group(3)) + (int(m.group(1)) if m.group(1) else 0)
        return 12.0 / paper_in if paper_in > 0 else None
    m = _SCALE_IMPERIAL_WHOLE.search(t)
    if m and int(m.group(1)) > 0:
        return 12.0 / int(m.group(1))
    m = _SCALE_RATIO.search(t)
    if m and 20 <= int(m.group(1)) <= 500:
        return float(m.group(1))
    return None


def snap_to_codebook(factor, tol=SNAP_TOL):
    if not factor or factor <= 0:
        return None
    best = min(SCALE_CODEBOOK, key=lambda c: abs(c - factor) / c)
    return best if abs(best - factor) / best <= tol else None


def scale_from_dimensions(words):
    """Factor measured from the drawing's own dimension chains, or None.

    `words` are (text, x, y) in PDF points. A chain prints each segment's length
    at its midpoint, so the distance between two adjacent dimension texts spans
    half of each: (a + b) / 2 millimetres. Median over every adjacent pair,
    rejected when the middle half disagrees by more than 25%.
    """
    toks = []
    for text, x, y in words:
        mm = parse_dim_text(text)
        if mm and mm > 0:
            toks.append((mm, x, y))
    votes = []
    for horizontal in (True, False):
        axis, cross = (1, 2) if horizontal else (2, 1)
        rows = {}
        for t in toks:
            rows.setdefault(int(t[cross] / 6.0), []).append(t)    # ~6 pt band
        for row in rows.values():
            row.sort(key=lambda t: t[axis])
            for a, b in zip(row, row[1:]):
                d_pt = abs(b[axis] - a[axis])
                if d_pt < 4.0:
                    continue
                votes.append(((a[0] + b[0]) / 2.0) / d_pt / PAPER_MM_PER_PT)
    if len(votes) < MIN_DIM_VOTES:
        return None
    votes.sort()
    med = votes[len(votes) // 2]
    q1, q3 = votes[len(votes) // 4], votes[(3 * len(votes)) // 4]
    if med <= 0 or (q3 - q1) / med > 0.25:
        return None
    return snap_to_codebook(med)


@dataclass
class Scale:
    factor: float
    source: str                  # scale_string | dimension_pair | default
    text: str = ""

    @property
    def mm_per_pt(self):
        return PAPER_MM_PER_PT * self.factor


@dataclass
class ScaleMap:
    """Scale anchors on one page. `at(x, y)` answers for a point in PDF space."""
    anchors: list = field(default_factory=list)     # (x, y, Scale)
    fallback: Scale = None
    nts: list = field(default_factory=list)          # (x, y) of "NOT TO SCALE"

    def at(self, x, y):
        best, best_d = None, None
        for ax, ay, sc in self.anchors:
            dy = y - ay                        # positive when the point sits above
            d = abs(x - ax) + (dy if dy >= 0 else 3.0 * -dy)
            if best_d is None or d < best_d:
                best, best_d = sc, d
        return best or self.fallback


def page_scales(lines, words):
    """ScaleMap from a page's text.

    `lines` are (text, x, y) text lines and `words` (text, x, y) single words,
    both in PDF points, y up. Scale strings are matched per line (they span
    several words); dimension chains per word.
    """
    anchors, nts = [], []
    for text, x, y in lines:
        f = scale_from_text(text)
        if f:
            anchors.append((x, y, Scale(f, "scale_string", text.strip()[:60])))
        elif _NTS.search(text or ""):
            nts.append((x, y))
    dim = scale_from_dimensions(words)
    if dim:
        fallback = Scale(dim, "dimension_pair", SCALE_CODEBOOK.get(dim, ""))
    elif anchors and len({round(a[2].factor, 3) for a in anchors}) == 1:
        fallback = anchors[0][2]
    else:
        fallback = Scale(DEFAULT_FACTOR, "default", SCALE_CODEBOOK[DEFAULT_FACTOR])
    return ScaleMap(anchors, fallback, nts)
