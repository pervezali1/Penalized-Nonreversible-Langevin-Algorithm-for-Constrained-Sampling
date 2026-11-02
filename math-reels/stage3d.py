"""The 3D stage of reels 38-40: a black stage with a soft pool of light on the floor, glossy solids under a slowly
orbiting camera, and maths in Computer Modern (matplotlib mathtext, composited straight into the frame).
Shared helpers only; each reel's scene decides what to draw and when."""

import functools
import math

import cairo
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["cmr10"]
matplotlib.rcParams["axes.formatter.use_mathtext"] = True
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402

from textkit import H, W, layer, over, rgb, surf_rgba  # noqa: E402

TAU = 2 * math.pi
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, GREEN, GOLD, MAGENTA, CYAN, BLUE = "#FF7A4D", "#34D399", "#F6C56E", "#F472B6", "#38BDF8", "#5B8DFF"
BASE = np.array([0.12, 0.62, 0.95])                          # the solids' cyan
LIGHT = np.array([-0.45, -0.35, 0.82]) / np.linalg.norm([-0.45, -0.35, 0.82])
BG = np.array(rgb("#020308"), np.float32)


# ---------------------------------------------------------------- Computer Modern text

_PARSER = MathTextParser("agg")


@functools.lru_cache(maxsize=8192)
def tex(s, size):
    """Alpha mask (float32), baseline row and advance width of the string s in Computer Modern."""
    r = _PARSER.parse(s, dpi=100, prop=FontProperties(size=size, family="serif"))
    a = np.asarray(r.image, np.float32) / 255.0
    return a, int(round(r.height - r.depth)), r.width


@functools.lru_cache(maxsize=1024)
def _halo(s, size, r):
    a, bl, w = tex(s, size)
    a = np.pad(a, r)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.GaussianBlur(cv2.dilate(a, k), (0, 0), r / 2), bl + r, w


def paste(img, s, size, x, base, col, alpha=1.0, align="l", halo=0):
    """Composite s onto img with its baseline at y = base, aligned l/c/r on x. Returns its width.
    halo > 0: first lay a soft dark outline of that radius under the text, so it reads over lines."""
    if alpha <= 0.003:
        return tex(s, size)[2]
    if halo:
        a, bl, w = _halo(s, size, halo)
        a = a * 0.85
        col_h = BG
    else:
        a, bl, w = tex(s, size)
    if halo:
        hh, ww = a.shape
        x0 = int(round(x - (halo if align == "l" else (ww / 2 if align == "c" else ww - halo))))
        y0 = int(round(base - bl))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
        if xa < xb and ya < yb:
            m = a[ya - y0:yb - y0, xa - x0:xb - x0, None] * alpha
            img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - m) + col_h * m
        a, bl, w = tex(s, size)
    hh, ww = a.shape
    x0 = int(round(x - (0 if align == "l" else (ww / 2 if align == "c" else ww))))
    y0 = int(round(base - bl))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if xa >= xb or ya >= yb:
        return w
    m = a[ya - y0:yb - y0, xa - x0:xb - x0, None] * alpha
    c = np.array(rgb(col), np.float32)
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - m) + c * m
    return w


def fmt(v, nd=2):
    """A number for the card: rounded, trailing zeros dropped, no negative zero, a true minus sign (mathtext)."""
    v = 0.0 if abs(v) < 0.5 * 10 ** -nd else float(v)
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def card_bg(ctx, x, y, w, h, a):
    ctx.set_source_rgba(0.035, 0.05, 0.11, 0.92 * a)
    pill(ctx, x, y, w, h, 20)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.25, 0.32, 0.5, 0.85 * a)
    ctx.set_line_width(1.3)
    ctx.stroke()


def brackets(ctx, x0, x1, y0, y1, a, col=(0.95, 0.96, 0.98)):
    ctx.set_source_rgba(*col, a)
    ctx.set_line_width(2.2)
    for xb, s in ((x0, 1), (x1, -1)):
        ctx.move_to(xb + 10 * s, y0)
        ctx.line_to(xb, y0)
        ctx.line_to(xb, y1)
        ctx.line_to(xb + 10 * s, y1)
        ctx.stroke()


def fixed(v, nd=2):
    """A number with exactly nd decimals, no negative zero."""
    s = f"{0.0 if abs(v) < 0.5 * 10 ** -nd else float(v):.{nd}f}"
    return s


def matrix_block(img, M, x, cy, size=26, row_h=46, cols=None, a=1.0, nd=2, measure=False, keep_zeros=False):
    """A matrix (any shape; 3x3 as before) with brackets, its left edge at x and its middle row's baseline near cy. Returns its width.
    keep_zeros: every entry with exactly nd decimals (for a full matrix), instead of trailing zeros dropped."""
    M = np.atleast_2d(np.asarray(M, float))
    nr, nc = M.shape
    entries = [[(fixed if keep_zeros else fmt)(M[r, c], nd) for c in range(nc)] for r in range(nr)]
    col_w = [max(max(tex(f"${entries[r][c]}$", size)[2] for r in range(nr)), 46) for c in range(nc)]
    gap = 22
    mw = sum(col_w) + (nc - 1) * gap + 2 * 20
    if measure:
        return mw
    surf, ctx = layer()
    brackets(ctx, x, x + mw, cy - nr / 2 * row_h + 6, cy + nr / 2 * row_h - 2, a)
    trgb, ta = surf_rgba(surf)
    over(img, trgb, ta)
    cx = x + 20
    for c in range(nc):
        for r in range(nr):
            col = cols[c] if cols else INK
            paste(img, f"${entries[r][c]}$", size, cx + col_w[c] / 2, cy + (r - (nr - 1) / 2) * row_h + 10, col, a, align="c")
        cx += col_w[c] + gap
    return mw


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


# ---------------------------------------------------------------- camera and projection

class Camera:
    def __init__(self, target, az_deg, el_deg, dist=10.5, focal=2150.0, centre=(540, 960)):
        az, el = math.radians(az_deg), math.radians(el_deg)
        self.tgt = np.asarray(target, float)
        self.eye = self.tgt + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        self.fwd = (self.tgt - self.eye) / np.linalg.norm(self.tgt - self.eye)
        r = np.cross(self.fwd, [0, 0, 1.0])
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.fwd)
        self.focal, self.centre = focal, centre

    def project(self, P):
        P = np.atleast_2d(P) - self.eye
        z = P @ self.fwd
        return np.stack([self.centre[0] + self.focal * (P @ self.right) / z, self.centre[1] - self.focal * (P @ self.up) / z], 1)

    def depth(self, P):
        return (np.atleast_2d(P) - self.eye) @ self.fwd


def floor_light(img, cam, z_floor, a, radius=3.9):
    surf, ctx = layer()
    n = 64
    tx, ty = cam.tgt[0], cam.tgt[1]
    for k in range(26, 0, -1):
        r = radius * k / 26
        ring = np.array([[tx + r * math.cos(TAU * i / n), ty + r * math.sin(TAU * i / n), z_floor] for i in range(n)])
        pts = cam.project(ring)
        ctx.set_source_rgba(0.04, 0.10, 0.26, 0.055 * a)
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()
    frgb, fa = surf_rgba(surf)
    over(img, frgb, fa)


def floor_shadow(img, cam, pts3, z_floor, a, strength=0.75):
    """A soft shadow: the points dropped onto the floor (with a slight slant), their hull blurred."""
    pts3 = np.asarray(pts3)
    foot = np.column_stack([pts3[:, 0] + 0.25 * (pts3[:, 2] - z_floor), pts3[:, 1] + 0.18 * (pts3[:, 2] - z_floor), np.full(len(pts3), z_floor)])
    hp = hull(cam.project(foot))
    if len(hp) >= 3:
        mask = np.zeros((H // 2, W // 2), np.float32)
        cv2.fillPoly(mask, [np.round(np.array(hp) / 2).astype(np.int32)], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), 14)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        img *= (1 - strength * a * mask)[..., None]


# ---------------------------------------------------------------- solids

def sphere_mesh(n_lat=14, n_lon=28):
    """Unit-sphere vertices and quads (lat-long), with the latitude/longitude index of each quad."""
    verts = []
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = TAU * j / n_lon
            verts.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    quads = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            quads.append((a, b, b + n_lon, a + n_lon, i, j))
    return np.array(verts), quads


def draw_mesh(ctx, cam, V, quads, a, base=BASE, alpha=0.9, grid=True, grid_every=(2, 4)):
    """A shaded closed mesh (painter's order, back faces culled), with optional latitude/longitude lines."""
    cen = V.mean(0)
    P = cam.project(V)
    items = []
    for q in quads:
        idx = q[:4]
        pts = V[list(idx)]
        n = np.cross(pts[2] - pts[0], pts[3] - pts[1])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        fc = pts.mean(0)
        if np.dot(n, fc - cen) < 0:
            n = -n
        if np.dot(n, cam.eye - fc) <= 0:
            continue
        items.append((float(cam.depth(fc)[0]), q, n))
    items.sort(key=lambda x: -x[0])
    for _, q, n in items:
        pts = P[list(q[:4])]
        lam = max(0.0, float(np.dot(n, LIGHT)))
        spec = max(0.0, float(np.dot(n, (LIGHT - cam.fwd) / np.linalg.norm(LIGHT - cam.fwd)))) ** 24
        col = np.clip(base * (0.38 + 0.62 * lam) + 0.08 + 0.55 * spec, 0, 1)
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source_rgba(*col, alpha * a)
        ctx.fill_preserve()
        ctx.set_source_rgba(*col, alpha * a)
        ctx.set_line_width(0.8)
        ctx.stroke()
        if grid:
            i, j = q[4], q[5]
            hi = np.clip(base * 0.5 + 0.55, 0, 1)
            ctx.set_line_width(1.0)
            ctx.set_source_rgba(*hi, 0.22 * a)
            if i % grid_every[0] == 0:
                ctx.move_to(*pts[0])
                ctx.line_to(*pts[1])
                ctx.stroke()
            if j % grid_every[1] == 0:
                ctx.move_to(*pts[0])
                ctx.line_to(*pts[3])
                ctx.stroke()


def arrow3(ctx, cam, p0, p1, col, a, width=4.0, head=22.0, glow=True):
    """An arrow from p0 to p1 in 3D, drawn as a 2D arrow between their projections."""
    P = cam.project(np.array([p0, p1]))
    o, tip = P[0], P[1]
    d = tip - o
    L = math.hypot(*d)
    if L < 3:
        return o, tip
    u = d / L
    nrm = np.array([-u[1], u[0]])
    hd = min(head, 0.45 * L)
    c = rgb(col) if isinstance(col, str) else col
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for lw, al in (((width * 2.3, 0.22), (width, 1.0)) if glow else ((width, 1.0),)):
        ctx.set_source_rgba(*c, al * a)
        ctx.set_line_width(lw)
        ctx.move_to(*o)
        ctx.line_to(*(tip - u * hd * 0.6))
        ctx.stroke()
    ctx.set_source_rgba(*c, a)
    ctx.move_to(*tip)
    ctx.line_to(*(tip - u * hd + nrm * hd * 0.42))
    ctx.line_to(*(tip - u * hd - nrm * hd * 0.42))
    ctx.close_path()
    ctx.fill()
    return o, tip


def fib_sphere(n):
    """n nearly evenly spread unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
