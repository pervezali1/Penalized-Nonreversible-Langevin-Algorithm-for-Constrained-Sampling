"""The determinant is a volume. A dark stage with a soft floor light and glossy 3D solids, and a maths card in
Computer Modern (matplotlib mathtext). A unit cube is reshaped by a 3x3 matrix that changes live: the cube's
three edges from the corner are the matrix's columns (coloured to match the card), the solid is the image of
the cube, and the card shows the matrix and its determinant, computed every frame. Stretch (diag 2, 1, 1.5:
det 3), shear (the same volume), a general matrix (det 2.233, checked with a million random points), then the
third column tipped into the plane of the other two (det 0: flat, no inverse) and pushed through (det < 0:
mirrored, drawn inside-out in magenta), and back. 50 s. frame(f) -> uint8 1080x1920 RGB."""

import functools
import importlib.util
import math
import os
import sys

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)

FPS = 60
N_FRAMES = 3000                       # 50 s
LOOP_FADE = (49.3, 49.85)
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, GREEN, GOLD, MAGENTA, CYAN = "#FF7A4D", "#34D399", "#F6C56E", "#F472B6", "#38BDF8"
COL_C = (VERM, GREEN, GOLD)           # the three columns / edges
seg = gds.seg
TAU = 2 * math.pi
BASE = np.array([0.12, 0.62, 0.95])   # the solid's cyan
INSIDE = np.array([0.93, 0.36, 0.62])  # inside-out (det < 0)
LIGHT = np.array([-0.45, -0.35, 0.82]) / np.linalg.norm([-0.45, -0.35, 0.82])
Z_FLOOR = -0.95
STAGE_C = (540, 960)                  # where the camera target lands on screen
FOCAL, DIST = 2150.0, 10.5
CARD = (70, 1430, 940, 250)
SWING_AZ, SWING_EL = 150.0, -12.0

I3 = np.eye(3)
S = np.diag([2.0, 1.0, 1.5])
SH = np.array([[2.0, 0.9, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.5]])
G = np.array([[2.0, 0.9, 0.4], [0.3, 1.0, 0.5], [-0.2, 0.4, 1.5]])
_N = np.cross(G[:, 0], G[:, 1])
_N /= np.linalg.norm(_N)                                     # the normal of the plane of the first two edges
F = G.copy()
F[:, 2] = G[:, 2] - np.dot(G[:, 2], _N) * _N                 # the third edge pushed straight into that plane
NEG = G.copy()
NEG[:, 2] = G[:, 2] - 2 * np.dot(G[:, 2], _N) * _N           # and straight through: its mirror image
# (start time, end time, from, to) of each morph; between morphs the matrix holds
MORPHS = [(9.4, 11.6, I3, S), (15.6, 17.6, S, SH), (21.8, 24.0, SH, G), (29.0, 32.4, G, F), (36.6, 39.0, F, NEG), (43.8, 47.2, NEG, I3)]
PHASES = [(0.0, "id"), (9.0, "stretch"), (15.2, "shear"), (21.4, "general"), (28.6, "flatten"), (32.6, "flat"), (36.2, "neg"), (43.4, "close")]

CUBE = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)       # index = 4i + 2j + k
FACES = [(0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def matrix_at(t):
    M = I3.copy()
    for t0, t1, A, B in MORPHS:
        if t >= t1:
            M = B
        elif t >= t0:
            k = smooth((t - t0) / (t1 - t0))
            M = A + (B - A) * k
            break
        else:
            break
    return M


def fmt(v):
    v = 0.0 if abs(v) < 0.005 else float(v)
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def monte_carlo(M, n=1_000_000, seed=38):
    rng = np.random.default_rng(seed)
    V = CUBE @ M.T
    lo, hi = V.min(0), V.max(0)
    P = lo + (hi - lo) * rng.random((n, 3))
    Q = np.linalg.solve(M, P.T).T
    inside = np.all((Q >= 0) & (Q <= 1), axis=1)
    return float(inside.mean() * np.prod(hi - lo))


# ---------------------------------------------------------------- Computer Modern text (matplotlib mathtext)

_PARSER = MathTextParser("agg")


@functools.lru_cache(maxsize=4096)
def tex(s, size):
    """Alpha mask (float32) and baseline row of the string s in Computer Modern at size (points at 100 dpi)."""
    r = _PARSER.parse(s, dpi=100, prop=FontProperties(size=size, family="serif"))
    a = np.asarray(r.image, np.float32) / 255.0
    return a, int(round(r.height - r.depth)), r.width


def paste(img, s, size, x, base, col, alpha=1.0, align="l"):
    """Composite string s onto img with its baseline at y = base; align l/c/r on x. Returns the width."""
    if alpha <= 0.003:
        return tex(s, size)[2]
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


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


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


class Scene(gds.Scene):
    def __init__(self):
        self.det = {k: float(np.linalg.det(M)) for k, M in (("I", I3), ("S", S), ("SH", SH), ("G", G), ("F", F), ("NEG", NEG))}
        self.mc = monte_carlo(G)
        self.facts = dict(det=self.det, mc_G=self.mc, G=G.tolist(), F=np.round(F, 4).tolist(), NEG=np.round(NEG, 4).tolist())
        self.cues = self._cues()
        self.bg = np.zeros((H, W, 3), np.float32) + np.array(rgb("#020308"), np.float32)

    # ------------------------------------------------------------ cues

    def _cues(self):
        d = self.det
        hook = [("The determinant is a volume.", "black", 54, INK, 300, 1), ("Watch a matrix reshape a cube.", "black", 40, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.8, 8.7, [("A 3 × 3 matrix moves the three edges of a cube.", "black", 40, INK, 300, 1), ("Its columns are where the edges land.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.0, 14.9, [("Stretch it: 2, 1 and 1.5.", "black", 48, INK, 300, 1), (f"The volume grows by exactly that: {d['S']:g}.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (15.2, 21.1, [("Now slant it.", "black", 52, INK, 300, 1), (f"Same base, same height: the volume stays {d['SH']:g}.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (21.4, 28.3, [("Any matrix: its volume is the determinant.", "black", 42, INK, 300, 1), (f"det = {d['G']:.2f}. A million random points measure {self.mc:.2f}.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (28.6, 32.3, [("Tip the third edge into the plane of the other two.", "black", 38, INK, 300, 1), ("The volume drains away.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (32.6, 35.9, [("det = 0: crushed flat.", "black", 52, MAGENTA, 300, 1), ("Nothing can un-flatten it. No inverse.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (36.2, 43.1, [("Keep going: it comes out the other side.", "black", 42, INK, 300, 1), (f"The mirror image: same volume, det = {d['NEG']:.2f}.".replace("-", "−"), "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (43.4, 49.0, [("The determinant: how much a matrix scales volume.", "black", 38, INK, 300, 1), ("0 means flattened. Negative means flipped.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (49.4, 50.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ camera

    def camera(self, t, M):
        # for the mirror image the camera swings round to the side the mirrored solid grows towards
        w = smooth(seg(t, 35.8, 38.8)) * (1 - smooth(seg(t, 43.4, 46.6)))
        az = math.radians(-58 + 14 * math.sin(TAU * t / 50.0) + SWING_AZ * w)
        el = math.radians(23 + 3 * math.sin(TAU * t / 25.0) + SWING_EL * w)
        # the target follows the solid's centre, eased (computed from the current matrix)
        tgt = (CUBE @ M.T).mean(0)
        tgt = np.array([tgt[0], tgt[1], 0.5 * tgt[2] + 0.2])
        eye = tgt + DIST * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, fwd, right, up, tgt

    def project(self, cam, P):
        eye, fwd, right, up, _ = cam
        P = np.atleast_2d(P) - eye
        z = P @ fwd
        return np.stack([STAGE_C[0] + FOCAL * (P @ right) / z, STAGE_C[1] - FOCAL * (P @ up) / z], 1)

    # ------------------------------------------------------------ the stage

    def draw_floor(self, img, cam, M, a):
        """The pool of light on the floor and the solid's soft shadow, straight into img."""
        _, _, _, _, tgt = cam
        surf, ctx = layer()
        n = 64
        for k in range(26, 0, -1):
            r = 3.9 * k / 26
            ring = np.array([[tgt[0] + r * math.cos(TAU * i / n), tgt[1] + r * math.sin(TAU * i / n), Z_FLOOR] for i in range(n)])
            pts = self.project(cam, ring)
            ctx.set_source_rgba(0.04, 0.10, 0.26, 0.055 * a)
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.fill()
        frgb, fa = surf_rgba(surf)
        over(img, frgb, fa)
        # the shadow: the solid's outline dropped straight down onto the floor, blurred
        V = CUBE @ M.T
        foot = np.column_stack([V[:, 0] + 0.25 * (V[:, 2] - Z_FLOOR), V[:, 1] + 0.18 * (V[:, 2] - Z_FLOOR), np.full(8, Z_FLOOR)])
        hp = hull(self.project(cam, foot))
        if len(hp) >= 3:
            mask = np.zeros((H // 2, W // 2), np.float32)
            cv2.fillPoly(mask, [np.round(np.array(hp) / 2).astype(np.int32)], 1.0)
            mask = cv2.GaussianBlur(mask, (0, 0), 14)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            img *= (1 - 0.75 * a * mask)[..., None]

    def draw_solid(self, ctx, cam, M, t, a):
        eye = cam[0]
        V = CUBE @ M.T
        P = self.project(cam, V)
        det = float(np.linalg.det(M))
        base = BASE if det >= 0 else INSIDE
        cen = V.mean(0)
        flat = abs(det) < 0.02
        vis_faces = []
        for f in FACES:
            q = V[list(f)]
            n = np.cross(q[1] - q[0], q[3] - q[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            fc = q.mean(0)
            if np.dot(n, fc - cen) < 0:
                n = -n
            if np.dot(n, eye - fc) > 0 or flat:
                vis_faces.append((f, n, np.dot(fc - eye, cam[1])))
        vis_faces.sort(key=lambda x: -x[2])
        # the ghost unit cube, dashed
        ctx.set_dash([6.0, 6.0])
        Pg = self.project(cam, CUBE)
        ctx.set_line_width(1.4)
        ctx.set_source_rgba(1, 1, 1, 0.22 * a)
        for i, j in EDGES:
            ctx.move_to(*Pg[i])
            ctx.line_to(*Pg[j])
        ctx.stroke()
        ctx.set_dash([])
        # faces: lambert shading and a vertical gradient for gloss
        for f, n, _ in vis_faces:
            q = P[list(f)]
            lam = max(0.0, float(np.dot(n, LIGHT)))
            col = base * (0.42 + 0.58 * lam) + 0.10
            top = q[np.argmin(q[:, 1])]
            bot = q[np.argmax(q[:, 1])]
            g = cairo.LinearGradient(top[0], top[1], bot[0], bot[1])
            g.add_color_stop_rgba(0, *np.clip(col * 1.12 + 0.04, 0, 1), (0.95 if not flat else 0.78) * a)
            g.add_color_stop_rgba(1, *np.clip(col * 0.86, 0, 1), (0.95 if not flat else 0.78) * a)
            ctx.move_to(*q[0])
            for p in q[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source(g)
            ctx.fill()
        # edges: visible ones glow, hidden ones dashed
        vis_set = set()
        for f, _, _ in vis_faces:
            for k in range(4):
                e = tuple(sorted((f[k], f[(k + 1) % 4])))
                vis_set.add(e)
        hi = np.clip(base * 0.5 + 0.55, 0, 1)
        for (i, j) in EDGES:
            if (i, j) in vis_set:
                for lw, al in ((7.0, 0.16), (2.2, 0.95)):
                    ctx.set_source_rgba(*hi, al * a)
                    ctx.set_line_width(lw)
                    ctx.move_to(*P[i])
                    ctx.line_to(*P[j])
                    ctx.stroke()
            else:
                ctx.set_dash([5.0, 6.0])
                ctx.set_source_rgba(*hi, 0.35 * a)
                ctx.set_line_width(1.4)
                ctx.move_to(*P[i])
                ctx.line_to(*P[j])
                ctx.stroke()
                ctx.set_dash([])
        # the three columns: edges from the corner, coloured, with arrowheads
        o = P[0]
        for c, idx in enumerate((4, 2, 1)):
            tip = P[idx]
            d = tip - o
            L = math.hypot(*d)
            if L < 4:
                continue
            u = d / L
            nrm = np.array([-u[1], u[0]])
            col = rgb(COL_C[c])
            for lw, al in ((9.0, 0.22), (4.0, 1.0)):
                ctx.set_source_rgba(*col, al * a)
                ctx.set_line_width(lw)
                ctx.move_to(*o)
                ctx.line_to(*(tip - u * 12))
                ctx.stroke()
            ctx.set_source_rgba(*col, a)
            ctx.move_to(*tip)
            ctx.line_to(*(tip - u * 22 + nrm * 9))
            ctx.line_to(*(tip - u * 22 - nrm * 9))
            ctx.close_path()
            ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.95 * a)
        ctx.arc(*o, 5.0, 0, TAU)
        ctx.fill()
        return P

    def tip_labels(self, img, P, a):
        for c, idx in enumerate((4, 2, 1)):
            o, tip = P[0], P[idx]
            d = tip - o
            L = math.hypot(*d)
            if L < 30:
                continue
            u = d / L
            pos = tip + u * 30
            paste(img, f"$c_{c + 1}$", 22, pos[0], pos[1] + 10, COL_C[c], a, align="c")

    # ------------------------------------------------------------ the card

    def phase(self, t):
        if t >= 49.0:
            return "id"                      # the loop: back to the opening card
        cur = PHASES[0][1]
        for t0, name in PHASES:
            if t >= t0:
                cur = name
        return cur

    def draw_card_bg(self, ctx, a):
        x, y, w, h = CARD
        ctx.set_source_rgba(0.035, 0.05, 0.11, 0.92 * a)
        pill(ctx, x, y, w, h, 20)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.25, 0.32, 0.5, 0.85 * a)
        ctx.set_line_width(1.3)
        ctx.stroke()

    def draw_matrix_brackets(self, ctx, x0, x1, y0, y1, a):
        ctx.set_source_rgba(0.95, 0.96, 0.98, a)
        ctx.set_line_width(2.2)
        for xb, s in ((x0, 1), (x1, -1)):
            ctx.move_to(xb + 10 * s, y0)
            ctx.line_to(xb, y0)
            ctx.line_to(xb, y1)
            ctx.line_to(xb + 10 * s, y1)
            ctx.stroke()

    def card(self, img, t, M, a):
        """The card: det [matrix] = value, and a second line for the phase. Text pasted into img after the background."""
        if a <= 0:
            return
        x, y, w, h = CARD
        det = float(np.linalg.det(M))
        ph = self.phase(t)
        size, row_h = 26, 46
        entries = [[fmt(M[r, c]) for c in range(3)] for r in range(3)]
        col_w = [max(tex(f"${entries[r][c]}$", size)[2] for r in range(3)) for c in range(3)]
        col_w = [max(cw, 52) for cw in col_w]
        gap = 26
        mw = sum(col_w) + 2 * gap + 2 * 22
        lw_det = tex(r"$\det$", 30)[2]
        val = fmt(det)
        lw_val = tex(f"$= {val}$", 30)[2]
        total = lw_det + 16 + mw + 16 + lw_val
        x0 = 540 - total / 2
        cy = y + 104
        surf, ctx = layer()
        mx0 = x0 + lw_det + 16
        self.draw_matrix_brackets(ctx, mx0, mx0 + mw, cy - 1.5 * row_h + 6, cy + 1.5 * row_h - 2, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        paste(img, r"$\det$", 30, x0, cy + 10, INK, a)
        cx = mx0 + 22
        for c in range(3):
            for r in range(3):
                paste(img, f"${entries[r][c]}$", size, cx + col_w[c] / 2, cy + (r - 1) * row_h + 10, COL_C[c], a, align="c")
            cx += col_w[c] + gap
        paste(img, f"$= {val}$", 30, mx0 + mw + 16, cy + 10, MAGENTA if abs(det) < 0.005 or det < 0 else CYAN, a)
        # the second line
        line = {"id": r"the unit cube: $\mathrm{volume} = 1$",
                "stretch": r"$\mathrm{volume} = 2 \times 1 \times 1.5 = 3$",
                "shear": r"slanted: same base, same height, $\mathrm{volume} = 3$",
                "general": rf"$1{{,}}000{{,}}000$ random points measure $\mathrm{{volume}} = {self.mc:.2f}$",
                "flatten": r"$c_3$ tips into the plane of $c_1$ and $c_2$",
                "flat": r"$\mathrm{volume} = 0$: flat, so no inverse",
                "neg": rf"$\det < 0$: the mirror image, the same volume ${abs(self.det['NEG']):.2f}$",
                "close": r"$\det M$ = how much $M$ scales volume"}[ph]
        paste(img, line, 22, 540, y + h - 26, INK2, a, align="c")

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        a = max(fade, ease_out_cubic(seg(t, LOOP_FADE[0] + 0.15, LOOP_FADE[1] + 0.1)))
        img = self.bg.copy()
        M = matrix_at(t)
        cam = self.camera(t, M)
        self.draw_floor(img, cam, M, a)
        surf, ctx = layer()
        P = self.draw_solid(ctx, cam, M, t, a)
        self.draw_card_bg(ctx, a)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "THE DETERMINANT OF A 3 × 3 MATRIX   ·   COMPUTED LIVE, EVERY FRAME"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self.tip_labels(img, P, a)
        self.card(img, t, M, a)
        return dither(img, f)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s")
    print("facts", sc.facts)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 300, 560, 690, 800, 1000, 1150, 1400, 1600, 1800, 1900, 2030, 2250, 2400, 2600, 2750, 2850, 2990]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (300, 1000, 1400, 2030, 2400):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
