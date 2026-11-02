"""PCA on real data. The 3D stage of reels 38-40 (stage3d.py), narrated by Kokoro (voicekit.py).

Fisher's 150 iris flowers (iris.csv), three measurements each (sepal length, petal length, petal width, in cm): one
point per flower in 3D. The SVD of the centred data gives the three principal directions and how much of the spread
lies along each. The cloud is flattened onto the best plane (the smallest direction dropped), turned to face the
camera (three measurements become two), squashed onto one line, and then coloured by species, which PCA never saw:
setosa sits apart. Every number on screen comes from the data. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import BLUE, CYAN, GOLD, GREEN, INK, INK2, INK3, MAGENTA, TAU, VERM, paste  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg

FPS = 60

# ---------------------------------------------------------------- the data and every number the reel states
_rows = [ln.strip().split(",") for ln in open(os.path.join(HERE, "iris.csv")) if ln[0].isdigit()]
RAW = np.array([[float(r[0]), float(r[2]), float(r[3])] for r in _rows])          # sepal length, petal length, petal width (cm)
SPECIES_NAMES = ["setosa", "versicolor", "virginica"]
SPECIES = np.array([SPECIES_NAMES.index(r[4]) for r in _rows])
FEATURES = ["sepal length", "petal length", "petal width"]
N = len(RAW)
MEAN = RAW.mean(0)
XC = RAW - MEAN
_U, SV, VT = np.linalg.svd(XC, full_matrices=False)
V = VT.T.copy()
if V[1, 0] < 0:                                   # PC1 points towards longer petals
    V[:, 0] *= -1
if V[2, 1] < 0:                                   # PC2: a fixed, readable orientation
    V[:, 1] *= -1
Q = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)   # world x <- sepal length, y <- petal width, z (up) <- petal length
if np.linalg.det(Q @ V) < 0:                      # right-handed in the world, so it can be turned to face the camera
    V[:, 2] *= -1
STD = SV / math.sqrt(N - 1)                       # spread (standard deviation) along each direction, cm
SHARE = SV ** 2 / np.sum(SV ** 2)
KEEP2, KEEP1 = SHARE[:2].sum(), SHARE[0]
PC = XC @ V                                       # principal coordinates
MOVE_MEAN, MOVE_MAX = float(np.abs(PC[:, 2]).mean()), float(np.abs(PC[:, 2]).max())
SET_MAX = float(PC[SPECIES == 0, 0].max())
REST_MIN = float(PC[SPECIES > 0, 0].min())
GAP = REST_MIN - SET_MAX
SPECIES_C = (VERM, GREEN, GOLD)
PC_C = (CYAN, MAGENTA, BLUE)

SCALE = 0.56                                      # world units per cm
Z0 = np.array([0.0, 0.0, 0.35])
Z_FLOOR = -2.2
CARD = (70, 1430, 940, 250)


def pct(v, nd=1):
    return f"{100 * v:.{nd}f}"


SCRIPT = [
    "A hundred and fifty flowers, three measurements each. Squash them flat, and lose almost nothing.",
    "Sepal length, petal length and petal width, in centimeters. Every flower is one point in 3D.",
    "Which directions does this cloud spread along the most? PCA answers that with the SVD of the data.",
    f"The first direction holds {round(100 * SHARE[0]):d} percent of the spread. The second, {round(100 * SHARE[1]):d} percent. The third, less than one.",
    f"Drop the third. Every point slides onto the best plane, on average just a seventh of a centimeter, and {int(100 * KEEP2):d} percent of the spread stays.",
    "Turn the plane to face us. Three measurements just became two.",
    f"Even a single line keeps {round(100 * KEEP1):d} percent. One number per flower.",
    "PCA never saw which species each flower is. Color them in: setosa sits on its own, more than a centimeter from the rest.",
    "That's PCA: the SVD of your data. Keep the directions with the most spread, and drop the rest.",
]
TIMING = os.path.join(HERE, "narration_timing.json")

# ---------------------------------------------------------------- the timeline, sized to the narration
# 0 hook, 1 the cloud, 2 directions, 3 shares, 4 flatten, 5 face us, 6 one line, 7 species, 8 close
SEG_MIN = [3.5, 6.5, 7.0, 7.5, 8.5, 6.0, 6.0, 8.5, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[9]
LOOP_FADE = (T_END, T_END + 0.55)
N_FRAMES = int(round((T_END + 0.7) * FPS))
AXES_OUT = (SEG[5], SEG[5] + 1.2)
PCS_IN = (SEG[2] + 1.0, SEG[2] + 2.4)
PLANE_IN = (SEG[4] + 0.2, SEG[4] + 1.2)
FLAT3 = (SEG[4] + 1.0, SEG[4] + 3.2)
FACE = (SEG[5] + 0.2, SEG[5] + 2.6)
FLAT2 = (SEG[6] + 0.3, SEG[6] + 2.3)
UNFLAT2 = (SEG[7] + 0.2, SEG[7] + 1.6)            # back to the plane, so the colours have room
COLOUR = (SEG[7] + 3.4, SEG[7] + 4.5)
BACK = (SEG[8] + 2.2, T_END - 0.2)                # everything returns to the opening cloud for the loop


def w(t, span):
    return smooth(seg(t, *span))


def state(t):
    back = w(t, BACK)
    k3 = w(t, FLAT3) * (1 - back)
    k2 = w(t, FLAT2) * (1 - w(t, UNFLAT2))
    kf = w(t, FACE) * (1 - back)
    col = w(t, COLOUR) * (1 - back)
    axes = 1 - w(t, AXES_OUT) * (1 - back)
    pcs = ease_out_cubic(seg(t, *PCS_IN)) * (1 - smooth(seg(t, SEG[7], SEG[7] + 0.8)))
    plane = ease_out_cubic(seg(t, *PLANE_IN)) * (1 - back)
    return dict(k3=k3, k2=k2, kf=kf, col=col, axes=axes, pcs=pcs, plane=plane, back=back)


def qa_frames():
    fr = []
    for k in range(9):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.3 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr[:17] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n=N, features=FEATURES, mean=MEAN.round(4).tolist(), std_cm=STD.round(4).tolist(), share=SHARE.round(5).tolist(),
                          keep2=float(KEEP2), keep1=float(KEEP1), directions=V.T.round(4).tolist(), move_mean_cm=MOVE_MEAN, move_max_cm=MOVE_MAX,
                          setosa_pc1_max=SET_MAX, others_pc1_min=REST_MIN, gap_cm=GAP, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self.bg = np.zeros((1920, 1080, 3), np.float32) + S3.BG

    def _cues(self):
        hook = [("150 flowers, 3 measurements.", "black", 50, INK, 300, 1), ("Squash them flat. Lose almost nothing.", "dbold", 38, INK2, 390, 0.95)]
        S_ = SEG
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Each flower is a point in 3D.", "black", 48, INK, 300, 1), ("Sepal length, petal length, petal width (cm).", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Which way does it spread?", "black", 50, INK, 300, 1), ("PCA: the SVD of the data.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [(f"One direction holds {pct(SHARE[0])}%.", "black", 48, INK, 300, 1), (f"The others: {pct(SHARE[1])}% and {pct(SHARE[2])}%.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Flatten onto the best plane.", "black", 48, INK, 300, 1), (f"{pct(KEEP2)}% of the spread stays.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("Turn it to face us.", "black", 50, INK, 300, 1), ("3 measurements became 2.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [(f"Even one line keeps {pct(KEEP1)}%.", "black", 48, INK, 300, 1), ("One number per flower.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], S_[8] - 0.3, [("PCA never saw the species.", "black", 48, INK, 300, 1), (f"Yet setosa sits apart: a {GAP:.2f} cm gap.", "dbold", 40, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[8], T_END - 0.3, [("PCA: keep the big directions.", "black", 48, INK, 300, 1), ("Drop the rest.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END + 0.1, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ camera and geometry

    def camera(self, t):
        L = N_FRAMES / FPS
        az = -40 + 12 * math.sin(TAU * t / L)
        el = 16 + 3 * math.sin(2 * TAU * t / L)
        return S3.Camera(Z0 + np.array([0, 0, -0.05]), az, el, dist=10.8, focal=2050.0, centre=(540, 905))

    def turn(self, cam, kf):
        """The rotation (about the centre of the cloud) that turns PC1 to screen-right, PC2 to screen-up, PC3 to us."""
        if kf <= 1e-6:
            return np.eye(3)
        B = Q @ V
        T = np.column_stack([cam.right, cam.up, -cam.fwd]) @ B.T
        if kf >= 1 - 1e-6:
            return T
        return Slerp([0, 1], Rotation.from_matrix([np.eye(3), T]))(kf).as_matrix()

    def world(self, feat_c, Rw):
        """Centred feature vectors (cm) -> world points."""
        return Z0 + (np.atleast_2d(feat_c) @ Q.T * SCALE) @ Rw.T

    # ------------------------------------------------------------ drawing

    def draw_world(self, img, t, a, st):
        cam = self.camera(t)
        Rw = self.turn(cam, st["kf"])
        P = PC.copy()
        P[:, 2] *= 1 - st["k3"]
        P[:, 1] *= 1 - st["k2"]
        pts_c = P @ V.T                                     # back to centred feature space
        W3 = self.world(pts_c, Rw)
        S3.floor_light(img, cam, Z_FLOOR, a)
        S3.floor_shadow(img, cam, W3, Z_FLOOR, a, strength=0.35)
        surf, ctx = layer()
        o = cam.project(Z0)[0]
        # the measurement axes, through the average flower
        self.axis_labels = []
        if st["axes"] > 0.01:
            ctx.set_dash([5.0, 7.0])
            for i in range(3):
                e = np.zeros(3)
                e[i] = 1.0
                ext = (RAW[:, i].max() - MEAN[i]) + 0.6
                lo = -(MEAN[i] - RAW[:, i].min()) - 0.4
                pa, pb = cam.project(self.world(np.array([lo * e, ext * e]), Rw))
                ctx.set_source_rgba(*rgb(INK3), 0.8 * st["axes"] * a)
                ctx.set_line_width(1.3)
                ctx.move_to(*pa)
                ctx.line_to(*pb)
                ctx.stroke()
                self.axis_labels.append((FEATURES[i], cam.project(self.world((ext + 0.25) * e, Rw))[0], st["axes"] * a))
            ctx.set_dash([])
        # the best plane (PC1, PC2) and, later, the best line (PC1)
        if st["plane"] > 0.01:
            a1, b1 = PC[:, 0].min() - 0.35, PC[:, 0].max() + 0.35
            a2, b2 = (PC[:, 1].min() - 0.3) * (1 - 0.85 * st["k2"]), (PC[:, 1].max() + 0.3) * (1 - 0.85 * st["k2"])
            corners = np.array([u1 * V[:, 0] + u2 * V[:, 1] for u1, u2 in ((a1, a2), (b1, a2), (b1, b2), (a1, b2))])
            cp = cam.project(self.world(corners, Rw))
            ctx.move_to(*cp[0])
            for p in cp[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source_rgba(*rgb(CYAN), 0.07 * st["plane"] * a)
            ctx.fill_preserve()
            ctx.set_source_rgba(*rgb(CYAN), 0.35 * st["plane"] * a)
            ctx.set_line_width(1.2)
            ctx.stroke()
        # the points, far to near
        dep = cam.depth(W3)
        P2 = cam.project(W3)
        base = np.array([0.62, 0.84, 0.98])
        cols = np.array([np.array(rgb(c)) for c in SPECIES_C])
        k = st["col"]
        for i in np.argsort(-dep):
            c = base * (1 - k) + cols[SPECIES[i]] * k
            r = 6.4 * (10.8 / max(dep[i], 1.0))
            x, y = P2[i]
            ctx.set_source_rgba(*(c * 0.55), 0.95 * a)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*c, 0.95 * a)
            ctx.arc(x - 0.25 * r, y - 0.25 * r, 0.72 * r, 0, TAU)
            ctx.fill()
        # the principal directions, through the centre, two spreads long each way
        self.pc_labels = []
        if st["pcs"] > 0.01:
            for i in range(3):
                show = st["pcs"] * (1 - (st["k3"] if i == 2 else 0)) * (1 - (st["k2"] if i == 1 else 0))
                if show < 0.01:
                    continue
                L = 2.0 * STD[i]
                tip = self.world(L * V[:, i], Rw)[0]
                tail = self.world(-L * V[:, i], Rw)[0]
                S3.arrow3(ctx, cam, Z0, tip, PC_C[i], show * a, width=4.4, head=22)
                S3.arrow3(ctx, cam, Z0, tail, PC_C[i], 0.5 * show * a, width=2.6, head=0.1, glow=False)
                self.pc_labels.append((i, cam.project(tip)[0], o, show * a))
        # the setosa gap, measured along PC1 once the colours are in
        self.gap = None
        gap_a = ease_out_cubic(seg(t, COLOUR[1] - 0.2, COLOUR[1] + 0.6)) * (1 - st["back"]) * st["kf"]
        if gap_a > 0.01:
            ys = PC[:, 1].max() * 1.25
            ends = []
            for v1, col in ((SET_MAX, VERM), (REST_MIN, GREEN)):
                pa, pb = cam.project(self.world(np.array([v1 * V[:, 0] - ys * V[:, 1], v1 * V[:, 0] + ys * V[:, 1]]), Rw))
                ctx.set_dash([4.0, 5.0])
                ctx.set_source_rgba(*rgb(col), 0.85 * gap_a * a)
                ctx.set_line_width(1.6)
                ctx.move_to(*pa)
                ctx.line_to(*pb)
                ctx.stroke()
                ctx.set_dash([])
                ends.append(pb)
            ctx.set_source_rgba(*rgb(INK), 0.9 * gap_a * a)
            ctx.set_line_width(1.8)
            ctx.move_to(*ends[0])
            ctx.line_to(*ends[1])
            ctx.stroke()
            self.gap = ((ends[0] + ends[1]) / 2, gap_a * a)
        ctx.set_source_rgba(1, 1, 1, 0.9 * a * (1 - st["kf"]))           # the average flower (hidden once it faces us)
        ctx.arc(*o, 4.0, 0, TAU)
        ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self.species_at = []
        if k > 0.01:
            for s in range(3):
                c = P2[SPECIES == s]
                self.species_at.append((s, c[:, 0].mean(), c[:, 1].min(), k * a))

    def labels(self, img, t, a, st):
        for name, p, al in self.axis_labels:
            x = min(max(float(p[0]), 110.0), 970.0)
            y = min(max(float(p[1]), 480.0), 1380.0)
            paste(img, rf"$\mathrm{{{name.replace(' ', chr(92) + ' ')}}}$", 22, x, y + 8, INK3, al, align="c", halo=4)
        placed = []
        for i, tip, o, al in self.pc_labels:
            lab = rf"$\mathrm{{PC}}{i + 1}:\ {STD[i]:.2f}\ \mathrm{{cm}}$"
            if t >= SEG[3]:
                lab = rf"$\mathrm{{PC}}{i + 1}:\ {pct(SHARE[i])}\%$"
            d = np.asarray(tip, float) - np.asarray(o, float)
            n = float(np.hypot(*d))
            d = d / n if n > 1e-6 else np.array([0.0, -1.0])
            m, bl, wd = S3.tex(lab, 22)
            hh = m.shape[0]
            cx, cy = np.asarray(tip, float) + d * (14 + abs(d[0]) * wd / 2 + abs(d[1]) * hh / 2)
            cx = min(max(cx, 24 + wd / 2), 1056 - wd / 2)
            for _ in range(20):
                box = (cx - wd / 2 - 4, cy - hh / 2 - 3, cx + wd / 2 + 4, cy + hh / 2 + 3)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                    break
                cy += 6
            placed.append((cx - wd / 2 - 4, cy - hh / 2 - 3, cx + wd / 2 + 4, cy + hh / 2 + 3))
            paste(img, lab, 22, cx, cy - hh / 2 + bl, PC_C[i], al, align="c", halo=4)
        for s, x, ytop, al in self.species_at:
            paste(img, rf"$\mathrm{{{SPECIES_NAMES[s]}}}$", 24, x, ytop - 22, SPECIES_C[s], al * st["kf"], align="c", halo=4)
        if self.gap is not None:
            p, al = self.gap
            paste(img, rf"${GAP:.2f}\ \mathrm{{cm}}$", 24, p[0], p[1] - 14, INK, al, align="c", halo=4)

    def card(self, img, t, a, st):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if t < SEG[2] or t >= T_END - 0.3:
            name = r"$X\ =$"
            lw = S3.tex(name, 30)[2]
            M = RAW[:3]
            mw = S3.matrix_block(img, M, 0, 0, size=24, row_h=36, measure=True, keep_zeros=True, nd=1)
            x0 = 540 - (lw + 16 + mw) / 2
            paste(img, name, 30, x0, y + 98, INK, a)
            S3.matrix_block(img, M, x0 + lw + 16, y + 90, size=24, row_h=36, a=a, keep_zeros=True, nd=1)
            paste(img, r"$\vdots$", 24, x0 + lw + 16 + mw / 2, y + 160, INK2, a, align="c")
            paste(img, r"$\mathrm{150\ flowers\ \times\ 3:\ sepal\ length,\ petal\ length,\ petal\ width\ (cm)}$", 21, 540, y + h - 20, INK2, a, align="c")
        elif t < SEG[3]:
            paste(img, r"$X_c = U\,\Sigma\,V^{\top}$", 44, 540, y + 104, INK, a, align="c")
            paste(img, r"$X_c:\ \mathrm{each\ measurement\ minus\ its\ average.}\ \ \mathrm{columns\ of}\ V:\ \mathrm{the\ directions}$", 21, 540, y + 168, INK2, a, align="c")
            paste(img, rf"$\mathrm{{spread\ along\ them:}}\ \ {STD[0]:.2f},\ {STD[1]:.2f},\ {STD[2]:.2f}\ \mathrm{{cm}}$", 24, 540, y + h - 22, INK2,
                  a * ease_out_cubic(seg(t, PCS_IN[1] - 0.3, PCS_IN[1] + 0.3)), align="c")
        elif t < SEG[4]:
            paste(img, r"$\mathrm{share}_i = \frac{\sigma_i^2}{\sigma_1^2 + \sigma_2^2 + \sigma_3^2}$", 36, 540, y + 100, INK, a, align="c")
            xs = (250, 540, 830)
            for i in range(3):
                al = a * ease_out_cubic(seg(t, SEG[3] + 0.4 + 0.5 * i, SEG[3] + 0.9 + 0.5 * i))
                paste(img, rf"${pct(SHARE[i])}\%$", 32, xs[i], y + 182, PC_C[i], al, align="c")
            paste(img, r"$\sigma_i:\ \mathrm{the\ singular\ values\ (the\ stretches\ of\ the\ SVD\ reel)}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[5]:
            paste(img, rf"$\mathrm{{kept}} = \frac{{\sigma_1^2 + \sigma_2^2}}{{\sigma_1^2 + \sigma_2^2 + \sigma_3^2}} = {pct(KEEP2)}\%$", 34, 540, y + 108, INK, a, align="c")
            paste(img, rf"$\mathrm{{each\ point\ moves\ along\ PC3:\ on\ average}}\ {MOVE_MEAN:.2f}\ \mathrm{{cm,\ at\ most}}\ {MOVE_MAX:.2f}\ \mathrm{{cm}}$", 21, 540, y + h - 26, INK2,
                  a * ease_out_cubic(seg(t, FLAT3[0], FLAT3[0] + 0.6)), align="c")
        elif t < SEG[6]:
            paste(img, r"$(\mathrm{sepal},\ \mathrm{petal},\ \mathrm{petal}) \ \rightarrow\ (\mathrm{PC1},\ \mathrm{PC2})$", 32, 540, y + 104, INK, a, align="c")
            paste(img, r"$\mathrm{each\ flower:\ its\ two\ coordinates\ along\ the\ first\ two\ directions}$", 21, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[7]:
            paste(img, rf"$\mathrm{{kept}} = \frac{{\sigma_1^2}}{{\sigma_1^2 + \sigma_2^2 + \sigma_3^2}} = {pct(KEEP1)}\%$", 34, 540, y + 108, INK, a, align="c")
            paste(img, r"$\mathrm{one\ number\ per\ flower:\ its\ position\ along\ PC1}$", 21, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[8]:
            paste(img, rf"$\mathrm{{setosa:\ PC1}} \leq {{{SET_MAX:.2f}}}\qquad \mathrm{{the\ others:\ PC1}} \geq {{{REST_MIN:.2f}}}$", 26, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\mathrm{{gap}} = {GAP:.2f}\ \mathrm{{cm}}$", 34, 540, y + 168, VERM, a * ease_out_cubic(seg(t, COLOUR[1] - 0.2, COLOUR[1] + 0.4)), align="c")
            paste(img, r"$\mathrm{PCA\ used\ only\ the\ measurements;\ the\ colours\ come\ afterwards}$", 21, 540, y + h - 18, INK3, a, align="c")
        else:
            paste(img, r"$\mathrm{PCA} = \mathrm{SVD\ of\ the\ centred\ data}$", 38, 540, y + 108, INK, a, align="c")
            paste(img, r"$\mathrm{keep\ the\ directions\ with\ the\ biggest}\ \sigma_i$", 22, 540, y + h - 26, INK2, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        a = max(fade, ease_out_cubic(seg(t, LOOP_FADE[0] + 0.15, LOOP_FADE[1] + 0.1)))
        st = state(t)
        img = self.bg.copy()
        self.draw_world(img, t, a, st)
        self.labels(img, t, a, st)
        self.card(img, t, a, st)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "FISHER'S IRIS DATA   ·   150 FLOWERS   ·   COMPUTED FROM THE DATA"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w_ / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    import cv2

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sc = Scene()
    print("facts", sc.facts)
    for line in SCRIPT:
        print("  >", line)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        ims.append(cv2.resize(sc.frame(fr)[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
        times.append(time.time() - t0)
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
