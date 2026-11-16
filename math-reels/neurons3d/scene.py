"""Neurons in 3D. The trained 2 -> 8 -> 8 -> 1 ReLU network of reels 31-32 (nnbuild/nnbuild_train.npz, the 8 + 8 run)
on the two-arm spiral, every field drawn as a surface over the input plane on the 3D stage (stage3d.py).
Act A: the output (height = the probability that a point belongs to the vermilion arm) through training, its
101 saved snapshots, the 400 points riding on it, the accuracy live. Act B: the eight layer-1 neurons, each
max(0, w.x + b): a flat sheet with one fold (the dashed line on the floor). Act C: the eight layer-2 neurons,
max(0, v.n + d): folds of folds. Act D: the output rebuilt from the layer-2 fields, added one at a time in the
order that helps the accuracy most, then back to the untrained surface for the loop. 30 s, no narration.
frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cairo
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import dither, ease_out_back, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
from stage3d import TAU  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg

FPS = 60
N_FRAMES = 1800                                   # 30 s
T_LOOP = N_FRAMES / FPS
NPZ = os.path.join(ROOT, "nnbuild", "nnbuild_train.npz")
SNAP_EVERY = 15                                   # nnbuild/nnsim.py: a snapshot every 15 Adam steps, 1,500 steps

INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GOLD, CYAN = "#FF7A4D", "#5B8DFF", "#F6C56E", "#38BDF8"
PANEL = np.array(rgb("#161C33"))
LIGHT = S3.LIGHT

# timeline (s)
TRAIN = (0.25, 6.3)                                # act A: snapshots 0 -> 100
T_B, T_C, T_D = 7.0, 13.2, 19.4                   # acts B, C, D
TILE_RISE0, TILE_DT, TILE_RISE = 0.7, 0.42, 0.55  # tiles rise one after another
ADD_T0, ADD_DT, ADD_MORPH = 20.6, 0.72, 0.42      # act D: a field added every ADD_DT
T_CLOSE = 26.6
BACK = (29.15, 29.85)                             # the trained surface melts back to the untrained one for the loop

# world: the input square [-L, L]^2 maps to [-S*L, S*L]^2; heights: a probability p sits at HZ * p
S = 1.55
HZ = 1.45
HZ_T = 1.9                                        # tiles: each neuron's field scaled to its own maximum
G_BIG, G_TILE = 80, 40
G_TILE2 = 60                                      # layer-2 tiles: finer, their creases run every which way
BLUR = 1.2                                        # the big surface is drawn smoothed over about one grid cell
SUPER = 3                                         # act D surfaces: exact on a 3x finer grid, cell-averaged (anti-aliased)
ROW_Y = (655, 935, 1215)                          # tile rows: 3, 3, 2
COL_X = ((180, 540, 900), (180, 540, 900), (360, 720))
TILE_BOX = (330, 250)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def az_el(t):
    return -58 + 360 * t / T_LOOP, 31 + 4 * math.sin(2 * TAU * t / T_LOOP)


def tile_az_el(t):
    return -58 + 13 * math.sin(TAU * t / 9.0), 30


def bilinear(field, grid, P):
    """field[i, j] at (x = grid[j], y = grid[i]); P (n, 2) -> values."""
    g0, dg = grid[0], grid[1] - grid[0]
    fx = np.clip((P[:, 0] - g0) / dg, 0, len(grid) - 1.0001)
    fy = np.clip((P[:, 1] - g0) / dg, 0, len(grid) - 1.0001)
    j, i = fx.astype(int), fy.astype(int)
    u, v = fx - j, fy - i
    return (field[i, j] * (1 - u) * (1 - v) + field[i, j + 1] * u * (1 - v) + field[i + 1, j] * (1 - u) * v + field[i + 1, j + 1] * u * v)


def resample(field, grid, g):
    """The field on a g x g grid spanning the same square (bilinear)."""
    xs = np.linspace(grid[0], grid[-1], g)
    XX, YY = np.meshgrid(xs, xs)
    return bilinear(field, grid, np.column_stack([XX.ravel(), YY.ravel()])).reshape(g, g)


def qa_frames():
    ts = [0.0, 1.5, 3.2, 5.0, 6.6, 8.6, 10.6, 12.6, 15.0, 17.2, 19.2, 20.4, 22.2, 24.4, 26.2, 27.6, 29.4, 29.97]
    return [min(N_FRAMES - 1, int(round(t * FPS))) for t in ts]


class Scene(gds.Scene):
    def __init__(self):
        d = np.load(NPZ)
        self.X, self.y, self.grid = d["X"], d["y"], d["grid"]
        self.lim = float(self.grid[-1])
        W1, b1, V, dv, a, c = [d[f"deep_p{i}"] for i in range(6)]
        self.W1, self.b1, self.V, self.dv, self.a, self.c = W1, b1, V, dv, a[:, 0], float(c[0])
        self.snaps = d["deep_fields"].astype(np.float64)               # (101, 96, 96) probabilities over training
        self.snap_acc = d["deep_acc"]
        n_snap = len(self.snaps)
        self.snap_big = np.stack([resample(f, self.grid, G_BIG) for f in self.snaps])
        self.snap_pts = np.stack([bilinear(f, self.grid, self.X) for f in self.snaps])
        # the trained network, exactly, on the grids and at the points
        xs_b = np.linspace(-self.lim, self.lim, G_BIG)
        xs_t = np.linspace(-self.lim, self.lim, G_TILE)
        xs_t2 = np.linspace(-self.lim, self.lim, G_TILE2)
        self.xs_b, self.xs_t, self.xs_t2 = xs_b, xs_t, xs_t2

        def hidden(P):
            n = np.maximum(P @ W1 + b1, 0)
            return n, np.maximum(n @ V + dv, 0)

        xs_f = np.linspace(-self.lim, self.lim, G_BIG * SUPER)
        GB = np.stack(np.meshgrid(xs_f, xs_f), -1).reshape(-1, 2)
        GT = np.stack(np.meshgrid(xs_t, xs_t), -1).reshape(-1, 2)
        GT2 = np.stack(np.meshgrid(xs_t2, xs_t2), -1).reshape(-1, 2)
        nB, mB = hidden(GB)
        nT, _ = hidden(GT)
        _, mT = hidden(GT2)
        nX, mX = hidden(self.X)
        self.l1_max = nT.max(0)
        self.l2_max = mT.max(0)
        assert (self.l1_max > 1e-6).all() and (self.l2_max > 1e-6).all(), "a neuron is off everywhere"
        self.n_tiles = [(nT[:, j] / self.l1_max[j]).reshape(G_TILE, G_TILE) for j in range(8)]
        self.m_tiles = [(mT[:, k] / self.l2_max[k]).reshape(G_TILE2, G_TILE2) for k in range(8)]
        self.mB = mB.reshape(G_BIG * SUPER, G_BIG * SUPER, 8)
        self.mX = mX
        logit_X = mX @ self.a + self.c
        self.acc_full = float(np.mean((logit_X > 0) == (self.y > 0.5)))
        self.right_full = int(np.sum((logit_X > 0) == (self.y > 0.5)))
        # the trained surface from the weights agrees with the last snapshot
        p_full_big = self.to_big(sigmoid(self.mB @ self.a + self.c))
        self.snap_gap = float(np.abs(p_full_big - self.snap_big[-1]).max())

        def acc(idx):
            lg = mX[:, idx] @ self.a[idx] + self.c if idx else np.full(len(self.y), self.c)
            return float(np.mean((lg > 0) == (self.y > 0.5)))

        order = []
        for _ in range(8):
            best = max((j for j in range(8) if j not in order), key=lambda j: acc(order + [j]))
            order.append(best)
        self.order = order
        self.partial_acc = [acc(order[:k]) for k in range(9)]
        self.partial_logit_big = [self.mB[:, :, order[:k]] @ self.a[order[:k]] + self.c if k else np.full((G_BIG * SUPER, G_BIG * SUPER), self.c) for k in range(9)]
        self.partial_logit_X = [mX[:, order[:k]] @ self.a[order[:k]] + self.c if k else np.full(len(self.y), self.c) for k in range(9)]
        self.tile_focal = self._fit_tile_focal()
        self.facts = dict(acc_full=self.acc_full, right_full=self.right_full, n_points=len(self.y), snapshots=n_snap,
                          steps=SNAP_EVERY * (n_snap - 1), snap_acc_first=float(self.snap_acc[0]), snap_acc_last=float(self.snap_acc[-1]),
                          order=[int(o) + 1 for o in order], partial_acc=self.partial_acc, a=[float(x) for x in self.a], c=self.c,
                          snap_vs_weights_max_gap=self.snap_gap, l1_max=self.l1_max.tolist(), l2_max=self.l2_max.tolist())
        self.cues = self._cues()
        self.bg = np.zeros((1920, 1080, 3), np.float32) + S3.BG

    # ------------------------------------------------------------ cues

    def _cues(self):
        hook = [("A neural network learning, in 3D.", "black", 50, INK, 300, 1),
                ("Height: how sure it is that a point is orange.", "dbold", 36, INK2, 390, 0.95)]
        pa = self.partial_acc
        return [
            (0.0, T_B - 0.35, hook, 0, 8, "none"),
            (T_B, T_C - 0.3, [("Take it apart. Layer 1: 8 neurons.", "black", 48, INK, 300, 1),
                              ("Each one: a flat sheet with a single fold.", "dbold", 38, GOLD, 390, 0.95)], 10, 8, "pop"),
            (T_C, T_D - 0.3, [("Layer 2: 8 more neurons.", "black", 50, INK, 300, 1),
                              ("Each one folds the folds below it.", "dbold", 38, CYAN, 390, 0.95)], 10, 8, "pop"),
            (T_D, T_CLOSE - 0.3, [("Now add layer 2 up, one neuron at a time.", "black", 42, INK, 300, 1),
                                  (f"1 neuron: {pa[1]:.1%} right.  All 8: {pa[8]:.1%}.", "dbold", 38, VERM, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, BACK[0] - 0.05, [("No single neuron knows the spiral.", "black", 48, INK, 300, 1),
                                       (f"The sum does: {self.right_full} of {len(self.y)} points right.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (BACK[1] - 0.25, T_LOOP + 0.3, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ state

    def snap_index(self, t):
        """Fractional training snapshot shown at time t (act A, and the loop melt)."""
        if t < TRAIN[0]:
            return 0.0
        if t < TRAIN[1]:
            u = (t - TRAIN[0]) / (TRAIN[1] - TRAIN[0])
            return (len(self.snaps) - 1) * u ** 1.25
        return float(len(self.snaps) - 1)

    def big_state(self, t):
        """(probability grid, probability at the points, alpha, label) of the big surface, or None."""
        if t < T_B:
            s = self.snap_index(t)
            i = int(math.floor(s))
            j = min(i + 1, len(self.snaps) - 1)
            u = s - i
            P = self.snap_big[i] * (1 - u) + self.snap_big[j] * u
            Q = self.snap_pts[i] * (1 - u) + self.snap_pts[j] * u
            a = 1 - smooth(seg(t, T_B - 0.45, T_B))
            return P, Q, a
        if t < T_D - 0.2:
            return None
        k, u = self.n_added(t)
        lg0, lg1 = self.partial_logit_big[max(0, k - 1)], self.partial_logit_big[k]
        lx0, lx1 = self.partial_logit_X[max(0, k - 1)], self.partial_logit_X[k]
        uu = smooth(u)
        P = self.to_big(sigmoid(lg0 * (1 - uu) + lg1 * uu))
        Q = sigmoid(lx0 * (1 - uu) + lx1 * uu)
        if t >= BACK[0]:
            m = smooth(seg(t, *BACK))
            P = P * (1 - m) + self.snap_big[0] * m
            Q = Q * (1 - m) + self.snap_pts[0] * m
        a = smooth(seg(t, T_D - 0.2, T_D + 0.35))
        return P, Q, a

    @staticmethod
    def to_big(Pf):
        """Average the fine-grid probabilities over each cell of the big mesh."""
        return cv2.resize(Pf, (G_BIG, G_BIG), interpolation=cv2.INTER_AREA)

    def n_added(self, t):
        """(fields added so far, progress of the latest one's morph)."""
        if t < ADD_T0:
            return 0, 1.0
        x = (t - ADD_T0) / ADD_DT
        k = min(8, int(x) + 1)
        return k, min(1.0, (t - ADD_T0 - (k - 1) * ADD_DT) / ADD_MORPH)

    def accuracy(self, t):
        if t >= BACK[1] - 0.25:
            return float(self.snap_acc[0]), 0
        if t < T_B:
            s = self.snap_index(t)
            return float(self.snap_acc[int(round(s))]), int(round(s)) * SNAP_EVERY
        k, _ = self.n_added(t)
        return self.partial_acc[k], None

    # ------------------------------------------------------------ 3D surfaces

    def surface_items(self, cam, xs, Z, cols, origin=(0.0, 0.0, 0.0)):
        """Quads of a height field as (depth, polygon (4, 2), rgb) after shading."""
        g = len(xs)
        ox, oy, oz = origin
        XW = S * xs
        XX, YY = np.meshgrid(XW, XW)
        V = np.stack([XX + ox, YY + oy, Z + oz], -1)                        # (g, g, 3)
        P = cam.project(V.reshape(-1, 3)).reshape(g, g, 2)
        p00, p01, p10, p11 = V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:]
        n = np.cross(p11 - p00, p10 - p01)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        n *= np.sign(n[..., 2:3] + 1e-9)                                   # normals point up
        cen = (p00 + p01 + p10 + p11) / 4
        to_eye = cam.eye - cen
        facing = np.sum(n * to_eye, -1) < 0
        n[facing] *= -1                                                    # seen from below: light the side we see
        lam = np.clip(np.sum(n * LIGHT, -1), 0, 1)
        hv = (LIGHT - cam.fwd) / np.linalg.norm(LIGHT - cam.fwd)
        spec = np.clip(np.sum(n * hv, -1), 0, 1) ** 28
        shade = np.clip(cols * (0.42 + 0.68 * lam[..., None]) + 0.32 * spec[..., None], 0, 1)
        depth = cam.depth(cen.reshape(-1, 3))
        quads = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(-1, 4, 2)
        return depth, quads, shade.reshape(-1, 3)

    @staticmethod
    def prob_colours(Pq):
        """Quad colours from probabilities: blue below 0.5, vermilion above, dark at the boundary."""
        cb, cv = np.array(rgb(BLUE)), np.array(rgb(VERM))
        w = np.abs(2 * Pq - 1)[..., None] ** 0.7
        tgt = np.where((Pq >= 0.5)[..., None], cv, cb)
        return PANEL + (tgt - PANEL) * (0.25 + 0.75 * w)

    def draw_big(self, img, t, P, Q, a):
        az, el = az_el(t)
        cam = S3.Camera((0.0, 0.0, 0.62), az, el, dist=10.5, focal=2050.0, centre=(540, 905))
        S3.floor_light(img, cam, 0.0, a, radius=3.4)
        P = cv2.GaussianBlur(P, (0, 0), BLUR, borderType=cv2.BORDER_REPLICATE)
        Z = HZ * P
        Pq = (P[:-1, :-1] + P[:-1, 1:] + P[1:, :-1] + P[1:, 1:]) / 4
        depth, quads, shade = self.surface_items(cam, self.xs_b, Z, self.prob_colours(Pq))
        # the points ride on the surface, a little above it
        pts3 = np.column_stack([S * self.X[:, 0], S * self.X[:, 1], HZ * Q + 0.035])
        pp = cam.project(pts3)
        pd = cam.depth(pts3) - 0.09
        right = (Q > 0.5) == (self.y > 0.5)
        surf, ctx = layer()
        order = np.argsort(-np.concatenate([depth, pd]))
        nq = len(depth)
        floor_square(ctx, cam, self.lim * S, 0.0, a)
        rv = 5.2 * cam.focal / 2050.0
        for idx in order:
            if idx < nq:
                q = quads[idx]
                ctx.move_to(*q[0])
                ctx.line_to(*q[1])
                ctx.line_to(*q[2])
                ctx.line_to(*q[3])
                ctx.close_path()
                ctx.set_source_rgba(*shade[idx], a)
                ctx.fill_preserve()
                ctx.set_line_width(0.6)
                ctx.stroke()
            else:
                k = idx - nq
                x, y = pp[k]
                col = rgb(VERM if self.y[k] > 0.5 else BLUE)
                ctx.arc(x, y, rv, 0, TAU)
                ctx.set_source_rgba(*col, a)
                ctx.fill_preserve()
                ctx.set_source_rgba(0.02, 0.02, 0.05, 0.9 * a)
                ctx.set_line_width(1.1)
                ctx.stroke()
                if not right[k]:
                    ctx.arc(x, y, rv + 4.5, 0, TAU)
                    ctx.set_source_rgba(*rgb(INK), 0.85 * a)
                    ctx.set_line_width(1.6)
                    ctx.stroke()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ tiles

    def _fit_tile_focal(self):
        """The largest focal length at which every tile, over the whole sway, fits its box."""
        L = S * self.lim
        corners = np.array([[sx * L, sy * L, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, HZ_T)])
        worst_w, worst_h = 0.0, 0.0
        for t in np.linspace(0, 9.0, 37):
            az, el = tile_az_el(t)
            cam = S3.Camera((0.0, 0.0, HZ_T * 0.42), az, el, dist=10.5, focal=1.0, centre=(0, 0))
            p = cam.project(corners)
            worst_w = max(worst_w, np.ptp(p[:, 0]))
            worst_h = max(worst_h, np.ptp(p[:, 1]))
        return min(TILE_BOX[0] / worst_w, TILE_BOX[1] / worst_h)

    def tile_slots(self):
        return [(x, y) for r, y in enumerate(ROW_Y) for x in COL_X[r]]

    def draw_tiles(self, img, t, which, t0, a_all):
        az, el = tile_az_el(t)
        surf, ctx = layer()
        items_d, items_q, items_c, items_a = [], [], [], []
        col = np.array(rgb(GOLD if which == 1 else CYAN))
        fields = self.n_tiles if which == 1 else self.m_tiles
        for j, (cx, cy) in enumerate(self.tile_slots()):
            rise = seg(t, t0 + TILE_RISE0 + TILE_DT * j, t0 + TILE_RISE0 + TILE_DT * j + TILE_RISE)
            if rise <= 0:
                continue
            h = ease_out_back(rise)
            cam = S3.Camera((0.0, 0.0, HZ_T * 0.42), az, el, dist=10.5, focal=self.tile_focal, centre=(cx, cy))
            F = fields[j]
            Fq = (F[:-1, :-1] + F[:-1, 1:] + F[1:, :-1] + F[1:, 1:]) / 4
            cols = PANEL + (col - PANEL) * (0.18 + 0.82 * Fq[..., None] ** 0.8)
            a = a_all * min(1.0, rise * 3)
            floor_square(ctx, cam, self.lim * S, 0.0, a * 0.9)
            if which == 1:
                self.fold_tile(ctx, cam, j, h, a)
                self.fold_line(ctx, cam, j, a)
            else:
                depth, quads, shade = self.surface_items(cam, self.xs_t2, HZ_T * h * F, cols)
            for dpt, q, s_ in (zip(depth, quads, shade) if which == 2 else ()):
                items_d.append(dpt + 1000.0 * (2 - (j // 3)))              # whole tiles drawn back row first
                items_q.append(q)
                items_c.append(s_)
                items_a.append(a)
            font(ctx, "dbold", 24)
            lab = f"{'n' if which == 1 else 'm'}{j + 1}"
            ctx.set_source_rgba(*rgb(INK2), 0.9 * a)
            ctx.move_to(cx - TILE_BOX[0] / 2 + 8, cy - TILE_BOX[1] / 2 + 6)
            ctx.show_text(lab)
        for k in np.argsort(-np.array(items_d)) if items_d else []:
            q = items_q[k]
            ctx.move_to(*q[0])
            ctx.line_to(*q[1])
            ctx.line_to(*q[2])
            ctx.line_to(*q[3])
            ctx.close_path()
            ctx.set_source_rgba(*items_c[k], items_a[k])
            ctx.fill_preserve()
            ctx.set_line_width(0.6)
            ctx.stroke()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def fold_tile(self, ctx, cam, j, h, a):
        """Neuron j exactly: the square cut by its fold line into the flat part (zero) and a planar ramp."""
        w = self.W1[:, j]
        b = float(self.b1[j])
        L = self.lim
        sq = [(-L, -L), (L, -L), (L, L), (-L, L)]

        def val(p):
            return w[0] * p[0] + w[1] * p[1] + b

        pos, neg = [], []
        for k in range(4):
            p, q = sq[k], sq[(k + 1) % 4]
            vp, vq = val(p), val(q)
            (pos if vp >= 0 else neg).append(p)
            if (vp >= 0) != (vq >= 0):
                u = vp / (vp - vq)
                x = (p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1]))
                pos.append(x)
                neg.append(x)
        zs = HZ_T * h / self.l1_max[j]
        col = np.array(rgb(GOLD))
        base = PANEL + (col - PANEL) * 0.18
        for poly, ramp in ((neg, False), (pos, True)):
            if len(poly) < 3:
                continue
            P3 = np.array([(S * x, S * y, zs * max(0.0, val((x, y))) if ramp else 0.0) for x, y in poly])
            pp = cam.project(P3)
            if ramp:
                n = np.array([-zs * w[0] / S, -zs * w[1] / S, 1.0])
                n /= np.linalg.norm(n)
                lam = max(0.0, float(np.dot(n, LIGHT)))
                lo = P3[np.argmin(P3[:, 2])]
                hi = P3[np.argmax(P3[:, 2])]
                g0, g1 = cam.project(np.array([lo, hi]))
                grad = cairo.LinearGradient(*g0, *g1)
                c_lo = np.clip(base * (0.42 + 0.68 * lam), 0, 1)
                c_hi = np.clip(col * (0.42 + 0.68 * lam) + 0.06, 0, 1)
                grad.add_color_stop_rgba(0, *c_lo, a)
                grad.add_color_stop_rgba(1, *c_hi, a)
                src = grad
            else:
                src = cairo.SolidPattern(*np.clip(base * 1.1, 0, 1), a)
            ctx.move_to(*pp[0])
            for p in pp[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source(src)
            ctx.fill()
            if ramp:
                # level lines at a quarter, half and three quarters of the neuron's maximum
                for lv in (0.25, 0.5, 0.75):
                    seg_pts = []
                    for k in range(4):
                        p, q = sq[k], sq[(k + 1) % 4]
                        vp, vq = val(p) - lv * self.l1_max[j], val(q) - lv * self.l1_max[j]
                        if (vp >= 0) != (vq >= 0):
                            u = vp / (vp - vq)
                            seg_pts.append((p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])))
                    if len(seg_pts) == 2:
                        e = cam.project(np.array([(S * x, S * y, zs * lv * self.l1_max[j]) for x, y in seg_pts]))
                        ctx.move_to(*e[0])
                        ctx.line_to(*e[1])
                        ctx.set_source_rgba(1, 1, 1, 0.22 * a)
                        ctx.set_line_width(1.2)
                        ctx.stroke()

    def fold_line(self, ctx, cam, j, a):
        """Where neuron j's sum is zero: the fold, dashed on the floor of its tile."""
        w1, w2, b = float(self.W1[0, j]), float(self.W1[1, j]), float(self.b1[j])
        nn_ = math.hypot(w1, w2)
        L = self.lim
        pts = []
        for s in np.linspace(-3 * L, 3 * L, 400):
            x0, y0 = -b * w1 / nn_ ** 2, -b * w2 / nn_ ** 2
            x, y = x0 - w2 / nn_ * s, y0 + w1 / nn_ * s
            if abs(x) <= L and abs(y) <= L:
                pts.append((S * x, S * y, 0.004))
        if len(pts) < 2:
            return
        p = cam.project(np.array([pts[0], pts[-1]]))
        ctx.save()
        ctx.set_dash([6.0, 5.0])
        ctx.set_source_rgba(*rgb(INK), 0.85 * a)
        ctx.set_line_width(1.8)
        ctx.move_to(*p[0])
        ctx.line_to(*p[1])
        ctx.stroke()
        ctx.restore()

    # ------------------------------------------------------------ readouts

    def readouts(self, ctx, t, a):
        acc, step = self.accuracy(t)
        if t < T_B or step == 0:
            items = (("TRAINING STEP", f"{step:,}", None), ("ACCURACY", f"{acc:.1%}", acc), ("POINTS RIGHT", f"{int(round(acc * 400))} / 400", None))
            note = "the 400 points ride on the surface: orange should sit high, blue low"
        elif t < T_C:
            items = (("LAYER", "1", None), ("NEURONS", "8", None), ("EACH ONE", "1 fold", None))
            note = "zero on one side of its line (dashed), a ramp on the other"
        elif t < T_D:
            items = (("LAYER", "2", None), ("NEURONS", "8", None), ("EACH ONE", "a fold of folds", None))
            note = "a weighted mix of the 8 sheets below it, cut off at zero"
        else:
            k, _ = self.n_added(t)
            items = (("NEURONS ADDED", f"{k} / 8", k / 8), ("ACCURACY", f"{acc:.1%}", acc), ("POINTS RIGHT", f"{int(round(acc * 400))} / 400", None))
            note = "the layer-2 surfaces, weighted and added, then squashed to 0-1"
        Y_LAB, Y_VAL, Y_BAR = 1452, 1514, 1540
        for (lab, val, frac), xc in zip(items, (190, 540, 890)):
            font(ctx, "bold", 21)
            tr = 0.12 * 21
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(INK2), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46 if len(val) < 9 else 34)
            w = ctx.text_extents(val).x_advance
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.move_to(xc - w / 2, Y_VAL)
            ctx.show_text(val)
            if frac is not None:
                ctx.set_line_width(6)
                ctx.set_source_rgba(*rgb("#2A3148"), a)
                ctx.move_to(xc - 110, Y_BAR)
                ctx.line_to(xc + 110, Y_BAR)
                ctx.stroke()
                ctx.set_source_rgba(*rgb(VERM), a)
                ctx.move_to(xc - 110, Y_BAR)
                ctx.line_to(xc - 110 + 220 * min(1.0, frac), Y_BAR)
                ctx.stroke()
        font(ctx, "regular", 27)
        w = ctx.text_extents(note).x_advance
        ctx.set_source_rgba(*rgb(INK2), 0.9 * a)
        ctx.move_to(540 - w / 2, 1612)
        ctx.show_text(note)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        img = self.bg.copy()
        st = self.big_state(t)
        if st is not None and st[2] > 0:
            self.draw_big(img, t, *st)
        if T_B - 0.35 <= t < T_C + 0.3:
            a = smooth(seg(t, T_B - 0.35, T_B + 0.1)) * (1 - smooth(seg(t, T_C - 0.25, T_C + 0.3)))
            self.draw_tiles(img, t, 1, T_B - 0.6, a)
        if T_C - 0.25 <= t < T_D + 0.1:
            a = smooth(seg(t, T_C - 0.25, T_C + 0.2)) * (1 - smooth(seg(t, T_D - 0.4, T_D + 0.1)))
            self.draw_tiles(img, t, 2, T_C - 0.6, a)
        surf, ctx = layer()
        self.readouts(ctx, t, 1.0)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "A REAL 2 → 8 → 8 → 1 NETWORK   ·   400 SPIRAL POINTS   ·   ALL COMPUTED"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


def floor_square(ctx, cam, L, z, a):
    c = cam.project(np.array([[-L, -L, z], [L, -L, z], [L, L, z], [-L, L, z]]))
    ctx.move_to(*c[0])
    for p in c[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgba(0.06, 0.08, 0.16, 0.85 * a)
    ctx.fill_preserve()
    ctx.set_source_rgba(*rgb(INK3), 0.55 * a)
    ctx.set_line_width(1.2)
    ctx.stroke()


if __name__ == "__main__":
    import time

    import cv2

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sc = Scene()
    print("facts", sc.facts)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)[..., ::-1]
        times.append(time.time() - t0)
        cv2.imwrite(f"{out}/f{fr:04d}.png", im)
        ims.append(cv2.resize(im, (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
