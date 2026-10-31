"""Backpropagation: one training step in slow motion, on the dark theme of reels 1-22 and 31-32, with no
formulas on screen. Two networks from reel 31 (bpsim.py): the one-neuron net after it has converged and
the 8 + 8 net in the middle of training. For each, one example the network gets wrong is sent forward
(gold pulses along the edges, neurons lit by their activation, a guess at the output), the error is shown
(guess minus answer), the blame is sent backward (magenta pulses, every edge then glowing by the size of
the gradient on its weight, neurons that were off getting none), and a plain gradient step is taken on
that one example: the weights tween to their new values, and the field, the guess and the accuracy are
recomputed live from the tweened weights. 46 s. frame(f) -> uint8 1080x1920 RGB."""

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
sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
from nnsim import GRID, forward  # noqa: E402
from textkit import (H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba,  # noqa: E402
                     tracked_width)

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)

FPS = 60
N_FRAMES = 2760                       # 46 s
LOOP_FADE = (45.3, 45.85)
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6"
GOLD = "#F6C56E"
PANEL_R = 18
seg, lerp = gds.seg, gds.lerp
TAU = 2 * math.pi
BG_TOP, BG_BOT = "#0E1226", "#05060C"
GRID_C, GRID_A = "#FFFFFF", 0.045
CARD_FILL, CARD_EDGE = "#121830", "#3A4460"
PANEL_BASE = "#161C33"
POINT_EDGE = "#05060C"
PANEL = (70, 1120, 520)               # the field: x, y, size
DIAG = (60, 470, 960, 560)            # the diagram: x, y, w, h
READ_CX, READ_Y0, READ_GAP = 835, 1158, 124
NPZ = os.path.join(HERE, "backprop_sim.npz")
T_CLOSE = 40.0

# the timeline of each stage (absolute seconds): fwd[li] is the window in which the signal crosses weight layer li,
# bwd[k] the window in which the blame crosses weight layer (last - k); guess/answer/error are when the readouts
# appear; nudge is the window over which the weights move to their new values
STAGES = [
    dict(key="one", sizes=[1], t0=0.6, t1=20.0, label="ONE NEURON, TRAINED",
         fwd=[(3.0, 4.0), (4.0, 4.9)], guess=4.9, answer=5.6, error=6.4,
         bwd=[(9.6, 10.6), (10.6, 11.7)], nudge=(14.8, 16.8)),
    dict(key="deep", sizes=[8, 8], t0=20.0, t1=46.5, label="8 + 8 NEURONS, MID-TRAINING",
         fwd=[(23.4, 24.3), (24.3, 25.2), (25.2, 26.0)], guess=26.0, answer=26.6, error=27.2,
         bwd=[(28.0, 28.9), (28.9, 29.9), (29.9, 30.9)], nudge=(34.4, 36.6)),
]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def mix(c0, c1, k):
    a, b = rgb(c0), rgb(c1)
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3))


class Scene(gds.Scene):
    def __init__(self):
        d = np.load(NPZ)
        self.X, self.y, self.grid = d["X"], d["y"], d["grid"]
        self.lim = float(self.grid[-1])
        gxx, gyy = np.meshgrid(self.grid, self.grid)
        self.GX = np.stack([gxx.ravel(), gyy.ravel()], 1)
        self.stages = []
        for st in STAGES:
            k = st["key"]
            nl = len(st["sizes"]) + 1
            s = dict(st, x=d[f"{k}_x"], t=float(d[f"{k}_t"]), i=int(d[f"{k}_i"]), p0=float(d[f"{k}_p0"]), p1=float(d[f"{k}_p1"]),
                     acc0=float(d[f"{k}_acc0"]), acc1=float(d[f"{k}_acc1"]), step=float(d[f"{k}_step"]), stop=int(d[f"{k}_stop"]),
                     n_par=int(d[f"{k}_n_par"]), n_w=int(d[f"{k}_n_w"]), f0=d[f"{k}_f0"], f1=d[f"{k}_f1"],
                     W=[d[f"{k}_W{li}"] for li in range(nl)], b=[d[f"{k}_b{li}"] for li in range(nl)],
                     gW=[d[f"{k}_gW{li}"] for li in range(nl)], gb=[d[f"{k}_gb{li}"] for li in range(nl)],
                     delta=[d[f"{k}_delta{li}"] for li in range(nl)], act=[d[f"{k}_act{li}"] for li in range(nl + 1)])
            s["W_after"] = [w - s["step"] * g for w, g in zip(s["W"], s["gW"])]
            s["b_after"] = [b - s["step"] * g for b, g in zip(s["b"], s["gb"])]
            for li in range(nl):
                assert np.allclose(s["W_after"][li], d[f"{k}_W{li}_after"])
            s["gmax"] = max(float(np.abs(g).max()) for g in s["gW"])
            s["dW"] = [a - b for a, b in zip(s["W_after"], s["W"])]
            s["dmax"] = max(float(np.abs(g).max()) for g in s["dW"])
            s["layers"] = [2] + st["sizes"] + [1]
            s["pos"] = self.layout(s["layers"])
            s["n_off"] = [int(np.sum(a <= 0)) for a in s["act"][1:nl]]
            self.stages.append(s)
        self.facts = {s["key"]: dict(i=s["i"], x=[round(float(v), 3) for v in s["x"]], p0=round(s["p0"], 3), p1=round(s["p1"], 3), acc0=round(s["acc0"], 4),
                                     acc1=round(s["acc1"], 4), step=s["step"], stop=s["stop"], n_par=s["n_par"], n_w=s["n_w"], n_off=s["n_off"],
                                     error=round(s["p0"] - s["t"], 3), n0=int(round(s["acc0"] * len(self.y))), n1=int(round(s["acc1"] * len(self.y)))) for s in self.stages}
        self.cues = self._cues()
        self._build_grid()
        self._palette()
        self._panel_masks()

    # ------------------------------------------------------------ cues (every number from the simulation)

    def _cues(self):
        f = self.facts
        hook = [("Backpropagation: how a network learns from one mistake.", "black", 40, INK, 300, 1), ("Forward: a guess. Backward: the blame.", "black", 38, INK2, 390, 1)]
        o, dp = f["one"], f["deep"]
        cues = [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.7, 5.7, [("One neuron, and one example it gets wrong.", "black", 42, INK, 300, 1), ("Forward: the inputs flow through to a guess.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.0, 9.2, [(f"The guess: {o['p0']:.2f}. The answer: 1.", "black", 44, INK, 300, 1), (f"Error {o['error']:+.2f}. That number is the only teacher.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (9.5, 14.4, [("Backward: the blame flows back along every edge.", "black", 40, INK, 300, 1), ("Each weight learns how much of the error was its fault.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (14.7, 19.5, [("Then every weight moves a little, against its blame.", "black", 40, INK, 300, 1), (f"Five numbers nudged once: the guess goes {o['p0']:.2f} → {o['p1']:.2f}.  Right: {o['n0']} → {o['n1']}.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (20.0, 23.2, [(f"Now the 8 + 8 network, mid-training: {dp['n0']} of 400 right.", "black", 40, INK, 300, 1), ("One point it gets wrong. The same three moves.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (23.5, 27.6, [("Forward again, layer by layer.", "black", 44, INK, 300, 1), ("Lit neurons are on. Dark ones are off: they passed nothing.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (27.9, 34.0, [("Backward: the blame fans out through every path.", "black", 40, INK, 300, 1), ("A neuron that was off gets none. Nothing went through it.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (34.3, 39.6, [(f"{dp['n_par']} numbers, each nudged once.", "black", 44, INK, 300, 1), (f"Guess {dp['p0']:.2f} → {dp['p1']:.2f}.   Right: {dp['n0']} → {dp['n1']} of 400.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, 45.0, [("Forward: guess. Backward: blame. Nudge every weight.", "black", 38, INK, 300, 1), ("Repeat a few thousand times. That is all training is.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (45.4, 46.1, hook, 12, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ geometry and the live network

    def layout(self, layers):
        x0, y0, w, h = DIAG
        cy = y0 + h / 2
        pos = []
        for li, n in enumerate(layers):
            cx = x0 + 70 + (w - 140) * li / (len(layers) - 1)
            gap = 58 if n > 3 else 84
            pos.append([(cx, cy + (i - (n - 1) / 2) * gap) for i in range(n)])
        return pos

    def live(self, s, t):
        """The weights at time t (tweened over the nudge window), and the field, accuracy and guess computed from them."""
        k = smooth(seg(t, *s["nudge"]))
        ps = []
        for li in range(len(s["W"])):
            ps.append(s["W"][li] + (s["W_after"][li] - s["W"][li]) * k)
            ps.append(s["b"][li] + (s["b_after"][li] - s["b"][li]) * k)
        field = sigmoid(forward(ps, self.GX)[0]).reshape(GRID, GRID).astype(np.float32)
        pX = sigmoid(forward(ps, self.X)[0])
        acc = float(np.mean((pX > 0.5) == (self.y > 0.5)))
        p = float(sigmoid(forward(ps, s["x"][None, :])[0][0]))
        return ps, field, acc, p, k

    def stage_alpha(self, s, t):
        return ease_out_cubic(seg(t, s["t0"], s["t0"] + 0.5)) * (1 - smooth(seg(t, s["t1"] - 0.5, s["t1"])))

    # ------------------------------------------------------------ the field

    def _panel_masks(self):
        px, py, n = PANEL
        surf, ctx = layer(n, n)
        pill(ctx, 0, 0, n, n, PANEL_R)
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()
        _, a = surf_rgba(surf)
        self.panel_mask = a.astype(np.float32)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
        self.panel_vign = (1 - 0.22 * np.clip(r, 0, 1.2) ** 2).astype(np.float32)
        pad = 48
        big = np.zeros((n + 2 * pad, n + 2 * pad), np.float32)
        big[pad + 10:pad + 10 + n, pad:pad + n] = self.panel_mask
        self.panel_shadow = cv2.GaussianBlur(big, (0, 0), 16) * 0.75
        self.panel_pad = pad

    def _palette(self):
        w = np.array(rgb(PANEL_BASE), np.float32)
        cb, cv = np.array(rgb(BLUE), np.float32), np.array(rgb(VERM), np.float32)
        lut = np.empty((256, 3), np.float32)
        for i, p in enumerate(np.linspace(0, 1, 256)):
            if p < 0.5:
                lut[i] = w + (cb - w) * 0.78 * (0.5 - p) * 2
            else:
                lut[i] = w + (cv - w) * 0.78 * (p - 0.5) * 2
        self.lut = lut

    def field_rgb(self, field):
        px, py, n = PANEL
        up = cv2.resize(field[::-1].astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)   # row 0 of the grid is y = -lim
        idx = np.clip((up * 255).round().astype(np.int32), 0, 255)
        return self.lut[idx], up

    def to_panel(self, x, y):
        px, py, n = PANEL
        return px + (x + self.lim) / (2 * self.lim) * n, py + (self.lim - y) / (2 * self.lim) * n

    def draw_panel(self, ctx, s, t, a, ps, field):
        px, py, n = PANEL
        ctx.save()
        pill(ctx, px, py, n, n, PANEL_R)
        ctx.clip()
        # fold lines of the first layer, one per neuron
        W1, B1 = ps[0], ps[1]
        ctx.set_source_rgba(*rgb(INK), (0.22 if len(s["sizes"]) > 1 else 0.5) * a)
        ctx.set_line_width(1.6)
        ctx.set_dash([7.0, 5.0])
        L = self.lim * 3
        for j in range(W1.shape[1]):
            w1, w2, b = float(W1[0, j]), float(W1[1, j]), float(B1[j])
            nn_ = math.hypot(w1, w2)
            if nn_ < 1e-6:
                continue
            x0, y0 = -b * w1 / nn_ ** 2, -b * w2 / nn_ ** 2
            dx, dy = -w2 / nn_, w1 / nn_
            ctx.move_to(*self.to_panel(x0 - dx * L, y0 - dy * L))
            ctx.line_to(*self.to_panel(x0 + dx * L, y0 + dy * L))
            ctx.stroke()
        ctx.set_dash([])
        # the decision boundary: the 0.5 contour of the upsampled field
        _, up = self.field_rgb(field)
        cs, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for lw, al in ((9.0, 0.16), (4.0, 0.30), (2.0, 0.95)):
            ctx.set_source_rgba(*rgb(INK), al * a)
            ctx.set_line_width(lw)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            for c in cs:
                if len(c) < 6:
                    continue
                pts = c[:, 0, :]
                ctx.move_to(px + pts[0][0] + 0.5, py + pts[0][1] + 0.5)
                for x, y in pts[1:]:
                    ctx.line_to(px + x + 0.5, py + y + 0.5)
                ctx.close_path()
                ctx.stroke()
        # the data
        for (x, y), c in zip(self.X, self.y):
            sx, sy = self.to_panel(x, y)
            col = rgb(VERM if c > 0.5 else BLUE)
            ctx.set_source_rgba(*col, 0.16 * a)
            ctx.arc(sx, sy, 6.0, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*col, 0.95 * a)
            ctx.arc(sx, sy, 2.9, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(POINT_EDGE), 0.8 * a)
            ctx.set_line_width(0.8)
            ctx.arc(sx, sy, 2.9, 0, TAU)
            ctx.stroke()
        # the example: a gold ring, breathing
        sx, sy = self.to_panel(*s["x"])
        br = 0.5 + 0.5 * math.sin(TAU * (t - s["t0"]) / 1.6)
        ctx.set_source_rgba(*rgb(GOLD), 0.22 * a)
        ctx.arc(sx, sy, 22 + 4 * br, 0, TAU)
        ctx.fill()
        ctx.set_source_rgba(*rgb(GOLD), (0.75 + 0.25 * br) * a)
        ctx.set_line_width(2.6)
        ctx.arc(sx, sy, 11 + 2 * br, 0, TAU)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(VERM), a)
        ctx.arc(sx, sy, 4.2, 0, TAU)
        ctx.fill()
        ctx.restore()
        for lw, al in ((5.0, 0.08), (1.4, 0.55)):
            ctx.set_source_rgba(*rgb(INK2), al * a)
            ctx.set_line_width(lw)
            pill(ctx, px + 0.5, py + 0.5, n - 1, n - 1, PANEL_R)
            ctx.stroke()
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, "x₁", px + n / 2 - 10, py + n + 30, tr)
        show_tracked(ctx, "x₂", px - 34, py + n / 2 + 8, tr)
        lab = "OUTPUT FIELD  ·  THE EXAMPLE RINGED"
        w = tracked_width(ctx, lab, tr)
        show_tracked(ctx, lab, px + n / 2 - w / 2, py - 16, tr)

    # ------------------------------------------------------------ the diagram

    def draw_diagram(self, ctx, s, t, a, ps, k_nudge, p_live):
        x0, y0, w, h = DIAG
        layers, pos = s["layers"], s["pos"]
        nl = len(layers) - 1
        n0, n1 = s["nudge"]
        blame_hold = 1 - smooth(seg(t, n0, n0 + 1.2))          # the blame glow goes once it has been used
        # signal strength leaving each node (forward), per layer, normalised within the layer
        fwd_str = []
        for li in range(nl):
            act = s["act"][li]
            v = np.abs(act) if li == 0 else np.maximum(act, 0)
            fwd_str.append(v / (v.max() + 1e-9))
        # --- edges: colour by the sign of the live weight, thickness by its size (per layer)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for li in range(nl):
            Wl = ps[2 * li]
            wmax = float(np.abs(Wl).max()) + 1e-9
            dense = layers[li] * layers[li + 1] > 20
            for i, p in enumerate(pos[li]):
                for j, q in enumerate(pos[li + 1]):
                    wv = float(Wl[i, j])
                    st = abs(wv) / wmax
                    col = VERM if wv >= 0 else BLUE
                    aa = ((0.16 + 0.5 * st) if dense else (0.28 + 0.6 * st)) * a
                    ctx.set_source_rgba(*rgb(col), 0.2 * aa)
                    ctx.set_line_width((1.6 + 4.0 * st) if dense else (2.5 + 6.0 * st))
                    ctx.move_to(*p)
                    ctx.line_to(*q)
                    ctx.stroke()
                    ctx.set_source_rgba(*rgb(col), aa)
                    ctx.set_line_width((0.8 + 1.8 * st) if dense else (1.0 + 2.6 * st))
                    ctx.move_to(*p)
                    ctx.line_to(*q)
                    ctx.stroke()
        # --- the blame on every weight: a magenta glow that builds as the blame crosses the layer and stays until the nudge
        for kk, (ba, bb) in enumerate(s["bwd"]):
            li = nl - 1 - kk
            g = ease_out_cubic(seg(t, ba + 0.3, bb + 0.2)) * blame_hold * a
            if g <= 0:
                continue
            G = s["gW"][li]
            for i, p in enumerate(pos[li]):
                for j, q in enumerate(pos[li + 1]):
                    st = (abs(float(G[i, j])) / s["gmax"]) ** 0.6
                    if st < 0.03:
                        continue
                    ctx.set_source_rgba(*rgb(MAGENTA), 0.22 * st * g)
                    ctx.set_line_width(3.0 + 9.0 * st)
                    ctx.move_to(*p)
                    ctx.line_to(*q)
                    ctx.stroke()
                    ctx.set_source_rgba(*rgb(MAGENTA), 0.9 * st * g)
                    ctx.set_line_width(1.2 + 2.2 * st)
                    ctx.move_to(*p)
                    ctx.line_to(*q)
                    ctx.stroke()
        # --- the nudge: a white shimmer on every edge, by how much its weight moved
        sh = ease_out_cubic(seg(t, n0, n0 + 0.35)) * (1 - smooth(seg(t, n0 + 0.6, n1))) * a
        if sh > 0:
            for li in range(nl):
                D = s["dW"][li]
                for i, p in enumerate(pos[li]):
                    for j, q in enumerate(pos[li + 1]):
                        st = (abs(float(D[i, j])) / s["dmax"]) ** 0.7
                        if st < 0.03:
                            continue
                        ctx.set_source_rgba(*rgb(INK), 0.18 * st * sh)
                        ctx.set_line_width(3.0 + 7.0 * st)
                        ctx.move_to(*p)
                        ctx.line_to(*q)
                        ctx.stroke()
                        ctx.set_source_rgba(*rgb(INK), 0.85 * st * sh)
                        ctx.set_line_width(1.0 + 1.6 * st)
                        ctx.move_to(*p)
                        ctx.line_to(*q)
                        ctx.stroke()
        # --- pulses: gold forward, magenta backward, each with a short trail
        def pulses(li, ph, strength, col, back):
            for i, p in enumerate(pos[li]):
                for j, q in enumerate(pos[li + 1]):
                    st = strength(i, j)
                    if st < 0.04:
                        continue
                    src, dst = (q, p) if back else (p, q)
                    env = min(1.0, ph / 0.08, (1 - ph) / 0.08) if 0 < ph < 1 else 0.0
                    for m in range(6):
                        u = ph - m * 0.035
                        if u < 0:
                            break
                        x = src[0] + (dst[0] - src[0]) * u
                        y = src[1] + (dst[1] - src[1]) * u
                        al = env * st * a * (1 - m / 6) ** 1.5
                        r_ = 2.4 if m == 0 else 1.6
                        if m == 0:
                            ctx.set_source_rgba(*rgb(col), 0.28 * al)
                            ctx.arc(x, y, 6.0, 0, TAU)
                            ctx.fill()
                        ctx.set_source_rgba(*rgb(col), 0.95 * al)
                        ctx.arc(x, y, r_, 0, TAU)
                        ctx.fill()

        for li, (fa, fb) in enumerate(s["fwd"]):
            ph = seg(t, fa, fb)
            if 0 < ph < 1:
                pulses(li, ease_out_cubic(ph) * 0.999, lambda i, j, li=li: float(fwd_str[li][i]), GOLD, False)
        for kk, (ba, bb) in enumerate(s["bwd"]):
            li = nl - 1 - kk
            ph = seg(t, ba, bb)
            if 0 < ph < 1:
                pulses(li, ease_out_cubic(ph) * 0.999, lambda i, j, li=li: (abs(float(s["gW"][li][i, j])) / s["gmax"]) ** 0.6, MAGENTA, True)
        # --- nodes
        dmax = max(float(np.abs(dl).max()) for dl in s["delta"]) + 1e-9
        for li, layer_pos in enumerate(pos):
            r = 20 if layers[li] <= 3 else 15
            hidden = 0 < li < nl
            out = li == nl
            # when the forward signal has arrived at this layer, and when the blame has
            lit = 1.0 if li == 0 else ease_out_cubic(seg(t, s["fwd"][li - 1][1] - 0.05, s["fwd"][li - 1][1] + 0.35))
            if out:
                lit = ease_out_cubic(seg(t, s["guess"], s["guess"] + 0.4))
            if out:
                blamed = ease_out_cubic(seg(t, s["error"], s["error"] + 0.4))
            elif hidden:
                ba, bb = s["bwd"][nl - 1 - li]
                blamed = ease_out_cubic(seg(t, bb - 0.05, bb + 0.35))
            else:
                blamed = 0.0
            blamed *= blame_hold
            acts = s["act"][li]
            amax = float(np.max(np.maximum(acts, 0))) + 1e-9 if hidden else 1.0
            for i, (x, y) in enumerate(layer_pos):
                fill = 0.0
                if hidden:
                    fill = (0.3 + 0.7 * float(max(acts[i], 0) / amax) ** 0.6) * lit if acts[i] > 0 else 0.0   # any neuron that is on is visibly lit
                elif li == 0:
                    fill = float(abs(s["x"][i]) / (np.abs(s["x"]).max() + 1e-9)) * 0.6
                dl = float(abs(s["delta"][li - 1][i]) / dmax) if li > 0 else 0.0
                # halo: gold by activation, magenta by blame
                g = cairo.RadialGradient(x, y, r * 0.6, x, y, r * 2.8)
                g.add_color_stop_rgba(0, *rgb(GOLD), 0.30 * fill * a)
                g.add_color_stop_rgba(1, *rgb(GOLD), 0.0)
                ctx.set_source(g)
                ctx.arc(x, y, r * 2.8, 0, TAU)
                ctx.fill()
                if dl * blamed > 0:
                    g = cairo.RadialGradient(x, y, r * 0.8, x, y, r * 3.2)
                    g.add_color_stop_rgba(0, *rgb(MAGENTA), 0.42 * dl * blamed * a)
                    g.add_color_stop_rgba(1, *rgb(MAGENTA), 0.0)
                    ctx.set_source(g)
                    ctx.arc(x, y, r * 3.2, 0, TAU)
                    ctx.fill()
                # the disc: dark, filled gold by activation (or by the guess at the output)
                ctx.set_source_rgba(*rgb(CARD_FILL), a)
                ctx.arc(x, y, r, 0, TAU)
                ctx.fill()
                if out and lit > 0:
                    ctx.set_source_rgba(*mix(BLUE, VERM, p_live), 0.85 * lit * a)
                    ctx.arc(x, y, r, 0, TAU)
                    ctx.fill()
                elif fill > 0:
                    ctx.set_source_rgba(*rgb(GOLD), 0.75 * fill * a)
                    ctx.arc(x, y, r, 0, TAU)
                    ctx.fill()
                rim = INK if (hidden and (fill > 0.02 or lit < 1)) else (INK if (li == 0 or out) else INK3)
                ctx.set_source_rgba(*rgb(rim), (0.9 if rim == INK else 0.55) * a)
                ctx.set_line_width(2.0)
                ctx.arc(x, y, r, 0, TAU)
                ctx.stroke()
                if dl * blamed > 0:
                    ctx.set_source_rgba(*rgb(MAGENTA), 0.95 * min(1.0, 0.35 + dl) * blamed * a)
                    ctx.set_line_width(2.6)
                    ctx.arc(x, y, r + 4.5, 0, TAU)
                    ctx.stroke()
                if li == 0:
                    font(ctx, "dbold", 20)
                    sx = "x₁" if i == 0 else "x₂"
                    ctx.set_source_rgba(*rgb(INK), a)
                    ctx.move_to(x - ctx.text_extents(sx).x_advance / 2, y + 7)
                    ctx.show_text(sx)
                    # the example's value, left of the node
                    font(ctx, "dbold", 21)
                    sv = f"{float(s['x'][i]):+.2f}"
                    ctx.set_source_rgba(*rgb(INK2), 0.95 * a)
                    ctx.move_to(x - r - 14 - ctx.text_extents(sv).x_advance, y + 7)
                    ctx.show_text(sv)
                elif out:
                    font(ctx, "dbold", 18)
                    ctx.set_source_rgba(*rgb(INK), a)
                    ctx.move_to(x - ctx.text_extents("out").x_advance / 2, y + 6)
                    ctx.show_text("out")
                    if lit > 0:
                        font(ctx, "dbold", 24)
                        sv = f"{p_live:.2f}"
                        ctx.set_source_rgba(*rgb(INK), 0.95 * lit * a)
                        ctx.move_to(x + r + 12, y + 8)
                        ctx.show_text(sv)
                else:
                    ctx.set_source_rgba(*rgb(INK if fill > 0.02 or lit < 1 else INK3), (0.9 if fill > 0.02 or lit < 1 else 0.6) * a)
                    ctx.set_line_width(2.0)
                    ctx.move_to(x - r * 0.6, y + r * 0.25)
                    ctx.line_to(x, y + r * 0.25)
                    ctx.line_to(x + r * 0.6, y - r * 0.4)
                    ctx.stroke()
        # the error pulse: rings leaving the output node when the error appears, and when the blame leaves it
        ox, oy = pos[nl][0]
        for t_ring in (s["error"], s["bwd"][0][0]):
            u = seg(t, t_ring, t_ring + 0.9)
            if 0 < u < 1:
                ctx.set_source_rgba(*rgb(MAGENTA), 0.8 * (1 - u) ** 1.5 * a)
                ctx.set_line_width(3.0 * (1 - u) + 0.5)
                ctx.arc(ox, oy, 22 + 70 * ease_out_cubic(u), 0, TAU)
                ctx.stroke()
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        # labels: the stage, and the layer names
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
        show_tracked(ctx, s["label"], x0 + 4, y0 + 14, tr)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        names = ["INPUTS"] + ([f"LAYER {i + 1}" for i in range(len(s["sizes"]))] if len(s["sizes"]) > 1 else ["THE NEURON"]) + ["OUTPUT"]
        for li, name in enumerate(names):
            w_ = tracked_width(ctx, name, tr)
            ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
            show_tracked(ctx, name, pos[li][0][0] - w_ / 2, y0 + h + 10, tr)
        # a legend for the two kinds of pulse
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        items = (("SIGNAL", GOLD), ("BLAME", MAGENTA))
        xx = x0 + w - 4
        for name, col in reversed(items):
            w_ = tracked_width(ctx, name, tr)
            xx -= w_
            ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
            show_tracked(ctx, name, xx, y0 + 14, tr)
            xx -= 16
            ctx.set_source_rgba(*rgb(col), 0.95 * a)
            ctx.arc(xx, y0 + 8, 5.0, 0, TAU)
            ctx.fill()
            xx -= 30

    # ------------------------------------------------------------ readouts

    def readouts(self, ctx, s, t, a, acc, p):
        rows = [("GUESS", f"{p:.2f}", p, mix(BLUE, VERM, p), ease_out_cubic(seg(t, s["guess"], s["guess"] + 0.4)), "probability of vermilion"),
                ("ANSWER", "1.00", None, rgb(VERM), ease_out_cubic(seg(t, s["answer"], s["answer"] + 0.4)), "this point is vermilion"),
                ("ERROR", f"{p - s['t']:+.2f}", None, rgb(MAGENTA), ease_out_cubic(seg(t, s["error"], s["error"] + 0.4)), "guess minus answer"),
                ("RIGHT", f"{int(round(acc * len(self.y)))}", acc, rgb(GREEN), 1.0, f"of {len(self.y)} points")]
        for r_, (lab, val, frac, col, ra, note) in enumerate(rows):
            if ra <= 0:
                continue
            aa = ra * a
            y = READ_Y0 + r_ * READ_GAP + 10 * (1 - ra)
            xl = READ_CX - 185
            font(ctx, "bold", 21)
            tr = 0.12 * 21
            ctx.set_source_rgba(*rgb(INK2), 0.95 * aa)
            show_tracked(ctx, lab, xl, y, tr)
            font(ctx, "regular", 19)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * aa)
            w_ = ctx.text_extents(note).x_advance
            ctx.move_to(READ_CX + 185 - w_, y)
            ctx.show_text(note)
            font(ctx, "dbold", 50)
            ctx.set_source_rgba(*col, 0.22 * aa)
            for dx, dy in ((-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5)):
                ctx.move_to(xl + dx, y + 56 + dy)
                ctx.show_text(val)
            ctx.set_source_rgba(*rgb(INK), aa)
            ctx.move_to(xl, y + 56)
            ctx.show_text(val)
            if frac is not None:
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                bx0, bx1, by = xl + 170, READ_CX + 185, y + 46
                ctx.set_line_width(6)
                ctx.set_source_rgba(*rgb("#2A3148"), aa)
                ctx.move_to(bx0, by)
                ctx.line_to(bx1, by)
                ctx.stroke()
                L = (bx1 - bx0) * min(1.0, frac)
                if L > 0.5:
                    ctx.set_source_rgba(*col, 0.25 * aa)
                    ctx.set_line_width(12)
                    ctx.move_to(bx0, by)
                    ctx.line_to(bx0 + L, by)
                    ctx.stroke()
                    ctx.set_source_rgba(*col, aa)
                    ctx.set_line_width(6)
                    ctx.move_to(bx0, by)
                    ctx.line_to(bx0 + L, by)
                    ctx.stroke()
                if lab == "GUESS":
                    # the halfway mark: above it the guess is vermilion, below it blue
                    ctx.set_source_rgba(*rgb(INK2), 0.8 * aa)
                    ctx.set_line_width(2)
                    ctx.move_to(bx0 + (bx1 - bx0) / 2, by - 10)
                    ctx.line_to(bx0 + (bx1 - bx0) / 2, by + 10)
                    ctx.stroke()
                ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        font(ctx, "bold", 21)
        tr = 0.12 * 21
        lab = f"400 POINTS   ·   ReLU   ·   ONE EXAMPLE, ONE PLAIN STEP OF SIZE {s['step']:g}"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1708, tr)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "EDGE: WEIGHT SIGN AND SIZE   ·   GLOW: ITS BLAME   ·   RING: THE NEURON'S BLAME"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1748, tr)

    # ------------------------------------------------------------ background and frame

    def _build_grid(self):
        surf, ctx = layer()
        ctx.set_source_rgba(*rgb(GRID_C), GRID_A)
        ctx.set_line_width(1.0)
        step = 54
        for x in range(step, W, step):
            ctx.move_to(x + 0.5, 0)
            ctx.line_to(x + 0.5, H)
        for y in range(step, H, step):
            ctx.move_to(0, y + 0.5)
            ctx.line_to(W, y + 0.5)
        ctx.stroke()
        grgb, ga = surf_rgba(surf)
        c0 = np.array(rgb(BG_TOP), np.float32) ** 2.2
        c1 = np.array(rgb(BG_BOT), np.float32) ** 2.2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = np.clip(np.sqrt((xx - W / 2) ** 2 + (yy - 1000) ** 2) / 1100, 0, 1)[..., None]
        base = (c0 * (1 - u) + c1 * u) ** (1 / 2.2)
        base = base * (1 - ga)[..., None] + grgb
        px, py, n = PANEL
        for (cx, cy, rad, col, amp) in ((px + n / 2, py + n / 2, 520.0, VERM, 0.08), (DIAG[0] + DIAG[2] / 2, DIAG[1] + DIAG[3] / 2, 560.0, BLUE, 0.08)):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
            g = np.clip(1 - d, 0, 1) ** 2 * amp
            base = base * (1 - g)[..., None] + np.array(rgb(col), np.float32) * g[..., None]
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        base = base * vig[..., None]
        self.bg = np.ascontiguousarray(base)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        px, py, n = PANEL
        pad = self.panel_pad
        for s in self.stages:
            a = self.stage_alpha(s, t) * fade
            if a <= 0:
                continue
            ps, field, acc, p, k = self.live(s, t)
            frgb, _ = self.field_rgb(field)
            shd = img[py - pad:py + n + pad, px - pad:px + n + pad]
            shd *= (1 - self.panel_shadow * a)[..., None]
            sub = img[py:py + n, px:px + n]
            am = (self.panel_mask * a)[..., None]
            sub[:] = sub * (1 - am) + (frgb * self.panel_vign[..., None]) * am
            surf, ctx = layer()
            self.draw_panel(ctx, s, t, a, ps, field)
            self.draw_diagram(ctx, s, t, a, ps, k, p)
            self.readouts(ctx, s, t, a, acc, p)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
        surf2, ctx2 = layer()
        self._headlines(ctx2, t)
        trgb, ta = surf_rgba(surf2)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s")
    print("facts", sc.facts)
    for s in sc.stages:
        _, fl, acc, p, k = sc.live(s, s["nudge"][1] + 1)
        assert k == 1 and abs(acc - s["acc1"]) < 1e-9 and abs(p - s["p1"]) < 1e-6 and np.abs(fl - s["f1"]).max() < 1e-5
        _, fl, acc, p, k = sc.live(s, s["nudge"][0] - 1)
        assert k == 0 and abs(acc - s["acc0"]) < 1e-9 and abs(p - s["p0"]) < 1e-6 and np.abs(fl - s["f0"]).max() < 1e-5
    print("live weights reproduce the saved before/after fields, guesses and accuracies")
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 200, 270, 330, 400, 560, 640, 720, 930, 1050, 1200, 1300, 1500, 1600, 1760, 1830, 2100, 2500]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (640, 930, 1760, 1830):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
