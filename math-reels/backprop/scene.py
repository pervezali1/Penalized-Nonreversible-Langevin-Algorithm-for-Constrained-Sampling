"""Backpropagation: one training step in slow motion, with every number on screen. Dark theme of reels 1-22 and
31-32. Two networks from reel 31 (bpsim.py): the one-neuron net at step 200 and the 8 + 8 net at step 150.
For each, one example the network gets wrong goes forward (gold pulses; the weights on the edges as number
pills; a ledger under the diagram doing the arithmetic as the signal arrives: multiply, add, fold, squash),
the error is shown (guess minus answer), the blame goes backward (magenta pulses; the blame on every weight
as a number, each one the blame arriving times the number that weight multiplied; neurons that were off
get 0), and a plain gradient step is taken: every pill ticks from its old value to its new one, new = old
minus step size times blame, and the field, the guess and the count of points right are recomputed live
from the tweened weights. 52 s. frame(f) -> uint8 1080x1920 RGB."""

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
N_FRAMES = 3120                       # 52 s
LOOP_FADE = (51.3, 51.85)
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
DIAG = (60, 500, 960, 540)            # the diagram: x, y, w, h
READ_CX, READ_Y0, READ_GAP = 835, 1158, 124
STRIP_Y = 452                         # the four moves, FORWARD / ERROR / BACKWARD / NUDGE
NPZ = os.path.join(HERE, "backprop_sim.npz")
T_CLOSE = 46.0

# the timeline of each stage (absolute seconds): fwd[li] is the window in which the signal crosses weight layer li,
# bwd[k] the window in which the blame crosses weight layer (last - k); guess/answer/error are when the readouts
# appear; nudge is the window over which the weights move to their new values
STAGES = [
    dict(key="one", sizes=[1], t0=0.6, t1=24.4, label="ONE NEURON, STEP 200 OF 1,500",
         fwd=[(3.0, 4.2), (4.6, 5.6)], guess=5.7, answer=6.6, error=7.2,
         bwd=[(10.3, 11.3), (11.5, 12.6)], nudge=(17.4, 19.6)),
    dict(key="deep", sizes=[8, 8], t0=24.4, t1=52.5, label="8 + 8 NEURONS, STEP 150 OF 1,500",
         fwd=[(28.3, 29.2), (29.2, 30.1), (30.1, 31.0)], guess=31.0, answer=31.6, error=32.2,
         bwd=[(33.2, 34.1), (34.1, 35.1), (35.1, 36.1)], nudge=(40.3, 42.5)),
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


def sg(v, nd=2):
    """Signed number, no negative zero."""
    v = float(v)
    if abs(v) < 0.5 * 10 ** -nd:
        v = 0.0
    return f"{v:+.{nd}f}"


def ug(v, nd=2):
    """Unsigned when positive, no negative zero."""
    v = float(v)
    if abs(v) < 0.5 * 10 ** -nd:
        v = 0.0
    return f"{v:.{nd}f}"


def rich(ctx, x, y, segs, a):
    """Draw a run of (text, kind, size, colour) segments on one baseline; returns the end x."""
    for text, kind, size, col in segs:
        font(ctx, kind, size)
        ctx.set_source_rgba(*rgb(col), a)
        ctx.move_to(x, y)
        ctx.show_text(text)
        x += ctx.text_extents(text).x_advance
    return x


def rich_width(ctx, segs):
    w = 0.0
    for text, kind, size, col in segs:
        font(ctx, kind, size)
        w += ctx.text_extents(text).x_advance
    return w


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
                     delta=[d[f"{k}_delta{li}"] for li in range(nl)], act=[d[f"{k}_act{li}"] for li in range(nl + 1)],
                     pre=[d[f"{k}_pre{li}"] for li in range(nl)])
            s["W_after"] = [w - s["step"] * g for w, g in zip(s["W"], s["gW"])]
            s["b_after"] = [b - s["step"] * g for b, g in zip(s["b"], s["gb"])]
            for li in range(nl):
                assert np.allclose(s["W_after"][li], d[f"{k}_W{li}_after"])
            s["gmax"] = max(float(np.abs(g).max()) for g in s["gW"])
            s["dW"] = [a - b for a, b in zip(s["W_after"], s["W"])]
            s["dmax"] = max(float(np.abs(g).max()) for g in s["dW"] + [a - b for a, b in zip(s["b_after"], s["b"])])
            s["layers"] = [2] + st["sizes"] + [1]
            s["pos"] = self.layout(s["layers"])
            s["n_off"] = [int(np.sum(a <= 0)) for a in s["act"][1:nl]]
            s["n0"], s["n1"] = int(round(s["acc0"] * len(self.y))), int(round(s["acc1"] * len(self.y)))
            s["logit"] = float(s["act"][nl][0])
            self.stages.append(s)
        self.facts = {s["key"]: dict(i=s["i"], x=[round(float(v), 3) for v in s["x"]], p0=round(s["p0"], 4), p1=round(s["p1"], 4), acc0=round(s["acc0"], 4),
                                     acc1=round(s["acc1"], 4), step=s["step"], stop=s["stop"], n_par=s["n_par"], n_w=s["n_w"], n_off=s["n_off"],
                                     error=round(s["p0"] - s["t"], 4), n0=s["n0"], n1=s["n1"], logit=round(s["logit"], 4)) for s in self.stages}
        self.cues = self._cues()
        self._build_grid()
        self._palette()
        self._panel_masks()

    # ------------------------------------------------------------ cues (every number from the simulation)

    def _cues(self):
        f = self.facts
        hook = [("Backpropagation: how a network learns from one mistake.", "black", 40, INK, 300, 1), ("Forward: a guess. Backward: the blame. Every number shown.", "black", 36, INK2, 390, 1)]
        o, dp = f["one"], f["deep"]
        so, sd = self.stages[0], self.stages[1]
        cues = [
            (0.0, 2.4, hook, 0, 8, "none"),
            (2.6, 6.2, [("One neuron, and one example it gets wrong.", "black", 42, INK, 300, 1), ("Forward: multiply by the weights, add, fold. Then squash to a guess.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (6.5, 9.8, [(f"The guess: {ug(o['p0'])}. The answer: 1.", "black", 44, INK, 300, 1), (f"Error {sg(o['error'])}. That number is the only teacher.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (10.0, 16.8, [("Backward: the blame flows back along every edge.", "black", 40, INK, 300, 1), ("Blame on a weight  =  the blame arriving  ×  the number it multiplied.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (17.0, 23.8, [("Then every weight moves a little, against its blame.", "black", 40, INK, 300, 1), (f"New = old − {so['step']:g} × blame. Five numbers nudged: the guess goes {ug(o['p0'])} → {ug(o['p1'])}.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (24.4, 27.8, [(f"Now the 8 + 8 network, mid-training: {dp['n0']} of 400 right.", "black", 40, INK, 300, 1), (f"One point it gets wrong. The same four moves, {dp['n_par']} numbers.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (28.0, 32.6, [("Forward, layer by layer: multiply, add, fold.", "black", 42, INK, 300, 1), ("Lit neurons are on. Dark ones passed 0.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (32.9, 39.6, [("Backward: the blame fans out through every path.", "black", 40, INK, 300, 1), ("A neuron that passed 0 gets blame 0. Nothing went through it.", "dbold", 35, INK2, 390, 0.95)], 10, 8, "pop"),
            (39.9, 45.6, [(f"{dp['n_par']} numbers, each nudged once: new = old − {sd['step']:g} × blame.", "black", 38, INK, 300, 1), (f"Guess {ug(dp['p0'])} → {ug(dp['p1'])}.   Right: {dp['n0']} → {dp['n1']} of 400.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, 51.0, [("Forward: multiply, add, fold. Backward: blame. Nudge every weight.", "black", 36, INK, 300, 1), ("Repeat a few thousand times. That is all training is.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (51.4, 52.1, hook, 12, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ geometry and the live network

    def layout(self, layers):
        x0, y0, w, h = DIAG
        pos = []
        for li, n in enumerate(layers):
            cx = x0 + 70 + (w - 140) * li / (len(layers) - 1)
            if n > 3:
                gap, cy = 52, y0 + 250
            else:
                gap, cy = 84, y0 + 140
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

    def phase(self, s, t):
        """Which of the four moves is under way: 0 none, 1 forward, 2 error, 3 backward, 4 nudge."""
        if t >= s["nudge"][0] - 0.3:
            return 4
        if t >= s["bwd"][0][0] - 0.3:
            return 3
        if t >= s["error"]:
            return 2
        if t >= s["fwd"][0][0] - 0.3:
            return 1
        return 0

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

    def number_pill(self, ctx, cx, cy, label, value, col, a, glow=0.0, w=122, h=36):
        """A rounded pill with a small label and a number, centred at (cx, cy)."""
        if a <= 0:
            return
        x, y = cx - w / 2, cy - h / 2
        if glow > 0:
            ctx.set_source_rgba(*rgb(col), 0.35 * glow * a)
            pill(ctx, x - 6, y - 6, w + 12, h + 12, 16)
            ctx.fill()
        ctx.set_source_rgba(*rgb(CARD_FILL), 0.94 * a)
        pill(ctx, x, y, w, h, 10)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(col), 0.75 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        font(ctx, "bold", 16)
        tr = 0.08 * 16
        lw = tracked_width(ctx, label, tr)
        font(ctx, "dbold", 22)
        vw = ctx.text_extents(value).x_advance
        total = lw + 10 + vw
        xs = cx - total / 2
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3 if col == INK else col), 0.95 * a)
        show_tracked(ctx, label, xs, cy + 5, tr)
        font(ctx, "dbold", 22)
        ctx.set_source_rgba(*rgb(INK if col == INK else col), a)
        ctx.move_to(xs + lw + 10, cy + 7)
        ctx.show_text(value)

    def draw_diagram(self, ctx, s, t, a, ps, k_nudge, p_live):
        x0, y0, w, h = DIAG
        layers, pos = s["layers"], s["pos"]
        nl = len(layers) - 1
        n0, n1 = s["nudge"]
        # the activations under the live weights (identical to the recorded ones until the nudge starts)
        _, acts_live, _ = forward(ps, s["x"][None, :])
        acts_live = [q[0] for q in acts_live]
        tiny = nl == 2                                        # the one-neuron network: numbers on the edges
        blame_hold = 1 - smooth(seg(t, n0, n0 + 1.2))         # the blame fades once it has been used
        fwd_str = []
        for li in range(nl):
            act = s["act"][li]
            v = np.abs(act) if li == 0 else np.maximum(act, 0)
            fwd_str.append(v / (v.max() + 1e-9))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        # --- edges: colour by the sign of the live weight, thickness by its size (per layer)
        for li in range(nl):
            Wl = ps[2 * li]
            wmax = float(np.abs(Wl).max()) + 1e-9
            dense = layers[li] * layers[li + 1] > 20
            for i, p in enumerate(pos[li]):
                for j, q in enumerate(pos[li + 1]):
                    wv = float(Wl[i, j])
                    st = abs(wv) / wmax
                    col = VERM if wv >= 0 else BLUE
                    aa = ((0.12 + 0.42 * st) if dense else (0.30 + 0.6 * st)) * a
                    ctx.set_source_rgba(*rgb(col), 0.2 * aa)
                    ctx.set_line_width((1.4 + 3.4 * st) if dense else (2.5 + 6.0 * st))
                    ctx.move_to(*p)
                    ctx.line_to(*q)
                    ctx.stroke()
                    ctx.set_source_rgba(*rgb(col), aa)
                    ctx.set_line_width((0.7 + 1.6 * st) if dense else (1.0 + 2.6 * st))
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
                        r_ = 2.6 if m == 0 else 1.7
                        if m == 0:
                            ctx.set_source_rgba(*rgb(col), 0.28 * al)
                            ctx.arc(x, y, 6.5, 0, TAU)
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
        lit_of, blamed_of = [], []
        for li, layer_pos in enumerate(pos):
            r = 20 if layers[li] <= 3 else 15
            hidden = 0 < li < nl
            out = li == nl
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
            lit_of.append(lit)
            blamed_of.append(blamed)
            acts = acts_live[li]
            amax = float(np.max(np.maximum(acts, 0))) + 1e-9 if hidden else 1.0
            for i, (x, y) in enumerate(layer_pos):
                fill = 0.0
                if hidden:
                    fill = (0.3 + 0.7 * float(max(acts[i], 0) / amax) ** 0.6) * lit if acts[i] > 0 else 0.0
                elif li == 0:
                    fill = float(abs(s["x"][i]) / (np.abs(s["x"]).max() + 1e-9)) * 0.6
                dl = float(abs(s["delta"][li - 1][i]) / dmax) if li > 0 else 0.0
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
                on = (not hidden) or fill > 0.02 or lit < 1
                ctx.set_source_rgba(*rgb(INK if on else INK3), (0.9 if on else 0.55) * a)
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
                    font(ctx, "dbold", 21)
                    sv = sg(s["x"][i])
                    ctx.set_source_rgba(*rgb(GOLD), 0.95 * a)
                    ctx.move_to(x - r - 14 - ctx.text_extents(sv).x_advance, y + 7)
                    ctx.show_text(sv)
                elif out:
                    font(ctx, "dbold", 18)
                    ctx.set_source_rgba(*rgb(INK), a)
                    ctx.move_to(x - ctx.text_extents("out").x_advance / 2, y + 6)
                    ctx.show_text("out")
                    if lit > 0:
                        font(ctx, "dbold", 24)
                        sv = ug(p_live)
                        ctx.set_source_rgba(*rgb(INK), 0.95 * lit * a)
                        ctx.move_to(x + r + 12, y + 8)
                        ctx.show_text(sv)
                else:
                    ctx.set_source_rgba(*rgb(INK if on else INK3), (0.9 if on else 0.6) * a)
                    ctx.set_line_width(2.0)
                    ctx.move_to(x - r * 0.6, y + r * 0.25)
                    ctx.line_to(x, y + r * 0.25)
                    ctx.line_to(x + r * 0.6, y - r * 0.4)
                    ctx.stroke()
                    # the number this neuron passed on (gold), then the blame that reached it (magenta)
                    if not tiny and (lit > 0 or blamed > 0):
                        if blamed > 0:
                            dv = float(s["delta"][li - 1][i])
                            txt, col, al = (sg(dv) if dv != 0 else "0"), (MAGENTA if dv != 0 else INK3), blamed
                        else:
                            av = float(acts[i])
                            txt, col, al = (ug(av) if av > 0 else "0"), (GOLD if av > 0 else INK3), lit
                        font(ctx, "dbold", 18)
                        tw = ctx.text_extents(txt).x_advance
                        ctx.set_source_rgba(*rgb(BG_BOT), 0.72 * al * a)
                        pill(ctx, x + r + 6, y - 12, tw + 12, 24, 7)
                        ctx.fill()
                        ctx.set_source_rgba(*rgb(col), 0.95 * al * a)
                        ctx.move_to(x + r + 12, y + 6)
                        ctx.show_text(txt)
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
        # --- the one-neuron network: every weight as a pill on its edge, its blame under it, both live
        if tiny:
            (xi1, yi1), (xi2, yi2) = pos[0]
            xn, yn = pos[1][0]
            xo, yo = pos[2][0]
            W1, B1, W2, B2 = ps[0], ps[1], ps[2], ps[3]
            a_in = ease_out_cubic(seg(t, s["t0"] + 0.3, s["t0"] + 0.8)) * a
            items = [  # (centre, label, live value, blame value, blame appears at, offset of the blame pill)
                (((xi1 + xn) / 2, (yi1 + yn) / 2 - 28), "w₁", float(W1[0, 0]), float(s["gW"][0][0, 0]), s["bwd"][1][1], -40),
                (((xi2 + xn) / 2, (yi2 + yn) / 2 + 28), "w₂", float(W1[1, 0]), float(s["gW"][0][1, 0]), s["bwd"][1][1], 40),
                ((xn, yn + 54), "b", float(B1[0]), float(s["gb"][0][0]), s["bwd"][1][1], 40),
                (((xn + xo) / 2, yn - 28), "a", float(W2[0, 0]), float(s["gW"][1][0, 0]), s["bwd"][0][1], -40),
                ((xo, yo + 54), "c", float(B2[0]), float(s["gb"][1][0]), s["bwd"][0][1], 40),
            ]
            glow = sh / max(a, 1e-9) if a > 0 else 0.0
            for (cx, cy), lab, val, bl, t_bl, off in items:
                self.number_pill(ctx, cx, cy, lab, sg(val), INK, a_in, glow=glow)
                ab = ease_out_cubic(seg(t, t_bl - 0.1, t_bl + 0.3)) * blame_hold * a
                self.number_pill(ctx, cx, cy + off, "BLAME", sg(bl), MAGENTA, ab, w=124)
            # the neuron's own numbers: what it passed on, then the blame that reached it
            lit_n, bl_n = lit_of[1], blamed_of[1]
            if bl_n > 0:
                self.number_pill(ctx, xn, yn - 54, "BLAME HERE", sg(float(s["delta"][0][0])), MAGENTA, bl_n * a, w=150)
            elif lit_n > 0:
                self.number_pill(ctx, xn, yn - 54, "PASSED ON", ug(float(acts_live[1][0])), GOLD, lit_n * a, w=150)
        # --- labels: the stage, the layer names, the legend
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
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        xx = x0 + w - 4
        for name, col in reversed((("SIGNAL", GOLD), ("BLAME", MAGENTA))):
            w_ = tracked_width(ctx, name, tr)
            xx -= w_
            ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
            show_tracked(ctx, name, xx, y0 + 14, tr)
            xx -= 16
            ctx.set_source_rgba(*rgb(col), 0.95 * a)
            ctx.arc(xx, y0 + 8, 5.0, 0, TAU)
            ctx.fill()
            xx -= 30

    # ------------------------------------------------------------ the ledger: the arithmetic as it happens

    def ledger_rows(self, s, t):
        """Rows of (appear time, column, label, segments) for the current page of the one-neuron ledger."""
        R, D = "regular", "dbold"
        S = 26
        x1, x2 = s["x"]
        W1, B1, W2, B2 = s["W"][0], s["b"][0], s["W"][1], s["b"][1]
        w1, w2, b, a_, c = float(W1[0, 0]), float(W1[1, 0]), float(B1[0]), float(W2[0, 0]), float(B2[0])
        n = float(s["act"][1][0])
        logit, p0, p1 = s["logit"], s["p0"], s["p1"]
        err = p0 - s["t"]
        g = s["gW"][0][:, 0]
        gb, ga, gc = float(s["gb"][0][0]), float(s["gW"][1][0, 0]), float(s["gb"][1][0])
        dn = float(s["delta"][0][0])
        stp = s["step"]
        ph = self.phase(s, t)

        def num(v, col=INK, signed=True):
            return (sg(v) if signed else ug(v), D, S, col)

        def op(txt):
            return (txt, R, S, INK3)

        if ph <= 2:
            fa, fb = s["fwd"]
            return [
                (fa[1], 0, 0, "SUM: EACH INPUT × ITS WEIGHT, ADDED, PLUS THE BIAS",
                 [num(w1), op(" × "), num(x1, GOLD), op("   +   "), num(w2), op(" × "), num(x2, GOLD), op("   +   "), num(b), op("   =   "), num(n, GOLD, False)]),
                (fa[1] + 0.3, 1, 0, "FOLD: BELOW ZERO WOULD BECOME 0", [num(n, GOLD, False), op("  is above zero, so it passes:  "), num(n, GOLD, False)]),
                (fb[1], 2, 0, "OUT: × THE OUTPUT WEIGHT, PLUS ITS BIAS, SQUASHED TO A PROBABILITY",
                 [num(a_), op(" × "), num(n, GOLD, False), op("   "), num(c), op("   =   "), num(logit), op("   →   guess  "), num(p0, GOLD, False)]),
                (s["error"], 3, 0, "ERROR: GUESS − ANSWER", [num(p0, GOLD, False), op("  −  "), ("1", D, S, VERM), op("   =   "), num(err, MAGENTA)]),
            ]
        if ph == 3:
            (ba, bb), (ba2, bb2) = s["bwd"]
            return [
                (bb, 0, 0, "BLAME ON c  (IT MULTIPLIED 1)", [num(err, MAGENTA), op(" × "), ("1", D, S, INK), op("   =   "), num(gc, MAGENTA)]),
                (bb, 0, 1, f"BLAME ON a  (IT MULTIPLIED {ug(n)})", [num(err, MAGENTA), op(" × "), num(n, GOLD, False), op("   =   "), num(ga, MAGENTA)]),
                (bb + 0.2, 1, 0, "BACK THROUGH THE NEURON  (IT WAS ON, SO THE BLAME PASSES)", [num(err, MAGENTA), op(" × "), num(a_), op("   =   "), num(dn, MAGENTA)]),
                (bb2, 2, 0, "BLAME ON w₁  (IT MULTIPLIED x₁)", [num(dn, MAGENTA), op(" × "), num(x1, GOLD), op("   =   "), num(float(g[0]), MAGENTA)]),
                (bb2, 2, 1, "BLAME ON w₂  (IT MULTIPLIED x₂)", [num(dn, MAGENTA), op(" × "), num(x2, GOLD), op("   =   "), num(float(g[1]), MAGENTA)]),
                (bb2 + 0.2, 3, 0, "BLAME ON b  (IT MULTIPLIED 1)", [num(dn, MAGENTA), op(" × "), ("1", D, S, INK), op("   =   "), num(gb, MAGENTA)]),
            ]
        n0 = s["nudge"][0]
        W1a, B1a, W2a, B2a = s["W_after"][0], s["b_after"][0], s["W_after"][1], s["b_after"][1]

        def upd(lab, old, bl, new):
            return [num(old), op(f"  −  {stp:g} × ("), num(bl, MAGENTA), op(")   =   "), num(new)]

        return [
            (n0 - 0.2, 0, 0, "NEW w₁  =  OLD  −  STEP SIZE × BLAME", upd("w₁", w1, float(g[0]), float(W1a[0, 0]))),
            (n0 - 0.2, 0, 1, "NEW w₂", upd("w₂", w2, float(g[1]), float(W1a[1, 0]))),
            (n0 + 0.1, 1, 0, "NEW b", upd("b", b, gb, float(B1a[0]))),
            (n0 + 0.1, 1, 1, "NEW a", upd("a", a_, ga, float(W2a[0, 0]))),
            (n0 + 0.4, 2, 0, "NEW c", upd("c", c, gc, float(B2a[0]))),
            (s["nudge"][1] + 0.3, 3, 0, "THE GUESS, AND THE POINTS RIGHT OF 400", [num(p0, GOLD, False), op("  →  "), num(p1, GOLD, False), op("        "), (f"{s['n0']}", D, S, GREEN), op("  →  "), (f"{s['n1']}", D, S, GREEN)]),
        ]

    def draw_ledger(self, ctx, s, t, a):
        x0, y0, w, h = DIAG
        if len(s["sizes"]) == 1:
            ph = self.phase(s, t)
            # the page fades out just before the next move starts
            t_next = {1: s["bwd"][0][0] - 0.3, 2: s["bwd"][0][0] - 0.3, 3: s["nudge"][0] - 0.3}.get(ph, None)
            page_a = (1 - smooth(seg(t, t_next - 0.4, t_next))) if t_next is not None else 1.0
            rows = self.ledger_rows(s, t)
            ys = [y0 + 272, y0 + 342, y0 + 412, y0 + 482]
            shared = {r[1] for r in rows if r[2] == 1}          # rows with a second column
            for (t_app, row, col, lab, segs) in rows:
                ra = ease_out_cubic(seg(t, t_app, t_app + 0.4)) * page_a * a
                if ra <= 0:
                    continue
                y = ys[row] + 8 * (1 - ease_out_cubic(seg(t, t_app, t_app + 0.4)))
                x = x0 + 10 if col == 0 else x0 + 500
                font(ctx, "bold", 17)
                tr = 0.1 * 17
                lw_ = tracked_width(ctx, lab, tr)
                ctx.set_source_rgba(*rgb(INK3), 0.95 * ra)
                show_tracked(ctx, lab, x, y, tr)
                wsum = rich_width(ctx, segs)
                assert max(wsum, lw_) < (460 if row in shared else 940), (lab, wsum, lw_)
                rich(ctx, x, y + 34, segs, ra)
        else:
            # the 8 + 8 network: one line, the output's numbers, then the count of what flowed back
            ph = self.phase(s, t)
            R, D, S = "regular", "dbold", 24
            if ph == 1 and t >= s["guess"]:
                lab, segs, t_app = "OUT: THE LAYER-2 NUMBERS × THEIR WEIGHTS, ADDED, PLUS THE BIAS, SQUASHED", [(sg(s["logit"]), D, S, INK), ("   →   guess  ", R, S, INK3), (ug(s["p0"]), D, S, GOLD)], s["guess"]
            elif ph == 2:
                lab, segs, t_app = "ERROR: GUESS − ANSWER", [(ug(s["p0"]), D, S, GOLD), ("  −  ", R, S, INK3), ("1", D, S, VERM), ("   =   ", R, S, INK3), (sg(s["p0"] - s["t"]), D, S, MAGENTA)], s["error"]
            elif ph == 3:
                o1, o2 = s["n_off"]
                lab, segs, t_app = f"BLAME FLOWING BACK: ONE NUMBER ON EACH OF THE {s['n_par']} WEIGHTS AND BIASES", [("8 numbers into layer 2", D, S, MAGENTA), (f"  ({o2} are 0)", R, S, INK3), ("      ·      8 into layer 1", D, S, MAGENTA), (f"  ({o1} are 0)", R, S, INK3)], s["bwd"][0][0]
            elif ph == 4:
                lab, segs, t_app = "NUDGE", [(f"new  =  old  −  {s['step']:g} × blame,   for all {s['n_par']} numbers at once", D, S, INK)], s["nudge"][0] - 0.2
            else:
                return
            ra = ease_out_cubic(seg(t, t_app, t_app + 0.4)) * a
            if ra <= 0:
                return
            y = y0 + 466
            font(ctx, "bold", 17)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * ra)
            assert tracked_width(ctx, lab, 1.7) < 940, lab
            show_tracked(ctx, lab, x0 + 10, y, 1.7)
            assert rich_width(ctx, segs) < 940, lab
            rich(ctx, x0 + 10, y + 33, segs, ra)

    # ------------------------------------------------------------ the strip of four moves, and the readouts

    def draw_strip(self, ctx, s, t, a):
        ph = self.phase(s, t)
        items = (("FORWARD", GOLD), ("ERROR", MAGENTA), ("BACKWARD", MAGENTA), ("NUDGE", INK))
        font(ctx, "bold", 19)
        tr = 0.14 * 19
        widths = [tracked_width(ctx, n, tr) for n, _ in items]
        gap = 54
        total = sum(widths) + gap * (len(items) - 1) + 22 * len(items)
        x = 540 - total / 2
        for k, ((name, col), w_) in enumerate(zip(items, widths)):
            on = (k + 1) == ph
            done = (k + 1) < ph
            cc = col if on else (INK2 if done else INK3)
            al = (1.0 if on else (0.7 if done else 0.45)) * a
            ctx.set_source_rgba(*rgb(cc), al)
            ctx.arc(x + 6, STRIP_Y - 7, 5.0 if on else 3.5, 0, TAU)
            ctx.fill()
            if on:
                ctx.set_source_rgba(*rgb(cc), 0.25 * a)
                ctx.arc(x + 6, STRIP_Y - 7, 11.0, 0, TAU)
                ctx.fill()
            ctx.set_source_rgba(*rgb(cc), al)
            show_tracked(ctx, name, x + 22, STRIP_Y, tr)
            if on:
                ctx.set_source_rgba(*rgb(cc), 0.8 * a)
                ctx.set_line_width(2.0)
                ctx.move_to(x + 22, STRIP_Y + 10)
                ctx.line_to(x + 22 + w_, STRIP_Y + 10)
                ctx.stroke()
            x += 22 + w_ + gap

    def readouts(self, ctx, s, t, a, acc, p):
        rows = [("GUESS", ug(p), p, mix(BLUE, VERM, p), ease_out_cubic(seg(t, s["guess"], s["guess"] + 0.4)), "probability of vermilion"),
                ("ANSWER", "1.00", None, rgb(VERM), ease_out_cubic(seg(t, s["answer"], s["answer"] + 0.4)), "this point is vermilion"),
                ("ERROR", sg(p - s["t"]), None, rgb(MAGENTA), ease_out_cubic(seg(t, s["error"], s["error"] + 0.4)), "guess minus answer"),
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
        lab = "NUMBERS ROUNDED TO 2 DECIMALS   ·   EDGE: WEIGHT SIGN, SIZE   ·   GLOW: ITS BLAME"
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
            self.draw_ledger(ctx, s, t, a)
            self.draw_strip(ctx, s, t, a)
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
    frames = [0, 200, 260, 300, 345, 400, 450, 700, 770, 1060, 1230, 1500, 1760, 1900, 2140, 2208, 2500, 2900]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (345, 450, 770, 1060, 1230, 2208, 2500):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
