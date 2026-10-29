"""A neural network, one neuron at a time, in the white style of reels 23-30 with the maths on screen. The
same two-arm spiral every time (neurons/train.py). Five networks trained from scratch with ReLU hidden
units and Adam (nnsim.py): 1, 2, 3 and 8 neurons in one layer, then two layers of 8. For each stage the
network diagram grows on the left, the output field (probability of the vermilion class over the input
plane) trains on the right from its random start, each first-layer neuron's fold line (w.x + b = 0) is
drawn on the field, the readouts count neurons, accuracy and training steps, and a maths card carries
the formula with the trained numbers. 48 s. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import (H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba,  # noqa: E402
                     tracked_width)

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)

FPS = 60
N_FRAMES = 2880                       # 48 s
LOOP_FADE = (47.3, 47.85)
INK, INK2, INK3 = gds.INK, gds.INK2, gds.INK3
VERM, BLUE, GREEN, MAGENTA = gds.AMBER, gds.BLUE, gds.GREEN, gds.MAGENTA
seg, lerp = gds.seg, gds.lerp
TAU = 2 * math.pi
BG = "#F4F5F9"
GRID_A = 0.05
CARD = (70, 1575, 940, 300)
PANEL = (540, 600, 480)               # the field: x, y, size
DIAG = (40, 560, 480, 560)            # the diagram: x, y, w, h
NPZ = os.path.join(HERE, "nnbuild_train.npz")

# stage: key, sizes, start time, training window (start, end) relative to the stage start, pill label
STAGES = [
    dict(key="one", sizes=[1], t0=2.8, train=(1.2, 5.6), label="1"),
    dict(key="two", sizes=[2], t0=11.3, train=(1.0, 5.0), label="2"),
    dict(key="three", sizes=[3], t0=18.6, train=(1.0, 5.0), label="3"),
    dict(key="eight", sizes=[8], t0=25.6, train=(1.0, 5.4), label="8"),
    dict(key="deep", sizes=[8, 8], t0=33.6, train=(1.2, 6.0), label="8 + 8"),
]
T_CLOSE = 42.8
COL_X = (190, 540, 890)


def sg(v, nd=2):
    """Signed number without a negative zero."""
    v = float(v)
    if abs(v) < 0.5 * 10 ** -nd:
        v = 0.0
    return f"{v:+.{nd}f}"


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


class Scene(gds.Scene):
    def __init__(self):
        d = np.load(NPZ)
        self.X, self.y, self.grid = d["X"], d["y"], d["grid"]
        self.lim = float(self.grid[-1])
        self.stages = []
        for st in STAGES:
            k = st["key"]
            s = dict(st, fields=d[f"{k}_fields"], W1=d[f"{k}_W1"], B1=d[f"{k}_B1"], acc=d[f"{k}_acc"], loss=d[f"{k}_loss"],
                     params=[d[f"{k}_p{i}"] for i in range(2 * (len(st["sizes"]) + 1))], seed=int(d[f"{k}_seed"]),
                     finals=d[f"{k}_finals"])
            s["n_snap"] = len(s["acc"])
            s["t1"] = STAGES[STAGES.index(st) + 1]["t0"] if STAGES.index(st) + 1 < len(STAGES) else T_CLOSE
            self.stages.append(s)
        self.facts = {s["key"]: dict(acc=float(s["acc"][-1]), acc0=float(s["acc"][0]), loss=float(s["loss"][-1]), seed=s["seed"],
                                     finals=[float(f) for f in s["finals"]]) for s in self.stages}
        self.cues = self._cues()
        self._build_grid()
        self._palette()

    # ------------------------------------------------------------ time

    def stage_at(self, t):
        """(stage, snapshot index (float), stage-local time) for time t; before the first stage: None."""
        cur = None
        for s in self.stages:
            if t >= s["t0"] - 0.4:
                cur = s
        if cur is None:
            return None, 0.0, 0.0
        tl = t - cur["t0"]
        a, b = cur["train"]
        prog = seg(tl, a, b)
        prog = 1 - (1 - prog) ** 2.2                       # fast early, slow late, like the loss itself
        return cur, prog * (cur["n_snap"] - 1), tl

    # ------------------------------------------------------------ cues (every number from the runs)

    def _cues(self):
        f = self.facts
        hook = [("A neural network, one neuron at a time.", "black", 50, INK, 300, 1), ("Same data every time. Watch what the output can do.", "black", 38, INK2, 390, 1)]
        S = {s["key"]: s for s in self.stages}

        def tr(k):
            return S[k]["t0"] + S[k]["train"][1] + 0.3

        cues = [
            (0.0, 2.4, hook, 0, 8, "none"),
            (S["one"]["t0"], tr("one") - 0.3, [("One neuron: a weighted sum, then a fold.", "black", 44, INK, 300, 1), ("It can draw one straight edge, and nothing else.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (tr("one"), S["two"]["t0"] - 0.3, [(f"Trained: {f['one']['acc']:.0%}.", "black", 46, VERM, 300, 1), ("A spiral is not a straight line.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S["two"]["t0"], tr("two") - 0.3, [("Two neurons, added.", "black", 48, INK, 300, 1), ("Two folds: the edge can bend once.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (tr("two"), S["three"]["t0"] - 0.3, [(f"Trained: {f['two']['acc']:.0%}. " + ("Barely better." if f["two"]["acc"] - f["one"]["acc"] < 0.01 else "Better."), "black", 44, VERM, 300, 1), ("Two straight edges. Still not a spiral.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S["three"]["t0"], tr("three") - 0.3, [("Three neurons.", "black", 48, INK, 300, 1), ("Three folds: corners, wedges, a band.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (tr("three"), S["eight"]["t0"] - 0.3, [(f"Trained: {f['three']['acc']:.0%}.", "black", 46, VERM, 300, 1), ("Every edge is still a straight line.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S["eight"]["t0"], tr("eight") - 0.3, [("A layer of eight.", "black", 48, INK, 300, 1), ("Eight folds, added: eight straight edges in the plane.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (tr("eight"), S["deep"]["t0"] - 0.3, [(f"Trained: {f['eight']['acc']:.0%}.", "black", 46, VERM, 300, 1), ("Better. But a spiral curves, and these edges can't.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (S["deep"]["t0"], tr("deep") - 0.3, [("Now the chain again: eight more, fed by the first eight.", "black", 36, INK, 300, 1), ("Folds of folds. Now the edges can curve.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (tr("deep"), T_CLOSE - 0.3, [(f"Trained: {f['deep']['acc']:.0%}.", "black", 46, VERM, 300, 1), ("Same data. Same rule. One more layer.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, 47.0, [("Neuron: one fold. Layer: many folds, added. Depth: folds of folds.", "black", 32, INK, 300, 1), ("That is the whole trick. Everything else is scale.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (47.4, 48.1, hook, 12, 0, "fade"),
        ]
        return cues

    # ------------------------------------------------------------ the field

    def _palette(self):
        w = np.array([1.0, 1.0, 1.0], np.float32)
        cb, cv = np.array(rgb(BLUE), np.float32), np.array(rgb(VERM), np.float32)
        ps = np.linspace(0, 1, 256)
        lut = np.empty((256, 3), np.float32)
        for i, p in enumerate(ps):
            if p < 0.5:
                k = (0.5 - p) * 2
                lut[i] = w + (cb - w) * 0.62 * k
            else:
                k = (p - 0.5) * 2
                lut[i] = w + (cv - w) * 0.62 * k
        self.lut = lut

    def field_rgb(self, field):
        px, py, n = PANEL
        up = cv2.resize(field.astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        idx = np.clip((up * 255).round().astype(np.int32), 0, 255)
        return self.lut[idx], up

    def to_panel(self, x, y):
        px, py, n = PANEL
        return px + (x + self.lim) / (2 * self.lim) * n, py + (self.lim - y) / (2 * self.lim) * n

    def draw_panel(self, ctx, st, snap, a):
        """Fold lines, decision boundary, data points and frame over the field (the field itself is pasted in frame())."""
        px, py, n = PANEL
        i0 = int(math.floor(snap))
        i1 = min(i0 + 1, st["n_snap"] - 1)
        k = snap - i0
        ctx.save()
        ctx.rectangle(px, py, n, n)
        ctx.clip()
        # fold lines of the first layer: w1 x + w2 y + b = 0, each one a neuron
        W1 = st["W1"][i0] * (1 - k) + st["W1"][i1] * k
        B1 = st["B1"][i0] * (1 - k) + st["B1"][i1] * k
        ctx.set_source_rgba(*rgb(INK), (0.22 if len(st["sizes"]) > 1 else 0.5) * a)
        ctx.set_line_width(1.6)
        ctx.set_dash([7.0, 5.0])
        L = self.lim * 3
        for j in range(W1.shape[1]):
            w1, w2, b = float(W1[0, j]), float(W1[1, j]), float(B1[j])
            nn_ = math.hypot(w1, w2)
            if nn_ < 1e-6:
                continue
            # a point on the line and its direction
            x0, y0 = -b * w1 / nn_ ** 2, -b * w2 / nn_ ** 2
            dx, dy = -w2 / nn_, w1 / nn_
            p = self.to_panel(x0 - dx * L, y0 - dy * L)
            q = self.to_panel(x0 + dx * L, y0 + dy * L)
            ctx.move_to(*p)
            ctx.line_to(*q)
            ctx.stroke()
        ctx.set_dash([])
        # decision boundary: the 0.5 contour of the upsampled field
        field = st["fields"][i0] * (1 - k) + st["fields"][i1] * k
        _, up = self.field_rgb(field)
        mask = (up > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ctx.set_source_rgba(*rgb(INK), 0.85 * a)
        ctx.set_line_width(2.2)
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
            ctx.set_source_rgba(*rgb(VERM if c > 0.5 else BLUE), 0.9 * a)
            ctx.arc(sx, sy, 3.2, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(1, 1, 1, 0.9 * a)
            ctx.set_line_width(0.8)
            ctx.arc(sx, sy, 3.2, 0, TAU)
            ctx.stroke()
        ctx.restore()
        ctx.set_source_rgba(*rgb(INK3), 0.7 * a)
        ctx.set_line_width(1.5)
        ctx.rectangle(px + 0.5, py + 0.5, n - 1, n - 1)
        ctx.stroke()
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        for s, x in (("x₁", px + n / 2 - 10), ):
            ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
            show_tracked(ctx, s, x, py + n + 30, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, "x₂", px - 34, py + n / 2 + 8, tr)
        lab = "OUTPUT OVER THE INPUT PLANE"
        w = tracked_width(ctx, lab, tr)
        show_tracked(ctx, lab, px + n / 2 - w / 2, py - 16, tr)

    # ------------------------------------------------------------ the diagram

    def draw_diagram(self, ctx, st, snap, a, tl):
        x0, y0, w, h = DIAG
        sizes = st["sizes"]
        cols = [x0 + 60] + [x0 + 60 + (w - 120) * (i + 1) / (len(sizes) + 1) for i in range(len(sizes))] + [x0 + w - 60]
        layers = [2] + sizes + [1]
        cy = y0 + h / 2

        def ys(n, big):
            gap = 72 if n <= 3 else 60
            return [cy + (i - (n - 1) / 2) * gap for i in range(n)]

        pos = [[(cols[li], yy) for yy in ys(n, n <= 3)] for li, n in enumerate(layers)]
        i0 = int(math.floor(snap))
        i1 = min(i0 + 1, st["n_snap"] - 1)
        k = snap - i0
        W1 = st["W1"][i0] * (1 - k) + st["W1"][i1] * k
        # neurons appear one by one at the stage start
        n_vis = [2] + [min(n, int(seg(tl, 0.0, 0.9) * n + 0.999)) for n in sizes] + [1]
        # edges
        for li in range(len(layers) - 1):
            Wl = W1 if li == 0 else st["params"][2 * li]
            wmax = float(np.abs(Wl).max()) + 1e-9
            for i, p in enumerate(pos[li][:n_vis[li]]):
                for j, q in enumerate(pos[li + 1][:n_vis[li + 1]]):
                    wv = float(Wl[i, j])
                    col = VERM if wv >= 0 else BLUE
                    aa = (0.25 + 0.75 * abs(wv) / wmax) * a if li == 0 else 0.35 * a
                    ctx.set_source_rgba(*rgb(col if li == 0 else INK3), aa)
                    ctx.set_line_width(1.0 + 3.0 * abs(wv) / wmax if li == 0 else 1.4)
                    ctx.move_to(*p)
                    ctx.line_to(*q)
                    ctx.stroke()
        # nodes
        for li, layer_pos in enumerate(pos):
            r = 20 if layers[li] <= 3 else 15
            for i, (x, y) in enumerate(layer_pos[:n_vis[li]]):
                ctx.set_source_rgba(1, 1, 1, a)
                ctx.arc(x, y, r, 0, TAU)
                ctx.fill()
                ctx.set_source_rgba(*rgb(INK if 0 < li < len(layers) - 1 else INK3), 0.9 * a)
                ctx.set_line_width(2.0)
                ctx.arc(x, y, r, 0, TAU)
                ctx.stroke()
                if li == 0:
                    font(ctx, "dbold", 20)
                    s = "x₁" if i == 0 else "x₂"
                    ctx.set_source_rgba(*rgb(INK), a)
                    ctx.move_to(x - ctx.text_extents(s).x_advance / 2, y + 7)
                    ctx.show_text(s)
                elif li == len(layers) - 1:
                    font(ctx, "dbold", 18)
                    s = "out"
                    ctx.set_source_rgba(*rgb(INK), a)
                    ctx.move_to(x - ctx.text_extents(s).x_advance / 2, y + 6)
                    ctx.show_text(s)
                else:
                    # the fold symbol: a tiny ReLU graph inside the node
                    ctx.set_source_rgba(*rgb(INK), 0.9 * a)
                    ctx.set_line_width(2.0)
                    ctx.move_to(x - r * 0.6, y + r * 0.25)
                    ctx.line_to(x, y + r * 0.25)
                    ctx.line_to(x + r * 0.6, y - r * 0.4)
                    ctx.stroke()
        # labels under the layers
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        names = ["INPUTS"] + ([f"LAYER {i + 1}" for i, n in enumerate(sizes)] if len(sizes) > 1 else [f"{sizes[0]} NEURON" + ("S" if sizes[0] > 1 else "")]) + (["OUT"] if len(sizes) > 1 else ["OUTPUT"])
        for li, name in enumerate(names):
            w_ = tracked_width(ctx, name, tr)
            ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
            show_tracked(ctx, name, cols[li] - w_ / 2, y0 + h + 10, tr)

    # ------------------------------------------------------------ readouts and the maths card

    def readouts(self, ctx, st, snap, a):
        i = int(round(snap))
        acc = float(st["acc"][i])
        step = i * 15
        n_neur = sum(st["sizes"])
        Y_LAB, Y_VAL, Y_BAR = 1404, 1466, 1489
        items = (("NEURONS", st["label"], None, INK), ("ACCURACY", f"{acc:.0%}", acc, VERM), ("TRAINING STEP", f"{step}", step / 1500, INK3))
        for (lab, val, frac, col), xc in zip(items, COL_X):
            font(ctx, "bold", 22)
            tr = 0.12 * 22
            w = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(INK2), 0.95 * a)
            show_tracked(ctx, lab, xc - w / 2, Y_LAB, tr)
            font(ctx, "dbold", 46)
            w = ctx.text_extents(val).x_advance
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.move_to(xc - w / 2, Y_VAL)
            ctx.show_text(val)
            if frac is not None:
                ctx.set_line_width(6)
                ctx.set_source_rgba(*rgb("#E5E7EB"), a)
                ctx.move_to(xc - 110, Y_BAR)
                ctx.line_to(xc + 110, Y_BAR)
                ctx.stroke()
                L = 220 * min(1.0, frac)
                ctx.set_source_rgba(*rgb(col), a)
                ctx.move_to(xc - 110, Y_BAR)
                ctx.line_to(xc - 110 + L, Y_BAR)
                ctx.stroke()
        font(ctx, "bold", 22)
        tr = 0.12 * 22
        lab = "TWO SPIRAL ARMS, 400 POINTS   ·   ReLU   ·   ADAM, STEP SIZE 0.01"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1372, tr)

    def card_lines(self, st, snap):
        i = int(round(snap))
        W1, B1 = st["W1"][i], st["B1"][i]
        acc = float(st["acc"][i])
        R, D = "regular", "dbold"
        k = st["key"]
        p = st["params"]
        if k == "one":
            a_, c_ = float(p[2][0, 0]), float(p[3][0])
            return "ONE NEURON", [
                ("n = max(0,  w₁·x₁ + w₂·x₂ + b)        a weighted sum, then a fold at zero", R, 27, INK),
                ("output = σ(a·n + c)                σ squashes any number into a probability", R, 27, INK),
                (f"now:  n = max(0,  {W1[0, 0]:+.2f}·x₁ {W1[1, 0]:+.2f}·x₂ {B1[0]:+.2f})      accuracy {acc:.0%}", D, 27, BLUE),
                ("Flat where the sum is negative, a ramp where it is positive: one straight fold.", R, 26, INK2)]
        if k == "two":
            return "TWO NEURONS, ADDED", [
                ("n₁ = max(0, w₁₁x₁ + w₂₁x₂ + b₁)     n₂ = max(0, w₁₂x₁ + w₂₂x₂ + b₂)", R, 25, INK),
                ("output = σ(a₁·n₁ + a₂·n₂ + c)", R, 28, INK),
                (f"now:  fold 1 at  {W1[0, 0]:+.2f}x₁ {W1[1, 0]:+.2f}x₂ {B1[0]:+.2f} = 0,   fold 2 at  {W1[0, 1]:+.2f}x₁ {W1[1, 1]:+.2f}x₂ {B1[1]:+.2f} = 0", D, 24, BLUE),
                ("Two folds added together: the field can bend once, along two straight lines.", R, 26, INK2)]
        if k == "three":
            return "THREE NEURONS", [
                ("output = σ(a₁·n₁ + a₂·n₂ + a₃·n₃ + c),   each n = max(0, w·x + b)", R, 26, INK),
                (f"now:  accuracy {acc:.0%}", D, 28, BLUE),
                ("Three straight folds can make a corner, a wedge or a band.", R, 27, INK),
                ("The dashed lines on the field are the folds: one per neuron.", R, 26, INK2)]
        if k == "eight":
            return "A LAYER OF EIGHT", [
                ("output = σ(a₁·n₁ + a₂·n₂ + … + a₈·n₈ + c),   each n = max(0, w·x + b)", R, 26, INK),
                (f"now:  accuracy {acc:.0%}", D, 28, BLUE),
                ("Eight folds: eight straight lines in the plane, and the field is flat between them.", R, 26, INK),
                ("More neurons, more straight edges. But a spiral is curved everywhere.", R, 26, INK2)]
        if k == "deep":
            return "THE CHAIN AGAIN: A SECOND LAYER", [
                ("layer 1:  nⱼ = max(0, wⱼ·x + bⱼ)      layer 2:  mₖ = max(0, vₖ·n + dₖ)", R, 26, INK),
                ("output = σ(a₁·m₁ + … + a₈·m₈ + c)", R, 28, INK),
                (f"now:  accuracy {acc:.0%}", D, 28, BLUE),
                ("A fold of an already folded field bends. Folds of folds: the edges curve.", R, 26, INK2)]
        return "", []

    def draw_card(self, ctx, t, fade):
        st, snap, tl = self.stage_at(t)
        if t >= T_CLOSE - 0.3:
            title, lines = "THE WHOLE TRICK", [
                ("A neuron:   max(0, w·x + b)          one straight fold in the input plane.", "regular", 27, INK),
                ("A layer:    a₁n₁ + a₂n₂ + …           many folds, added. Straight edges.", "regular", 27, INK),
                ("Another layer:   folds of the folded field.   The edges curve.", "regular", 27, INK),
                (f"Here: 1 → {self.facts['one']['acc']:.0%},  2 → {self.facts['two']['acc']:.0%},  3 → {self.facts['three']['acc']:.0%},  8 → {self.facts['eight']['acc']:.0%},  8 + 8 → {self.facts['deep']['acc']:.0%}.", "dbold", 27, BLUE)]
            a = ease_out_cubic(seg(t, T_CLOSE - 0.3, T_CLOSE)) * fade
        elif st is not None and t >= st["t0"]:
            title, lines = self.card_lines(st, snap)
            a = ease_out_cubic(seg(t, st["t0"], st["t0"] + 0.4)) * (1 - smooth(seg(t, st["t1"] - 0.4, st["t1"] - 0.1))) * fade
        else:
            return
        if a <= 0 or not lines:
            return
        x, y, w, h = CARD
        ctx.set_source_rgba(1, 1, 1, 0.88 * a)
        pill(ctx, x, y, w, h, 18)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(INK3), 0.45 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
        show_tracked(ctx, title, x + 34, y + 44, tr)
        yy = y + 92
        gap = 52 if len(lines) <= 4 else 44
        for text, kind, size, col in lines:
            font(ctx, kind, size)
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(x + 34, yy)
            ctx.show_text(text)
            yy += gap

    # ------------------------------------------------------------ background and frame

    def _build_grid(self):
        surf, ctx = layer()
        ctx.set_source_rgba(*rgb(INK2), GRID_A)
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
        base = np.ones((H, W, 3), np.float32) * np.array(rgb(BG), np.float32)
        base = base * (1 - ga)[..., None] + grgb
        self.bg = np.ascontiguousarray(base)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        st, snap, tl = self.stage_at(t)
        if st is not None and fade > 0:
            a_in = ease_out_cubic(seg(t, st["t0"] - 0.4, st["t0"] + 0.1)) * fade
            # the field, pasted under the overlays
            i0 = int(math.floor(snap))
            i1 = min(i0 + 1, st["n_snap"] - 1)
            k = snap - i0
            field = st["fields"][i0] * (1 - k) + st["fields"][i1] * k
            frgb, _ = self.field_rgb(field)
            px, py, n = PANEL
            sub = img[py:py + n, px:px + n]
            sub[:] = sub * (1 - a_in) + frgb * a_in
            surf, ctx = layer()
            self.draw_panel(ctx, st, snap, a_in)
            self.draw_diagram(ctx, st, snap, a_in, tl)
            self.readouts(ctx, st, snap, a_in)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
        if fade > 0:
            surf3, ctx3 = layer()
            self.draw_card(ctx3, t, fade)
            trgb, ta = surf_rgba(surf3)
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
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 180, 300, 420, 560, 700, 820, 960, 1100, 1250, 1400, 1560, 1700, 1900, 2050, 2250, 2500, 2700]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (560, 2250):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
