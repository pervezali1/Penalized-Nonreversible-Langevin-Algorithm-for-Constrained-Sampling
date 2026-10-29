"""What each neuron is looking at: the sixteen hidden neurons of reel 31's trained two-layer network
(nnbuild/nnbuild_train.npz, the 8 + 8 run), one field each, on the dark theme. Layer 1: eight tiles, each
neuron's activation max(0, w.x + b) over the input plane, bright where the neuron is on, with its fold
line. Layer 2: eight tiles, each a fold of a weighted sum of the eight fields above: the shapes curve.
Then the output assembles: the layer-2 fields are added one at a time with their output weights, in the
order that helps the accuracy most, and the big field shows the running sum with its decision boundary
and the live accuracy. 45 s. frame(f) -> uint8 1080x1920 RGB."""

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
N_FRAMES = 2700                       # 45 s
LOOP_FADE = (44.3, 44.85)
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GOLD = "#FF7A4D", "#5B8DFF", "#F6C56E"
seg, lerp = gds.seg, gds.lerp
TAU = 2 * math.pi
BG_TOP, BG_BOT = "#0E1226", "#05060C"
GRID_C, GRID_A = "#FFFFFF", 0.045
CARD_FILL, CARD_EDGE = "#121830", "#3A4460"
PANEL_BASE = "#161C33"
POINT_EDGE = "#05060C"
CARD = (70, 1575, 940, 300)
COL_X = (190, 540, 890)
NPZ = os.path.join(ROOT, "nnbuild", "nnbuild_train.npz")

# timeline
T_A, T_B, T_C, T_CLOSE = 2.8, 13.3, 24.8, 40.4
TRANS = 0.7                           # layout transition length
HILITE_DT = 0.9                       # highlight dwell per neuron
ADD_T0, ADD_DT = 27.0, 1.25           # the output assembles: first field added, then one every ADD_DT


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def sg(v, nd=2):
    v = float(v)
    if abs(v) < 0.5 * 10 ** -nd:
        v = 0.0
    return f"{v:+.{nd}f}"


SUB = "₀₁₂₃₄₅₆₇₈₉"


def sub(i):
    return SUB[i]


# ---------------------------------------------------------------- layouts: tile rects per stage (x, y, size)
def grid_2x4(x0, y0, size, gap):
    return [(x0 + (i % 4) * (size + gap), y0 + (i // 4) * (size + gap), size) for i in range(8)]


def strip_8(x0, y0, size, gap):
    return [(x0 + i * (size + gap), y0, size) for i in range(8)]


def col_4x2(x0, y0, size, gap):
    return [(x0 + (i % 2) * (size + gap), y0 + (i // 2) * (size + gap), size) for i in range(8)]


LAYOUT = {
    "A": dict(l1=grid_2x4(80, 600, 215, 20), l2=None, out=None),
    "B": dict(l1=strip_8(68, 560, 112, 8), l2=grid_2x4(80, 720, 215, 20), out=None),
    "C": dict(l1=None, l2=col_4x2(60, 610, 108, 14), out=(420, 600, 480)),
}


class Scene(gds.Scene):
    def __init__(self):
        d = np.load(NPZ)
        self.X, self.y, self.grid = d["X"], d["y"], d["grid"]
        self.lim = float(self.grid[-1])
        W1, b1, V, dv, a, c = [d[f"deep_p{i}"] for i in range(6)]
        self.W1, self.b1, self.V, self.dv, self.a, self.c = W1, b1, V, dv, a[:, 0], float(c[0])
        self.acc_full = float(d["deep_acc"][-1])
        G = len(self.grid)
        GX = np.stack(np.meshgrid(self.grid, self.grid), -1).reshape(-1, 2)
        n = np.maximum(GX @ W1 + b1, 0)
        m = np.maximum(n @ V + dv, 0)
        self.n_fields = [n[:, j].reshape(G, G) / max(1e-9, n[:, j].max()) for j in range(8)]
        self.m_fields = [m[:, k].reshape(G, G) / max(1e-9, m[:, k].max()) for k in range(8)]
        self.m_raw = m.reshape(G, G, 8)
        # the output, assembled in the order that helps the accuracy most
        nX = np.maximum(self.X @ W1 + b1, 0)
        mX = np.maximum(nX @ V + dv, 0)

        def acc(idx):
            logit = mX[:, idx] @ self.a[idx] + self.c if idx else np.full(len(self.y), self.c)
            return float(np.mean((logit > 0) == (self.y > 0.5)))

        order = []
        for _ in range(8):
            best = max((j for j in range(8) if j not in order), key=lambda j: acc(order + [j]))
            order.append(best)
        self.order = order
        self.partial_acc = [acc(order[:k]) for k in range(9)]
        self.partial = []
        for k in range(9):
            logit = self.m_raw[:, :, order[:k]] @ self.a[order[:k]] + self.c if k else np.full((G, G), self.c)
            self.partial.append(1 / (1 + np.exp(-logit)))
        self.facts = dict(order=[int(o) + 1 for o in order], partial_acc=self.partial_acc, a=[float(x) for x in self.a], c=self.c,
                          acc_full=self.acc_full, l1_max=[float(x) for x in n.max(0)], l2_max=[float(x) for x in m.max(0)])
        self.cues = self._cues()
        self._build_grid()
        self._palettes()

    # ------------------------------------------------------------ time and layout

    def stage(self, t):
        if t < T_B:
            return "A"
        if t < T_C:
            return "B"
        return "C"

    def rects(self, t):
        """Tile rects for layer 1, layer 2 and the output at time t, interpolated across the layout transitions."""
        def mix(r0, r1, k):
            if r0 is None or r1 is None:
                return r1 if k >= 0.5 else r0
            return tuple(a + (b - a) * k for a, b in zip(r0, r1))

        def mixl(l0, l1, k):
            if l0 is None and l1 is None:
                return None
            if l0 is None:
                return l1
            if l1 is None:
                return l0
            return [mix(a, b, k) for a, b in zip(l0, l1)]

        if t < T_B:
            L = LAYOUT["A"]
            return L["l1"], L["l2"], L["out"], 1.0, 0.0, 0.0
        if t < T_C:
            k = smooth(seg(t, T_B, T_B + TRANS))
            A, B = LAYOUT["A"], LAYOUT["B"]
            return mixl(A["l1"], B["l1"], k), B["l2"], None, 1.0, 1.0, 0.0
        k = smooth(seg(t, T_C, T_C + TRANS))
        B, C = LAYOUT["B"], LAYOUT["C"]
        return B["l1"], mixl(B["l2"], C["l2"], k), C["out"], 1.0 - k, 1.0, k

    def highlight(self, t):
        """(layer, index) of the highlighted neuron, or (None, None)."""
        if T_A + 2.4 <= t < T_B - 0.4:
            j = int((t - T_A - 2.4) / HILITE_DT)
            return (1, j) if j < 8 else (None, None)
        if T_B + 2.9 <= t < T_C - 0.4:
            j = int((t - T_B - 2.9) / HILITE_DT)
            return (2, j) if j < 8 else (None, None)
        return None, None

    def n_added(self, t):
        if t < ADD_T0:
            return 0, 0.0
        x = (t - ADD_T0) / ADD_DT
        k = min(8, int(x) + 1)
        return k, min(1.0, x - (k - 1)) if k < 9 else 1.0

    # ------------------------------------------------------------ cues (every number from the trained network)

    def _cues(self):
        o = self.facts["order"]
        pa = self.partial_acc
        hook = [("What is each neuron looking at?", "black", 48, INK, 300, 1), ("The sixteen neurons of the last reel, one field each.", "black", 36, INK2, 390, 1)]
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (T_A, T_A + 2.6, [("Layer 1: eight neurons, eight fields.", "black", 46, INK, 300, 1), ("Bright where the neuron is on, dark where it is off.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_A + 2.9, T_B - 0.3, [("Each one is a straight fold.", "black", 48, INK, 300, 1), ("Off on one side of its line, a ramp on the other. Nothing else.", "dbold", 34, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_B, T_B + 3.0, [("Layer 2: eight more, each fed by all eight above.", "black", 40, INK, 300, 1), ("A fold of a weighted sum of folds.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_B + 3.3, T_C - 0.3, [("Folds of folds: the shapes curve.", "black", 48, INK, 300, 1), ("No neuron sees the spiral. Each sees one piece of it.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_C, ADD_T0 - 0.2, [("The output: the eight fields of layer 2, weighted and added.", "black", 36, INK, 300, 1), ("Added one at a time, in the order that helps most.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (ADD_T0, ADD_T0 + 8 * ADD_DT + 0.6, [("Watch the spiral assemble.", "black", 48, INK, 300, 1), (f"{pa[1]:.0%} with one field, {pa[4]:.0%} with four, {pa[8]:.0%} with all eight.", "dbold", 36, VERM, 390, 0.95)], 10, 8, "pop"),
            (ADD_T0 + 8 * ADD_DT + 0.9, T_CLOSE - 0.3, [(f"All eight: {pa[8]:.0%}. Take any one away and it drops.", "black", 38, INK, 300, 1), ("The shape lives in the sum, not in any neuron.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, 44.0, [("Sixteen simple fields. One spiral.", "black", 48, INK, 300, 1), ("Each neuron is simple. The shape is in how they add up.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (44.4, 45.1, hook, 12, 0, "fade"),
        ]

    # ------------------------------------------------------------ palettes and tiles

    def _palettes(self):
        base = np.array(rgb(PANEL_BASE), np.float32)
        gold = np.array(rgb(GOLD), np.float32)
        ps = np.linspace(0, 1, 256)
        self.lut_act = np.stack([base + (gold - base) * (p ** 0.8) for p in ps]).astype(np.float32)
        cb, cv = np.array(rgb(BLUE), np.float32), np.array(rgb(VERM), np.float32)
        lut = np.empty((256, 3), np.float32)
        for i, p in enumerate(ps):
            if p < 0.5:
                lut[i] = base + (cb - base) * 0.78 * (0.5 - p) * 2
            else:
                lut[i] = base + (cv - base) * 0.78 * (p - 0.5) * 2
        self.lut_out = lut

    def paste(self, img, field, rect, lut, alpha):
        x, y, s = rect
        x, y, s = int(round(x)), int(round(y)), int(round(s))
        if s <= 2 or alpha <= 0:
            return
        up = cv2.resize(field.astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC)
        idx = np.clip((up * 255).round().astype(np.int32), 0, 255)
        tile = lut[idx]
        sub = img[y:y + s, x:x + s]
        sub[:] = sub * (1 - alpha) + tile * alpha

    def to_rect(self, x, y, rect):
        rx, ry, s = rect
        return rx + (x + self.lim) / (2 * self.lim) * s, ry + (self.lim - y) / (2 * self.lim) * s

    def fold_line(self, ctx, j, rect, a):
        w1, w2, b = float(self.W1[0, j]), float(self.W1[1, j]), float(self.b1[j])
        nn_ = math.hypot(w1, w2)
        if nn_ < 1e-6:
            return
        x0, y0 = -b * w1 / nn_ ** 2, -b * w2 / nn_ ** 2
        dx, dy = -w2 / nn_, w1 / nn_
        L = self.lim * 3
        rx, ry, s = rect
        ctx.save()
        ctx.rectangle(rx, ry, s, s)
        ctx.clip()
        ctx.set_source_rgba(*rgb(INK), 0.7 * a)
        ctx.set_line_width(1.5)
        ctx.set_dash([5.0, 4.0])
        ctx.move_to(*self.to_rect(x0 - dx * L, y0 - dy * L, rect))
        ctx.line_to(*self.to_rect(x0 + dx * L, y0 + dy * L, rect))
        ctx.stroke()
        ctx.restore()

    def tile_frame(self, ctx, rect, label, a, hi=False, col=INK3):
        rx, ry, s = rect
        ctx.set_source_rgba(*rgb(INK if hi else col), (0.95 if hi else 0.6) * a)
        ctx.set_line_width(3.0 if hi else 1.2)
        ctx.rectangle(rx + 0.5, ry + 0.5, s - 1, s - 1)
        ctx.stroke()
        if label and s >= 90:
            font(ctx, "dbold", 20 if s >= 150 else 16)
            ctx.set_source_rgba(*rgb(INK), 0.95 * a)
            ctx.move_to(rx + 8, ry + (24 if s >= 150 else 20))
            ctx.show_text(label)

    def draw_points(self, ctx, rect, a, r=3.0):
        for (x, y), c in zip(self.X, self.y):
            sx, sy = self.to_rect(x, y, rect)
            ctx.set_source_rgba(*rgb(VERM if c > 0.5 else BLUE), 0.9 * a)
            ctx.arc(sx, sy, r, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(POINT_EDGE), 0.9 * a)
            ctx.set_line_width(0.8)
            ctx.arc(sx, sy, r, 0, TAU)
            ctx.stroke()

    def draw_boundary(self, ctx, field, rect, a):
        rx, ry, s = rect
        s = int(round(s))
        up = cv2.resize(field.astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC)
        cs, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ctx.save()
        ctx.rectangle(rx, ry, s, s)
        ctx.clip()
        ctx.set_source_rgba(*rgb(INK), 0.85 * a)
        ctx.set_line_width(2.2)
        for c in cs:
            if len(c) < 6:
                continue
            pts = c[:, 0, :]
            ctx.move_to(rx + pts[0][0] + 0.5, ry + pts[0][1] + 0.5)
            for x, y in pts[1:]:
                ctx.line_to(rx + x + 0.5, ry + y + 0.5)
            ctx.close_path()
            ctx.stroke()
        ctx.restore()

    # ------------------------------------------------------------ readouts and card

    def readouts(self, ctx, t, a):
        st = self.stage(t)
        k, _ = self.n_added(t)
        if st == "A":
            items = (("LAYER", "1", None), ("FIELDS ON SCREEN", "8", None), ("ACCURACY", "–", None))
        elif st == "B":
            items = (("LAYER", "2", None), ("FIELDS ON SCREEN", "8 + 8", None), ("ACCURACY", "–", None))
        else:
            items = (("OUTPUT", "sum", None), ("FIELDS ADDED", f"{k} / 8", k / 8), ("ACCURACY", f"{self.partial_acc[k]:.0%}", self.partial_acc[k]))
        Y_LAB, Y_VAL, Y_BAR = 1404, 1466, 1489
        for (lab, val, frac), xc in zip(items, COL_X):
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
                ctx.set_source_rgba(*rgb("#2A3148"), a)
                ctx.move_to(xc - 110, Y_BAR)
                ctx.line_to(xc + 110, Y_BAR)
                ctx.stroke()
                ctx.set_source_rgba(*rgb(VERM), a)
                ctx.move_to(xc - 110, Y_BAR)
                ctx.line_to(xc - 110 + 220 * min(1.0, frac), Y_BAR)
                ctx.stroke()
        font(ctx, "bold", 22)
        tr = 0.12 * 22
        lab = f"THE 8 + 8 NETWORK OF REEL 31   ·   TRAINED TO {self.acc_full:.0%} ON THE SPIRAL"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1372, tr)

    def card_content(self, t):
        st = self.stage(t)
        R, D = "regular", "dbold"
        layer_, j = self.highlight(t)
        if t >= T_CLOSE - 0.3:
            return "SIXTEEN FIELDS, ONE SHAPE", [
                ("Layer 1:   nⱼ = max(0, wⱼ·x + bⱼ)        eight straight folds.", R, 27, INK),
                ("Layer 2:   mₖ = max(0, vₖ·n + dₖ)        folds of those: curved pieces.", R, 27, INK),
                ("Output:    σ(a₁m₁ + … + a₈m₈ + c)         the pieces, weighted and added.", R, 27, INK),
                (f"No single neuron knows the spiral. Together: {self.acc_full:.0%}.", D, 27, VERM)]
        if st == "A":
            lines = [("nⱼ = max(0,  wⱼ·x + bⱼ)        bright where the sum is positive, dark where it is not", R, 26, INK),
                     ("The dashed line on each tile is where the sum is zero: the fold.", R, 27, INK)]
            if layer_ == 1:
                lines.append((f"n{sub(j + 1)} = max(0,  {sg(self.W1[0, j])}·x₁ {sg(self.W1[1, j])}·x₂ {sg(self.b1[j])})", D, 28, GOLD))
            else:
                lines.append(("Eight neurons, eight lines, eight ramps.", D, 28, GOLD))
            lines.append(("A single neuron cannot curve. That is all a layer of them can draw: straight creases.", R, 25, INK2))
            return "LAYER 1", lines
        if st == "B":
            lines = [("mₖ = max(0,  vₖ·n + dₖ)        a weighted sum of the eight layer-1 fields, then a fold", R, 26, INK)]
            if layer_ == 2:
                v = self.V[:, j]
                top = np.argsort(-np.abs(v))[:3]
                terms = " ".join(f"{sg(v[i], 1)}·n{sub(i + 1)}" for i in top)
                lines.append((f"m{sub(j + 1)} = max(0,  {terms} … {sg(self.dv[j], 1)})", D, 28, GOLD))
                lines.append((f"its three biggest inputs; it is on where that mix is positive", R, 26, INK))
            else:
                lines.append(("Each layer-2 field is a fold of a mix of the eight above.", D, 28, GOLD))
                lines.append(("Where the mix crosses zero follows the old creases, so the new crease bends.", R, 26, INK))
            lines.append(("Folds of folds. This is the step that buys curves.", R, 26, INK2))
            return "LAYER 2", lines
        k, _ = self.n_added(t)
        o = self.order
        terms = " ".join(f"{sg(self.a[o[i]], 1)}·m{sub(o[i] + 1)}" for i in range(k))
        lines = [("output = σ(a₁·m₁ + a₂·m₂ + … + a₈·m₈ + c)        σ: squash to a probability", R, 26, INK),
                 ((f"so far:  σ({terms} {sg(self.c, 1)})" if k else f"so far:  σ({sg(self.c, 1)})   nothing added yet"), D, 25 if k >= 5 else 27, GOLD),
                 (f"accuracy with {k} of 8 fields: {self.partial_acc[k]:.0%}", D, 27, VERM),
                 ("Positive weights push towards vermilion, negative towards blue. The shape is in the sum.", R, 25, INK2)]
        return "THE OUTPUT", lines

    def draw_card(self, ctx, t, fade):
        title, lines = self.card_content(t)
        a = fade * ease_out_cubic(seg(t, T_A - 0.2, T_A + 0.4))
        if a <= 0 or not lines:
            return
        x, y, w, h = CARD
        ctx.set_source_rgba(*rgb(CARD_FILL), 0.90 * a)
        pill(ctx, x, y, w, h, 18)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(CARD_EDGE), 0.9 * a)
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
        self.bg = np.ascontiguousarray(base)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        if t < T_A - 0.3 or fade <= 0:
            surf2, ctx2 = layer()
            self._headlines(ctx2, t)
            trgb, ta = surf_rgba(surf2)
            over(img, trgb, ta)
            return dither(img, f)
        l1, l2, out_rect, a_l1, a_l2, a_out = self.rects(t)
        st = self.stage(t)
        layer_, hj = self.highlight(t)
        k_add, k_frac = self.n_added(t)
        # tiles appear one by one at each stage start
        def appear(t0, i, dt=0.22):
            return ease_out_cubic(seg(t, t0 + i * dt, t0 + i * dt + 0.35))
        # ---- paste the fields
        a1 = [appear(T_A, i) * a_l1 * fade for i in range(8)]
        a2 = [appear(T_B + TRANS, i) * a_l2 * fade for i in range(8)] if l2 is not None else [0] * 8
        if l1 is not None:
            for j in range(8):
                self.paste(img, self.n_fields[j], l1[j], self.lut_act, a1[j])
        if l2 is not None:
            for j in range(8):
                self.paste(img, self.m_fields[j], l2[j], self.lut_act, a2[j])
        if out_rect is not None and a_out > 0:
            # the running sum: cross-fade between k-1 and k fields while the k-th is being added
            if k_add == 0:
                fld = self.partial[0]
            else:
                fld = self.partial[k_add - 1] * (1 - k_frac) + self.partial[k_add] * k_frac
            self.paste(img, fld, out_rect, self.lut_out, a_out * fade)
        # ---- overlays
        surf, ctx = layer()
        if l1 is not None:
            for j in range(8):
                if a1[j] <= 0:
                    continue
                self.fold_line(ctx, j, l1[j], a1[j])
                self.tile_frame(ctx, l1[j], f"n{sub(j + 1)}", a1[j], hi=(layer_ == 1 and hj == j))
        if l2 is not None:
            for j in range(8):
                if a2[j] <= 0:
                    continue
                hi = (layer_ == 2 and hj == j)
                lab = f"m{sub(j + 1)}"
                self.tile_frame(ctx, l2[j], lab, a2[j], hi=hi)
                if st == "C" and a_out > 0.5:
                    # the output weight of this field, and whether it has been added yet
                    pos = self.order.index(j)
                    added = pos < k_add
                    rx, ry, s = l2[j]
                    font(ctx, "dbold", 17)
                    txt = f"×{sg(self.a[j], 1)}"
                    col = VERM if self.a[j] > 0 else BLUE
                    ctx.set_source_rgba(*rgb(col), (0.95 if added else 0.35) * a2[j])
                    ctx.move_to(rx + s - ctx.text_extents(txt).x_advance - 6, ry + s - 8)
                    ctx.show_text(txt)
                    if not added:
                        ctx.set_source_rgba(0, 0, 0, 0.45 * a2[j])
                        ctx.rectangle(rx, ry, s, s)
                        ctx.fill()
                    elif pos == k_add - 1 and k_frac < 1:
                        # the field being added right now: a bright ring and a line to the output
                        ctx.set_source_rgba(*rgb(GOLD), 0.9 * a2[j])
                        ctx.set_line_width(3.0)
                        ctx.rectangle(rx - 2, ry - 2, s + 4, s + 4)
                        ctx.stroke()
                        ox, oy, osz = out_rect
                        ctx.set_source_rgba(*rgb(GOLD), 0.6 * a2[j])
                        ctx.set_line_width(2.0)
                        ctx.move_to(rx + s, ry + s / 2)
                        ctx.line_to(ox, oy + osz / 2)
                        ctx.stroke()
        if out_rect is not None and a_out > 0:
            fld = self.partial[k_add] if k_frac >= 1 or k_add == 0 else self.partial[k_add - 1] * (1 - k_frac) + self.partial[k_add] * k_frac
            if k_add > 0:
                self.draw_boundary(ctx, fld, out_rect, a_out * fade)
            self.draw_points(ctx, out_rect, a_out * fade)
            self.tile_frame(ctx, out_rect, "", a_out * fade, col=INK2)
            font(ctx, "bold", 20)
            tr = 0.12 * 20
            lab = "OUTPUT: THE WEIGHTED SUM"
            w = tracked_width(ctx, lab, tr)
            ox, oy, osz = out_rect
            ctx.set_source_rgba(*rgb(INK2), 0.85 * a_out * fade)
            show_tracked(ctx, lab, ox + osz / 2 - w / 2, oy - 16, tr)
        # layer captions
        font(ctx, "bold", 20)
        tr = 0.12 * 20
        if l1 is not None and a_l1 > 0:
            rx, ry, s = l1[0]
            lab = "LAYER 1   ·   max(0, w·x + b)   ·   EIGHT FOLDS"
            ctx.set_source_rgba(*rgb(INK2), 0.85 * a_l1 * fade * appear(T_A, 0))
            show_tracked(ctx, lab, rx, ry - 16, tr)
        if l2 is not None and a_l2 > 0 and st != "C":
            rx, ry, s = l2[0]
            lab = "LAYER 2   ·   max(0, v·n + d)   ·   FOLDS OF FOLDS"
            ctx.set_source_rgba(*rgb(INK2), 0.85 * a_l2 * fade * appear(T_B + TRANS, 0))
            show_tracked(ctx, lab, rx, ry - 16, tr)
        if st == "C" and a_out > 0.5:
            rx, ry, s = l2[0]
            lab = "LAYER 2  ×  WEIGHT"
            ctx.set_source_rgba(*rgb(INK2), 0.85 * a_out * fade)
            show_tracked(ctx, lab, rx, ry - 16, tr)
        self.readouts(ctx, t, fade * ease_out_cubic(seg(t, T_A, T_A + 0.5)))
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
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
    frames = [0, 200, 320, 480, 640, 790, 830, 960, 1100, 1300, 1500, 1560, 1640, 1800, 1950, 2100, 2250, 2400]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (480, 1100, 1950):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
