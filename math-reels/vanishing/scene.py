"""The vanishing gradient, measured, on the dark theme of reels 1-22 and 31-34. Two networks of 10 hidden
layers x 16 neurons on the two-arm spiral (vgsim.py), sigmoid on the left, ReLU on the right. A ladder of
11 rows per network (weight layer 11 at the top, next to the output; layer 1 at the bottom, next to the
input) fills from the top down as the blame travels backward: each row a log-scale bar and the measured
size of that layer's gradient at the start; between the sigmoid rows the factor it shrank by. Then why:
sigma'(z) never exceeds 0.25 (the measured average is drawn on it), ReLU'(z) is 1 for a neuron that is on.
Then both are trained with plain gradient descent (step 0.1, 3,000 steps): the ladder switches to how far
each layer's weights have moved, and the output fields, losses and counts train live from the snapshots.
50 s. frame(f) -> uint8 1080x1920 RGB."""

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
sys.path.insert(0, HERE)
from textkit import (H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba,  # noqa: E402
                     tracked_width)

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)

FPS = 60
N_FRAMES = 3000                       # 50 s
LOOP_FADE = (49.3, 49.85)
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6"
GOLD = "#F6C56E"
seg = gds.seg
TAU = 2 * math.pi
BG_TOP, BG_BOT = "#0E1226", "#05060C"
GRID_C, GRID_A = "#FFFFFF", 0.045
CARD_FILL, CARD_EDGE = "#121830", "#3A4460"
PANEL_BASE = "#161C33"
POINT_EDGE = "#05060C"
NPZ = os.path.join(HERE, "vanishing_sim.npz")

COLS = {"sigmoid": 40, "relu": 560}   # left edge of each column
COL_W = 480
COL_NAME = {"sigmoid": "SIGMOID", "relu": "ReLU"}
COL_ACC = {"sigmoid": BLUE, "relu": GOLD}
ROW_Y0, ROW_GAP = 612, 58             # baseline of the top row (layer 11), spacing
BAR_X, BAR_W = 64, 236                # bar offset and width inside a column
FIELD_Y, FIELD_N = 1300, 250          # the output fields (training phase)
PLOT_Y, PLOT_H = 1300, 250            # the slope plots (why phase)

# timeline
T_SETUP, T_BWD, ROW_DT = 2.6, 5.4, 0.78
T_SUM, T_WHY, T_WHY2, T_TRAIN = 14.8, 19.6, 23.6, 27.6
TRAIN_WIN = (29.0, 40.0)
T_CLOSE = 42.6
G_DEC = (-10.0, -1.0)                 # log10 range of the gradient bars
M_DEC = (-7.0, 0.0)                   # log10 range of the "moved" bars

SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m} × 10" + str(int(e)).translate(SUPER)


def pct(v):
    x = 100 * float(v)
    if x >= 10:
        return f"{x:.0f}%"
    if x >= 1:
        return f"{x:.1f}%"
    s = f"{x:.2g}"
    if "e" in s:
        s = f"{x:.10f}".rstrip("0")
        nz = len(s.split(".")[1]) - len(s.split(".")[1].lstrip("0"))
        s = f"{x:.{nz + 2}f}"
    return s + "%"


def pill(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


class Scene(gds.Scene):
    def __init__(self):
        if not os.path.exists(NPZ):
            import vgsim
            vgsim.main(NPZ)
        d = np.load(NPZ)
        self.X, self.y, self.grid = d["X"], d["y"], d["grid"]
        self.lim = float(self.grid[-1])
        self.n = {}
        for a in ("sigmoid", "relu"):
            self.n[a] = dict(g=d[f"{a}_g"], slope=d[f"{a}_slope_mean"], seed=int(d[f"{a}_seed"]), finals=d[f"{a}_finals"],
                             g10=d[f"{a}_g_median10"], loss=d[f"{a}_loss"], acc=d[f"{a}_acc"], rel=d[f"{a}_rel"],
                             pmin=d[f"{a}_pmin"], pmax=d[f"{a}_pmax"], field=d[f"{a}_field"])
        self.n_snap = len(self.n["relu"]["loss"])
        self.n_layers = len(self.n["relu"]["g"])
        S, R = self.n["sigmoid"], self.n["relu"]
        self.facts = dict(
            sig_g=[float(v) for v in S["g"]], relu_g=[float(v) for v in R["g"]],
            sig_ratio=float(S["g"][0] / S["g"][-1]), relu_ratio=float(R["g"][0] / R["g"][-1]),
            sig_factors=[float(S["g"][i + 1] / S["g"][i]) for i in range(self.n_layers - 1)],
            sig_slope=[float(v) for v in S["slope"]], relu_on=[float(v) for v in R["slope"]],
            sig_seed=S["seed"], relu_seed=R["seed"], sig_finals=[float(v) for v in S["finals"]], relu_finals=[float(v) for v in R["finals"]],
            sig_loss=float(S["loss"][-1]), relu_loss=float(R["loss"][-1]), sig_acc=float(S["acc"][-1]), relu_acc=float(R["acc"][-1]),
            sig_p=(float(S["pmin"][-1]), float(S["pmax"][-1])), relu_p=(float(R["pmin"][-1]), float(R["pmax"][-1])),
            sig_moved=[float(v) for v in S["rel"][-1]], relu_moved=[float(v) for v in R["rel"][-1]],
            sig_g10=[float(v) for v in S["g10"]], relu_g10=[float(v) for v in R["g10"]])
        self.cues = self._cues()
        self._build_grid()
        self._palette()
        self._panel_masks()

    # -------------------------------------------------------------- cues (every number from vgsim.py)

    def _cues(self):
        F = self.facts
        million = 1 / F["sig_ratio"] / 1e6
        hook = [("Ten layers deep, the first layer can't hear the error.", "black", 40, INK, 300, 1), ("The vanishing gradient, measured.", "black", 38, INK2, 390, 1)]
        slope_avg = float(np.mean(F["sig_slope"]))
        gm = (1 / F["sig_ratio"]) ** (1 / (len(F["sig_g"]) - 1))     # the average shrink per layer, geometric
        self.gm = gm
        return [
            (0.0, 2.4, hook, 0, 8, "none"),
            (T_SETUP, T_BWD - 0.3, [("Two networks, 10 hidden layers of 16.", "black", 44, INK, 300, 1), ("Same data, same depth. Only the fold differs.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_BWD, T_SUM - 0.3, [("Backward: the blame flows down, layer by layer.", "black", 40, INK, 300, 1), ("The size of each layer's gradient at the start.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_SUM, T_WHY - 0.3, [(f"Sigmoid: layer 1 gets {million:.0f} million times less.", "black", 40, INK, 300, 1), (f"ReLU: layer 1 still gets {F['relu_ratio']:.0%} of what layer 11 gets.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_WHY, T_WHY2 - 0.3, [("Why: every layer multiplies the blame by σ′(z).", "black", 40, INK, 300, 1), (f"σ′ is never above 0.25. Here it averages {slope_avg:.2f}: ÷4 from the fold alone.", "dbold", 33, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_WHY2, T_TRAIN - 0.3, [("ReLU′ is exactly 1 for a neuron that is on.", "black", 42, INK, 300, 1), ("The blame passes through it whole.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_TRAIN, 33.2, [("Now train both: plain gradient descent.", "black", 44, INK, 300, 1), ("Step size 0.1, 3,000 steps. How far does each layer move?", "dbold", 35, INK2, 390, 0.95)], 10, 8, "pop"),
            (33.5, 37.7, [(f"Sigmoid: layer 1 moves {pct(F['sig_moved'][0])}.", "black", 42, INK, 300, 1), (f"ReLU: layer 1 moves {pct(F['relu_moved'][0])}.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (38.0, T_CLOSE - 0.3, [(f"Sigmoid: every guess is {F['sig_p'][0]:.3f}. A coin flip.", "black", 40, INK, 300, 1), (f"ReLU: {F['relu_acc']:.0%} right, every layer learning.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_CLOSE, 49.0, [("A deep network needs the blame to survive the trip down.", "black", 36, INK, 300, 1), (f"ReLU passes it on. Sigmoid shrinks it {gm:.0f}× a layer, ten times over.", "dbold", 35, INK2, 390, 0.95)], 10, 8, "pop"),
            (49.4, 50.1, hook, 12, 0, "fade"),
        ]

    # -------------------------------------------------------------- time helpers

    def train_snap(self, t):
        u = seg(t, *TRAIN_WIN)
        u = 1 - (1 - u) ** 2.2
        return u * (self.n_snap - 1)

    def lerp_snap(self, arr, sn):
        i0 = int(math.floor(sn))
        i1 = min(i0 + 1, self.n_snap - 1)
        k = sn - i0
        return arr[i0] * (1 - k) + arr[i1] * k

    def row_time(self, li):
        """When the blame reaches weight layer li (0 = layer 1, at the bottom)."""
        return T_BWD + (self.n_layers - 1 - li) * ROW_DT

    # -------------------------------------------------------------- the field

    def _panel_masks(self):
        n = FIELD_N
        surf, ctx = layer(n, n)
        pill(ctx, 0, 0, n, n, 14)
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()
        _, a = surf_rgba(surf)
        self.panel_mask = a.astype(np.float32)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
        self.panel_vign = (1 - 0.22 * np.clip(r, 0, 1.2) ** 2).astype(np.float32)

    def _palette(self):
        w = np.array(rgb(PANEL_BASE), np.float32)
        cb, cv = np.array(rgb(BLUE), np.float32), np.array(rgb(VERM), np.float32)
        lut = np.empty((256, 3), np.float32)
        for i, p in enumerate(np.linspace(0, 1, 256)):
            lut[i] = w + ((cb - w) * 0.78 * (0.5 - p) * 2 if p < 0.5 else (cv - w) * 0.78 * (p - 0.5) * 2)
        self.lut = lut

    def field_rgb(self, field):
        n = FIELD_N
        up = cv2.resize(field[::-1].astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)   # row 0 of the grid is y = -lim
        idx = np.clip((up * 255).round().astype(np.int32), 0, 255)
        return self.lut[idx], up

    def field_xy(self, act):
        return COLS[act] + 14, FIELD_Y

    def to_field(self, act, x, y):
        fx, fy = self.field_xy(act)
        return fx + (x + self.lim) / (2 * self.lim) * FIELD_N, fy + (self.lim - y) / (2 * self.lim) * FIELD_N

    def draw_field_overlay(self, ctx, act, up, a):
        fx, fy = self.field_xy(act)
        n = FIELD_N
        ctx.save()
        pill(ctx, fx, fy, n, n, 14)
        ctx.clip()
        if float(up.max() - up.min()) > 0.02:             # a decision boundary only where the field has one
            cs, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for lw, al in ((6.0, 0.16), (1.8, 0.95)):
                ctx.set_source_rgba(*rgb(INK), al * a)
                ctx.set_line_width(lw)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                for c in cs:
                    if len(c) < 6:
                        continue
                    pts = c[:, 0, :]
                    ctx.move_to(fx + pts[0][0] + 0.5, fy + pts[0][1] + 0.5)
                    for x, y in pts[1:]:
                        ctx.line_to(fx + x + 0.5, fy + y + 0.5)
                    ctx.close_path()
                    ctx.stroke()
        for (x, y), c in zip(self.X, self.y):
            sx, sy = self.to_field(act, x, y)
            col = rgb(VERM if c > 0.5 else BLUE)
            ctx.set_source_rgba(*col, 0.95 * a)
            ctx.arc(sx, sy, 2.0, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(POINT_EDGE), 0.7 * a)
            ctx.set_line_width(0.6)
            ctx.arc(sx, sy, 2.0, 0, TAU)
            ctx.stroke()
        ctx.restore()
        for lw, al in ((4.0, 0.08), (1.2, 0.55)):
            ctx.set_source_rgba(*rgb(INK2), al * a)
            ctx.set_line_width(lw)
            pill(ctx, fx + 0.5, fy + 0.5, n - 1, n - 1, 14)
            ctx.stroke()

    # -------------------------------------------------------------- the ladder

    def ladder(self, ctx, t, a):
        """Column headers, the 11 rows: the starting gradient (blame) of each layer, or, while training, how far it has moved."""
        if a <= 0:
            return
        L = self.n_layers
        # which mode: 0 = gradients, 1 = moved
        mode = smooth(seg(t, T_TRAIN + 0.4, T_TRAIN + 1.2)) * (1 - smooth(seg(t, LOOP_FADE[0], LOOP_FADE[1])))
        hook_full = 1 - smooth(seg(t, 2.2, 2.7))           # the opening shows the finished ladder
        end_full = ease_out_cubic(seg(t, 49.45, 49.95))
        sn = self.train_snap(t)
        for act, x0 in COLS.items():
            N = self.n[act]
            # header
            font(ctx, "black", 40)
            name = COL_NAME[act]
            w_ = ctx.text_extents(name).x_advance
            ctx.set_source_rgba(*rgb(COL_ACC[act]), a)
            ctx.move_to(x0 + COL_W / 2 - w_ / 2, ROW_Y0 - 74)
            ctx.show_text(name)
            font(ctx, "bold", 16)
            tr = 0.12 * 16
            sub = "BLAME REACHING EACH LAYER (GRADIENT SIZE)" if mode < 0.5 else "HOW FAR EACH LAYER HAS MOVED"
            sa = (1 - 2 * mode) if mode < 0.5 else (2 * mode - 1)
            sw = tracked_width(ctx, sub, tr)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a * sa)
            show_tracked(ctx, sub, x0 + COL_W / 2 - sw / 2, ROW_Y0 - 42, tr)
            # decade ticks under the bars
            dec = G_DEC if mode < 0.5 else M_DEC
            for k in range(int(dec[1] - dec[0]) + 1):
                xx = x0 + BAR_X + BAR_W * k / (dec[1] - dec[0])
                ctx.set_source_rgba(*rgb(INK3), 0.18 * a)
                ctx.set_line_width(1.0)
                ctx.move_to(xx, ROW_Y0 - 20)
                ctx.line_to(xx, ROW_Y0 + (L - 1) * ROW_GAP + 10)
                ctx.stroke()
            for r in range(L):
                li = L - 1 - r                              # row 0 is the top: weight layer 11
                y = ROW_Y0 + r * ROW_GAP
                font(ctx, "bold", 18)
                lab = f"{li + 1}"
                ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
                ctx.move_to(x0 + 30 - ctx.text_extents(lab).x_advance, y)
                ctx.show_text(lab)
                # track
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.set_source_rgba(*rgb("#232A44"), 0.9 * a)
                ctx.set_line_width(12)
                ctx.move_to(x0 + BAR_X, y - 6)
                ctx.line_to(x0 + BAR_X + BAR_W, y - 6)
                ctx.stroke()
                # the gradient value (mode 0) and the moved value (mode 1)
                g = float(N["g"][li])
                appear = max(ease_out_cubic(seg(t, self.row_time(li), self.row_time(li) + 0.45)), hook_full, end_full)
                if mode < 1:
                    fr = (math.log10(g) - G_DEC[0]) / (G_DEC[1] - G_DEC[0])
                    self.bar(ctx, x0, y, fr, MAGENTA, appear * a * (1 - mode), sci(g), act, li, appear)
                if mode > 0:
                    mv = float(self.lerp_snap(N["rel"][:, li], sn))
                    fr = (math.log10(max(mv, 1e-12)) - M_DEC[0]) / (M_DEC[1] - M_DEC[0]) if mv > 0 else 0.0
                    self.bar(ctx, x0, y, fr, GREEN, a * mode, pct(mv) if mv > 0 else "0%", act, li, 1.0)
                # the factor between sigmoid rows (gradient mode only)
                if act == "sigmoid" and li < L - 1 and mode < 1:
                    fa = ease_out_cubic(seg(t, self.row_time(li) + 0.2, self.row_time(li) + 0.6))
                    fa = max(fa, hook_full, end_full) * (1 - mode) * a
                    if fa > 0:
                        f_ = float(N["g"][li + 1] / N["g"][li])
                        s = f"÷{f_:.1f}"
                        font(ctx, "dbold", 16)
                        ctx.set_source_rgba(*rgb(MAGENTA), 0.75 * fa)
                        ctx.move_to(x0 + COL_W - 2 - ctx.text_extents(s).x_advance, y - ROW_GAP / 2 + 2)
                        ctx.show_text(s)
            # the layer axis
            font(ctx, "bold", 15)
            tr = 0.12 * 15
            for s, yy in (("OUTPUT SIDE", ROW_Y0 - 20), ("INPUT SIDE", ROW_Y0 + (L - 1) * ROW_GAP + 30)):
                ctx.set_source_rgba(*rgb(INK3), 0.7 * a)
                show_tracked(ctx, s, x0 + 8, yy, tr)
            # the travelling blame: a magenta dot moving down the rows while the blame flows
            if T_BWD - 0.1 < t < self.row_time(0) + 0.5:
                u = (t - T_BWD) / ROW_DT
                yy = ROW_Y0 - 6 + min(max(u, 0), L - 1) * ROW_GAP
                al = a * min(1.0, (t - T_BWD + 0.1) / 0.3, (self.row_time(0) + 0.5 - t) / 0.3)
                g_ = cairo.RadialGradient(x0 + 44, yy, 1, x0 + 44, yy, 16)
                g_.add_color_stop_rgba(0, *rgb(MAGENTA), 0.9 * al)
                g_.add_color_stop_rgba(1, *rgb(MAGENTA), 0.0)
                ctx.set_source(g_)
                ctx.arc(x0 + 44, yy, 16, 0, TAU)
                ctx.fill()

    def bar(self, ctx, x0, y, fr, col, al, label, act, li, appear):
        if al <= 0:
            return
        fr = max(0.0, min(1.0, fr)) * ease_out_cubic(min(1.0, appear * 1.2))
        L_ = BAR_W * fr
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        if L_ > 0.5:
            ctx.set_source_rgba(*rgb(col), 0.22 * al)
            ctx.set_line_width(18)
            ctx.move_to(x0 + BAR_X, y - 6)
            ctx.line_to(x0 + BAR_X + L_, y - 6)
            ctx.stroke()
            ctx.set_source_rgba(*rgb(col), 0.95 * al)
            ctx.set_line_width(10)
            ctx.move_to(x0 + BAR_X, y - 6)
            ctx.line_to(x0 + BAR_X + L_, y - 6)
            ctx.stroke()
        font(ctx, "dbold", 23)
        ctx.set_source_rgba(*rgb(INK), al)
        ctx.move_to(x0 + BAR_X + BAR_W + 18, y + 2)
        ctx.show_text(label)

    # -------------------------------------------------------------- the why: slopes

    def slope_plots(self, ctx, t, a):
        if a <= 0:
            return
        F = self.facts
        for act, x0 in COLS.items():
            show = ease_out_cubic(seg(t, T_WHY if act == "sigmoid" else T_WHY2, (T_WHY if act == "sigmoid" else T_WHY2) + 0.5)) * a
            if show <= 0:
                continue
            px, py, pw, ph = x0 + 40, PLOT_Y + 46, COL_W - 70, PLOT_H - 76
            zs = np.linspace(-6, 6, 241)
            ys = 1 / (1 + np.exp(-zs))
            ds = ys * (1 - ys) if act == "sigmoid" else (zs > 0).astype(float)

            def P(z, v):
                return px + (z + 6) / 12 * pw, py + ph - v * ph

            # axes and gridlines at 0, 0.25 and 1
            for v, lab in ((0.0, "0"), (0.25, "0.25"), (1.0, "1")):
                ctx.set_source_rgba(*rgb(INK3), (0.5 if v == 0 else 0.22) * show)
                ctx.set_line_width(1.0)
                ctx.set_dash([] if v == 0 else [5.0, 5.0])
                ctx.move_to(*P(-6, v))
                ctx.line_to(*P(6, v))
                ctx.stroke()
                ctx.set_dash([])
                font(ctx, "bold", 16)
                ctx.set_source_rgba(*rgb(INK3), 0.95 * show)
                ctx.move_to(px - 4 - ctx.text_extents(lab).x_advance, P(0, v)[1] + 6)
                ctx.show_text(lab)
            ctx.set_source_rgba(*rgb(INK3), 0.4 * show)
            ctx.move_to(*P(0, 0))
            ctx.line_to(*P(0, 1.05))
            ctx.stroke()
            # the curve, drawn on
            k = ease_out_cubic(seg(t, (T_WHY if act == "sigmoid" else T_WHY2) + 0.2, (T_WHY if act == "sigmoid" else T_WHY2) + 1.4))
            m = max(2, int(len(zs) * k))
            for lw, al in ((9.0, 0.18), (3.2, 1.0)):
                ctx.set_source_rgba(*rgb(MAGENTA), al * show)
                ctx.set_line_width(lw)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                for i in range(m):
                    xx, yy = P(zs[i], ds[i])
                    if i == 0 or (act == "relu" and zs[i - 1] <= 0 < zs[i]):
                        if i > 0 and act == "relu":
                            ctx.line_to(*P(0, 0))
                            ctx.stroke()
                            ctx.move_to(*P(0, 1))
                        else:
                            ctx.move_to(xx, yy)
                    else:
                        ctx.line_to(xx, yy)
                ctx.stroke()
            font(ctx, "dbold", 24)
            ctx.set_source_rgba(*rgb(INK), show)
            title = "σ′(z), the sigmoid's slope" if act == "sigmoid" else "ReLU′(z), the ReLU's slope"
            ctx.move_to(px - 10, py - 22)
            ctx.show_text(title)
            font(ctx, "regular", 19)
            ctx.set_source_rgba(*rgb(INK2), 0.95 * show * k)
            if act == "sigmoid":
                avg = float(np.mean(F["sig_slope"]))
                note = f"never above 0.25;  measured here: {avg:.2f}"
                xx, yy = P(0, 0.25)
                ctx.set_source_rgba(*rgb(MAGENTA), show * k)
                ctx.arc(xx, yy, 5.5, 0, TAU)
                ctx.fill()
            else:
                on = float(np.mean(F["relu_on"]))
                note = f"1 when on, 0 when off;  on here: {on:.0%} of neurons"
            ctx.set_source_rgba(*rgb(INK2), 0.95 * show * k)
            ctx.move_to(px, py + ph + 34)
            ctx.show_text(note)
            font(ctx, "bold", 15)
            ctx.set_source_rgba(*rgb(INK3), 0.9 * show)
            ctx.move_to(px + pw - ctx.text_extents("z").x_advance, py + ph + 14)
            ctx.show_text("z")

    def rule_card(self, ctx, t, a):
        """While the blame flows: what multiplies it at each layer (in words)."""
        if a <= 0:
            return
        x, y, w, h = 70, 1300, 940, 250
        ctx.set_source_rgba(*rgb(CARD_FILL), 0.80 * a)
        pill(ctx, x, y, w, h, 20)
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgb(CARD_EDGE), 0.9 * a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        lines = [("One layer down, the blame is multiplied by", "regular", 30, INK2, 1356, 0.0),
                 ("its weights  ×  the slope of the fold", "dbold", 40, INK, 1424, 0.4),
                 ("Ten layers down: ten of these multiplications.", "regular", 30, INK2, 1500, 1.0)]
        for text, kind, size, col, yy, delay in lines:
            la = ease_out_cubic(seg(t, T_BWD + 0.6 + delay, T_BWD + 1.1 + delay)) * a
            if la <= 0:
                continue
            font(ctx, kind, size)
            tw = ctx.text_extents(text).x_advance
            assert tw < w - 40, text
            if "slope" in text:
                a_ = "its weights  ×  "
                font(ctx, kind, size)
                w0 = ctx.text_extents(a_).x_advance
                ctx.set_source_rgba(*rgb(INK), la)
                ctx.move_to(540 - tw / 2, yy)
                ctx.show_text(a_)
                ctx.set_source_rgba(*rgb(MAGENTA), la)
                ctx.move_to(540 - tw / 2 + w0, yy)
                ctx.show_text("the slope of the fold")
            else:
                ctx.set_source_rgba(*rgb(col), la)
                ctx.move_to(540 - tw / 2, yy)
                ctx.show_text(text)

    # -------------------------------------------------------------- training readouts

    def train_readouts(self, ctx, t, a):
        if a <= 0:
            return
        sn = self.train_snap(t)
        step = int(round(sn)) * 25
        for act, x0 in COLS.items():
            N = self.n[act]
            fx, fy = self.field_xy(act)
            rx = fx + FIELD_N + 22
            loss = float(self.lerp_snap(N["loss"], sn))
            acc = float(N["acc"][int(round(sn))])
            pmin, pmax = float(N["pmin"][int(round(sn))]), float(N["pmax"][int(round(sn))])
            items = (("LOSS", f"{loss:.3f}"), ("RIGHT", f"{int(round(acc * len(self.y)))} / {len(self.y)}"),
                     ("GUESSES", f"{pmin:.3f}–{pmax:.3f}"))
            for k, (lab, val) in enumerate(items):
                yy = fy + 26 + k * 82
                font(ctx, "bold", 16)
                tr = 0.12 * 16
                ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
                show_tracked(ctx, lab, rx, yy, tr)
                font(ctx, "dbold", 30 if k < 2 else 25)
                ctx.set_source_rgba(*rgb(INK), a)
                ctx.move_to(rx, yy + 38)
                ctx.show_text(val)
        font(ctx, "bold", 19)
        tr = 0.12 * 19
        lab = f"TRAINING STEP {step:,} / 3,000   ·   PLAIN GRADIENT DESCENT, STEP SIZE 0.1"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK2), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, FIELD_Y - 26, tr)

    def footer(self, ctx, a):
        if a <= 0:
            return
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "2 → 16 × 10 → 1   ·   THE SPIRAL, 400 POINTS   ·   GRADIENTS AVERAGED OVER ALL POINTS"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK2), 0.8 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        lab = "BARS ON A LOG SCALE: EACH TICK IS 10×   ·   MEDIAN OF 6 SEEDS"
        w = tracked_width(ctx, lab, tr)
        assert w < 1000, w
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1788, tr)

    # -------------------------------------------------------------- background and frame

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
        for (cx, cy, rad, col, amp) in ((280, 900, 560.0, BLUE, 0.06), (800, 900, 560.0, GOLD, 0.045), (540, 1450, 520.0, MAGENTA, 0.035)):
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
            g = np.clip(1 - d, 0, 1) ** 2 * amp
            base = base * (1 - g)[..., None] + np.array(rgb(col), np.float32) * g[..., None]
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        base = base * vig[..., None]
        # a hairline between the two columns
        surf2, ctx2 = layer()
        ctx2.set_source_rgba(*rgb(CARD_EDGE), 0.8)
        ctx2.set_line_width(1.0)
        ctx2.move_to(540.5, 500)
        ctx2.line_to(540.5, 1690)
        ctx2.stroke()
        lrgb, la = surf_rgba(surf2)
        self.bg = np.ascontiguousarray(base)
        self.divider = (lrgb, la)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        img = self.bg.copy()
        a_all = max(fade, ease_out_cubic(seg(t, 49.45, 49.95)))
        over(img, self.divider[0] * a_all, self.divider[1] * a_all)
        # fields (training phase), pasted before the overlays
        a_tr = ease_out_cubic(seg(t, T_TRAIN + 0.6, T_TRAIN + 1.3)) * fade
        sn = self.train_snap(t)
        ups = {}
        if a_tr > 0:
            for act in COLS:
                field = self.lerp_snap(self.n[act]["field"], sn)
                frgb, up = self.field_rgb(field)
                ups[act] = up
                fx, fy = self.field_xy(act)
                sub = img[fy:fy + FIELD_N, fx:fx + FIELD_N]
                am = (self.panel_mask * a_tr)[..., None]
                sub[:] = sub * (1 - am) + (frgb * self.panel_vign[..., None]) * am
        surf, ctx = layer()
        self.ladder(ctx, t, a_all)
        a_rule = ease_out_cubic(seg(t, T_BWD + 0.4, T_BWD + 0.9)) * (1 - smooth(seg(t, T_WHY - 0.5, T_WHY - 0.1))) * fade
        self.rule_card(ctx, t, a_rule)
        a_why = ease_out_cubic(seg(t, T_WHY - 0.2, T_WHY + 0.3)) * (1 - smooth(seg(t, T_TRAIN - 0.2, T_TRAIN + 0.4))) * fade
        self.slope_plots(ctx, t, a_why)
        if a_tr > 0:
            for act in COLS:
                self.draw_field_overlay(ctx, act, ups[act], a_tr)
            self.train_readouts(ctx, t, a_tr)
        self.footer(ctx, a_all)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s")
    for k, v in sc.facts.items():
        print(" ", k, v)
    for c in sc.cues:
        print(round(c[0], 2), round(c[1], 2), c[2][0][0], "|", c[2][1][0])
    frames = [0, 200, 340, 520, 700, 900, 1000, 1180, 1300, 1450, 1650, 1780, 1900, 2050, 2280, 2450, 2700, 2990]
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        times.append(time.time() - t0)
        ims.append(cv2.resize(im[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (0, 1000, 1300, 1500, 2280, 2450):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b_, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a).mean(), "step", np.abs(c - b_).mean())
