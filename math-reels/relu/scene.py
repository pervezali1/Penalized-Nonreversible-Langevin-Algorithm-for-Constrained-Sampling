"""Ten layers without an activation are one neuron. Dark theme of reels 31-50, maths in Computer Modern (stage3d.tex),
narrated by Kokoro (voicekit). The spiral of reels 31-47 and a network of 10 layers of 8 neurons (681 numbers):
trained with no activation it reaches 62.0%, exactly what one neuron gets, and its layers collapse one by one into
a single neuron (3 numbers, the same output to 14 decimal places); from the same start with ReLU after every
neuron it reaches 98.5% (relusim.py). On screen: the network (edges coloured by the weights as they train), the map
of what it predicts, the score, the count of numbers, and the training curves. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import cairo
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import paste  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "nnbuild"))
sys.path.insert(0, os.path.join(ROOT, "neurons"))
from nnsim import DATA_SEED, LIM  # noqa: E402
from train import spiral  # noqa: E402

_spec = importlib.util.spec_from_file_location("gds_scene", os.path.join(ROOT, "gdsimple", "scene.py"))
gds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gds)
seg = gds.seg
FPS = 60
TAU = 2 * math.pi
INK, INK2, INK3 = "#F3F5FA", "#B3BBCC", "#6F788C"
VERM, BLUE, GREEN, MAGENTA, GOLD, CYAN = "#FF7A4D", "#5B8DFF", "#34D399", "#F472B6", "#F6C56E", "#38BDF8"
BG_TOP, BG_BOT = "#0E1226", "#05060C"
PANEL_BASE = "#161C33"
POINT_EDGE = "#0B0E1A"

sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "relu_fields.npz")):          # the archive is not committed: rebuild it (a minute)
    import relusim
    relusim.main()
with open(os.path.join(HERE, "relu_results.json")) as fh:
    R = json.load(fh)
F = np.load(os.path.join(HERE, "relu_fields.npz"))
F_LIN, F_REL, S_LIN, S_REL = F["f_lin"], F["f_rel"], F["s_lin"], F["s_rel"]
SIZES = R["sizes"]
STEPS, SNAP = R["steps"], R["snap"]
ACC_LIN, ACC_REL = np.array(R["acc_linear"]), np.array(R["acc_relu"])
ACC_ONE = R["acc_one"][-1]
N_PARAMS = R["n_params"]
COLLAPSE = R["collapse"]                                 # numbers left after merging 1..11 weight layers
DECIMALS = R["decimals"]
WE, BE = R["w_eff"], R["b_eff"]
X, Y = spiral(np.random.default_rng(DATA_SEED))


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


def unpack(vec):
    ps, k = [], 0
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        ps.append(vec[k:k + a * b].reshape(a, b))
        k += a * b
        ps.append(vec[k:k + b])
        k += b
    return ps


assert (N_PARAMS, R["right_linear"], R["right_one"], round(1000 * ACC_REL[-1]), COLLAPSE[-1]["numbers"], DECIMALS) == (681, 248, 248, 985, 3, 14), "script numbers changed"
SCRIPT = [
    "This network has ten layers and six hundred and eighty-one numbers. And the only thing it can ever draw is a straight line.",
    "Train it on the spiral, with fifteen hundred steps of backpropagation: sixty-two percent. Exactly what one single neuron gets.",
    "The problem: there's no activation. Each layer just multiplies by a matrix, and adds some numbers. And a matrix times a matrix is just one matrix.",
    "So the ten layers collapse, one by one, into a single neuron. Three numbers. The same answer for every point, to fourteen decimal places.",
    "Now add one tiny step after every neuron: if the number is negative, make it zero. That's ReLU.",
    "Same ten layers. Same starting numbers. Same training. Ninety-eight point five percent.",
    "Those bends are what let a network learn a shape. Without them, all that depth does nothing.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 train it, 2 why, 3 the collapse, 4 ReLU, 5 train with ReLU, 6 close
SEG_MIN = [6.0, 9.0, 9.5, 10.0, 7.5, 8.5, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_LIN = (SEG[1] + 0.8, word_t(1, "sixty-two"))           # linear training, steps 0 -> 1,500
T_ONE = word_t(1, "Exactly")
T_COL = (word_t(3, "collapse") - 0.2, word_t(3, "Three numbers") + 0.2)   # the ten hidden layers merge
T_EXP = (SEG[4] + 0.1, SEG[4] + 1.1)                     # the layers come back
T_KINK = word_t(4, "if the number")
T_REL = (SEG[5] + 0.5, word_t(5, "Ninety-eight"))        # ReLU training
HOME = (T_END - 0.65, T_END - 0.05)                      # back to the opening for the loop

NET = (96, 984, 650, 27)                                 # x of the input column, x of the output, centre y, spacing
PANEL = (70, 860, 470)                                   # the map: x, y, size
RIGHT = 590                                              # the right-hand column
CHART = (610, 1170, 390, 150)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= HOME[0]:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("trainlin", "why", "collapse", "relu", "trainrelu", "close")[k - 1]
    return "close"


def relu_on(t):
    md = mode(t)
    if md in ("relu", ) and t < T_KINK:
        return 0.0
    if md == "relu":
        return smooth(seg(t, T_KINK, T_KINK + 0.8))
    return 1.0 if md in ("trainrelu", "close") else 0.0


def step_at(t):
    md = mode(t)
    if md == "trainlin":
        return int(round(STEPS * smooth(seg(t, *T_LIN)) ** 1.0))
    if md == "relu":
        return 0
    if md == "trainrelu":
        return int(round(STEPS * seg(t, *T_REL) ** 1.3))
    return STEPS


def merged_at(t):
    """How many of the ten hidden layers have merged into the one before (continuous, 0..10)."""
    md = mode(t)
    if md == "collapse":
        return 10 * smooth(seg(t, *T_COL))
    if md == "relu":
        return 10 * (1 - smooth(seg(t, *T_EXP)))
    return 0.0


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int(0.5 * (T_COL[0] + T_COL[1]) * FPS), int((T_KINK + 1.0) * FPS), int(0.6 * (T_REL[0] + T_REL[1]) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n_params=N_PARAMS, acc_linear=float(ACC_LIN[-1]), acc_one=ACC_ONE, acc_relu=float(ACC_REL[-1]), collapsed=dict(w=WE, b=BE),
                          numbers=[c["numbers"] for c in COLLAPSE], max_diff=R["max_diff"], decimals=DECIMALS, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        self._palette()
        self._panel_frame()

    def _cues(self):
        S_ = SEG
        hook = [("10 layers. 681 numbers.", "black", 52, INK, 300, 1), ("It can only draw a straight line.", "dbold", 42, VERM, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Trained on the spiral.", "black", 50, INK, 300, 1), (f"{pc(ACC_LIN[-1])}%. One neuron gets {pc(ACC_ONE)}%.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("No activation.", "black", 52, INK, 300, 1), ("Each layer: times a matrix, plus numbers.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Ten layers collapse into one.", "black", 48, INK, 300, 1), (f"{N_PARAMS} numbers → 3.", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Add one step: ReLU.", "black", 52, MAGENTA, 300, 1), ("Negative? Make it zero.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("Same network. Same start.", "black", 50, INK, 300, 1), (f"With ReLU: {pc(ACC_REL[-1])}%.", "dbold", 44, MAGENTA, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("The bends do the learning.", "black", 50, INK, 300, 1), ("Without them, depth does nothing.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END - 0.3, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    def _build_bg(self):
        surf, ctx = layer()
        ctx.set_source_rgba(1, 1, 1, 0.045)
        ctx.set_line_width(1.0)
        for x in range(54, W, 54):
            ctx.move_to(x + 0.5, 0)
            ctx.line_to(x + 0.5, H)
        for y in range(54, H, 54):
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
        vig = 1 - 0.30 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.35, 0, 1) ** 2
        self.bg = np.ascontiguousarray(base * vig[..., None])

    def _palette(self):
        w = np.array(rgb(PANEL_BASE), np.float32)
        cb, cv = np.array(rgb(BLUE), np.float32), np.array(rgb(VERM), np.float32)
        lut = np.empty((256, 3), np.float32)
        for i, p in enumerate(np.linspace(0, 1, 256)):
            lut[i] = w + (cb - w) * 0.78 * (0.5 - p) * 2 if p < 0.5 else w + (cv - w) * 0.78 * (p - 0.5) * 2
        self.lut = lut

    def _panel_frame(self):
        px, py, n = PANEL
        surf, ctx = layer()
        S3.pill(ctx, px, py, n, n, 18)
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()
        _, ta = surf_rgba(surf)
        self.panel_mask = ta[py:py + n, px:px + n].copy()

    # ------------------------------------------------------------ state

    def state(self, t):
        """(field, accuracy, weights, ReLU 0..1, merged 0..10, colour) at time t."""
        md = mode(t)
        s = step_at(t)
        k = min(s // SNAP, len(F_LIN) - 1)
        ron = relu_on(t)
        if md == "hook":
            u = smooth(seg(t, *HOME)) if t >= HOME[0] else 1.0
            f_ = F_REL[-1] * (1 - u) + F_LIN[-1] * u
            return f_, (float(ACC_REL[-1]) if u < 0.5 else float(ACC_LIN[-1])), (S_REL[-1] if u < 0.5 else S_LIN[-1]), 1 - u, 0.0, (MAGENTA if u < 0.5 else INK2)
        if md in ("trainlin", "why", "collapse"):
            return F_LIN[k], float(ACC_LIN[s]), S_LIN[k], 0.0, merged_at(t), INK2
        if md == "relu":
            return F_LIN[-1] if t < T_EXP[1] else F_REL[0], float(ACC_LIN[-1]) if t < T_EXP[1] else float(ACC_REL[0]), S_LIN[-1] if t < T_EXP[0] else S_REL[0], ron, merged_at(t), INK2
        return F_REL[k], float(ACC_REL[s]), S_REL[k], 1.0, 0.0, MAGENTA

    # ------------------------------------------------------------ the network

    def draw_net(self, ctx, t, vec, ron, u, a):
        x0, x1, yc, sp = NET
        ps = unpack(vec)
        nh = len(SIZES) - 2
        cs = [float(np.clip(u - i, 0, 1)) for i in range(nh)]       # hidden column i+1 merging into the one before
        gaps = [1 - c for c in cs] + [1.0]
        cum = np.concatenate([[0.0], np.cumsum(gaps)])
        xs = [x0 + (x1 - x0) * cum[i] / cum[-1] for i in range(nh + 2)]

        def ys(n):
            return [yc + (j - (n - 1) / 2) * sp for j in range(n)]
        vis = [0] + [i + 1 for i in range(nh) if cs[i] < 1] + [nh + 1]
        alpha_col = [1.0] + [1 - c for c in cs] + [1.0]
        solid = [0] + [i + 1 for i in range(nh) if cs[i] <= 0] + [nh + 1]

        def edges(ia, ib, fade):
            # the product of the weights between two drawn columns (the merged layers in between)
            Wm = ps[2 * ia]
            for m in range(ia + 1, ib):
                Wm = Wm @ ps[2 * m]
            mx = np.abs(Wm).max() + 1e-12
            ya, yb = ys(SIZES[ia]), ys(SIZES[ib])
            for p in range(Wm.shape[0]):
                for q in range(Wm.shape[1]):
                    w_ = Wm[p, q] / mx
                    ctx.set_source_rgba(*rgb(VERM if w_ > 0 else BLUE), a * (0.12 + 0.75 * abs(w_) ** 0.8) * fade)
                    ctx.set_line_width(0.6 + 1.6 * abs(w_))
                    ctx.move_to(xs[ia], ya[p])
                    ctx.line_to(xs[ib], yb[q])
                    ctx.stroke()
        # a column that is merging fades out with its edges, while the merged edges that skip it fade in
        for ia, ib in zip(vis[:-1], vis[1:]):
            edges(ia, ib, min(alpha_col[ia], alpha_col[ib]))
        for ia, ib in zip(solid[:-1], solid[1:]):
            if ib - ia > 1 and any(0 < cs[m - 1] < 1 for m in range(ia + 1, ib)):
                edges(ia, ib, max(cs[m - 1] for m in range(ia + 1, ib) if 0 < cs[m - 1] < 1))
        for ci in range(nh + 2):
            al = alpha_col[ci]
            if al <= 0.01:
                continue
            for yy in ys(SIZES[ci]):
                ctx.set_source_rgba(*rgb(PANEL_BASE), a * al)
                ctx.arc(xs[ci], yy, 11, 0, TAU)
                ctx.fill()
                hid = 0 < ci < nh + 1
                ctx.set_source_rgba(*rgb(MAGENTA if (hid and ron > 0.5) else INK2), a * al * 0.9)
                ctx.set_line_width(1.6)
                ctx.arc(xs[ci], yy, 11, 0, TAU)
                ctx.stroke()
                if hid and ron > 0:
                    # ReLU, drawn inside the neuron: flat, then rising
                    ctx.set_line_width(2.0)
                    xx = xs[ci]
                    ctx.set_source_rgba(*rgb(MAGENTA), a * al * ron)
                    ctx.move_to(xx - 7, yy + 4)
                    ctx.line_to(xx - 0.5, yy + 4)
                    ctx.line_to(xx + 6, yy - 5)
                    ctx.stroke()
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        top = yc - 4 * sp - 30
        show_tracked(ctx, "INPUT", xs[0] - 22, top, 0.1 * 16)
        s = "OUTPUT"
        show_tracked(ctx, s, xs[-1] - 58, top, 0.1 * 16)
        nvis = len(vis) - 2
        lab = f"{nvis} LAYER{'S' if nvis != 1 else ''} OF 8 NEURONS" if nvis else "NO LAYERS LEFT: ONE NEURON"
        lab += "   ·   ReLU" if ron > 0.5 else ("   ·   NO ACTIVATION" if nvis else "")
        w_ = tracked_width(ctx, lab, 0.1 * 16)
        show_tracked(ctx, lab, 540 - w_ / 2, top, 0.1 * 16)

    # ------------------------------------------------------------ the map

    def to_panel(self, x, y):
        px, py, n = PANEL
        return px + (x + LIM) / (2 * LIM) * n, py + (LIM - y) / (2 * LIM) * n

    def draw_map(self, img, f_, acc, col, a):
        px, py, n = PANEL
        up = cv2.resize(f_[::-1].astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        rgbf = self.lut[np.clip((up * 255).round().astype(np.int32), 0, 255)]
        m = self.panel_mask[..., None] * a
        sub = img[py:py + n, px:px + n]
        sub[:] = sub * (1 - m) + rgbf * m
        surf, ctx = layer()
        ctx.save()
        S3.pill(ctx, px, py, n, n, 18)
        ctx.clip()
        cs, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for lw, al in ((7.0, 0.14), (2.0, 0.9)):
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
        for (x, y), c in zip(X, Y):
            sx, sy = self.to_panel(x, y)
            ctx.set_source_rgba(*rgb(VERM if c > 0.5 else BLUE), 0.95 * a)
            ctx.arc(sx, sy, 3.4, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(POINT_EDGE), 0.8 * a)
            ctx.set_line_width(0.8)
            ctx.arc(sx, sy, 3.4, 0, TAU)
            ctx.stroke()
        ctx.restore()
        ctx.set_source_rgba(*rgb(INK2), 0.55 * a)
        ctx.set_line_width(1.4)
        S3.pill(ctx, px, py, n, n, 18)
        ctx.stroke()
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "WHAT IT PREDICTS, EVERYWHERE", px, py - 14, 0.1 * 16)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ the right-hand column

    def draw_right(self, ctx, t, acc, col, u, a):
        x = RIGHT
        md = mode(t)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "OF THE 400 POINTS, RIGHT", x, 874, 0.1 * 16)
        font(ctx, "black", 76)
        s = f"{pc(acc)}%"
        ctx.set_source_rgba(*rgb(col), a)
        ctx.move_to(x, 952)
        ctx.show_text(s)
        if md == "trainlin" and t >= T_ONE - 0.3 or md in ("why", "collapse"):
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(INK2), a * smooth(seg(t, T_ONE - 0.3, T_ONE + 0.3)) if md == "trainlin" else a)
            ctx.move_to(x, 994)
            ctx.show_text(f"one single neuron: {pc(ACC_ONE)}%")
        if md == "relu" and t >= T_EXP[1]:
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(INK2), a * smooth(seg(t, T_EXP[1], T_EXP[1] + 0.4)))
            ctx.move_to(x, 994)
            ctx.show_text("back to the random start")
        if md in ("trainrelu", "close"):
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(x, 994)
            ctx.show_text(f"without ReLU: {pc(ACC_LIN[-1])}%")
        # the numbers in the network
        j = int(math.floor(u + 1e-9))
        n_now = COLLAPSE[j]["numbers"] if j < len(COLLAPSE) else 3
        if u >= 10 - 1e-6:
            n_now = COLLAPSE[-1]["numbers"]
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "NUMBERS IN THE NETWORK", x, 1046, 0.1 * 16)
        font(ctx, "black", 64)
        ctx.set_source_rgba(*rgb(GOLD), a)
        ctx.move_to(x, 1116)
        ctx.show_text(f"{n_now}")
        self.chart(ctx, t, a)

    def chart(self, ctx, t, a):
        x0, y0, w, h = CHART
        md = mode(t)
        lo = 0.5

        def gx(s):
            return x0 + s / STEPS * w

        def gy(v):
            return y0 + h - (v - lo) / (1 - lo) * h
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        ctx.set_source_rgba(1, 1, 1, 0.07 * a)
        for v in (0.625, 0.75, 0.875, 1.0):
            ctx.move_to(x0, gy(v))
            ctx.line_to(x0 + w, gy(v))
        ctx.stroke()
        font(ctx, "dbold", 16)
        ctx.set_source_rgba(*rgb(INK3), a)
        for v, s in ((0.5, "50%"), (1.0, "100%")):
            ctx.move_to(x0 - 8 - ctx.text_extents(s).x_advance, gy(v) + 5)
            ctx.show_text(s)
        s = f"{STEPS:,} STEPS"
        ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, y0 + h + 22)
        ctx.show_text(s)
        s_lin = step_at(t) if md == "trainlin" else (0 if md == "hook" and t < SEG[1] else STEPS)
        if md == "hook":
            s_lin = STEPS
        s_rel = step_at(t) if md == "trainrelu" else (STEPS if md in ("close", "hook") else -1)
        if md == "hook" and t >= HOME[0]:
            s_rel = STEPS if smooth(seg(t, *HOME)) < 0.5 else -1
        elif md == "hook":
            s_rel = -1
        for acc, upto, col in ((ACC_LIN, s_lin, INK2), (ACC_REL, s_rel, MAGENTA)):
            if upto < 0:
                continue
            ctx.set_source_rgba(*rgb(col), a)
            ctx.set_line_width(3.0)
            ctx.move_to(gx(0), gy(max(lo, acc[0])))
            for s in range(5, upto + 1, 5):
                ctx.line_to(gx(s), gy(max(lo, acc[s])))
            ctx.line_to(gx(upto), gy(max(lo, acc[upto])))
            ctx.stroke()
            ctx.arc(gx(upto), gy(max(lo, acc[upto])), 4.5, 0, TAU)
            ctx.fill()
        font(ctx, "dbold", 17)
        for acc, upto, col, lab in ((ACC_LIN, s_lin, INK2, "no activation"), (ACC_REL, s_rel, MAGENTA, "with ReLU")):
            if upto >= STEPS:
                ctx.set_source_rgba(*rgb(col), a)
                ctx.move_to(x0 + w - ctx.text_extents(lab).x_advance, gy(max(lo, acc[STEPS])) + 36)
                ctx.show_text(lab)
        font(ctx, "bold", 15)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "ACCURACY, BY TRAINING STEP", x0, y0 - 14, 0.1 * 15)

    # ------------------------------------------------------------ the card

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        md = mode(t)
        if md == "relu" and t >= T_KINK - 0.4:
            # a small plot of ReLU
            gx0, gy0, gs = 120, y + 150, 70
            al = a * smooth(seg(t, T_KINK - 0.4, T_KINK + 0.4))
            ctx.set_source_rgba(*rgb(INK3), al)
            ctx.set_line_width(1.2)
            ctx.move_to(gx0 - gs, gy0)
            ctx.line_to(gx0 + gs, gy0)
            ctx.move_to(gx0, gy0 + 10)
            ctx.line_to(gx0, gy0 - gs)
            ctx.stroke()
            ctx.set_source_rgba(*rgb(MAGENTA), al)
            ctx.set_line_width(4)
            ctx.move_to(gx0 - gs, gy0)
            ctx.line_to(gx0, gy0)
            ctx.line_to(gx0 + gs * 0.85, gy0 - gs * 0.85)
            ctx.stroke()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if md in ("hook", "close"):
            paste(img, r"$\mathrm{no\ activation:}\ \ 681\ \mathrm{numbers} = 3\ \mathrm{numbers},\ \ 62.0\%$", 28, 540, y + 96, INK2, a, align="c")
            paste(img, rf"$\mathrm{{the\ same\ network\ with\ ReLU:}}\ \ {pc(ACC_REL[-1])}\%$", 30, 540, y + 158, MAGENTA, a, align="c")
            paste(img, r"$2 \rightarrow 8 \rightarrow 8 \rightarrow \cdots \rightarrow 8 \rightarrow 1\ \ (10\ \mathrm{layers}),\ \ \mathrm{400\ spiral\ points}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "trainlin":
            paste(img, r"$h_k = h_{k-1}\, W_k + b_k\qquad (k = 1, \ldots, 10)$", 34, 540, y + 100, INK, a, align="c")
            paste(img, r"$p = \sigma(h_{10}\, W_{11} + b_{11}),\qquad \mathrm{1{,}500\ steps\ of\ backpropagation}$", 23, 540, y + 168, INK2, a, align="c")
            paste(img, rf"$\mathrm{{one\ neuron}},\ p = \sigma(w_1 x + w_2 y + b),\ \mathrm{{trained\ the\ same\ way}}:\ {pc(ACC_ONE)}\%$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "why":
            paste(img, r"$(x\, W_1 + b_1)\, W_2 + b_2 \;=\; x\, (W_1 W_2) + (b_1 W_2 + b_2)$", 32, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{with\ no\ activation,\ two\ layers\ are\ just\ one\ layer}$", 23, 540, y + 168, INK2, a, align="c")
        elif md == "collapse":
            w1, w2 = WE
            sgn = "+" if w2 >= 0 else "-"
            sb = "+" if BE >= 0 else "-"
            paste(img, rf"$p = \sigma\left({w1:.4f}\, x\ {sgn}\ {abs(w2):.4f}\, y\ {sb}\ {abs(BE):.4f}\right)$".replace("-0.", "{-}0."), 34, 540, y + 100, GOLD, a, align="c")
            paste(img, rf"$W_1 W_2 \cdots W_{{11}}:\ \mathrm{{every\ output\ the\ same\ to}}\ {DECIMALS}\ \mathrm{{decimal\ places}}$", 24, 540, y + 166, INK2, a, align="c")
            paste(img, r"$\mathrm{a\ single\ neuron\ trained\ alone:\ the\ same\ line,\ to\ 4\ decimals}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "relu":
            paste(img, r"$\mathrm{ReLU}(z) = \max(0,\ z)$", 40, 600, y + 104, MAGENTA, a, align="c")
            paste(img, r"$h_k = \mathrm{ReLU}(h_{k-1}\, W_k + b_k)$", 30, 600, y + 170, INK, a, align="c")
            paste(img, r"$\mathrm{the\ same\ 681\ numbers,\ the\ same\ random\ start}$", 21, 600, y + h - 18, INK3, a, align="c")
        else:
            paste(img, r"$h_k = \mathrm{ReLU}(h_{k-1}\, W_k + b_k):\ \ \mathrm{every\ neuron\ bends}$", 28, 540, y + 100, MAGENTA, a, align="c")
            paste(img, rf"$\mathrm{{the\ same\ 1{{,}}500\ steps:}}\ \ {R['right_relu']}\ \mathrm{{of\ the}}\ 400\ \mathrm{{points\ right}}$", 24, 540, y + 166, INK2, a, align="c")
            paste(img, r"$\mathrm{9\ random\ starts:\ ReLU,\ median\ 98.5\%;\ no\ activation,\ 62.0\%\ every\ time}$", 20, 540, y + h - 18, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        f_, acc, vec, ron, u, col = self.state(t)
        self.draw_map(img, f_, acc, col, a)
        surf, ctx = layer()
        self.draw_net(ctx, t, vec, ron, u, a)
        self.draw_right(ctx, t, acc, col, u, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "400 SPIRAL POINTS   ·   REAL TRAINING RUNS   ·   ALL COMPUTED"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w_ / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


if __name__ == "__main__":
    import time

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
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]))
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
