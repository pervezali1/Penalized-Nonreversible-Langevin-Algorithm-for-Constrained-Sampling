"""Move the digit one pixel. Dark theme of reels 31-51, maths in Computer Modern (stage3d.tex), narrated by Kokoro
(voicekit). The handwriting network of reel 49 reads 96.7% of the 540 unseen digits; moved one pixel right it reads
46.5%, one pixel left 39.1%. Its neurons are wired to fixed pixel positions. A small convolutional network (32
filters of 3 x 3 slid over every position, the strongest response of each kept, 650 numbers, trained the same way)
reads 95.9%, and 96.5% / 95.9% after the same moves (cnnsim.py). On screen: the digit sliding, both networks'
probabilities, the accuracy by pixels moved, one neuron's weights under the ink, a filter sliding over the digit
and its responses before and after the move. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from textkit import H, W, dither, ease_out_cubic, font, layer, over, rgb, show_tracked, smooth, surf_rgba, tracked_width  # noqa: E402
import stage3d as S3  # noqa: E402
import voicekit  # noqa: E402
from stage3d import paste  # noqa: E402

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

sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "cnn_results.json")):
    import cnnsim
    cnnsim.main()
with open(os.path.join(HERE, "cnn_results.json")) as fh:
    R = json.load(fh)
sys.path.insert(0, os.path.join(ROOT, "adversarial"))
sys.path.insert(0, os.path.join(ROOT, "digits"))
import advsim  # noqa: E402
_MLP, XTE, YTE = advsim.trained()                      # the network of reel 49 (retrained exactly, about a second)
SHOW = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
ACC = {k: {int(s): v for s, v in d.items()} for k, d in R["acc"].items()}
PS = {s["dx"]: s for s in R["per_shift"]}
DIG = R["digit"]
X0 = np.array(PS[0]["x"])
KF = R["filter"]
FILT = np.array(R["filters"][KF]).reshape(3, 3)
BC = R["bc"][KF]
N_MLP, N_CNN = R["n_mlp"], R["n_cnn"]
STRIP = R["strip"]


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


def pp(v):
    """A probability as a percentage that never rounds up to 100%."""
    if v < 0.999:
        return f"{100 * v:.1f}%"
    if v < 0.9999:
        return f"{100 * v:.2f}%"
    return ">99.99%"


def shifted(x64, dx):
    I = np.asarray(x64).reshape(8, 8)
    O = np.zeros_like(I)
    if dx >= 0:
        O[:, dx:] = I[:, :8 - dx]
    else:
        O[:, :8 + dx] = I[:, -dx:]
    return O.ravel()


# the neuron shown: the one the 0 excites most before the move
_h0 = np.maximum(X0 @ _MLP[0] + _MLP[1], 0)
_h1 = np.maximum(shifted(X0, 1) @ _MLP[0] + _MLP[1], 0)
NEURON = int(np.argmax(_h0))
H_BEFORE, H_AFTER = float(_h0[NEURON]), float(_h1[NEURON])

assert (round(1000 * ACC["mlp"][1]), round(1000 * ACC["mlp"][-1]), round(100 * ACC["cnn"][0]), round(100 * ACC["cnn"][1]), round(100 * ACC["cnn"][-1]), round(100 * ACC["cnn"][2]), DIG) == (465, 391, 96, 96, 96, 94, 0), "script numbers changed"
assert int(np.argmax(PS[1]["p_mlp"])) == 2 and int(np.argmax(PS[-1]["p_mlp"])) == 4 and int(np.argmax(PS[-2]["p_mlp"])) == 1
SCRIPT = [
    "This network reads handwriting: ninety-seven percent right. Move every digit one pixel to the right, and it falls to forty-six.",
    "Same digits, same network. One pixel to the left: thirty-nine percent. This zero becomes a two, a four, a one.",
    "Each of its neurons is wired to particular pixels. Move the ink by one pixel, and it lands on different weights.",
    "Image networks do something else. They use small filters, three by three pixels, and slide each one over every position in the image.",
    "Then they keep each filter's strongest response, wherever it was. Move the digit, and the responses move with it. The strongest one doesn't change at all.",
    "Thirty-two of these filters, trained on the same digits: ninety-six percent, and still ninety-six percent after a one-pixel move. With about a quarter of the numbers.",
    "That's convolution: a pattern means the same thing, wherever it shows up.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the shifts, 2 why, 3 a filter slides, 4 the strongest response, 5 the result, 6 close
SEG_MIN = [6.0, 8.0, 8.0, 9.0, 10.0, 10.5, 5.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_MOVE = word_t(0, "to the right")
T_LEFT = word_t(1, "One pixel")
T_TWO, T_FOUR, T_ONE = word_t(1, "a two"), word_t(1, "a four"), word_t(1, "a one")
T_SLIDE = (word_t(3, "slide") - 0.3, SEG[4] - 0.4)
T_MOVE4 = word_t(4, "Move the digit")
T_RES = word_t(5, "and still")

DIGIT = (70, 470, 340)
BAR_M = (450, 488, 250)
BAR_C = (760, 488, 250)
ROW2 = 900
STRIP_ = (250, 1222, 760)
CHART = (170, 948, 800, 190)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("shift", "why", "conv", "max", "result", "close")[k - 1]
    return "close"


def shift_at(t):
    """(the shift shown, the shift it is sliding from, how far along the slide is 0..1)."""
    md = mode(t)
    keys = []
    if md == "hook":
        keys = [(0.0, 0)] if t >= T_END - 0.6 or t < T_MOVE + 0.4 else [(0.0, 0), (T_MOVE + 0.4, 1)]
    elif md == "shift":
        keys = [(SEG[1], 1), (T_LEFT + 0.3, -1), (T_TWO - 0.3, 1), (T_FOUR - 0.3, -1), (T_ONE - 0.3, -2), (SEG[2] - 0.6, 0)]
    elif md == "why":
        keys = [(SEG[2], 0)] + [(SEG[2] + 1.6 + 1.8 * k, 1 if k % 2 == 0 else 0) for k in range(int((SEG[3] - SEG[2] - 1.6) / 1.8))]
    elif md == "conv":
        keys = [(SEG[3], 0)]
    elif md == "max":
        keys = [(SEG[4], 0), (T_MOVE4 + 0.3, 1)]
    elif md == "result":
        keys = [(SEG[5], 1), (SEG[5] + 1.2, 0), (T_RES + 0.2, 1), (T_RES + 1.6, -1), (T_RES + 3.0, 2), (T_RES + 4.4, -2), (SEG[6] - 0.7, 0)]
    else:
        keys = [(SEG[6], 0)]
    cur, prev, t_at = keys[0][1], keys[0][1], keys[0][0]
    for tk, s in keys:
        if t >= tk:
            prev, cur, t_at = cur, s, tk
    return cur, prev, min(1.0, (t - t_at) / 0.28)


def cnn_on(t):
    md = mode(t)
    if md == "hook":
        return 1 - smooth(seg(t, T_END - 0.6, T_END - 0.1)) if t >= T_END - 0.6 else 0.0
    if md in ("shift", "why"):
        return 0.0
    if md == "conv":
        return smooth(seg(t, SEG[3] + 0.3, SEG[3] + 1.1))
    return 1.0


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_MOVE + 1.2) * FPS), int(0.5 * (T_SLIDE[0] + T_SLIDE[1]) * FPS), int((T_MOVE4 + 1.0) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(acc=ACC, digit=DIG, index=R["index"], filter=KF, max_before=R["max0"], max_after=R["max1"], neuron=NEURON, h_before=H_BEFORE,
                          h_after=H_AFTER, n_mlp=N_MLP, n_cnn=N_CNN, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("Move the digit one pixel.", "black", 50, INK, 300, 1), (f"{pc(ACC['mlp'][0])}% → {pc(ACC['mlp'][1])}%.", "dbold", 44, VERM, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Same digits. Same network.", "black", 50, INK, 300, 1), (f"One pixel left: {pc(ACC['mlp'][-1])}%.", "dbold", 42, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Each neuron is wired to pixels.", "black", 46, INK, 300, 1), ("Move the ink: different weights.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("One 3 × 3 filter, slid everywhere.", "black", 44, MAGENTA, 300, 1), ("The same 9 weights at every position.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Keep the strongest response.", "black", 48, INK, 300, 1), ("Wherever it was.", "dbold", 42, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [(f"Convolution: {pc(ACC['cnn'][0])}%.", "black", 50, MAGENTA, 300, 1), (f"One pixel right: {pc(ACC['cnn'][1])}%. Left: {pc(ACC['cnn'][-1])}%.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("A pattern is a pattern,", "black", 50, INK, 300, 1), ("wherever it shows up.", "dbold", 44, MAGENTA, 390, 0.95)], 10, 8, "pop"),
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

    # ------------------------------------------------------------ pieces

    def grid_bg(self, ctx, x0, y0, size, a):
        c = size / 8
        ctx.set_source_rgba(*rgb(PANEL_BASE), a)
        for r in range(8):
            for q in range(8):
                ctx.rectangle(x0 + q * c + 1, y0 + r * c + 1, c - 2, c - 2)
        ctx.fill()

    def pixels(self, ctx, x0, y0, size, img64, a, offset=0.0, frame=True):
        """The digit's 64 pixels; offset (in pixels, fractional while sliding) moves the ink inside the frame."""
        c = size / 8
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        self.grid_bg(ctx, x0, y0, size, a)
        ctx.save()
        ctx.rectangle(x0, y0, size, size)
        ctx.clip()
        for r in range(8):
            for q in range(8):
                v = float(np.clip(img64[r * 8 + q], 0, 1))
                if v <= 0.002:
                    continue
                ctx.set_source_rgba(*(base * (1 - v) + ink * v), a)
                ctx.rectangle(x0 + (q + offset) * c + 1, y0 + r * c + 1, c - 2, c - 2)
                ctx.fill()
        ctx.restore()
        if frame:
            ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
            ctx.set_line_width(1.2)
            ctx.rectangle(x0, y0, size, size)
            ctx.stroke()

    def weights(self, ctx, x0, y0, size, w64, a, ink64=None):
        """A neuron's 64 weights as a picture (red +, blue -); the ink of a digit drawn on top as white squares."""
        c = size / 8
        base = np.array(rgb(PANEL_BASE))
        m = np.abs(w64).max() + 1e-12
        for p in range(64):
            g = w64[p] / m
            col = np.array(rgb(VERM if g > 0 else BLUE))
            k = abs(g) ** 0.8
            ctx.set_source_rgba(*(base * (1 - k) + col * k), a)
            ctx.rectangle(x0 + (p % 8) * c + 1, y0 + (p // 8) * c + 1, c - 2, c - 2)
            ctx.fill()
            if ink64 is not None and ink64[p] > 0.05:
                s = c * 0.5 * float(ink64[p])
                ctx.set_source_rgba(*rgb(INK), 0.95 * a)
                ctx.rectangle(x0 + (p % 8) * c + c / 2 - s / 2, y0 + (p // 8) * c + c / 2 - s / 2, s, s)
                ctx.fill()
        ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
        ctx.set_line_width(1.2)
        ctx.rectangle(x0, y0, size, size)
        ctx.stroke()

    def fmap(self, ctx, x0, y0, size, m64, a, upto=64, ring=None, vmax=None):
        """A filter's response at each of the 64 positions (gold: stronger), filled up to `upto` positions."""
        c = size / 8
        base = np.array(rgb(PANEL_BASE))
        gold = np.array(rgb(GOLD))
        vm = vmax or (max(m64) + 1e-12)
        self.grid_bg(ctx, x0, y0, size, a)
        for p in range(min(64, upto)):
            k = min(1.0, max(0.0, m64[p] / vm)) ** 0.8
            ctx.set_source_rgba(*(base * (1 - k) + gold * k), a)
            ctx.rectangle(x0 + (p % 8) * c + 1, y0 + (p // 8) * c + 1, c - 2, c - 2)
            ctx.fill()
        if ring is not None:
            ctx.set_source_rgba(*rgb(INK), a)
            ctx.set_line_width(3.0)
            ctx.rectangle(x0 + (ring % 8) * c - 1, y0 + (ring // 8) * c - 1, c + 2, c + 2)
            ctx.stroke()
        ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
        ctx.set_line_width(1.2)
        ctx.rectangle(x0, y0, size, size)
        ctx.stroke()

    def filt(self, ctx, x0, y0, size, a):
        c = size / 3
        base = np.array(rgb(PANEL_BASE))
        m = np.abs(FILT).max()
        for r in range(3):
            for q in range(3):
                g = FILT[r, q] / m
                col = np.array(rgb(VERM if g > 0 else BLUE))
                k = abs(g) ** 0.8
                ctx.set_source_rgba(*(base * (1 - k) + col * k), a)
                ctx.rectangle(x0 + q * c + 1.5, y0 + r * c + 1.5, c - 3, c - 3)
                ctx.fill()
        ctx.set_source_rgba(*rgb(MAGENTA), 0.9 * a)
        ctx.set_line_width(2.5)
        ctx.rectangle(x0, y0, size, size)
        ctx.stroke()

    def bars(self, ctx, x0, y0, w, probs, a, truth=DIG, title="", sub=""):
        top = int(np.argmax(probs))
        for d in range(10):
            yy = y0 + d * 30
            font(ctx, "dbold", 21)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(x0, yy + 21)
            ctx.show_text(str(d))
            ctx.set_source_rgba(0.12, 0.15, 0.25, a)
            S3.pill(ctx, x0 + 24, yy + 7, w - 24, 16, 8)
            ctx.fill()
            col = (GREEN if d == truth else VERM) if d == top else INK3
            ctx.set_source_rgba(*rgb(col), a)
            S3.pill(ctx, x0 + 24, yy + 7, max(4.0, (w - 24) * float(probs[d])), 16, 8)
            ctx.fill()
        s = f"reads {top}  ·  {pp(probs[top])}"
        font(ctx, "dbold", 28)
        ctx.set_source_rgba(*rgb(GREEN if top == truth else VERM), a)
        ctx.move_to(x0 + w / 2 - ctx.text_extents(s).x_advance / 2, y0 + 340)
        ctx.show_text(s)
        self.label(ctx, title, x0, y0 - 34, a, INK2)
        self.label(ctx, sub, x0, y0 - 12, a)

    def label(self, ctx, s, x, y, a, col=INK3):
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        show_tracked(ctx, s, x, y, 0.1 * 16)

    def chart(self, ctx, t, a, dx_now):
        x0, y0, w, h = CHART
        lo = 0.0

        def gx(dx):
            return x0 + (dx + 2) / 4 * w

        def gy(v):
            return y0 + h - (v - lo) / (1 - lo) * h
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        ctx.set_source_rgba(1, 1, 1, 0.07 * a)
        for v in (0.25, 0.5, 0.75, 1.0):
            ctx.move_to(x0, gy(v))
            ctx.line_to(x0 + w, gy(v))
        ctx.stroke()
        # where we are
        ctx.set_source_rgba(1, 1, 1, 0.12 * a)
        ctx.rectangle(gx(dx_now) - 22, y0 - 4, 44, h + 4)
        ctx.fill()
        font(ctx, "dbold", 18)
        for dx in range(-2, 3):
            s = {-2: "2 left", -1: "1 left", 0: "none", 1: "1 right", 2: "2 right"}[dx]
            ctx.set_source_rgba(*rgb(INK if dx == dx_now else INK2), a)
            ctx.move_to(gx(dx) - ctx.text_extents(s).x_advance / 2, y0 + h + 28)
            ctx.show_text(s)
        ctx.set_source_rgba(*rgb(INK3), a)
        for v, s in ((0.0, "0"), (0.5, "50%"), (1.0, "100%")):
            ctx.move_to(x0 - 12 - ctx.text_extents(s).x_advance, gy(v) + 6)
            ctx.show_text(s)
        con = cnn_on(t)
        for key, col, al in (("mlp", VERM, 1.0), ("cnn", MAGENTA, con)):
            if al <= 0.01:
                continue
            ctx.set_source_rgba(*rgb(col), a * al)
            ctx.set_line_width(3.5)
            for i, dx in enumerate(range(-2, 3)):
                (ctx.move_to if i == 0 else ctx.line_to)(gx(dx), gy(ACC[key][dx]))
            ctx.stroke()
            for dx in range(-2, 3):
                ctx.arc(gx(dx), gy(ACC[key][dx]), 6 if dx == dx_now else 4.5, 0, TAU)
                ctx.fill()
        font(ctx, "dbold", 20)
        for key, col, al, yy in (("mlp", VERM, 1.0, gy(0.68)), ("cnn", MAGENTA, con, gy(0.80))):
            if al <= 0.01:
                continue
            s = f"{'handwriting network' if key == 'mlp' else 'convolutional'}: {pc(ACC[key][dx_now])}%"
            ctx.set_source_rgba(*rgb(col), a * al)
            ctx.move_to(gx(-1.95), yy)
            ctx.show_text(s)
        self.label(ctx, f"RIGHT, OF THE {R['n_test']} UNSEEN DIGITS, BY PIXELS MOVED", x0 - 100, y0 - 26, a)

    def strip(self, ctx, t, a):
        x0, y0, w = STRIP_
        tile, n = 58, 10
        gap = (w - n * tile) / (n - 1)
        con = cnn_on(t)
        for d in range(10):
            o = STRIP[d]
            xx = x0 + d * (tile + gap)
            self.pixels(ctx, xx, y0, tile, shifted(XTE[SHOW[d]], 1), a, frame=False)
            font(ctx, "dbold", 24)
            for val, yy, al in ((o["mlp_r"], y0 + tile + 30, 1.0), (o["cnn_r"], y0 + tile + 60, con)):
                if al <= 0.01:
                    continue
                s = str(val)
                ctx.set_source_rgba(*rgb(GREEN if val == d else VERM), a * al)
                ctx.move_to(xx + tile / 2 - ctx.text_extents(s).x_advance / 2, yy)
                ctx.show_text(s)
        self.label(ctx, "TEN UNSEEN DIGITS, MOVED ONE PIXEL RIGHT: WHAT EACH NETWORK READS", 70, y0 - 14, a)
        font(ctx, "bold", 15)
        ctx.set_source_rgba(*rgb(INK2), 0.95 * a)
        show_tracked(ctx, "HANDWRITING", 70, y0 + tile + 28, 0.1 * 15)
        ctx.set_source_rgba(*rgb(INK2), 0.95 * a * con)
        show_tracked(ctx, "CONVOLUTIONAL", 70, y0 + tile + 58, 0.1 * 15)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        dx, prev, fr = shift_at(t)
        off = prev + (dx - prev) * smooth(fr)
        surf, ctx = layer()
        x, y, size = DIGIT
        self.pixels(ctx, x, y, size, X0, a, offset=off)
        lab = "AN UNSEEN 0" + ("" if dx == 0 else f", MOVED {abs(dx)} PIXEL{'S' if abs(dx) > 1 else ''} {'RIGHT' if dx > 0 else 'LEFT'}")
        self.label(ctx, lab, x, y - 14, a)
        con = cnn_on(t)
        pm = PS[dx]["p_mlp"] if fr >= 1 else PS[prev]["p_mlp"]
        pcn = PS[dx]["p_cnn"] if fr >= 1 else PS[prev]["p_cnn"]
        self.bars(ctx, *BAR_M, pm, a, title="HANDWRITING NETWORK", sub=f"{N_MLP:,} NUMBERS")
        if con > 0.01:
            self.bars(ctx, *BAR_C, pcn, a * con, title="CONVOLUTIONAL", sub=f"{N_CNN} NUMBERS")
        # row 2: what each part is doing
        if md == "why":
            ws = _MLP[0][:, NEURON]
            sh = dx if fr >= 1 else prev
            ink = shifted(X0, sh)
            self.weights(ctx, 70, ROW2 + 30, 250, ws, a, ink64=ink)
            hv = float(np.maximum(ink @ ws + _MLP[1][NEURON], 0))
            self.label(ctx, f"NEURON {NEURON + 1} OF 32: ITS 64 WEIGHTS, AND THE INK ON THEM", 70, ROW2 + 14, a)
            font(ctx, "dbold", 28)
            ctx.set_source_rgba(*rgb(INK2), a)
            for j, s in enumerate(("red: ink here excites it", "blue: ink here silences it")):
                ctx.move_to(370, ROW2 + 80 + 40 * j)
                ctx.show_text(s)
            font(ctx, "black", 46)
            ctx.set_source_rgba(*rgb(GOLD if sh == 0 else VERM), a)
            ctx.move_to(370, ROW2 + 200)
            ctx.show_text(f"output: {hv:.2f}")
            font(ctx, "dbold", 24)
            ctx.set_source_rgba(*rgb(INK3), a)
            ctx.move_to(370, ROW2 + 244)
            ctx.show_text(f"before the move: {H_BEFORE:.2f}   moved right: {H_AFTER:.2f}")
        elif md == "conv":
            # the filter slides over the digit, filling its response map
            n_pos = int(64 * seg(t, *T_SLIDE)) if t >= T_SLIDE[0] else 0
            p = min(63, n_pos)
            c = size / 8
            if t >= T_SLIDE[0] and n_pos < 64:
                r_, q_ = p // 8, p % 8
                ctx.set_source_rgba(*rgb(MAGENTA), a)
                ctx.set_line_width(4)
                ctx.save()
                ctx.rectangle(x - 2, y - 2, size + 4, size + 4)
                ctx.clip()
                ctx.rectangle(x + (q_ - 1) * c, y + (r_ - 1) * c, 3 * c, 3 * c)
                ctx.stroke()
                ctx.restore()
            self.filt(ctx, 70, ROW2 + 40, 180, a)
            self.label(ctx, "ONE FILTER: 3 × 3 WEIGHTS", 70, ROW2 + 24, a)
            self.fmap(ctx, 330, ROW2 + 40, 240, np.array(PS[0]["fmaps"][KF]).ravel(), a, upto=n_pos)
            self.label(ctx, "ITS RESPONSE AT EVERY POSITION", 330, ROW2 + 24, a)
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(INK2), a)
            for j, s in enumerate(("the same 9 weights,", "at all 64 positions:", f"position {min(n_pos, 64)} of 64")):
                ctx.move_to(610, ROW2 + 90 + 40 * j)
                ctx.show_text(s)
        elif md == "max":
            sh = dx if fr >= 1 else prev
            m0 = np.array(PS[0]["fmaps"][KF]).ravel()
            m1 = np.array(PS[1]["fmaps"][KF]).ravel()
            vm = max(m0.max(), m1.max())
            self.fmap(ctx, 70, ROW2 + 40, 220, m0, a, ring=int(np.argmax(m0)), vmax=vm)
            self.label(ctx, "RESPONSE: BEFORE", 70, ROW2 + 24, a)
            al1 = a if sh == 1 else 0.25 * a
            self.fmap(ctx, 330, ROW2 + 40, 220, m1, al1, ring=int(np.argmax(m1)), vmax=vm)
            self.label(ctx, "MOVED ONE PIXEL RIGHT", 330, ROW2 + 24, al1)
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.move_to(600, ROW2 + 90)
            ctx.show_text(f"strongest: {m0.max():.2f}")
            ctx.set_source_rgba(*rgb(GOLD), al1)
            ctx.move_to(600, ROW2 + 130)
            ctx.show_text(f"strongest: {m1.max():.2f}")
            ctx.set_source_rgba(*rgb(INK2), al1)
            ctx.move_to(600, ROW2 + 190)
            ctx.show_text("all 32 filters' strongest")
            ctx.move_to(600, ROW2 + 226)
            ctx.show_text(f"responses: unchanged")
        else:
            self.chart(ctx, t, a, dx if fr >= 0.5 else prev)
        self.strip(ctx, t, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close"):
            paste(img, rf"$\mathrm{{handwriting\ network:}}\ \ {pc(ACC['mlp'][0])}\% \;\rightarrow\; {pc(ACC['mlp'][1])}\%\ \ \mathrm{{(one\ pixel\ right)}}$", 25, 540, y + 96, VERM, a, align="c")
            paste(img, rf"$\mathrm{{convolutional:}}\ \ {pc(ACC['cnn'][0])}\% \;\rightarrow\; {pc(ACC['cnn'][1])}\%$", 25, 540, y + 156, MAGENTA, a * (1.0 if md == "close" else cnn_on(t)), align="c")
            paste(img, r"$\mathrm{the\ same\ 540\ unseen\ digits\ (UCI,\ 8 \times 8\ pixels),\ moved\ sideways}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "shift":
            s = r"\qquad ".join(rf"{lab}:\ {pc(ACC['mlp'][d])}\%" for d, lab in ((-1, r"\mathrm{1\ left}"), (0, r"\mathrm{not\ moved}"), (1, r"\mathrm{1\ right}")))
            s2 = r"\qquad ".join(rf"{lab}:\ {pc(ACC['mlp'][d])}\%" for d, lab in ((-2, r"\mathrm{2\ left}"), (2, r"\mathrm{2\ right}")))
            paste(img, "$" + s + "$", 26, 540, y + 88, VERM, a, align="c")
            paste(img, "$" + s2 + "$", 24, 540, y + 136, INK2, a, align="c")
            paste(img, r"$\mathrm{the\ columns\ that\ slide\ in\ are\ blank;\ the\ edge\ columns\ hold\ 0.3\%\ of\ the\ ink}$", 20, 540, y + 182, INK3, a, align="c")
            paste(img, r"$\mathrm{it\ never\ saw\ a\ moved\ digit\ in\ training,\ and\ neither\ will\ the\ next\ network}$", 20, 540, y + h - 18, INK3, a, align="c")
        elif md == "why":
            paste(img, r"$h_j = \mathrm{ReLU}\!\left(\sum_{p=1}^{64} x_p\, W_{pj} + b_j\right)$", 34, 540, y + 110, INK, a, align="c")
            paste(img, r"$\mathrm{weight}\ W_{pj}\ \mathrm{belongs\ to\ pixel}\ p:\ \mathrm{moved\ ink\ meets\ other\ weights}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "conv":
            paste(img, r"$r(i, j) = \mathrm{ReLU}\!\left(\sum_{3 \times 3} x \cdot w + b\right)$", 32, 540, y + 110, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{at\ every\ position}\ (i, j):\ \mathrm{one\ set\ of\ 9\ weights,\ shared\ by\ all\ 64}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "max":
            paste(img, r"$\mathrm{strongest} = \max_{i, j}\ r(i, j)$", 36, 540, y + 104, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{this\ filter:}}\ {R['max0']:.2f}\ \mathrm{{before,}}\ {R['max1']:.2f}\ \mathrm{{after;\ all\ 32:\ the\ same\ to\ the\ last\ digit}}$", 21, 540, y + 168, INK2, a, align="c")
            paste(img, r"$\mathrm{then\ 32\ strongest\ responses} \rightarrow 10\ \mathrm{probabilities}$", 21, 540, y + h - 18, INK3, a, align="c")
        else:
            paste(img, r"$32 \times 9 + 32 + 32 \times 10 + 10 = 650\ \mathrm{numbers}\qquad (\mathrm{vs}\ 2{,}410)$", 26, 540, y + 100, MAGENTA, a, align="c")
            paste(img, rf"$\mathrm{{moved\ 1\ right:}}\ {pc(ACC['cnn'][1])}\%,\ \ \mathrm{{1\ left:}}\ {pc(ACC['cnn'][-1])}\%,\ \ \mathrm{{2\ right:}}\ {pc(ACC['cnn'][2])}\%,\ \ \mathrm{{2\ left:}}\ {pc(ACC['cnn'][-2])}\%$", 20, 540, y + 164, INK2, a, align="c")
            paste(img, r"$\mathrm{trained\ like\ the\ handwriting\ network:\ the\ same\ 1{,}257\ digits,\ never\ moved}$", 20, 540, y + h - 18, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        self.draw_main(img, t, a)
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "REAL HANDWRITING   ·   540 UNSEEN DIGITS   ·   ALL COMPUTED"
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
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]))
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
