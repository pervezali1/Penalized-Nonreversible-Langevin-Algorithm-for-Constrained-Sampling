"""The network that is sure about static. Dark theme of reels 31-52, maths in Computer Modern (stage3d.tex), narrated by
Kokoro (voicekit). The handwriting network of reel 49 (96.7% on 540 unseen digits) shown 1,000 images of random
static: for 817 of them it is more than 90% sure of a digit, mostly a 4 or a 1; a letter A is "a 4" at 99.96%.
Softmax has no "none of these". The usual fix, an 11th answer trained on 600 static images, catches 999 of 1,000 new
static images and still reads 96.5% of digits, but only 72.8% of a different noise it never saw (staticsim.py).
frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "static_results.json")):
    import staticsim
    staticsim.main()
with open(os.path.join(HERE, "static_results.json")) as fh:
    R = json.load(fh)
_D = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))
_NPZ = os.path.join(ROOT, "digits", "dg_results.npz")
if not os.path.exists(_NPZ):
    sys.path.insert(0, os.path.join(ROOT, "digits"))
    import dgsim
    dgsim.main()
_DZ = np.load(_NPZ)
X3 = _DZ["Xte"][_D["show"][3]]
P3 = _DZ["probs_show"][-1][3]
ST = R["static"]
FX = R["fix"]
N = R["n_static"]
N90 = int(round(ST["over90"] * N))
N99 = int(round(ST["over99"] * N))
BIG, BIG_P = np.array(R["big"]), np.array(R["big_p"])
A_X, A_P = np.array(R["letter_a_x"]), np.array(R["letter_a"])
SHOWN, SHOWN_P, SHOWN_Q = np.array(R["shown"]), np.array(R["shown_p"]), np.array(FX["shown_p"])
BIG_Q, A_Q = np.array(FX["big_p"]), np.array(FX["letter_a"])
CAUGHT = int(round(FX["static"]["called_not_digit"] * N))
DOTS = FX["dots"]["called_not_digit"]


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


def pp(v):
    """A probability as a percentage that never rounds up to 100%."""
    if v < 0.999:
        return f"{100 * v:.1f}%"
    if v < 0.9999:
        return f"{100 * v:.2f}%"
    return ">99.99%"


assert (int(np.argmax(BIG_P)), pc(BIG_P.max()), N90, int(np.argmax(A_P)), pc(A_P.max(), 2), CAUGHT, pc(FX["acc"]), pc(DOTS, 0)) == (4, "99.7", 817, 4, "99.96", 999, "96.5", "73"), "script numbers changed"
SCRIPT = [
    "I showed this network pure static. It's ninety-nine point seven percent sure it's a four.",
    "It's the network from the handwriting reel: ninety-six point seven percent right, on digits it has never seen.",
    "So I made a thousand images of random static. For eight hundred and seventeen of them, it was more than ninety percent sure of a digit. Mostly a four, or a one.",
    "It isn't broken. Its answer is ten probabilities, and they always add up to one. There's no option for none of these. It can only ever say which digit.",
    "And it's not just static. Draw it a letter A: a four, ninety-nine point nine six percent sure.",
    "The usual fix: give it an eleventh answer, not a digit, and train it on six hundred static images too. Now new static gets not a digit nine hundred and ninety-nine times out of a thousand, and it still reads ninety-six point five percent of real digits.",
    "But show it a different kind of noise, scattered dots, and it catches only seventy-three percent. A network only learns to reject what it was shown.",
    "Ninety-nine percent sure means sure among the answers it knows. Not sure that it's right.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 1,000 static images, 3 softmax has no "none", 4 the letter A, 5 the fix, 6 other noise, 7 close
SEG_MIN = [5.0, 7.0, 10.0, 9.0, 6.5, 13.0, 9.0, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[8]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_FILL = (SEG[2] + 0.6, word_t(2, "For eight"))           # the histogram of the 1,000 fills
T_NOW = word_t(5, "Now new")

BIG_T = (70, 480, 330)
BARS = (470, 486, 540)
STRIP = (70, 1214, 940)
HIST = (150, 930, 820, 190)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 8):
        if t < SEG[k + 1]:
            return ("net", "static", "why", "letter", "fix", "dots", "close")[k - 1]
    return "close"


def qa_frames():
    fr = []
    for k in range(8):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_NOW + 1.5) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(big=dict(read=int(np.argmax(BIG_P)), p=float(BIG_P.max())), over90=N90, over99=N99, classes=ST["classes"], letter_a=float(A_P.max()),
                          caught=CAUGHT, fix_acc=FX["acc"], dots=DOTS, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        self.hist_s = np.array(R["hist_static"], float)
        self.hist_r = np.array(R["hist_real"], float)

    def _cues(self):
        S_ = SEG
        hook = [("Pure static.", "black", 54, INK, 300, 1), (f"\"A {int(np.argmax(BIG_P))}\", {pc(BIG_P.max())}% sure.", "dbold", 46, VERM, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("The handwriting network.", "black", 50, INK, 300, 1), (f"{pc(R['acc'])}% right on unseen digits.", "dbold", 42, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [(f"{N:,} images of static.", "black", 52, INK, 300, 1), (f"{N90} read with over 90% certainty.", "dbold", 40, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Ten answers, adding up to 100%.", "black", 44, INK, 300, 1), ("No \"none of these\".", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("A letter A.", "black", 54, INK, 300, 1), (f"\"A 4\", {pc(A_P.max(), 2)}% sure.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("An 11th answer: \"not a digit\".", "black", 46, MAGENTA, 300, 1), (f"Static caught: {CAUGHT} of {N:,}.", "dbold", 42, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [("A different noise.", "black", 52, INK, 300, 1), (f"Caught: only {pc(DOTS, 0)}%.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[7], T_END - 0.3, [("99% sure of an answer", "black", 50, INK, 300, 1), ("is not 99% sure it's right.", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
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

    def pixels(self, ctx, x0, y0, size, img64, a, frame=True):
        c = size / 8
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        for r in range(8):
            for q in range(8):
                v = float(np.clip(img64[r * 8 + q], 0, 1))
                ctx.set_source_rgba(*(base * (1 - v) + ink * v), a)
                ctx.rectangle(x0 + q * c + 1, y0 + r * c + 1, c - 2, c - 2)
                ctx.fill()
        if frame:
            ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
            ctx.set_line_width(1.2)
            ctx.rectangle(x0, y0, size, size)
            ctx.stroke()

    def label(self, ctx, s, x, y, a, col=INK3):
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        show_tracked(ctx, s, x, y, 0.1 * 16)

    def bars(self, ctx, probs, a, truth=None, eleven=False):
        """The ten (or eleven) probabilities, as horizontal bars, with the reading underneath."""
        x0, y0, w = BARS
        n = 11 if eleven else 10
        row = 30 if eleven else 32
        top = int(np.argmax(probs))
        for d in range(n):
            yy = y0 + d * row
            font(ctx, "dbold", 21)
            ctx.set_source_rgba(*rgb(MAGENTA if d == 10 else INK2), a)
            s = "none" if d == 10 else str(d)
            ctx.move_to(x0 + 50 - ctx.text_extents(s).x_advance, yy + 21)
            ctx.show_text(s)
            ctx.set_source_rgba(0.12, 0.15, 0.25, a)
            S3.pill(ctx, x0 + 62, yy + 7, w - 62 - 110, 16, 8)
            ctx.fill()
            col = (GREEN if (truth is not None and d == truth) or (truth is None and d == 10) else VERM) if d == top else INK3
            ctx.set_source_rgba(*rgb(col), a)
            S3.pill(ctx, x0 + 62, yy + 7, max(4.0, (w - 62 - 110) * float(probs[d])), 16, 8)
            ctx.fill()
            font(ctx, "dbold", 18)
            v = pp(float(probs[d])) if probs[d] >= 0.0005 else "0%"
            ctx.set_source_rgba(*rgb(col if d == top else INK3), a)
            ctx.move_to(x0 + w - ctx.text_extents(v).x_advance, yy + 20)
            ctx.show_text(v)
        # the sum
        font(ctx, "dbold", 20)
        ctx.set_source_rgba(*rgb(GOLD), a)
        s = f"total: {pc(float(np.sum(probs)), 0)}%"
        ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, y0 + n * row + 60)
        ctx.show_text(s)
        reading = "not a digit" if top == 10 else f"\"a {top}\""
        s = f"reads {reading}  ·  {pp(float(probs[top]))}"
        font(ctx, "dbold", 30)
        ok = (truth is not None and top == truth) or (truth is None and top == 10)
        ctx.set_source_rgba(*rgb(GREEN if ok else VERM), a)
        ctx.move_to(x0 + 62, y0 + n * row + 26)
        ctx.show_text(s)

    def strip(self, ctx, t, a, fixed):
        x0, y0, w = STRIP
        tile, n = 70, 10
        gap = (w - n * tile) / (n - 1)
        for k in range(10):
            xx = x0 + k * (tile + gap)
            self.pixels(ctx, xx, y0, tile, SHOWN[k], a, frame=False)
            p = SHOWN_Q[k] if fixed else SHOWN_P[k]
            top = int(np.argmax(p))
            s = "none" if top == 10 else str(top)
            font(ctx, "dbold", 24 if top == 10 else 26)
            ctx.set_source_rgba(*rgb(GREEN if top == 10 else VERM), a)
            ctx.move_to(xx + tile / 2 - ctx.text_extents(s).x_advance / 2, y0 + tile + 30)
            ctx.show_text(s)
            font(ctx, "bold", 15)
            s2 = pp(float(p[top]))
            ctx.set_source_rgba(*rgb(INK3), a)
            ctx.move_to(xx + tile / 2 - ctx.text_extents(s2).x_advance / 2, y0 + tile + 52)
            ctx.show_text(s2)
        self.label(ctx, "TEN IMAGES OF STATIC: WHAT IT READS" + (", WITH THE 11TH ANSWER" if fixed else ""), x0, y0 - 14, a)

    def hist(self, ctx, t, a):
        """How sure it is: the 1,000 static images and the 540 real digits, by the probability of its answer."""
        x0, y0, w, h = HIST
        fill = smooth(seg(t, *T_FILL)) if mode(t) == "static" else 1.0
        hs = self.hist_s * fill
        hs_n, hr_n = hs / N, self.hist_r / self.hist_r.sum()
        mxv = max(hs_n.max(), hr_n.max(), 1e-9)
        bw = w / 10
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        for b in range(10):
            for k, (vals, col) in enumerate(((hr_n, GREEN), (hs_n, VERM))):
                hh = vals[b] / mxv * h
                ctx.set_source_rgba(*rgb(col), 0.9 * a)
                ctx.rectangle(x0 + b * bw + 6 + k * (bw - 12) / 2, y0 + h - hh, (bw - 12) / 2 - 2, hh)
                ctx.fill()
        font(ctx, "dbold", 17)
        ctx.set_source_rgba(*rgb(INK2), a)
        for b, s in ((0, "0%"), (5, "50%"), (10, "100%")):
            ctx.move_to(x0 + b * bw - ctx.text_extents(s).x_advance / 2, y0 + h + 24)
            ctx.show_text(s)
        font(ctx, "dbold", 22)
        ctx.set_source_rgba(*rgb(GREEN), a)
        ctx.move_to(x0 + 10, y0 + 24)
        ctx.show_text(f"real digits ({int(self.hist_r.sum())})")
        ctx.set_source_rgba(*rgb(VERM), a)
        ctx.move_to(x0 + 10, y0 + 52)
        ctx.show_text(f"static ({int(round(fill * N)):,})")
        self.label(ctx, "SHARE OF IMAGES, BY HOW SURE IT IS OF ITS ANSWER", x0 - 80, y0 - 20, a)

    def counters(self, ctx, t, a, show_dots):
        x0, y0 = 70, 940
        rows = [(f"{CAUGHT} of {N:,}", "new static, called \"not a digit\"", GREEN),
                (f"{pc(FX['acc'])}%", "of the 540 real digits still read right", GREEN)]
        if show_dots:
            rows.append((f"{pc(DOTS)}%", "of a different noise (dots) caught", VERM))
        for j, (big, small, col) in enumerate(rows):
            font(ctx, "black", 52)
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(x0, y0 + 60 + 86 * j)
            ctx.show_text(big)
            bw = ctx.text_extents(big).x_advance
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(x0 + bw + 22, y0 + 52 + 86 * j)
            ctx.show_text(small)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        surf, ctx = layer()
        x, y, size = BIG_T
        if md == "net":
            self.pixels(ctx, x, y, size, X3, a)
            self.label(ctx, "AN UNSEEN 3", x, y - 14, a)
            self.bars(ctx, P3, a, truth=3)
        elif md == "letter":
            self.pixels(ctx, x, y, size, A_X, a)
            self.label(ctx, "A LETTER A, IN 8 × 8 PIXELS", x, y - 14, a)
            self.bars(ctx, A_P, a, truth=-1)
        elif md == "fix":
            self.pixels(ctx, x, y, size, BIG, a)
            self.label(ctx, "THE SAME STATIC", x, y - 14, a)
            self.bars(ctx, BIG_Q, a, eleven=True)
        elif md == "dots":
            self.pixels(ctx, x, y, size, np.array(R["dot_example"]["x"]), a)
            self.label(ctx, "DOTS: ONE OF THE 27% IT MISSES", x, y - 14, a)
            # the 11-class network on this example
            self.bars(ctx, np.array(R["dot_example"]["p11"]), a, eleven=True)
        else:
            self.pixels(ctx, x, y, size, BIG, a)
            self.label(ctx, "RANDOM STATIC: EVERY PIXEL A RANDOM SHADE", x, y - 14, a)
            self.bars(ctx, BIG_P, a, truth=-1)
        if md in ("fix", "dots"):
            self.counters(ctx, t, a, md == "dots")
        else:
            self.hist(ctx, t, a)
        self.strip(ctx, t, a, fixed=md in ("fix", "dots"))
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close", "static"):
            paste(img, rf"${N:,}\ \mathrm{{static\ images:}}\ {N90}\ \mathrm{{read\ with}}\ p > 0.9,\ {N99}\ \mathrm{{with}}\ p > 0.99$".replace(",", "{,}", 1), 24, 540, y + 96, VERM, a, align="c")
            c = ST["classes"]
            paste(img, rf"$\mathrm{{read\ as}}\ 4:\ {c[4]},\ \ 1:\ {c[1]},\ \ 2:\ {c[2]},\ \ 7:\ {c[7]}$", 24, 540, y + 156, INK2, a, align="c")
            paste(img, r"$\mathrm{static:\ every\ pixel\ a\ random\ shade\ of\ ink,\ 0\ to\ 16}$", 20, 540, y + h - 18, INK3, a, align="c")
        elif md == "net":
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 32\ \mathrm{neurons} \;\rightarrow\; 10\ \mathrm{probabilities}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{a\ typical\ real\ digit\ (the\ median):}\ \mathrm{over}\ 99.99\%\ \mathrm{sure}$" if R['real']['median_max'] > 0.9999 else rf"$\mathrm{{a\ typical\ real\ digit:}}\ {pc(R['real']['median_max'], 2)}\%\ \mathrm{{sure}}$", 26, 540, y + 162, GREEN, a, align="c")
        elif md == "why":
            paste(img, r"$p_d = \frac{e^{z_d}}{e^{z_0} + e^{z_1} + \cdots + e^{z_9}}$", 40, 540, y + 116, GOLD, a, align="c")
            paste(img, r"$p_0 + p_1 + \cdots + p_9 = 1\ \mathrm{always:\ some\ digit\ has\ to\ win}$", 26, 540, y + h - 26, INK2, a, align="c")
        elif md == "letter":
            paste(img, rf"$p(4) = {A_P.max():.4f}\ \ \mathrm{{for\ a\ letter\ it\ has\ never\ seen}}$", 30, 540, y + 100, VERM, a, align="c")
            paste(img, r"$\mathrm{it\ was\ only\ ever\ shown\ digits:\ every\ input\ becomes\ some\ digit}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "fix":
            paste(img, r"$p_0 + \cdots + p_9 + p_{\mathrm{none}} = 1$", 36, 540, y + 100, MAGENTA, a, align="c")
            paste(img, rf"$\mathrm{{trained\ on\ the\ 1{{,}}257\ digits\ +\ {FX['n_train_static']}\ static\ images;\ tested\ on\ new\ ones}}$", 21, 540, y + h - 26, INK2, a, align="c")
        else:
            paste(img, rf'$\mathrm{{dots\ (each\ pixel\ inked\ with\ chance\ 0.3):\ {pc(DOTS)}\%\ caught}}$', 26, 540, y + 100, VERM, a, align="c")
            paste(img, rf'$\mathrm{{the\ letter\ A:\ caught,\ but\ only\ with}}\ p_{{\mathrm{{none}}}} = {A_Q.max():.2f}$', 24, 540, y + 160, INK2, a, align="c")
            paste(img, r"$\mathrm{it\ learns\ to\ reject\ the\ noise\ it\ was\ shown}$", 21, 540, y + h - 18, INK3, a, align="c")

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
        lab = "THE HANDWRITING REEL'S NETWORK   ·   1,000 IMAGES OF STATIC   ·   ALL COMPUTED"
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
