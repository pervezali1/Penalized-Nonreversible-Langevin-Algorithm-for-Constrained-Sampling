"""A network learns to read handwriting. Dark theme of reels 31-48, maths in Computer Modern (stage3d.tex), narrated by
Kokoro (voicekit). 1,797 real handwritten digits (8 x 8 pixels, digits.csv) and a network of 64 pixels -> 32
neurons -> 10 probabilities, trained on 1,257 of them and tested on the 540 it never sees (dgsim.py). On screen:
the digit as its 64 pixels, the 32 neurons as pictures of their weights (they change as it learns), the ten
probabilities, a strip of ten unseen digits with the network's answers, and the accuracy on the 540. Then
backpropagation carried all the way to the pixels (which pixels pushed the answer which way), and one of its 18
mistakes. frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "dg_results.npz")):       # the archive is not committed: rebuild it (seconds)
    import dgsim
    dgsim.main()
with open(os.path.join(HERE, "dg_results.json")) as fh:
    R = json.load(fh)
D = np.load(os.path.join(HERE, "dg_results.npz"))
XTE, YTE, PRED = D["Xte"], D["yte"], D["pred"]
PROBS = D["probs_show"]                          # (steps + 1, 10 shown digits, 10 probabilities)
W1S = D["W1s"]                                   # every 10 steps
BLAME = {int(i): b for i, b in zip(D["blame_idx"], D["blame"])}
SHOW = R["show"]                                 # one unseen example of each digit 0..9
ACC = np.array(R["acc_test"])
STEPS = R["steps"]
N_TEST, N_TRAIN, N_RIGHT = R["n_test"], R["n_train"], R["n_right"]
N_WRONG = N_TEST - N_RIGHT
FIRST90 = R["first_90"]
_rows = [ln.strip().split(",") for ln in open(os.path.join(HERE, "digits.csv")) if ln[0].isdigit()]
ALL = np.array([[float(v) for v in r[:64]] for r in _rows]) / 16.0
# the mistake shown: the 4 it reads as a 9
_wi = [k for k, (tt, pp) in enumerate(zip(R["wrong_true"], R["wrong_pred"])) if (tt, pp) == (4, 9)][0]
WRONG_I, WRONG_P = R["wrong"][_wi], R["wrong_p"][_wi]
BLAME_DIGIT = 3                                  # the digit whose pixel blame is shown (its unseen example)


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


SCRIPT = [
    "This network learned to read handwriting. Here's exactly how.",
    "These are real handwritten digits, scanned at 8 by 8 pixels. One thousand seven hundred and ninety-seven of them.",
    "Each pixel is one input: sixty-four numbers. They go through thirty-two neurons, and come out as ten probabilities, one for each digit.",
    f"Train it on {N_TRAIN} digits, and test it on {N_TEST} it never sees. At the start, it's guessing: {round(100 * ACC[0]):d} percent. By step {FIRST90}: ninety percent.",
    f"After three hundred steps, {N_RIGHT} of the {N_TEST} unseen digits are right. And every neuron's weights have changed: each one now looks for its own pattern of pixels.",
    "Backpropagation doesn't stop at the first layer. Send the blame all the way back, and every pixel finds out how it pushed the answer.",
    f"It still makes {N_WRONG} mistakes, like this four, which it reads as a nine.",
    "Sixty-four pixels, thirty-two neurons, ten probabilities, and backpropagation. That's a network learning to read.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the data, 2 the network, 3 training, 4 result, 5 pixel blame, 6 a mistake, 7 close
SEG_MIN = [3.5, 7.0, 9.0, 11.0, 8.0, 9.0, 7.0, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[8]
N_FRAMES = int(round((T_END + 0.7) * FPS))
TRAIN = (SEG[3] + 1.8, SEG[4] + 2.4)             # steps 0 -> 300
MOSAIC = (SEG[1] + 0.2, SEG[2] - 0.2)

DIGIT = (70, 470, 360)                           # the big digit: x, y, size
NEUR = (468, 560, 300)                           # the neuron pictures: x, y, width (8 x 4 tiles)
BARS = (812, 476, 200)                           # the ten probabilities: x, y, width
STRIP = (70, 905, 940)
CHART = (130, 1120, 850, 240)
CARD = (70, 1430, 940, 250)


def step_at(t):
    if t < SEG[1] - 0.2 or t >= SEG[4] + 0.6:
        return STEPS
    if t < TRAIN[0]:
        return 0
    # slowly up to the step where it passes 90%, timed to the words "by step ...", then on to the end
    tb = (VOICE_AT[3] + 0.8 * VOICE_DUR[3]) if VOICE_DUR else (TRAIN[0] + 0.6 * (TRAIN[1] - TRAIN[0]))
    if t < tb:
        return int(round(FIRST90 * seg(t, TRAIN[0], tb)))
    return int(round(FIRST90 + (STEPS - FIRST90) * seg(t, tb, TRAIN[1]) ** 1.2))


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    if t < SEG[2]:
        return "mosaic"
    if t < SEG[5]:
        return "train"
    if t < SEG[6]:
        return "blame"
    if t < SEG[7]:
        return "wrong"
    return "close"


def qa_frames():
    fr = []
    for k in range(8):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n=R["n"], n_train=N_TRAIN, n_test=N_TEST, acc_start=float(ACC[0]), acc_end=float(ACC[-1]), first_90=FIRST90, n_right=N_RIGHT, n_wrong=N_WRONG,
                          wrong_shown=dict(index=WRONG_I, true=4, pred=9, p=WRONG_P), segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("It learned to read handwriting.", "black", 48, INK, 300, 1), (f"{N_RIGHT} of {N_TEST} unseen digits right.", "dbold", 42, GREEN, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Real handwritten digits.", "black", 50, INK, 300, 1), (f"8 × 8 pixels. {R['n']:,} of them.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("64 pixels in. 10 probabilities out.", "black", 46, INK, 300, 1), ("32 neurons in between.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [(f"Trained on {N_TRAIN:,}. Tested on {N_TEST}.", "black", 46, INK, 300, 1), (f"Step 0: {pc(ACC[0])}%.  Step {FIRST90}: 90%.", "dbold", 42, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"Step {STEPS}: {pc(ACC[-1])}%.", "black", 52, GREEN, 300, 1), ("Each neuron: its own pattern of pixels.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The blame reaches the pixels.", "black", 48, MAGENTA, 300, 1), ("Red: more ink here helps. Blue: hurts.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [(f"{N_WRONG} mistakes out of {N_TEST}.", "black", 50, VERM, 300, 1), (f"This 4: \"a 9\", {round(100 * WRONG_P):d}% sure.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], T_END - 0.3, [("64 pixels, 32 neurons, 10 answers.", "black", 46, INK, 300, 1), ("And backpropagation.", "dbold", 42, MAGENTA, 390, 0.95)], 10, 8, "pop"),
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

    def pixels(self, ctx, x0, y0, size, img64, a, blame=None, frame=True):
        c = size / 8
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        if blame is not None:
            bm = np.abs(blame).max() + 1e-12
        for r in range(8):
            for q in range(8):
                v = float(img64[r * 8 + q])
                col = base * (1 - v) + ink * v
                if blame is not None:
                    g = float(blame[r * 8 + q]) / bm
                    tint = np.array(rgb(VERM)) if g > 0 else np.array(rgb(BLUE))
                    k = min(1.0, abs(g)) ** 0.7
                    col = base * (1 - k) + tint * k
                ctx.set_source_rgba(*col, a)
                ctx.rectangle(x0 + q * c + 1, y0 + r * c + 1, c - 2, c - 2)
                ctx.fill()
                if blame is not None and v > 0.05:
                    s = c * 0.42 * v
                    ctx.set_source_rgba(*ink, 0.9 * a)
                    ctx.rectangle(x0 + q * c + c / 2 - s / 2, y0 + r * c + c / 2 - s / 2, s, s)
                    ctx.fill()
        if frame:
            ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
            ctx.set_line_width(1.2)
            ctx.rectangle(x0, y0, size, size)
            ctx.stroke()

    def neurons(self, ctx, W1, a):
        x0, y0, w = NEUR
        tile = (w - 7 * 6) / 8
        for j in range(32):
            r, q = j // 8, j % 8
            wv = W1[:, j]
            m = np.abs(wv).max() + 1e-12
            tx, ty = x0 + q * (tile + 6), y0 + r * (tile + 6)
            c = tile / 8
            for p in range(64):
                g = wv[p] / m
                col = np.array(rgb(VERM)) if g > 0 else np.array(rgb(BLUE))
                k = abs(g) ** 0.8
                base = np.array(rgb(PANEL_BASE))
                ctx.set_source_rgba(*(base * (1 - k) + col * k), a)
                ctx.rectangle(tx + (p % 8) * c, ty + (p // 8) * c, c + 0.4, c + 0.4)
                ctx.fill()
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "32 NEURONS (THEIR WEIGHTS)", x0, y0 - 16, 0.1 * 16)

    def bars(self, ctx, probs, a, truth=None):
        x0, y0, w = BARS
        top = int(np.argmax(probs))
        for d in range(10):
            yy = y0 + d * 36
            font(ctx, "dbold", 24)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(x0, yy + 24)
            ctx.show_text(str(d))
            ctx.set_source_rgba(0.12, 0.15, 0.25, a)
            S3.pill(ctx, x0 + 26, yy + 8, w - 26, 18, 9)
            ctx.fill()
            col = GREEN if (truth is None or d == truth) and d == top else (VERM if d == top else INK3)
            ctx.set_source_rgba(*rgb(col), a)
            S3.pill(ctx, x0 + 26, yy + 8, max(4.0, (w - 26) * float(probs[d])), 18, 9)
            ctx.fill()

    def strip(self, ctx, img, t, step, a, mistakes=False):
        x0, y0, w = STRIP
        n = 10
        tile = 70
        gap = (w - n * tile) / (n - 1)
        items = []
        if mistakes:
            for k in range(min(10, N_WRONG)):
                i = R["wrong"][k]
                items.append((XTE[i], int(YTE[i]), int(R["wrong_pred"][k])))
        else:
            for d in range(10):
                p = PROBS[step][d]
                items.append((XTE[SHOW[d]], d, int(np.argmax(p))))
        for k, (im, tru, prd) in enumerate(items):
            xx = x0 + k * (tile + gap)
            self.pixels(ctx, xx, y0, tile, im, a, frame=False)
            font(ctx, "dbold", 26)
            ok = prd == tru
            s = f"{prd}" if not mistakes else f"{tru}→{prd}"
            ctx.set_source_rgba(*rgb(GREEN if ok else VERM), a)
            sw = ctx.text_extents(s).x_advance
            ctx.move_to(xx + tile / 2 - sw / 2, y0 + tile + 32)
            ctx.show_text(s)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        lab = f"{N_WRONG} OF ITS MISTAKES: TRUE → READ AS" if mistakes else "TEN UNSEEN DIGITS: WHAT IT READS"
        show_tracked(ctx, lab, x0, y0 - 14, 0.1 * 16)

    def chart(self, ctx, img, t, step, a):
        x0, y0, w, h = CHART

        def gx(s):
            return x0 + s / STEPS * w

        def gy(v):
            return y0 + h - v * h
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
        ctx.set_source_rgba(*rgb(GREEN), 0.95 * a)
        ctx.set_line_width(3.0)
        ctx.move_to(gx(0), gy(ACC[0]))
        for s in range(1, step + 1):
            ctx.line_to(gx(s), gy(ACC[s]))
        ctx.stroke()
        ctx.arc(gx(step), gy(ACC[step]), 5, 0, TAU)
        ctx.fill()
        if step >= FIRST90:
            ctx.set_source_rgba(1, 1, 1, 0.9 * a)
            ctx.set_line_width(2.0)
            ctx.arc(gx(FIRST90), gy(ACC[FIRST90]), 8, 0, TAU)
            ctx.stroke()
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, f"RIGHT ON THE {N_TEST} UNSEEN DIGITS, BY TRAINING STEP", x0, y0 - 14, 0.1 * 16)
        font(ctx, "dbold", 44)
        s = f"{pc(ACC[step])}%"
        ctx.set_source_rgba(*rgb(GREEN), a)
        ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, y0 + h - 30)
        ctx.show_text(s)
        font(ctx, "bold", 17)
        s2 = f"STEP {step}  ·  {int(round(ACC[step] * N_TEST))} OF {N_TEST}"
        ctx.set_source_rgba(*rgb(INK2), a)
        ctx.move_to(x0 + w - ctx.text_extents(s2).x_advance - 6, y0 + h - 84)
        ctx.show_text(s2)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        step = step_at(t)
        surf, ctx = layer()
        if md == "mosaic":
            # sixty real digits, in a grid
            k_in = ease_out_cubic(seg(t, MOSAIC[0], MOSAIC[0] + 1.0))
            x0, y0, tile, gap = 70, 470, 84, 10
            for k in range(60):
                r, q = k // 10, k % 10
                al = a * min(1.0, max(0.0, k_in * 60 - k * 0.6) / 6 + (k_in >= 1))
                self.pixels(ctx, x0 + q * (tile + gap), y0 + r * (tile + gap), tile, ALL[k * 29], min(1.0, al), frame=False)
        else:
            if md in ("hook", "train", "close"):
                d = int((t - SEG[2]) / 0.9) % 10 if md == "train" else BLAME_DIGIT
                i = SHOW[d]
                probs = PROBS[step][d]
                self.pixels(ctx, *DIGIT[:2], DIGIT[2], XTE[i], a)
                self.bars(ctx, probs, a, truth=d)
                lab = f"AN UNSEEN {d}"
            elif md == "blame":
                d = BLAME_DIGIT
                i = SHOW[d]
                self.pixels(ctx, *DIGIT[:2], DIGIT[2], XTE[i], a, blame=BLAME[i])
                self.bars(ctx, PROBS[STEPS][d], a, truth=d)
                lab = f"BLAME ON EACH PIXEL, FOR \"{d}\""
            else:
                self.pixels(ctx, *DIGIT[:2], DIGIT[2], XTE[WRONG_I], a)
                lab = "AN UNSEEN 4"
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, lab, DIGIT[0], DIGIT[1] - 14, 0.1 * 16)
            k = min(step // 10, len(W1S) - 1)
            self.neurons(ctx, W1S[k], a)
            # arrows: pixels -> neurons -> probabilities
            for xa, xb in ((DIGIT[0] + DIGIT[2] + 6, NEUR[0] - 8), (NEUR[0] + NEUR[2] + 6, BARS[0] - 8)):
                yy = 650
                ctx.set_source_rgba(*rgb(INK3), 0.8 * a)
                ctx.set_line_width(2.0)
                ctx.move_to(xa, yy)
                ctx.line_to(xb - 6, yy)
                ctx.stroke()
                ctx.move_to(xb, yy)
                ctx.line_to(xb - 10, yy - 6)
                ctx.line_to(xb - 10, yy + 6)
                ctx.close_path()
                ctx.fill()
            self.strip(ctx, img, t, step, a, mistakes=(md == "wrong"))
            self.chart(ctx, img, t, step, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if md == "wrong":
            surf, ctx = layer()
            self.bars(ctx, self.wrong_probs_vec(), a, truth=4)
            trgb, ta = surf_rgba(surf)
            over(img, trgb, ta)
        if md != "mosaic":
            font_lbl = r"$\mathrm{probability}$"
            paste(img, font_lbl, 18, BARS[0] + BARS[2], BARS[1] - 8, INK3, a, align="r")

    def wrong_probs_vec(self):
        # the network's ten probabilities for the shown mistake, recomputed from the final weights
        return self._wp

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close"):
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 32\ \mathrm{neurons} \;\rightarrow\; 10\ \mathrm{probabilities}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\mathrm{{trained\ on}}\ {N_TRAIN:,}\ \mathrm{{digits;\ tested\ on\ the\ other}}\ {N_TEST}:\ {pc(ACC[-1])}\%$".replace(",", "{,}"), 26, 540, y + 162, GREEN, a, align="c")
            paste(img, r"$\mathrm{UCI\ handwritten\ digits,\ 8 \times 8\ pixels}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "mosaic":
            paste(img, r"$\mathrm{a\ digit} = 64\ \mathrm{numbers:\ the\ ink\ in\ each\ pixel},\ 0\ \mathrm{to}\ 16$", 30, 540, y + 108, INK, a, align="c")
            paste(img, r"$\mathrm{the\ UCI\ handwritten\ digits:\ 1{,}797\ digits\ written\ by\ 13\ people}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "train" and t < SEG[3]:
            paste(img, r"$h = \mathrm{ReLU}(x\,W_1 + b_1),\qquad p = \mathrm{softmax}(h\,W_2 + b_2)$", 32, 540, y + 100, INK, a, align="c")
            paste(img, r"$x:\ 64\ \mathrm{pixels}\qquad h:\ 32\ \mathrm{neurons}\qquad p:\ 10\ \mathrm{probabilities}$", 24, 540, y + 162, INK2, a, align="c")
            paste(img, r"$\mathrm{each\ neuron\ picture:\ its\ 64\ weights\ (red\ +,\ blue\ -)}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "train":
            paste(img, r"$L = -\log p_{\mathrm{right\ digit}},\quad \mathrm{averaged\ over\ the\ training\ digits}$", 30, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{each\ step:\ backpropagation\ for\ all\ 2{,}410\ numbers,\ then\ one\ Adam\ step}$", 22, 540, y + 162, INK2, a, align="c")
            paste(img, r"$\mathrm{the\ accuracy\ is\ on\ the\ 540\ digits\ it\ never\ trains\ on}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "blame":
            paste(img, r"$\frac{\partial\, \log p_{\mathrm{right}}}{\partial\, \mathrm{pixel}}$", 40, 540, y + 110, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{the\ chain\ rule\ through\ both\ layers,\ down\ to\ the\ 64\ inputs}$", 22, 540, y + h - 26, INK2, a, align="c")
        else:
            paste(img, rf"$\mathrm{{read\ as\ a}}\ 9:\ p = {WRONG_P:.2f}\qquad \mathrm{{as\ a}}\ 4:\ p = {self._wp[4]:.2f}$", 30, 540, y + 100, VERM, a, align="c")
            paste(img, rf"${N_RIGHT}\ \mathrm{{right}},\ {N_WRONG}\ \mathrm{{wrong,\ out\ of}}\ {N_TEST}\ \mathrm{{unseen\ digits}}$", 26, 540, y + 162, INK2, a, align="c")
            paste(img, r"$\mathrm{its\ top\ is\ closed,\ like\ a\ 9's}$", 21, 540, y + h - 18, INK3, a, align="c")

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
        lab = "REAL HANDWRITING   ·   540 TEST DIGITS IT NEVER TRAINED ON   ·   ALL COMPUTED"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w_ / 2, 1752, tr)
        self._headlines(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return dither(img, f)


def _wrong_probs():
    """The final network's ten probabilities for the mistake shown (saved by dgsim.py)."""
    return D["wrong_probs"][list(R["wrong"]).index(WRONG_I)]


Scene._wp = property(lambda self: _wrong_probs())


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
    ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
