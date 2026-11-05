"""How to fool a neural network. Dark theme of reels 31-49, maths in Computer Modern (stage3d.tex), narrated by Kokoro
(voicekit). The handwriting network of reel 49 (64 pixels -> 32 neurons -> 10 probabilities, 96.7% on 540 unseen
digits) and its unseen 3, read as a 3 with p = 0.999. Backpropagation carried to the pixels gives the direction
that makes "3" less likely; pushing every pixel one and then two shades of ink (out of 16) that way turns it into
"a 9" (p = 0.992). Then the same 3 nudged toward each of the ten digits, and all 540 unseen digits nudged along
their own gradients against random nudges of the same size (advsim.py). frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "adv_results.json")):
    import advsim
    advsim.main()
with open(os.path.join(HERE, "adv_results.json")) as fh:
    R = json.load(fh)
_D = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))
_DZ = np.load(os.path.join(ROOT, "digits", "dg_results.npz")) if os.path.exists(os.path.join(ROOT, "digits", "dg_results.npz")) else None
if _DZ is None:
    sys.path.insert(0, os.path.join(ROOT, "digits"))
    import dgsim
    dgsim.main()
    _DZ = np.load(os.path.join(ROOT, "digits", "dg_results.npz"))
XTE = _DZ["Xte"]
SHOW = _D["show"]
X3 = np.array(R["x3"])
P3 = np.array(R["p3"])
G3 = np.array(R["grad3"])
PK = [np.array(p) for p in R["p3_by_k"]]                     # the 3's probabilities after 0..4 shades
XK = [X3] + [np.array(R["one"][str(k)]["x"]) for k in (1, 2, 3)]
K_FOOL = R["k_first"]                                          # 2
P_FOOL = PK[K_FOOL]
READ_FOOL = int(np.argmax(P_FOOL))                             # 9
CHANGED = R["one"][str(K_FOOL)]["changed"]
TG = {int(t): v for t, v in R["targeted"].items()}
K_TG = R["k_targeted_max"]
K_TG_MIN = min(v["k"] for t, v in TG.items() if t != 3)
CG, CR = R["curve_gradient"], R["curve_random"]
RA, RR = R["right_after"], R["right_after_random"]
STRIP3 = R["strip3"]
ACC = R["acc"]
N_TEST = R["n_test"]
DIR = np.sign(G3) * (((G3 > 0) & (X3 < 1)) | ((G3 < 0) & (X3 > 0)))     # the way each pixel can still move


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (K_FOOL, READ_FOOL, round(100 * P_FOOL[READ_FOOL]), RA["3"], RR["3"], K_TG) == (2, 9, 99, 4, 450, 4), "script numbers changed"
SCRIPT = [
    "I changed this three by two shades of ink. Now the network is ninety-nine percent sure it's a nine.",
    "This is the network from the handwriting reel. It reads ninety-six point seven percent of digits it has never seen, and it's ninety-nine point nine percent sure this is a three.",
    "Backpropagation carries the blame all the way back to the pixels: for each one, which way would make three less likely. Red: add ink. Blue: take some away.",
    "So push every pixel that way. One shade, out of sixteen: still a three, but only sixty-three percent sure. Two shades: a nine. Ninety-nine percent sure.",
    "Or pick the answer you want, and push toward it instead. The same three can be read as every digit from zero to nine. Never more than four shades.",
    "And it's not just this three. Push all five hundred and forty test digits three shades along their own gradients, and four are still right. Push them the same amount at random: four hundred and fifty.",
    "The gradient that teaches a network is also a map for fooling it. That's called an adversarial example.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 the gradient, 3 the nudge, 4 any answer, 5 all 540, 6 close
SEG_MIN = [5.0, 9.0, 9.0, 10.0, 9.5, 12.0, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    """When the narration reaches `phrase` in line i (by its position in the text)."""
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_ONE = word_t(3, "One shade")
T_TWO = word_t(3, "Two shades")
T_TG0, T_TG1 = word_t(4, "The same three"), word_t(4, "Never more")
T_ALL0, T_ALL1 = word_t(5, "Push all"), word_t(5, "Push them the same")

TRIO = (70, 480, 280)                    # three tiles: the digit, the nudge, the nudged digit (x of the first, y, size)
TX = (70, 400, 730)
BARS_Y = 800
STRIP = (70, 1214, 940)
CHART = (150, 905, 820, 210)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("net", "dir", "nudge", "target", "all", "close")[k - 1]
    return "close"


def k_at(t):
    md = mode(t)
    if md in ("hook", "close", "target"):
        return K_FOOL
    if md in ("net", "dir"):
        return 0
    if md == "nudge":
        return 0 if t < T_ONE else (1 if t < T_TWO else 2)
    return 3


def n_targets(t):
    """How many of the ten targeted 3s are on the strip."""
    if t < T_TG0 - 0.4:
        return 0
    return min(10, 1 + int((t - (T_TG0 - 0.4)) / max(0.3, (T_TG1 - T_TG0 + 0.4) / 10)))


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_TWO + 1.2) * FPS), int((T_ALL1 + 1.5) * FPS), int((T_TG1 + 0.5) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(acc=ACC, p3=float(P3[3]), k_fool=K_FOOL, read_fool=READ_FOOL, p_fool=float(P_FOOL[READ_FOOL]), p3_one_shade=float(PK[1][3]),
                          changed=CHANGED, k_targeted_max=K_TG, right_after=RA, right_after_random=RR, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [(f"I changed this 3 by {K_FOOL} shades.", "black", 48, INK, 300, 1), (f"Now: \"a {READ_FOOL}\", {pc(P_FOOL[READ_FOOL])}% sure.", "dbold", 42, VERM, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("The handwriting network.", "black", 50, INK, 300, 1), (f"{pc(ACC)}% right. This 3: {pc(P3[3])}% sure.", "dbold", 42, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Backpropagation, down to the pixels.", "black", 44, MAGENTA, 300, 1), ("Which way makes \"3\" less likely?", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Push every pixel that way.", "black", 50, INK, 300, 1), (f"{K_FOOL} shades out of 16: \"a {READ_FOOL}\".", "dbold", 42, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Or pick the answer you want.", "black", 50, INK, 300, 1), (f"Every digit, 0 to 9. At most {K_TG} shades.", "dbold", 40, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [(f"All {N_TEST} digits, 3 shades.", "black", 50, INK, 300, 1), (f"Along the gradient: {RA['3']} right. At random: {RR['3']}.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("The gradient that teaches it", "black", 48, INK, 300, 1), ("is a map for fooling it.", "dbold", 44, MAGENTA, 390, 0.95)], 10, 8, "pop"),
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

    def change_map(self, ctx, x0, y0, size, d16, a, frame=True):
        """A nudge, pixel by pixel: red = ink added, blue = ink taken away, brighter = more shades (4 = full)."""
        c = size / 8
        base = np.array(rgb(PANEL_BASE))
        for p in range(64):
            v = float(d16[p])
            k = min(1.0, abs(v) / 4) ** 0.6
            col = np.array(rgb(VERM if v > 0 else BLUE))
            ctx.set_source_rgba(*(base * (1 - k) + col * k), a)
            ctx.rectangle(x0 + (p % 8) * c + 1, y0 + (p // 8) * c + 1, c - 2, c - 2)
            ctx.fill()
        if frame:
            ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
            ctx.set_line_width(1.2)
            ctx.rectangle(x0, y0, size, size)
            ctx.stroke()

    def bars(self, ctx, x0, y0, w, probs, a, truth=3):
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

    def reading(self, ctx, x0, w, y, probs, a, truth=3, size=30, wrong=VERM):
        top = int(np.argmax(probs))
        s = f"reads {top}  ·  {pc(probs[top])}%"
        font(ctx, "dbold", size)
        ctx.set_source_rgba(*rgb(GREEN if top == truth else wrong), a)
        ctx.move_to(x0 + w / 2 - ctx.text_extents(s).x_advance / 2, y)
        ctx.show_text(s)

    def label(self, ctx, s, x, y, a, col=INK3):
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        show_tracked(ctx, s, x, y, 0.1 * 16)

    def strip(self, ctx, t, a):
        x0, y0, w = STRIP
        tile, n = 70, 10
        gap = (w - n * tile) / (n - 1)
        md = mode(t)
        items = []
        if md == "target":
            nt = n_targets(t)
            for d in range(10):
                if d < nt:
                    v = TG[d]
                    items.append((np.array(v["x"]), f"{d}", GOLD if d != 3 else GREEN, 1.0))
                else:
                    items.append((X3, "", INK3, 0.25))
            lab = f"THE SAME 3, NUDGED TOWARD EACH DIGIT: WHAT IT READS"
        elif md == "all":
            for d in range(10):
                o = STRIP3[d]
                items.append((np.array(o["x"]), f"{o['read']}", GREEN if o["read"] == d else VERM, 1.0))
            lab = "TEN UNSEEN DIGITS, 3 SHADES ALONG THEIR GRADIENT: WHAT IT READS"
        else:
            for d in range(10):
                items.append((XTE[SHOW[d]], f"{d}", GREEN, 1.0))
            lab = "TEN UNSEEN DIGITS: WHAT IT READS"
        for k, (im, s, col, al) in enumerate(items):
            xx = x0 + k * (tile + gap)
            self.pixels(ctx, xx, y0, tile, im, a * al, frame=False)
            if s:
                font(ctx, "dbold", 26)
                ctx.set_source_rgba(*rgb(col), a)
                ctx.move_to(xx + tile / 2 - ctx.text_extents(s).x_advance / 2, y0 + tile + 32)
                ctx.show_text(s)
        self.label(ctx, lab, x0, y0 - 14, a)

    def chart(self, ctx, t, a):
        x0, y0, w, h = CHART
        kk = 4 * smooth(seg(t, T_ALL0 - 0.3, T_ALL1 + 0.6)) if mode(t) == "all" else 0.0
        kr = 4 * smooth(seg(t, T_ALL1 - 0.2, T_ALL1 + 2.2)) if mode(t) == "all" else 0.0

        def gx(k):
            return x0 + k / 4 * w

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
        font(ctx, "dbold", 18)
        for k in range(5):
            ctx.set_source_rgba(*rgb(INK2), a)
            s = str(k)
            ctx.move_to(gx(k) - ctx.text_extents(s).x_advance / 2, y0 + h + 26)
            ctx.show_text(s)
        for v, s in ((0.0, "0"), (0.5, "50%"), (1.0, "100%")):
            ctx.move_to(x0 - 12 - ctx.text_extents(s).x_advance, gy(v) + 6)
            ctx.show_text(s)
        for curve, upto, col in ((CR, kr, BLUE), (CG, kk, MAGENTA)):
            if upto <= 0:
                continue
            ctx.set_source_rgba(*rgb(col), a)
            ctx.set_line_width(3.5)
            ctx.move_to(gx(0), gy(curve[0]))
            kf = int(math.floor(upto))
            for k in range(1, kf + 1):
                ctx.line_to(gx(k), gy(curve[k]))
            if kf < 4:
                fr_ = upto - kf
                ctx.line_to(gx(upto), gy(curve[kf] * (1 - fr_) + curve[kf + 1] * fr_))
            ctx.stroke()
            for k in range(0, kf + 1):
                ctx.arc(gx(k), gy(curve[k]), 5.5, 0, TAU)
                ctx.fill()
            if upto >= 3:
                ctx.set_source_rgba(1, 1, 1, 0.9 * a)
                ctx.set_line_width(2.0)
                ctx.arc(gx(3), gy(curve[3]), 10, 0, TAU)
                ctx.stroke()
        font(ctx, "dbold", 26)
        # the readouts at 3 shades, in the free space between the two curves
        if kr >= 3:
            s = f"at random: {RR['3']} of {N_TEST} right"
            ctx.set_source_rgba(*rgb(BLUE), a)
            ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, gy(0.50))
            ctx.show_text(s)
        if kk >= 3:
            s = f"along the gradient: {RA['3']} of {N_TEST} right"
            ctx.set_source_rgba(*rgb(MAGENTA), a)
            ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, gy(0.30))
            ctx.show_text(s)
        self.label(ctx, f"RIGHT, OF THE {N_TEST} UNSEEN DIGITS, BY SHADES PUSHED (EVERY PIXEL)", x0 - 80, y0 - 30, a)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        k = k_at(t)
        surf, ctx = layer()
        x_tile, y_tile, size = TRIO
        if md == "all":
            # one unseen digit at a time, nudged 3 shades along its own gradient
            d = min(9, max(0, int((t - SEG[5] - 0.4) / ((SEG[6] - SEG[5] - 0.6) / 10))))
            o = STRIP3[d]
            xo, xa = XTE[SHOW[d]], np.array(o["x"])
            self.pixels(ctx, TX[0], y_tile, size, xo, a)
            self.change_map(ctx, TX[1], y_tile, size, np.round((xa - xo) * 16), a)
            self.pixels(ctx, TX[2], y_tile, size, xa, a)
            self.label(ctx, f"AN UNSEEN {d}", TX[0], y_tile - 14, a)
            self.label(ctx, "NUDGE: 3 SHADES", TX[1], y_tile - 14, a)
            self.label(ctx, "NUDGED", TX[2], y_tile - 14, a)
            font(ctx, "dbold", 30)
            for xx, s, col in ((TX[0], f"reads {d}", GREEN), (TX[2], f"reads {o['read']}", GREEN if o["read"] == d else VERM)):
                ctx.set_source_rgba(*rgb(col), a)
                ctx.move_to(xx + size / 2 - ctx.text_extents(s).x_advance / 2, y_tile + size + 44)
                ctx.show_text(s)
            self.chart(ctx, t, a)
        else:
            if md == "target":
                nt = n_targets(t)
                tgt = (nt - 1) if nt > 0 else None
                if tgt is None or tgt == 3:
                    xa, pa, dm, lab_n = XK[K_FOOL], PK[K_FOOL], np.round((XK[K_FOOL] - X3) * 16), f"NUDGE: ±{K_FOOL} SHADES"
                    if tgt == 3:
                        xa, pa, dm, lab_n = X3, P3, np.zeros(64), "TARGET 3: NO NUDGE"
                else:
                    v = TG[tgt]
                    xa, pa = np.array(v["x"]), np.array(v["p"])
                    dm = np.round((xa - X3) * 16)
                    lab_n = f"TOWARD {tgt}: ±{v['k']} SHADES"
            else:
                xa, pa = XK[k], PK[k]
                dm = np.round((xa - X3) * 16)
                lab_n = f"NUDGE: ±{k} SHADE{'S' if k != 1 else ''}" if k else "NO NUDGE YET"
            self.pixels(ctx, TX[0], y_tile, size, X3, a)
            if md == "dir":
                # the direction alone: where each pixel would have to go
                al = smooth(seg(t, SEG[2] + 0.6, SEG[2] + 2.0))
                self.change_map(ctx, TX[1], y_tile, size, np.zeros(64), a)
                self.change_map(ctx, TX[1], y_tile, size, DIR * 2.2, a * al, frame=False)
                lab_n = "WHICH WAY EACH PIXEL GOES"
            elif md == "net":
                self.change_map(ctx, TX[1], y_tile, size, np.zeros(64), a)
            else:
                self.change_map(ctx, TX[1], y_tile, size, dm, a)
            self.pixels(ctx, TX[2], y_tile, size, xa, a)
            self.label(ctx, "AN UNSEEN 3", TX[0], y_tile - 14, a)
            self.label(ctx, lab_n, TX[1], y_tile - 14, a)
            self.label(ctx, "WHAT THE NETWORK GETS", TX[2], y_tile - 14, a)
            self.bars(ctx, TX[0], BARS_Y, size, P3, a)
            self.bars(ctx, TX[2], BARS_Y, size, pa, a)
            self.reading(ctx, TX[0], size, BARS_Y + 344, P3, a)
            self.reading(ctx, TX[2], size, BARS_Y + 344, pa, a, wrong=GOLD if md == "target" else VERM)
            # the middle column: how big the nudge is
            if md not in ("net", "dir"):
                kk = int(np.abs(dm).max())
                font(ctx, "black", 64)
                s = f"{kk}"
                ctx.set_source_rgba(*rgb(GOLD), a)
                ctx.move_to(TX[1] + size / 2 - ctx.text_extents(s).x_advance / 2, BARS_Y + 90)
                ctx.show_text(s)
                font(ctx, "dbold", 24)
                for j, s in enumerate((f"shade{'s' if kk != 1 else ''} of ink, at most,", "per pixel (of 16)", f"{int(np.sum(dm != 0))} of 64 pixels changed")):
                    ctx.set_source_rgba(*rgb(INK2 if j < 2 else INK3), a)
                    ctx.move_to(TX[1] + size / 2 - ctx.text_extents(s).x_advance / 2, BARS_Y + 136 + 34 * j)
                    ctx.show_text(s)
            font(ctx, "black", 52)
            ctx.set_source_rgba(*rgb(INK3), a)
            for xx, s in ((TX[1] - 25, "+"), (TX[2] - 25, "=")):
                ctx.move_to(xx - ctx.text_extents(s).x_advance / 2, y_tile + size / 2 + 18)
                ctx.show_text(s)
        if md == "all":
            font(ctx, "black", 52)
            ctx.set_source_rgba(*rgb(INK3), a)
            for xx, s in ((TX[1] - 25, "+"), (TX[2] - 25, "=")):
                ctx.move_to(xx - ctx.text_extents(s).x_advance / 2, y_tile + size / 2 + 18)
                ctx.show_text(s)
        self.strip(ctx, t, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if md != "all":
            paste(img, r"$\mathrm{probability}$", 18, TX[0] + size, BARS_Y - 8, INK3, a, align="r")
            paste(img, r"$\mathrm{probability}$", 18, TX[2] + size, BARS_Y - 8, INK3, a, align="r")

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close"):
            paste(img, rf"$p(3) = {P3[3]:.3f}\ \ \longrightarrow\ \ p({READ_FOOL}) = {P_FOOL[READ_FOOL]:.3f}$", 34, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\mathrm{{every\ pixel\ moved\ at\ most}}\ {K_FOOL}\ \mathrm{{shades\ (of\ 16),\ along\ the\ gradient}}$", 23, 540, y + 160, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{the\ network\ of\ the\ handwriting\ reel,\ on\ its\ 540\ unseen\ digits}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "net":
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 32\ \mathrm{neurons} \;\rightarrow\; 10\ \mathrm{probabilities}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, rf"$522\ \mathrm{{of\ the}}\ 540\ \mathrm{{unseen\ digits\ right}}:\ {pc(ACC)}\%$", 26, 540, y + 162, GREEN, a, align="c")
            paste(img, r"$\mathrm{a\ pixel's\ ink:\ 0\ to\ 16\ shades}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif md == "dir":
            paste(img, r"$L = -\log p(3),\qquad \mathrm{direction} = \mathrm{sign}\!\left(\frac{\partial L}{\partial\, \mathrm{pixel}}\right)$", 32, 540, y + 108, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{the\ chain\ rule\ through\ both\ layers,\ down\ to\ the\ 64\ pixels}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "nudge":
            paste(img, r"$\mathrm{pixel} \;\leftarrow\; \mathrm{pixel} + k \cdot \mathrm{sign}\!\left(\frac{\partial L}{\partial\, \mathrm{pixel}}\right)$", 32, 540, y + 108, INK, a, align="c")
            paste(img, rf"$k = 1:\ p(3) = {PK[1][3]:.2f}\qquad k = 2:\ p(3) = {PK[2][3]:.3f},\ p(9) = {PK[2][9]:.3f}$", 23, 540, y + 182, INK2, a, align="c")
            paste(img, r"$\mathrm{kept\ inside\ 0\ to\ 16:\ still\ a\ real\ scan}$", 21, 540, y + h - 16, INK3, a, align="c")
        elif md == "target":
            paste(img, r"$\mathrm{toward\ digit}\ d:\ \ \mathrm{pixel} \;\leftarrow\; \mathrm{pixel} - \mathrm{sign}\!\left(\frac{\partial\,(-\log p(d))}{\partial\, \mathrm{pixel}}\right)$", 30, 540, y + 108, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{40\ small\ steps,\ never\ more\ than}}\ k\ \mathrm{{shades\ from\ the\ 3}}\ \ (k = {K_TG_MIN}\ \mathrm{{to}}\ {K_TG})$", 22, 540, y + h - 26, INK2, a, align="c")
        else:
            paste(img, rf"$3\ \mathrm{{shades,\ along\ each\ digit's\ own\ gradient}}:\ {RA['3']}\ \mathrm{{of}}\ {N_TEST}\ \mathrm{{right}}$", 26, 540, y + 96, MAGENTA, a, align="c")
            paste(img, rf"$3\ \mathrm{{shades,\ random\ directions}}:\ {RR['3']}\ \mathrm{{of}}\ {N_TEST}\ \mathrm{{right}}$", 26, 540, y + 152, BLUE, a, align="c")
            paste(img, r"$\mathrm{random:\ median\ of\ 20\ draws.\ \ The\ direction\ is\ what\ matters.}$", 21, 540, y + h - 18, INK3, a, align="c")

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
        lab = "REAL HANDWRITING   ·   THE HANDWRITING REEL'S NETWORK   ·   ALL COMPUTED"
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
