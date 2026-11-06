"""Store every weight in one bit (quantization). Dark theme of reels 31-57, maths in Computer Modern (stage3d.tex),
narrated by Kokoro (voicekit). The handwriting network of reel 49 (2,368 weights of 32 bits, 96.7%): rounded to 8 bits
it still reads 96.7%, to 4 bits 93.3%; down to 1 bit (only the sign of each weight, one size per layer) it falls to
50.2%; with 100 steps of training with the rounding built in, the 1-bit network reads 93.7% (quantsim.py). On screen:
the 32 neurons' weight pictures, the first layer's 2,048 weight values snapping onto the allowed values, the size,
the score, and ten unseen digits with what it reads. frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "quant_results.json")):
    import quantsim
    quantsim.main()
with open(os.path.join(HERE, "quant_results.json")) as fh:
    R = json.load(fh)
_NPZ = os.path.join(ROOT, "digits", "dg_results.npz")
if not os.path.exists(_NPZ):
    sys.path.insert(0, os.path.join(ROOT, "digits"))
    import dgsim
    dgsim.main()
XTE = np.load(_NPZ)["Xte"]
ROWS = {r["bits"]: r for r in R["rows"]}
RT = R["one_bit_retrained"]
NW = R["n_weights"]
SHOW = R["show"]
W_OF = {32: np.array(ROWS[32]["W1"]), 8: np.array(ROWS[8]["W1"]), 4: np.array(ROWS[4]["W1"]), 1: np.array(ROWS[1]["W1"]), "rt": np.array(RT["W1"])}
ACC_OF = {32: ROWS[32]["acc"], 8: ROWS[8]["acc"], 4: ROWS[4]["acc"], 1: ROWS[1]["acc"], "rt": RT["acc"]}
READS_OF = {32: ROWS[32]["reads"], 8: ROWS[8]["reads"], 4: ROWS[4]["reads"], 1: ROWS[1]["reads"], "rt": RT["reads"]}
BITS_OF = {32: 32, 8: 8, 4: 4, 1: 1, "rt": 1}
FLIPS1 = np.sign(W_OF["rt"]) != np.sign(W_OF[32])
OTH = R["others"]


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (NW, pc(ACC_OF[32]), pc(ACC_OF[8]), pc(ACC_OF[4], 0), pc(ACC_OF[1], 0), pc(RT["acc"]), RT["sign_flips"], 2 ** 8 - 1, 2 ** 4 - 1) == \
    (2368, "96.7", "96.7", "93", "50", "93.7", 163, 255, 15), "script numbers changed"
SCRIPT = [
    "Every weight in this network is now just one bit: plus or minus. It still reads ninety-three point seven percent of handwriting.",
    "It's the handwriting network. Each of its two thousand three hundred and sixty-eight weights is stored in thirty-two bits, and it reads ninety-six point seven percent.",
    "Round every weight to eight bits, at most two hundred and fifty-five values: still ninety-six point seven. Four bits, fifteen values: ninety-three.",
    "Now one bit. Each weight keeps only its sign, plus or minus the same size. Just rounding, it falls to fifty percent.",
    "But train it a hundred more steps with the rounding built in. A hundred and sixty-three signs flip, and it's back to ninety-three point seven, with weights thirty-two times smaller.",
    "That's quantization: how big AI models squeeze onto laptops and phones, often at eight or four bits per weight.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 rounding, 3 one bit, 4 retrain, 5 close
SEG_MIN = [6.5, 8.5, 10.0, 8.0, 11.0, 8.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[6]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_8 = word_t(2, "Round every")
T_4 = word_t(2, "Four bits")
T_1 = word_t(3, "Each weight keeps")
T_RT = word_t(4, "A hundred and sixty-three")
SNAP = 0.45                                       # the cross-fade when the weights change (seconds)

TILES = (70, 470, 940)
HIST = (110, 1092, 860, 132)
STRIP = (70, 1306, 940)
CARD = (70, 1430, 940, 250)
XR = 1.5                                          # the histogram shows weights from -1.5 to 1.5


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 6):
        if t < SEG[k + 1]:
            return ("net", "round", "one", "retrain", "close")[k - 1]
    return "close"


def state(t):
    """(previous version, current version, cross-fade 0..1). Versions: 32, 8, 4, 1 bits, or "rt" (1 bit retrained)."""
    md = mode(t)
    if md in ("hook", "close"):
        return "rt", "rt", 1.0
    if md == "net":
        return 32, 32, 1.0
    if md == "round":
        if t < T_8:
            return 32, 32, 1.0
        if t < T_4:
            return 32, 8, smooth(seg(t, T_8, T_8 + SNAP))
        return 8, 4, smooth(seg(t, T_4, T_4 + SNAP))
    if md == "one":
        if t < T_1:
            return 4, 4, 1.0
        return 4, 1, smooth(seg(t, T_1, T_1 + SNAP))
    if t < T_RT:
        return 1, 1, 1.0
    return 1, "rt", smooth(seg(t, T_RT, T_RT + SNAP))


def qa_frames():
    fr = []
    for k in range(6):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_4 + 1.0) * FPS), int((T_1 + 1.0) * FPS), int((T_RT + 1.2) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n_weights=NW, acc={str(k): v for k, v in ACC_OF.items()}, flips=RT["sign_flips"], others=OTH, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        self.hist = {k: self._hist(v) for k, v in W_OF.items()}

    def _cues(self):
        S_ = SEG
        hook = [("Every weight: one bit.", "black", 56, INK, 300, 1), (f"Still {pc(RT['acc'])}% right.", "dbold", 46, GREEN, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("The handwriting network.", "black", 50, INK, 300, 1), (f"32 bits per weight: {pc(ACC_OF[32])}%.", "dbold", 44, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Round to 8 bits, then 4.", "black", 52, INK, 300, 1), (f"{pc(ACC_OF[8])}%, then {pc(ACC_OF[4])}%.", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Now 1 bit: just the sign.", "black", 52, INK, 300, 1), (f"It falls to {pc(ACC_OF[1])}%.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Retrain, rounding built in.", "black", 50, INK, 300, 1), (f"{pc(RT['acc'])}%, with weights 32× smaller.", "dbold", 42, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[5], T_END - 0.3, [("That's quantization.", "black", 54, INK, 300, 1), ("How big models fit on phones.", "dbold", 44, GREEN, 390, 0.95)], 10, 8, "pop"),
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

    @staticmethod
    def _hist(w):
        """Counts of the 2,048 first-layer weights: in 171 bins over -1.5..1.5 (a weight beyond goes in the end bin)."""
        nb = 171
        e = np.linspace(-XR, XR, nb + 1)
        c, _ = np.histogram(np.clip(w.ravel(), -XR + 1e-9, XR - 1e-9), bins=e)
        return c, int(np.sum(np.abs(w) > XR))

    # ------------------------------------------------------------ pieces

    def tiles(self, ctx, va, vb, k, a):
        x0, y0, w = TILES
        gap = 10
        tile = (w - 7 * gap) / 8
        c = tile / 8
        base = np.array(rgb(PANEL_BASE))
        red, blue = np.array(rgb(VERM)), np.array(rgb(BLUE))

        def colours(Wn):
            mx = np.abs(Wn).max(0, keepdims=True) + 1e-12       # each neuron scaled to its own largest weight
            g = Wn / mx
            kk = (np.minimum(1.0, np.abs(g)) ** 0.8)[..., None]
            return base * (1 - kk) + np.where(g[..., None] > 0, red, blue) * kk
        col = colours(W_OF[vb]) if k >= 1 else colours(W_OF[va]) * (1 - k) + colours(W_OF[vb]) * k
        flash = 0.0
        if vb == "rt" and va == 1:
            flash = math.sin(math.pi * min(1.0, k)) if k < 1 else 0.0
        for j in range(32):
            r, q = j // 8, j % 8
            tx, ty = x0 + q * (tile + gap), y0 + r * (tile + gap)
            for p in range(64):
                cc = col[p, j]
                if flash > 0 and FLIPS1[p, j]:
                    cc = cc * (1 - flash) + np.ones(3) * flash
                ctx.set_source_rgba(*cc, a)
                ctx.rectangle(tx + (p % 8) * c, ty + (p // 8) * c, c + 0.4, c + 0.4)
                ctx.fill()
            ctx.set_source_rgba(*rgb(INK3), 0.4 * a)
            ctx.set_line_width(1.0)
            ctx.rectangle(tx, ty, tile, tile)
            ctx.stroke()
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "32 NEURONS, THEIR 64 WEIGHTS EACH (RED +, BLUE −)", x0, y0 - 14, 0.1 * 16)
        # bits, size and the score, under the tiles
        v = vb if k >= 0.5 else va
        bits = BITS_OF[v]
        yb = y0 + 4 * (tile + gap) + 46
        font(ctx, "black", 54)
        ctx.set_source_rgba(*rgb(GOLD), a)
        s = f"{bits} bit" + ("s" if bits > 1 else "")
        ctx.move_to(x0, yb + 20)
        ctx.show_text(s)
        w1 = ctx.text_extents(s).x_advance
        font(ctx, "dbold", 24)
        ctx.set_source_rgba(*rgb(INK2), a)
        ctx.move_to(x0 + w1 + 14, yb + 12)
        ctx.show_text("per weight")
        font(ctx, "dbold", 20)
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(x0, yb + 48)
        ctx.show_text(f"all {NW:,} weights: {int(NW * bits / 8):,} bytes")
        acc = ACC_OF[v]
        colr = GREEN if acc > 0.95 or v == "rt" else (GOLD if acc > 0.9 else VERM)
        font(ctx, "black", 54)
        s2 = f"{pc(acc)}%"
        ctx.set_source_rgba(*rgb(colr), a)
        ctx.move_to(x0 + w - ctx.text_extents(s2).x_advance, yb + 20)
        ctx.show_text(s2)
        font(ctx, "dbold", 20)
        s3 = {32: "the full network", 8: "rounded to 8 bits", 4: "rounded to 4 bits", 1: "rounded to 1 bit", "rt": f"1 bit + {R['retrain_steps']} steps of retraining"}[v]
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(x0 + w - ctx.text_extents(s3).x_advance, yb + 48)
        ctx.show_text(s3)

    def histogram(self, ctx, va, vb, k, a):
        x0, y0, w, h = HIST
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        font(ctx, "dbold", 17)
        ctx.set_source_rgba(*rgb(INK2), a)
        for v in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
            s = f"{v:g}".replace("-", "−")
            gx = x0 + (v + XR) / (2 * XR) * w
            ctx.move_to(gx - ctx.text_extents(s).x_advance / 2, y0 + h + 22)
            ctx.show_text(s)
        nb = len(self.hist[32][0])
        bw = w / nb
        for v, al in ((va, 1 - k), (vb, k)):
            if al <= 0.003:
                continue
            c, beyond = self.hist[v]
            top = c.max()
            for i, n in enumerate(c):
                if n == 0:
                    continue
                hh = h * (n / top) ** 0.5            # square-root heights, so the small counts show
                ctx.set_source_rgba(*rgb(GOLD if v != 32 else INK2), 0.95 * al * a)
                ctx.rectangle(x0 + i * bw + 0.3, y0 + h - hh, max(1.2, bw - 0.6), hh)
                ctx.fill()
        v = vb if k >= 0.5 else va
        c, beyond = self.hist[v]
        nvals = len(np.unique(W_OF[v]))
        font(ctx, "dbold", 19)
        s = "2,048 different values" if v == 32 else f"{nvals} different values"
        ctx.set_source_rgba(*rgb(GOLD if v != 32 else INK2), a)
        ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, y0 + 14)
        ctx.show_text(s)
        if beyond:
            font(ctx, "dbold", 16)
            s = f"{beyond} beyond ±1.5 drawn at the ends"
            ctx.set_source_rgba(*rgb(INK3), a)
            ctx.move_to(x0, y0 + 14)
            ctx.show_text(s)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "THE 2,048 FIRST-LAYER WEIGHTS, BY VALUE", x0, y0 - 18, 0.1 * 16)

    def strip(self, ctx, v, a):
        x0, y0, w = STRIP
        tile, n = 56, 10
        gap = (w - n * tile) / (n - 1)
        reads = READS_OF[v]
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        for k in range(10):
            xx = x0 + k * (tile + gap)
            c = tile / 8
            img = XTE[SHOW[k]]
            for p in range(64):
                val = float(img[p])
                ctx.set_source_rgba(*(base * (1 - val) + ink * val), a)
                ctx.rectangle(xx + (p % 8) * c + 0.5, y0 + (p // 8) * c + 0.5, c - 1, c - 1)
                ctx.fill()
            s = str(reads[k])
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(GREEN if reads[k] == k else VERM), a)
            ctx.move_to(xx + tile + 6, y0 + tile / 2 + 10)
            ctx.show_text(s)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "TEN UNSEEN DIGITS: WHAT IT READS", x0, y0 - 14, 0.1 * 16)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        va, vb, k = state(t)
        surf, ctx = layer()
        self.tiles(ctx, va, vb, k, a)
        self.histogram(ctx, va, vb, k, a)
        self.strip(ctx, vb if k >= 0.5 else va, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md == "net":
            paste(img, r"$64 \times 32 + 32 \times 10 = 2{,}368\ \mathrm{weights}$", 30, 540, y + 96, INK, a, align="c")
            paste(img, r"$2{,}368 \times 32\ \mathrm{bits} = 75{,}776\ \mathrm{bits} = 9{,}472\ \mathrm{bytes}$", 28, 540, y + 162, GOLD, a, align="c")
        elif md == "round":
            paste(img, r"$w \;\rightarrow\; s \cdot \mathrm{round}(w / s),\quad s = \max|w| \,/\, 127\ (8\ \mathrm{bits})\ \mathrm{or}\ /\,7\ (4\ \mathrm{bits})$", 23, 540, y + 96, GOLD, a, align="c")
            paste(img, r"$\mathrm{one}\ s\ \mathrm{per\ layer:\ 8\ bits\ allow\ 255\ values,\ 4\ bits\ allow\ 15}$", 22, 540, y + 156, INK2, a, align="c")
            paste(img, rf"$\mathrm{{(3\ bits,\ 7\ values:\ {pc(ROWS[3]['acc'])}\%)}}$", 20, 540, y + h - 22, INK3, a, align="c")
        elif md == "one":
            paste(img, r"$w \;\rightarrow\; s \cdot \mathrm{sign}(w),\qquad s = \mathrm{the\ mean\ of}\ |w|\ \mathrm{in\ the\ layer}$", 28, 540, y + 96, VERM, a, align="c")
            paste(img, r"$\mathrm{two\ values\ per\ layer:}\ +s\ \mathrm{and}\ -s$", 24, 540, y + 156, INK2, a, align="c")
        else:
            lo1, hi1 = min(o["one_bit"] for o in OTH), max(o["one_bit"] for o in OTH)
            lo2, hi2 = min(o["one_bit_retrained"] for o in OTH), max(o["one_bit_retrained"] for o in OTH)
            paste(img, rf"$\mathrm{{{R['retrain_steps']}\ Adam\ steps\ with\ the\ rounding\ inside\ the\ network}}$", 26, 540, y + 92, GREEN, a, align="c")
            paste(img, rf"$\mathrm{{6\ other\ splits:\ 1\ bit}}\ {pc(lo1, 0)}\ \mathrm{{to}}\ {pc(hi1, 0)}\%,\ \ \mathrm{{retrained}}\ {pc(lo2)}\ \mathrm{{to}}\ {pc(hi2)}\%$", 23, 540, y + 152, INK2, a, align="c")
            paste(img, rf"$\mathrm{{the\ {R['n_biases']}\ biases\ stay\ at\ 32\ bits;\ {RT['sign_flips']}\ of\ the\ {f'{NW:,}'.replace(',', '{,}')}\ signs\ flipped}}$", 20, 540, y + h - 22, INK3, a, align="c")

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
        lab = "THE HANDWRITING REEL'S NETWORK   ·   540 UNSEEN DIGITS   ·   ALL COMPUTED"
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
    print("facts", {k: v for k, v in sc.facts.items() if k != "others"})
    for line in SCRIPT:
        print("  >", line)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)[..., ::-1]
        cv2.imwrite(f"{out}/f{fr:04d}.png", im)
        ims.append(cv2.resize(im, (360, 640), interpolation=cv2.INTER_AREA))
        times.append(time.time() - t0)
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)]))
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
