"""How many examples does a network need? Dark theme of reels 31-55, maths in Computer Modern (stage3d.tex), narrated
by Kokoro (voicekit). The handwriting network of reel 49 trained on 1, 2, 3, 5, 10, 20 and 50 examples of each digit
and on all 1,257 (fewsim.py; each size the median of 10 random picks): 10 pictures -> 58.7%, 100 -> 89.4%, 500 ->
95.1%, 1,257 -> 96.7%. On screen: the training set itself, the accuracy against the number of pictures (log scale,
with the spread over the 10 picks), and ten unseen digits with what the network reads. frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "few_results.json")):
    import fewsim
    fewsim.main()
with open(os.path.join(HERE, "few_results.json")) as fh:
    R = json.load(fh)
sys.path.insert(0, os.path.join(ROOT, "digits"))
import dgsim  # noqa: E402
_X, _Y = dgsim.load()
_TR = np.array(R["train_index"])
XTR, YTR = _X[_TR], _Y[_TR]
_NPZ = os.path.join(ROOT, "digits", "dg_results.npz")
if not os.path.exists(_NPZ):
    dgsim.main()
XTE = np.load(_NPZ)["Xte"]
SIZES = {s["n"]: s for s in R["sizes"]}
NS_ = [s["n"] for s in R["sizes"]]
NFULL = R["n_train"]
SHOW = R["show"]


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


GAIN1 = SIZES[100]["median"] - SIZES[10]["median"]
GAIN2 = SIZES[NFULL]["median"] - SIZES[500]["median"]
assert (pc(SIZES[10]["median"], 0), pc(SIZES[100]["median"], 0), pc(SIZES[500]["median"], 0), pc(SIZES[NFULL]["median"]), NFULL, round(100 * GAIN1), round(100 * GAIN2, 1)) == \
    ("59", "89", "95", "96.7", 1257, 31, 1.6), "script numbers changed"
SCRIPT = [
    "This network learned to read from just ten pictures, one of each digit. It still gets fifty-nine percent of new handwriting right.",
    "It's the network from the handwriting reel. Normally it learns from twelve hundred and fifty-seven digits. So what happens with fewer?",
    "One example of each digit: fifty-nine percent. It gets all ten of its own pictures right, every time. It just has no idea how differently other people write.",
    "Ten of each, a hundred pictures: eighty-nine percent. Fifty of each: ninety-five.",
    "All twelve hundred and fifty-seven: ninety-six point seven. Going from ten pictures to a hundred bought thirty-one points. The last seven hundred and fifty-seven bought less than two.",
    "More data always helps. But each new picture helps a little less than the one before.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 one of each, 3 more, 4 all of them, 5 close
SEG_MIN = [7.0, 8.5, 10.0, 7.5, 13.0, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[6]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_FIFTY = word_t(3, "Fifty of each")
T_GAIN1, T_GAIN2 = word_t(4, "Going from"), word_t(4, "The last")

MOSAIC = (70, 470, 940, 420)
CHART = (150, 1010, 820, 190)
STRIP = (70, 1290, 940)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 6):
        if t < SEG[k + 1]:
            return ("net", "one", "more", "all", "close")[k - 1]
    return "close"


def n_at(t):
    md = mode(t)
    if md in ("hook", "one"):
        return 10
    if md == "more":
        return 100 if t < T_FIFTY - 0.2 else 500
    return NFULL


def qa_frames():
    fr = []
    for k in range(6):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_FIFTY + 1.0) * FPS), int((T_GAIN2 + 1.5) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(sizes={n: (SIZES[n]["median"], SIZES[n]["lo"], SIZES[n]["hi"]) for n in NS_}, gain_10_100=GAIN1, gain_500_all=GAIN2,
                          segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("10 pictures. One of each digit.", "black", 46, INK, 300, 1), (f"{pc(SIZES[10]['median'])}% on new handwriting.", "dbold", 44, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("The handwriting network.", "black", 50, INK, 300, 1), (f"Usually: {NFULL:,} pictures, {pc(SIZES[NFULL]['median'])}%.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("One of each digit.", "black", 52, INK, 300, 1), (f"Its own 10: 100%. New ones: {pc(SIZES[10]['median'])}%.", "dbold", 40, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("More pictures.", "black", 54, INK, 300, 1), (f"100: {pc(SIZES[100]['median'])}%.   500: {pc(SIZES[500]['median'])}%.", "dbold", 44, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"All {NFULL:,}: {pc(SIZES[NFULL]['median'])}%.", "black", 52, GREEN, 300, 1), (f"The last 757 bought {pc(GAIN2)} points.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], T_END - 0.3, [("More data always helps.", "black", 50, INK, 300, 1), ("Each picture, a little less.", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
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
        self._mos = {}

    # ------------------------------------------------------------ pieces

    def mosaic_img(self, n):
        """The training set of size n as a picture (cached): k per digit in rows, or all of them in a grid."""
        if n in self._mos:
            return self._mos[n]
        x0, y0, w, h = MOSAIC
        pick = SIZES[n]["shown"]["pick"]
        surf, ctx = layer()
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        if n < NFULL:
            k = n // 10
            cols, rows = (5, 2) if k == 1 else (k, 10)
            s = min(w / cols, h / rows)
            items = [(r, q, pick[r * k + q] if k > 1 else pick[r * 5 + q]) for r in range(rows) for q in range(cols)]
        else:
            order = np.argsort(YTR, kind="stable")
            cols = int(math.ceil(math.sqrt(n * w / h)))
            rows = int(math.ceil(n / cols))
            s = min(w / cols, h / rows)
            items = [(i // cols, i % cols, int(order[i])) for i in range(n)]
        ox = x0 + (w - cols * s) / 2
        c = s / 8
        pad = max(0.5, s * 0.06)
        for r, q, i in items:
            img = XTR[i]
            for p in range(64):
                v = float(img[p])
                if v <= 0.01:
                    continue
                ctx.set_source_rgba(*(base * (1 - v) + ink * v), 1.0)
                ctx.rectangle(ox + q * s + pad + (p % 8) * (s - 2 * pad) / 8, y0 + r * s + pad + (p // 8) * (s - 2 * pad) / 8, (s - 2 * pad) / 8 + 0.3, (s - 2 * pad) / 8 + 0.3)
            ctx.fill()
        trgb, ta = surf_rgba(surf)
        self._mos[n] = (trgb, ta)
        return self._mos[n]

    def label(self, ctx, s, x, y, a, col=INK3):
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        show_tracked(ctx, s, x, y, 0.1 * 16)

    def chart(self, ctx, t, n, a):
        x0, y0, w, h = CHART
        lo, hi = math.log10(10), math.log10(NFULL)

        def gx(m):
            return x0 + (math.log10(m) - lo) / (hi - lo) * w

        def gy(v):
            return y0 + h - (v - 0.5) / 0.5 * h
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
        font(ctx, "dbold", 17)
        ctx.set_source_rgba(*rgb(INK2), a)
        for m in (10, 20, 50, 100, 200, 500, NFULL):
            s = f"{m:,}"
            ctx.move_to(gx(m) - ctx.text_extents(s).x_advance / 2, y0 + h + 22)
            ctx.show_text(s)
        for v, s in ((0.5, "50%"), (0.75, "75%"), (1.0, "100%")):
            ctx.move_to(x0 - 10 - ctx.text_extents(s).x_advance, gy(v) + 6)
            ctx.show_text(s)
        shown = [m for m in NS_ if m <= n]
        # the spread over the 10 picks, then the medians
        for m in shown:
            sz = SIZES[m]
            ctx.set_source_rgba(*rgb(GREEN), 0.35 * a)
            ctx.set_line_width(6)
            ctx.move_to(gx(m), gy(sz["lo"]))
            ctx.line_to(gx(m), gy(sz["hi"]))
            ctx.stroke()
        ctx.set_source_rgba(*rgb(GREEN), a)
        ctx.set_line_width(3.2)
        for i, m in enumerate(shown):
            (ctx.move_to if i == 0 else ctx.line_to)(gx(m), gy(SIZES[m]["median"]))
        ctx.stroke()
        for m in shown:
            ctx.arc(gx(m), gy(SIZES[m]["median"]), 5, 0, TAU)
            ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.95 * a)
        ctx.set_line_width(2.2)
        ctx.arc(gx(n), gy(SIZES[n]["median"]), 10, 0, TAU)
        ctx.stroke()
        md = mode(t)
        if md in ("all", "close", "hook") and n == NFULL or (md == "all"):
            # the two gains, as brackets
            for (m1, m2, tt, col) in ((10, 100, T_GAIN1, GOLD), (500, NFULL, T_GAIN2, VERM)):
                al = a * (smooth(seg(t, tt - 0.2, tt + 0.6)) if md == "all" else 1.0)
                if al <= 0.01:
                    continue
                g = SIZES[m2]["median"] - SIZES[m1]["median"]
                ctx.set_source_rgba(*rgb(col), al)
                ctx.set_line_width(2.0)
                xm = gx(m2) + 14
                ctx.move_to(xm, gy(SIZES[m1]["median"]))
                ctx.line_to(xm, gy(SIZES[m2]["median"]))
                ctx.stroke()
                font(ctx, "dbold", 21)
                s = f"+{pc(g)} points"
                tx = xm + 8 if m2 != NFULL else xm - 8 - ctx.text_extents(s).x_advance - 28
                ty = (gy(SIZES[m1]["median"]) + gy(SIZES[m2]["median"])) / 2 + 7 + (24 if m2 == NFULL else 0)
                ctx.move_to(tx, ty)
                ctx.show_text(s)
        self.label(ctx, "RIGHT ON 540 UNSEEN DIGITS, BY NUMBER OF TRAINING PICTURES", x0 - 80, y0 - 20, a)

    def strip(self, ctx, n, a):
        x0, y0, w = STRIP
        tile, k = 56, 10
        gap = (w - k * tile) / (k - 1)
        reads = SIZES[n]["shown"]["reads"]
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        for d in range(10):
            xx = x0 + d * (tile + gap)
            c = tile / 8
            img = XTE[SHOW[d]]
            for p in range(64):
                v = float(img[p])
                ctx.set_source_rgba(*(base * (1 - v) + ink * v), a)
                ctx.rectangle(xx + (p % 8) * c + 0.5, y0 + (p // 8) * c + 0.5, c - 1, c - 1)
                ctx.fill()
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(GREEN if reads[d] == d else VERM), a)
            ctx.move_to(xx + tile + 6, y0 + tile / 2 + 10)
            ctx.show_text(str(reads[d]))
        self.label(ctx, "TEN UNSEEN DIGITS: WHAT IT READS", x0, y0 - 14, a)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        n = n_at(t)
        mrgb, ma = self.mosaic_img(n)
        over(img, mrgb * a, ma * a)
        surf, ctx = layer()
        x0, y0, w, h = MOSAIC
        lab = f"THE TRAINING SET: {n:,} PICTURES" + (f" ({n // 10} OF EACH DIGIT)" if n < NFULL else " (ALL OF THEM)")
        self.label(ctx, lab, x0, y0 - 14, a)
        # the score of this training set
        sz = SIZES[n]
        font(ctx, "black", 52)
        s = f"{pc(sz['median'])}%"
        ctx.set_source_rgba(*rgb(GREEN if n >= 100 else GOLD), a)
        ctx.move_to(x0 + w - ctx.text_extents(s).x_advance, y0 + h + 58)
        ctx.show_text(s)
        font(ctx, "dbold", 21)
        ctx.set_source_rgba(*rgb(INK3), a)
        s2 = "right on 540 new digits (median of 10 picks)" if n < NFULL else "right on 540 new digits"
        ctx.move_to(x0, y0 + h + 50)
        ctx.show_text(s2)
        self.chart(ctx, t, n, a)
        self.strip(ctx, n, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        S_ = SIZES
        if md in ("hook", "close"):
            paste(img, rf"$10 \rightarrow {pc(S_[10]['median'])}\%\quad 100 \rightarrow {pc(S_[100]['median'])}\%\quad 500 \rightarrow {pc(S_[500]['median'])}\%\quad 1{{,}}257 \rightarrow {pc(S_[NFULL]['median'])}\%$", 25, 540, y + 100, GREEN, a, align="c")
            paste(img, r"$\mathrm{pictures} \rightarrow \mathrm{right\ on\ 540\ new\ digits\ (median\ of\ 10\ random\ picks)}$", 21, 540, y + 162, INK2, a, align="c")
            paste(img, r"$\mathrm{the\ same\ network,\ start\ and\ 300\ training\ steps\ every\ time}$", 20, 540, y + h - 18, INK3, a, align="c")
        elif md == "net":
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 32\ \mathrm{neurons} \;\rightarrow\; 10\ \mathrm{probabilities}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\mathrm{{trained\ on\ all\ {NFULL:,}\ training\ digits:}}\ {pc(S_[NFULL]['median'])}\%$".replace(",", "{,}", 1), 26, 540, y + 162, GREEN, a, align="c")
        elif md == "one":
            paste(img, rf"$\mathrm{{its\ own\ 10\ pictures:}}\ 100\%\qquad \mathrm{{540\ new\ ones:}}\ {pc(S_[10]['median'])}\%$", 28, 540, y + 100, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{over\ 10\ random\ picks\ of\ the\ 10\ pictures:}}\ {pc(S_[10]['lo'])}\ \mathrm{{to}}\ {pc(S_[10]['hi'])}\%$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "more":
            paste(img, rf"$100:\ {pc(S_[100]['median'])}\%\ \ ({pc(S_[100]['lo'])}{{-}}{pc(S_[100]['hi'])})\qquad 500:\ {pc(S_[500]['median'])}\%\ \ ({pc(S_[500]['lo'])}{{-}}{pc(S_[500]['hi'])})$", 26, 540, y + 100, GREEN, a, align="c")
            paste(img, r"$\mathrm{it\ still\ gets\ every\ one\ of\ its\ own\ training\ pictures\ right}$", 22, 540, y + h - 26, INK2, a, align="c")
        else:
            paste(img, rf"$10 \rightarrow 100\ \mathrm{{pictures:}}\ +{pc(GAIN1)}\ \mathrm{{points}}\qquad 500 \rightarrow 1{{,}}257:\ +{pc(GAIN2)}\ \mathrm{{points}}$", 23, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{each\ tenfold\ more\ pictures\ buys\ fewer\ points}$", 22, 540, y + h - 26, INK2, a, align="c")

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
    print("facts", {k: v for k, v in sc.facts.items()})
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
