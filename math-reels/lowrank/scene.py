"""Compress a photo with the SVD (low rank). Dark theme of reels 31-59, maths in Computer Modern (stage3d.tex),
narrated by Kokoro (voicekit). A 300 x 451 colour photo (scikit-image's CC0 "chelsea") is 405,900 numbers; the SVD
splits each colour into pieces, each a column times a row with a strength, strongest first. 1 piece: plaid; 5: blurry
shapes; 20: the cat, from 11.1% of the numbers; 50: 27.8% of the numbers, off by 5.2 shades of 255 on average
(svdsim.py). On screen: the photo rebuilt from k pieces, the first three pieces, the numbers kept, and the strengths
of all 300 pieces. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import cv2
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
import svdsim  # noqa: E402
if not os.path.exists(os.path.join(HERE, "svd_results.json")):
    svdsim.main()
with open(os.path.join(HERE, "svd_results.json")) as fh:
    R = json.load(fh)
ROWS = {r["k"]: r for r in R["rows"]}
NUM = R["numbers"]
STR = [np.array(s) for s in R["strengths"]]
M_, N_ = R["shape"][:2]
BREAK_EVEN = NUM / (3 * (M_ + N_ + 1))


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (NUM, M_, N_, pc(1 - ROWS[20]["share"], 0), pc(ROWS[20]["share"], 0), pc(ROWS[50]["share"], 0), round(ROWS[50]["rms"])) == \
    (405900, 300, 451, "89", "11", "28", 5), "script numbers changed"
SCRIPT = [
    "I threw away eighty-nine percent of the numbers in this photo. It's still obviously a cat.",
    "To a computer, a photo is a grid of numbers: three hundred rows, four hundred and fifty-one columns, three colors. Four hundred and five thousand nine hundred numbers.",
    "Linear algebra can split any grid into simple pieces. Each one is just a column times a row, with a strength. That's the singular value decomposition, and the pieces come sorted, strongest first.",
    "Keep only the strongest piece: plaid. Five pieces: blurry shapes. Twenty pieces: the cat, from eleven percent of the numbers.",
    "Fifty pieces, twenty-eight percent of the numbers, and the average pixel is off by only five shades out of two hundred and fifty-five. The strengths drop fast, so the first pieces carry most of the photo.",
    "That's a low-rank approximation. The same idea, called LoRA, fine-tunes giant AI models: instead of changing a huge matrix, it learns a few columns times rows.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the photo, 2 the pieces, 3 keep 1, 5, 20, 4 keep 50, 5 close
SEG_MIN = [6.5, 10.0, 12.0, 10.0, 12.0, 10.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[6]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_K1 = word_t(3, "Keep only")
T_K5 = word_t(3, "Five pieces")
T_K20 = word_t(3, "Twenty pieces")
T_K50 = word_t(4, "Fifty pieces")
T_DROP = word_t(4, "The strengths")
T_P = [SEG[2] + 0.6, SEG[2] + 1.5, SEG[2] + 2.4]
FADE = 0.4

IMG = (70, 470, 940, int(round(940 * M_ / N_)))
INFO_Y = 1160
PLOT = (150, 1250, 840, 120)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 6):
        if t < SEG[k + 1]:
            return ("photo", "pieces", "keep", "more", "close")[k - 1]
    return "close"


def view(t):
    """(previous k, current k, cross-fade); k = 0 is the original photo."""
    md = mode(t)
    if md in ("hook", "close"):
        return 20, 20, 1.0
    if md in ("photo", "pieces"):
        return 0, 0, 1.0
    if md == "keep":
        if t < T_K1:
            return 0, 0, 1.0
        if t < T_K5:
            return 0, 1, smooth(seg(t, T_K1, T_K1 + FADE))
        if t < T_K20:
            return 1, 5, smooth(seg(t, T_K5, T_K5 + FADE))
        return 5, 20, smooth(seg(t, T_K20, T_K20 + FADE))
    if t < T_K50:
        return 20, 20, 1.0
    return 20, 50, smooth(seg(t, T_K50, T_K50 + FADE))


def qa_frames():
    fr = []
    for k in range(6):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_K1 + 1.0) * FPS), int((T_K5 + 1.0) * FPS), int((T_K20 + 1.0) * FPS), int((T_K50 + 1.0) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(numbers=NUM, rows=R["rows"], segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        im, _, _ = svdsim.photo()
        X, svds = svdsim.decompose(im)
        x0, y0, w, h = IMG
        fit = lambda A: cv2.resize(np.clip(A, 0, 255).astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)  # noqa: E731
        self.pics = {0: fit(X)}
        for k in (1, 5, 20, 50):
            self.pics[k] = fit(svdsim.rebuild(svds, k))
        tw = 286
        th = int(round(tw * M_ / N_))
        small = lambda A: cv2.resize(np.clip(A, 0, 255).astype(np.float32) / 255.0, (tw, th), interpolation=cv2.INTER_AREA).clip(0, 1)  # noqa: E731
        self.pieces = [small(svdsim.piece(svds, 0))] + [small(128 + 2 * svdsim.piece(svds, i)) for i in (1, 2)]
        sw = 600
        sh = int(round(sw * M_ / N_))
        self.sum3 = cv2.resize(np.clip(svdsim.rebuild(svds, 3), 0, 255).astype(np.float32) / 255.0, (sw, sh), interpolation=cv2.INTER_CUBIC).clip(0, 1)
        self.tw, self.th, self.sw, self.sh = tw, th, sw, sh

    def _cues(self):
        S_ = SEG
        r20, r50 = ROWS[20], ROWS[50]
        hook = [(f"{pc(1 - r20['share'], 0)}% of the numbers: gone.", "black", 52, INK, 300, 1), ("Still obviously a cat.", "dbold", 46, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("A photo is a grid of numbers.", "black", 48, INK, 300, 1), (f"300 \u00d7 451 \u00d7 3 = {NUM:,}.", "dbold", 46, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Split it into pieces:", "black", 52, INK, 300, 1), ("each a column \u00d7 a row.", "dbold", 46, CYAN, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Keep the strongest pieces.", "black", 50, INK, 300, 1), (f"20 pieces: {pc(r20['share'])}% of the numbers.", "dbold", 42, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"50 pieces: {pc(r50['share'])}%.", "black", 52, INK, 300, 1), (f"Off by {r50['rms']:.1f} shades of 255.", "dbold", 44, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[5], T_END - 0.3, [("That's low rank.", "black", 54, INK, 300, 1), ("The idea behind LoRA.", "dbold", 46, GREEN, 390, 0.95)], 10, 8, "pop"),
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

    @staticmethod
    def put(img, pic, x, y, a=1.0):
        h, w = pic.shape[:2]
        sub = img[y:y + h, x:x + w]
        sub[:] = sub * (1 - a) + pic * a

    def photo(self, img, ctx, t, a):
        md = mode(t)
        x0, y0, w, h = IMG
        if md == "pieces":
            gap = (w - 3 * self.tw) // 2
            for i in range(3):
                ap = smooth(seg(t, T_P[i], T_P[i] + 0.5)) * a
                if ap <= 0:
                    continue
                px = x0 + i * (self.tw + gap)
                self.put(img, self.pieces[i], px, y0 + 34, ap)
                ctx.set_source_rgba(*rgb(INK3), 0.7 * ap)
                ctx.set_line_width(1.5)
                ctx.rectangle(px, y0 + 34, self.tw, self.th)
                ctx.stroke()
                font(ctx, "dbold", 21)
                ctx.set_source_rgba(*rgb(CYAN), ap)
                s = f"piece {i + 1}"
                ctx.move_to(px + self.tw / 2 - ctx.text_extents(s).x_advance / 2, y0 + 20)
                ctx.show_text(s)
                if i < 2:
                    font(ctx, "black", 34)
                    ctx.set_source_rgba(*rgb(INK2), ap)
                    ctx.move_to(px + self.tw + gap / 2 - 10, y0 + 34 + self.th / 2 + 12)
                    ctx.show_text("+")
            font(ctx, "dbold", 18)
            ctx.set_source_rgba(*rgb(INK3), a)
            s = "pieces 2 and 3 can be negative: drawn on gray (gray = 0)"
            ctx.move_to(540 - ctx.text_extents(s).x_advance / 2, y0 + 34 + self.th + 30)
            ctx.show_text(s)
            a3 = smooth(seg(t, T_P[2] + 0.8, T_P[2] + 1.4)) * a
            if a3 > 0:
                sx, sy = 540 - self.sw // 2, y0 + 34 + self.th + 64
                self.put(img, self.sum3, sx, sy, a3)
                font(ctx, "dbold", 21)
                ctx.set_source_rgba(*rgb(GOLD), a3)
                s = "= the first 3 pieces added up"
                ctx.move_to(540 - ctx.text_extents(s).x_advance / 2, sy + self.sh + 30)
                ctx.show_text(s)
            return
        ka, kb, k = view(t)
        pic = self.pics[kb] if k >= 1 else self.pics[ka] * (1 - k) + self.pics[kb] * k
        self.put(img, pic, x0, y0, a)
        ctx.set_source_rgba(*rgb(INK3), 0.6 * a)
        ctx.set_line_width(1.5)
        ctx.rectangle(x0, y0, w, h)
        ctx.stroke()

    def info(self, ctx, t, a):
        md = mode(t)
        if md == "pieces":
            return
        ka, kb, k = view(t)
        kk = kb if k >= 0.5 else ka
        x0, w = IMG[0], IMG[2]
        y = INFO_Y
        font(ctx, "black", 54)
        if kk == 0:
            s = "the original"
            ctx.set_source_rgba(*rgb(INK), a)
        else:
            s = f"{kk} piece" + ("s" if kk > 1 else "")
            ctx.set_source_rgba(*rgb(GOLD), a)
        ctx.move_to(x0, y)
        ctx.show_text(s)
        n = NUM if kk == 0 else ROWS[kk]["numbers"]
        font(ctx, "black", 40)
        s2 = f"{n:,}"
        ctx.set_source_rgba(*rgb(INK if kk == 0 else GOLD), a)
        x2 = x0 + w - 330
        ctx.move_to(x2, y - 6)
        ctx.show_text(s2)
        font(ctx, "dbold", 20)
        ctx.set_source_rgba(*rgb(INK2), a)
        s3 = "numbers" if kk == 0 else f"numbers ({pc(ROWS[kk]['share'])}% of {NUM:,})"
        ctx.move_to(x2, y + 22)
        ctx.show_text(s3)
        if kk:
            font(ctx, "dbold", 20)
            ctx.set_source_rgba(*rgb(INK3), a)
            ctx.move_to(x0, y + 30)
            ctx.show_text(f"average error: {ROWS[kk]['rms']:.1f} shades of 255")

    def plot(self, ctx, t, a):
        x0, y0, w, h = PLOT
        md = mode(t)
        ka, kb, k = view(t)
        kk = kb if k >= 0.5 else ka
        lo, hi = 0.0, 5.0                                   # log10 of the strength
        gx = lambda i: x0 + (i - 1) / (len(STR[0]) - 1) * w  # noqa: E731
        gy = lambda v: y0 + h - (math.log10(max(v, 1.0)) - lo) / (hi - lo) * h  # noqa: E731
        if kk and md != "pieces":
            ctx.set_source_rgba(*rgb(GOLD), 0.14 * a)
            ctx.rectangle(x0, y0, gx(kk) - x0, h)
            ctx.fill()
            ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
            ctx.set_line_width(2.0)
            ctx.move_to(gx(kk), y0 - 4)
            ctx.line_to(gx(kk), y0 + h)
            ctx.stroke()
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        for c, col in zip(range(3), (VERM, GREEN, BLUE)):
            ctx.set_source_rgba(*rgb(col), 0.9 * a)
            ctx.set_line_width(2.0)
            for i, v in enumerate(STR[c]):
                (ctx.move_to if i == 0 else ctx.line_to)(gx(i + 1), gy(v))
            ctx.stroke()
        font(ctx, "dbold", 16)
        ctx.set_source_rgba(*rgb(INK2), a)
        for v, s in ((1, "1"), (100, "100"), (10000, "10,000")):
            ctx.move_to(x0 - 10 - ctx.text_extents(s).x_advance, gy(v) + 5)
            ctx.show_text(s)
        for i in (1, 100, 200, 300):
            s = str(i)
            ctx.move_to(gx(i) - ctx.text_extents(s).x_advance / 2, y0 + h + 20)
            ctx.show_text(s)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "STRENGTH OF EACH OF THE 300 PIECES (LOG SCALE; RED, GREEN, BLUE)", x0 - 80, y0 - 14, 0.1 * 16)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        surf, ctx = layer()
        self.photo(img, ctx, t, a)
        self.info(ctx, t, a)
        self.plot(ctx, t, a)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "CHELSEA THE CAT (SCIKIT-IMAGE SAMPLE PHOTO, CC0)", IMG[0], IMG[1] - 14, 0.1 * 16)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close", "keep"):
            paste(img, r"$X \;\approx\; \sigma_1 u_1 v_1^{\top} + \sigma_2 u_2 v_2^{\top} + \cdots + \sigma_k u_k v_k^{\top}$", 32, 540, y + 96, GOLD, a, align="c")
            paste(img, r"$k\ \mathrm{pieces\ of\ 3\ colors:}\ 3 \times k \times (300 + 451 + 1)\ \mathrm{numbers}$", 23, 540, y + 160, INK2, a, align="c")
            paste(img, r"$k = 20:\ 45{,}120\ \mathrm{numbers\ instead\ of}\ 405{,}900$", 21, 540, y + h - 20, INK3, a, align="c")
        elif md == "photo":
            paste(img, r"$300 \times 451\ \mathrm{pixels} \times 3\ \mathrm{colors} = 405{,}900\ \mathrm{numbers}$", 30, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{each\ from\ 0\ (dark)\ to\ 255\ (bright)}$", 24, 540, y + 162, INK2, a, align="c")
        elif md == "pieces":
            paste(img, r"$\sigma_i\, u_i\, v_i^{\top}:\ \mathrm{a\ column}\ u_i\ (300\ \mathrm{numbers})\ \times\ \mathrm{a\ row}\ v_i^{\top}\ (451)$", 26, 540, y + 96, CYAN, a, align="c")
            paste(img, r"$\mathrm{strength}\ \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{300} \geq 0\ \ \mathrm{(the\ singular\ values)}$", 23, 540, y + 158, INK2, a, align="c")
            paste(img, r"$\mathrm{all\ 300\ pieces\ added\ up\ give\ back\ the\ photo\ exactly}$", 20, 540, y + h - 20, INK3, a, align="c")
        elif md == "more":
            r = ROWS
            paste(img, rf"$\mathrm{{average\ error:\ 1\ piece}}\ {r[1]['rms']:.1f},\ \ 5:\ {r[5]['rms']:.1f},\ \ 20:\ {r[20]['rms']:.1f},\ \ 50:\ {r[50]['rms']:.1f}\ \mathrm{{shades}}$", 22, 540, y + 96, GREEN, a, align="c")
            paste(img, rf"$\sigma_1 = {STR[0][0]:,.0f}\ \ \mathrm{{but}}\ \ \sigma_{{50}} = {STR[0][49]:,.0f}\ \ \mathrm{{(red)}}$".replace(",", "{,}"), 26, 540, y + 158, INK2, a, align="c")
            paste(img, rf"$\mathrm{{past\ {math.ceil(BREAK_EVEN)}\ pieces\ it\ takes\ more\ numbers\ than\ the\ photo\ itself}}$", 20, 540, y + h - 20, INK3, a, align="c")
        else:
            paste(img, r"$W \;\rightarrow\; W + B\,A,\qquad B:\ d \times r,\ \ A:\ r \times d,\ \ r\ \mathrm{small}$", 30, 540, y + 96, GREEN, a, align="c")
            paste(img, r"$\mathrm{LoRA\ (Low\ Rank\ Adaptation):\ train\ only}\ B\ \mathrm{and}\ A$", 24, 540, y + 160, INK2, a, align="c")

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
        lab = "THE SINGULAR VALUE DECOMPOSITION   \u00b7   ALL COMPUTED"
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
