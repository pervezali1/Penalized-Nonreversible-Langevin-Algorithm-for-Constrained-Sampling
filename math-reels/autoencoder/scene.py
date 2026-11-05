"""Two numbers per digit (an autoencoder). Dark theme of reels 31-53, maths in Computer Modern (stage3d.tex), narrated
by Kokoro (voicekit). The UCI handwritten digits of reel 49: a network squeezes each one to 2 numbers and rebuilds it
from them, trained only to rebuild (no labels). On screen: the 540 unseen digits as points on the 2-number map,
blob at the start and sorted by digit at the end (colours added afterwards), the same map from PCA (reel 43), the
5-nearest-neighbour score of both, rebuilt digits, and the decoder's pictures across the map (aesim.py).
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
DIGIT_COLS = ["#FF7A4D", "#5B8DFF", "#34D399", "#F472B6", "#F6C56E", "#38BDF8", "#A78BFA", "#F87171", "#A3E635", "#E5E7EB"]

sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "ae_results.npz")):          # the archive is not committed: rebuild it (a few minutes)
    import aesim
    aesim.main()
with open(os.path.join(HERE, "ae_results.json")) as fh:
    R = json.load(fh)
D = np.load(os.path.join(HERE, "ae_results.npz"))
YTE, XTE = D["yte"], D["Xte"]
AE, PCA, SNAPS, GRID = D["ae_codes"], D["pca_codes"], D["snaps"], D["grid"]
AE_REC, PCA_REC = D["ae_rec"], D["pca_rec"]
SHOW = R["show"]
KNN_P, KNN_A = R["pca"]["knn"], R["ae"]["knn"]
MSE_P, MSE_A = R["pca"]["mse"], R["ae"]["mse"]
KS = sorted(R["knn_by_seed"].values())


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (round(100 * KNN_P), round(100 * KNN_A), round(100 * (1 - MSE_A / MSE_P))) == (63, 72, 24), "script numbers changed"
SCRIPT = [
    "This network squeezes every handwritten digit into just two numbers. Nobody ever told it which digit is which.",
    "Sixty-four pixels go in. Two numbers in the middle. Sixty-four pixels come out. Its only job is to rebuild the digit it was given.",
    "So every digit becomes one point on a map. At the start, the map is a blob. Then backpropagation trains it to rebuild better, and better.",
    "Now colour each point by its real digit. The network never saw these labels, and still the zeros, the fours and the sixes each found their own corner. The fives, sevens and eights still crowd the middle: two numbers isn't much room.",
    "PCA, the straight-line squeeze from the flowers reel, does a decent job too. A simple test: do a digit's five nearest neighbours on the map name it? PCA: sixty-three percent. The network, with its bends: seventy-two.",
    "And the map works backwards. Pick any point, and the second half of the network draws the digit that lives there.",
    "Two numbers, learned from nothing but the pictures. That's an autoencoder.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 the map forms, 3 the colours, 4 against PCA, 5 the map backwards, 6 close
SEG_MIN = [6.0, 9.0, 10.0, 13.0, 14.0, 9.0, 6.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_TRAIN = (word_t(2, "Then backpropagation") - 0.2, SEG[3] - 0.4)
T_COL = word_t(3, "Now colour")
T_PCA = word_t(4, "PCA, the")
T_TEST = word_t(4, "A simple test")
T_BACK = word_t(5, "Pick any")

MAP = (70, 470, 600)                  # the map panel: x, y, size
RX = 700                              # the right-hand column
STRIP = (70, 1150, 940)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("net", "train", "colour", "pca", "back", "close")[k - 1]
    return "close"


def snap_at(t):
    md = mode(t)
    if md == "train":
        return int(round((len(SNAPS) - 1) * smooth(seg(t, *T_TRAIN)) ** 1.6))
    return len(SNAPS) - 1


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_TEST + 2.0) * FPS), int((T_BACK + 1.5) * FPS), N_FRAMES - 1]


def _box(C):
    lo, hi = np.percentile(C, 1, 0), np.percentile(C, 99, 0)
    c, s = (lo + hi) / 2, (hi - lo).max() * 0.56 + 1e-9
    return c, s


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(knn_pca=KNN_P, knn_ae=KNN_A, mse_pca=MSE_P, mse_ae=MSE_A, var_kept_pca=R["pca"]["var_kept"], seed=R["seed"],
                          knn_by_seed=KS, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        self.box_ae, self.box_pca = _box(AE), _box(PCA)
        self.box_snap = [_box(s) for s in SNAPS]

    def _cues(self):
        S_ = SEG
        hook = [("Every digit, squeezed to 2 numbers.", "black", 44, INK, 300, 1), ("It was never told which is which.", "dbold", 42, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("64 pixels → 2 numbers → 64 pixels.", "black", 44, INK, 300, 1), ("Its only job: rebuild the digit.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Each digit: one point on a map.", "black", 46, INK, 300, 1), ("Training sorts the map.", "dbold", 42, MAGENTA, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Colours added only now.", "black", 50, INK, 300, 1), ("It never saw a label.", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("PCA: a straight-line squeeze.", "black", 46, INK, 300, 1), (f"Neighbours right: {pc(KNN_P)}% vs {pc(KNN_A)}%.", "dbold", 40, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The map works backwards.", "black", 50, INK, 300, 1), ("Any point → a digit.", "dbold", 44, MAGENTA, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("Two numbers, learned", "black", 50, INK, 300, 1), ("from nothing but pictures.", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
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

    def panel(self, ctx, a):
        x0, y0, n = MAP
        ctx.set_source_rgba(*rgb(PANEL_BASE), 0.85 * a)
        S3.pill(ctx, x0, y0, n, n, 18)
        ctx.fill()
        ctx.set_source_rgba(*rgb(INK2), 0.5 * a)
        ctx.set_line_width(1.4)
        S3.pill(ctx, x0, y0, n, n, 18)
        ctx.stroke()

    def points(self, ctx, C, box, a, colour=1.0, labels=0.0):
        x0, y0, n = MAP
        c, s = box
        P = (C - c) / s                                     # about -0.9 .. 0.9
        px = x0 + n / 2 + P[:, 0] * n / 2
        py = y0 + n / 2 - P[:, 1] * n / 2
        ctx.save()
        S3.pill(ctx, x0, y0, n, n, 18)
        ctx.clip()
        white = np.array(rgb(INK2))
        for i in range(len(C)):
            col = white * (1 - colour) + np.array(rgb(DIGIT_COLS[int(YTE[i])])) * colour
            ctx.set_source_rgba(*col, 0.85 * a)
            ctx.arc(px[i], py[i], 4.2, 0, TAU)
            ctx.fill()
        if labels > 0:
            font(ctx, "black", 34)
            for d in range(10):
                m = YTE == d
                mx, my = np.median(px[m]), np.median(py[m])
                s_ = str(d)
                w_ = ctx.text_extents(s_).x_advance
                ctx.set_source_rgba(0.02, 0.03, 0.06, 0.75 * labels * a)
                ctx.arc(mx, my - 12, 22, 0, TAU)
                ctx.fill()
                ctx.set_source_rgba(*rgb(DIGIT_COLS[d]), labels * a)
                ctx.move_to(mx - w_ / 2, my)
                ctx.show_text(s_)
        ctx.restore()

    def mosaic(self, ctx, a):
        x0, y0, n = MAP
        k = 9
        cell = (n - 40) / k
        for r in range(k):
            for q in range(k):
                self.pixels(ctx, x0 + 20 + q * cell + 3, y0 + 20 + r * cell + 3, cell - 6, GRID[r, q], a, frame=False)

    def bottleneck(self, ctx, t, a, i):
        """The right-hand column: a digit, its 2 numbers, and the digit rebuilt from them."""
        x = RX
        self.label(ctx, "THE TRAINED NETWORK:", x, 464, a)
        self.label(ctx, "AN UNSEEN DIGIT", x, 486, a)
        self.pixels(ctx, x, 500, 150, XTE[i], a)
        font(ctx, "dbold", 26)
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(x + 168, 590)
        ctx.show_text("64 pixels")
        cx, cy = AE[i]
        font(ctx, "black", 34)
        ctx.set_source_rgba(*rgb(GOLD), a)
        ctx.move_to(x, 720)
        ctx.show_text(f"({cx:.2f}, {cy:.2f})")
        font(ctx, "dbold", 22)
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(x, 752)
        ctx.show_text("2 numbers")
        self.label(ctx, "REBUILT FROM THEM", x, 806, a)
        self.pixels(ctx, x, 820, 150, AE_REC[i], a)
        font(ctx, "dbold", 26)
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(x + 168, 910)
        ctx.show_text("64 pixels")
        # arrows
        ctx.set_source_rgba(*rgb(INK3), 0.8 * a)
        ctx.set_line_width(2.0)
        for ya, yb in ((660, 690), (762, 792)):
            ctx.move_to(x + 75, ya)
            ctx.line_to(x + 75, yb - 6)
            ctx.stroke()
            ctx.move_to(x + 75, yb)
            ctx.line_to(x + 69, yb - 10)
            ctx.line_to(x + 81, yb - 10)
            ctx.close_path()
            ctx.fill()

    def scores(self, ctx, t, a):
        x = RX
        self.label(ctx, "DO A DIGIT'S 5 NEAREST", x, 486, a)
        self.label(ctx, "NEIGHBOURS NAME IT?", x, 508, a)
        u1 = smooth(seg(t, T_TEST + 0.6, T_TEST + 1.6))
        u2 = smooth(seg(t, T_TEST + 2.2, T_TEST + 3.2))
        for j, (lab, v, col, u) in enumerate((("PCA", KNN_P, INK2, u1), ("the network", KNN_A, GREEN, u2))):
            yy = 580 + 150 * j
            font(ctx, "dbold", 26)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(x, yy)
            ctx.show_text(lab)
            font(ctx, "black", 64)
            ctx.set_source_rgba(*rgb(col), a * u)
            ctx.move_to(x, yy + 70)
            ctx.show_text(f"{pc(v)}%")
        font(ctx, "dbold", 21)
        ctx.set_source_rgba(*rgb(INK3), a)
        for j, s in enumerate(("of the 540 unseen digits,", "on the 2-number maps", f"(networks from 9 starts: {pc(KS[0], 0)}–{pc(KS[-1], 0)}%)")):
            ctx.move_to(x, 900 + 30 * j)
            ctx.show_text(s)

    def strip(self, ctx, t, a, show_pca):
        x0, y0, w = STRIP
        tile, n = 70, 10
        gap = (w - n * tile) / (n - 1)
        rec = PCA_REC if show_pca else AE_REC
        for k in range(10):
            i = SHOW[k]
            xx = x0 + k * (tile + gap)
            self.pixels(ctx, xx, y0, tile, XTE[i], a, frame=False)
            self.pixels(ctx, xx, y0 + tile + 14, tile, rec[i], a, frame=False)
        self.label(ctx, "TEN UNSEEN DIGITS (TOP), REBUILT FROM 2 NUMBERS " + ("BY PCA" if show_pca else "BY THE NETWORK") + " (BELOW)", x0, y0 - 14, a)

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        surf, ctx = layer()
        self.panel(ctx, a)
        x0, y0, n = MAP
        if md == "back":
            u = smooth(seg(t, T_BACK - 0.3, T_BACK + 0.9))
            self.points(ctx, AE, self.box_ae, a * (1 - u), colour=1.0)
            self.mosaic(ctx, a * u)
            self.label(ctx, "THE MAP, BACKWARDS: THE DIGIT AT EACH POINT", x0, y0 - 14, a)
        elif md == "pca":
            u = smooth(seg(t, T_PCA - 0.3, T_PCA + 0.9))
            if u < 1:
                self.points(ctx, AE, self.box_ae, a * (1 - u), colour=1.0, labels=1.0)
            self.points(ctx, PCA, self.box_pca, a * u, colour=1.0, labels=1.0)
            self.label(ctx, "THE SAME DIGITS, SQUEEZED BY PCA", x0, y0 - 14, a)
        elif md in ("train", "net"):
            k = snap_at(t)
            self.points(ctx, SNAPS[k], self.box_snap[k], a, colour=0.0)
            lab = f"540 UNSEEN DIGITS AS POINTS · TRAINING STEP {k * (R['steps'] // (len(SNAPS) - 1)):,}" if md == "train" else "540 UNSEEN DIGITS ON THE NETWORK'S 2-NUMBER MAP"
            self.label(ctx, lab, x0, y0 - 14, a)
        else:
            col = smooth(seg(t, T_COL - 0.2, T_COL + 1.0)) if md == "colour" else 1.0
            self.points(ctx, AE, self.box_ae, a, colour=col, labels=col)
            self.label(ctx, "540 UNSEEN DIGITS ON THE NETWORK'S 2-NUMBER MAP", x0, y0 - 14, a)
        if md == "pca":
            self.scores(ctx, t, a)
        else:
            i = SHOW[3] if md == "hook" else SHOW[int(t / 1.4) % 10]          # fixed in the hook, so the loop closes
            self.bottleneck(ctx, t, a, i)
        self.strip(ctx, t, a, show_pca=(md == "pca"))
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close", "colour"):
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 64 \;\rightarrow\; 2\ \mathrm{numbers} \;\rightarrow\; 64 \;\rightarrow\; 64\ \mathrm{pixels}$", 28, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{trained\ only\ to\ rebuild;\ labels\ only\ colour\ the\ map,\ afterwards}$", 22, 540, y + 162, GOLD, a, align="c")
            paste(img, r"$\mathrm{UCI\ handwritten\ digits:\ 1{,}257\ to\ train\ on,\ 540\ unseen}$", 20, 540, y + h - 18, INK3, a, align="c")
        elif md == "net":
            paste(img, r"$L = \frac{1}{64}\sum_{p=1}^{64}\left(\hat{x}_p - x_p\right)^2$", 38, 540, y + 116, INK, a, align="c")
            paste(img, r"$x:\ \mathrm{the\ digit},\quad \hat{x}:\ \mathrm{what\ comes\ out}.\ \ \mathrm{No\ labels\ anywhere}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "train":
            paste(img, r"$\mathrm{backpropagation\ +\ Adam:\ 3{,}000\ steps,\ 1{,}257\ training\ digits}$", 22, 540, y + 100, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{each\ map\ rescaled\ to\ fit\ the\ panel}$", 21, 540, y + h - 26, INK3, a, align="c")
        elif md == "pca":
            paste(img, rf"$\mathrm{{PCA}}:\ {pc(KNN_P)}\%\qquad \mathrm{{the\ network}}:\ {pc(KNN_A)}\%$", 30, 540, y + 96, GREEN, a, align="c")
            paste(img, rf"$\mathrm{{PCA\ keeps\ {pc(R['pca']['var_kept'], 0)}\%\ of\ the\ spread;\ the\ network\ rebuilds\ with}}\ {pc(1 - MSE_A / MSE_P, 0)}\%\ \mathrm{{less\ error}}$", 20, 540, y + 156, INK2, a, align="c")
            paste(img, rf"$\mathrm{{the\ run\ shown\ is\ the\ median\ of\ 9\ random\ starts\ ({pc(KS[0], 0)}\ to\ {pc(KS[-1], 0)}\%)}}$", 20, 540, y + h - 18, INK3, a, align="c")
        else:
            paste(img, r"$\mathrm{a\ 9 \times 9\ grid\ of\ points\ on\ the\ map,\ fed\ to\ the\ second\ half}$", 24, 540, y + 100, MAGENTA, a, align="c")
            paste(img, r"$(u, v) \;\rightarrow\; 64 \;\rightarrow\; 64\ \mathrm{pixels}$", 30, 540, y + 166, INK, a, align="c")

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
        lab = "REAL HANDWRITING   ·   NO LABELS IN TRAINING   ·   ALL COMPUTED"
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
