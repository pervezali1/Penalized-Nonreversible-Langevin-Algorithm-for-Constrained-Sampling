"""Delete most of a network (pruning). Dark theme of reels 31-54, maths in Computer Modern (stage3d.tex), narrated by
Kokoro (voicekit). The handwriting network of reel 49 (2,368 weights, 96.7%): half its weights deleted at random
leaves 40%; the smallest half deleted leaves 90%; with 100 steps of retraining and the deleted weights held at zero,
60% deleted reads 96.7% again, 80% deleted 93.1%, and 90% deleted only 77.6% (prunesim.py). On screen: the 32
neurons' weight pictures with the deleted weights in black, the count of weights left, the accuracy against the
share deleted for the three ways, and ten unseen digits with what it reads. frame(f) -> uint8 1080x1920 RGB."""

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
DEAD = "#04050A"

sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "prune_results.json")):
    import prunesim
    prunesim.main()
with open(os.path.join(HERE, "prune_results.json")) as fh:
    R = json.load(fh)
_NPZ = os.path.join(ROOT, "digits", "dg_results.npz")
if not os.path.exists(_NPZ):
    sys.path.insert(0, os.path.join(ROOT, "digits"))
    import dgsim
    dgsim.main()
XTE = np.load(_NPZ)["Xte"]
ROWS = {round(r["frac"], 2): r for r in R["rows"]}
FRACS = [round(r["frac"], 2) for r in R["rows"]]
W1 = np.array(R["W1"])
NW = R["n_weights"]
ACC0 = R["acc"]
SHOW = R["show"]
RND50 = R["random50"]


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (NW, pc(ACC0), pc(RND50["acc"], 0), pc(ROWS[0.5]["smallest"], 0), pc(ROWS[0.6]["retrained"]), pc(ROWS[0.8]["retrained"], 0), pc(ROWS[0.9]["retrained"], 0)) == \
    (2368, "96.7", "40", "90", "96.7", "93", "78"), "script numbers changed"
SCRIPT = [
    "I deleted sixty percent of this network's weights. It still reads ninety-six point seven percent of handwriting, exactly as well as before.",
    "It's the handwriting network: two thousand three hundred and sixty-eight weights, ninety-six point seven percent right on digits it has never seen.",
    "Delete half of them at random, and it falls apart: forty percent.",
    "Delete the smallest half instead, the weights closest to zero, and it still reads ninety percent. Most of its weights were barely doing anything.",
    "Then let the survivors retrain for a hundred steps, with the deleted ones held at zero. Sixty percent gone: ninety-six point seven, the same as the full network. Eighty percent gone: ninety-three.",
    "But there's a limit. Delete ninety percent, and even after retraining it reads only seventy-eight.",
    "That's pruning: one of the tricks that shrink big networks to fit on a phone.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 random, 3 smallest, 4 retrain, 5 the limit, 6 close
SEG_MIN = [6.5, 8.5, 6.0, 10.0, 14.0, 7.5, 6.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_SIXTY = word_t(4, "Sixty percent gone")
T_EIGHTY = word_t(4, "Eighty percent gone")

TILES = (70, 470, 940)               # the 32 neuron pictures: x, y, width (8 across, 4 down)
CHART = (150, 1112, 820, 160)
STRIP = (70, 1346, 940)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("net", "random", "smallest", "retrain", "limit", "close")[k - 1]
    return "close"


def state(t):
    """(share deleted, way: full / random / smallest / retrained)."""
    md = mode(t)
    if md == "net":
        return 0.0, "full"
    if md == "random":
        return 0.5, "random"
    if md == "smallest":
        return 0.5, "smallest"
    if md == "retrain":
        if t < T_SIXTY - 0.2:
            return 0.5, "retrained"
        if t < T_EIGHTY - 0.2:
            return 0.6, "retrained"
        return 0.8, "retrained"
    if md == "limit":
        return 0.9, "retrained"
    return 0.6, "retrained"


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_SIXTY + 1.0) * FPS), int((T_EIGHTY + 1.0) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n_weights=NW, acc=ACC0, random50=RND50["acc"], rows={f: dict(smallest=ROWS[f]["smallest"], retrained=ROWS[f]["retrained"], random=ROWS[f]["random"]) for f in FRACS},
                          segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("60% of its weights: deleted.", "black", 48, INK, 300, 1), (f"Still {pc(ROWS[0.6]['retrained'])}%. Same as before.", "dbold", 44, GREEN, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("The handwriting network.", "black", 50, INK, 300, 1), (f"{NW:,} weights. {pc(ACC0)}% right.", "dbold", 44, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Delete half, at random.", "black", 52, INK, 300, 1), (f"It falls apart: {pc(ROWS[0.5]['random'])}%.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Delete the smallest half.", "black", 50, INK, 300, 1), (f"Still {pc(ROWS[0.5]['smallest'])}%.", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Retrain the survivors.", "black", 52, INK, 300, 1), (f"60% gone: {pc(ROWS[0.6]['retrained'])}%.  80% gone: {pc(ROWS[0.8]['retrained'])}%.", "dbold", 40, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("There's a limit.", "black", 52, INK, 300, 1), (f"90% gone: {pc(ROWS[0.9]['retrained'])}%.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("That's pruning.", "black", 54, INK, 300, 1), ("How big networks get small.", "dbold", 44, GREEN, 390, 0.95)], 10, 8, "pop"),
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

    def weights_now(self, frac, way):
        """The 64 x 32 first-layer weights and which of them are deleted."""
        if way == "full" or frac == 0:
            return W1, np.ones_like(W1, bool)
        if way == "random":
            m = np.array(RND50["mask1"], bool)
            return W1 * m, m
        row = ROWS[frac]
        m = np.array(row["mask1"], bool)
        w = np.array(row["w1_retrained"] if way == "retrained" else row["w1_smallest"])
        return w, m

    def tiles(self, ctx, frac, way, a):
        x0, y0, w = TILES
        gap = 10
        tile = (w - 7 * gap) / 8
        c = tile / 8
        Wn, M = self.weights_now(frac, way)
        base = np.array(rgb(PANEL_BASE))
        dead = np.array(rgb(DEAD))
        for j in range(32):
            r, q = j // 8, j % 8
            tx, ty = x0 + q * (tile + gap), y0 + r * (tile + gap)
            mx = np.abs(W1[:, j]).max() + 1e-12
            for p in range(64):
                if not M[p, j]:
                    col = dead
                else:
                    g = Wn[p, j] / mx
                    k = min(1.0, abs(g)) ** 0.8
                    col = base * (1 - k) + np.array(rgb(VERM if g > 0 else BLUE)) * k
                ctx.set_source_rgba(*col, a)
                ctx.rectangle(tx + (p % 8) * c, ty + (p // 8) * c, c + 0.4, c + 0.4)
                ctx.fill()
            ctx.set_source_rgba(*rgb(INK3), 0.4 * a)
            ctx.set_line_width(1.0)
            ctx.rectangle(tx, ty, tile, tile)
            ctx.stroke()
        lab = "32 NEURONS, THEIR 64 WEIGHTS EACH (BLACK: DELETED)"
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, lab, x0, y0 - 14, 0.1 * 16)
        # the count and the score, under the tiles
        kept = NW if way == "full" or frac == 0 else (RND50["kept"] if way == "random" else ROWS[frac]["kept"])
        acc = ACC0 if way == "full" or frac == 0 else (RND50["acc"] if way == "random" else ROWS[frac][way])
        yb = y0 + 4 * (tile + gap) + 46
        font(ctx, "black", 54)
        ctx.set_source_rgba(*rgb(GOLD), a)
        s = f"{kept:,}"
        ctx.move_to(x0, yb + 20)
        ctx.show_text(s)
        w1 = ctx.text_extents(s).x_advance
        font(ctx, "dbold", 24)
        ctx.set_source_rgba(*rgb(INK2), a)
        ctx.move_to(x0 + w1 + 14, yb + 12)
        ctx.show_text(f"of {NW:,} weights left")
        col = {"full": GREEN, "random": BLUE, "smallest": GOLD, "retrained": GREEN}[way]
        font(ctx, "black", 54)
        s2 = f"{pc(acc)}%"
        ctx.set_source_rgba(*rgb(col), a)
        ctx.move_to(x0 + w - ctx.text_extents(s2).x_advance, yb + 20)
        ctx.show_text(s2)
        font(ctx, "dbold", 20)
        s3 = {"full": "the full network", "random": "half deleted: one random choice", "smallest": "smallest deleted", "retrained": "smallest deleted + retrained"}[way]
        ctx.set_source_rgba(*rgb(INK3), a)
        ctx.move_to(x0 + w - ctx.text_extents(s3).x_advance, yb + 48)
        ctx.show_text(s3)

    def chart(self, ctx, t, frac, way, a):
        x0, y0, w, h = CHART
        md = mode(t)

        def gx(f):
            return x0 + f / 0.95 * w

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
        font(ctx, "dbold", 17)
        ctx.set_source_rgba(*rgb(INK2), a)
        for f, s in ((0.0, "0%"), (0.5, "50%"), (0.9, "90%")):
            ctx.move_to(gx(f) - ctx.text_extents(s).x_advance / 2, y0 + h + 22)
            ctx.show_text(s)
        for v, s in ((0.0, "0"), (0.5, "50%"), (1.0, "100%")):
            ctx.move_to(x0 - 10 - ctx.text_extents(s).x_advance, gy(v) + 6)
            ctx.show_text(s)
        order = ["random", "smallest", "retrained"]
        shown = {"hook": order, "close": order, "net": [], "random": ["random"], "smallest": ["random", "smallest"], "retrain": order, "limit": order}[md]
        cols = dict(random=BLUE, smallest=GOLD, retrained=GREEN)
        names = dict(random="at random", smallest="smallest first", retrained="smallest first + retrain")
        for k in shown:
            upto = 0.95
            if md == way.replace("retrained", "retrain") or (md == "random" and k == "random") or (md == "smallest" and k == "smallest"):
                upto = 0.95
            ctx.set_source_rgba(*rgb(cols[k]), a)
            ctx.set_line_width(3.2)
            pts = [(f, ROWS[f][k]) for f in FRACS if f <= upto + 1e-9]
            for i, (f, v) in enumerate(pts):
                (ctx.move_to if i == 0 else ctx.line_to)(gx(f), gy(v))
            ctx.stroke()
            for f, v in pts:
                ctx.arc(gx(f), gy(v), 4, 0, TAU)
                ctx.fill()
        # where we are
        if way != "full":
            v = RND50["acc"] if way == "random" else ROWS[frac][way]
            ctx.set_source_rgba(1, 1, 1, 0.95 * a)
            ctx.set_line_width(2.2)
            ctx.arc(gx(frac), gy(v), 10, 0, TAU)
            ctx.stroke()
        # legend
        font(ctx, "dbold", 19)
        for j, k in enumerate(shown):
            ctx.set_source_rgba(*rgb(cols[k]), a)
            ctx.move_to(x0 + 20, y0 + h - 60 - 26 * (len(shown) - 1 - j) + 0)
            ctx.show_text(names[k])
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, "RIGHT ON 540 UNSEEN DIGITS, BY SHARE OF WEIGHTS DELETED", x0 - 80, y0 - 20, 0.1 * 16)

    def strip(self, ctx, frac, way, a):
        x0, y0, w = STRIP
        tile, n = 56, 10
        gap = (w - n * tile) / (n - 1)
        if way == "full" or frac == 0:
            reads = list(range(10))
        elif way == "random":
            reads = RND50["reads"]
        else:
            reads = ROWS[frac]["reads_smallest" if way == "smallest" else "reads_retrained"]
        for k in range(10):
            xx = x0 + k * (tile + gap)
            c = tile / 8
            base = np.array(rgb(PANEL_BASE))
            ink = np.array(rgb(INK))
            img = XTE[SHOW[k]]
            for p in range(64):
                v = float(img[p])
                ctx.set_source_rgba(*(base * (1 - v) + ink * v), a)
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
        frac, way = state(t)
        surf, ctx = layer()
        self.tiles(ctx, frac, way, a)
        self.chart(ctx, t, frac, way, a)
        self.strip(ctx, frac, way, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close", "retrain"):
            paste(img, r"$\mathrm{delete\ the}\ w\ \mathrm{with\ the\ smallest}\ |w|,\ \mathrm{then\ retrain\ the\ rest}$", 25, 540, y + 96, GREEN, a, align="c")
            paste(img, rf"$\mathrm{{{R['retrain_steps']}\ Adam\ steps\ on\ the\ 1{{,}}257\ training\ digits;\ deleted\ weights\ held\ at\ 0}}$", 20, 540, y + 156, INK2, a, align="c")
            paste(img, r"$\mathrm{both\ layers\ together:\ 64 \times 32 + 32 \times 10 = 2{,}368\ weights}$", 20, 540, y + h - 18, INK3, a, align="c")
        elif md == "net":
            paste(img, r"$64\ \mathrm{pixels} \;\rightarrow\; 32\ \mathrm{neurons} \;\rightarrow\; 10\ \mathrm{probabilities}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, r"$64 \times 32 + 32 \times 10 = 2{,}368\ \mathrm{weights}$", 28, 540, y + 162, GOLD, a, align="c")
        elif md == "random":
            r = ROWS[0.5]["random_range"]
            paste(img, rf"$\mathrm{{half\ deleted\ at\ random:\ median\ of\ {R['n_random']}\ tries}}\ {pc(ROWS[0.5]['random'])}\%\ \ ({pc(r[0], 0)}\ \mathrm{{to}}\ {pc(r[1], 0)}\%)$", 22, 540, y + 100, BLUE, a, align="c")
            paste(img, r"$\mathrm{the\ big\ weights\ go\ as\ often\ as\ the\ tiny\ ones}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif md == "smallest":
            paste(img, r"$\mathrm{delete\ every}\ w\ \mathrm{with}\ |w| < \mathrm{the\ median}\ |w|$", 30, 540, y + 100, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{1{{,}}184\ weights\ left:}}\ {pc(ROWS[0.5]['smallest'])}\%,\ \ \mathrm{{no\ retraining}}$", 24, 540, y + 162, INK2, a, align="c")
        else:
            paste(img, rf"$\mathrm{{90\%\ deleted\ (237\ weights\ left):}}\ {pc(ROWS[0.9]['retrained'])}\%;\ \ \mathrm{{95\%\ deleted:}}\ {pc(ROWS[0.95]['retrained'])}\%$", 23, 540, y + 100, VERM, a, align="c")
            paste(img, r"$\mathrm{too\ few\ weights\ left\ to\ hold\ what\ it\ learned}$", 22, 540, y + h - 26, INK2, a, align="c")

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
    print("facts", {k: v for k, v in sc.facts.items() if k != "rows"})
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
