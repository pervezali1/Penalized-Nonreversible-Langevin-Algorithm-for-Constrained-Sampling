"""Fix the one-pixel problem without touching the network (data augmentation). Dark theme of reels 31-61, maths in
Computer Modern (stage3d.tex), narrated by Kokoro (voicekit). The handwriting network of reel 49 reads 96.7% of the
540 unseen digits but only 46.5% when every digit moves one pixel right (reel 52). Trained the same way on five copies
of each training picture (as it is, and moved one pixel left, right, up and down) it reads 94.3% of the moved digits
and still 96.5% of the unmoved ones; moved two pixels, a move it never saw, 52.6% (augsim.py). On screen: a training
picture and its four shifted copies, the score on each kind of move for both networks, and ten moved digits with
what each network reads. frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "aug_results.json")):
    import augsim
    augsim.main()
with open(os.path.join(HERE, "aug_results.json")) as fh:
    R = json.load(fh)
for _d in ("digits", "convolution", "nearest"):
    sys.path.insert(0, os.path.join(ROOT, _d))
from cnnsim import shift  # noqa: E402
from nnsim import split  # noqa: E402
import dgsim  # noqa: E402

XTR, YTR, XTE, YTE = split(dgsim.SPLIT_SEED)
SHOW = R["show"]
TESTS = R["tests"]
PL, AU = R["plain"], R["aug"]
OTH = R["others"]
_THREES = np.flatnonzero(YTR == 3)
TRAIN3 = int(_THREES[np.argmin(((XTR[_THREES] - XTR[_THREES].mean(0)) ** 2).sum(1))])   # the most typical training 3


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (pc(PL["unmoved"]), pc(PL["1 right"], 0), R["n_aug"], pc(AU["1 right"], 0), pc(AU["1 left"], 0), pc(AU["1 up"], 0), pc(AU["1 down"], 0),
        pc(AU["unmoved"]), pc(AU["2 right"], 0), R["steps"]) == ("96.7", "46", 6285, "94", "94", "93", "95", "96.5", "53", 300), "script numbers changed"
SCRIPT = [
    "This network failed when a digit moved one pixel. I fixed it without changing a single line of the network. I just changed what it saw.",
    "It's the handwriting network: ninety-six point seven percent. But move every digit one pixel to the right, and it reads only forty-six percent.",
    "So I gave it five copies of every training picture: as it is, and moved one pixel left, right, up and down. Same network, same three hundred steps of training.",
    "Now, moved one pixel right: ninety-four percent. Left: ninety-four. Up and down: ninety-three and ninety-five. And unmoved digits: still ninety-six point five.",
    "But it only learns what it's shown. Move a digit two pixels, a move it never saw, and it drops to fifty-three percent.",
    "That's data augmentation: when you can't get more data, make more from what you have. Image models are routinely trained on shifted, flipped and cropped copies.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the problem, 2 the copies, 3 the result, 4 the limit, 5 close
SEG_MIN = [7.0, 9.0, 10.5, 11.0, 8.5, 10.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[6]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_MOVE = word_t(1, "But move")
T_COPIES = word_t(2, "as it is")
T_RES = [word_t(3, "Now, moved"), word_t(3, "Left"), word_t(3, "Up and down"), word_t(3, "Up and down") + 0.6, word_t(3, "And unmoved")]
T_TWO = word_t(4, "Move a digit two")

PANEL_Y = 470
CHART = (150, 860, 830, 230)
STRIP_Y = 1206
CARD = (70, 1430, 940, 250)
SHIFTS = [("unmoved", 0, 0), ("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)]


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 6):
        if t < SEG[k + 1]:
            return ("problem", "copies", "result", "limit", "close")[k - 1]
    return "close"


def qa_frames():
    fr = []
    for k in range(6):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_MOVE + 1.0) * FPS), int((T_RES[4] + 1.0) * FPS), int((T_TWO + 1.2) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(plain=PL, aug=AU, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("Same network. Different pictures.", "black", 48, INK, 300, 1), (f"Moved digits: {pc(PL['1 right'], 0)}% → {pc(AU['1 right'], 0)}%.", "dbold", 46, GREEN, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [(f"{pc(PL['unmoved'])}% on unseen digits.", "black", 50, INK, 300, 1), (f"Moved 1 pixel: {pc(PL['1 right'], 0)}%.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("5 copies of every picture.", "black", 50, INK, 300, 1), ("Same network, same 300 steps.", "dbold", 44, CYAN, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [(f"Moved 1 pixel: {pc(AU['1 right'], 0)}%.", "black", 54, INK, 300, 1), (f"Unmoved: still {pc(AU['unmoved'])}%.", "dbold", 46, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("It learns only what it's shown.", "black", 48, INK, 300, 1), (f"Moved 2 pixels: {pc(AU['2 right'], 0)}%.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[5], T_END - 0.3, [("That's data augmentation.", "black", 52, INK, 300, 1), ("More data from the data you have.", "dbold", 42, GREEN, 390, 0.95)], 10, 8, "pop"),
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
    def label(ctx, s, x, y, a, col=INK3, size=16, center=False):
        font(ctx, "bold", size)
        ctx.set_source_rgba(*rgb(col), 0.95 * a)
        tr = 0.1 * size
        if center:
            x -= tracked_width(ctx, s, tr) / 2
        show_tracked(ctx, s, x, y, tr)

    @staticmethod
    def digit(ctx, x0, y0, s, img8, a, frame_col=INK3, fw=1.6):
        c = s / 8
        base = np.array(rgb(PANEL_BASE))
        ink = np.array(rgb(INK))
        for p in range(64):
            v = float(img8[p])
            ctx.set_source_rgba(*(base * (1 - v) + ink * v), a)
            ctx.rectangle(x0 + (p % 8) * c, y0 + (p // 8) * c, c + 0.4, c + 0.4)
            ctx.fill()
        ctx.set_source_rgba(*rgb(frame_col), 0.8 * a)
        ctx.set_line_width(fw)
        ctx.rectangle(x0, y0, s, s)
        ctx.stroke()

    def panel(self, ctx, t, a):
        md = mode(t)
        y0 = PANEL_Y
        if md in ("copies",) or (md in ("hook", "close", "result", "limit")):
            # a training picture and its four shifted copies
            ap = a if md != "copies" else smooth(seg(t, T_COPIES - 0.4, T_COPIES + 0.4)) * a
            self.label(ctx, "ONE TRAINING PICTURE AND ITS 4 SHIFTED COPIES", 540, y0 + 6, a, center=True)
            s = 130
            gap = 34
            x0 = 540 - (5 * s + 4 * gap) / 2
            x = TRAIN3
            for k, (name, dx, dy) in enumerate(SHIFTS):
                al = a if k == 0 else ap
                img = shift(XTR[x:x + 1], dx, dy)[0]
                xx = x0 + k * (s + gap)
                self.digit(ctx, xx, y0 + 34, s, img, al, CYAN if k else INK2, 2.0)
                font(ctx, "dbold", 20)
                lab = name if k == 0 else f"1 px {name}"
                ctx.set_source_rgba(*rgb(CYAN if k else INK2), al)
                ctx.move_to(xx + s / 2 - ctx.text_extents(lab).x_advance / 2, y0 + 34 + s + 30)
                ctx.show_text(lab)
            font(ctx, "dbold", 21)
            ctx.set_source_rgba(*rgb(INK3), ap)
            s2 = f"{R['n_train']:,} pictures × 5 = {R['n_aug']:,} to train on"
            ctx.move_to(540 - ctx.text_extents(s2).x_advance / 2, y0 + 34 + s + 70)
            ctx.show_text(s2)
            return
        # the problem: one unseen 3, as it is and moved one pixel right, and what the network reads
        s = 200
        i = SHOW[3]
        moved = shift(XTE[i:i + 1], 1, 0)[0]
        am = smooth(seg(t, T_MOVE - 0.2, T_MOVE + 0.5)) * a
        self.label(ctx, "AN UNSEEN 3", 290, y0 + 6, a, center=True)
        self.digit(ctx, 190, y0 + 30, s, XTE[i], a)
        r0 = R["reads"]["unmoved"]["plain"][3]
        font(ctx, "dbold", 24)
        ctx.set_source_rgba(*rgb(GREEN), a)
        lab = f"reads {r0}"
        ctx.move_to(290 - ctx.text_extents(lab).x_advance / 2, y0 + 30 + s + 36)
        ctx.show_text(lab)
        if am > 0:
            self.label(ctx, "THE SAME 3, 1 PIXEL RIGHT", 790, y0 + 6, am, center=True)
            self.digit(ctx, 690, y0 + 30, s, moved, am, VERM)
            r1 = R["reads"]["1 right"]["plain"][3]
            ctx.set_source_rgba(*rgb(VERM), am)
            lab = f"reads {r1}"
            font(ctx, "dbold", 24)
            ctx.move_to(790 - ctx.text_extents(lab).x_advance / 2, y0 + 30 + s + 36)
            ctx.show_text(lab)
            font(ctx, "black", 40)
            ctx.set_source_rgba(*rgb(INK2), am)
            ctx.move_to(520, y0 + 30 + s / 2 + 14)
            ctx.show_text("→")

    def chart(self, ctx, t, a):
        md = mode(t)
        x0, y0, w, h = CHART
        names = ["unmoved", "1 right", "1 left", "1 up", "1 down", "2 right", "2 left"]
        labels = ["unmoved", "1 right", "1 left", "1 up", "1 down", "2 right", "2 left"]
        gw = w / len(names)
        bw = gw * 0.34

        def gy(v):
            return y0 + h - v * h
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        ctx.set_source_rgba(1, 1, 1, 0.07 * a)
        for v in (0.25, 0.5, 0.75, 1.0):
            ctx.move_to(x0, gy(v))
            ctx.line_to(x0 + w, gy(v))
        ctx.stroke()
        font(ctx, "dbold", 16)
        ctx.set_source_rgba(*rgb(INK2), a)
        for v, s in ((0.0, "0"), (0.5, "50%"), (1.0, "100%")):
            ctx.move_to(x0 - 10 - ctx.text_extents(s).x_advance, gy(v) + 6)
            ctx.show_text(s)
        # how much of each series to show
        if md == "problem":
            plain_on = [1.0, smooth(seg(t, T_MOVE, T_MOVE + 0.6))] + [smooth(seg(t, T_MOVE + 0.4, T_MOVE + 1.0))] * 5
            aug_on = [0.0] * 7
        elif md == "copies":
            plain_on, aug_on = [1.0] * 7, [0.0] * 7
        elif md == "result":
            plain_on = [1.0] * 7
            aug_on = [smooth(seg(t, T_RES[0], T_RES[0] + 0.5)), smooth(seg(t, T_RES[1], T_RES[1] + 0.5)), smooth(seg(t, T_RES[2], T_RES[2] + 0.5)),
                      smooth(seg(t, T_RES[3], T_RES[3] + 0.5)), smooth(seg(t, T_RES[4], T_RES[4] + 0.5)), 0.0, 0.0]
            aug_on = [aug_on[4], aug_on[0], aug_on[1], aug_on[2], aug_on[3], 0.0, 0.0]
        elif md == "limit":
            g = smooth(seg(t, T_TWO, T_TWO + 0.6))
            plain_on, aug_on = [1.0] * 7, [1.0] * 5 + [g, g]
        else:
            plain_on, aug_on = [1.0] * 7, [1.0] * 7
        for k, nm in enumerate(names):
            cx = x0 + (k + 0.5) * gw
            for v, on, col, off in ((PL[nm], plain_on[k], BLUE, -bw - 2), (AU[nm], aug_on[k], GREEN, 2)):
                if on <= 0:
                    continue
                hh = h * v * on
                ctx.set_source_rgba(*rgb(col), 0.92 * a)
                ctx.rectangle(cx + off, y0 + h - hh, bw, hh)
                ctx.fill()
                if on > 0.9:
                    font(ctx, "dbold", 15)
                    s = f"{pc(v, 0)}"
                    ctx.set_source_rgba(*rgb(col), a)
                    ctx.move_to(cx + off + bw / 2 - ctx.text_extents(s).x_advance / 2, y0 + h - hh - 6)
                    ctx.show_text(s)
            font(ctx, "dbold", 16)
            ctx.set_source_rgba(*rgb(VERM if nm.startswith("2") and md in ("limit", "close", "hook") else INK2), a)
            s = labels[k]
            ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, y0 + h + 22)
            ctx.show_text(s)
        self.label(ctx, "RIGHT ON THE 540 UNSEEN DIGITS, MOVED BY (PIXELS)", x0 - 80, y0 - 58, a)
        font(ctx, "dbold", 18)
        ctx.set_source_rgba(*rgb(BLUE), a)
        ctx.move_to(x0 - 80, y0 - 28)
        s1 = "■ trained on the pictures"
        ctx.show_text(s1)
        if md not in ("problem", "copies"):
            ctx.set_source_rgba(*rgb(GREEN), a)
            ctx.move_to(x0 - 80 + ctx.text_extents(s1).x_advance + 40, y0 - 28)
            ctx.show_text("■ trained on the 5 copies of each")

    def strip(self, ctx, t, a):
        md = mode(t)
        if md == "problem" and t < T_MOVE:
            return
        y0 = STRIP_Y
        tile, n = 52, 10
        x0, w = 70, 940
        gap = (w - n * tile) / (n - 1)
        rows = [("plain", BLUE, "trained on the pictures")]
        if md not in ("problem", "copies"):
            rows.append(("aug", GREEN, "trained on the 5 copies"))
        self.label(ctx, "TEN UNSEEN DIGITS, MOVED 1 PIXEL RIGHT: WHAT EACH NETWORK READS", x0, y0 - 14, a)
        for k in range(10):
            xx = x0 + k * (tile + gap)
            img = shift(XTE[SHOW[k]:SHOW[k] + 1], 1, 0)[0]
            self.digit(ctx, xx, y0, tile, img, a, INK3, 1.0)
            for r, (key, col, _) in enumerate(rows):
                v = R["reads"]["1 right"][key][k]
                font(ctx, "black", 26)
                ctx.set_source_rgba(*rgb(GREEN if v == k else VERM), a)
                s = str(v)
                ctx.move_to(xx + tile / 2 - ctx.text_extents(s).x_advance / 2, y0 + tile + 34 + 34 * r)
                ctx.show_text(s)
        parts = []
        for r, (key, col, name) in enumerate(rows):
            right = sum(1 for k in range(10) if R["reads"]["1 right"][key][k] == k)
            ctx.set_source_rgba(*rgb(col), a)
            ctx.arc(x0 - 26, y0 + tile + 26 + 34 * r, 7, 0, TAU)
            ctx.fill()
            parts.append((f"{name}: {right} of 10 right", col))
        font(ctx, "dbold", 18)
        xx = x0
        for s, col in parts:
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(xx, y0 + tile + 34 + 34 * len(rows) + 4)
            ctx.show_text(s)
            xx += ctx.text_extents(s).x_advance + 40

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        surf, ctx = layer()
        self.panel(ctx, t, a)
        self.chart(ctx, t, a)
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
        lo = min(o["plain"]["1 right"] for o in OTH), max(o["plain"]["1 right"] for o in OTH)
        hi = min(o["aug"]["1 right"] for o in OTH), max(o["aug"]["1 right"] for o in OTH)
        if md in ("hook", "close", "result"):
            paste(img, rf"$\mathrm{{1\ px\ right:}}\ {pc(PL['1 right'])}\% \rightarrow {pc(AU['1 right'])}\%;\ \ \mathrm{{unmoved:}}\ {pc(PL['unmoved'])}\% \rightarrow {pc(AU['unmoved'])}\%$", 24, 540, y + 96, GREEN, a, align="c")
            paste(img, rf"$\mathrm{{6\ other\ random\ splits,\ 1\ pixel\ right:}}\ {pc(lo[0], 0)}\mathrm{{-}}{pc(lo[1], 0)}\%\ \rightarrow\ {pc(hi[0], 0)}\mathrm{{-}}{pc(hi[1], 0)}\%$", 23, 540, y + 156, INK2, a, align="c")
            paste(img, r"$\mathrm{same\ network,\ same\ start,\ same\ 300\ steps;\ only\ the\ pictures\ differ}$", 20, 540, y + h - 22, INK3, a, align="c")
        elif md == "problem":
            paste(img, r"$\mathrm{each\ pixel\ position\ has\ its\ own\ weights\ in\ this\ network}$", 24, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{move\ the\ ink\ and\ it\ lands\ on\ weights\ that\ never\ saw\ it}$", 24, 540, y + 162, VERM, a, align="c")
        elif md == "copies":
            paste(img, r"$x'(r,\ c) = x(r - \Delta r,\ c - \Delta c),\quad (\Delta r,\ \Delta c) \in \{(0,0),\ (0,\pm1),\ (\pm1,0)\}$", 20, 540, y + 100, CYAN, a, align="c")
            paste(img, r"$\mathrm{what\ slides\ in\ at\ the\ edge\ is\ blank;\ each\ copy\ keeps\ its\ label}$", 22, 540, y + 162, INK2, a, align="c")
        else:
            paste(img, rf"$\mathrm{{moved\ 2\ pixels\ right:\ plain}}\ {pc(PL['2 right'])}\%,\ \ \mathrm{{augmented}}\ {pc(AU['2 right'])}\%$", 26, 540, y + 100, VERM, a, align="c")
            paste(img, r"$\mathrm{shifts\ of\ 2\ were\ never\ in\ the\ training\ copies}$", 23, 540, y + 162, INK2, a, align="c")

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
