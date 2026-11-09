"""Ten networks, the same mistakes. Dark theme of reels 31-62, maths in Computer Modern (stage3d.tex), narrated by
Kokoro (voicekit). The handwriting network trained ten times from ten random starts reads 519 to 523 of the 540
unseen digits; their vote reads 523, no better than the best of them. Ten digits fool all ten networks (eight of them
with one shared wrong answer); nearest neighbour (reel 57) reads eight of those ten correctly (enssim.py). On screen:
the ten digits as a challenge, every network's answer, the real answers, the scores, and nearest neighbour's answers.
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
if not os.path.exists(os.path.join(HERE, "ens_results.json")):
    import enssim
    enssim.main()
with open(os.path.join(HERE, "ens_results.json")) as fh:
    R = json.load(fh)
FOOL = R["fooled"]
SR = R["single_right"]
NT = R["n_test"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

assert (len(FOOL), R["unanimous"], min(SR), max(SR), R["vote_right"], R["nn_right"], NT, [f["true"] for f in FOOL]) == \
    (10, 8, 519, 523, 523, 8, 540, [1, 6, 8, 8, 4, 3, 3, 6, 5, 9]), "script numbers changed"
ANSWERS = ", ".join(WORDS[f["true"]] for f in FOOL)
SCRIPT = [
    "These ten handwritten digits fooled ten different neural networks. Every single one of them. Can you read them?",
    "I trained the same network ten times, from ten random starts. Alone, each one reads between five hundred and nineteen and five hundred and twenty-three of the five hundred and forty unseen digits.",
    "So I let them vote. The vote gets five hundred and twenty-three right: no better than the best network on its own.",
    "Because they don't make different mistakes. They all learned from the same pictures. On these ten digits, all ten networks were wrong, and on eight of them, all ten gave the same wrong answer.",
    f"Here are the real answers: {ANSWERS}. How many did you get?",
    "The lazy nearest neighbor method, which just compares pixels, reads eight of these ten correctly. A vote only helps when the voters make different mistakes.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook (the challenge), 1 ten networks, 2 the vote, 3 the same mistakes, 4 the answers, 5 nearest neighbour
SEG_MIN = [7.5, 11.0, 8.0, 11.5, 9.0, 11.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[6]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_VOTE = word_t(2, "The vote gets")
T_SAME = word_t(3, "On these ten")
T_ANS0, T_ANS1 = word_t(4, ANSWERS[:8]), word_t(4, "How many")
T_REVEAL = [T_ANS0 + (T_ANS1 - T_ANS0) * k / 10 for k in range(10)]
T_NN = word_t(5, "reads eight")

TILE, GAPX = 150, 40
GX0 = (W - (5 * TILE + 4 * GAPX)) // 2
GY = (510, 800)
PLOT = (140, 1180, 800, 150)
CARD = (70, 1430, 940, 250)
LO, HI = 515, 526


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 6):
        if t < SEG[k + 1]:
            return ("nets", "vote", "same", "answers", "nn")[k - 1]
    return "nn"


def qa_frames():
    fr = []
    for k in range(6):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_VOTE + 1.0) * FPS), int((T_ANS1 + 0.5) * FPS), int((T_NN + 1.0) * FPS), N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(single=SR, vote=R["vote_right"], fooled=[(f["true"], f["top"], f["top_count"], f["nn"]) for f in FOOL], segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("10 digits that fooled", "black", 54, INK, 300, 1), ("10 neural networks. Can you?", "dbold", 46, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("One network, 10 random starts.", "black", 48, INK, 300, 1), (f"{min(SR)} to {max(SR)} of {NT} right.", "dbold", 46, BLUE, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Let them vote:", "black", 54, INK, 300, 1), (f"{R['vote_right']}. No better than the best.", "dbold", 46, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("They make the same mistakes.", "black", 48, INK, 300, 1), ("Same pictures, same blind spots.", "dbold", 44, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("The real answers.", "black", 54, INK, 300, 1), ("How many did you get?", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[5], T_END - 0.3, [(f"Nearest neighbor: {R['nn_right']} of 10.", "black", 50, INK, 300, 1), ("Votes need different mistakes.", "dbold", 44, GREEN, 390, 0.95)], 10, 8, "pop"),
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
        ctx.set_source_rgba(*rgb(frame_col), 0.85 * a)
        ctx.set_line_width(fw)
        ctx.rectangle(x0, y0, s, s)
        ctx.stroke()

    def grid(self, ctx, t, a):
        md = mode(t)
        show_nets = md in ("same", "answers", "nn") and (md != "same" or t >= T_SAME - 0.2)
        self.label(ctx, "10 OF THE 540 UNSEEN DIGITS", 540, GY[0] - 18, a, center=True)
        for k, f in enumerate(FOOL):
            r, c = divmod(k, 5)
            x, y = GX0 + c * (TILE + GAPX), GY[r]
            revealed = (md == "nn") or (md == "answers" and t >= T_REVEAL[k])
            fc = GREEN if revealed else INK3
            self.digit(ctx, x, y, TILE, f["pixels"], a, fc, 2.0 if revealed else 1.6)
            cx = x + TILE / 2
            yy = y + TILE + 32
            if show_nets:
                font(ctx, "dbold", 21)
                s = f"nets: {f['top']} (×{f['top_count']})"
                ctx.set_source_rgba(*rgb(VERM), a)
                ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, yy)
                ctx.show_text(s)
            if revealed:
                font(ctx, "black", 28)
                s = f"it's a {f['true']}"
                ap = a if md != "answers" else smooth(seg(t, T_REVEAL[k], T_REVEAL[k] + 0.25)) * a
                ctx.set_source_rgba(*rgb(GREEN), ap)
                ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, yy + 32)
                ctx.show_text(s)
            elif not show_nets or md in ("hook", "nets", "vote"):
                font(ctx, "black", 30)
                ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
                ctx.move_to(cx - ctx.text_extents("?").x_advance / 2, yy + 8)
                ctx.show_text("?")
            if md == "nn" and t >= T_NN - 0.3:
                ok = f["nn"] == f["true"]
                font(ctx, "dbold", 19)
                s = f"nearest: {f['nn']}"
                ap = smooth(seg(t, T_NN - 0.3, T_NN + 0.3)) * a
                ctx.set_source_rgba(*rgb(CYAN if ok else VERM), ap)
                ctx.move_to(cx - ctx.text_extents(s).x_advance / 2, yy + 62)
                ctx.show_text(s)

    def plot(self, ctx, t, a):
        md = mode(t)
        x0, y0, w, h = PLOT

        def gx(v):
            return x0 + (v - LO) / (HI - LO) * w
        ctx.set_source_rgba(*rgb(INK3), 0.8 * a)
        ctx.set_line_width(1.2)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        font(ctx, "dbold", 17)
        ctx.set_source_rgba(*rgb(INK2), a)
        for v in range(LO, HI + 1):
            ctx.move_to(gx(v), y0 + h)
            ctx.line_to(gx(v), y0 + h + 6)
            if v % 2 == 1:
                s = str(v)
                ctx.move_to(gx(v) - ctx.text_extents(s).x_advance / 2, y0 + h + 26)
                ctx.show_text(s)
        ctx.stroke()
        self.label(ctx, f"DIGITS READ RIGHT, OUT OF {NT}", x0 - 60, y0 - 20, a)
        stack = {}
        for k, v in enumerate(SR):
            ap = a if md not in ("nets",) else smooth(seg(t, SEG[1] + 0.6 + 0.25 * k, SEG[1] + 0.9 + 0.25 * k)) * a
            n = stack.get(v, 0)
            stack[v] = n + 1
            cy = y0 + h - 20 - n * 30
            ctx.set_source_rgba(*rgb(BLUE), ap)
            ctx.arc(gx(v), cy, 11, 0, TAU)
            ctx.fill()
        font(ctx, "dbold", 18)
        ctx.set_source_rgba(*rgb(BLUE), a)
        ctx.move_to(x0, y0 + 16)
        ctx.show_text("● one network (10 starts)")
        if md in ("vote", "same", "answers", "nn", "hook") and (md != "vote" or t >= T_VOTE - 0.2):
            av = a if md != "vote" else smooth(seg(t, T_VOTE - 0.2, T_VOTE + 0.4)) * a
            v = R["vote_right"]
            ctx.set_source_rgba(*rgb(GOLD), av)
            ctx.set_line_width(4)
            ctx.move_to(gx(v), y0 - 4)
            ctx.line_to(gx(v), y0 + h)
            ctx.stroke()
            ctx.move_to(x0, y0 + 44)
            ctx.show_text("│ the vote of all 10")

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        surf, ctx = layer()
        self.grid(ctx, t, a)
        self.plot(ctx, t, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        med = int(R["median_right"])
        if md in ("hook",):
            paste(img, r"$\mathrm{all\ 10\ networks\ got\ every\ one\ of\ these\ wrong}$", 30, 540, y + 100, GOLD, a, align="c")
            paste(img, r"$\mathrm{pause\ and\ read\ them;\ the\ answers\ are\ coming}$", 24, 540, y + 162, INK2, a, align="c")
        elif md == "nets":
            paste(img, r"$64 \rightarrow 32 \rightarrow 10,\ \mathrm{300\ steps,\ the\ same\ 1{,}257\ training\ digits}$", 25, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\mathrm{{only\ the\ random\ start\ differs\ (seeds\ 1\ to\ 10);\ median\ {med}\ right}}$", 22, 540, y + 162, BLUE, a, align="c")
        elif md == "vote":
            paste(img, r"$\mathrm{the\ vote:\ each\ digit\ gets\ the\ answer\ most\ networks\ give}$", 26, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\mathrm{{vote}}\ {R['vote_right']},\ \ \mathrm{{best\ single}}\ {max(SR)},\ \ \mathrm{{median\ single}}\ {med}$", 26, 540, y + 162, GOLD, a, align="c")
        elif md == "same":
            paste(img, rf"$\mathrm{{{R['wrong_any']}\ digits\ fool\ at\ least\ one\ network;\ {R['wrong_all']}\ fool\ all\ ten}}$", 27, 540, y + 100, VERM, a, align="c")
            paste(img, rf"$\mathrm{{on\ {R['unanimous']}\ of\ those\ 10,\ all\ ten\ give\ the\ same\ wrong\ answer}}$", 24, 540, y + 162, INK2, a, align="c")
        elif md == "answers":
            paste(img, r"$\mathrm{the\ real\ answers\ are\ the\ labels\ in\ the\ data\ set}$", 26, 540, y + 100, GREEN, a, align="c")
            paste(img, r"$\mathrm{comment\ how\ many\ you\ read\ right}$", 24, 540, y + 162, INK2, a, align="c")
        else:
            paste(img, rf"$\mathrm{{nearest\ neighbor:\ {R['nn_right']}\ of\ these\ 10\ right}}$", 28, 540, y + 100, CYAN, a, align="c")
            paste(img, r"$\mathrm{different\ methods\ fail\ on\ different\ digits}$", 24, 540, y + 162, INK2, a, align="c")

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
        lab = "REAL HANDWRITTEN DIGITS   ·   540 UNSEEN   ·   ALL COMPUTED"
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
