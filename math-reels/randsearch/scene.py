"""A million random networks against gradient descent. Dark theme of reels 31-35, 41 and 42, maths in Computer Modern
(stage3d.tex), narrated by Kokoro (voicekit). The 8 + 8 network of reels 31, 33 and 41 (105 numbers) on the same
spiral (rssim.py): first random search, a million networks with every number drawn at random, the map showing a
few ordinary tries and then the best network found so far, the best score climbing to 69% and then stopping;
then gradient descent from one random start, following the slope backpropagation computes, passing the best of
the million at step 94 and reaching 95.5% at step 1,500. The chart puts both on one axis: runs of the network
(log scale). Then why guessing can't work (10^105 networks for ten values per number). frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import cairo
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
PANEL_BASE, POINT_EDGE = "#161C33", "#05060C"

# ---------------------------------------------------------------- the runs (rssim.py) and every number the reel states
sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "rs_fields.npz")):       # the archives are not committed: rebuild them (about 8 minutes)
    import rssim
    rssim.main()
if not os.path.exists(os.path.join(HERE, "rs_steps.npz")):
    import rssim
    rssim.save_steps()
with open(os.path.join(HERE, "rs_results.json")) as fh:
    R = json.load(fh)
F = np.load(os.path.join(HERE, "rs_fields.npz"))
ST = np.load(os.path.join(HERE, "rs_steps.npz"))
X, Y = F["X"], F["y"]
GD_FIELDS = F["gd_fields"]                       # every 15 steps
GD_ACC = np.array(R["gd_acc"])
N_RANDOM = R["n_random"]
BEST = R["best_random"]
CURVE = R["curve"]                               # (networks tried, best so far) at each improvement
REC_N, REC_ACC, REC_FIELD = ST["rec_n"], ST["rec_acc"], ST["rec_field"]
SMP_N, SMP_ACC, SMP_FIELD = ST["sample_n"], ST["sample_acc"], ST["sample_field"]
assert int(REC_N[-1]) == CURVE[-1][0] and abs(float(REC_ACC[-1]) - BEST) < 1e-12
BEATS = R["gd_beats_random_at"]
GD_STEPS = R["gd_steps"]
LAST_REC = int(REC_N[-1])
AFTER = {int(k): v for k, v in R["best_after"].items()}


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


def best_after(n):
    b = 0.0
    for k, a in CURVE:
        if k <= n:
            b = a
    return b


SCRIPT = [
    f"I tried a million random networks on this spiral. The best one gets {round(100 * BEST):d} percent right.",
    "Same network as the backpropagation video: a hundred and five numbers. Each try picks all of them at random, and counts how many of the four hundred points it gets right.",
    f"Ten tries: {round(100 * AFTER[10]):d} percent. A thousand: {round(100 * AFTER[1000]):d}. A hundred thousand: {round(100 * AFTER[100000]):d}. "
    f"Then nothing. Not one better network in the last {round((N_RANDOM - LAST_REC) / 1000):d} thousand.",
    "Now gradient descent. One random start, and at every step, it follows the slope that backpropagation computes.",
    f"Step {BEATS}: it already beats the best of the million. Step fifteen hundred: ninety-five and a half percent.",
    "And it needed about three thousand runs of the network. The random search used a million.",
    "Guessing can't work. Even ten choices for each of the hundred and five numbers makes ten to the power of a hundred and five networks. "
    "The universe has about ten to the eighty atoms.",
    "The slope says which way to move every number, all at once. That's what backpropagation buys you.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 a random try, 2 a million tries, 3 gradient descent, 4 it wins, 5 the work, 6 why guessing fails, 7 close
SEG_MIN = [3.5, 7.0, 8.5, 6.0, 7.5, 6.0, 9.0, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[8]
N_FRAMES = int(round((T_END + 0.7) * FPS))
TRIES = (SEG[1] + 1.0, SEG[2] - 0.4)            # the first 40 tries, one by one
SEARCH = (SEG[2] + 0.2, SEG[3] - 0.6)            # 40 -> 1,000,000 tries (log time)
GD = (SEG[3] + 1.2, SEG[5] - 0.8)                # 0 -> 1,500 steps
HOME = (T_END - 1.2, T_END - 0.1)                # back to the opening for the loop

PANEL = (260, 440, 560)                          # the map: x, y, size
LIM = 1.05
CHART = (150, 1240, 820, 136)
CARD = (70, 1430, 940, 250)


def tries_at(t):
    """Networks tried so far."""
    if t < TRIES[0]:
        return 0 if t >= SEG[1] else N_RANDOM
    if t < TRIES[1]:
        return 1 + int(39 * seg(t, *TRIES) ** 1.3)
    if t < SEARCH[0]:
        return 40
    u = smooth(seg(t, *SEARCH))
    return int(round(10 ** (math.log10(40) + (6 - math.log10(40)) * u)))


def gd_step(t):
    """Steps taken: slowly up to the step that beats the best random network (as the narration reaches it), then on."""
    if t < GD[0]:
        return 0
    tb = SEG[4] + 0.6
    if t < tb:
        return int(round(BEATS * seg(t, GD[0], tb)))
    return int(round(BEATS + (GD_STEPS - BEATS) * seg(t, tb, GD[1]) ** 1.3))


def mode(t):
    if t < SEG[1] - 0.2 or t >= HOME[1]:
        return "hook"
    if t < GD[0]:
        return "random"
    if t < HOME[0]:
        return "gd"
    return "home"


def qa_frames():
    fr = []
    for k in range(8):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n_random=N_RANDOM, best_random=BEST, best_after=AFTER, last_record=LAST_REC, gd_final=float(GD_ACC[-1]), beats_at=BEATS,
                          gd_at_beats=float(GD_ACC[BEATS]), passes_gd=R["passes_gd"], segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        self._palette()
        self._panel_frame()

    def _cues(self):
        S_ = SEG
        hook = [("1,000,000 random networks.", "black", 52, INK, 300, 1), (f"The best: {pc(BEST)}%.", "black", 50, GOLD, 390, 1)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Every number at random.", "black", 50, INK, 300, 1), ("Then count the points it gets right.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [(f"A million tries: {pc(BEST)}%.", "black", 50, GOLD, 300, 1), (f"Nothing better after try {LAST_REC:,}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Gradient descent.", "black", 52, MAGENTA, 300, 1), ("Follow the slope from backpropagation.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"Step {BEATS}: better than all of them.", "black", 46, MAGENTA, 300, 1), (f"Step {GD_STEPS:,}: {pc(GD_ACC[-1])}%.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("3,000 runs vs 1,000,000.", "black", 50, INK, 300, 1), ("Less work, far better.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [("Why guessing can't work:", "black", 50, INK, 300, 1), ("10¹⁰⁵ networks, 10⁸⁰ atoms.", "dbold", 44, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[7], T_END - 0.3, [("The slope says which way.", "black", 50, INK, 300, 1), ("That's what backpropagation buys.", "dbold", 40, MAGENTA, 390, 0.95)], 10, 8, "pop"),
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

    def _palette(self):
        w = np.array(rgb(PANEL_BASE), np.float32)
        cb, cv = np.array(rgb(BLUE), np.float32), np.array(rgb(VERM), np.float32)
        lut = np.empty((256, 3), np.float32)
        for i, p in enumerate(np.linspace(0, 1, 256)):
            lut[i] = w + (cb - w) * 0.78 * (0.5 - p) * 2 if p < 0.5 else w + (cv - w) * 0.78 * (p - 0.5) * 2
        self.lut = lut

    def _panel_frame(self):
        px, py, n = PANEL
        m = np.zeros((n, n), np.float32)
        surf, ctx = layer()
        S3.pill(ctx, px, py, n, n, 18)
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.fill()
        _, ta = surf_rgba(surf)
        self.panel_mask = ta[py:py + n, px:px + n].copy()

    # ------------------------------------------------------------ the map

    def to_panel(self, x, y):
        px, py, n = PANEL
        return px + (x + LIM) / (2 * LIM) * n, py + (LIM - y) / (2 * LIM) * n

    def field_now(self, t):
        """(field, accuracy shown, colour of the score, label) for the map at time t."""
        md = mode(t)
        if md == "hook":
            return REC_FIELD[-1], BEST, GOLD, f"THE BEST OF {N_RANDOM:,} RANDOM NETWORKS"
        if md == "random":
            n = tries_at(t)
            if t < TRIES[1] and t >= TRIES[0]:
                k = min(n, len(SMP_N)) - 1
                return SMP_FIELD[k], float(SMP_ACC[k]), INK, f"RANDOM NETWORK #{n}"
            if t < TRIES[0]:
                return REC_FIELD[-1], BEST, GOLD, f"THE BEST OF {N_RANDOM:,} RANDOM NETWORKS"
            i = int(np.searchsorted(REC_N, n, side="right")) - 1
            return REC_FIELD[i], float(REC_ACC[i]), GOLD, f"BEST SO FAR: NETWORK #{int(REC_N[i]):,}"
        if md == "gd":
            s = gd_step(t)
            k = min(s // 15, len(GD_FIELDS) - 1)
            return GD_FIELDS[k], float(GD_ACC[s]), MAGENTA, f"GRADIENT DESCENT, STEP {s:,}"
        u = smooth(seg(t, *HOME))
        f_ = GD_FIELDS[-1] * (1 - u) + REC_FIELD[-1] * u
        return f_, BEST if u > 0.5 else float(GD_ACC[-1]), GOLD if u > 0.5 else MAGENTA, (f"THE BEST OF {N_RANDOM:,} RANDOM NETWORKS" if u > 0.5 else f"GRADIENT DESCENT, STEP {GD_STEPS:,}")

    def draw_map(self, img, t, a):
        px, py, n = PANEL
        f_, acc, col, lab = self.field_now(t)
        up = cv2.resize(f_[::-1].astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        rgbf = self.lut[np.clip((up * 255).round().astype(np.int32), 0, 255)]
        dim = 1 - 0.55 * smooth(seg(t, SEG[6] + 0.2, SEG[6] + 1.0)) * (1 - smooth(seg(t, SEG[7] - 0.2, SEG[7] + 0.6)))
        m = self.panel_mask[..., None] * a
        sub = img[py:py + n, px:px + n]
        sub[:] = sub * (1 - m) + rgbf * dim * m
        surf, ctx = layer()
        ctx.save()
        S3.pill(ctx, px, py, n, n, 18)
        ctx.clip()
        cs, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for lw, al in ((7.0, 0.14), (2.0, 0.9)):
            ctx.set_source_rgba(*rgb(INK), al * a * dim)
            ctx.set_line_width(lw)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            for c in cs:
                if len(c) < 6:
                    continue
                pts = c[:, 0, :]
                ctx.move_to(px + pts[0][0] + 0.5, py + pts[0][1] + 0.5)
                for x, y in pts[1:]:
                    ctx.line_to(px + x + 0.5, py + y + 0.5)
                ctx.close_path()
                ctx.stroke()
        for (x, y), c in zip(X, Y):
            sx, sy = self.to_panel(x, y)
            cc = rgb(VERM if c > 0.5 else BLUE)
            ctx.set_source_rgba(*cc, 0.95 * a * dim)
            ctx.arc(sx, sy, 3.4, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(POINT_EDGE), 0.8 * a * dim)
            ctx.set_line_width(0.8)
            ctx.arc(sx, sy, 3.4, 0, TAU)
            ctx.stroke()
        ctx.restore()
        ctx.set_source_rgba(*rgb(INK2), 0.55 * a)
        ctx.set_line_width(1.4)
        S3.pill(ctx, px, py, n, n, 18)
        ctx.stroke()
        # the label and the score
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, lab, px, py + n + 34, tr)
        font(ctx, "dbold", 44)
        s = f"{pc(acc)}%"
        ctx.set_source_rgba(*rgb(col), a)
        ctx.move_to(px + n - ctx.text_extents(s).x_advance, py + n + 52)
        ctx.show_text(s)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        lab2 = f"OF 400 POINTS RIGHT: {int(round(acc * 400))}"
        show_tracked(ctx, lab2, px, py + n + 62, 0.12 * 16)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ the chart: accuracy against runs of the network

    def draw_chart(self, img, t, a):
        cx, cy, cw, ch = CHART
        lo, hi = 50.0, 100.0

        def gx(runs):
            return cx + math.log10(max(runs, 1)) / 6 * cw

        def gy(acc):
            return cy + ch - (100 * acc - lo) / (hi - lo) * ch
        surf, ctx = layer()
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(cx, cy)
        ctx.line_to(cx, cy + ch)
        ctx.line_to(cx + cw, cy + ch)
        ctx.stroke()
        ctx.set_source_rgba(1, 1, 1, 0.07 * a)
        for v in (0.6, 0.7, 0.8, 0.9, 1.0):
            ctx.move_to(cx, gy(v))
            ctx.line_to(cx + cw, gy(v))
        for e in range(1, 7):
            ctx.move_to(gx(10 ** e), cy)
            ctx.line_to(gx(10 ** e), cy + ch)
        ctx.stroke()
        md = mode(t)
        # random search: the best so far, as a staircase
        n_now = N_RANDOM if md in ("hook", "gd", "home") else max(1, tries_at(t))
        if md == "random" and t < TRIES[0]:
            n_now = 0
        if n_now >= 1:
            ctx.set_source_rgba(*rgb(GOLD), 0.95 * a)
            ctx.set_line_width(3.0)
            pts = [(1, CURVE[0][1])] + [(k, v) for k, v in CURVE[1:] if k <= n_now]
            ctx.move_to(gx(1), gy(pts[0][1]))
            prev = pts[0][1]
            for k, v in pts[1:]:
                ctx.line_to(gx(k), gy(prev))
                ctx.line_to(gx(k), gy(v))
                prev = v
            ctx.line_to(gx(n_now), gy(prev))
            ctx.stroke()
            ctx.arc(gx(n_now), gy(prev), 5, 0, TAU)
            ctx.fill()
        # gradient descent: accuracy against runs (one forward and one backward run per step)
        s_now = GD_STEPS if md in ("hook", "home") or t >= GD[1] else gd_step(t)
        gd_a = 1.0 if md != "random" else 0.0
        if md == "hook":
            gd_a = 1.0
        if gd_a > 0 and s_now >= 0:
            ctx.set_source_rgba(*rgb(MAGENTA), 0.95 * a * gd_a)
            ctx.set_line_width(3.0)
            ctx.move_to(gx(1), gy(GD_ACC[0]))
            for s in range(1, s_now + 1):
                ctx.line_to(gx(2 * s), gy(GD_ACC[s]))
            ctx.stroke()
            ctx.arc(gx(max(1, 2 * s_now)), gy(GD_ACC[s_now]), 5, 0, TAU)
            ctx.fill()
            if s_now >= BEATS and md == "gd":
                xb, yb = gx(2 * BEATS), gy(GD_ACC[BEATS])
                ctx.set_source_rgba(1, 1, 1, 0.9 * a)
                ctx.set_line_width(2.0)
                ctx.arc(xb, yb, 9, 0, TAU)
                ctx.stroke()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        for v in (0.5, 0.75, 1.0):
            paste(img, rf"${round(100 * v):d}\%$", 18, cx - 10, gy(v) + 6, INK3, a, align="r")
        for e, lab in ((0, "1"), (2, "100"), (4, "10^{4}"), (6, "10^{6}")):
            paste(img, rf"${lab}$", 18, gx(10 ** e), cy + ch + 24, INK3, a, align="c")
        paste(img, r"$\mathrm{points\ right\ vs\ runs\ of\ the\ network\ (log\ scale)}$", 18, cx + cw, cy - 10, INK3, a, align="r")
        if md == "gd" and gd_step(t) >= BEATS:
            paste(img, rf"$\mathrm{{step}}\ {BEATS}$", 20, gx(2 * BEATS) + 14, gy(GD_ACC[BEATS]) - 12, INK, a, align="l", halo=3)

    def draw_counters(self, img, t, a):
        md = mode(t)
        surf, ctx = layer()
        cols = []
        n = tries_at(t) if md == "random" else N_RANDOM
        cols.append((150, "RANDOM NETWORKS TRIED", f"{n:,}", GOLD, 1.0))
        if md in ("gd", "hook", "home") or t >= GD[0]:
            s = gd_step(t) if md == "gd" else GD_STEPS
            cols.append((620, "GRADIENT DESCENT STEPS", f"{s:,}", MAGENTA, 1.0 if md != "hook" else 1.0))
        for x, lab, val, col, al in cols:
            font(ctx, "bold", 17)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a * al)
            show_tracked(ctx, lab, x, 1110, 0.12 * 17)
            font(ctx, "dbold", 38)
            ctx.set_source_rgba(*rgb(col), a * al)
            ctx.move_to(x, 1152)
            ctx.show_text(val)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ the card

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if t < SEG[1] or t >= SEG[7]:
            paste(img, rf"$\mathrm{{best\ of}}\ 1{{,}}000{{,}}000\ \mathrm{{random\ networks}}:\ {pc(BEST)}\%$", 30, 540, y + 96, GOLD, a, align="c")
            steps = f"{GD_STEPS:,}".replace(",", "{,}")
            paste(img, rf"$\mathrm{{gradient\ descent,}}\ {steps}\ \mathrm{{steps}}:\ {pc(GD_ACC[-1])}\%$", 30, 540, y + 152, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{the\ same\ 2 \to 8 \to 8 \to 1\ network,\ the\ same\ 400\ points}$", 21, 540, y + h - 20, INK2, a, align="c")
        elif t < SEG[2]:
            paste(img, r"$\mathrm{a\ try:\ all\ 105\ numbers\ drawn\ at\ random}$", 32, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{score} = \mathrm{points\ on\ the\ right\ side\ of\ the\ white\ line} \,/\, 400$", 26, 540, y + 164, INK2, a, align="c")
            paste(img, r"$\mathrm{weights\ spread\ as\ at\ the\ start\ of\ training;\ biases\ too}$", 21, 540, y + h - 20, INK3, a, align="c")
        elif t < SEG[3]:
            rows = [(10, AFTER[10]), (1000, AFTER[1000]), (100000, AFTER[100000]), (1000000, AFTER[1000000])]
            xs = (215, 430, 650, 870)
            for (nn, v), xx in zip(rows, xs):
                lab = f"{nn:,}".replace(",", "{,}")
                paste(img, rf"${lab}$", 24, xx, y + 84, INK2, a, align="c")
                paste(img, rf"${pc(v)}\%$", 32, xx, y + 140, GOLD, a, align="c")
            paste(img, r"$\mathrm{best\ after\ this\ many\ tries}$", 21, 540, y + h - 20, INK3, a, align="c")
        elif t < SEG[4]:
            paste(img, r"$w \;\leftarrow\; w - \eta\,\frac{\partial L}{\partial w}\quad \mathrm{for\ all\ 105\ numbers}$", 34, 540, y + 110, MAGENTA, a, align="c")
            paste(img, r"$\frac{\partial L}{\partial w}:\ \mathrm{from\ backpropagation\ (one\ run\ forward,\ one\ back)};\ \mathrm{Adam},\ \eta = 0.01$", 21, 540, y + h - 22, INK2, a, align="c")
        elif t < SEG[5]:
            paste(img, rf"$\mathrm{{step}}\ {BEATS}:\ {pc(GD_ACC[BEATS], 2)}\% > {pc(BEST)}\%$", 34, 540, y + 100, MAGENTA, a, align="c")
            paste(img, rf"$\mathrm{{step}}\ 1{{,}}500:\ {pc(GD_ACC[-1])}\%$", 30, 540, y + 162, INK, a, align="c")
            paste(img, r"$\mathrm{from\ one\ random\ start\ (the\ start\ of\ the\ backpropagation\ reel)}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[6]:
            paste(img, rf"$\mathrm{{random\ search}}:\ 1{{,}}000{{,}}000\ \mathrm{{runs}}\ \rightarrow\ {pc(BEST)}\%$", 30, 540, y + 96, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{gradient\ descent}}:\ 1{{,}}500 \times 2\ \mathrm{{runs}}\ \rightarrow\ {pc(GD_ACC[-1])}\%$", 30, 540, y + 154, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{a\ step\ is\ one\ run\ forward\ and\ one\ run\ back}$", 21, 540, y + h - 20, INK3, a, align="c")
        else:
            paste(img, r"$10\ \mathrm{values\ for\ each\ of\ 105\ numbers}:\ 10^{105}\ \mathrm{networks}$", 27, 540, y + 96, GOLD, a, align="c")
            paste(img, r"$\mathrm{atoms\ in\ the\ observable\ universe}:\ \approx 10^{80}$", 30, 540, y + 156, INK2, a * ease_out_cubic(seg(t, SEG[6] + 4.0, SEG[6] + 4.8)), align="c")
            paste(img, r"$\mathrm{every\ extra\ number\ multiplies\ the\ choices\ by\ 10}$", 21, 540, y + h - 20, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        self.draw_map(img, t, a)
        self.draw_counters(img, t, a)
        self.draw_chart(img, t, a)
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "THE 8 + 8 NETWORK OF REELS 31–41   ·   THE SPIRAL, 400 POINTS   ·   REAL RUNS"
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
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
