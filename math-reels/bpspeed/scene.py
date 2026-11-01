"""Why backpropagation is fast: every slope for the price of about two runs. Dark theme of reels 31-35, maths in
Computer Modern (stage3d.tex), narrated by Kokoro (voicekit). The 8 + 8 network of reels 31 and 33 (105 numbers):
first the obvious way, nudging each number by +-0.00001 and running the network twice (the counter climbs to
210 runs, the slopes fill a bar chart one by one), then backpropagation (one gold pass forward, one magenta pass
back, all 105 slopes at once) and the comparison (the bars coincide; the largest difference is measured). Then a
network with a million weights, timed on this machine (speedsim.py): nudging would take 2,010,002 runs (hours),
backpropagation one forward and one backward pass (a fraction of a second). frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys

import cairo
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
CARD_FILL, CARD_EDGE = "#121830", "#3A4460"

with open(os.path.join(HERE, "speed_results.json")) as fh:
    RES = json.load(fh)
SM, BIG = RES["small"], RES["big"]
G_BP, G_FD = np.array(SM["g_bp"]), np.array(SM["g_fd"])
NP = SM["n_par"]


def say_time(s):
    """A duration in words for the narration."""
    if s >= 3600:
        h = s / 3600
        return f"about {round(h):d} hours" if h >= 3 else f"about {h:.1f} hours".replace(".0 ", " ")
    if s >= 60:
        return f"about {round(s / 60):d} minutes"
    if s >= 1:
        return f"about {s:.1f} seconds"
    return f"about {round(s * 1000):d} milliseconds"


def show_time(s):
    if s >= 3600:
        return f"{s / 3600:.1f} hours"
    if s >= 60:
        return f"{s / 60:.1f} minutes"
    if s >= 1:
        return f"{s:.2f} s"
    return f"{s * 1000:.0f} ms"


DIFF = SM["max_abs_diff"]
DIFF_WORDS = ("less than one ten-billionth" if DIFF < 1e-10 else "less than one billionth" if DIFF < 1e-9 else f"{DIFF:.1e}")
RATIO = BIG["ratio_bp_to_forward"]
SCRIPT = [
    "Backpropagation's real superpower isn't cleverness. It's speed.",
    f"To train a network, you need the slope of the error for every single number in it. This one has {NP}.",
    "The obvious way: nudge one number a tiny bit, run the network, and see how the error changes.",
    f"Two runs per number, to measure the slope both ways. {NP} numbers: {SM['passes_fd']} runs.",
    f"Backpropagation: one run forward, one run backward, and all {NP} slopes come out at once.",
    f"And they're the same slopes. The biggest difference between the two is {DIFF_WORDS}.",
    f"Now a network with a million weights. Nudging needs two million runs. On this computer, that's {say_time(BIG['fd_estimate_s'])}.",
    f"Backpropagation gets every one of those slopes in {say_time(BIG['t_backprop'])}.",
    f"That's why deep learning works at all: each training step gets every slope for the price of about {round(RATIO):d} runs.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the network, 2 nudge, 3 nudge count, 4 backprop, 5 agree, 6 big nudging, 7 big backprop, 8 close
SEG_MIN = [2.8, 6.0, 6.0, 6.5, 7.0, 6.0, 7.0, 6.0, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[9]
LOOP_FADE = (T_END, T_END + 0.55)
N_FRAMES = int(round((T_END + 0.7) * FPS))
NUDGE = (SEG[2] + 0.6, SEG[4] - 0.6)               # the nudges: 105 numbers, slow at first, then a blur
BP_FWD = (SEG[4] + 0.6, SEG[4] + 1.8)
BP_BWD = (SEG[4] + 2.0, SEG[4] + 3.4)
BIG_IN = (SEG[6] - 0.3, SEG[6] + 0.5)
BIG_OUT = (SEG[8] + 2.0, SEG[8] + 3.0)

DIAG = (120, 470, 840, 430)                       # x, y, w, h of the network diagram
CHART = (90, 1110, 900, 270)                      # the slopes, as bars
CARD = (70, 1430, 940, 250)
LAYERS = [2, 8, 8, 1]
# the 105 numbers in order: W0 (2x8), b0 (8), W1 (8x8), b1 (8), W2 (8x1), b2 (1)
PARAMS = []
for li in range(3):
    a_, b_ = LAYERS[li], LAYERS[li + 1]
    PARAMS += [("w", li, i, j) for i in range(a_) for j in range(b_)]
    PARAMS += [("b", li, j, None) for j in range(b_)]
assert len(PARAMS) == NP


def nudge_index(t):
    """How many numbers have been nudged by time t (float): the first few slowly, then faster and faster."""
    u = seg(t, *NUDGE)
    return NP * (0.035 * u / 0.25 if u < 0.25 else 0.035 + 0.965 * ((u - 0.25) / 0.75) ** 1.6)


def qa_frames():
    fr = []
    for k in range(9):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr[:17] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(n_par=NP, passes_fd=SM["passes_fd"], max_abs_diff=DIFF, big=BIG, segments=[round(v, 2) for v in SEG])
        x0, y0, w, h = DIAG
        self.pos = []
        for li, n in enumerate(LAYERS):
            cx = x0 + w * li / (len(LAYERS) - 1)
            gap = 48 if n > 2 else 120
            cy = y0 + h / 2
            self.pos.append([(cx, cy + (i - (n - 1) / 2) * gap) for i in range(n)])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("Backpropagation's superpower:", "black", 52, INK, 300, 1), ("speed.", "black", 52, GOLD, 390, 1)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Training needs every slope.", "black", 50, INK, 300, 1), (f"This network has {NP} numbers.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("The obvious way: nudge one number.", "black", 46, INK, 300, 1), ("Run the network. Watch the error.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Two runs per number.", "black", 50, INK, 300, 1), (f"{NP} numbers: {SM['passes_fd']} runs.", "dbold", 42, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Backpropagation: 2 runs.", "black", 52, MAGENTA, 300, 1), (f"Forward, backward: all {NP} slopes at once.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The same slopes.", "black", 52, INK, 300, 1), (f"Largest difference: {DIFF:.0e}.".replace("e-", " × 10⁻").replace("⁻11", "⁻¹¹").replace("⁻10", "⁻¹⁰").replace("⁻12", "⁻¹²"), "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [("A million weights:", "black", 52, INK, 300, 1), (f"nudging = {BIG['passes_fd']:,} runs ≈ {show_time(BIG['fd_estimate_s'])}.", "dbold", 38, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[7], S_[8] - 0.3, [(f"Backpropagation: {show_time(BIG['t_backprop'])}.", "black", 52, MAGENTA, 300, 1), (f"{BIG['speedup']:,.0f} times faster, same slopes.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[8], T_END - 0.3, [("Every slope, for about 2 runs.", "black", 50, INK, 300, 1), ("That's why deep learning works.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END + 0.1, T_END + 0.8, hook, 12, 0, "fade"),
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

    # ------------------------------------------------------------ the network

    def draw_net(self, ctx, t, a):
        if a <= 0:
            return
        k = nudge_index(t)
        cur = int(k) if seg(t, *NUDGE) < 1 and t >= NUDGE[0] else -1
        fwd, bwd = seg(t, *BP_FWD), seg(t, *BP_BWD)
        lit = (t >= BP_BWD[1] - 0.2) and t < SEG[6]
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for pi, (kind, li, i, j) in enumerate(PARAMS):
            if kind != "w":
                continue
            p, q = self.pos[li][i], self.pos[li + 1][j]
            done = (pi < k and NUDGE[0] <= t < SEG[4]) or lit
            col = GOLD if pi == cur else (MAGENTA if lit else (INK2 if done else INK3))
            al = 1.0 if pi == cur else (0.55 if done else 0.22)
            if pi == cur:
                ctx.set_source_rgba(*rgb(GOLD), 0.3 * a)
                ctx.set_line_width(9)
                ctx.move_to(*p)
                ctx.line_to(*q)
                ctx.stroke()
            ctx.set_source_rgba(*rgb(col), al * a)
            ctx.set_line_width(3.0 if pi == cur else 1.2)
            ctx.move_to(*p)
            ctx.line_to(*q)
            ctx.stroke()
        # the two backprop sweeps: a band of light crossing the layers
        for ph, col, rev in ((fwd, GOLD, False), (bwd, MAGENTA, True)):
            if 0 < ph < 1:
                xs = DIAG[0] + DIAG[2] * (1 - ph if rev else ph)
                g = cairo.LinearGradient(xs - 80, 0, xs + 80, 0)
                g.add_color_stop_rgba(0, *rgb(col), 0)
                g.add_color_stop_rgba(0.5, *rgb(col), 0.32 * a)
                g.add_color_stop_rgba(1, *rgb(col), 0)
                ctx.set_source(g)
                ctx.rectangle(xs - 80, DIAG[1] - 20, 160, DIAG[3] + 40)
                ctx.fill()
        # nodes; a bias being nudged rings its node
        for li, layer_pos in enumerate(self.pos):
            for i, (x, y) in enumerate(layer_pos):
                r = 15 if LAYERS[li] > 2 else 20
                ctx.set_source_rgba(*rgb(CARD_FILL), a)
                ctx.arc(x, y, r, 0, TAU)
                ctx.fill()
                ctx.set_source_rgba(*rgb(INK2), 0.85 * a)
                ctx.set_line_width(2.0)
                ctx.arc(x, y, r, 0, TAU)
                ctx.stroke()
        if cur >= 0 and PARAMS[cur][0] == "b":
            _, li, j, _ = PARAMS[cur]
            x, y = self.pos[li + 1][j]
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.set_line_width(3.0)
            ctx.arc(x, y, 22, 0, TAU)
            ctx.stroke()
        # the output pulses twice per nudge (the two runs)
        if cur >= 0:
            frac = (k - int(k))
            x, y = self.pos[-1][0]
            for half in (0, 1):
                u = (frac - 0.5 * half) * 2
                if 0 < u < 1:
                    ctx.set_source_rgba(*rgb(GOLD), 0.7 * (1 - u) * a)
                    ctx.set_line_width(2.5)
                    ctx.arc(x, y, 22 + 30 * u, 0, TAU)
                    ctx.stroke()
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        for li, name in enumerate(("INPUTS", "8 NEURONS", "8 NEURONS", "OUTPUT")):
            w_ = tracked_width(ctx, name, tr)
            ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
            show_tracked(ctx, name, self.pos[li][0][0] - w_ / 2, DIAG[1] + DIAG[3] + 26, tr)

    # ------------------------------------------------------------ the slopes as bars, and the run counters

    def draw_chart(self, ctx, t, a):
        if a <= 0:
            return
        x0, y0, w, h = CHART
        mid = y0 + h / 2
        bw = w / NP
        gm = math.sqrt(np.abs(G_BP).max())
        k = nudge_index(t)
        show_fd = (t >= NUDGE[0]) and t < SEG[6]
        bp_on = ease_out_cubic(seg(t, BP_BWD[1] - 0.2, BP_BWD[1] + 0.3)) if t < SEG[6] else 0.0
        ctx.set_source_rgba(*rgb(INK3), 0.5 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, mid)
        ctx.line_to(x0 + w, mid)
        ctx.stroke()
        for i in range(NP):
            hb = math.copysign(math.sqrt(abs(G_FD[i])) / gm, G_FD[i]) * (h / 2 - 8)
            if show_fd and i < k:
                ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
                ctx.rectangle(x0 + i * bw + 1, mid - max(hb, 0), bw - 2, abs(hb) + 0.5)
                ctx.fill()
            if bp_on > 0:
                hp = math.copysign(math.sqrt(abs(G_BP[i])) / gm, G_BP[i]) * (h / 2 - 8)
                ctx.set_source_rgba(*rgb(MAGENTA), 0.95 * bp_on * a)
                ctx.set_line_width(1.6)
                ctx.rectangle(x0 + i * bw + 1, mid - max(hp, 0), bw - 2, abs(hp) + 0.5)
                ctx.stroke()
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, f"THE {NP} SLOPES  ·  BAR HEIGHT: SLOPE (SQUARE-ROOT SCALE)", x0, y0 - 12, tr)

    def draw_counters(self, ctx, t, a):
        if a <= 0:
            return
        k = nudge_index(t)
        fresh = t >= SEG[6]                       # after the comparison the network comes back as it began
        runs_fd = int(2 * min(NP, k)) if (t >= NUDGE[0] and not fresh) else 0
        runs_bp = (1 if t >= BP_FWD[1] else 0) + (1 if t >= BP_BWD[1] else 0)
        bp_on = 0.0 if fresh else ease_out_cubic(seg(t, SEG[4], SEG[4] + 0.5))
        for (lab, val, col, xc, on) in (("NUDGING: RUNS", runs_fd, GOLD, 300, 1.0), ("BACKPROPAGATION: RUNS", runs_bp, MAGENTA, 780, bp_on)):
            al = a * on
            if al <= 0:
                continue
            font(ctx, "bold", 19)
            tr = 0.12 * 19
            w_ = tracked_width(ctx, lab, tr)
            ctx.set_source_rgba(*rgb(INK2), 0.95 * al)
            show_tracked(ctx, lab, xc - w_ / 2, 975, tr)
            font(ctx, "dbold", 58)
            s = f"{val}"
            w_ = ctx.text_extents(s).x_advance
            ctx.set_source_rgba(*rgb(col), al)
            ctx.move_to(xc - w_ / 2, 1046)
            ctx.show_text(s)

    # ------------------------------------------------------------ the million-weight comparison

    def draw_big(self, ctx, img, t, a):
        if a <= 0:
            return
        x0, x1 = 120, 960
        lo, hi = -3.0, 5.5                          # log10 seconds: 1 ms .. ~3.5 days
        def xof(s):
            return x0 + (x1 - x0) * (math.log10(s) - lo) / (hi - lo)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        lab = f"A NETWORK WITH {BIG['n_par']:,} WEIGHTS  ·  2 → 1000 → 1000 → 1  ·  TIMED ON THIS COMPUTER"
        w_ = tracked_width(ctx, lab, tr)
        show_tracked(ctx, lab, 540 - w_ / 2, 560, tr)
        for s, name in ((1e-3, "1 ms"), (1.0, "1 s"), (60, "1 min"), (3600, "1 hour"), (86400, "1 day")):
            x = xof(s)
            ctx.set_source_rgba(*rgb(INK3), 0.35 * a)
            ctx.set_line_width(1.0)
            ctx.move_to(x, 640)
            ctx.line_to(x, 1030)
            ctx.stroke()
            font(ctx, "bold", 16)
            w_ = ctx.text_extents(name).x_advance
            ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
            ctx.move_to(x - w_ / 2, 1062)
            ctx.show_text(name)
        rows = ((f"NUDGING: {BIG['passes_fd']:,} RUNS", BIG["fd_estimate_s"], GOLD, 720, (SEG[6] + 0.6, SEG[6] + 3.6)),
                ("BACKPROPAGATION: 1 RUN FORWARD, 1 BACK", BIG["t_backprop"], MAGENTA, 900, (SEG[7] + 0.4, SEG[7] + 1.4)))
        for name, sec, col, y, win in rows:
            g = ease_out_cubic(seg(t, *win))
            if g <= 0:
                continue
            font(ctx, "bold", 19)
            tr = 0.12 * 19
            ctx.set_source_rgba(*rgb(INK2), a * min(1.0, g * 3))
            show_tracked(ctx, name, x0, y - 30, tr)
            xe = x0 + (xof(sec) - x0) * g
            ctx.set_source_rgba(*rgb(col), 0.25 * a)
            pill = S3.pill
            pill(ctx, x0, y - 6, max(12, xe - x0) + 8, 52, 12)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), 0.95 * a)
            pill(ctx, x0, y, max(12, xe - x0), 40, 10)
            ctx.fill()
            font(ctx, "dbold", 34)
            s = show_time(sec)
            sw = ctx.text_extents(s).x_advance
            tx = xe + 16 if xe + 16 + sw < W - 40 else xe - sw - 16
            ctx.set_source_rgba(*rgb(INK if tx > xe else "#05060C"), a * g)
            ctx.move_to(tx, y + 32)
            ctx.show_text(s)

    # ------------------------------------------------------------ the card

    def card(self, img, t, a):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if t < SEG[2] or t >= T_END - 0.3:
            paste(img, r"$\mathrm{training\ needs}\ \frac{\partial L}{\partial w}\ \mathrm{for\ every\ number}\ w$", 32, 540, y + 128, INK, a, align="c")
            paste(img, rf"$\mathrm{{here:}}\ {NP}\ \mathrm{{numbers}}\ (88\ \mathrm{{weights}},\ 17\ \mathrm{{biases}})$", 22, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[4]:
            paste(img, r"$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$", 36, 540, y + 120, GOLD, a, align="c")
            paste(img, r"$\epsilon = 0.00001\ \mathrm{;\ two\ runs\ of\ the\ network\ per\ number}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[5]:
            paste(img, r"$\mathrm{chain\ rule,\ layer\ by\ layer:}\ \delta_j = \mathrm{ReLU}'(z_j)\,\sum_k w_{jk}\,\delta_k$", 28, 540, y + 116, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{one\ run\ forward,\ one\ run\ back:\ every}\ \partial L / \partial w\ \mathrm{at\ once}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[6]:
            m, e = f"{DIFF:.1e}".split("e")
            paste(img, rf"$\max_w\ \left|\,\mathrm{{nudge}} - \mathrm{{backprop}}\,\right| = {m} \times 10^{{{int(e)}}}$", 32, 540, y + 120, INK, a, align="c")
            paste(img, r"$\mathrm{the\ same\ slopes,\ to\ the\ precision\ of\ the\ nudge}$", 22, 540, y + h - 26, INK2, a, align="c")
        else:
            nfd = f"{BIG['passes_fd']:,}".replace(",", "{,}")
            paste(img, rf"$\mathrm{{nudging}}: 2n\ \mathrm{{runs}} = {nfd}\ \times\ {BIG['t_forward'] * 1000:.1f}\,\mathrm{{ms}}$", 28, 540, y + 96, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{backprop}}: \approx 2\ \mathrm{{runs}} = {BIG['t_backprop'] * 1000:.0f}\,\mathrm{{ms}}\ \ ({RATIO:.1f} \times\ \mathrm{{one\ run}})$", 28, 540, y + 152, MAGENTA,
                  a * ease_out_cubic(seg(t, SEG[7], SEG[7] + 0.5)), align="c")
            paste(img, r"$\mathrm{times\ measured\ on\ this\ computer\ (median\ of\ repeated\ runs)}$", 21, 540, y + h - 22, INK2, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        a = max(fade, ease_out_cubic(seg(t, LOOP_FADE[0] + 0.15, LOOP_FADE[1] + 0.1)))
        img = self.bg.copy()
        big = smooth(seg(t, *BIG_IN)) * (1 - smooth(seg(t, *BIG_OUT)))
        surf, ctx = layer()
        self.draw_net(ctx, t, a * (1 - big))
        self.draw_chart(ctx, t, a * (1 - big))
        self.draw_counters(ctx, t, a * (1 - big))
        self.draw_big(ctx, img, t, a * big)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "THE 8 + 8 NETWORK OF REELS 31–33   ·   THE SPIRAL, 400 POINTS   ·   ALL COMPUTED"
        w = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        show_tracked(ctx, lab, 540 - w / 2, 1752, tr)
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
    print("facts", {k: v for k, v in sc.facts.items() if k != "big"})
    for line in SCRIPT:
        print("  >", line)
    frames = qa_frames()
    ims, times = [], []
    for fr in frames:
        t0 = time.time()
        ims.append(cv2.resize(sc.frame(fr)[..., ::-1], (360, 640), interpolation=cv2.INTER_AREA))
        times.append(time.time() - t0)
    print(f"frame time: mean {np.mean(times[1:]):.3f}s max {max(times[1:]):.3f}s")
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
