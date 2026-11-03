"""Why AI runs on matrix multiplication. Dark theme of reels 31-46, maths in Computer Modern (stage3d.tex), narrated
by Kokoro (voicekit). One 512 x 512 matrix multiplication, timed on this computer (mmsim.py): plain Python fills the
answer one entry at a time (shown at 2x speed, the timer is real time), NumPy on one core and on all four fill it at
once. Then the times as bars on a log scale, and the million-weight network of reel 41, whose forward pass is three
matrix multiplications: NumPy against plain Python (estimated from the measured loop speed). frame(f) -> uint8 RGB."""

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
CORE_C = (CYAN, GREEN, MAGENTA, BLUE)

with open(os.path.join(HERE, "mm_results.json")) as fh:
    R = json.load(fh)
N = R["n"]
T_PY, T_NP1, T_NPA = R["t_python_s"], R["t_numpy_1core_s"], R["t_numpy_all_s"]
SPEEDUP = R["speedup_all"]
CORES = R["cores"]
T_NET, T_NET_PY = R["t_net_numpy_s"], R["t_net_python_est_s"]
G = 16                                            # the matrices are drawn as 16 x 16 blocks of 32 x 32 entries
PLAY = 2.0                                        # the Python run is shown at 2x speed


def ms(s):
    return f"{s * 1000:.1f} ms"


def words_ms(s):
    v = s * 1000
    return f"{v:.1f}".replace(".", " point ").replace(" point 0", "") + " milliseconds"


SCRIPT = [
    f"Same multiplication. Python: {round(T_PY):d} seconds. NumPy: two milliseconds.",
    "Every layer of a neural network is one matrix multiplication: inputs times weights. Here, two 512 by 512 matrices: 134 million multiply-adds.",
    f"Plain Python does them one at a time. It takes {T_PY:.1f} seconds.",
    f"NumPy hands the same job to compiled code that does many multiply-adds per instruction. On one core: {round(T_NP1 * 1000):d} milliseconds.",
    f"On all {CORES} cores: {T_NPA * 1000:.1f} milliseconds. The same answer, almost {round(SPEEDUP / 1000):d} thousand times faster.",
    f"The million-weight network from the backpropagation video: one forward pass is three of these. NumPy: {round(T_NET * 1000):d} milliseconds. Plain Python: about {round(T_NET_PY):d} seconds.",
    "That's why AI runs on matrix multiplication. It's the one job computers have learned to do absurdly fast, and GPUs take it to thousands of cores.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 a layer is a matmul, 2 Python, 3 NumPy 1 core, 4 all cores + bars, 5 the network, 6 close
SEG_MIN = [3.5, 8.0, 8.0, 7.0, 8.0, 9.0, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))
PY_RUN = (SEG[2] + 0.5, SEG[2] + 0.5 + T_PY / PLAY)
NP1_AT = SEG[3] + 2.6
NPA_AT = SEG[4] + 0.6
BARS = (SEG[4] + 2.6, SEG[5] - 0.2)
NET = (SEG[5], SEG[6] - 0.2)

GRID_A = (90, 900, 330)                           # x, y, size
GRID_B = (450, 540, 330)
GRID_C = (450, 900, 330)
CARD = (70, 1430, 940, 250)


def fill_state(t):
    """(entries of C done as a fraction, mode, time shown)."""
    if t < SEG[1] - 0.2:
        return 1.0, "cores", T_NPA
    if t < PY_RUN[0]:
        return 0.0, "python", 0.0
    if t < SEG[3]:
        u = seg(t, *PY_RUN)
        return u, "python", u * T_PY
    if t < NP1_AT:
        return 1.0 if t < SEG[3] + 0.6 else 0.0, "python" if t < SEG[3] + 0.6 else "one", T_PY if t < SEG[3] + 0.6 else 0.0
    if t < SEG[4]:
        return 1.0, "one", T_NP1
    if t < NPA_AT:
        return 0.0, "cores", 0.0
    return 1.0, "cores", T_NPA


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.3 * (b - a)) * FPS), int((a + 0.6 * (b - a)) * FPS), int((a + 0.9 * (b - a)) * FPS)]
    return fr[:20] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = {k: R[k] for k in ("n", "madds", "t_python_s", "t_numpy_1core_s", "t_numpy_all_s", "speedup_all", "max_abs_diff", "t_net_numpy_s", "t_net_python_est_s")}
        self.facts["segments"] = [round(v, 2) for v in SEG]
        self.cues = self._cues()
        self._build_bg()
        rng = np.random.default_rng(0)
        self.va, self.vb = rng.normal(size=(G, G)), rng.normal(size=(G, G))
        self.vc = self.va @ self.vb

    def _cues(self):
        S_ = SEG
        hook = [("Same multiplication.", "black", 52, INK, 300, 1), (f"Python {T_PY:.1f} s.  NumPy {ms(T_NPA)}.", "black", 44, GOLD, 390, 1)]
        madds = f"{R['madds']:,}"
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("A layer is a matrix multiplication.", "black", 46, INK, 300, 1), (f"512 × 512: {madds} multiply-adds.", "dbold", 36, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Python: one at a time.", "black", 50, GOLD, 300, 1), (f"{T_PY:.1f} seconds.", "dbold", 44, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [(f"NumPy, 1 core: {ms(T_NP1)}.", "black", 50, CYAN, 300, 1), ("Many multiply-adds per instruction.", "dbold", 38, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [(f"All {CORES} cores: {ms(T_NPA)}.", "black", 50, MAGENTA, 300, 1), (f"Same answer. {SPEEDUP:,.0f}× faster.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("A million-weight network:", "black", 50, INK, 300, 1), (f"NumPy {T_NET * 1000:.0f} ms.  Python ≈ {T_NET_PY:.0f} s.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("AI runs on matrix multiplication.", "black", 46, INK, 300, 1), ("Computers do it absurdly fast.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
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

    # ------------------------------------------------------------ the matrices

    def grid(self, ctx, box, vals, a, done=None, col_done=None, hl_row=None, hl_col=None, core_cols=False):
        x0, y0, n = box
        c = n / G
        vmax = np.abs(vals).max()
        for i in range(G):
            for j in range(G):
                v = vals[i, j] / vmax
                if done is not None and not done[i, j]:
                    ctx.set_source_rgba(0.10, 0.12, 0.2, 0.9 * a)
                else:
                    base = np.array(rgb(col_done)) if col_done else (np.array(rgb(VERM)) if v > 0 else np.array(rgb(BLUE)))
                    if core_cols:
                        base = np.array(rgb(CORE_C[(i >= G // 2) * 2 + (j >= G // 2)]))
                    k = 0.25 + 0.6 * abs(v) if not (col_done or core_cols) else 0.35 + 0.5 * abs(v)
                    ctx.set_source_rgba(*(base * k + 0.06), a)
                ctx.rectangle(x0 + j * c + 1, y0 + i * c + 1, c - 2, c - 2)
                ctx.fill()
        if hl_row is not None:
            ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
            ctx.set_line_width(2.4)
            ctx.rectangle(x0, y0 + hl_row * c, n, c)
            ctx.stroke()
        if hl_col is not None:
            ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
            ctx.set_line_width(2.4)
            ctx.rectangle(x0 + hl_col * c, y0, c, n)
            ctx.stroke()
        ctx.set_source_rgba(*rgb(INK2), 0.6 * a)
        ctx.set_line_width(1.2)
        ctx.rectangle(x0, y0, n, n)
        ctx.stroke()

    def draw_matrices(self, img, t, a):
        if a <= 0.01:
            return
        frac, md, shown = fill_state(t)
        surf, ctx = layer()
        cells = G * G
        k = int(frac * cells + 1e-9)
        done = np.zeros((G, G), bool)
        done.flat[:k] = True
        hl_r = hl_c = None
        if md == "python" and 0 < frac < 1:
            hl_r, hl_c = k // G, k % G
        flash = 0.0
        if md in ("one", "cores") and frac >= 1:
            t_at = NP1_AT if md == "one" else NPA_AT
            flash = max(0.0, 1 - (t - t_at) / 0.6) if t >= t_at and t < t_at + 0.6 else 0.0
        self.grid(ctx, GRID_A, self.va, a, hl_row=hl_r)
        self.grid(ctx, GRID_B, self.vb, a, hl_col=hl_c)
        self.grid(ctx, GRID_C, self.vc, a, done=done, col_done=(GOLD if md == "python" else (CYAN if md == "one" else None)), core_cols=(md == "cores"))
        if hl_r is not None:
            c = GRID_C[2] / G
            ctx.set_source_rgba(1, 1, 1, a)
            ctx.set_line_width(3)
            ctx.rectangle(GRID_C[0] + hl_c * c, GRID_C[1] + hl_r * c, c, c)
            ctx.stroke()
        if flash > 0:
            ctx.set_source_rgba(1, 1, 1, 0.35 * flash * a)
            ctx.rectangle(*GRID_C[:2], GRID_C[2], GRID_C[2])
            ctx.fill()
        if md == "cores" and frac >= 1:
            x0, y0, n = GRID_C
            ctx.set_line_width(2.5)
            for q in range(4):
                ctx.set_source_rgba(*rgb(CORE_C[q]), 0.95 * a)
                ctx.rectangle(x0 + (q % 2) * n / 2 + 2, y0 + (q // 2) * n / 2 + 2, n / 2 - 4, n / 2 - 4)
                ctx.stroke()
        # labels
        font(ctx, "bold", 17)
        tr = 0.12 * 17
        for box, lab in ((GRID_A, "A: INPUTS (512 × 512)"), (GRID_B, "B: WEIGHTS (512 × 512)"), (GRID_C, "C = A B")):
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, lab, box[0], box[1] - 14, tr)
        # the readouts, right of B
        rx = 800
        rows = [("MULTIPLY-ADDS DONE", f"{int(frac * R['madds']):,}", INK)]
        name = {"python": "PLAIN PYTHON", "one": "NUMPY, 1 CORE", "cores": f"NUMPY, {CORES} CORES"}[md]
        colr = {"python": GOLD, "one": CYAN, "cores": MAGENTA}[md]
        tshow = f"{shown:.2f} s" if md == "python" else ms(shown)
        rows.append((name, tshow, colr))
        for i, (lab, val, col) in enumerate(rows):
            yy = 580 + i * 120
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, lab, rx, yy, 0.1 * 16)
            font(ctx, "dbold", 30 if i == 0 else 40)
            ctx.set_source_rgba(*rgb(col), a)
            ctx.move_to(rx, yy + 44)
            ctx.show_text(val)
        if md == "python" and 0 < frac < 1:
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(GOLD), 0.9 * a)
            show_tracked(ctx, "▶▶ SHOWN AT 2× SPEED", rx, 580 + 2 * 120, 0.1 * 16)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ the time bars (log scale)

    def draw_bars(self, img, t, a, rows, title, win):
        if a <= 0.01:
            return
        surf, ctx = layer()
        x0, x1 = 120, 960
        lo, hi = -3.5, 2.0

        def xof(s):
            return x0 + (x1 - x0) * (math.log10(s) - lo) / (hi - lo)
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        w_ = tracked_width(ctx, title, tr)
        show_tracked(ctx, title, 540 - w_ / 2, 640, tr)
        for s, name in ((1e-3, "1 ms"), (1e-2, "10 ms"), (0.1, "0.1 s"), (1.0, "1 s"), (10.0, "10 s"), (100.0, "100 s")):
            x = xof(s)
            ctx.set_source_rgba(*rgb(INK3), 0.35 * a)
            ctx.set_line_width(1.0)
            ctx.move_to(x, 720)
            ctx.line_to(x, 1160)
            ctx.stroke()
            font(ctx, "bold", 16)
            w_ = ctx.text_extents(name).x_advance
            ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
            ctx.move_to(x - w_ / 2, 1192)
            ctx.show_text(name)
        for k, (name, sec, col, dashed) in enumerate(rows):
            y = 790 + k * 130
            g = ease_out_cubic(seg(t, win[0] + 0.5 * k, win[0] + 0.5 * k + 0.9))
            if g <= 0:
                continue
            font(ctx, "bold", 19)
            ctx.set_source_rgba(*rgb(INK2), a * min(1.0, g * 3))
            show_tracked(ctx, name, x0, y - 30, 0.12 * 19)
            xe = x0 + (xof(sec) - x0) * g
            if dashed:
                ctx.set_dash([8.0, 6.0])
                ctx.set_source_rgba(*rgb(col), 0.95 * a)
                ctx.set_line_width(2.5)
                S3.pill(ctx, x0, y, max(12, xe - x0), 40, 10)
                ctx.stroke()
                ctx.set_dash([])
            else:
                ctx.set_source_rgba(*rgb(col), 0.95 * a)
                S3.pill(ctx, x0, y, max(12, xe - x0), 40, 10)
                ctx.fill()
            font(ctx, "dbold", 32)
            s = f"{sec:.1f} s" if sec >= 1 else ms(sec)
            if dashed:
                s = "≈ " + s
            sw = ctx.text_extents(s).x_advance
            tx = xe + 16 if xe + 16 + sw < W - 40 else xe - sw - 16
            ctx.set_source_rgba(*rgb(INK if tx > xe else "#05060C"), a * g)
            ctx.move_to(tx, y + 31)
            ctx.show_text(s)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ the card

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        madds = f"{R['madds']:,}".replace(",", "{,}")
        if t < SEG[1] or t >= SEG[6]:
            paste(img, r"$c_{ij} = \sum_{k=1}^{512} a_{ik}\, b_{kj}$", 40, 540, y + 108, INK, a, align="c")
            paste(img, rf"$512 \times 512 \times 512 = {madds}\ \mathrm{{multiply\ adds}}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[2]:
            paste(img, r"$h = \mathrm{ReLU}(x\,W + b)$", 40, 540, y + 100, INK, a, align="c")
            paste(img, r"$x\,W:\ \mathrm{every\ input\ times\ every\ weight,\ added\ up}$", 26, 540, y + 162, INK2, a, align="c")
            paste(img, r"$\mathrm{inputs\ times\ a\ layer's\ weights:\ one\ matrix\ multiplication}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[3]:
            paste(img, r"$\mathrm{for}\ i:\ \ \mathrm{for}\ j:\ \ \mathrm{for}\ k:\ \ s \leftarrow s + a_{ik}\, b_{kj}$", 34, 540, y + 104, GOLD, a, align="c")
            paste(img, rf"$\mathrm{{about}}\ {R['python_madds_per_s'] / 1e6:.0f}\ \mathrm{{million\ multiply\ adds\ per\ second}}$", 26, 540, y + 166, INK2, a, align="c")
            paste(img, rf"$\mathrm{{Python}}\ {R['python']},\ \mathrm{{one\ at\ a\ time}}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[4]:
            paste(img, r"$C = A\,@\,B$", 44, 540, y + 104, CYAN, a, align="c")
            paste(img, rf"$\mathrm{{about}}\ {R['madds'] / T_NP1 / 1e9:.0f}\ \mathrm{{billion\ multiply\ adds\ per\ second\ on\ one\ core}}$", 26, 540, y + 166, INK2, a, align="c")
            paste(img, rf"$\mathrm{{NumPy}}\ {R['numpy']}\ \mathrm{{with\ OpenBLAS:\ compiled\ and\ vectorized}}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[5]:
            m, e = f"{R['max_abs_diff']:.1e}".split("e")
            paste(img, rf"$\max\,|C_{{\mathrm{{Python}}}} - C_{{\mathrm{{NumPy}}}}| = {m} \times 10^{{{int(e)}}}$", 32, 540, y + 100, INK, a, align="c")
            sp = f"{SPEEDUP:,.0f}".replace(",", "{,}")
            paste(img, rf"${T_PY:.1f}\,\mathrm{{s}}\ /\ {T_NPA * 1000:.2f}\,\mathrm{{ms}} = {sp}\times$", 30, 540, y + 160, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{same\ answer\ to\ the\ last\ rounding\ error;\ medians\ of\ repeated\ runs}$", 20, 540, y + h - 18, INK3, a, align="c")
        else:
            nm = f"{R['net_madds']:,}".replace(",", "{,}")
            paste(img, rf"$400 \times (2 \cdot 1000 + 1000 \cdot 1000 + 1000) = {nm}$", 27, 540, y + 96, INK, a, align="c")
            paste(img, rf"$\mathrm{{NumPy}}:\ {T_NET * 1000:.0f}\,\mathrm{{ms}}\qquad \mathrm{{Python}}:\ \approx {T_NET_PY:.0f}\,\mathrm{{s}}\ \mathrm{{(estimated)}}$", 28, 540, y + 156, INK2, a, align="c")
            paste(img, r"$\mathrm{one\ forward\ pass,\ 400\ points;\ Python\ at\ its\ measured\ speed}$", 20, 540, y + h - 18, INK3, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        a = 1.0
        img = self.bg.copy()
        bars_a = smooth(seg(t, BARS[0] - 0.4, BARS[0] + 0.2)) * (1 - smooth(seg(t, BARS[1] - 0.2, BARS[1] + 0.3)))
        net_a = smooth(seg(t, NET[0], NET[0] + 0.5)) * (1 - smooth(seg(t, NET[1] - 0.2, NET[1] + 0.4)))
        mat_a = max(0.0, 1 - max(bars_a, net_a))
        self.draw_matrices(img, t, a * mat_a)
        self.draw_bars(img, t, a * bars_a, [("PLAIN PYTHON", T_PY, GOLD, False), ("NUMPY, 1 CORE", T_NP1, CYAN, False), (f"NUMPY, {CORES} CORES", T_NPA, MAGENTA, False)],
                       "ONE 512 × 512 MULTIPLICATION  ·  TIMED ON THIS COMPUTER", BARS)
        self.draw_bars(img, t, a * net_a, [("PLAIN PYTHON (ESTIMATED)", T_NET_PY, GOLD, True), (f"NUMPY, {CORES} CORES", T_NET, MAGENTA, False)],
                       "ONE FORWARD PASS: 2 → 1000 → 1000 → 1, 400 POINTS", (NET[0] + 2.0, NET[1]))
        self.card(img, t, a)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "TIMED ON THIS COMPUTER   ·   SAME NUMBERS, SAME ANSWER"
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
