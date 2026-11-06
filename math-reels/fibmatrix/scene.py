"""Every Fibonacci number is a power of one matrix, and the golden ratio is its eigenvalue. Dark theme of reels 31-60,
maths in Computer Modern (stage3d.tex), narrated by Kokoro (voicekit). A = [[1, 1], [1, 0]] sends (a, b) to (a + b, a):
from (1, 0) the arrow runs through the Fibonacci pairs; from any start its direction swings onto the eigenvector
(phi, 1), and each step stretches it by phi = 1.618...; the other eigenvalue (-0.618) shrinks away, so phi^100 / sqrt 5
is F(100) to within 6 x 10^-22; and repeated squaring finds F(1,000,000) with 25 matrix multiplications (fibsim.py).
On screen: the plane with the arrows and both eigenvector lines, the multiplication, the ratios, the digits, the
timings. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import json
import math
import os
import sys
from decimal import Decimal

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
if not os.path.exists(os.path.join(HERE, "fib_results.json")):
    import fibsim
    fibsim.main()
with open(os.path.join(HERE, "fib_results.json")) as fh:
    R = json.load(fh)
PATHS = R["paths"]
EIG = R["eig_angle"]
PSI_ANG = R["psi_angle"]
PHI = R["phi"]
BIG = R["big"]
RAT = {r["n"]: r for r in R["ratios"]}
ERR = Decimal(R["binet_err"])
PCOL = [GOLD, CYAN, MAGENTA]


assert (R["F100"], PHI[:5], round(BIG["digits"], -3), f"{BIG['t_add']:.1f}", round(BIG["t_mat"] * 4) / 4, BIG["mults"], ERR < Decimal("1e-21"), RAT[6]["num"], RAT[10]["num"]) == \
    ("354224848179261915075", "1.618", 209000, "7.7", 0.25, 25, True, 13, 89), "script numbers changed"
SCRIPT = [
    "Every Fibonacci number is hiding in the powers of one tiny matrix. And its eigenvalue is the golden ratio.",
    "Take this matrix: one, one, one, zero. Times the pair one, zero, you get one, one. Again: two, one. Then three, two. Five, three. Each step adds the pair and shifts it: the Fibonacci rule.",
    "Now watch the arrow's direction. Start it anywhere: within a few steps it swings onto the same line, and stays there. That line is an eigenvector. Along it, the matrix only stretches.",
    "And every step stretches it by the same factor. Thirteen over eight, twenty-one over thirteen, eighty-nine over fifty-five: one point six one eight. The golden ratio. It's the eigenvalue: lambda squared equals lambda plus one.",
    "The other eigenvalue, minus zero point six one eight, shrinks away. So the hundredth Fibonacci number is the golden ratio to the hundredth power over root five, off by less than a billionth of a trillionth.",
    "And squaring the matrix is fast. The millionth Fibonacci number has two hundred and nine thousand digits. A million additions: seven point seven seconds. Twenty-five matrix multiplications: a quarter of a second.",
    "One matrix. Its eigenvector is the golden direction, its eigenvalue the golden ratio, and every Fibonacci number falls out.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the matrix, 2 the direction, 3 the ratio, 4 Binet, 5 speed, 6 close
SEG_MIN = [6.5, 14.0, 11.0, 13.0, 12.0, 13.0, 7.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_M = [word_t(1, "you get"), word_t(1, "Again"), word_t(1, "Then three"), word_t(1, "Five, three")]
T_DIR = word_t(2, "Start it anywhere")
T_EIGLINE = word_t(2, "That line")
DIR_DT = 0.62
T_LAMBDA = word_t(3, "It's the eigenvalue")
T_PSI = word_t(4, "So the hundredth")
T_ADD = word_t(5, "A million additions")
T_MAT = word_t(5, "Twenty-five")

O = (540, 790)            # the origin of the plane
UNIT = 44                 # pixels per unit (the real-size arrows)
RAD = 270                 # length of the direction arrows
LOW_Y = 1150
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("matrix", "direction", "ratio", "binet", "speed", "close")[k - 1]
    return "close"


def m_step(t):
    """In the matrix segment: how many multiplications have happened (fractional), 0..4."""
    k = 0.0
    for i, tm in enumerate(T_M):
        k = max(k, i + smooth(seg(t, tm, tm + 0.45)))
    return k if t >= T_M[0] else 0.0


def d_step(t):
    """Direction steps shown (fractional), 0..STEPS; full after the direction segment."""
    md = mode(t)
    if md == "direction":
        return float(np.clip((t - T_DIR) / DIR_DT, 0, R["steps"])) if t >= T_DIR else 0.0
    if md == "matrix":
        return None
    return float(R["steps"])


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_M[3] + 1.0) * FPS), int((T_DIR + 2.0) * FPS), int((T_MAT + 1.5) * FPS), N_FRAMES - 1]


def fmt_int(s):
    s = str(s)
    neg = s.startswith("-")
    s = s.lstrip("-")
    out = ",".join([s[max(0, i - 3):i] for i in range(len(s), 0, -3)][::-1])
    return ("−" if neg else "") + out


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(F100=R["F100"], binet_err=R["binet_err"][:10], big={k: BIG[k] for k in ("digits", "t_add", "t_mat", "mults")}, segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()

    def _cues(self):
        S_ = SEG
        hook = [("Every Fibonacci number:", "black", 52, INK, 300, 1), ("one matrix, multiplied.", "dbold", 48, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("(a, b) → (a + b, a)", "black", 54, INK, 300, 1), ("1, 1, 2, 3, 5, 8, …", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Start anywhere.", "black", 54, INK, 300, 1), ("It swings onto one line.", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("Each step: × 1.618…", "black", 54, INK, 300, 1), ("The golden ratio is the eigenvalue.", "dbold", 40, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("F(100) = φ¹⁰⁰ / √5", "black", 54, INK, 300, 1), ("to 21 decimal places.", "dbold", 46, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("The millionth: 209,000 digits.", "black", 48, INK, 300, 1), (f"{BIG['t_add']:.1f} s adding, {BIG['t_mat']:.2f} s squaring.", "dbold", 44, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("One matrix.", "black", 56, INK, 300, 1), ("Golden ratio = eigenvalue.", "dbold", 48, GOLD, 390, 0.95)], 10, 8, "pop"),
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

    # ------------------------------------------------------------ drawing helpers

    @staticmethod
    def arrow(ctx, x0, y0, x1, y1, col, a, width=5.0, head=20):
        ang = math.atan2(y1 - y0, x1 - x0)
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1:
            return
        ex, ey = x1 - head * 0.7 * math.cos(ang), y1 - head * 0.7 * math.sin(ang)
        ctx.set_source_rgba(*rgb(col), a)
        ctx.set_line_width(width)
        ctx.move_to(x0, y0)
        ctx.line_to(ex, ey)
        ctx.stroke()
        ctx.move_to(x1, y1)
        ctx.line_to(x1 - head * math.cos(ang - 0.4), y1 - head * math.sin(ang - 0.4))
        ctx.line_to(x1 - head * math.cos(ang + 0.4), y1 - head * math.sin(ang + 0.4))
        ctx.close_path()
        ctx.fill()

    @staticmethod
    def bracket(ctx, x, y0, y1, left, col, a):
        ctx.set_source_rgba(*rgb(col), a)
        ctx.set_line_width(2.6)
        d = 12 if left else -12
        ctx.move_to(x + d, y0)
        ctx.line_to(x, y0)
        ctx.line_to(x, y1)
        ctx.line_to(x + d, y1)
        ctx.stroke()

    def matrix(self, img, ctx, x, ymid, rows, size, col, a, cw=None):
        """A matrix of Computer Modern entries with square brackets; returns its width."""
        nr, nc = len(rows), len(rows[0])
        cw = cw or [max(48, max(len(str(r[j])) for r in rows) * size * 0.62) for j in range(nc)]
        lh = size * 1.55
        w = sum(cw) + 30
        top = ymid - nr * lh / 2
        cx = x + 15
        for j in range(nc):
            for i in range(nr):
                v = rows[i][j]
                paste(img, f"${v}$", size, cx + cw[j] / 2, top + (i + 0.72) * lh, col if not isinstance(col, list) else col[i][j], a, align="c")
            cx += cw[j]
        self.bracket(ctx, x, top, top + nr * lh, True, INK2, a)
        self.bracket(ctx, x + w, top, top + nr * lh, False, INK2, a)
        return w

    # ------------------------------------------------------------ the plane

    def plane(self, ctx, t, a):
        md = mode(t)
        ox, oy = O
        # axes and grid
        ctx.set_source_rgba(1, 1, 1, 0.08 * a)
        ctx.set_line_width(1.0)
        for k in range(-10, 11):
            ctx.move_to(ox + k * UNIT, oy - 300)
            ctx.line_to(ox + k * UNIT, oy + 300)
            ctx.move_to(ox - 440, oy + k * UNIT)
            ctx.line_to(ox + 440, oy + k * UNIT)
        ctx.stroke()
        ctx.set_source_rgba(*rgb(INK3), 0.9 * a)
        ctx.set_line_width(1.6)
        ctx.move_to(ox - 440, oy)
        ctx.line_to(ox + 440, oy)
        ctx.move_to(ox, oy - 300)
        ctx.line_to(ox, oy + 300)
        ctx.stroke()
        s = d_step(t)
        show_lines = md not in ("matrix",) and (md != "direction" or t >= T_EIGLINE - 0.3)
        if md == "direction":
            la = smooth(seg(t, T_EIGLINE - 0.3, T_EIGLINE + 0.4))
        else:
            la = 1.0
        if show_lines:
            for ang, col, dash, lab in ((EIG, GOLD, None, "eigenvector (φ, 1): stretched × 1.618"), (PSI_ANG, INK3, [10, 8], "eigenvector (ψ, 1): × (−0.618), shrinks")):
                c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                L = 300 / max(abs(s_), 1e-9) if abs(s_) * 440 > abs(c) * 300 else 440 / abs(c)
                ctx.set_source_rgba(*rgb(col), 0.95 * la * a)
                ctx.set_line_width(2.6 if dash is None else 2.0)
                if dash:
                    ctx.set_dash(dash)
                ctx.move_to(ox - c * L, oy + s_ * L)
                ctx.line_to(ox + c * L, oy - s_ * L)
                ctx.stroke()
                ctx.set_dash([])
                font(ctx, "dbold", 19)
                ctx.set_source_rgba(*rgb(col), la * a)
                if dash is None:
                    ctx.move_to(ox + 150, oy - 300 + 22)
                else:
                    ctx.move_to(ox - 430, oy - 60)
                ctx.show_text(lab)
        if md == "matrix":
            k = m_step(t)
            pairs = PATHS[0]["pairs"]
            i = int(math.floor(k))
            f = k - i
            p0, p1 = pairs[i], pairs[min(i + 1, len(pairs) - 1)]
            x = p0[0] + (p1[0] - p0[0]) * f
            y = p0[1] + (p1[1] - p0[1]) * f
            for j in range(i):                        # the earlier pairs, faint
                q = pairs[j]
                self.arrow(ctx, ox, oy, ox + q[0] * UNIT, oy - q[1] * UNIT, GOLD, 0.22 * a, 3.0, 14)
            self.arrow(ctx, ox, oy, ox + x * UNIT, oy - y * UNIT, GOLD, a)
            font(ctx, "black", 30)
            q = pairs[i] if f < 0.5 else p1
            lab = f"({q[0]}, {q[1]})"
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.move_to(ox + x * UNIT + 14, oy - y * UNIT - 14)
            ctx.show_text(lab)
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, "THE PAIR AS AN ARROW (ONE GRID SQUARE = 1)", 100, 470, 1.6)
            return
        # directions only: every arrow drawn the same length
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, f"DIRECTION ONLY (ARROWS DRAWN THE SAME LENGTH)  ·  STEP {int(s)}", 100, 470, 1.6)
        widths = [4.5, 8.0, 6.0]
        for pi in (2, 1, 0):                                  # the Fibonacci arrow last, thin, on top
            path = PATHS[pi]
            angs = path["angles"]
            i = int(math.floor(s))
            f = smooth(s - i) if i < len(angs) - 1 else 0.0
            a0, a1 = angs[min(i, len(angs) - 1)], angs[min(i + 1, len(angs) - 1)]
            d = (a1 - a0 + 180) % 360 - 180
            ang = math.radians(a0 + d * f)
            for j in range(min(i, len(angs) - 1) + 1):
                aj = math.radians(angs[j])
                ctx.set_source_rgba(*rgb(PCOL[pi]), 0.35 * a)
                ctx.arc(ox + RAD * math.cos(aj), oy - RAD * math.sin(aj), 5, 0, TAU)
                ctx.fill()
            self.arrow(ctx, ox, oy, ox + RAD * math.cos(ang), oy - RAD * math.sin(ang), PCOL[pi], a, widths[pi], 20 + 2 * (widths[pi] - 4.5))
        # legend: the starts
        font(ctx, "dbold", 20)
        for pi, path in enumerate(PATHS):
            s0 = path["start"]
            ctx.set_source_rgba(*rgb(PCOL[pi]), a)
            ctx.move_to(800, oy + 222 + 28 * pi)
            ctx.show_text(f"start ({s0[0]}, {s0[1]})".replace("-", "−"))

    # ------------------------------------------------------------ the lower panel

    def lower(self, img, ctx, t, a):
        md = mode(t)
        y = LOW_Y
        if md == "matrix":
            k = m_step(t)
            i = min(int(round(k)), 4)
            pairs = PATHS[0]["pairs"]
            p, q = pairs[max(i - 1, 0)], pairs[i]
            if i == 0:
                p, q = pairs[0], pairs[1]
            xx = 120
            w = self.matrix(img, ctx, xx, y + 70, [[1, 1], [1, 0]], 40, INK, a)
            w2 = self.matrix(img, ctx, xx + w + 30, y + 70, [[p[0]], [p[1]]], 40, GOLD, a)
            paste(img, "$=$", 44, xx + w + w2 + 80, y + 84, INK2, a, align="c")
            ar = a if (k >= 0.5 or t >= T_M[0]) else 0.35 * a
            self.matrix(img, ctx, xx + w + w2 + 120, y + 70, [[f"{p[0]} + {p[1]}"], [p[0]]], 40, GREEN, ar, cw=[170])
            font(ctx, "dbold", 22)
            ctx.set_source_rgba(*rgb(INK2), a)
            seq = ", ".join(str(pp[0]) for pp in pairs[1:i + 2]) if i else "1"
            ctx.move_to(120, y + 200)
            ctx.show_text(f"so far: {seq}, …")
        elif md == "direction":
            s = d_step(t)
            i = int(min(s, R["steps"]))
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, "THE PAIRS", 100, y + 10, 1.6)
            for pi, path in enumerate(PATHS):
                pr = path["pairs"][max(0, i - 3):i + 1]
                txt = "   ".join(f"({fmt_int(q[0])}, {fmt_int(q[1])})" for q in pr)
                font(ctx, "dbold", 26)
                ctx.set_source_rgba(*rgb(PCOL[pi]), a)
                ctx.move_to(100, y + 56 + 48 * pi)
                ctx.show_text(txt)
        elif md in ("ratio", "hook", "close"):
            rows = [6, 7, 8, 9, 10, 11, 12]
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, "EACH STEP'S STRETCH  (GREEN: DIGITS OF φ = 1.6180339887… ALREADY RIGHT)", 100, y - 4, 1.6)
            for k, n in enumerate(rows):
                ap = a if md != "ratio" else smooth(seg(t, SEG[3] + 0.4 + 0.55 * k, SEG[3] + 0.8 + 0.55 * k)) * a
                if ap <= 0:
                    continue
                r = RAT[n]
                yy = y + 34 + k * 34
                font(ctx, "dbold", 24)
                ctx.set_source_rgba(*rgb(INK2), ap)
                lab = f"{r['num']} / {r['den']}"
                ctx.move_to(260 - ctx.text_extents(lab).x_advance, yy)
                ctx.show_text(lab)
                rs = r["ratio"][:12].ljust(12, "0")
                good = 2 + r["right"]
                font(ctx, "black", 26)
                x = 300
                ctx.set_source_rgba(*rgb(GREEN), ap)
                ctx.move_to(x, yy)
                ctx.show_text(rs[:good])
                x += ctx.text_extents(rs[:good]).x_advance
                ctx.set_source_rgba(*rgb(INK3), ap)
                ctx.move_to(x, yy)
                ctx.show_text(rs[good:])
        elif md == "binet":
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, "THE HUNDREDTH FIBONACCI NUMBER, TWO WAYS", 100, y - 4, 1.6)
            b = R["binet100"]
            ip, fp = b.split(".")
            font(ctx, "dbold", 22)
            ctx.set_source_rgba(*rgb(INK2), a)
            ctx.move_to(100, y + 42)
            ctx.show_text("adding, 99 times:")
            font(ctx, "black", 34)
            ctx.set_source_rgba(*rgb(GOLD), a)
            ctx.move_to(100, y + 86)
            ctx.show_text(fmt_int(R["F100"]))
            ab = smooth(seg(t, T_PSI, T_PSI + 0.6)) * a
            font(ctx, "dbold", 22)
            ctx.set_source_rgba(*rgb(INK2), ab)
            ctx.move_to(100, y + 140)
            ctx.show_text("φ¹⁰⁰ / √5:")
            font(ctx, "black", 34)
            ctx.set_source_rgba(*rgb(GOLD), ab)
            ctx.move_to(100, y + 184)
            s1 = fmt_int(ip) + "."
            ctx.show_text(s1)
            x = 100 + ctx.text_extents(s1).x_advance
            font(ctx, "dbold", 22)
            tail = fp[:25]
            zeros = len(tail) - len(tail.lstrip("0"))
            ctx.set_source_rgba(*rgb(GREEN), ab)
            ctx.move_to(x + 4, y + 184)
            ctx.show_text(tail[:zeros])
            x2 = x + 4 + ctx.text_extents(tail[:zeros]).x_advance
            ctx.set_source_rgba(*rgb(VERM), ab)
            ctx.move_to(x2, y + 184)
            ctx.show_text(tail[zeros:zeros + 3] + "…")
            font(ctx, "dbold", 19)
            ctx.set_source_rgba(*rgb(INK3), ab)
            ctx.move_to(100, y + 220)
            ctx.show_text(f"{zeros} zeros after the point before it differs")
        elif md == "speed":
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
            show_tracked(ctx, f"F(1,000,000): {BIG['digits']:,} DIGITS, {BIG['head']}…{BIG['tail']}", 100, y - 4, 1.6)
            rows = [("a million additions", BIG["t_add"], VERM, T_ADD), (f"{BIG['mults']} matrix multiplications", BIG["t_mat"], GREEN, T_MAT)]
            for k, (name, sec, col, t0) in enumerate(rows):
                yy = y + 50 + k * 92
                g = smooth(seg(t, t0 - 0.2, t0 + 0.8))
                font(ctx, "dbold", 24)
                ctx.set_source_rgba(*rgb(INK2), a)
                ctx.move_to(100, yy)
                ctx.show_text(name)
                ctx.set_source_rgba(*rgb(PANEL_BASE), a)
                ctx.rectangle(100, yy + 14, 700, 26)
                ctx.fill()
                ctx.set_source_rgba(*rgb(col), a)
                ctx.rectangle(100, yy + 14, max(3, 700 * g * sec / BIG["t_add"]), 26)
                ctx.fill()
                if g > 0.05:
                    font(ctx, "black", 34)
                    ctx.set_source_rgba(*rgb(col), a * min(1, g * 2))
                    ctx.move_to(820, yy + 40)
                    ctx.show_text(f"{sec:.2f} s")
            font(ctx, "dbold", 18)
            ctx.set_source_rgba(*rgb(INK3), a)
            ctx.move_to(100, y + 240)
            ctx.show_text(f"this machine, Python {R['python']}, median of 3 runs; both give the same {BIG['digits']:,} digits")

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        surf, ctx = layer()
        self.plane(ctx, t, a)
        self.lower(img, ctx, t, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    def card(self, img, t, a):
        x, y, w_, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w_, h, a)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        md = mode(t)
        if md in ("hook", "close"):
            paste(img, r"$A^n \binom{1}{0} = \binom{F_{n+1}}{F_n},\qquad A\binom{a}{b} = \binom{a + b}{a}$", 32, 540, y + 100, GOLD, a, align="c")
            paste(img, r"$\lambda^2 = \lambda + 1:\quad \varphi = \frac{1 + \sqrt{5}}{2} = 1.618\ldots,\quad \psi = \frac{1 - \sqrt{5}}{2} = -0.618\ldots$", 25, 540, y + 182, INK2, a, align="c")
        elif md == "matrix":
            paste(img, r"$A \binom{a}{b} = \binom{a + b}{a}:\ \ \mathrm{add\ the\ pair,\ shift\ it\ along}$", 32, 540, y + 104, INK, a, align="c")
            paste(img, r"$\mathrm{from}\ (1,\ 0)\ \mathrm{the\ pairs\ are}\ (F_{n+1},\ F_n):\ 1,\ 1,\ 2,\ 3,\ 5,\ 8,\ 13,\ \ldots$", 22, 540, y + 184, INK2, a, align="c")
        elif md == "direction":
            paste(img, r"$A \binom{\varphi}{1} = \binom{\varphi + 1}{\varphi} = \varphi \binom{\varphi}{1}\qquad (\mathrm{because}\ \varphi^2 = \varphi + 1)$", 30, 540, y + 108, GOLD, a, align="c")
            paste(img, r"$\mathrm{any\ start\ off\ the\ dashed\ line\ ends\ up\ along}\ (\varphi,\ 1)\ \mathrm{or}\ (-\varphi,\ -1)$", 22, 540, y + 190, INK2, a, align="c")
        elif md == "ratio":
            al = smooth(seg(t, T_LAMBDA - 0.2, T_LAMBDA + 0.5))
            paste(img, r"$\frac{F_{n+1}}{F_n} \;\rightarrow\; \varphi = 1.6180339887\ldots$", 34, 540, y + 100, GREEN, a, align="c")
            paste(img, r"$\det(A - \lambda I) = \lambda^2 - \lambda - 1 = 0$", 28, 540, y + 182, GOLD, a * al, align="c")
        elif md == "binet":
            paste(img, r"$F_n = \frac{\varphi^n - \psi^n}{\sqrt{5}},\qquad |\psi| = 0.618\ldots < 1$", 32, 540, y + 100, GOLD, a, align="c")
            err = f"{float(ERR):.1e}".split("e")
            paste(img, rf"$\frac{{\varphi^{{100}}}}{{\sqrt{{5}}}} - F_{{100}} = \frac{{\psi^{{100}}}}{{\sqrt{{5}}}} = {err[0]} \times 10^{{{int(err[1])}}}$", 28, 540, y + 190, GREEN, a, align="c")
        else:
            paste(img, rf"$A^{{1{{,}}000{{,}}000}}:\ {BIG['squarings']}\ \mathrm{{squarings}} + {BIG['extra']}\ \mathrm{{multiplications}} = {BIG['mults']}$", 30, 540, y + 100, GREEN, a, align="c")
            paste(img, r"$A^{8} = ((A^2)^2)^2:\ \mathrm{each\ squaring\ doubles\ the\ power}$", 24, 540, y + 176, INK2, a, align="c")

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
        lab = "EXACT INTEGERS AND 80-DIGIT DECIMALS   ·   ALL COMPUTED"
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
