"""PageRank is an eigenvector. Dark theme of reels 31-58, maths in Computer Modern (stage3d.tex), narrated by Kokoro
(voicekit). A web of six pages; 100,000 surfers click links at random (15% of the time they jump to any page); within
a dozen steps the crowd stops changing: C holds 37.4%, A, with a single link in, 34.3%, because that link comes from
C. The steady crowd is the eigenvector of the click matrix with eigenvalue 1. Then the same rule on Python's standard
library: after sys, the second-ranked module is functools (imported by 37 modules), not os (imported by 79)
(prsim.py). On screen: the web with the surfers as dots, the share on each page, the eigenvector, and the ranked
modules. frame(f) -> uint8 1080x1920 RGB."""

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
if not os.path.exists(os.path.join(HERE, "pr_results.json")):
    import prsim
    prsim.main()
with open(os.path.join(HERE, "pr_results.json")) as fh:
    R = json.load(fh)
NAMES = R["names"]
LINKS = R["links"]
EIG = np.array(R["eigvec"])
SHARES = np.array(R["shares"])
WALKS = np.array(R["walks"])
STEPS = R["steps"]
TOP = R["top"]
INB = R["inbound"]
POS = dict(A=(300, 600), B=(780, 600), C=(540, 790), D=(220, 960), E=(540, 1030), F=(860, 960))
COL = dict(A=GREEN, B=CYAN, C=GOLD, D=MAGENTA, E=BLUE, F=VERM)


def pc(v, nd=1):
    return f"{100 * v:.{nd}f}"


assert (pc(EIG[2], 0), pc(EIG[0], 0), INB["C"], INB["A"], R["settle_step"] <= 12, R["n_modules"], R["n_links"], TOP[0]["module"], TOP[1]["module"],
        TOP[1]["imported_by"], R["place"]["os"], R["in_place"]["os"], [t["imported_by"] for t in TOP if t["module"] == "os"][0], R["n_surfers"]) == \
    ("37", "34", 5, 1, True, 185, 1254, "sys", "functools", 37, 5, 2, 79, 100000), "script numbers changed"
SCRIPT = [
    "Google's original ranking was a single eigenvector. Here it is, on a web of six pages.",
    "Each arrow is a link. Page C has five links pointing at it. Page A has just one.",
    "Drop a hundred thousand surfers on random pages. Every step, each one clicks a random link, or, fifteen percent of the time, jumps to any page.",
    "Within a dozen steps, the crowd stops changing. C ends up with thirty-seven percent of the surfers. A, with its one link, gets thirty-four, because that link comes from C.",
    "That steady crowd is an eigenvector. Write the clicks as a matrix, and multiplying by it leaves this vector unchanged: eigenvalue one. That's PageRank.",
    "Now Python's standard library: a hundred and eighty-five modules, one thousand two hundred and fifty-four imports. After sys, second place isn't os, imported by seventy-nine modules. It's functools, imported by thirty-seven, because the modules that matter import it.",
    "One matrix, multiplied until it stops changing. Google's founders described it in nineteen ninety-eight.",
]
TIMING = os.path.join(HERE, "narration_timing.json")
# 0 hook, 1 the web, 2 surfers, 3 it settles, 4 the eigenvector, 5 Python, 6 close
SEG_MIN = [6.5, 7.0, 9.5, 11.0, 10.0, 15.0, 7.0]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[7]
N_FRAMES = int(round((T_END + 0.7) * FPS))


def word_t(i, phrase):
    line = SCRIPT[i]
    if not VOICE_DUR:
        return SEG[i] + (SEG[i + 1] - SEG[i]) * line.index(phrase) / len(line)
    return VOICE_AT[i] + VOICE_DUR[i] * line.index(phrase) / len(line)


T_DROP = word_t(2, "Drop")
T_S0 = word_t(2, "Every step")
T_S1 = T_S0 + 5.6                                 # the first 7 steps, slowly
T_S2 = word_t(3, "C ends up")                     # then on to step 60
T_FIVE = word_t(1, "Page C")
T_ONE = word_t(1, "Page A")
T_MATRIX = word_t(4, "Write the clicks")
T_LIST = word_t(5, "After sys")
T_FT = word_t(5, "It's functools")

GRAPH_LABEL_Y = 500
BARS = (170, 1140, 740, 150)
CARD = (70, 1430, 940, 250)


def mode(t):
    if t < SEG[1] - 0.2 or t >= T_END - 0.6:
        return "hook"
    for k in range(1, 7):
        if t < SEG[k + 1]:
            return ("web", "surf", "settle", "eigen", "python", "close")[k - 1]
    return "close"


def step_at(t):
    """Surfer step (fractional) shown at time t; None before the surfers are dropped."""
    md = mode(t)
    if md in ("hook", "eigen", "python", "close"):
        return float(STEPS)
    if md == "web":
        return None
    if t < T_DROP:
        return None
    if t < T_S0:
        return 0.0
    if t < T_S1:
        return 7.0 * (t - T_S0) / (T_S1 - T_S0)
    if t < T_S2:
        return 7.0 + (STEPS - 7.0) * smooth(seg(t, T_S1, T_S2))
    return float(STEPS)


def shares_at(s):
    if s is None:
        return np.full(6, 1 / 6)
    k = int(math.floor(s))
    if k >= STEPS:
        return SHARES[STEPS]
    f = smooth(s - k)
    return SHARES[k] * (1 - f) + SHARES[k + 1] * f


def qa_frames():
    fr = []
    for k in range(7):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.35 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr + [int((T_S0 + 1.3) * FPS), int((T_FT + 1.0) * FPS), N_FRAMES - 1]


def radius(share):
    return 30 + 62 * math.sqrt(max(share, 0) / EIG.max())


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(eig=dict(zip(NAMES, [round(v, 4) for v in EIG])), settle=R["settle_step"], top=TOP[:6], python=R["python"], segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._build_bg()
        rng = np.random.default_rng(7)
        a = rng.random(R["n_show"]) * TAU
        rr = np.sqrt(rng.random(R["n_show"])) * 0.78
        self.jit = np.stack([np.cos(a) * rr, np.sin(a) * rr], 1)

    def _cues(self):
        S_ = SEG
        hook = [("Google's first ranking:", "black", 52, INK, 300, 1), ("one eigenvector.", "dbold", 50, GOLD, 390, 0.95)]
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("Six pages. Ten links.", "black", 54, INK, 300, 1), ("C: 5 links in.  A: just 1.", "dbold", 44, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("100,000 random surfers.", "black", 52, INK, 300, 1), ("Click a link, or jump (15%).", "dbold", 42, CYAN, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [("The crowd settles.", "black", 54, INK, 300, 1), (f"C {pc(EIG[2])}%.  A {pc(EIG[0])}%, from 1 link.", "dbold", 42, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("That crowd is an eigenvector.", "black", 48, INK, 300, 1), ("G r = r.  Eigenvalue 1.", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("Python's standard library:", "black", 50, INK, 300, 1), ("#2 is functools, not os.", "dbold", 46, GREEN, 390, 0.95)], 10, 8, "pop"),
            (S_[6], T_END - 0.3, [("One matrix, multiplied", "black", 52, INK, 300, 1), ("until it stops changing.", "dbold", 46, GOLD, 390, 0.95)], 10, 8, "pop"),
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

    # ------------------------------------------------------------ the web

    def arrow(self, ctx, p, q, rp, rq, bend, col, a, width=3.0):
        (x0, y0), (x1, y1) = p, q
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        cx, cy = (x0 + x1) / 2 + nx * bend, (y0 + y1) / 2 + ny * bend
        # start and end on the circles, toward the control point
        def onc(px, py, r, tx, ty):
            vx, vy = tx - px, ty - py
            ll = math.hypot(vx, vy)
            return px + vx / ll * r, py + vy / ll * r
        sx, sy = onc(x0, y0, rp + 4, cx, cy)
        ex, ey = onc(x1, y1, rq + 8, cx, cy)
        ctx.set_source_rgba(*rgb(col), a)
        ctx.set_line_width(width)
        ctx.move_to(sx, sy)
        ctx.curve_to(sx + (cx - sx) * 0.66, sy + (cy - sy) * 0.66, ex + (cx - ex) * 0.66, ey + (cy - ey) * 0.66, ex, ey)
        ctx.stroke()
        ang = math.atan2(ey - cy, ex - cx)
        h = 16
        ctx.move_to(ex + 4 * math.cos(ang), ey + 4 * math.sin(ang))
        ctx.line_to(ex - h * math.cos(ang - 0.42), ey - h * math.sin(ang - 0.42))
        ctx.line_to(ex - h * math.cos(ang + 0.42), ey - h * math.sin(ang + 0.42))
        ctx.close_path()
        ctx.fill()

    def web(self, ctx, t, a):
        md = mode(t)
        s = step_at(t)
        sh = shares_at(s)
        rad = {c: (52 if s is None else radius(sh[i])) for i, c in enumerate(NAMES)}
        hl_c = md == "web" and t >= T_FIVE - 0.1 and t < T_ONE - 0.1
        hl_a = md == "web" and t >= T_ONE - 0.1
        for src, ts in LINKS.items():
            for dst in ts:
                back = src in LINKS.get(dst, "")
                bend = 26 if back else 0
                col, al, wd = INK3, 0.9, 3.0
                if (hl_c and dst == "C") or (hl_a and dst == "A"):
                    col, al, wd = COL[dst], 1.0, 5.0
                self.arrow(ctx, POS[src], POS[dst], rad[src], rad[dst], bend, col, al * a, wd)
        # the surfers (600 of the 100,000), at or between pages
        if s is not None:
            k = int(math.floor(s))
            f = smooth(s - k) if k < STEPS else 0.0
            p0 = WALKS[min(k, STEPS)]
            p1 = WALKS[min(k + 1, STEPS)]
            fast = 7 <= s < STEPS
            for i in range(len(p0)):
                j0, j1 = int(p0[i]), int(p1[i])
                c0, c1 = NAMES[j0], NAMES[j1]
                x0, y0 = POS[c0][0] + self.jit[i, 0] * rad[c0], POS[c0][1] + self.jit[i, 1] * rad[c0]
                x1, y1 = POS[c1][0] + self.jit[i, 0] * rad[c1], POS[c1][1] + self.jit[i, 1] * rad[c1]
                ff = (1.0 if f > 0.5 else 0.0) if fast else f
                x, y = x0 + (x1 - x0) * ff, y0 + (y1 - y0) * ff
                ctx.set_source_rgba(*rgb(INK), 0.85 * a)
                ctx.arc(x, y, 2.6, 0, TAU)
                ctx.fill()
        for i, c in enumerate(NAMES):
            x, y = POS[c]
            r = rad[c]
            ctx.set_source_rgba(*rgb(PANEL_BASE), 0.55 * a)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(COL[c]), a)
            ctx.set_line_width(3.5)
            ctx.arc(x, y, r, 0, TAU)
            ctx.stroke()
            font(ctx, "black", 34)
            ctx.set_source_rgba(*rgb(COL[c]), a)
            lx = x - r - 34 if c in "AD" else x + r + 12
            if c in "CE":
                lx = x + r + 12
            ctx.move_to(lx, y + 12)
            ctx.show_text(c)
            font(ctx, "dbold", 20)
            ctx.set_source_rgba(*rgb(INK2), a)
            if md == "web":
                s_ = f"{INB[c]} link{'s' if INB[c] > 1 else ''} in"
            elif s is None:
                s_ = ""
            else:
                s_ = f"{pc(sh[i])}%"
            if s_:
                ex = ctx.text_extents(s_).x_advance
                ctx.move_to(lx if c not in "AD" else x - r - 12 - ex, y + 40)
                ctx.show_text(s_)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        lab = "A WEB OF SIX PAGES (MADE UP); EACH ARROW IS A LINK"
        if s is not None:
            lab = f"SURFERS: 600 OF THE 100,000 DRAWN  \u00b7  STEP {min(STEPS, int(s))}"
        show_tracked(ctx, lab, 70, GRAPH_LABEL_Y - 36, 0.1 * 16)

    def bars(self, ctx, t, a):
        md = mode(t)
        s = step_at(t)
        sh = shares_at(s)
        x0, y0, w, h = BARS
        bw = w / 6
        top = 0.40
        ctx.set_source_rgba(*rgb(INK3), 0.75 * a)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, y0 + h)
        ctx.line_to(x0 + w, y0 + h)
        ctx.stroke()
        show_eig = md in ("eigen", "close", "hook", "settle") and (md != "settle" or t >= T_S2)
        for i, c in enumerate(NAMES):
            bx = x0 + i * bw + bw * 0.2
            hh = h * sh[i] / top
            ctx.set_source_rgba(*rgb(COL[c]), 0.9 * a)
            ctx.rectangle(bx, y0 + h - hh, bw * 0.6, hh)
            ctx.fill()
            if show_eig:
                ey = y0 + h - h * EIG[i] / top
                ctx.set_source_rgba(1, 1, 1, 0.95 * a)
                ctx.set_line_width(2.4)
                ctx.move_to(bx - 8, ey)
                ctx.line_to(bx + bw * 0.6 + 8, ey)
                ctx.stroke()
            font(ctx, "black", 24)
            ctx.set_source_rgba(*rgb(COL[c]), a)
            ctx.move_to(bx + bw * 0.3 - ctx.text_extents(c).x_advance / 2, y0 + h + 28)
            ctx.show_text(c)
        font(ctx, "bold", 16)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        lab = "SHARE OF THE SURFERS ON EACH PAGE" + ("   \u2014  WHITE: THE EIGENVECTOR" if show_eig else "")
        show_tracked(ctx, lab, x0 - 100, y0 - 16, 0.1 * 16)

    # ------------------------------------------------------------ Python

    def modules(self, ctx, t, a):
        x0, y0 = 90, 520
        font(ctx, "bold", 17)
        ctx.set_source_rgba(*rgb(INK3), 0.95 * a)
        show_tracked(ctx, f"PYTHON {R['python']} STANDARD LIBRARY: {R['n_modules']} MODULES, {R['n_links']:,} IMPORTS", x0, y0 - 30, 0.1 * 17)
        font(ctx, "bold", 15)
        show_tracked(ctx, "RANK", x0, y0 + 6, 1.5)
        show_tracked(ctx, "PAGERANK", x0 + 400, y0 + 6, 1.5)
        show_tracked(ctx, "IMPORTED BY", x0 + 720, y0 + 6, 1.5)
        rows = TOP[:6]
        rmax = rows[0]["rank"]
        for k, row in enumerate(rows):
            ap = smooth(seg(t, SEG[5] + 0.3 + 0.18 * k, SEG[5] + 0.7 + 0.18 * k)) * a
            y = y0 + 60 + k * 112
            m = row["module"]
            col = GREEN if m == "functools" else (GOLD if m == "os" else INK2)
            emph = (m == "functools" and t >= T_FT - 0.2) or (m == "os" and t >= T_LIST)
            if emph:
                ctx.set_source_rgba(*rgb(col), 0.10 * ap)
                ctx.rectangle(x0 - 20, y - 50, 920, 96)
                ctx.fill()
            font(ctx, "black", 40)
            ctx.set_source_rgba(*rgb(col if emph else INK3), ap)
            ctx.move_to(x0, y + 14)
            ctx.show_text(f"{row['place']}")
            font(ctx, "black", 44)
            ctx.set_source_rgba(*rgb(col if emph or m == "sys" else INK), ap)
            ctx.move_to(x0 + 70, y + 14)
            ctx.show_text(m)
            bw = 220 * row["rank"] / rmax
            ctx.set_source_rgba(*rgb(col if emph else INK3), 0.9 * ap)
            ctx.rectangle(x0 + 400, y - 10, bw, 22)
            ctx.fill()
            font(ctx, "dbold", 22)
            ctx.set_source_rgba(*rgb(INK2), ap)
            ctx.move_to(x0 + 400 + bw + 10, y + 9)
            ctx.show_text(f"{pc(row['rank'])}%")
            font(ctx, "black", 36)
            s_ = f"{row['imported_by']}"
            ctx.set_source_rgba(*rgb(col if emph else INK2), ap)
            ctx.move_to(x0 + 720, y + 14)
            ctx.show_text(s_)
            font(ctx, "dbold", 19)
            ctx.set_source_rgba(*rgb(INK3), ap)
            ctx.move_to(x0 + 720 + ctx.text_extents(s_).x_advance + 64, y + 10)
            ctx.show_text(f"(#{row['place_by_imports']})")

    # ------------------------------------------------------------ the frame

    def draw_main(self, img, t, a):
        md = mode(t)
        surf, ctx = layer()
        if md == "python":
            self.modules(ctx, t, a)
        else:
            self.web(ctx, t, a)
            if step_at(t) is not None:
                self.bars(ctx, t, a)
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
            paste(img, r"$G\,r = r,\qquad G = 0.85\,M + 0.15 \cdot \frac{1}{6}$", 34, 540, y + 96, GOLD, a, align="c")
            paste(img, r"$M_{ij} = 1\,/\,(\mathrm{links\ out\ of}\ j)\ \ \mathrm{if}\ j\ \mathrm{links\ to}\ i,\ \mathrm{else}\ 0$", 23, 540, y + 162, INK2, a, align="c")
            paste(img, r"$r:\ \mathrm{the\ eigenvector\ of}\ G\ \mathrm{with\ eigenvalue}\ 1\ \mathrm{(Brin\ and\ Page,\ 1998)}$", 21, 540, y + h - 20, INK3, a, align="c")
        elif md == "web":
            paste(img, r"$\mathrm{each\ page\ splits\ its\ visitors\ evenly\ among\ its\ links}$", 28, 540, y + 100, INK, a, align="c")
            paste(img, r"$\mathrm{E\ links\ to\ C,\ D,\ F:\ each\ gets}\ \frac{1}{3}\ \mathrm{of\ E's\ clicks}$", 26, 540, y + 168, INK2, a, align="c")
        elif md == "surf":
            paste(img, r"$\mathrm{each\ step:\ chance}\ 0.85,\ \mathrm{click\ one\ of\ the\ page's\ links\ at\ random}$", 22, 540, y + 96, CYAN, a, align="c")
            paste(img, r"$\mathrm{with\ chance}\ 0.15,\ \mathrm{jump\ to\ any\ of\ the\ 6\ pages}$", 23, 540, y + 150, INK2, a, align="c")
            paste(img, r"$100{,}000\ \mathrm{surfers,\ started\ on\ random\ pages\ (600\ drawn)}$", 20, 540, y + h - 22, INK3, a, align="c")
        elif md == "settle":
            paste(img, rf"$\mathrm{{after\ {R['settle_step']}\ steps\ the\ shares\ stay\ within\ 0.1\ points\ of\ their\ final\ values}}$", 22, 540, y + 96, INK, a, align="c")
            paste(img, r"$\mathrm{A's\ one\ link\ is\ from\ C,\ which\ sends\ all\ its\ clicks\ to\ A}$", 25, 540, y + 156, GREEN, a, align="c")
            paste(img, r"$\mathrm{C's\ five\ come\ from\ A,\ B,\ D,\ E,\ F}$", 21, 540, y + h - 22, INK3, a, align="c")
        elif md == "eigen":
            am = smooth(seg(t, T_MATRIX - 0.3, T_MATRIX + 0.4))
            e = EIG
            paste(img, rf"$r = ({e[0]:.3f},\ {e[1]:.3f},\ {e[2]:.3f},\ {e[3]:.3f},\ {e[4]:.3f},\ {e[5]:.3f})$", 28, 540, y + 92, INK, a, align="c")
            paste(img, r"$G\,r = 1 \cdot r$", 40, 540, y + 162, GOLD, a * am, align="c")
            gap = np.abs(SHARES[-1] - EIG).max()
            paste(img, rf"$\mathrm{{the\ 100{{,}}000\ surfers\ after\ {STEPS}\ steps:\ within\ {100 * gap:.1f}\ points\ of}}\ r$", 20, 540, y + h - 20, INK3, a, align="c")
        else:
            nc = R["n_modules"] - R["n_with_source"]
            paste(img, r"$\mathrm{a\ link:\ module\ X\ imports\ module\ Y;\ same\ 0.85\ /\ 0.15\ rule}$", 24, 540, y + 96, INK, a, align="c")
            ip = R["importers_places"]
            paste(img, rf"$\mathrm{{functools\ is\ imported\ by\ \#{ip['functools'][0]}\ and\ \#{ip['functools'][1]};\ os'\ best\ importer\ is\ \#{ip['os'][0]}}}$", 23, 540, y + 156, GREEN, a, align="c")
            paste(img, rf"$\mathrm{{{nc}\ modules\ are\ written\ in\ C\ (sys,\ math,\ \ldots):\ no\ source,\ so\ no\ links\ out}}$", 20, 540, y + h - 20, INK3, a, align="c")

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
        lab = "PAGERANK   \u00b7   100,000 SIMULATED SURFERS   \u00b7   ALL COMPUTED"
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
