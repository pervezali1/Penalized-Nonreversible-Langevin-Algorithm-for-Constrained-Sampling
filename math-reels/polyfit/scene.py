"""Overfitting, measured. Dark theme of reels 31-42, maths in Computer Modern (stage3d.tex), narrated by Kokoro
(voicekit). 20 noisy points and polynomials of rising degree (pfsim.py): the error on the 20 points keeps falling
and degree 19 passes through all of them, but on 5,000 new points from the same source the error falls, then
explodes (a U on a log scale). The best degree is 3. Two fixes, measured: 200 points instead of 20, or a small
penalty on big coefficients (its strength chosen by cross-validation on the 20 points). frame(f) -> uint8 1080x1920 RGB."""

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
sys.path.insert(0, HERE)
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

RES_P = os.path.join(HERE, "pf_results.json")
if not os.path.exists(RES_P):
    import pfsim
    with open(RES_P, "w") as fh:
        json.dump(pfsim.main(), fh)
with open(RES_P) as fh:
    R = json.load(fh)
XS, YS = np.array(R["x"]), np.array(R["y"])
XB, YB = np.array(R["xb"]), np.array(R["yb"])
XT, YT = np.array(R["xt"]), np.array(R["yt"])
GRID = np.array(R["grid"])
CURVES = np.array(R["curves"])
TRAIN, TEST = np.array(R["train"]), np.array(R["test"])
BIG_CURVE, RIDGE_CURVE, TRUTH = np.array(R["big_curve"]), np.array(R["ridge_curve"]), np.array(R["truth"])
BEST = R["best_degree"]
DMAX = len(CURVES) - 1
_band = [d for d in range(DMAX + 1) if TEST[d] <= 1.01 * TEST[BEST]]
BAND = (min(_band), max(_band))   # the degrees within 1% of the best on new points
assert list(range(BAND[0], BAND[1] + 1)) == _band

PLOT = (90, 470, 900, 560)                         # x0, y0, w, h
PX, PY = (-1.05, 1.05), (-2.3, 2.3)
CHART = (150, 1240, 830, 150)
LOG_LO, LOG_HI = -2.0, 5.0
CARD = (70, 1430, 940, 250)

WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def say(v):
    """0.26 -> 'zero point two six'."""
    s = f"{v:.2f}"
    a, b = s.split(".")
    return f"{WORDS[a] if a in WORDS else a} point {' '.join(WORDS[c] for c in b)}"


def err_txt(v):
    if v < 0.005:
        return "0.00"
    if v >= 10:
        return f"{v:,.0f}"
    return f"{v:.2f}"


SCRIPT = [
    "This curve passes through every single point. That's the problem.",
    "Twenty noisy points. Let's fit a polynomial, starting with a straight line.",
    "Raise the degree, and the curve bends to follow the points. The error on these twenty points keeps falling.",
    f"At degree {DMAX}, it passes through all twenty. Error: zero.",
    f"Now test it on five thousand new points from the same source. The error jumps to {round(TEST[DMAX])}.",
    f"Too simple misses the shape. Too flexible memorizes the noise. Degrees {BAND[0]} to {BAND[1]} all do about the same: {say(TEST[BEST])}.",
    f"Fix one: more data. With two hundred points, degree {DMAX} behaves: {say(R['big_test'])} on new points.",
    f"Fix two: a small penalty on big coefficients. Back to twenty points, degree {DMAX}: {say(R['ridge_test'])}.",
    "A perfect score on your own data isn't the goal. The error on new data is.",
]
TIMING = os.path.join(HERE, "narration_timing.json")

# ---------------------------------------------------------------- the timeline, sized to the narration
# 0 hook, 1 a line, 2 degree up, 3 degree 19, 4 new points, 5 the U, 6 more data, 7 penalty, 8 close
SEG_MIN = [4.0, 6.0, 8.5, 6.0, 8.0, 8.5, 8.0, 8.5, 6.5]
VOICE_DUR = voicekit.load_timing(TIMING, SCRIPT)
SEG, VOICE_AT = voicekit.segments(SEG_MIN, VOICE_DUR)
T_END = SEG[9]
N_FRAMES = int(round((T_END + 0.7) * FPS))
UP1 = (SEG[2] + 0.6, SEG[3] - 0.6)                 # degree 1 -> 9
UP2 = (SEG[3] + 0.2, SEG[3] + 2.8)                 # 9 -> 19
DOWN = (SEG[5] + 3.4, SEG[5] + 5.6)                # 19 -> the best degree
TEST_IN = (SEG[4] + 0.6, SEG[4] + 1.6)
SWEEP = (SEG[4] + 2.0, SEG[5] + 1.5)               # the test-error line drawn across the degrees
BIG = (SEG[6] + 0.6, SEG[6] + 2.4)
RIDGE = (SEG[7] + 0.6, SEG[7] + 2.6)
HOME = (T_END - 1.6, T_END - 0.2)                  # back to the opening for the loop


def deg_at(t):
    """The degree shown (fractional while morphing)."""
    if t < SEG[1] - 0.2:
        return float(DMAX)
    if t < UP1[0]:
        return float(DMAX) + (1 - DMAX) * smooth(seg(t, SEG[1] - 0.2, SEG[1] + 0.8))
    if t < UP2[0]:
        return 1 + 8 * seg(t, *UP1)
    if t < DOWN[0]:
        return 9 + 10 * seg(t, *UP2)
    if t < HOME[0]:
        return DMAX + (BEST - DMAX) * smooth(seg(t, *DOWN))
    return BEST + (DMAX - BEST) * smooth(seg(t, *HOME))


def curve_at(d):
    """The curve for degree d: whole degrees are the fits; between them, a quick blend."""
    lo = int(math.floor(d))
    hi = min(lo + 1, DMAX)
    f = smooth(min(1.0, (d - lo) / 0.45))
    return CURVES[lo] * (1 - f) + CURVES[hi] * f, (lo if f < 0.5 else hi)


def qa_frames():
    fr = []
    for k in range(9):
        a, b = SEG[k], SEG[k + 1]
        fr += [int((a + 0.3 * (b - a)) * FPS), int((a + 0.85 * (b - a)) * FPS)]
    return fr[:17] + [N_FRAMES - 1]


class Scene(gds.Scene):
    def __init__(self):
        self.facts = dict(train=[round(v, 6) for v in TRAIN], test=[round(v, 4) for v in TEST], best_degree=BEST, big_test=R["big_test"], big_train=R["big_train"],
                          ridge_lambda=R["ridge_lambda"], ridge_test=R["ridge_test"], ridge_train=R["ridge_train"], noise_floor=R["truth_test"], segments=[round(v, 2) for v in SEG])
        self.cues = self._cues()
        self._make_bg()

    def _cues(self):
        hook = [("It passes through every point.", "black", 50, INK, 300, 1), ("That's the problem.", "dbold", 42, VERM, 390, 0.95)]
        S_ = SEG
        return [
            (0.0, S_[1] - 0.4, hook, 0, 8, "none"),
            (S_[1], S_[2] - 0.3, [("20 noisy points.", "black", 50, INK, 300, 1), ("Fit a polynomial. Start with a line.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[2], S_[3] - 0.3, [("Raise the degree.", "black", 50, INK, 300, 1), ("The error on the 20 points keeps falling.", "dbold", 38, CYAN, 390, 0.95)], 10, 8, "pop"),
            (S_[3], S_[4] - 0.3, [(f"Degree {DMAX}: through all 20.", "black", 50, INK, 300, 1), ("Error on them: 0.", "dbold", 40, CYAN, 390, 0.95)], 10, 8, "pop"),
            (S_[4], S_[5] - 0.3, [("Now 5,000 new points.", "black", 50, INK, 300, 1), (f"Error: {err_txt(TEST[DMAX])}.", "dbold", 42, VERM, 390, 0.95)], 10, 8, "pop"),
            (S_[5], S_[6] - 0.3, [("Too simple. Too flexible.", "black", 50, INK, 300, 1), (f"Best on new points: degrees {BAND[0]}–{BAND[1]}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[6], S_[7] - 0.3, [("Fix 1: more data.", "black", 50, MAGENTA, 300, 1), (f"200 points, degree {DMAX}: {err_txt(R['big_test'])}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[7], S_[8] - 0.3, [("Fix 2: a penalty on big coefficients.", "black", 44, GREEN, 300, 1), (f"20 points, degree {DMAX}: {err_txt(R['ridge_test'])}.", "dbold", 40, INK2, 390, 0.95)], 10, 8, "pop"),
            (S_[8], T_END - 0.3, [("Zero error on your data", "black", 50, INK, 300, 1), ("isn't the goal. New data is.", "dbold", 42, INK2, 390, 0.95)], 10, 8, "pop"),
            (T_END - 0.3, T_END + 0.8, hook, 12, 0, "fade"),
        ]

    def _make_bg(self):
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

    # ------------------------------------------------------------ the plot

    def px(self, x):
        x0, _, w, _ = PLOT
        return x0 + (np.asarray(x) - PX[0]) / (PX[1] - PX[0]) * w

    def py(self, y):
        _, y0, _, h = PLOT
        return y0 + h - (np.clip(np.asarray(y), -60, 60) - PY[0]) / (PY[1] - PY[0]) * h

    def draw_plot(self, ctx, t):
        x0, y0, w, h = PLOT
        ctx.set_source_rgba(0.09, 0.11, 0.2, 0.55)
        S3.pill(ctx, x0 - 10, y0 - 10, w + 20, h + 20, 18)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.10)
        ctx.set_line_width(1.0)
        ctx.move_to(x0, self.py(0))
        ctx.line_to(x0 + w, self.py(0))
        ctx.stroke()
        # the new points (a sample of 400 of the 5,000), faint
        home = smooth(seg(t, *HOME))
        tp = ease_out_cubic(seg(t, *TEST_IN)) * (1 - smooth(seg(t, SEG[6] - 0.2, SEG[6] + 0.6)) * 0.7) * (1 - home)
        if tp > 0.01:
            ctx.set_source_rgba(*rgb(INK3), 0.55 * tp)
            for xv, yv in zip(XT, YT):
                ctx.arc(float(self.px(xv)), float(self.py(yv)), 2.6, 0, TAU)
                ctx.fill()
        # the true curve, from the U on
        tr = ease_out_cubic(seg(t, SEG[5] + 5.0, SEG[5] + 6.0)) * (1 - home)
        if tr > 0.01:
            ctx.set_dash([7.0, 6.0])
            ctx.set_source_rgba(*rgb(INK2), 0.6 * tr)
            ctx.set_line_width(2.0)
            ctx.move_to(float(self.px(GRID[0])), float(self.py(TRUTH[0])))
            for xv, yv in zip(GRID[1:], TRUTH[1:]):
                ctx.line_to(float(self.px(xv)), float(self.py(yv)))
            ctx.stroke()
            ctx.set_dash([])
        # the data: 20 points, or 200 for the first fix
        big = smooth(seg(t, *BIG)) * (1 - smooth(seg(t, SEG[7], SEG[7] + 0.8)))
        if big > 0.01:
            ctx.set_source_rgba(*rgb(MAGENTA), 0.9 * big)
            for xv, yv in zip(XB, YB):
                ctx.arc(float(self.px(xv)), float(self.py(yv)), 4.2, 0, TAU)
                ctx.fill()
        for xv, yv in zip(XS, YS):
            ctx.set_source_rgba(0.04, 0.05, 0.1, (1 - big))
            ctx.arc(float(self.px(xv)), float(self.py(yv)), 9.0, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(*rgb(CYAN), 1 - big)
            ctx.arc(float(self.px(xv)), float(self.py(yv)), 7.0, 0, TAU)
            ctx.fill()
        # the fitted curve
        d = deg_at(t)
        cv, d_show = curve_at(d)
        ridge = smooth(seg(t, *RIDGE)) * (1 - smooth(seg(t, SEG[8] + 0.2, SEG[8] + 1.4)))
        if big > 0.01:
            cv = cv * (1 - big) + BIG_CURVE * big
        if ridge > 0.01:
            cv = cv * (1 - ridge) + RIDGE_CURVE * ridge
        col = np.array(rgb(GOLD))
        if big > 0.01:
            col = col * (1 - big) + np.array(rgb(MAGENTA)) * big
        if ridge > 0.01:
            col = col * (1 - ridge) + np.array(rgb(GREEN)) * ridge
        ctx.save()
        ctx.rectangle(x0 - 8, y0 - 8, w + 16, h + 16)
        ctx.clip()
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        for lw, al in ((9.0, 0.18), (3.6, 1.0)):
            ctx.set_source_rgba(*col, al)
            ctx.set_line_width(lw)
            ctx.move_to(float(self.px(GRID[0])), float(self.py(cv[0])))
            for xv, yv in zip(GRID[1:], cv[1:]):
                ctx.line_to(float(self.px(xv)), float(self.py(yv)))
            ctx.stroke()
        ctx.restore()
        return d_show, big, ridge

    # ------------------------------------------------------------ the readouts and the U

    def readouts(self, img, t, d_show, big, ridge):
        surf, ctx = layer()
        home = smooth(seg(t, *HOME))
        new_a = ease_out_cubic(seg(t, SEG[4] + 3.9, SEG[4] + 4.6)) * (1 - home)
        if big > 0.5:
            tr_v, te_v, n_lab = R["big_train"], R["big_test"], "ON ITS 200 POINTS"
        elif ridge > 0.5:
            tr_v, te_v, n_lab = R["ridge_train"], R["ridge_test"], "ON ITS 20 POINTS"
        else:
            tr_v, te_v, n_lab = TRAIN[d_show], TEST[d_show], "ON ITS 20 POINTS"
        d_lab = DMAX if (big > 0.5 or ridge > 0.5) else d_show
        cols = ((100, "DEGREE", f"{d_lab}" + ("  + penalty" if ridge > 0.5 else ""), INK, 1.0),
                (430, "ERROR " + n_lab, err_txt(tr_v), CYAN, 1.0),
                (760, "ON 5,000 NEW POINTS", err_txt(te_v), VERM, new_a))
        for x, lab, val, col, al in cols:
            if al < 0.01:
                continue
            font(ctx, "bold", 16)
            ctx.set_source_rgba(*rgb(INK3), 0.95 * al)
            show_tracked(ctx, lab, x, 1086, 0.12 * 16)
            font(ctx, "dbold", 40 if x > 100 else 44)
            ctx.set_source_rgba(*rgb(col), al)
            ctx.move_to(x, 1140)
            ctx.show_text(val)
        # the U: error against degree, log scale
        cx, cy, cw, ch = CHART
        ch_a = ease_out_cubic(seg(t, SEG[2] + 0.2, SEG[2] + 1.0)) * (1 - home)
        if ch_a > 0.01:
            def gx(dd):
                return cx + dd / DMAX * cw

            def gy(v):
                lv = math.log10(max(v, 10 ** LOG_LO))
                return cy + ch - (min(lv, LOG_HI) - LOG_LO) / (LOG_HI - LOG_LO) * ch
            ctx.set_source_rgba(*rgb(INK3), 0.7 * ch_a)
            ctx.set_line_width(1.0)
            ctx.move_to(cx, cy)
            ctx.line_to(cx, cy + ch)
            ctx.line_to(cx + cw, cy + ch)
            ctx.stroke()
            ctx.set_source_rgba(1, 1, 1, 0.06 * ch_a)
            for e in range(int(LOG_LO), int(LOG_HI) + 1):
                ctx.move_to(cx, gy(10 ** e))
                ctx.line_to(cx + cw, gy(10 ** e))
            ctx.stroke()
            # training error up to the highest degree reached so far
            reached = DMAX if t >= UP2[1] else (int(deg_at(t)) if t >= UP1[0] else 1)
            for vals, col, upto, al in ((TRAIN, CYAN, reached, 1.0),
                                        (TEST, VERM, int(round(DMAX * seg(t, *SWEEP))), ease_out_cubic(seg(t, SWEEP[0] - 0.3, SWEEP[0])))):
                if upto < 0 or al < 0.01:
                    continue
                ctx.set_source_rgba(*rgb(col), 0.95 * ch_a * al)
                ctx.set_line_width(2.6)
                ctx.move_to(gx(0), gy(vals[0]))
                for dd in range(1, upto + 1):
                    ctx.line_to(gx(dd), gy(vals[dd]))
                ctx.stroke()
                for dd in range(0, upto + 1):
                    ctx.arc(gx(dd), gy(vals[dd]), 3.6, 0, TAU)
                    ctx.fill()
            # the current degree
            if t >= SEG[1]:
                ctx.set_source_rgba(1, 1, 1, 0.35 * ch_a)
                ctx.set_line_width(1.5)
                ctx.move_to(gx(d_show), cy)
                ctx.line_to(gx(d_show), cy + ch)
                ctx.stroke()
            # the best degree, and the two fixes at degree 19
            best_a = ease_out_cubic(seg(t, SEG[5] + 5.2, SEG[5] + 6.0)) * ch_a
            if best_a > 0.01:
                ctx.set_source_rgba(1, 1, 1, best_a)
                ctx.set_line_width(2.2)
                S3.pill(ctx, gx(BAND[0]) - 10, gy(TEST[BEST]) - 10, gx(BAND[1]) - gx(BAND[0]) + 20, 20, 10)
                ctx.stroke()
            for v, col, t0 in ((R["big_test"], MAGENTA, BIG[1]), (R["ridge_test"], GREEN, RIDGE[1])):
                al = ease_out_cubic(seg(t, t0 - 0.2, t0 + 0.4)) * ch_a
                if al > 0.01:
                    ctx.set_source_rgba(*rgb(col), al)
                    ctx.arc(gx(DMAX), gy(v), 7.5, 0, TAU)
                    ctx.fill()
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        if ch_a > 0.01:
            for e, lab in ((-2, "0.01"), (0, "1"), (2, "100"), (4, "10^4")):
                y = cy + ch - (e - LOG_LO) / (LOG_HI - LOG_LO) * ch
                paste(img, rf"${lab.replace('10^4', '10^{4}')}$", 18, cx - 10, y + 6, INK3, ch_a, align="r")
            for dd in (0, 5, 10, 15, 19):
                paste(img, rf"${dd}$", 18, cx + dd / DMAX * cw, cy + ch + 24, INK3, ch_a, align="c")
            paste(img, r"$\mathrm{error\ vs\ degree\ (log\ scale)}$", 18, cx + cw, cy - 8, INK3, ch_a, align="r")

    def card(self, img, t, d_show, big, ridge):
        x, y, w, h = CARD
        surf, ctx = layer()
        S3.card_bg(ctx, x, y, w, h, 1.0)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        a = 1.0
        if t < SEG[1] or t >= T_END - 0.3:
            paste(img, rf"$p(x) = c_0 + c_1 x + c_2 x^2 + \cdots + c_{{{DMAX}}}\,x^{{{DMAX}}}$", 34, 540, y + 100, INK, a, align="c")
            paste(img, rf"$\mathrm{{{DMAX + 1}\ coefficients\ for\ 20\ points:\ it\ can\ hit\ every\ one}}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[4]:
            dd = d_show
            terms = {0: r"c_0", 1: r"c_0 + c_1 x", 2: r"c_0 + c_1 x + c_2 x^2"}
            body = terms.get(dd, rf"c_0 + c_1 x + \cdots + c_{{{dd}}}\,x^{{{dd}}}")
            paste(img, rf"$p(x) = {body}$", 34, 540, y + 90, INK, a, align="c")
            paste(img, r"$\mathrm{error} = \sqrt{\mathrm{mean\ of\ the\ squared\ misses}}$", 28, 540, y + 158, CYAN, a, align="c")
            paste(img, r"$\mathrm{least\ squares\ picks\ the\ coefficients\ that\ miss\ the\ 20\ points\ the\ least}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[5]:
            paste(img, rf"$\mathrm{{degree}}\ {DMAX}:\ \ \mathrm{{error\ on\ its\ 20\ points}} = {err_txt(TRAIN[DMAX])}$", 30, 540, y + 92, CYAN, a, align="c")
            paste(img, rf"$\mathrm{{on\ 5{{,}}000\ new\ points}} = {err_txt(TEST[DMAX]).replace(',', '{,}')}$", 30, 540, y + 152, VERM,
                  ease_out_cubic(seg(t, SEG[4] + 3.9, SEG[4] + 4.6)), align="c")
            paste(img, r"$\mathrm{near\ the\ ends\ it\ swings\ far\ outside\ the\ plot}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[6]:
            paste(img, rf"$\mathrm{{degrees}}\ {BAND[0]}\ \mathrm{{to}}\ {BAND[1]}:\ {TEST[BAND[0]:BAND[1] + 1].min():.3f}\ \mathrm{{to}}\ {TEST[BAND[0]:BAND[1] + 1].max():.3f}\ \mathrm{{on\ new\ points}}$", 30, 540, y + 92, INK, a, align="c")
            paste(img, rf"$\mathrm{{the\ true\ curve\ itself\ scores}}\ {err_txt(R['truth_test'])}:\ \mathrm{{the\ noise}}$", 26, 540, y + 156, INK2,
                  ease_out_cubic(seg(t, SEG[5] + 5.0, SEG[5] + 6.0)), align="c")
            paste(img, r"$\mathrm{too\ simple\ misses\ the\ shape;\ too\ flexible\ fits\ the\ noise}$", 21, 540, y + h - 18, INK3, a, align="c")
        elif t < SEG[7]:
            paste(img, rf"$200\ \mathrm{{points,\ degree}}\ {DMAX}:\ \ {err_txt(R['big_test'])}\ \mathrm{{on\ new\ points}}$", 32, 540, y + 100, MAGENTA, a, align="c")
            paste(img, r"$\mathrm{with\ 200\ points\ to\ follow,\ 20\ coefficients\ can't\ chase\ the\ noise}$", 22, 540, y + h - 26, INK2, a, align="c")
        elif t < SEG[8]:
            paste(img, r"$\mathrm{minimise}\ \ \sum_i\,(y_i - p(x_i))^2 + \lambda \sum_k c_k^2$", 32, 540, y + 72, INK, a, align="c")
            paste(img, rf"$\lambda = {R['ridge_lambda']:.2f}:\ \ {err_txt(R['ridge_test'])}\ \mathrm{{on\ new\ points}}$", 30, 540, y + 180, GREEN,
                  ease_out_cubic(seg(t, RIDGE[1] - 0.3, RIDGE[1] + 0.3)), align="c")
            paste(img, r"$\lambda\ \mathrm{chosen\ by\ cross{-}validation\ on\ the\ 20\ points;}\ c_k\mathrm{:\ Legendre\ coefficients}$", 20, 540, y + h - 16, INK3, a, align="c")
        else:
            paste(img, r"$\mathrm{goal:\ a\ small\ error\ on\ data\ it\ has\ never\ seen}$", 32, 540, y + 108, INK, a, align="c")
            paste(img, r"$\mathrm{a\ perfect\ fit\ to\ the\ data\ you\ have\ is\ often\ the\ worst\ sign}$", 22, 540, y + h - 26, INK2, a, align="c")

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        img = self.bg.copy()
        surf, ctx = layer()
        d_show, big, ridge = self.draw_plot(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self.readouts(img, t, d_show, big, ridge)
        self.card(img, t, d_show, big, ridge)
        surf, ctx = layer()
        font(ctx, "bold", 18)
        tr = 0.12 * 18
        lab = "20 POINTS   ·   5,000 NEW ONES TO TEST ON   ·   ALL COMPUTED"
        w_ = tracked_width(ctx, lab, tr)
        ctx.set_source_rgba(*rgb(INK3), 0.9)
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
    cv2.imwrite(f"{out}/sheet.png", np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]))
    for fr in frames[1::3]:
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a_, b_, c_ = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b_ - a_).mean(), "step", np.abs(c_ - b_).mean())
