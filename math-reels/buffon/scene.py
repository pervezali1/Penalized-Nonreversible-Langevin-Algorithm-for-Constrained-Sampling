"""Buffon's needle. Needles as long as the gap rain onto floorboards; pi ~ 2 x needles / crossings,
correct digits in green. Then why: one needle rotates while a graph fills with its crossing chance
|sin theta|, whose average over all angles is 2/pi. Then 1901: Lazzarini's 3,408 needles and six
digits of pi, one crossing more or fewer and he'd have had two; 1,000 simulated repeats of his
experiment and how many hit his number. frame(f) -> uint8 1080x1920 RGB."""

import importlib.util
import math
import os
import sys

import cv2
import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_cubic, font, layer, linear_to_srgb, over, rgb,  # noqa: E402
                     show_cells, show_tracked, smooth, srgb_to_linear, surf_rgba, tracked_width)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ruin = _load("ruin_scene", ("ruin", "scene.py"))
_mc = _load("montecarlo_scene", ("montecarlo", "scene.py"))
digits_correct = _mc.digits_correct

FPS = 60
N_FRAMES = 1500                     # 25 s
SEED = 3
N_MAX = 100_000
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#49E68A"
CX = 470
BOARD = (60, 660, 1020, 1560)       # the floor: horizontal cracks every GAP px
GAP = 100.0                         # crack spacing, and the needle length (l = d)
RAIN_T0, RAIN_T1 = 2.6, 10.6
WHY_T0, WHY_T1 = 12.8, 17.4         # the rotating needle and the |sin| graph
LAZ_T0 = 17.6                       # Lazzarini
LAZ_N, LAZ_C, LAZ_RATIO = 3408, 1808, 5 / 6
LAZ_REPEATS = 1000
LOOP_FADE = (24.3, 24.85)
MAX_TEXT_W = 820


@njit(cache=True)
def splat_segments(acc, xa, ya, xb, yb, cross, n, col_cross, col_miss, gain):
    for i in range(n):
        dx = xb[i] - xa[i]
        dy = yb[i] - ya[i]
        length = math.sqrt(dx * dx + dy * dy)
        m = int(length / 0.9) + 1
        col = col_cross if cross[i] else col_miss
        for k in range(m):
            u = (k + 0.5) / m
            x = xa[i] + dx * u
            y = ya[i] + dy * u
            xi = int(x)
            yi = int(y)
            if xi < 0 or yi < 0 or xi >= acc.shape[1] - 1 or yi >= acc.shape[0] - 1:
                continue
            fx = x - xi
            fy = y - yi
            for c in range(3):
                v = col[c] * gain
                acc[yi, xi, c] += v * (1 - fx) * (1 - fy)
                acc[yi, xi + 1, c] += v * fx * (1 - fy)
                acc[yi + 1, xi, c] += v * (1 - fx) * fy
                acc[yi + 1, xi + 1, c] += v * fx * fy


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        x0, y0, x1, y1 = BOARD
        # needles: centre uniform on the board, angle uniform in [0, pi)
        self.cx = rng.random(N_MAX) * (x1 - x0 - 2 * GAP) + x0 + GAP
        self.cy = rng.random(N_MAX) * (y1 - y0) + y0
        self.th = rng.random(N_MAX) * math.pi
        hx, hy = 0.5 * GAP * np.cos(self.th), 0.5 * GAP * np.sin(self.th)
        self.xa, self.ya, self.xb, self.yb = self.cx - hx, self.cy - hy, self.cx + hx, self.cy + hy
        rel = (self.cy - y0) % GAP
        dist = np.minimum(rel, GAP - rel)                     # distance from centre to nearest crack
        self.cross = dist <= np.abs(hy)
        self.cum = np.cumsum(self.cross).astype(np.int64)
        self.est_final = 2 * N_MAX / self.cum[-1]
        self.digits_final = digits_correct(self.est_final)
        # Lazzarini's experiment, repeated
        r2 = np.random.default_rng(7)
        self.laz_counts = np.zeros(LAZ_REPEATS, np.int64)
        for r in range(LAZ_REPEATS):
            t = r2.random(LAZ_N) * math.pi
            y = r2.random(LAZ_N)
            self.laz_counts[r] = int((np.minimum(y, 1 - y) <= LAZ_RATIO / 2 * np.sin(t)).sum())
        self.laz_hits = int((self.laz_counts == LAZ_C).sum())
        self.laz_est = 5 * LAZ_N / (3 * LAZ_C)
        self.laz_lo, self.laz_hi = 5 * LAZ_N / (3 * (LAZ_C + 1)), 5 * LAZ_N / (3 * (LAZ_C - 1))
        self.col_cross = np.array(srgb_to_linear(rgb(AMBER)), np.float32)
        self.col_miss = np.array(srgb_to_linear(rgb(CYAN)), np.float32) * 0.5
        self.bg = _ruin.Scene._background(self)
        self.cues = self._cues()

    def _cues(self):
        hook = [("Drop a needle on floorboards.", "black", 56, WHITE, 300, 1), ("Count how often it crosses a crack. That's π.", "dbold", 40, WHITE, 392, 0.9)]
        return [
            (0.0, 2.4, hook, 0, 6, "none"),
            (2.6, 5.2, [("Needle as long as the gap.", "black", 54, WHITE, 312, 1), ("It crosses a crack 2/π of the time. Why?", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (6.0, 8.4, [("Watch the digits.", "black", 60, AMBER, 350, 1)], 8, 6, "pop"),
            (10.8, 12.6, [(f"100,000 needles: π ≈ {self.est_final:.4f}", "black", 56, WHITE, 312, 1), (f"Correct digits: {self.digits_final}.", "black", 60, GREEN, 404, 1)], 6, 6, "chaos"),
            (WHY_T0 + 0.2, 15.4, [("Why π? The needle's angle.", "black", 54, WHITE, 312, 1), ("Crossing chance = |sin θ|.", "dbold", 46, AMBER, 396, 0.9)], 8, 6, "pop"),
            (15.6, WHY_T1, [("Average of |sin θ| over all angles: 2/π.", "black", 46, GREEN, 350, 1)], 8, 6, "pop"),
            (LAZ_T0 + 0.2, 20.0, [("1901. Lazzarini: 3,408 needles.", "black", 52, WHITE, 312, 1), (f"π ≈ {self.laz_est:.7f}. Six digits.", "dbold", 46, AMBER, 396, 0.9)], 8, 6, "pop"),
            (20.3, 22.4, [(f"One crossing more or fewer: {self.laz_lo:.4f} or {self.laz_hi:.4f}.", "black", 42, MAGENTA, 312, 1), ("Only exactly 1,808 gives six digits.", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (22.7, 24.5, [(f"1,000 repeats: {self.laz_hits} hit 1,808.", "black", 56, WHITE, 312, 1), ("He stopped counting at the right moment.", "dbold", 42, MAGENTA, 396, 0.9)], 8, 6, "pop"),
            (24.6, 25.2, hook, 10, 0, "fade"),
        ]

    # ------------------------------------------------------------ schedules

    def needles_shown(self, t):
        if t < RAIN_T0:
            return 0
        if t >= RAIN_T1:
            return N_MAX
        u = (t - RAIN_T0) / (RAIN_T1 - RAIN_T0)
        return max(1, min(N_MAX, int(round(10 ** (5 * u)))))

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        if fade < 1.0 and t > LOOP_FADE[0]:
            a = self._section_frame(t)
            b = self._rain_frame(0.0)
            out = a * fade + b * (1 - fade)
        else:
            out = self._section_frame(t)
        _ruin.Scene._headlines(self, out, f, t)
        return dither(out, f)

    def _section_frame(self, t):
        # crossfades of 0.4 s between the three sections
        if t < WHY_T0 - 0.4:
            return self._rain_frame(t)
        if t < WHY_T0:
            u = smooth((t - (WHY_T0 - 0.4)) / 0.4)
            return self._rain_frame(t) * (1 - u) + self._why_frame(t) * u
        if t < LAZ_T0 - 0.4:
            return self._why_frame(t)
        if t < LAZ_T0:
            u = smooth((t - (LAZ_T0 - 0.4)) / 0.4)
            return self._why_frame(t) * (1 - u) + self._laz_frame(t) * u
        return self._laz_frame(t)

    def _compose(self, acc, k=1.5):
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-k * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        return linear_to_srgb(lin).astype(np.float32)

    # ------------------------------------------------------------ the rain

    def _rain_frame(self, t):
        n = self.needles_shown(t)
        acc = np.zeros((H, W, 3), np.float32)
        if n > 0:
            gain = float(min(1.2, 1.2 * (300.0 / n) ** 0.88)) if n > 300 else 1.2
            gain = max(gain, 0.005)
            splat_segments(acc, self.xa, self.ya, self.xb, self.yb, self.cross, n, self.col_cross, self.col_miss, gain)
        img = self._compose(acc)
        self._rain_hud(img, t, n)
        return img

    def _floor(self, ctx, x0, y0, x1, y1, alpha=0.35):
        ctx.set_line_width(2)
        ctx.set_source_rgba(1, 1, 1, alpha)
        y = y0
        while y <= y1 + 0.5:
            ctx.move_to(x0, y + 0.5)
            ctx.line_to(x1, y + 0.5)
            ctx.stroke()
            y += GAP

    def _rain_hud(self, img, t, n):
        surf, ctx = layer()
        x0, y0, x1, y1 = BOARD
        self._floor(ctx, x0, y0, x1, y1)
        crossing = int(self.cum[n - 1]) if n else 0
        est = 2 * n / crossing if crossing else 0.0
        k = digits_correct(est) if crossing else 0
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "NEEDLES", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "CROSSING", 0.14 * 24)
        show_tracked(ctx, "CROSSING", 910 - w, 470, 0.14 * 24)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22})
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{n:,}", 70, 545, big)
        s = f"{crossing:,}"
        ctx.set_source_rgb(*rgb(AMBER))
        show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)
        font(ctx, "bold", 26)
        lab = "π ≈ 2 × NEEDLES ÷ CROSSING  ="
        ctx.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(ctx, lab, 70, 626, 0.06 * 26)
        lw = tracked_width(ctx, lab, 0.06 * 26)
        font(ctx, "dbold", 44)
        cw = 30
        sx = 70 + lw + 26
        if crossing:
            s = f"{est:.6f}" if est < 10 else f"{est:.4f}"
            di = 0
            for ch in s:
                if ch == ".":
                    ctx.set_source_rgba(1, 1, 1, 0.9)
                    ctx.move_to(sx, 632)
                    ctx.show_text(ch)
                    sx += 14
                    continue
                ok = di < k
                ctx.set_source_rgba(*(rgb(GREEN) if ok else (1, 1, 1)), 0.95 if ok else 0.45)
                ctx.move_to(sx + cw / 2 - ctx.text_extents(ch).x_advance / 2, 632)
                ctx.show_text(ch)
                sx += cw
                di += 1
            font(ctx, "bold", 22)
            ctx.set_source_rgba(*rgb(GREEN), 0.9)
            lab2 = f"{k} DIGIT{'S' if k != 1 else ''}"
            show_tracked(ctx, lab2, 910 - tracked_width(ctx, lab2, 0.08 * 22), 632, 0.08 * 22)
        font(ctx, "bold", 20)
        ctx.set_source_rgba(1, 1, 1, 0.45)
        show_tracked(ctx, "NEEDLE = GAP      π = 3.14159265…", x0 + 10, y1 + 44, 0.08 * 20)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)

    # ------------------------------------------------------------ why: the angle

    def _why_frame(self, t):
        u = min(1.0, max(0.0, (t - WHY_T0 - 0.2) / (WHY_T1 - WHY_T0 - 0.8)))
        theta = math.pi * u
        img = linear_to_srgb(self.bg).astype(np.float32)
        # left: a small floor (cracks every 200 px, twice life-size) with one needle rotating at its centre
        fx0, fy0, fx1, fy1 = 70, 760, 470, 1460
        L = 2.0 * GAP
        surf2, ctx2 = layer()
        ctx2.set_line_width(2)
        ctx2.set_source_rgba(1, 1, 1, 0.3)
        for yy in (fy0 + 50, fy0 + 250, fy0 + 450, fy0 + 650):
            ctx2.move_to(fx0, yy + 0.5)
            ctx2.line_to(fx1, yy + 0.5)
            ctx2.stroke()
        cxn, cyn = (fx0 + fx1) / 2, fy0 + 350 - 60     # centre 60 px above a crack-midline: 40 px from the crack at fy0+250
        # the projection across the cracks: L|sin theta|
        proj = L * abs(math.sin(theta))
        crosses = proj / 2 >= abs(cyn - (fy0 + 250))
        ctx2.set_line_width(10)
        ctx2.set_source_rgba(*rgb(AMBER if crosses else CYAN), 0.95)
        ctx2.move_to(cxn - L / 2 * math.cos(theta), cyn - L / 2 * math.sin(theta))
        ctx2.line_to(cxn + L / 2 * math.cos(theta), cyn + L / 2 * math.sin(theta))
        ctx2.stroke()
        # vertical extent bar
        ctx2.set_line_width(4)
        ctx2.set_dash([6, 6])
        ctx2.set_source_rgba(*rgb(GREEN), 0.9)
        ctx2.move_to(fx1 - 30, cyn - proj / 2)
        ctx2.line_to(fx1 - 30, cyn + proj / 2)
        ctx2.stroke()
        ctx2.set_dash([])
        font(ctx2, "bold", 22)
        ctx2.set_source_rgba(*rgb(GREEN), 0.95)
        s = f"|sin θ| = {abs(math.sin(theta)):.2f}"
        ctx2.move_to(fx1 - 30 - ctx2.text_extents(s).x_advance - 12, cyn + 8)
        ctx2.show_text(s)
        font(ctx2, "bold", 22)
        ctx2.set_source_rgba(1, 1, 1, 0.7)
        s = f"θ = {math.degrees(theta):.0f}°"
        ctx2.move_to(fx0, fy0 + 20)
        ctx2.show_text(s)
        # right: the graph of |sin theta| filling in
        gx0, gy0, gx1, gy1 = 540, 820, 900, 1400

        def gpx(th):
            return gx0 + (gx1 - gx0) * th / math.pi

        def gpy(v):
            return gy1 - (gy1 - gy0) * v

        ctx2.set_line_width(1)
        ctx2.set_source_rgba(1, 1, 1, 0.25)
        ctx2.move_to(gx0, gy1 + 0.5)
        ctx2.line_to(gx1, gy1 + 0.5)
        ctx2.stroke()
        ctx2.move_to(gx0 + 0.5, gy0)
        ctx2.line_to(gx0 + 0.5, gy1)
        ctx2.stroke()
        # average line 2/pi
        avg = 2 / math.pi
        ctx2.set_dash([5, 7])
        ctx2.set_source_rgba(*rgb(GREEN), 0.8 if u >= 1 else 0.3)
        ctx2.move_to(gx0, gpy(avg) + 0.5)
        ctx2.line_to(gx1, gpy(avg) + 0.5)
        ctx2.stroke()
        ctx2.set_dash([])
        # filled area so far
        if theta > 0:
            ths = np.linspace(0, theta, 120)
            ctx2.move_to(gpx(0), gpy(0))
            for th in ths:
                ctx2.line_to(gpx(th), gpy(abs(math.sin(th))))
            ctx2.line_to(gpx(theta), gpy(0))
            ctx2.close_path()
            ctx2.set_source_rgba(*rgb(AMBER), 0.28)
            ctx2.fill()
            ctx2.set_line_width(4)
            ctx2.set_source_rgba(*rgb(AMBER), 0.95)
            ctx2.move_to(gpx(0), gpy(0))
            for th in ths:
                ctx2.line_to(gpx(th), gpy(abs(math.sin(th))))
            ctx2.stroke()
            ctx2.set_source_rgba(*rgb(GREEN), 0.95)
            ctx2.arc(gpx(theta), gpy(abs(math.sin(theta))), 7, 0, 2 * math.pi)
            ctx2.fill()
        font(ctx2, "bold", 20)
        ctx2.set_source_rgba(1, 1, 1, 0.5)
        for th, lab in ((0, "0°"), (math.pi / 2, "90°"), (math.pi, "180°")):
            ctx2.move_to(gpx(th) - ctx2.text_extents(lab).x_advance / 2, gy1 + 30)
            ctx2.show_text(lab)
        for v, lab in ((1.0, "1"), (avg, "2/π = 0.637")):
            ctx2.set_source_rgba(*(rgb(GREEN) if v < 1 else (1, 1, 1)), 0.8)
            ctx2.move_to(gx1 + 10, gpy(v) + 7)
            ctx2.show_text(lab)
        font(ctx2, "bold", 22)
        ctx2.set_source_rgba(1, 1, 1, 0.65)
        show_tracked(ctx2, "CROSSING CHANCE  vs  ANGLE", gx0, gy0 - 24, 0.08 * 22)
        # running average readout
        if theta > 0:
            run_avg = (1 - math.cos(theta)) / theta
            font(ctx2, "bold", 24)
            ctx2.set_source_rgba(1, 1, 1, 0.55)
            show_tracked(ctx2, "AVERAGE SO FAR", 70, 470, 0.14 * 24)
            big = {str(c): 50 for c in range(10)}
            big.update({".": 22})
            font(ctx2, "dbold", 76)
            ctx2.set_source_rgb(*(rgb(GREEN) if u >= 1 else rgb(WHITE)))
            show_cells(ctx2, f"{run_avg:.4f}", 70, 545, big)
            font(ctx2, "bold", 24)
            ctx2.set_source_rgba(1, 1, 1, 0.55)
            w = tracked_width(ctx2, "2/π", 0.14 * 24)
            show_tracked(ctx2, "2/π", 910 - w, 470, 0.14 * 24)
            font(ctx2, "dbold", 76)
            ctx2.set_source_rgb(*rgb(GREEN))
            s = f"{avg:.4f}"
            show_cells(ctx2, s, 910 - cells_width(ctx2, s, big), 545, big)
        trgb, ta = surf_rgba(surf2)
        over(img, trgb, ta)
        return img

    # ------------------------------------------------------------ Lazzarini

    def _laz_frame(self, t):
        u = min(1.0, max(0.0, (t - LAZ_T0 - 0.3) / 2.0))
        n_show = int(LAZ_REPEATS * ease_out_cubic(u))
        img = linear_to_srgb(self.bg).astype(np.float32)
        surf, ctx = layer()
        x0, y0, x1, y1 = 90, 780, 880, 1420
        lo, hi = 1700, 1920
        counts = np.bincount(self.laz_counts[:n_show], minlength=hi + 1)[lo:hi + 1] if n_show else np.zeros(hi - lo + 1, int)
        peak = max(1, np.bincount(self.laz_counts, minlength=hi + 1)[lo:hi + 1].max())
        bw = (x1 - x0) / (hi - lo + 1)

        def bx(c):
            return x0 + (c - lo) * bw

        for c in range(lo, hi + 1):
            h = counts[c - lo]
            if h == 0:
                continue
            hh = (y1 - y0 - 40) * h / peak
            col = AMBER if c == LAZ_C else WHITE
            ctx.set_source_rgba(*rgb(col), 0.95 if c == LAZ_C else 0.45)
            ctx.rectangle(bx(c) + 0.5, y1 - hh, max(1.0, bw - 1), hh)
            ctx.fill()
        # axis and labels
        ctx.set_line_width(1)
        ctx.set_source_rgba(1, 1, 1, 0.3)
        ctx.move_to(x0, y1 + 0.5)
        ctx.line_to(x1, y1 + 0.5)
        ctx.stroke()
        font(ctx, "bold", 20)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        for c in (1700, 1750, 1800, 1850, 1900):
            s = f"{c:,}"
            ctx.move_to(bx(c) + bw / 2 - ctx.text_extents(s).x_advance / 2, y1 + 30)
            ctx.show_text(s)
        s = "CROSSINGS IN 3,408 NEEDLES"
        ctx.move_to(x1 - ctx.text_extents(s).x_advance, y1 + 56)
        ctx.show_text(s)
        # the 1,808 marker and its neighbours
        if t >= LAZ_T0 + 0.3:
            ctx.set_line_width(2)
            ctx.set_source_rgba(*rgb(AMBER), 0.9)
            ctx.move_to(bx(LAZ_C) + bw / 2, y0 - 10)
            ctx.line_to(bx(LAZ_C) + bw / 2, y1)
            ctx.stroke()
            font(ctx, "bold", 24)
            ctx.set_source_rgba(*rgb(AMBER), 0.95)
            s = f"1,808  →  {self.laz_est:.7f}"
            ctx.move_to(bx(LAZ_C) + bw / 2 + 14, y0 + 14)
            ctx.show_text(s)
            font(ctx, "bold", 20)
            ctx.set_source_rgba(*rgb(AMBER), 0.8)
            s = "= 355/113"
            ctx.move_to(bx(LAZ_C) + bw / 2 + 14, y0 + 42)
            ctx.show_text(s)
        if t >= 20.3:
            font(ctx, "bold", 22)
            ctx.set_source_rgba(*rgb(MAGENTA), 0.95)
            for c, est, dy in ((LAZ_C - 1, self.laz_hi, 90), (LAZ_C + 1, self.laz_lo, 130)):
                s = f"{c:,} → {est:.4f}"
                ctx.move_to(bx(LAZ_C) + bw / 2 - ctx.text_extents(s).x_advance - 14 if c < LAZ_C else bx(LAZ_C) + bw / 2 + 14, y0 + dy)
                ctx.show_text(s)
        # HUD
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "REPEATS", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "HIT 1,808 EXACTLY", 0.14 * 24)
        show_tracked(ctx, "HIT 1,808 EXACTLY", 910 - w, 470, 0.14 * 24)
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22})
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{n_show:,}", 70, 545, big)
        hits = int((self.laz_counts[:n_show] == LAZ_C).sum())
        s = f"{hits}"
        ctx.set_source_rgb(*rgb(AMBER))
        show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.5)
        show_tracked(ctx, "NEEDLE = 5/6 OF THE GAP, AS HE USED      π ≈ 5 × 3,408 ÷ (3 × CROSSINGS)", 70, 626, 0.06 * 24)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        return img


if __name__ == "__main__":
    import time

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sc = Scene()
    print(f"init {time.time() - t0:.1f}s est {sc.est_final:.6f} digits {sc.digits_final} laz hits {sc.laz_hits} "
          f"laz est {sc.laz_est:.7f} lo {sc.laz_lo:.4f} hi {sc.laz_hi:.4f}")
    frames = [0, 170, 220, 280, 340, 420, 500, 580, 660, 760, 800, 880, 960, 1010, 1080, 1150, 1260, 1380]
    ims = []
    for fr in frames:
        t0 = time.time()
        im = sc.frame(fr)
        print(f"  frame {fr} {time.time() - t0:.2f}s")
        ims.append(cv2.resize(im[..., ::-1], (360, 640)))
    rows = [np.hstack(ims[i:i + 6]) for i in range(0, 18, 6)]
    cv2.imwrite(f"{out}/sheet.png", np.vstack(rows))
    for fr in (500, 880, 1380):
        cv2.imwrite(f"{out}/f{fr}.png", sc.frame(fr)[..., ::-1])
    a, b, c = (sc.frame(i).astype(np.float32) for i in (N_FRAMES - 1, 0, 1))
    print("seam", np.abs(b - a).mean(), "step", np.abs(c - b).mean())
