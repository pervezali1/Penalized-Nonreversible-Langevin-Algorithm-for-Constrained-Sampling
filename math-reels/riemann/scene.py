"""The Riemann hypothesis in one picture. Top: how many primes there are up to x (the amber staircase), and Riemann's
explicit formula built from the zeta zeros found so far (cyan). Bottom: a point climbs the critical line (left) while
the zeta function's output at that point is traced on the right; every time the output passes through 0, a zero is
marked on the line and its wave is added to the cyan curve, which turns into the staircase of the primes.
The first 2.4 s continue the end of the loop (hundreds of zeros in), then the picture clears and the story starts.
    frame(f) -> uint8 1080x1920 RGB"""

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
from rsim import DUR, FPS, HIT_T, N_FRAMES, STEP, SWEEP, T_WRAP, T_ZETA, WIPE, X_MAX, tau  # noqa: E402,F401

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
SLATE, DIM = "#7C88B5", "#C9CEE6"
CX = 470
MAX_TEXT_W = 820
# chart A: the prime count
AX0, AX1, AY0, AY1, C_TOP = 96, 866, 1010, 640, 16.0
# panel B: the line (input) and the zeta plane (output)
GX, GY0, GY1, GS, G_BELOW = 150, 1558, 1150, 10.5, 26.0
OX, OY = 430, 1352
CLIP_B = (236, 1136, 668, 436)
T_CURVE = 9.0                                    # the formula's curve appears (no zeros yet)
T_HYP = (17.2, 22.7)                             # the line is highlighted
NEWS = "Nov 20, 2026: a proof was announced."


def ax(x):
    return AX0 + (AX1 - AX0) * x / X_MAX


def ay(c):
    return AY0 - (AY0 - AY1) * c / C_TOP


def zoom(h):
    """Pixels per unit in the zeta plane: 110 while the first loops are drawn, then zooming out to 56 (the rare
    very large loops run off the panel)."""
    if h <= 27:
        return 110.0
    return max(56.0, 110.0 * (h / 27.0) ** -0.458)


class Scene:
    def __init__(self, data_path):
        d = np.load(data_path)
        with open(data_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.zeta, self.gamma, self.hit = d["zeta"], d["gamma"], d["hit"]
        self.x, self.base, self.terms = d["x"], d["base"], d["terms"].astype(np.float64)
        self.primes = d["primes"]
        self.px = ax(self.x)
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        hook = [("The $1,000,000 math problem.", "black", 70, WHITE, 300, 1), (NEWS, "dbold", 46, AMBER, 392, 1)]
        return [
            (0.0, 2.35, hook, 0, 6, "none"),
            (2.6, 4.8, [("The Riemann Hypothesis.", "black", 80, WHITE, 312, 1),
                        ("What is it? Start with the primes.", "dbold", 48, WHITE, 400, 0.9)], 8, 6, "pop"),
            (5.0, 6.9, [("2, 3, 5, 7, 11, 13…", "black", 80, AMBER, 312, 1),
                        ("They look random. No pattern.", "dbold", 50, WHITE, 400, 0.9)], 8, 6, "pop"),
            (7.1, 9.4, [("1859: Riemann builds a function.", "black", 58, WHITE, 312, 1),
                        ("Feed it one line of numbers. Watch.", "dbold", 46, WHITE, 396, 0.9)], 8, 6, "pop"),
            (HIT_T[0], 11.6, [("Sometimes it hits zero.", "black", 72, CYAN, 312, 1),
                              ("Each zero is a wave.", "dbold", 54, WHITE, 400, 0.95)], 6, 6, "pop"),
            (11.8, 14.0, [("Add the waves", "black", 76, WHITE, 312, 1), ("to the prime count…", "black", 62, WHITE, 396, 1)], 8, 6, "pop"),
            (14.2, 17.0, [("…and the primes appear.", "black", 76, CYAN, 312, 1),
                          ("The zeros know where every prime is.", "dbold", 46, WHITE, 400, 0.95)], 6, 6, "chaos"),
            (T_HYP[0], 19.8, [("Riemann's guess:", "black", 80, AMBER, 312, 1),
                              ("every zero sits on this one line.", "dbold", 52, WHITE, 400, 1)], 8, 6, "pop"),
            (20.0, 22.6, [("12 trillion zeros checked.", "black", 70, WHITE, 312, 1),
                          ("All on the line. A proof for all: $1,000,000.", "dbold", 42, AMBER, 400, 1)], 8, 6, "pop"),
            (22.8, 25.6, [("Why care?", "black", 88, AMBER, 312, 1),
                          ("Primes lock your cards and messages.", "dbold", 48, WHITE, 400, 0.95)], 8, 6, "pop"),
            (25.8, 29.3, [("This is the law of the primes.", "black", 62, WHITE, 312, 1),
                          ("Whole areas of maths already assume it.", "dbold", 44, CYAN, 396, 1)], 8, 6, "pop"),
            (29.45, DUR + 0.2, hook, 12, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1050) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ state

    def state(self, t):
        """(te, alpha of the running picture, swept x) at video time t."""
        if t < WIPE[1]:                                   # the end of the loop, continuing, then fading out
            te = min(t + DUR, T_WRAP)
            a = 1.0 if t < WIPE[0] else 1 - smooth((t - WIPE[0]) / (WIPE[1] - WIPE[0] - 0.05))
            return te, a, X_MAX
        xs = 0.0 if t < SWEEP[0] else X_MAX * min(1.0, (t - SWEEP[0]) / (SWEEP[1] - SWEEP[0]))
        return t, 1.0, xs

    def weights(self, te):
        """How much of each zero's wave is in: the first three ease in slowly so the ripple can be seen."""
        w = np.zeros(len(self.hit))
        age = te - self.hit
        w[:3] = np.clip(age[:3] / 0.55, 0, 1)
        w[3:] = np.clip(age[3:] / 0.12, 0, 1)
        return w * w * (3 - 2 * w)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        te, a, xs = self.state(t)
        g_surf, g = layer()
        h_surf, hud = layer()
        n_found = int(np.searchsorted(self.hit, te, side="right")) if te >= T_ZETA else 0
        self._chart(g, hud, t, te, a, xs, n_found)
        self._panel(g, hud, t, te, a, n_found)
        self._hud(hud, t, a, xs, n_found)
        g_rgb, g_a = surf_rgba(g_surf)
        band = slice(590, 1600)
        emis = np.zeros((H, W, 3), np.float32)
        sub = g_rgb[band] / np.maximum(g_a[band], 1e-4)[..., None]
        emis[band] = srgb_to_linear(sub).astype(np.float32) * g_a[band][..., None] * 1.15
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.4 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.3 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.4 * (emis + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        trgb, ta = surf_rgba(h_surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)
        return dither(img, f)

    # ------------------------------------------------------------ chart A: the primes

    def _chart(self, g, hud, t, te, a, xs, n_found):
        # axes
        hud.set_source_rgba(1, 1, 1, 0.35)
        hud.set_line_width(2)
        hud.move_to(AX0, AY1 - 6)
        hud.line_to(AX0, AY0)
        hud.line_to(AX1 + 8, AY0)
        hud.stroke()
        font(hud, "bold", 20)
        for c in (5, 10, 15):
            hud.set_source_rgba(1, 1, 1, 0.07)
            hud.move_to(AX0, ay(c))
            hud.line_to(AX1, ay(c))
            hud.stroke()
            s = str(c)
            hud.set_source_rgba(1, 1, 1, 0.5)
            hud.move_to(AX0 - 10 - hud.text_extents(s).x_advance, ay(c) + 7)
            hud.show_text(s)
        hud.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(hud, "HOW MANY PRIMES SO FAR", AX0, AY1 - 28, 0.1 * 20)
        # the number line: every whole number a dot, the primes lit and named
        for m in range(1, int(X_MAX) + 1):
            hud.set_source_rgba(1, 1, 1, 0.22)
            hud.arc(ax(m), AY0 + 26, 2.2, 0, 2 * math.pi)
            hud.fill()
        font(hud, "bold", 19)
        for p in self.primes:
            if p > xs:
                break
            k = min(1.0, (xs - p) / 0.8) if xs < X_MAX else 1.0
            g.set_source_rgba(*rgb(AMBER), a)
            g.arc(ax(p), AY0 + 26, 3.2 + 3.0 * (1 - k), 0, 2 * math.pi)
            g.fill()
            s = str(p)
            hud.set_source_rgba(*rgb(AMBER), 0.95 * a)
            hud.move_to(ax(p) - hud.text_extents(s).x_advance / 2, AY0 + 58)
            hud.show_text(s)
        if 0 < xs < X_MAX:                                # the sweep line
            hud.set_source_rgba(1, 1, 1, 0.5)
            hud.set_line_width(2)
            hud.move_to(ax(xs), AY0 + 34)
            hud.line_to(ax(xs), AY1)
            hud.stroke()
        # the true staircase up to xs
        a_c = smooth((te - T_CURVE) / 0.5) if te >= T_CURVE else 0.0
        if xs > 0:
            g.set_source_rgba(*rgb(AMBER), a * (1.0 - 0.5 * a_c))
            g.set_line_width(5)
            g.move_to(ax(0), ay(0))
            c = 0
            for p in self.primes:
                if p > xs:
                    break
                g.line_to(ax(p), ay(c))
                c += 1
                g.line_to(ax(p), ay(c))
            g.line_to(ax(xs), ay(c))
            g.stroke()
        # Riemann's formula with the zeros found so far
        if a_c > 0:
            wts = self.weights(te)
            n_use = int(np.count_nonzero(wts))
            curve = self.base + wts[:n_use] @ self.terms[:n_use]
            py = ay(np.clip(curve, -0.8, C_TOP + 0.3))
            g.set_source_rgba(*rgb(CYAN), a * a_c)
            g.set_line_width(4.5)
            g.move_to(self.px[0], py[0])
            for xx, yy in zip(self.px[1:], py[1:]):
                g.line_to(xx, yy)
            g.stroke()
            # the newest wave on its own, four times its real height, while the zeros still come one by one
            a_w = 1 - smooth((te - 12.6) / 0.6)
            if n_found and a_w > 0:
                k = n_found - 1
                age = te - self.hit[k]
                aw = a * a_w * min(1.0, age / 0.15)
                m = (self.x >= 8) & (self.x <= 38)
                wy = ay(14.4 + 4 * self.terms[k][m])
                g.set_source_rgba(*rgb(WHITE), 0.85 * aw)
                g.set_line_width(2.5)
                g.move_to(self.px[m][0], wy[0])
                for xx, yy in zip(self.px[m][1:], wy[1:]):
                    g.line_to(xx, yy)
                g.stroke()
                font(hud, "bold", 19)
                hud.set_source_rgba(1, 1, 1, 0.8 * aw)
                show_tracked(hud, f"ZERO {k + 1}'S WAVE  (DRAWN 4X TALLER)", ax(8), ay(14.4) + 56, 0.08 * 19)
        # legend
        font(hud, "bold", 20)
        s = f"■ BUILT FROM {n_found:,} ZERO{'S' if n_found != 1 else ''}"
        if a_c > 0:
            hud.set_source_rgba(*rgb(CYAN), 0.95 * a * a_c)
            show_tracked(hud, s, AX1 - tracked_width(hud, s, 0.06 * 20), AY1 - 28, 0.06 * 20)

    # ------------------------------------------------------------ panel B: the line and the zeta plane

    def _panel(self, g, hud, t, te, a, n_found):
        h = tau(te) if te >= T_ZETA else 0.0
        bot = max(0.0, h - G_BELOW)
        a_hyp = smooth((t - T_HYP[0]) / 0.4) * (1 - smooth((t - T_HYP[1]) / 0.5)) if T_HYP[0] <= t < T_HYP[1] + 0.5 else 0.0
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.4 * t)
        # the line
        if a_hyp > 0:                                      # the strip around it, and two places no zero has been found
            hud.set_source_rgba(*rgb(AMBER), 0.07 * a_hyp)
            hud.rectangle(GX - 44, GY1, 88, GY0 - GY1)
            hud.fill()
            hud.set_line_width(2)
            for gx, gy in ((GX - 26, 1236), (GX + 27, 1452)):
                hud.set_source_rgba(1, 1, 1, 0.5 * a_hyp)
                hud.arc(gx, gy, 7, 0, 2 * math.pi)
                hud.stroke()
                hud.set_source_rgba(*rgb(MAGENTA), 0.9 * a_hyp)
                hud.move_to(gx - 9, gy - 9)
                hud.line_to(gx + 9, gy + 9)
                hud.move_to(gx + 9, gy - 9)
                hud.line_to(gx - 9, gy + 9)
                hud.stroke()
        col_line = [(1 - a_hyp) * 0.6 + a_hyp * c for c in rgb(AMBER)] if a_hyp > 0 else (0.6, 0.6, 0.6)
        g.set_source_rgba(*col_line, 0.55 + 0.45 * a_hyp * (0.6 + 0.4 * pulse))
        g.set_line_width(2.5 + 3.5 * a_hyp)
        g.move_to(GX, GY0)
        g.line_to(GX, GY1)
        g.stroke()
        font(hud, "bold", 20)
        hud.set_source_rgba(1, 1, 1, 0.6)
        s = "THE LINE"
        show_tracked(hud, s, GX - tracked_width(hud, s, 0.1 * 20) / 2, GY1 - 16, 0.1 * 20)
        show_tracked(hud, "RIEMANN'S ZETA FUNCTION: ITS OUTPUT", CLIP_B[0] + 24, GY1 - 16, 0.1 * 20)
        # the plane: axes through 0
        sc = zoom(h)
        hud.set_source_rgba(1, 1, 1, 0.16)
        hud.set_line_width(2)
        hud.move_to(CLIP_B[0], OY)
        hud.line_to(CLIP_B[0] + CLIP_B[2], OY)
        hud.move_to(OX, CLIP_B[1])
        hud.line_to(OX, CLIP_B[1] + CLIP_B[3])
        hud.stroke()
        hud.set_dash([5, 7])
        hud.set_source_rgba(1, 1, 1, 0.2)
        hud.arc(OX, OY, sc, 0, 2 * math.pi)                # the unit circle: shows the zoom
        hud.stroke()
        hud.set_dash([])
        font(hud, "black", 26)
        hud.set_source_rgba(1, 1, 1, 0.75)
        hud.move_to(OX - 26, OY + 32)
        hud.show_text("0")
        if te < T_ZETA:                                    # waiting to start: the point at the foot of the line
            g.set_source_rgba(*rgb(WHITE), 0.55 + 0.35 * pulse)
            g.arc(GX, GY0, 6, 0, 2 * math.pi)
            g.fill()
            z0 = self.zeta[0]
            g.arc(OX + sc * z0.real, OY - sc * z0.imag, 6, 0, 2 * math.pi)
            g.fill()
            return
        # zeros found so far, as beads on the line
        lo = np.searchsorted(self.gamma, bot)
        font(hud, "bold", 19)
        for k in range(lo, n_found):
            y = GY0 - (self.gamma[k] - bot) * GS
            age = te - self.hit[k]
            fl = max(0.0, 1 - age / 0.35)
            col = rgb(AMBER) if a_hyp > 0.5 else rgb(CYAN)
            g.set_source_rgba(*[c + (1 - c) * fl for c in col], a)
            g.arc(GX, y, 4.6 + 5 * fl, 0, 2 * math.pi)
            g.fill()
            if k < 3:
                hud.set_source_rgba(*rgb(CYAN), 0.95 * a)
                hud.move_to(GX - 20 - hud.text_extents(f"{self.gamma[k]:.2f}").x_advance, y + 7)
                hud.show_text(f"{self.gamma[k]:.2f}")
        hy = GY0 - (h - bot) * GS
        g.set_source_rgba(*rgb(WHITE), a)
        g.arc(GX, hy, 7, 0, 2 * math.pi)
        g.fill()
        font(hud, "bold", 19)
        hud.set_source_rgba(1, 1, 1, 0.75 * a)
        hud.move_to(GX + 16, hy + 7)
        hud.show_text(f"{h:.0f}" if h >= 100 else f"{h:.1f}")
        # the output: zeta along the line, the last stretch of it
        win = h if h <= 26 else max(9.0, 26 - (h - 26) * 0.5)
        i1 = h / STEP
        i0 = max(0, int((h - win) / STEP))
        idx = np.arange(i0, int(i1) + 1)
        zz = self.zeta[idx]
        fr = i1 - int(i1)
        head = self.zeta[int(i1)] * (1 - fr) + self.zeta[int(i1) + 1] * fr
        pts = np.stack([OX + sc * np.append(zz.real, head.real), OY - sc * np.append(zz.imag, head.imag)], 1)
        g.save()
        g.rectangle(*CLIP_B)
        g.clip()
        n = len(pts)
        K = 16
        for c in range(K):
            j0, j1 = c * (n - 1) // K, (c + 1) * (n - 1) // K + 1
            if j1 - j0 < 2:
                continue
            u = (c + 0.5) / K                              # 0 oldest .. 1 newest
            g.set_source_rgba(*rgb(CYAN), a * (0.08 + 0.92 * u ** 1.6))
            g.set_line_width(2.5 + 2.0 * u)
            g.move_to(*pts[j0])
            for p in pts[j0 + 1:j1]:
                g.line_to(*p)
            g.stroke()
        for k in range(max(0, n_found - 4), n_found):     # a ring at 0 each time the output passes through it
            age = te - self.hit[k]
            if 0 <= age < 0.4:
                u = age / 0.4
                g.set_source_rgba(*rgb(WHITE), a * (1 - u))
                g.set_line_width(4)
                g.arc(OX, OY, 10 + 70 * u, 0, 2 * math.pi)
                g.stroke()
        g.set_source_rgba(*rgb(WHITE), a)
        g.arc(pts[-1][0], pts[-1][1], 8, 0, 2 * math.pi)
        g.fill()
        g.restore()

    # ------------------------------------------------------------ HUD

    def _hud(self, ctx, t, a, xs, n_found):
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "PRIMES UP TO 50", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "ZEROS FOUND", 0.14 * 24)
        show_tracked(ctx, "ZEROS FOUND", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        n_p = int(np.searchsorted(self.primes, xs, side="right"))
        if a < 1:                                         # the old picture's numbers fade out, then zeros fade in
            old, new = max(0.0, 2 * a - 1), max(0.0, 1 - 2 * a)
            s = f"{n_found:,}" if old > 0 else "0"
            ctx.set_source_rgba(*rgb(AMBER), max(old, new))
            show_cells(ctx, str(n_p) if old > 0 else "0", 70, 545, big)
            ctx.set_source_rgba(*rgb(CYAN), max(old, new))
            show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)
            return
        ctx.set_source_rgb(*rgb(AMBER))
        show_cells(ctx, str(n_p), 70, 545, big)
        ctx.set_source_rgb(*rgb(CYAN))
        s = f"{n_found:,}"
        show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)

    def _headlines(self, img, f, t):
        for (t_in, t_out, lines, n_in, n_out, style) in self.cues:
            if not (t_in - 1e-6 <= t < t_out + n_out / FPS):
                continue
            k_in = (t - t_in) * FPS
            k_out = (t - t_out) * FPS
            alpha = 1.0
            if n_in and k_in < n_in:
                alpha = k_in / n_in if style == "fade" else ease_out_cubic(k_in / n_in)
            if k_out >= 0 and n_out:
                alpha = min(alpha, 1 - k_out / n_out)
            if alpha <= 0:
                continue
            sc, dy = 1.0, 0.0
            if style == "pop" and k_in < n_in:
                u = ease_out_cubic(k_in / n_in)
                sc, dy = 0.92 + 0.08 * u, 18 * (1 - u)
            if style == "chaos" and k_in < n_in:
                sc = 1.30 - 0.30 * ease_out_back(k_in / n_in)
            surf, ctx = layer()
            tops, bots = [], []
            for (s, kind, size, colr, base, la) in lines:
                font(ctx, kind, size)
                w = ctx.text_extents(s).x_advance
                fs = min(1.0, MAX_TEXT_W / w)
                cy = base - size * 0.36
                ctx.save()
                ctx.translate(CX, cy + dy)
                ctx.scale(sc * fs, sc * fs)
                ctx.translate(-CX, -cy)
                ctx.set_source_rgba(*rgb(colr), la)
                ctx.move_to(CX - w / 2, base)
                ctx.show_text(s)
                ctx.restore()
                tops.append(base - size)
                bots.append(base + size * 0.3)
            y0, y1 = max(0, int(min(tops) - 60)), min(H, int(max(bots) + 60))
            trgb, ta = surf_rgba(surf)
            band = slice(y0, y1)
            a_band = ta[band]
            if style == "chaos":
                gl = cv2.GaussianBlur(a_band, (0, 0), 14) * 0.8 * alpha
                img[band] += gl[..., None] * np.array(rgb(CYAN), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
