"""What if one zero left the line? Top: the prime staircase (amber) and Riemann's formula built from the zeros (cyan);
bottom left: the critical strip with the first zeros as beads; bottom right: the wave the first zero adds to the prime
count. The first zero is pushed from real part 0.5 to 0.8: its wave swells, the staircase bends, and in the wide view
(x up to a billion and beyond) that one wave grows without limit while the on-the-line wave stays flat.
The first 2.4 s continue the end of the loop, then the picture clears and the story starts.
    frame(f) -> uint8 1080x1920 RGB"""

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "riemann"))
sys.path.insert(0, HERE)
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
from rsim import X_MAX  # noqa: E402
from wsim import (BUILD, DUR, FPS, N_FRAMES, PUSH, SIGMA_OFF, SWEEP, T_WRAP, WAVE_IN, WIPE, X_LEFT, ZOOM,  # noqa: E402,F401
                  sigma, x_edge)

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
SLATE = "#7C88B5"
CX = 470
MAX_TEXT_W = 820
# chart A: the prime count
AX0, AX1, AY0, AY1, C_TOP = 96, 866, 1010, 640, 16.0
# the strip: real part 0 .. 1 across, height 0 .. 46 up
SX0, SX1, SY0, SY1, T_TOP = 92, 332, 1560, 1160, 46.0
# the wave panel
PXL, PXR, PCY, PH = 418, 884, 1356, 172


def ax(x):
    return AX0 + (AX1 - AX0) * x / X_MAX


def ay(c):
    return AY0 - (AY0 - AY1) * c / C_TOP


def mix(c0, c1, u):
    return tuple(a + (b - a) * u for a, b in zip(rgb(c0), rgb(c1)))


def fmt(v):
    if v < 10:
        return f"{v:.2f}"
    return f"{v:,.0f}"


class Scene:
    def __init__(self, riemann_npz, rogue_npz):
        d = np.load(riemann_npz)
        r = np.load(rogue_npz)
        with open(rogue_npz.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.gamma, self.primes = d["gamma"], d["primes"]
        self.x, self.base, self.terms = d["x"], d["base"], d["terms"].astype(np.float64)
        self.rest = self.terms[1:].sum(0)                 # every zero but the first, all in
        self.sigs, self.wtop = r["sigs"], r["wtop"].astype(np.float64)
        self.xb, self.on_b, self.off_b = r["xb"], r["on_b"], r["off_b"]
        self.lxb = np.log(self.xb)
        self.cum_off = np.maximum.accumulate(np.abs(self.off_b))
        self.px = ax(self.x)
        self.n_zeros = len(self.gamma)
        self.real_max = int(round(self.meta["real_dev_max"]))
        self.rogue_1e9 = int(round(self.meta["swing_off"]["1e9"], -3))
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        hook = [("What if ONE zero left the line?", "black", 70, WHITE, 300, 1),
                ("The Riemann Hypothesis, part 2.", "dbold", 46, AMBER, 392, 1)]
        return [
            (0.0, 2.35, hook, 0, 6, "none"),
            (2.6, 5.0, [("The zeros build the primes.", "black", 68, WHITE, 312, 1),
                        ("Every zero ever found sits on one line.", "dbold", 46, WHITE, 400, 0.9)], 8, 6, "pop"),
            (5.2, 7.4, [("Riemann's guess: they ALL do.", "black", 64, AMBER, 312, 1),
                        ("Why would that matter?", "dbold", 54, WHITE, 400, 0.95)], 8, 6, "pop"),
            (7.6, 10.4, [("Push ONE zero off the line.", "black", 70, MAGENTA, 312, 1),
                         ("From 0.5 to 0.8. Nothing else changes.", "dbold", 46, WHITE, 400, 0.95)], 8, 6, "pop"),
            (10.6, 13.0, [("Its wave swells.", "black", 84, WHITE, 312, 1),
                          ("The prime staircase bends.", "dbold", 54, MAGENTA, 400, 1)], 8, 6, "pop"),
            (ZOOM[0], 16.4, [("Now zoom out.", "black", 88, WHITE, 312, 1),
                             ("Follow that wave up to a billion.", "dbold", 50, WHITE, 400, 0.95)], 8, 6, "pop"),
            (16.6, 19.6, [("It never stops growing.", "black", 78, MAGENTA, 312, 1),
                          ("It drowns out every other zero.", "dbold", 50, WHITE, 400, 0.95)], 8, 6, "pop"),
            (19.8, 23.0, [(f"Real primes stray by {self.real_max:,} at most.", "black", 58, CYAN, 312, 1),
                          (f"With one rogue zero: {self.rogue_1e9:,}. And rising.", "black", 50, MAGENTA, 396, 1)], 6, 6, "chaos"),
            (23.2, 26.2, [("On the line, every wave", "black", 66, WHITE, 312, 1),
                          ("is as small as it can be.", "black", 66, CYAN, 396, 1)], 8, 6, "pop"),
            (26.4, 29.3, [("12 trillion zeros checked.", "black", 70, WHITE, 312, 1),
                          ("None off the line. Ever.", "dbold", 56, AMBER, 400, 1)], 8, 6, "pop"),
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
        """(effective time, alpha of the running picture)."""
        if t < WIPE[1]:
            a = 1.0 if t < WIPE[0] else 1 - smooth((t - WIPE[0]) / (WIPE[1] - WIPE[0] - 0.05))
            return min(t + DUR, T_WRAP), a
        return t, 1.0

    def wave_top(self, sig):
        """The first zero's wave on the top chart's grid, for a real part between 0.5 and 0.8."""
        u = (sig - self.sigs[0]) / (self.sigs[-1] - self.sigs[0]) * (len(self.sigs) - 1)
        i = min(int(u), len(self.sigs) - 2)
        fr = u - i
        return self.wtop[i] * (1 - fr) + self.wtop[i + 1] * fr

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        te, a = self.state(t)
        sig = sigma(te)
        off = (sig - 0.5) / (SIGMA_OFF - 0.5)             # 0 on the line .. 1 fully off
        g_surf, g = layer()
        h_surf, hud = layer()
        self._chart(g, hud, te, a, sig, off)
        self._strip(g, hud, t, te, a, sig, off)
        size = self._wave_panel(g, hud, te, a, sig, off)
        self._hud(hud, te, a, sig, off, size)
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

    def _chart(self, g, hud, te, a, sig, off):
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
        for m in range(1, int(X_MAX) + 1):
            hud.set_source_rgba(1, 1, 1, 0.22)
            hud.arc(ax(m), AY0 + 26, 2.2, 0, 2 * math.pi)
            hud.fill()
        font(hud, "bold", 19)
        for p in self.primes:
            g.set_source_rgba(*rgb(AMBER), a)
            g.arc(ax(p), AY0 + 26, 3.2, 0, 2 * math.pi)
            g.fill()
            s = str(p)
            hud.set_source_rgba(*rgb(AMBER), 0.95 * a)
            hud.move_to(ax(p) - hud.text_extents(s).x_advance / 2, AY0 + 58)
            hud.show_text(s)
        # the true staircase
        g.set_source_rgba(*rgb(AMBER), a * 0.55)
        g.set_line_width(5)
        g.move_to(ax(0), ay(0))
        c = 0
        for p in self.primes:
            g.line_to(ax(p), ay(c))
            c += 1
            g.line_to(ax(p), ay(c))
        g.line_to(ax(X_MAX), ay(c))
        g.stroke()
        # Riemann's formula: rebuilt zero by zero in the recap, then with the first zero wherever it has been pushed
        if te < BUILD[0]:
            return
        n_used = self.n_zeros * smooth((te - BUILD[0]) / (BUILD[1] - BUILD[0])) ** 1.6 if te < BUILD[1] else self.n_zeros
        if n_used >= self.n_zeros:
            curve = self.base + self.rest + self.wave_top(sig)
        else:
            w = np.clip(n_used - np.arange(self.n_zeros), 0, 1)
            curve = self.base + w @ self.terms
        py = ay(np.clip(curve, -0.8, C_TOP + 0.3))
        col = mix(CYAN, MAGENTA, off)
        if off > 0 and n_used >= self.n_zeros:
            py0 = ay(np.clip(self.base + self.rest + self.wtop[0], -0.8, C_TOP + 0.3))
            g.set_source_rgba(*rgb(MAGENTA), 0.32 * a * off)
            g.move_to(self.px[0], py[0])
            for xx, yy in zip(self.px[1:], py[1:]):
                g.line_to(xx, yy)
            for xx, yy in zip(self.px[::-1], py0[::-1]):
                g.line_to(xx, yy)
            g.close_path()
            g.fill()
            g.set_source_rgba(*rgb(CYAN), 0.75 * a * off)
            g.set_line_width(2.5)
            g.move_to(self.px[0], py0[0])
            for xx, yy in zip(self.px[1:], py0[1:]):
                g.line_to(xx, yy)
            g.stroke()
        g.set_source_rgba(*col, a)
        g.set_line_width(4.5)
        g.move_to(self.px[0], py[0])
        for xx, yy in zip(self.px[1:], py[1:]):
            g.line_to(xx, yy)
        g.stroke()
        font(hud, "bold", 20)
        if off > 0.02:
            s = f"■ ZERO 1 MOVED TO {sig:.2f}"
        else:
            s = f"■ BUILT FROM {int(n_used):,} ZEROS"
        hud.set_source_rgba(*col, 0.95 * a)
        show_tracked(hud, s, AX1 - tracked_width(hud, s, 0.06 * 20), AY1 - 28, 0.06 * 20)

    # ------------------------------------------------------------ the strip

    def _strip(self, g, hud, t, te, a, sig, off):
        xl = (SX0 + SX1) / 2
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.4 * t)
        hud.set_source_rgba(1, 1, 1, 0.045)
        hud.rectangle(SX0, SY1, SX1 - SX0, SY0 - SY1)
        hud.fill()
        hud.set_line_width(2)
        hud.set_dash([5, 7])
        hud.set_source_rgba(1, 1, 1, 0.3)
        for x in (SX0, SX1):
            hud.move_to(x, SY0)
            hud.line_to(x, SY1)
            hud.stroke()
        hud.set_dash([])
        g.set_source_rgba(*rgb(AMBER), 0.8)
        g.set_line_width(3)
        g.move_to(xl, SY0)
        g.line_to(xl, SY1)
        g.stroke()
        font(hud, "bold", 20)
        hud.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(hud, "WHERE THE ZEROS SIT", SX0, SY1 - 46, 0.1 * 20)
        font(hud, "bold", 19)
        for s, x, c in (("0", SX0, (1, 1, 1)), ("0.5", xl, rgb(AMBER)), ("1", SX1, (1, 1, 1))):
            hud.set_source_rgba(*c, 0.75)
            hud.move_to(x - hud.text_extents(s).x_advance / 2, SY1 - 12)
            hud.show_text(s)
        if te < BUILD[0]:
            return
        n_show = int(np.searchsorted(self.gamma, T_TOP))
        for k in range(n_show):
            t_in = BUILD[0] + 0.2 * k
            if te < t_in:
                continue
            pop = max(0.0, 1 - (te - t_in) / 0.3)
            y = SY0 - self.gamma[k] * (SY0 - SY1) / T_TOP
            if k == 0:
                x = SX0 + sig * (SX1 - SX0)
                if off > 0:                                # the trail back to the line
                    g.set_source_rgba(*rgb(MAGENTA), 0.7 * a)
                    g.set_line_width(2.5)
                    g.set_dash([5, 6])
                    g.move_to(xl, y)
                    g.line_to(x, y)
                    g.stroke()
                    g.set_dash([])
                    hud.set_source_rgba(1, 1, 1, 0.45 * a)   # where it was
                    hud.set_line_width(2)
                    hud.new_sub_path()
                    hud.arc(xl, y, 6, 0, 2 * math.pi)
                    hud.stroke()
                col = mix(CYAN, MAGENTA, off)
                g.set_source_rgba(*col, a)
                g.arc(x, y, 6.5 + 5 * pop + 2.5 * off * pulse, 0, 2 * math.pi)
                g.fill()
                font(hud, "black", 22)
                if off > 0.02:
                    hud.set_source_rgba(*rgb(MAGENTA), a)
                    hud.move_to(x - hud.text_extents(f"{sig:.2f}").x_advance / 2, y + 34)
                    hud.show_text(f"{sig:.2f}")
                font(hud, "bold", 19)
                hud.set_source_rgba(*col, 0.95 * a)
                s = "ZERO 1"
                hud.move_to(SX0 + 8, y + 7)
                hud.show_text(s)
            else:
                g.set_source_rgba(*rgb(CYAN), a)
                g.arc(xl, y, 5.5 + 5 * pop, 0, 2 * math.pi)
                g.fill()

    # ------------------------------------------------------------ the wave panel

    def _wave_panel(self, g, hud, te, a, sig, off):
        """Draws the first zero's wave; returns its largest size over the range in view (None before it is drawn)."""
        font(hud, "bold", 20)
        hud.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(hud, "ZERO 1'S WAVE IN THE PRIME COUNT", PXL, SY1 - 46, 0.1 * 20)
        hud.set_source_rgba(1, 1, 1, 0.3)
        hud.set_line_width(2)
        hud.move_to(PXL, PCY)
        hud.line_to(PXR, PCY)
        hud.stroke()
        hud.set_source_rgba(1, 1, 1, 0.3)
        hud.move_to(PXL, PCY - PH - 8)
        hud.line_to(PXL, PCY + PH + 8)
        hud.stroke()
        if te < WAVE_IN[0]:
            return None
        a_lin = 1 - smooth((te - ZOOM[0]) / 0.15)
        a_wide = smooth((te - ZOOM[0] - 0.15) / 0.15)
        font(hud, "bold", 19)
        size = None
        if a_lin > 0:                                      # x from 0 to 50, one unit of height = PH pixels
            m = self.x >= 4
            xs = PXL + (PXR - PXL) * self.x[m] / X_MAX
            w_on, w_now = self.wtop[0][m], self.wave_top(sig)[m]
            reveal = PXL + (PXR - PXL) * min(1.0, (te - WAVE_IN[0]) / (WAVE_IN[1] - WAVE_IN[0]))
            for wv, col, alpha, lw in ((w_on, rgb(CYAN), 1.0 - 0.45 * min(1.0, off * 4), 3.5), (w_now, rgb(MAGENTA), min(1.0, off * 4), 4.0)):
                if alpha <= 0:
                    continue
                g.set_source_rgba(*col, a * a_lin * alpha)
                g.set_line_width(lw)
                first = True
                for xx, yy in zip(xs, PCY - PH * np.clip(wv, -1.1, 1.1)):
                    if xx > reveal:
                        break
                    (g.move_to if first else g.line_to)(xx, yy)
                    first = False
                g.stroke()
            for v in (10, 20, 30, 40, 50):
                s = str(v)
                hud.set_source_rgba(1, 1, 1, 0.5 * a_lin)
                hud.move_to(PXL + (PXR - PXL) * v / X_MAX - hud.text_extents(s).x_advance / 2, PCY + PH + 34)
                hud.show_text(s)
            hud.set_source_rgba(1, 1, 1, 0.5 * a_lin)
            hud.move_to(PXL + 8, PCY - PH + 6)
            hud.show_text("+1")
            size = float(np.abs(w_now).max())
        if a_wide > 0:                                     # x from 10 to the moving right edge, log scale, auto height
            xe = x_edge(te)
            n = int(np.searchsorted(self.xb, xe, side="right"))
            top = 1.25 * self.cum_off[n - 1]
            lx0, lx1 = math.log(X_LEFT), math.log(xe)
            xs = PXL + (PXR - PXL) * (self.lxb[:n] - lx0) / (lx1 - lx0)
            for wv, col, lw in ((self.off_b, rgb(MAGENTA), 2.6), (self.on_b, rgb(CYAN), 3.2)):
                g.set_source_rgba(*col, a * a_wide)
                g.set_line_width(lw)
                ys = PCY - PH * wv[:n] / top
                g.move_to(xs[0], ys[0])
                for xx, yy in zip(xs[1:], ys[1:]):
                    g.line_to(xx, yy)
                g.stroke()
            for name, v in (("THOUSAND", 1e3), ("MILLION", 1e6), ("BILLION", 1e9), ("TRILLION", 1e12)):
                if v > xe:
                    continue
                xx = PXL + (PXR - PXL) * (math.log(v) - lx0) / (lx1 - lx0)
                hud.set_source_rgba(1, 1, 1, 0.14 * a_wide * a)
                hud.move_to(xx, PCY - PH)
                hud.line_to(xx, PCY + PH)
                hud.stroke()
                hud.set_source_rgba(1, 1, 1, 0.55 * a_wide * a)
                wd = hud.text_extents(name).x_advance
                hud.move_to(min(xx - wd / 2, PXR - wd), PCY + PH + 34)
                hud.show_text(name)
            s = "+" + fmt(top)
            hud.set_source_rgba(1, 1, 1, 0.55 * a_wide * a)
            hud.move_to(PXL + 8, PCY - PH + 6)
            hud.show_text(s)
            size = float(self.cum_off[n - 1])
        # legend
        font(hud, "bold", 19)
        hud.set_source_rgba(*rgb(CYAN), 0.95 * a)
        show_tracked(hud, "■ ON THE LINE", PXL + 8, PCY + PH - 8, 0.05 * 19)
        if off > 0.02:
            s = "■ OFF THE LINE"
            hud.set_source_rgba(*rgb(MAGENTA), 0.95 * a)
            show_tracked(hud, s, PXR - tracked_width(hud, s, 0.05 * 19), PCY + PH - 8, 0.05 * 19)
        return size

    # ------------------------------------------------------------ HUD

    def _hud(self, ctx, te, a, sig, off, size):
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "ZERO 1: REAL PART", 70, 470, 0.14 * 24)
        lab = "ITS WAVE REACHES"
        show_tracked(ctx, lab, 910 - tracked_width(ctx, lab, 0.14 * 24), 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        col = mix(CYAN, MAGENTA, off)
        ctx.set_source_rgba(*col, a)
        show_cells(ctx, f"{sig:.2f}", 70, 545, big)
        if size is not None:
            s = fmt(size)
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
                img[band] += gl[..., None] * np.array(rgb(MAGENTA), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
