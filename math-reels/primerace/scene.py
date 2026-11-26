"""The prime race: odd primes split by their remainder when divided by 4 into Team 1 and Team 3. The rules on a number
line (each prime drops into its lane), the race to 1,000 as two bars, Team 3's lead up to 30,000 (with a x300 zoom on
26,861, where Team 1 leads for the first time, for two numbers), then the lead up to a billion on a log scale with
Team 1's four bursts marked, and the fair count (each odd prime square adds half a point to Team 1) drawn over it.
The first 2.4 s continue the end of the loop.   frame(f) -> uint8 1080x1920 RGB"""

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

FPS, N_FRAMES = 60, 1800
DUR = N_FRAMES / FPS
WIPE = (2.4, 2.9)
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
SLATE = "#7C88B5"
T3, T1 = AMBER, CYAN
CX = 470
MAX_TEXT_W = 820
# rules stage
NL_Y, NL_N = 700, 60
LANE3_Y, LANE1_Y, LANE_X0, LANE_X1 = 880, 1040, 232, 878
RULES = (3.0, 4.8)                                # 0 -> 60
RACE = (4.8, 9.4)                                 # 60 -> 1,000
# charts
PX0, PX1, PY0, PY1 = 112, 880, 650, 1170
IX0, IX1, IY0, IY1 = 112, 880, 1262, 1490
LIN_KEYS = [(9.7, 1000.0), (12.2, 26700.0), (12.6, 26861.0), (13.4, 26863.0), (13.9, 26900.0), (15.2, 30000.0)]
LIN_RANGE = (-6.0, 34.0)
ZOOM = (26800, 26900)
LOG = (15.5, 20.5)                                # 30,000 -> 10^9
LOG_LEFT = 1e4
FAIR = (24.8, 26.6)
SQUARES = (21.3, 27.3)


def fmt(n):
    return f"{int(n):,}"


def x_of(te):
    """How far the race has counted at effective time te."""
    if te < RULES[0]:
        return 0.0
    if te < RULES[1]:
        return 60 * (te - RULES[0]) / (RULES[1] - RULES[0])
    if te < RACE[1]:
        return 60 * (1000 / 60) ** ((te - RACE[0]) / (RACE[1] - RACE[0]))
    if te < LIN_KEYS[0][0]:
        return 1000.0
    for (t0, x0), (t1, x1) in zip(LIN_KEYS, LIN_KEYS[1:]):
        if te < t1:
            return x0 + (x1 - x0) * (te - t0) / (t1 - t0)
    if te < LOG[0]:
        return 30000.0
    if te < LOG[1]:
        u = smooth((te - LOG[0]) / (LOG[1] - LOG[0]))
        return math.exp(math.log(3e4) + (math.log(1e9) - math.log(3e4)) * u)
    return 1e9


def time_of(x):
    """Inverse of x_of on the rules and linear stages (when the count reaches x)."""
    lo, hi = RULES[0], LIN_KEYS[-1][0]
    for _ in range(50):
        mid = (lo + hi) / 2
        if x_of(mid) < x:
            lo = mid
        else:
            hi = mid
    return hi


def mix(c0, c1, u):
    return tuple(a + (b - a) * u for a, b in zip(rgb(c0), rgb(c1)))


class Scene:
    def __init__(self, data_path):
        d = np.load(data_path)
        with open(data_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.want, self.d_log, self.pi_log = d["want"], d["d_log"], d["pi_log"]
        self.d_lo, self.d_hi, self.f_lo, self.f_hi, self.f_log = d["d_lo"], d["d_hi"], d["f_lo"], d["f_hi"], d["f_log"]
        self.d_lin, self.pi_lin = d["d_lin"], d["pi_lin"]
        self.lw = np.log(np.maximum(self.want, 1))
        self.run_max = np.maximum.accumulate(np.where(self.want >= LOG_LEFT, self.d_hi, 0))
        runs = d["neg_runs"]
        eps = []
        for a, b in runs:                                 # group Team 1's leads into bursts
            if eps and a < eps[-1][1] * 1.1:
                eps[-1][1] = int(b)
            else:
                eps.append([int(a), int(b)])
        self.bursts = eps
        self.primes = [p for p in self.meta["first_primes"] if p <= NL_N]
        self.t_prime = {p: time_of(p) for p in self.primes}
        self.share = 100 * self.meta["share_team3_ahead"]
        self.fair3 = 100 * self.meta["fair_share_team3_ahead"]
        self.fair1 = 100 * self.meta["fair_share_team1_ahead"]
        c = self.meta["checkpoints"]["1e3"]
        self.c1000 = (c["team3"], c["team1"])
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        hook = [("Primes split into two teams.", "black", 70, WHITE, 300, 1), ("One almost never loses.", "dbold", 52, AMBER, 392, 1)]
        return [
            (0.0, 2.35, hook, 0, 6, "none"),
            (2.6, 4.75, [("Divide each prime by 4.", "black", 72, WHITE, 312, 1),
                         ("Remainder 1: Team 1.  Remainder 3: Team 3.", "dbold", 44, WHITE, 400, 0.95)], 8, 6, "pop"),
            (4.9, 7.0, [("It should be a tie.", "black", 76, WHITE, 312, 1), ("Team 3 takes the lead…", "dbold", 54, AMBER, 400, 1)], 8, 6, "pop"),
            (7.2, 9.4, [("…and keeps it.", "black", 84, AMBER, 312, 1),
                        (f"Up to 1,000: {self.c1000[0]} to {self.c1000[1]}.", "dbold", 52, WHITE, 400, 0.95)], 8, 6, "pop"),
            (9.6, 12.45, [("Race to 30,000.", "black", 80, WHITE, 312, 1), ("Does Team 1 ever get ahead?", "dbold", 52, CYAN, 400, 1)], 8, 6, "pop"),
            (12.6, 15.2, [("26,861: Team 1 leads!", "black", 76, CYAN, 312, 1), ("For exactly 2 numbers.", "dbold", 54, WHITE, 400, 1)], 6, 6, "chaos"),
            (15.4, 18.2, [("Now race to a billion.", "black", 72, WHITE, 312, 1),
                          (f"Team 1 gets ahead in just {len(self.bursts)} bursts.", "dbold", 48, CYAN, 400, 1)], 8, 6, "pop"),
            (18.4, 21.1, [(f"Team 3 leads {self.share:.2f}%", "black", 76, AMBER, 312, 1),
                          ("of all the numbers up to a billion.", "dbold", 48, WHITE, 400, 0.95)], 6, 6, "chaos"),
            (21.3, 24.6, [("Why? Squares.", "black", 88, WHITE, 312, 1),
                          ("Every odd square is a Team 1 number.", "dbold", 48, CYAN, 400, 1)], 8, 6, "pop"),
            (24.8, 27.3, [("Count each one as half a point:", "black", 56, WHITE, 312, 1),
                          (f"the race is fair. {self.fair3:.0f}% to {self.fair1:.0f}%.", "black", 58, CYAN, 396, 1)], 8, 6, "pop"),
            (27.5, 29.3, [("Its swings come from zeros,", "black", 66, WHITE, 312, 1), ("just like Riemann's.", "black", 66, AMBER, 396, 1)], 8, 6, "pop"),
            (29.45, DUR + 0.2, hook, 12, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1050) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ the race state

    def at(self, x):
        """(shown count, D, odd primes counted) at count x."""
        if x <= 30000:
            n = int(x)
            return n, int(self.d_lin[n]), int(self.pi_lin[n])
        k = int(np.searchsorted(self.want, x, side="right")) - 1
        return int(self.want[k]), int(self.d_log[k]), int(self.pi_log[k])

    def state(self, t):
        if t < WIPE[1]:
            a = 1.0 if t < WIPE[0] else 1 - smooth((t - WIPE[0]) / (WIPE[1] - WIPE[0] - 0.05))
            return t + DUR, a
        return t, 1.0

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        te, a = self.state(t)
        x = x_of(te)
        g_surf, g = layer()
        h_surf, hud = layer()
        if te < 9.55:
            a_r = smooth((te - 2.9) / 0.3) * (1 - smooth((te - 9.4) / 0.15))
            self._rules(g, hud, te, x, a_r)
        if 9.4 <= te < 15.5:
            a_l = smooth((te - 9.5) / 0.3) * (1 - smooth((te - 15.3) / 0.2))
            self._lin(g, hud, te, x, a_l)
        if te >= 15.3:
            a_g = smooth((te - 15.35) / 0.3) * a
            self._log(g, hud, te, x, a_g)
        self._hud(hud, te, x, a)
        g_rgb, g_a = surf_rgba(g_surf)
        band = slice(590, 1560)
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

    # ------------------------------------------------------------ stage 1: the rules, the race to 1,000

    def _nx(self, n):
        return 95 + (n - 1) * (873 - 95) / (NL_N - 1)

    def _rules(self, g, hud, te, x, a):
        if a <= 0:
            return
        a_nl = a * (1 - smooth((te - 5.0) / 0.5))                     # the number line goes once the race speeds up
        if a_nl > 0:
            font(hud, "bold", 19)
            for n in range(1, NL_N + 1):
                passed = x >= n
                xx = self._nx(n)
                if n in self.primes and n > 2:
                    col = T3 if n % 4 == 3 else T1
                    if passed:
                        g.set_source_rgba(*rgb(col), a_nl)
                        g.arc(xx, NL_Y, 5.5, 0, 2 * math.pi)
                        g.fill()
                        s = str(n)
                        hud.set_source_rgba(*rgb(col), a_nl)
                        hud.move_to(xx - hud.text_extents(s).x_advance / 2, NL_Y - 18)
                        hud.show_text(s)
                        continue
                hud.set_source_rgba(1, 1, 1, (0.35 if passed else 0.16) * a_nl)
                hud.new_sub_path()
                hud.arc(xx, NL_Y, 2.4, 0, 2 * math.pi)
                hud.fill()
            if x >= 2:
                hud.set_source_rgba(1, 1, 1, 0.5 * a_nl)
                hud.move_to(self._nx(2) - 4, NL_Y - 18)
                hud.show_text("2")
            if 0 < x < NL_N:
                xx = self._nx(max(1.0, x))
                hud.set_source_rgba(1, 1, 1, 0.8 * a_nl)
                hud.set_line_width(2)
                hud.move_to(xx, NL_Y + 14)
                hud.line_to(xx, NL_Y + 30)
                hud.stroke()
            font(hud, "bold", 19)
            hud.set_source_rgba(1, 1, 1, 0.55 * a_nl)
            show_tracked(hud, "THE NUMBERS 1 TO 60 (2 IS THE ONLY EVEN PRIME: IT SITS OUT)", 95, NL_Y + 58, 0.06 * 19)
        # lanes
        for y, name, sub, col in ((LANE3_Y, "TEAM 3", "REMAINDER 3", T3), (LANE1_Y, "TEAM 1", "REMAINDER 1", T1)):
            font(hud, "black", 32)
            hud.set_source_rgba(*rgb(col), a)
            hud.move_to(62, y + 4)
            hud.show_text(name)
            font(hud, "bold", 17)
            hud.set_source_rgba(*rgb(col), 0.75 * a)
            show_tracked(hud, sub, 62, y + 30, 0.08 * 17)
            hud.set_source_rgba(1, 1, 1, 0.12 * a)
            hud.set_line_width(2)
            hud.move_to(LANE_X0, y + 40)
            hud.line_to(LANE_X1, y + 40)
            hud.stroke()
        # beads for the primes up to 60, flying in from the number line
        a_b = a * (1 - smooth((te - 4.9) / 0.4))
        if a_b > 0:
            k3 = k1 = 0
            for p in self.primes:
                if p == 2:
                    continue
                team3 = p % 4 == 3
                k = k3 if team3 else k1
                if team3:
                    k3 += 1
                else:
                    k1 += 1
                tp = self.t_prime[p]
                if te < tp:
                    continue
                u = min(1.0, (te - tp) / 0.35)
                ue = ease_out_cubic(u)
                bx, by = LANE_X0 + 38 + k * 66, (LANE3_Y if team3 else LANE1_Y)
                sx, sy = self._nx(p), NL_Y
                xx = sx + (bx - sx) * ue
                yy = sy + (by - sy) * ue - 60 * math.sin(math.pi * u)
                col = T3 if team3 else T1
                g.set_source_rgba(*rgb(col), a_b)
                g.set_line_width(3)
                g.arc(xx, yy, 25, 0, 2 * math.pi)
                g.stroke()
                g.set_source_rgba(*rgb(col), 0.22 * a_b)
                g.arc(xx, yy, 25, 0, 2 * math.pi)
                g.fill()
                font(hud, "black", 22)
                s = str(p)
                hud.set_source_rgba(1, 1, 1, a_b * u)
                hud.move_to(xx - hud.text_extents(s).x_advance / 2, yy + 8)
                hud.show_text(s)
        # bars: the race to 1,000
        a_bar = a * smooth((te - 4.9) / 0.4)
        if a_bar > 0:
            _, D, pi = self.at(x)
            c3, c1 = (pi + D) // 2, (pi - D) // 2
            top = max(c3, c1, 1)
            for y, c, col in ((LANE3_Y, c3, T3), (LANE1_Y, c1, T1)):
                ln = (LANE_X1 - 90 - LANE_X0) * c / top
                g.set_source_rgba(*rgb(col), 0.85 * a_bar)
                g.rectangle(LANE_X0, y - 22, ln, 44)
                g.fill()
                font(hud, "black", 36)
                s = str(c)
                hud.set_source_rgba(*rgb(col), a_bar)
                hud.move_to(LANE_X0 + ln + 14, y + 13)
                hud.show_text(s)

    # ------------------------------------------------------------ stage 2: the lead up to 30,000

    def _axes(self, hud, a, title):
        hud.set_source_rgba(1, 1, 1, 0.35 * a)
        hud.set_line_width(2)
        hud.move_to(PX0, PY0 - 6)
        hud.line_to(PX0, PY1)
        hud.line_to(PX1 + 6, PY1)
        hud.stroke()
        font(hud, "bold", 20)
        hud.set_source_rgba(1, 1, 1, 0.6 * a)
        show_tracked(hud, title, PX0, PY0 - 22, 0.1 * 20)

    def _lin(self, g, hud, te, x, a):
        if a <= 0:
            return
        lo, hi = LIN_RANGE
        ys = lambda v: PY1 - (PY1 - PY0) * (v - lo) / (hi - lo)  # noqa: E731
        xs = lambda n: PX0 + (PX1 - PX0) * n / 30000.0  # noqa: E731
        self._axes(hud, a, "TEAM 3'S LEAD (TEAM 3 MINUS TEAM 1)")
        font(hud, "bold", 19)
        for v in (10, 20, 30):
            hud.set_source_rgba(1, 1, 1, 0.07 * a)
            hud.move_to(PX0, ys(v))
            hud.line_to(PX1, ys(v))
            hud.stroke()
            s = str(v)
            hud.set_source_rgba(1, 1, 1, 0.5 * a)
            hud.move_to(PX0 - 12 - hud.text_extents(s).x_advance, ys(v) + 7)
            hud.show_text(s)
        hud.set_source_rgba(1, 1, 1, 0.45 * a)
        hud.move_to(PX0, ys(0))
        hud.line_to(PX1, ys(0))
        hud.stroke()
        hud.move_to(PX0 - 24, ys(0) + 7)
        hud.show_text("0")
        for v in (10000, 20000, 30000):
            s = fmt(v)
            hud.set_source_rgba(1, 1, 1, 0.5 * a)
            hud.move_to(xs(v) - hud.text_extents(s).x_advance / 2, PY1 + 32)
            hud.show_text(s)
        # the lead, one pixel column at a time (its full range inside each column, so nothing is skipped)
        n_end = int(min(x, 30000))
        cols = int((PX1 - PX0) * n_end / 30000) + 1
        edges = np.linspace(0, n_end, cols + 1).astype(int)
        g.set_line_width(2.2)
        for c in range(cols):
            seg = self.d_lin[edges[c]:max(edges[c + 1], edges[c] + 1) + 1]
            mn, mx = int(seg.min()), int(seg.max())
            xx = PX0 + c + 0.5
            if mx > 0:
                g.set_source_rgba(*rgb(T3), a)
                g.move_to(xx, ys(max(mn, 0.4)) + 1)
                g.line_to(xx, ys(mx) - 1)
                g.stroke()
            if mn < 0:
                g.set_source_rgba(*rgb(T1), a)
                g.move_to(xx, ys(0))
                g.line_to(xx, ys(mn) + 2)
                g.stroke()
            if mn <= 0 <= mx:
                g.set_source_rgba(*rgb(WHITE), 0.8 * a)
                g.arc(xx, ys(0), 1.4, 0, 2 * math.pi)
                g.fill()
        if te >= 12.6:                                     # mark 26,861 on the big chart
            k = smooth((te - 12.6) / 0.3) * a
            hud.set_source_rgba(*rgb(T1), k)
            hud.set_line_width(2.5)
            hud.new_sub_path()
            hud.arc(xs(26861), ys(-1), 12, 0, 2 * math.pi)
            hud.stroke()
        # the x300 zoom on 26,800 .. 26,900
        if te >= 11.9:
            k = smooth((te - 11.9) / 0.3) * a
            z0, z1 = ZOOM
            zx = lambda n: IX0 + (IX1 - IX0) * (n - z0) / (z1 - z0)  # noqa: E731
            zy = lambda v: IY1 - 30 - (IY1 - IY0 - 50) * (v + 2) / 7.0  # noqa: E731
            hud.set_source_rgba(1, 1, 1, 0.05 * k)
            hud.rectangle(IX0, IY0, IX1 - IX0, IY1 - IY0)
            hud.fill()
            font(hud, "bold", 19)
            hud.set_source_rgba(1, 1, 1, 0.6 * k)
            show_tracked(hud, "ZOOM ×300: THE NUMBERS 26,800 TO 26,900", IX0 + 12, IY0 + 26, 0.08 * 19)
            hud.set_source_rgba(1, 1, 1, 0.4 * k)
            hud.set_line_width(2)
            hud.move_to(IX0, zy(0))
            hud.line_to(IX1, zy(0))
            hud.stroke()
            hud.move_to(IX0 + 6, zy(0) - 8)
            hud.show_text("0")
            n_end = int(min(max(x, z0), z1))
            prev = None
            for n in range(z0, n_end + 1):
                v = int(self.d_lin[n])
                col = T3 if v > 0 else (T1 if v < 0 else WHITE)
                g.set_source_rgba(*rgb(col), k)
                g.set_line_width(4)
                if prev is not None and prev != v:
                    g.move_to(zx(n), zy(prev))
                    g.line_to(zx(n), zy(v))
                    g.stroke()
                g.move_to(zx(n), zy(v))
                g.line_to(zx(min(n + 1, z1)), zy(v))
                g.stroke()
                prev = v
            # the primes in the window, as ticks in their team's colour
            for p in range(z0, z1 + 1):
                if p > n_end or not self._isprime(p):
                    continue
                col = T3 if p % 4 == 3 else T1
                hud.set_source_rgba(*rgb(col), k)
                hud.rectangle(zx(p) - 1.5, IY1 - 22, 3, 14)
                hud.fill()
            if x >= 26861:
                kk = smooth((te - 12.6) / 0.3) * k
                font(hud, "black", 24)
                s = "26,861: TEAM 1 AHEAD"
                hud.set_source_rgba(*rgb(T1), kk)
                hud.move_to(zx(26861) + 22, zy(-1) + 9)
                hud.show_text(s)

    @staticmethod
    def _isprime(n):
        if n < 2 or n % 2 == 0:
            return n == 2
        r = int(n ** 0.5)
        for q in range(3, r + 1, 2):
            if n % q == 0:
                return False
        return True

    # ------------------------------------------------------------ stage 3: up to a billion

    def _log(self, g, hud, te, x, a):
        if a <= 0:
            return
        xe = max(x, 3e4)
        lx0, lx1 = math.log(LOG_LEFT), math.log(xe)
        xs = lambda v: PX0 + (PX1 - PX0) * (math.log(v) - lx0) / (lx1 - lx0)  # noqa: E731
        k_end = int(np.searchsorted(self.want, xe, side="right"))
        M = max(40.0, float(self.run_max[k_end - 1]))
        a_f = smooth((te - FAIR[0]) / 0.3)
        f_draw = 0.0 if te < FAIR[0] else min(1.0, (te - FAIR[0]) / (FAIR[1] - FAIR[0]))
        lo = -0.25 * M
        if a_f > 0:
            fmin = float(self.f_lo[(self.want >= LOG_LEFT) & (np.arange(len(self.want)) < k_end)].min())
            lo = lo + (min(lo, 1.1 * fmin) - lo) * a_f
        hi = 1.12 * M
        ys = lambda v: PY1 - (PY1 - PY0) * (v - lo) / (hi - lo)  # noqa: E731
        self._axes(hud, a, "TEAM 3'S LEAD, ALL THE WAY UP")
        font(hud, "bold", 19)
        hud.set_source_rgba(1, 1, 1, 0.45 * a)
        hud.set_line_width(2)
        hud.move_to(PX0, ys(0))
        hud.line_to(PX1, ys(0))
        hud.stroke()
        hud.move_to(PX0 - 24, ys(0) + 7)
        hud.show_text("0")
        top = int(round(M, -1 if M < 1000 else -2))
        s = "+" + fmt(top)
        hud.set_source_rgba(1, 1, 1, 0.5 * a)
        hud.move_to(PX0 + 8, ys(top) + 26)
        hud.show_text(s)
        hud.set_source_rgba(1, 1, 1, 0.07 * a)
        hud.move_to(PX0, ys(top))
        hud.line_to(PX1, ys(top))
        hud.stroke()
        for name, v in (("10,000", 1e4), ("MILLION", 1e6), ("BILLION", 1e9)):
            if v > xe * 1.0001 or v < LOG_LEFT:
                continue
            xx = xs(v)
            hud.set_source_rgba(1, 1, 1, 0.12 * a)
            hud.move_to(xx, PY0)
            hud.line_to(xx, PY1)
            hud.stroke()
            hud.set_source_rgba(1, 1, 1, 0.55 * a)
            wd = hud.text_extents(name).x_advance
            hud.move_to(min(max(xx - wd / 2, PX0), PX1 - wd), PY1 + 32)
            hud.show_text(name)
        # Team 1's bursts, as bands
        for i, (b0, b1) in enumerate(self.bursts):
            if b0 > xe:
                continue
            x0, x1 = xs(max(b0, LOG_LEFT)), xs(min(b1, xe))
            if x1 - x0 < 6:
                c = (x0 + x1) / 2
                x0, x1 = c - 3, c + 3
            hud.set_source_rgba(*rgb(T1), 0.18 * a)
            hud.rectangle(x0, PY0, x1 - x0, PY1 - PY0)
            hud.fill()
            font(hud, "black", 20)
            s = fmt(b0)
            hud.set_source_rgba(*rgb(T1), 0.95 * a)
            wd = hud.text_extents(s).x_advance
            yy = PY0 + 26 + 28 * (i % 2)
            hud.move_to(min(max((x0 + x1) / 2 - wd / 2, PX0 + 4), PX1 - wd), yy)
            hud.show_text(s)
        # the lead, column by column (its full range inside each column)
        cols = PX1 - PX0
        bx = PX0 + (PX1 - PX0) * (self.lw[:k_end] - lx0) / (lx1 - lx0)
        m = self.want[:k_end] >= LOG_LEFT
        bx, dlo, dhi = bx[m], self.d_lo[:k_end][m], self.d_hi[:k_end][m]
        flo, fhi = self.f_lo[:k_end][m], self.f_hi[:k_end][m]
        ci = np.clip((bx - PX0).astype(int), 0, cols - 1)
        cmin = np.full(cols, 10 ** 9)
        cmax = np.full(cols, -10 ** 9)
        np.minimum.at(cmin, ci, dlo)
        np.maximum.at(cmax, ci, dhi)
        g.set_line_width(2.4)
        last = -1
        for c in range(cols):
            if cmax[c] < -10 ** 8:
                continue
            xx = PX0 + c + 0.5
            if last >= 0 and c - last > 1:                  # fill gaps at the left, where the samples are sparse
                pass
            if cmax[c] > 0:
                g.set_source_rgba(*rgb(T3), a)
                g.move_to(xx, ys(max(cmin[c], 0)) - 1)
                g.line_to(xx, ys(cmax[c]) - 1.5)
                g.stroke()
            if cmin[c] < 0:
                g.set_source_rgba(*rgb(T1), a)
                g.move_to(xx, ys(0))
                g.line_to(xx, ys(cmin[c]) + 1.5)
                g.stroke()
            last = c
        # the fair count, drawn left to right
        if f_draw > 0:
            fmin = np.full(cols, 1e18)
            fmax = np.full(cols, -1e18)
            np.minimum.at(fmin, ci, flo)
            np.maximum.at(fmax, ci, fhi)
            c_end = int(cols * f_draw)
            g.set_source_rgba(*rgb(WHITE), 0.85 * a * a_f)
            g.set_line_width(2.0)
            for c in range(c_end):
                if fmax[c] < -1e17:
                    continue
                xx = PX0 + c + 0.5
                g.move_to(xx, ys(fmin[c]))
                g.line_to(xx, ys(fmax[c]) - 1)
                g.stroke()
            font(hud, "bold", 19)
            s = "WHITE: THE FAIR COUNT (SQUARES ADDED BACK)"
            hud.set_source_rgba(1, 1, 1, 0.85 * a * a_f)
            show_tracked(hud, s, PX1 - tracked_width(hud, s, 0.06 * 19), PY1 + 64, 0.06 * 19)
        # the squares
        if SQUARES[0] <= te < SQUARES[1] + 0.4:
            k = a * (1 - smooth((te - SQUARES[1]) / 0.4))
            items = [(3, 9), (5, 25), (7, 49), (11, 121), (13, 169), (17, 289)]
            for i, (q, sqv) in enumerate(items):
                t_in = SQUARES[0] + 0.2 + 0.22 * i
                if te < t_in:
                    continue
                u = ease_out_back(min(1.0, (te - t_in) / 0.3))
                col_i, row = i % 3, i // 3
                bx0 = IX0 + 6 + col_i * 258
                by0 = IY0 + 14 + row * 112
                hud.set_source_rgba(*rgb(T1), 0.12 * k)
                hud.rectangle(bx0, by0, 240, 94)
                hud.fill()
                hud.set_source_rgba(*rgb(T1), 0.9 * k)
                hud.set_line_width(2)
                hud.rectangle(bx0, by0, 240, 94)
                hud.stroke()
                font(hud, "black", int(36 * min(1.0, u) + 0.5))
                s = f"{q}² = {sqv}"
                hud.set_source_rgba(1, 1, 1, k)
                hud.move_to(bx0 + 120 - hud.text_extents(s).x_advance / 2, by0 + 46)
                hud.show_text(s)
                font(hud, "bold", 21)
                s = f"{sqv} ÷ 4 → REMAINDER 1"
                hud.set_source_rgba(*rgb(T1), k)
                hud.move_to(bx0 + 120 - hud.text_extents(s).x_advance / 2, by0 + 78)
                hud.show_text(s)

    # ------------------------------------------------------------ HUD

    def _hud(self, ctx, te, x, a):
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        n, D, pi = self.at(x)
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "COUNTED UP TO", 70, 470, 0.14 * 24)
        if D > 0:
            lab, val, col = "TEAM 3 AHEAD BY", fmt(D), T3
        elif D < 0:
            lab, val, col = "TEAM 1 AHEAD BY", fmt(-D), T1
        else:
            lab, val, col = "SCORE", "TIED", WHITE
        show_tracked(ctx, lab, 910 - tracked_width(ctx, lab, 0.14 * 24), 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgba(*rgb(WHITE), 1)
        show_cells(ctx, fmt(n), 70, 545, big)
        if val[0].isdigit():
            ctx.set_source_rgba(*rgb(col), 1)
            show_cells(ctx, val, 910 - cells_width(ctx, val, big), 545, big)
        else:
            font(ctx, "dbold", 60)
            ctx.set_source_rgba(*rgb(col), 1)
            ctx.move_to(910 - ctx.text_extents(val).x_advance, 542)
            ctx.show_text(val)
        c3, c1 = (pi + D) // 2, (pi - D) // 2
        font(ctx, "bold", 22)
        s3, s1 = f"TEAM 3: {fmt(c3)}", f"TEAM 1: {fmt(c1)}"
        ctx.set_source_rgba(*rgb(T3), 0.9)
        show_tracked(ctx, s3, 70, 590, 0.08 * 22)
        ctx.set_source_rgba(*rgb(T1), 0.9)
        show_tracked(ctx, s1, 910 - tracked_width(ctx, s1, 0.08 * 22), 590, 0.08 * 22)

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
