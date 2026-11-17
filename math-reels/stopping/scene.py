"""The 37% rule: when do you stop looking? One life of 100 people shown bar by bar (taller = better match), the rule
"say no to the first 37, then yes to the first one better than all of them", then 10,000 lives filling a 100x100 grid
(cyan = found the very best, white = someone else, magenta = alone), then every cutoff from 0 to 99 as a curve, and
the catch: the share who end up alone, which crosses the curve at its peak.  frame(f) -> uint8 1080x1920 RGB."""

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
import stopsim as sim  # noqa: E402

FPS = 60
N_FRAMES = 1560                                  # 26 s
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
SLATE, DIM, STUB = "#7C88B5", "#C9CEE6", "#3A4366"
CX = 470
MAX_TEXT_W = 820
N, R = sim.N, sim.R
# the row of 100 people
BX0, BPITCH, BW = 70.0, 8.1, 5.4
DEMO_G, LIVES_G = (1250, 520), (880, 225)       # (baseline y, tallest bar) for the two layouts
WAVE_K = 13                                      # the waiting crowd sways 13 times per loop
# times (s)
LOOK0, LOOK1 = 2.5, 4.6                          # the first 37 appear
LEAP0, LEAP_DT = 4.75, 0.075                     # then one every 75 ms until the yes
LIVES0, SLOW_DT, SLOW_N, FAST1 = 9.4, 0.42, 6, 14.5
C0, C1 = 16.7, 19.4                              # the curve draws, cutoff 0 -> 99
A0, A1 = 20.15, 21.0                             # the "alone" line draws
LOOP_FADE = (25.3, 26.0)
GRID_C, GRID_PITCH, GRID_CELL = (470, 1300), 4.2, 3.4
# the curve chart
PX0, PX1, PY0, PY1 = 130, 860, 660, 1400          # 0..99 across, 0..100% up


def ordinal(k):
    return {1: "THE BEST", 2: "2ND BEST", 3: "3RD BEST"}.get(k, f"{k}TH BEST")


class Scene:
    def __init__(self):
        self.ranks = sim.lives()
        self.pick, self.thr = sim.outcome(self.ranks)
        got = np.take_along_axis(self.ranks, np.maximum(self.pick, 0)[:, None], 1)[:, 0]
        self.picked_rank = np.where(self.pick >= 0, got, 0)
        self.code = np.where(self.pick < 0, 2, np.where(self.picked_rank == N, 0, 1))     # 0 best, 1 settled, 2 alone
        self.cum = [np.concatenate([[0], np.cumsum(self.code == k)]) for k in range(3)]
        self.curve = sim.per_cutoff(self.ranks)
        self.curve_exact = np.array([sim.exact(r) for r in range(N)])
        b = self.ranks.argmax(1)
        self.alone_curve = np.array([float((b < r).mean()) for r in range(N)])
        self.pct = [100 * float((self.code == k).mean()) for k in range(3)]
        self.demo = int(np.argmax(self.code == 0))
        self.dpick = int(self.pick[self.demo])
        self.T_YES = LEAP0 + (self.dpick - R) * LEAP_DT
        self.REST0, self.REST1 = self.T_YES + 0.65, self.T_YES + 1.25
        self.T_BEST = self.REST1 + 0.1
        self.cols = {k: np.array(srgb_to_linear(rgb(c)), np.float32) for k, c in
                     (("cyan", CYAN), ("mag", MAGENTA), ("dim", DIM), ("amber", AMBER))}
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        hook = [("When do you stop looking?", "black", 74, WHITE, 300, 1),
                ("100 people. One at a time. No going back.", "dbold", 42, WHITE, 392, 0.85)]
        return [
            (0.0, 2.3, hook, 0, 6, "none"),
            (2.45, 4.5, [("The 37% rule:", "black", 84, AMBER, 312, 1), ("say no to the first 37.", "dbold", 52, WHITE, 400, 0.95)], 8, 6, "pop"),
            (4.65, self.T_YES - 0.05, [("Then say yes to the first one", "black", 56, WHITE, 312, 1),
                                       ("better than all of them.", "black", 56, CYAN, 396, 1)], 8, 6, "pop"),
            (self.T_YES + 0.05, self.T_BEST - 0.12, [(f"Person {self.dpick + 1}: yes.", "black", 84, AMBER, 340, 1)], 6, 6, "pop"),
            (self.T_BEST, 9.05, [("The best of all 100.", "black", 76, CYAN, 340, 1)], 6, 6, "chaos"),
            (9.2, 11.0, [("Luck? Run 10,000 lives.", "black", 66, WHITE, 312, 1),
                         ("Same rule. New random order each time.", "dbold", 42, WHITE, 396, 0.85)], 8, 6, "pop"),
            (14.6, 16.45, [(f"{self.pct[0]:.1f}% found THE best.", "black", 70, CYAN, 312, 1),
                           (f"Picking at random: {100 / N:.0f}%.", "dbold", 46, WHITE, 400, 0.85)], 6, 6, "chaos"),
            (16.6, 18.15, [("Why 37?", "black", 84, AMBER, 312, 1), ("Try every number.", "dbold", 52, WHITE, 400, 0.9)], 8, 6, "pop"),
            (18.3, 20.0, [("Too few: you jump too soon.", "black", 54, WHITE, 312, 1),
                          ("Too many: the best already left.", "black", 54, WHITE, 396, 1)], 8, 6, "pop"),
            (20.15, 22.3, [("The catch:", "black", 76, MAGENTA, 312, 1),
                           (f"{100 * self.alone_curve[R]:.0f}% of the time, you end up alone.", "dbold", 46, WHITE, 400, 1)], 8, 6, "pop"),
            (22.45, 25.15, [("The best one was", "black", 66, WHITE, 312, 1), (f"in the first {R}.", "black", 66, MAGENTA, 396, 1)], 8, 6, "pop"),
            (25.3, 26.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1000) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ schedule

    def reveal_time(self, k):
        """When person k of the example life appears."""
        if k < R:
            return LOOK0 + k * (LOOK1 - LOOK0) / R
        if k <= self.dpick:
            return LEAP0 + (k - R) * LEAP_DT
        return self.REST0 + (k - self.dpick - 1) * (self.REST1 - self.REST0) / max(1, N - 1 - self.dpick)

    def lives_done(self, t):
        if t < LIVES0:
            return 0
        t_fast = LIVES0 + SLOW_N * SLOW_DT
        if t < t_fast:
            return min(SLOW_N, int((t - LIVES0) // SLOW_DT) + 1)
        if t >= FAST1:
            return sim.LIVES
        u = (t - t_fast) / (FAST1 - t_fast)
        return min(sim.LIVES, SLOW_N + int((sim.LIVES - SLOW_N) * u ** 2.2))

    def phase(self, t):
        if t >= LOOP_FADE[0]:
            return "loop"
        if t < LOOK0:
            return "hook"
        if t < LIVES0 - 0.2:
            return "demo"
        if t < C0:
            return "lives"
        return "curve"

    # ------------------------------------------------------------ frame

    def geom(self, t, ph):
        if ph in ("lives", "curve"):
            u = smooth((t - (LIVES0 - 0.2)) / 0.45)
            return tuple(a + (b - a) * u for a, b in zip(DEMO_G, LIVES_G))
        return DEMO_G

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        ph = self.phase(t)
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if ph == "loop" else 0.0
        a_chart = 1 - smooth((t - C0) / 0.35) if t >= C0 else 1.0           # bars and grid fade out
        a_curve = smooth((t - C0) / 0.35) * (1 - mix) if t >= C0 else 0.0
        if ph == "loop" and mix > 0.5:
            a_chart, ph_draw = smooth((mix - 0.5) / 0.5), "hook"
        else:
            ph_draw = ph
        self.base_y, self.bmax = self.geom(t, ph_draw)
        self.t = t
        emis = np.zeros((H, W, 3), np.float32)
        done = self.lives_done(t) if ph_draw in ("lives", "curve") else 0
        if ph_draw == "lives" or (ph_draw == "curve" and a_chart > 0):
            self._grid(emis, done, a_chart)
        gsurf, gctx = layer()
        hsurf, hctx = layer()
        if ph_draw in ("hook", "demo") or (ph_draw in ("lives", "curve") and a_chart > 0):
            self._people(gctx, hctx, t, ph_draw, done, a_chart)
        if a_curve > 0:
            self._curve(gctx, hctx, t, a_curve)
        g_rgb, g_a = surf_rgba(gsurf)
        band = slice(560, 1560)
        lin_g = np.zeros_like(emis)
        sub = g_rgb[band] / np.maximum(g_a[band], 1e-4)[..., None]
        lin_g[band] = srgb_to_linear(sub).astype(np.float32) * g_a[band][..., None]
        emis += lin_g * 1.1
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.35 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.25 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.4 * (emis + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(hctx, t, ph_draw, done)
        trgb, ta = surf_rgba(hsurf)
        over(img, trgb, ta)
        self._headlines(img, f, t)
        return dither(img, f)

    # ------------------------------------------------------------ the row of people

    def _stub(self, k):
        return 30 + 10 * (0.5 + 0.5 * math.sin(2 * math.pi * WAVE_K * self.t / (N_FRAMES / FPS) - 0.22 * k))

    def _bar(self, ctx, k, h, col, a):
        x = BX0 + k * BPITCH
        BASE_Y = self.base_y
        ctx.set_source_rgba(*rgb(col), a)
        ctx.arc(x + BW / 2, BASE_Y - h - 6, 3.0, 0, 2 * math.pi)
        ctx.fill()
        r = min(BW / 2, h / 2)
        ctx.new_sub_path()
        ctx.arc(x + BW - r, BASE_Y - h + r, r, -math.pi / 2, 0)
        ctx.line_to(x + BW, BASE_Y)
        ctx.line_to(x, BASE_Y)
        ctx.arc(x + r, BASE_Y - h + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        ctx.fill()

    def _people(self, g, hud, t, ph, done, a):
        q = a if ph == "hook" else (1 - smooth((t - LOOK0 + 0.1) / 0.4) if ph == "demo" else 0.0)
        if q > 0:                                        # the big question mark over the waiting crowd
            font(g, "black", 300)
            g.set_source_rgba(*rgb(SLATE), 0.42 * q)
            g.move_to(CX - g.text_extents("?").x_advance / 2, self.base_y - 120)
            g.show_text("?")
        if ph == "hook":
            for k in range(N):
                self._bar(g, k, self._stub(k), STUB, 0.95 * a)
            return
        if ph == "demo":
            self._demo_life(g, hud, t)
            return
        life = max(0, done - 1)
        slow = done <= SLOW_N
        ranks, pick, code = self.ranks[life], int(self.pick[life]), int(self.code[life])
        fade = (1.0 if slow else 0.55) * a
        for k in range(N):
            h = self.bmax * ranks[k] / N
            if k == pick:
                col, aa = (CYAN if code == 0 else AMBER), 1.0
            elif code == 2 and ranks[k] == N:
                col, aa = MAGENTA, 1.0
            elif k < R:
                col, aa = SLATE, 0.8
            elif pick >= 0 and k > pick:
                col, aa = "#5A6080", 0.6
            else:
                col, aa = DIM, 0.45
            self._bar(g, k, h, col, aa * fade)
        self._threshold(hud, self.thr[life], "BEST OF THE FIRST 37", 0.6 * fade)
        self._divider(hud, 0.7 * fade)
        # outcome label under the row
        if code == 0:
            s, col = f"LIFE {life + 1}: FOUND THE BEST", CYAN
        elif code == 1:
            s, col = f"LIFE {life + 1}: SETTLED FOR THE {ordinal(N + 1 - int(self.picked_rank[life]))}", AMBER
        else:
            s, col = f"LIFE {life + 1}: ALONE (THE BEST CAME EARLY)", MAGENTA
        font(hud, "black", 26)
        hud.set_source_rgba(*rgb(col), fade)
        w = tracked_width(hud, s, 0.05 * 26)
        show_tracked(hud, s, CX - w / 2, self.base_y + 46, 0.05 * 26)

    def _demo_life(self, g, hud, t):
        ranks = self.ranks[self.demo]
        best_shown = t >= self.T_BEST
        for k in range(N):
            tk = self.reveal_time(k)
            if t < tk:
                self._bar(g, k, self._stub(k), STUB, 0.95)
                continue
            u = min(1.0, (t - tk) / 0.12)
            h0 = self._stub(k)
            h = max(4.0, h0 + (self.bmax * ranks[k] / N - h0) * ease_out_back(u, 1.2))
            if k == self.dpick:
                col, aa = (CYAN if best_shown else AMBER), 1.0
            elif k < R:
                col, aa = SLATE, 0.85
            elif k < self.dpick:
                col, aa = DIM, 0.5
            else:
                col, aa = "#5A6080", 0.65
            self._bar(g, k, h, col, aa)
        # the running best of the first 37
        n_look = sum(1 for k in range(R) if t >= self.reveal_time(k))
        if n_look:
            thr = int(ranks[:n_look].max())
            done_look = n_look == R
            self._threshold(hud, thr, "BEST OF THE FIRST 37" if done_look else "BEST SO FAR", 0.75)
        self._divider(hud, 0.8 if t >= LOOK0 else 0.0)
        top = self.base_y - self.bmax
        font(hud, "bold", 22)
        hud.set_source_rgba(1, 1, 1, 0.6)
        show_tracked(hud, "TALLER = BETTER MATCH", BX0, top - 40, 0.1 * 22)
        s = f"LIFE {self.demo + 1} OF {sim.LIVES:,}"
        w = tracked_width(hud, s, 0.1 * 22)
        show_tracked(hud, s, BX0 + N * BPITCH - w, top - 40, 0.1 * 22)
        # region labels
        hud.set_source_rgba(*rgb(SLATE), 1)
        s = "FIRST 37: LOOKING"
        w = tracked_width(hud, s, 0.06 * 22)
        show_tracked(hud, s, BX0 + R * BPITCH / 2 - w / 2, self.base_y + 44, 0.06 * 22)
        if t >= LEAP0 - 0.15:
            hud.set_source_rgba(*rgb(AMBER), 1)
            s = "THEN: FIRST ONE BETTER = YES"
            w = tracked_width(hud, s, 0.06 * 22)
            show_tracked(hud, s, BX0 + (R + N) * BPITCH / 2 - w / 2, self.base_y + 44, 0.06 * 22)
        if t >= self.T_YES:
            x = BX0 + self.dpick * BPITCH + BW / 2
            y = self.base_y - self.bmax * ranks[self.dpick] / N - 26
            s, col = ("THE BEST", CYAN) if best_shown else ("YES", AMBER)
            font(hud, "black", 34)
            hud.set_source_rgba(*rgb(col), 1)
            hud.move_to(x - hud.text_extents(s).x_advance / 2, y)
            hud.show_text(s)

    def _threshold(self, hud, thr, label, a):
        y = self.base_y - self.bmax * thr / N - 0.5
        hud.set_source_rgba(1, 1, 1, a)
        hud.set_line_width(2)
        hud.set_dash([7, 6])
        hud.move_to(BX0 - 6, y)
        hud.line_to(BX0 + N * BPITCH, y)
        hud.stroke()
        hud.set_dash([])
        font(hud, "bold", 20)
        w = tracked_width(hud, label, 0.08 * 20)
        show_tracked(hud, label, BX0 + N * BPITCH - w, y - 22, 0.08 * 20)

    def _divider(self, hud, a):
        if a <= 0:
            return
        x = BX0 + R * BPITCH - (BPITCH - BW) / 2
        hud.set_source_rgba(*rgb(AMBER), a)
        hud.set_line_width(2)
        hud.set_dash([5, 6])
        hud.move_to(x, self.base_y + 10)
        hud.line_to(x, self.base_y - self.bmax - 14)
        hud.stroke()
        hud.set_dash([])

    def _grid(self, emis, n, a):
        cells = np.zeros((sim.LIVES, 3), np.float32)
        if n:
            c = self.code[:n]
            cells[:n][c == 0] = self.cols["cyan"] * 0.6
            cells[:n][c == 1] = self.cols["dim"] * 0.16
            cells[:n][c == 2] = self.cols["mag"] * 0.55
        img = cv2.resize(cells.reshape(100, 100, 3), (420, 420), interpolation=cv2.INTER_NEAREST)
        x0, y0 = GRID_C[0] - 210, GRID_C[1] - 210
        emis[y0:y0 + 420, x0:x0 + 420] += img * a

    # ------------------------------------------------------------ the curve

    def _xy(self, r, p):
        return PX0 + (PX1 - PX0) * r / (N - 1), PY1 - (PY1 - PY0) * p

    def _curve(self, g, hud, t, a):
        u = smooth((t - C0) / (C1 - C0))
        r_now = (N - 1) * u
        # axes and ticks
        hud.set_source_rgba(1, 1, 1, 0.35 * a)
        hud.set_line_width(2)
        hud.move_to(PX0, PY0 - 10)
        hud.line_to(PX0, PY1)
        hud.line_to(PX1 + 10, PY1)
        hud.stroke()
        font(hud, "bold", 24)
        for p in (0.25, 0.5, 0.75, 1.0):
            x, y = self._xy(0, p)
            hud.set_source_rgba(1, 1, 1, 0.08 * a)
            hud.move_to(PX0, y)
            hud.line_to(PX1, y)
            hud.stroke()
            s = f"{100 * p:.0f}%"
            hud.set_source_rgba(1, 1, 1, 0.5 * a)
            hud.move_to(PX0 - 12 - hud.text_extents(s).x_advance, y + 7)
            hud.show_text(s)
        for r in (0, 20, 40, 60, 80, 99):
            x, _ = self._xy(r, 0)
            s = str(r)
            hud.set_source_rgba(1, 1, 1, 0.5 * a)
            hud.move_to(x - hud.text_extents(s).x_advance / 2, PY1 + 36)
            hud.show_text(s)
        font(hud, "bold", 22)
        s = "SAY NO TO THE FIRST ... PEOPLE"
        w = tracked_width(hud, s, 0.1 * 22)
        hud.set_source_rgba(1, 1, 1, 0.65 * a)
        show_tracked(hud, s, (PX0 + PX1) / 2 - w / 2, PY1 + 80, 0.1 * 22)
        # legend
        font(hud, "bold", 22)
        lx, ly = PX0 + 30, PY0 + 40
        hud.set_source_rgba(*rgb(CYAN), a)
        hud.set_line_width(5)
        hud.move_to(lx, ly - 8)
        hud.line_to(lx + 40, ly - 8)
        hud.stroke()
        show_tracked(hud, "10,000 LIVES", lx + 54, ly, 0.06 * 22)
        hud.set_source_rgba(1, 1, 1, 0.6 * a)
        hud.set_line_width(2)
        hud.set_dash([4, 6])
        hud.move_to(lx, ly + 32)
        hud.line_to(lx + 40, ly + 32)
        hud.stroke()
        hud.set_dash([])
        show_tracked(hud, "EXACT FORMULA", lx + 54, ly + 40, 0.06 * 22)
        # exact curve (thin dashed), then the 10,000-life curve up to r_now
        hud.set_source_rgba(1, 1, 1, 0.35 * a)
        hud.set_line_width(2)
        hud.set_dash([4, 6])
        for r in range(N):
            x, y = self._xy(r, self.curve_exact[r])
            (hud.move_to if r == 0 else hud.line_to)(x, y)
        hud.stroke()
        hud.set_dash([])
        self._polyline(g, self.curve, r_now, CYAN, a)
        x, y = self._xy(r_now, np.interp(r_now, np.arange(N), self.curve))
        g.set_source_rgba(*rgb(WHITE), a)
        g.arc(x, y, 9, 0, 2 * math.pi)
        g.fill()
        font(hud, "black", 26)
        if r_now >= 80:
            x, y = self._xy(80, self.curve[80])
            hud.set_source_rgba(*rgb(CYAN), a * smooth((r_now - 80) / 6))
            hud.move_to(x - 30, y - 22)
            hud.show_text("FOUND THE BEST")
        if r_now >= R:
            k = smooth((r_now - R) / 8) * a
            x, y = self._xy(R, self.curve[R])
            hud.set_source_rgba(*rgb(AMBER), 0.8 * k)
            hud.set_line_width(2)
            hud.set_dash([5, 6])
            hud.move_to(x, PY1)
            hud.line_to(x, y)
            hud.stroke()
            hud.set_dash([])
            s = f"{R} → {100 * self.curve[R]:.1f}%"
            font(hud, "black", 30)
            hud.set_source_rgba(*rgb(AMBER), k)
            hud.move_to(x - hud.text_extents(s).x_advance / 2, y - 26)
            hud.show_text(s)
        # the catch: the share who end up alone
        if t >= A0:
            v = smooth((t - A0) / (A1 - A0))
            self._polyline(g, self.alone_curve, (N - 1) * v, MAGENTA, a)
            if v >= 1:
                k = smooth((t - A1) / 0.4) * a
                x, y = self._xy(70, self.alone_curve[70])
                font(hud, "black", 26)
                hud.set_source_rgba(*rgb(MAGENTA), k)
                hud.move_to(x - 236, y - 6)
                hud.show_text("ENDED UP ALONE")
                x, y = self._xy(R, self.alone_curve[R])
                pulse = 0.5 + 0.5 * math.sin(2 * math.pi * 1.2 * (t - A1))
                hud.set_source_rgba(*rgb(WHITE), k * (0.55 + 0.45 * pulse))
                hud.set_line_width(3)
                hud.arc(x, y, 20, 0, 2 * math.pi)
                hud.stroke()

    def _polyline(self, g, vals, r_now, col, a):
        if r_now <= 0:
            return
        g.set_source_rgba(*rgb(col), a)
        g.set_line_width(6)
        n = int(math.floor(r_now))
        for r in range(n + 1):
            x, y = self._xy(r, vals[r])
            (g.move_to if r == 0 else g.line_to)(x, y)
        if n < N - 1:
            x, y = self._xy(r_now, np.interp(r_now, np.arange(N), vals))
            g.line_to(x, y)
        g.stroke()

    # ------------------------------------------------------------ HUD

    def _hud(self, ctx, t, ph, done):
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        if ph == "hook":
            lab = ("MET", "STATUS")
            left, right, rcol = "0", "?", AMBER
        elif ph == "demo":
            lab = ("MET", "STATUS")
            met = sum(1 for k in range(self.dpick + 1) if t >= self.reveal_time(k))
            left = str(met)
            if t >= self.T_YES:
                right, rcol = "YES", AMBER
            elif met >= R:
                right, rcol = "READY", WHITE
            else:
                right, rcol = "JUST LOOKING", SLATE
        elif ph == "lives":
            lab = ("LIVES", "FOUND THE BEST")
            left = f"{done:,}"
            right, rcol = (f"{100 * self.cum[0][done] / done:.1f}%" if done else "?"), CYAN
        else:
            if t < A0:
                r_now = int(round((N - 1) * smooth((t - C0) / (C1 - C0))))
                lab = ("SAY NO TO THE FIRST", "FOUND THE BEST")
                left, right, rcol = str(r_now), f"{100 * self.curve[r_now]:.1f}%", CYAN
            else:
                lab = ("SAY NO TO THE FIRST", "ENDED UP ALONE")
                left, right, rcol = str(R), f"{100 * self.alone_curve[R]:.1f}%", MAGENTA
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, lab[0], 70, 470, 0.14 * 24)
        w = tracked_width(ctx, lab[1], 0.14 * 24)
        show_tracked(ctx, lab[1], 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, left, 70, 545, big)
        if right[0].isdigit():
            font(ctx, "dbold", 76)
            ctx.set_source_rgb(*rgb(rcol))
            show_cells(ctx, right, 910 - cells_width(ctx, right, big), 545, big)
        else:
            font(ctx, "dbold", 56)
            ctx.set_source_rgb(*rgb(rcol))
            ctx.move_to(910 - ctx.text_extents(right).x_advance, 540)
            ctx.show_text(right)
        if ph == "lives":
            entries = [(f"■ THE BEST {100 * self.cum[0][done] / max(done, 1):.1f}%", CYAN),
                       (f"■ SOMEONE ELSE {100 * self.cum[1][done] / max(done, 1):.1f}%", DIM),
                       (f"■ ALONE {100 * self.cum[2][done] / max(done, 1):.1f}%", MAGENTA)]
            font(ctx, "bold", 21)
            ws = [tracked_width(ctx, s, 0.04 * 21) for s, _ in entries]
            gap = 24
            x = CX - (sum(ws) + gap * 2) / 2
            for (s, c), w in zip(entries, ws):
                ctx.set_source_rgba(*rgb(c), 0.9 if done else 0.0)
                show_tracked(ctx, s, x, GRID_C[1] - 228, 0.04 * 21)
                x += w + gap

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
                g = cv2.GaussianBlur(a_band, (0, 0), 14) * 0.8 * alpha
                img[band] += g[..., None] * np.array(rgb(CYAN), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
