"""The medical test paradox: 10,000 glowing people, 10 of them sick; a 99%-accurate test sweeps the crowd; the
positives fly together into one block: one row of sick people, the rest false alarms; then the positives are tested
again and the false alarms go dark.  frame(f) -> uint8 1080x1920 RGB."""

import math
import os
import sys

import cv2
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
import bsim  # noqa: E402

FPS = 60
N_FRAMES = 1320                                  # 22 s
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
PEOPLE = "#AEB6D8"
CX = 470
MAX_TEXT_W = 820
N = bsim.N
# the crowd: 100 x 100 at 8 px
GX0, GY0, GP = CX - 400 + 4, 660, 8.0
# the block of positives
BCOLS, BP, BC = 10, 44.0, (CX, 1070)
# times (s)
T_SICK = 2.6                                     # the 10 sick light up
SCAN0, SCAN1 = 5.3, 7.9                          # the test sweeps the crowd, row by row
GATHER0, GATHER1 = 9.0, 10.4                     # positives fly into the block, everyone else dims
RETEST0, RETEST1 = 16.6, 18.0                    # second test: false alarms go dark one by one
LOOP0, LOOP1 = 21.2, 22.0                        # everyone flies home


@njit(cache=True)
def splat_points(acc, pts, cols, w, sigma):
    rad = int(sigma * 2.5) + 1
    inv = -0.5 / (sigma * sigma)
    for i in range(pts.shape[0]):
        cx, cy = pts[i, 0], pts[i, 1]
        x0, y0 = int(cx), int(cy)
        for yy in range(y0 - rad, y0 + rad + 2):
            if yy < 0 or yy >= acc.shape[0]:
                continue
            for xx in range(x0 - rad, x0 + rad + 2):
                if xx < 0 or xx >= acc.shape[1]:
                    continue
                v = w * math.exp(((xx - cx) ** 2 + (yy - cy) ** 2) * inv)
                acc[yy, xx, 0] += cols[i, 0] * v
                acc[yy, xx, 1] += cols[i, 1] * v
                acc[yy, xx, 2] += cols[i, 2] * v


def lin(h):
    return np.array(srgb_to_linear(rgb(h)), np.float32)


class Scene:
    def __init__(self):
        self.sick, self.pos1, self.pos2 = bsim.people()
        i = np.arange(N)
        self.home = np.stack([GX0 + (i % 100) * GP, GY0 + (i // 100) * GP], 1).astype(np.float64)
        self.row = i // 100
        self.phase = 0.13 * (i // 100 + i % 100)
        # block slots: sick positives first (row 0), then the false alarms, in crowd order
        pos_ids = np.nonzero(self.pos1)[0]
        order = np.concatenate([pos_ids[self.sick[pos_ids]], pos_ids[~self.sick[pos_ids]]])
        self.block = np.zeros((N, 2))
        n_rows = math.ceil(len(order) / BCOLS)
        for k, p in enumerate(order):
            r, c = divmod(k, BCOLS)
            self.block[p] = (BC[0] + (c - (BCOLS - 1) / 2) * BP, BC[1] + (r - (n_rows - 1) / 2) * BP)
        self.n_pos = int(self.pos1.sum())
        self.n_tp = int((self.pos1 & self.sick).sum())
        self.n_fp = self.n_pos - self.n_tp
        both = self.pos1 & self.pos2
        self.n_pos2 = int(both.sum())
        self.n_tp2 = int((both & self.sick).sum())
        self.pct1 = 100 * self.n_tp / self.n_pos
        self.pct2 = 100 * self.n_tp2 / self.n_pos2
        # the retest reveal order: the false alarms that test negative go dark one by one
        rng = np.random.default_rng(7)
        fa = np.nonzero(self.pos1 & ~self.sick & ~self.pos2)[0]
        self.dark_t = np.full(N, np.inf)
        self.dark_t[rng.permutation(fa)] = RETEST0 + (RETEST1 - RETEST0) * np.arange(len(fa)) / max(1, len(fa) - 1)
        self.c_people, self.c_sick, self.c_amber = lin(PEOPLE), lin(MAGENTA), lin(AMBER)
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        hook = [("This test is 99% accurate.", "black", 70, WHITE, 300, 1),
                ("You test positive. Are you sick?", "dbold", 48, WHITE, 392, 0.9)]
        return [
            (0.0, 2.4, hook, 0, 6, "none"),
            (T_SICK, 4.9, [("1 in 1,000 people has it.", "black", 66, WHITE, 312, 1),
                           (f"Here: {int(self.sick.sum())} out of {N:,}.", "dbold", 52, MAGENTA, 400, 1)], 8, 6, "pop"),
            (5.2, 7.9, [(f"Test all {N:,}.", "black", 76, WHITE, 312, 1), ("It's right 99% of the time.", "dbold", 50, WHITE, 400, 0.9)], 8, 6, "pop"),
            (8.1, 10.4, [(f"{self.n_pos} positive tests.", "black", 76, AMBER, 312, 1), ("Now look closer.", "dbold", 50, WHITE, 400, 0.9)], 8, 6, "pop"),
            (10.8, 13.4, [(f"Only {self.pct1:.0f}% are sick.", "black", 84, MAGENTA, 316, 1),
                          (f"The other {self.n_fp}: false alarms.", "dbold", 50, WHITE, 404, 0.9)], 6, 6, "chaos"),
            (13.6, 16.3, [(f"1% of {N - int(self.sick.sum()):,} healthy people", "black", 56, AMBER, 312, 1),
                          (f"outnumbers all {int(self.sick.sum())} sick ones.", "black", 56, WHITE, 396, 1)], 8, 6, "pop"),
            (16.5, 18.3, [("Test them again.", "black", 80, WHITE, 340, 1)], 8, 6, "pop"),
            (18.5, 21.1, [(f"Positive twice: {self.pct2:.0f}% sick.", "black", 70, CYAN, 312, 1),
                          ("One positive test isn't a diagnosis.", "dbold", 46, WHITE, 400, 0.95)], 6, 6, "chaos"),
            (LOOP0, LOOP1 + 0.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1050) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ state at time t

    def scan_row(self, t):
        """How many rows the first test has swept (fractional)."""
        return 100 * smooth((t - SCAN0) / (SCAN1 - SCAN0)) if t >= SCAN0 else 0.0

    def tested(self, t):
        if t >= LOOP0:
            return np.zeros(N, bool)
        return self.row < self.scan_row(t)

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        acc = np.zeros((H, W, 3), np.float32)
        tested = self.tested(t)
        flag = tested & self.pos1                             # glowing amber: tested positive
        # where everyone is: home, or (positives) in the block
        g = smooth((t - GATHER0) / (GATHER1 - GATHER0)) if t >= GATHER0 else 0.0
        if t >= LOOP0:
            g = 1 - smooth((t - LOOP0) / (LOOP1 - LOOP0 - 0.15))
        P = self.home.copy()
        if g > 0:
            m = self.pos1
            # each one flies on a slight arc, staggered by crowd row
            k = np.clip(g * 1.25 - 0.25 * self.row[m] / 99, 0, 1)
            k = k * k * (3 - 2 * k)
            a, b = self.home[m], self.block[m]
            P[m] = a + (b - a) * k[:, None]
            P[m, 1] -= 60 * np.sin(np.pi * k)
        # brightness of the crowd: dims once the positives leave
        dim = 1 - 0.75 * g
        # the crowd (not positive, or not yet tested)
        sick_on = smooth((t - T_SICK) / 0.4) if T_SICK <= t < LOOP0 else (1 - smooth((t - LOOP0) / 0.5) if t >= LOOP0 else 0.0)
        rest = ~flag & ~(self.sick & (sick_on > 0))
        shimmer = 0.78 + 0.22 * np.sin(2 * np.pi * 11 * t / (N_FRAMES / FPS) - self.phase[rest])   # 11 waves per loop
        cols = (self.c_people[None, :] * (0.75 * dim) * shimmer[:, None]).astype(np.float32)
        splat_points(acc, P[rest], cols, 1.15, 0.85)
        # the sick (lit from T_SICK), drawn bigger and pulsing
        if sick_on > 0:
            s = self.sick & ~flag
            pulse = 0.75 + 0.25 * math.sin(2 * math.pi * 1.5 * t)
            cols = np.repeat(self.c_sick[None, :] * sick_on * pulse * 1.4, int(s.sum()), 0)
            splat_points(acc, P[s], cols, 1.0, 3.2)
        # the positives
        if flag.any():
            size = 1.6 + 3.6 * g
            dark = t >= self.dark_t
            for group, col in ((flag & self.sick, self.c_sick), (flag & ~self.sick, self.c_amber)):
                lit = group & ~dark
                if lit.any():
                    cols = np.repeat(col[None, :] * 1.3, int(lit.sum()), 0)
                    splat_points(acc, P[lit], cols, 0.9, size)
                gone = group & dark
                if gone.any():
                    cols = np.repeat(self.c_people[None, :] * 0.25, int(gone.sum()), 0)
                    splat_points(acc, P[gone], cols, 0.6, size * 0.7)
        # the scan line
        if SCAN0 <= t < SCAN1 + 0.3:
            y = GY0 + self.scan_row(t) * GP - GP / 2
            a = 1 - smooth((t - SCAN1) / 0.3) if t >= SCAN1 else 1.0
            band = slice(max(0, int(y) - 2), int(y) + 3)
            acc[band, int(GX0 - 10):int(GX0 + 100 * GP + 2)] += lin(CYAN) * 0.9 * a
        np.clip(acc, 0, 4, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.35 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin_img = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin_img).astype(np.float32)
        surf, ctx = layer()
        self._labels(ctx, t, g)
        self._hud(ctx, t, tested)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)
        return dither(img, f)

    # ------------------------------------------------------------ labels and HUD

    def _labels(self, ctx, t, g):
        # rings around the 10 sick when they first light up
        if T_SICK <= t < 5.0:
            a = smooth((t - T_SICK) / 0.3) * (1 - smooth((t - 4.6) / 0.4))
            ctx.set_source_rgba(*rgb(MAGENTA), 0.9 * a)
            ctx.set_line_width(2.5)
            r = 16 + 6 * smooth((t - T_SICK) / 0.5)
            for p in self.home[self.sick]:
                ctx.arc(p[0], p[1], r, 0, 2 * math.pi)
                ctx.stroke()
        # block legend once the positives have gathered
        if GATHER1 - 0.2 <= t < LOOP0:
            a = smooth((t - GATHER1 + 0.2) / 0.4) * (1 - smooth((t - LOOP0) / 0.3) if t >= LOOP0 else 1.0)
            top = self.block[self.pos1, 1].min()
            bot = self.block[self.pos1, 1].max()
            font(ctx, "black", 30)
            s1 = f"■ SICK: {self.n_tp}"
            s2 = f"■ FALSE ALARMS: {self.n_fp}"
            if t >= RETEST1:
                s2 = f"■ STILL POSITIVE: {self.n_pos2 - self.n_tp2}"
            w1, w2 = ctx.text_extents(s1).x_advance, ctx.text_extents(s2).x_advance
            gap = 40
            x = CX - (w1 + w2 + gap) / 2
            ctx.set_source_rgba(*rgb(MAGENTA), a)
            ctx.move_to(x, top - 40)
            ctx.show_text(s1)
            ctx.set_source_rgba(*rgb(AMBER), a)
            ctx.move_to(x + w1 + gap, top - 40)
            ctx.show_text(s2)
            font(ctx, "bold", 22)
            s = "EVERYONE WHO TESTED POSITIVE"
            w = tracked_width(ctx, s, 0.1 * 22)
            ctx.set_source_rgba(1, 1, 1, 0.6 * a)
            show_tracked(ctx, s, CX - w / 2, bot + 56, 0.1 * 22)
            if t >= 10.8:
                k = smooth((t - 10.8) / 0.4) * a
                times = 2 if t >= RETEST1 else 1
                s = f"THE EXACT MATHS: {100 * bsim.exact(times):.1f}% SICK"
                font(ctx, "black", 26)
                w = tracked_width(ctx, s, 0.06 * 26)
                ctx.set_source_rgba(*rgb(CYAN if times == 2 else MAGENTA), 0.9 * k)
                show_tracked(ctx, s, CX - w / 2, bot + 96, 0.06 * 26)

    def _hud(self, ctx, t, tested):
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        if t < SCAN0 or t >= LOOP0:
            lab = ("PEOPLE", "SICK")
            left = f"{N:,}"
            right, rcol = (str(int(self.sick.sum())) if T_SICK <= t < LOOP0 else "?"), MAGENTA
        elif t < GATHER1:
            lab = ("TESTED", "POSITIVE")
            left = f"{int(tested.sum()):,}"
            right, rcol = str(int((tested & self.pos1).sum())), AMBER
        elif t < RETEST0:
            lab = ("POSITIVE", "ACTUALLY SICK")
            left = str(self.n_pos)
            right, rcol = f"{self.pct1:.1f}%", MAGENTA
        else:
            still = int((self.pos1 & ~(t >= self.dark_t)).sum())
            lab = ("STILL POSITIVE", "ACTUALLY SICK")
            left = str(still)
            right, rcol = f"{100 * self.n_tp / still:.1f}%", (CYAN if t >= RETEST1 else MAGENTA)
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, lab[0], 70, 470, 0.14 * 24)
        w = tracked_width(ctx, lab[1], 0.14 * 24)
        show_tracked(ctx, lab[1], 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, left, 70, 545, big)
        if right[0].isdigit():
            ctx.set_source_rgb(*rgb(rcol))
            show_cells(ctx, right, 910 - cells_width(ctx, right, big), 545, big)
        else:
            font(ctx, "dbold", 56)
            ctx.set_source_rgb(*rgb(AMBER))
            ctx.move_to(910 - ctx.text_extents(right).x_advance, 540)
            ctx.show_text(right)

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
