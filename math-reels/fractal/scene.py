"""The chaos game: random dice draw a Sierpinski triangle, then four slightly different random
rules grow a Barnsley fern. Points accumulate per frame.  frame(f) -> uint8 1080x1920 RGB."""

import math
import os
import sys

import cv2
import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)

FPS = 60
N_FRAMES = 1200
SEED = 11
CYAN, MAGENTA, AMBER, WHITE, GREEN = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF", "#3DFF8F"
CX = 470
BOX_C = (470, 1130)
BOX_H = 400
N_TRI, N_FERN = 50_000, 200_000
TRI_SLOW_T0, TRI_SLOW_DT, TRI_SLOW_N = 0.6, 0.25, 12
TRI_FAST_T0, TRI_FAST_T1 = 3.6, 6.8
WIPE = (7.3, 7.7)
FERN_SLOW_T0, FERN_SLOW_DT, FERN_SLOW_N = 7.8, 0.22, 8
FERN_FAST_T0, FERN_FAST_T1 = 9.6, 15.6
LOOP_FADE = (19.4, 20.0)
MAX_TEXT_W = 820

# Barnsley fern: (a, b, c, d, e, f, probability)
FERN_MAPS = [(0.00, 0.00, 0.00, 0.16, 0.0, 0.00, 0.01),
             (0.85, 0.04, -0.04, 0.85, 0.0, 1.60, 0.85),
             (0.20, -0.26, 0.23, 0.22, 0.0, 1.60, 0.07),
             (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07)]


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


def chaos_triangle(rng, n):
    corners = np.array([[0.0, 1.0], [-0.95, -0.7], [0.95, -0.7]])
    p = np.array([0.1, 0.1])
    pts = np.empty((n, 2))
    which = rng.integers(0, 3, n)
    dice = np.empty(n, np.int64)
    for i in range(n):
        k = which[i]
        dice[i] = 2 * k + 1 + rng.integers(0, 2)      # die 1-2 -> A, 3-4 -> B, 5-6 -> C
        p = (p + corners[k]) / 2
        pts[i] = p
    return pts, which, dice, corners


def chaos_fern(rng, n):
    probs = np.array([m[6] for m in FERN_MAPS])
    which = rng.choice(4, size=n, p=probs)
    p = np.array([0.0, 0.0])
    pts = np.empty((n, 2))
    for i in range(n):
        a, b, c, d, e, f, _ = FERN_MAPS[which[i]]
        p = np.array([a * p[0] + b * p[1] + e, c * p[0] + d * p[1] + f])
        pts[i] = p
    return pts, which


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.tri, self.tri_which, self.tri_dice, self.corners = chaos_triangle(rng, N_TRI)
        self.fern, self.fern_which = chaos_fern(rng, N_FERN)
        self.tri_px = np.stack([BOX_C[0] + self.tri[:, 0] * BOX_H, BOX_C[1] - self.tri[:, 1] * BOX_H], 1)
        fx = (self.fern[:, 0] + 0.0) / 5.2 * 2       # x in about [-2.2, 2.7] -> [-0.85, 1.05]
        fy = self.fern[:, 1] / 10.0 * 2 - 1          # y in [0, 10] -> [-1, 1]
        self.fern_px = np.stack([BOX_C[0] + (fx - 0.1) * BOX_H, BOX_C[1] - fy * BOX_H * 0.98], 1)
        self.tri_cols = np.array([srgb_to_linear(rgb(c)) for c in (CYAN, MAGENTA, AMBER)], np.float32)[self.tri_which]
        fern_palette = np.array([srgb_to_linear(rgb(c)) for c in (AMBER, GREEN, CYAN, "#9DFF6A")], np.float32)
        self.fern_cols = fern_palette[self.fern_which]
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        hook = [("Roll a die.", "black", 92, WHITE, 300, 1), ("Move halfway to a corner.", "dbold", 50, WHITE, 392, 0.9)]
        return [
            (0.0, 2.2, hook, 0, 6, "none"),
            (2.5, 4.4, [("Repeat 50,000 times.", "black", 76, WHITE, 350, 1)], 8, 6, "pop"),
            (5.2, 7.2, [("A perfect triangle.", "black", 76, CYAN, 312, 1), ("From pure randomness.", "dbold", 50, WHITE, 396, 0.9)], 8, 6, "pop"),
            (7.8, 9.6, [("Now change the rules slightly.", "black", 60, AMBER, 350, 1)], 8, 6, "pop"),
            (11.0, 13.4, [("Still random. Still dice.", "black", 66, WHITE, 312, 1), ("200,000 points.", "dbold", 50, WHITE, 396, 0.85)], 8, 6, "pop"),
            (14.6, 17.3, [("That's a fern.", "black", 110, GREEN, 330, 1), ("4 rules. 0 drawings.", "dbold", 54, WHITE, 404, 1)], 6, 6, "chaos"),
            (17.6, 19.3, [("That's what a fractal is:", "black", 58, WHITE, 312, 1), ("simple rules, repeated forever.", "dbold", 46, AMBER, 396, 1)], 8, 6, "pop"),
            (19.45, 20.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - BOX_C[0]) ** 2 + (yy - BOX_C[1]) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#0C1024")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#04050A")), np.float32)
        u = np.clip(r / 1000, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ point counts

    def tri_count(self, t):
        if t < TRI_SLOW_T0:
            return 0
        if t < TRI_FAST_T0:
            return min(TRI_SLOW_N, int((t - TRI_SLOW_T0) / TRI_SLOW_DT) + 1)
        if t >= TRI_FAST_T1:
            return N_TRI
        u = (t - TRI_FAST_T0) / (TRI_FAST_T1 - TRI_FAST_T0)
        return min(N_TRI, TRI_SLOW_N + int((N_TRI - TRI_SLOW_N) * u ** 2.5))

    def fern_count(self, t):
        if t < FERN_SLOW_T0:
            return 0
        if t < FERN_FAST_T0:
            return min(FERN_SLOW_N, int((t - FERN_SLOW_T0) / FERN_SLOW_DT) + 1)
        if t >= FERN_FAST_T1:
            return N_FERN
        u = (t - FERN_FAST_T0) / (FERN_FAST_T1 - FERN_FAST_T0)
        return min(N_FERN, FERN_SLOW_N + int((N_FERN - FERN_SLOW_N) * u ** 2.2))

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        wipe = smooth((t - WIPE[0]) / (WIPE[1] - WIPE[0])) if WIPE[0] <= t < WIPE[1] else (1.0 if t >= WIPE[1] else 0.0)
        acc = np.zeros((H, W, 3), np.float32)
        n_tri = self.tri_count(t)
        n_fern = self.fern_count(t)
        phase = "fern" if t >= WIPE[0] else "tri"
        if phase == "tri" or wipe < 1:
            k = n_tri if phase == "tri" else N_TRI
            if k:
                w = 0.09 if k < 2000 else 0.09 * (2000 / k) ** 0.35
                w *= (1 - wipe)
                splat_points(acc, self.tri_px[:k], self.tri_cols[:k], w, 1.6 if k < 2000 else 1.1)
        if n_fern:
            w = 0.09 if n_fern < 2000 else 0.09 * (2000 / n_fern) ** 0.33
            splat_points(acc, self.fern_px[:n_fern], self.fern_cols[:n_fern], w * fade, 1.5 if n_fern < 2000 else 1.0)
        # the walking point and its jump during the slow phases
        self._walker(acc, t, phase, n_tri, n_fern)
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, phase, n_tri, n_fern, fade)
        return dither(img, f)

    def _walker(self, acc, t, phase, n_tri, n_fern):
        if phase == "tri" and 0 < n_tri <= TRI_SLOW_N:
            age = (t - TRI_SLOW_T0) % TRI_SLOW_DT
            a = 1 - age / TRI_SLOW_DT
            p1 = self.tri_px[n_tri - 1]
            p0 = self.tri_px[n_tri - 2] if n_tri >= 2 else np.array([BOX_C[0] + 0.1 * BOX_H, BOX_C[1] - 0.1 * BOX_H])
            self._segment(acc, p0, p1, a)
            splat_points(acc, p1[None], np.array([[1.0, 1.0, 1.0]], np.float32), 1.5, 3.0)
        if phase == "fern" and 0 < n_fern <= FERN_SLOW_N:
            age = (t - FERN_SLOW_T0) % FERN_SLOW_DT
            a = 1 - age / FERN_SLOW_DT
            p1 = self.fern_px[n_fern - 1]
            p0 = self.fern_px[n_fern - 2] if n_fern >= 2 else np.array([BOX_C[0] - 0.1 * BOX_H, BOX_C[1] + BOX_H * 0.98])
            self._segment(acc, p0, p1, a)
            splat_points(acc, p1[None], np.array([[1.0, 1.0, 1.0]], np.float32), 1.5, 3.0)

    def _segment(self, acc, p0, p1, a):
        n = int(np.linalg.norm(p1 - p0) / 3) + 2
        pts = p0[None] + (p1 - p0)[None] * np.linspace(0, 1, n)[:, None]
        splat_points(acc, np.ascontiguousarray(pts), np.full((n, 3), 0.9, np.float32), 0.08 * a, 1.4)

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, phase, n_tri, n_fern, fade):
        surf, ctx = layer()
        if fade < 0.5:                       # loop tail: HUD matches frame 0 (dice rules, corners, no points)
            phase, n_tri, n_fern = "tri", 0, 0
        big = {str(c): 50 for c in range(10)}
        big[","] = 22
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "POINTS", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "RULES", 0.14 * 24)
        show_tracked(ctx, "RULES", 910 - w, 470, 0.14 * 24)
        n = n_tri if phase == "tri" else n_fern
        if fade < 0.5:
            n = 0
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{n:,}", 70, 545, big)
        rules = "3" if (phase == "tri" or fade < 0.5) else "4"
        ctx.set_source_rgb(*(rgb(CYAN) if rules == "3" else rgb(GREEN)))
        tw = cells_width(ctx, rules, big)
        show_cells(ctx, rules, 910 - tw, 545, big)
        # rule pills with the last roll highlighted
        if phase == "tri":
            labels = [("1-2  →  A", CYAN), ("3-4  →  B", MAGENTA), ("5-6  →  C", AMBER)]
            last = int(self.tri_which[max(0, n_tri - 1)]) if n_tri else -1
        else:
            labels = [("1%  stem", AMBER), ("85%  main", GREEN), ("7%  left", CYAN), ("7%  right", "#9DFF6A")]
            last = int(self.fern_which[max(0, n_fern - 1)]) if n_fern else -1
        font(ctx, "bold", 22)
        total = sum(tracked_width(ctx, s, 0.05 * 22) + 36 for s, _ in labels) + 12 * (len(labels) - 1)
        x = CX - total / 2
        for i, (s, col) in enumerate(labels):
            lw = tracked_width(ctx, s, 0.05 * 22)
            on = (i == last) and fade > 0.5
            ctx.set_source_rgba(*rgb(col), 0.35 if on else 0.10)
            ctx.rectangle(x, 586, lw + 36, 40)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), 1.0 if on else 0.6)
            show_tracked(ctx, s, x + 18, 614, 0.05 * 22)
            x += lw + 36 + 12
        # a die showing the current roll during the triangle phase
        if phase == "tri" and n_tri:
            face = int(self.tri_dice[n_tri - 1])
            self._die(ctx, 880 - 60, 660, 60, face, 1.0 if n_tri <= TRI_SLOW_N else 0.7)
        # corners
        if phase == "tri" or (phase == "fern" and t < WIPE[1]):
            a = 1.0 if phase == "tri" else 1 - smooth((t - WIPE[0]) / (WIPE[1] - WIPE[0]))
            for (cx, cy), name, col in zip(self.corners, "ABC", (CYAN, MAGENTA, AMBER)):
                px, py = BOX_C[0] + cx * BOX_H, BOX_C[1] - cy * BOX_H
                ctx.set_source_rgba(*rgb(col), 0.9 * a)
                ctx.arc(px, py, 9, 0, 2 * math.pi)
                ctx.fill()
                font(ctx, "black", 30)
                ctx.set_source_rgba(*rgb(col), a)
                dx = -40 if name == "B" else (18 if name == "C" else -10)
                dy = -22 if name == "A" else 42
                ctx.move_to(px + dx, py + dy)
                ctx.show_text(name)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

    def _die(self, ctx, x, y, s, face, alpha):
        ctx.save()
        ctx.translate(x, y)
        r = s * 0.18
        ctx.new_sub_path()
        ctx.arc(s - r, r, r, -math.pi / 2, 0)
        ctx.arc(s - r, s - r, r, 0, math.pi / 2)
        ctx.arc(r, s - r, r, math.pi / 2, math.pi)
        ctx.arc(r, r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        ctx.set_source_rgba(*rgb("#F2EFE6"), 0.92 * alpha)
        ctx.fill()
        pos = {1: [(0.5, 0.5)], 2: [(0.25, 0.25), (0.75, 0.75)], 3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
               4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
               5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
               6: [(0.25, 0.22), (0.75, 0.22), (0.25, 0.5), (0.75, 0.5), (0.25, 0.78), (0.75, 0.78)]}[face]
        ctx.set_source_rgba(*rgb("#141828"), alpha)
        for (px, py) in pos:
            ctx.arc(px * s, py * s, s * 0.08, 0, 2 * math.pi)
            ctx.fill()
        ctx.restore()

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
                img[band] += g[..., None] * np.array(rgb(GREEN), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
