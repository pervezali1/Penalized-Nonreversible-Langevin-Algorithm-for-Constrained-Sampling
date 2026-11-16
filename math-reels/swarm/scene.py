"""Frame renderer for the swarm reel: reel 04's 10,000 glowing particles on the same shaded landscape, first alone
(plain gradient descent), then talking (particle swarm optimisation), with faint lines to the best spot found so far
while the news spreads.  frame(f) -> uint8 1080x1920 RGB."""

import json
import math
import os
import sys

import cv2
import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pendulum"))
sys.path.insert(0, os.path.join(ROOT, "descent"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)
from raster import splat_trails  # noqa: E402  (pendulum numba trail splatter)
from landscape import grad  # noqa: E402
from swarmsim import (EXPO_A, EXPO_B, FPS, GLOBAL, LOOP, N_FRAMES, PSO_STEPS, RESET, RUN_A, RUN_B, F_MIN,  # noqa: E402
                      found, frame_step, loss)

CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
CX = 470
SQ_C = (470, 1120)
SQ_H = 450
TRAIL = 10
WEIGHT = 0.05
MAX_TEXT_W = 820


def to_px(P):
    return np.stack([SQ_C[0] + P[..., 0] * SQ_H, SQ_C[1] + P[..., 1] * SQ_H], -1)


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


class Scene:
    def __init__(self, data_path):
        d = np.load(data_path)
        with open(data_path.replace(".npz", ".json")) as fh:
            self.meta = json.load(fh)
        self.pos, self.P0 = d["pos"], d["P0"]
        self.found_a, self.found_b = d["found_a"], d["found_b"]
        self.gbest = d["gbest"]
        self.talk_idx = np.random.default_rng(66).choice(len(self.P0), 240, replace=False)
        self.pct_a = int(round(100 * self.meta["found_a"]))
        self.pct_b = int(round(100 * self.meta["found_b"]))
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_amber = np.array(srgb_to_linear(rgb(AMBER)), np.float32)
        self.bg = self._landscape()
        self.cues = self._cues()

    # ------------------------------------------------------------ setup

    def _cues(self):
        hook = [("10,000 AIs.", "black", 96, WHITE, 300, 1), ("This time, they talk to each other.", "dbold", 46, WHITE, 392, 0.85)]
        return [
            (0.0, 2.2, hook, 0, 6, "none"),
            (3.0, 4.8, [("First, alone.", "black", 76, WHITE, 312, 1), ("Each one rolls downhill.", "dbold", 52, WHITE, 396, 0.85)], 8, 6, "pop"),
            (5.4, 7.3, [("Most get stuck", "black", 76, WHITE, 312, 1), ("in a trap.", "black", 76, MAGENTA, 396, 1)], 8, 6, "pop"),
            (7.6, 9.6, [(f"Only {self.pct_a}%", "black", 110, MAGENTA, 320, 1), ("found the best answer.", "dbold", 52, WHITE, 400, 1)], 8, 6, "pop"),
            (10.5, 12.4, [("Same hill. Same starts.", "dbold", 54, WHITE, 300, 0.85), ("Now they talk.", "black", 84, CYAN, 392, 1)], 8, 6, "pop"),
            (13.2, 15.3, [("Everyone heads for", "black", 66, WHITE, 312, 1), ("the best spot anyone found.", "black", 58, CYAN, 392, 1)], 8, 6, "pop"),
            (16.6, 18.5, [(f"{self.pct_b}%", "black", 150, CYAN, 330, 1), ("found the best answer.", "dbold", 52, WHITE, 404, 1)], 6, 6, "chaos"),
            (18.7, 19.5, [("They never felt the slope.", "black", 62, WHITE, 312, 1), ("They talked.", "dbold", 56, AMBER, 396, 1)], 8, 6, "pop"),
            (19.55, 20.2, hook, 10, 0, "fade"),
        ]

    def _landscape(self):
        """Static shaded height map with contour lines; the deep well's floor is tinted cyan."""
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        P = np.stack([(xx - SQ_C[0]) / SQ_H, (yy - SQ_C[1]) / SQ_H], -1)
        h = loss(P)
        g = grad(P)
        inside = (np.abs(P[..., 0]) <= 1.0) & (np.abs(P[..., 1]) <= 1.0)
        # lighting from the upper left
        nx, ny = -g[..., 0], -g[..., 1]
        shade = 0.5 + 0.5 * (nx * -0.6 + ny * -0.8) / np.sqrt(1 + nx * nx + ny * ny)
        base = np.array(srgb_to_linear(rgb("#0B1026")), np.float32)
        dark = np.array(srgb_to_linear(rgb("#04050B")), np.float32)
        depth = np.clip((h - F_MIN) / 1.8, 0, 1)
        col = dark + (base - dark) * (0.35 + 0.65 * depth)[..., None]
        col *= (0.55 + 0.9 * shade)[..., None]
        # contours
        frac = (h / 0.07) % 1.0
        line = np.exp(-((np.minimum(frac, 1 - frac)) / 0.035) ** 2)
        col += (line * 0.045)[..., None]
        # the best answer glows faintly cyan
        well = np.exp(-((P[..., 0] - GLOBAL[0]) ** 2 + (P[..., 1] - GLOBAL[1]) ** 2) / (2 * 0.2 ** 2))
        col += (well * 0.05)[..., None] * self.c_cyan
        col[~inside] = dark * 0.6
        # soft frame edge
        edge = np.clip(1 - (np.maximum(np.abs(P[..., 0]), np.abs(P[..., 1])) - 1) / 0.08, 0, 1)
        col = col * (0.3 + 0.7 * edge[..., None]) + dark * 0.6 * (1 - edge[..., None]) * 0
        return col.astype(np.float32)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        P = self.pos[f]
        mode = self._mode(f)
        # particle colours
        if mode == "reveal_a" or mode == "reset":
            cols = np.where(self.found_a[:, None], self.c_cyan, self.c_mag)
        elif mode == "reveal_b" or mode == "loop":
            cols = np.where(self.found_b[:, None], self.c_cyan, self.c_mag)
        else:
            u = np.clip((loss(P) - F_MIN) / 1.6, 0, 1) ** 0.6
            cols = self.c_cyan * (1 - u[:, None]) + self.c_amber * u[:, None]
        if mode in ("reset", "loop"):            # fade outcome colours toward neutral while flying back
            a, b = (RESET if mode == "reset" else LOOP)
            k = smooth((f - a) / (b - a))
            cols = cols * (1 - k) + np.array([0.8, 0.8, 0.9], np.float32) * k
        cols = cols.astype(np.float32)

        acc = np.zeros((H, W, 3), np.float32)
        pts = to_px(P).astype(np.float64)
        idx = [(f - k) % N_FRAMES for k in range(TRAIL + 1)]
        trails = to_px(self.pos[idx]).astype(np.float32)
        if mode not in ("hook",):
            splat_trails(acc, trails, cols, WEIGHT, 1.0, 0.0, 0.0)
        splat_points(acc, pts, cols, WEIGHT * 2.5, 1.7)

        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, P, mode)
        return dither(img, f)

    def _mode(self, f):
        if f < RUN_A[0]:
            return "hook"
        if f < 456:
            return "run_a"
        if f < RESET[0]:
            return "reveal_a"
        if f < RESET[1]:
            return "reset"
        if f < 996:
            return "run_b"
        if f < LOOP[0]:
            return "reveal_b"
        if f < LOOP[1]:
            return "loop"
        return "hook"

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, P, mode):
        surf, ctx = layer()
        big = {str(k): 50 for k in range(10)}
        big.update({",": 22, "%": 60})
        if mode in ("hook", "run_a", "reveal_a", "reset"):
            step = frame_step(f, *RUN_A, expo=EXPO_A) if mode != "hook" else 0
            if mode == "reset":
                step = 0
            mom = False
        else:
            step = frame_step(f, *RUN_B, steps=PSO_STEPS, expo=EXPO_B) if mode != "loop" else 0
            mom = mode != "loop"
        if mode == "reveal_a":
            pct = self.pct_a
        elif mode == "reveal_b":
            pct = self.pct_b
        else:   # live share of the particles on screen in the deep well (228 of the random starts, 2%, already are)
            pct = int(round(100 * found(P).mean()))
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "STEP", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "FOUND THE BEST ANSWER", 0.14 * 24)
        show_tracked(ctx, "FOUND THE BEST ANSWER", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{int(step):,}", 70, 545, big)
        s = f"{pct}%"
        ctx.set_source_rgb(*(rgb(CYAN) if pct >= 60 else rgb(MAGENTA) if pct >= 25 else rgb(AMBER)))
        tw = cells_width(ctx, s, big)
        show_cells(ctx, s, 910 - tw, 545, big)
        # momentum pill
        label = "TALKING  ON" if mom else "TALKING  OFF"
        font(ctx, "bold", 26)
        lw = tracked_width(ctx, label, 0.08 * 26)
        x0, y0, bw, bh, r = CX - lw / 2 - 18, 586, lw + 36, 44, 22
        ctx.new_sub_path()
        ctx.arc(x0 + bw - r, y0 + r, r, -math.pi / 2, 0)
        ctx.arc(x0 + bw - r, y0 + bh - r, r, 0, math.pi / 2)
        ctx.arc(x0 + r, y0 + bh - r, r, math.pi / 2, math.pi)
        ctx.arc(x0 + r, y0 + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        c = rgb(CYAN) if mom else (1, 1, 1)
        ctx.set_source_rgba(*c, 0.14)
        ctx.fill()
        ctx.set_source_rgba(*c, 0.9 if mom else 0.6)
        show_tracked(ctx, label, CX - lw / 2, y0 + 31, 0.08 * 26)
        # talking: faint lines from a sample of particles to the best spot found so far, while the news spreads
        a_talk = 0.0
        if mode in ("run_b", "reveal_b"):
            k_talk = (f - RUN_B[0]) / FPS
            a_talk = max(0.0, min(1.0, k_talk / 0.25)) * (1 - smooth(min(1.0, max(0.0, (k_talk - 2.6) / 1.4))))
            if a_talk > 0:
                g = to_px(self.gbest[f])
                pts = to_px(P[self.talk_idx])
                ctx.set_line_width(1.0)
                ctx.set_source_rgba(*rgb(AMBER), 0.22 * a_talk)
                for x, y in pts:
                    ctx.move_to(x, y)
                    ctx.line_to(*g)
                ctx.stroke()
                ctx.set_line_width(2.5)
                ctx.set_source_rgba(*rgb(AMBER), 0.95 * a_talk)
                ctx.arc(*g, 20 + 4 * math.sin(2 * math.pi * 2.0 * k_talk), 0, 2 * math.pi)
                ctx.stroke()
                font(ctx, "bold", 20)
                s = "BEST SPOT SO FAR"
                show_tracked(ctx, s, g[0] - tracked_width(ctx, s, 0.12 * 20) / 2, g[1] - 34, 0.12 * 20)
        # the best-answer marker
        gx, gy = to_px(np.array(GLOBAL))
        ctx.set_line_width(1.5)
        ctx.set_source_rgba(*rgb(CYAN), 0.5)
        ctx.arc(gx, gy, 14, 0, 2 * math.pi)
        ctx.stroke()
        font(ctx, "bold", 20)
        ctx.set_source_rgba(*rgb(CYAN), 0.7 * (1 - a_talk))
        s = "BEST ANSWER"
        show_tracked(ctx, s, gx - tracked_width(ctx, s, 0.12 * 20) / 2, gy + 44, 0.12 * 20)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

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
