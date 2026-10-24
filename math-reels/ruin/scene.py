"""Gambler's ruin: 1,000 bankrolls betting $1 at a time on roulette red/black (p = 18/38), drawn
as glowing walks that sink into the floor; then the same 1,000 betting everything once.
frame(f) -> uint8 1080x1920 RGB."""

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
N_FRAMES = 1320                     # 22 s
SEED = 12
N, START, TARGET = 1000, 100, 200
P_WIN = 18 / 38
MAX_BETS = 7000
DS = 7                              # draw every 7th bet (1,000 samples per walk)
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
CX = 470
CHART = (90, 720, 880, 1440)        # x0, y0, x1, y1 of the chart area
SIM_T0, SIM_T1 = 5.0, 12.0
ALLIN_T0, ALLIN_HIT = 13.2, 14.8
LOOP_FADE = (21.4, 22.0)
MAX_TEXT_W = 820


@njit(cache=True)
def splat_walks(acc, paths, alive_until, n_show, cols_alive, col_broke, col_dbl, x0, y0, xw, yh, w):
    """paths: (N, S) bankroll samples; alive_until: sample index at which each walk ended (or S)."""
    n_walks, S = paths.shape
    for i in range(n_walks):
        end = min(n_show, alive_until[i])
        for s in range(1, end + 1):
            if s >= S:
                break
            xa = x0 + xw * (s - 1) / (S - 1)
            xb = x0 + xw * s / (S - 1)
            ya = y0 + yh * (1 - paths[i, s - 1] / 200.0)
            yb = y0 + yh * (1 - paths[i, s] / 200.0)
            dx = xb - xa
            dy = yb - ya
            length = math.sqrt(dx * dx + dy * dy)
            n = int(length / 0.8) + 1
            for k in range(n):
                u = (k + 0.5) / n
                x = xa + dx * u
                y = ya + dy * u
                xi = int(x)
                yi = int(y)
                if xi < 0 or yi < 0 or xi >= acc.shape[1] - 1 or yi >= acc.shape[0] - 1:
                    continue
                fx = x - xi
                fy = y - yi
                for c in range(3):
                    v = cols_alive[i, c] * w * 0.8
                    acc[yi, xi, c] += v * (1 - fx) * (1 - fy)
                    acc[yi, xi + 1, c] += v * fx * (1 - fy)
                    acc[yi + 1, xi, c] += v * (1 - fx) * fy
                    acc[yi + 1, xi + 1, c] += v * fx * fy
        # end marker
        if alive_until[i] <= n_show and alive_until[i] < S:
            s = alive_until[i]
            x = x0 + xw * s / (S - 1)
            y = y0 + yh * (1 - paths[i, s] / 200.0)
            col = col_broke if paths[i, s] <= 0 else col_dbl
            for yy in range(int(y) - 4, int(y) + 6):
                for xx in range(int(x) - 4, int(x) + 6):
                    if 0 <= xx < acc.shape[1] and 0 <= yy < acc.shape[0]:
                        d2 = (xx - x) ** 2 + (yy - y) ** 2
                        g = math.exp(-d2 / 4.0) * 0.9
                        for c in range(3):
                            acc[yy, xx, c] += col[c] * g


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        steps = np.where(rng.random((N, MAX_BETS)) < P_WIN, 1, -1)
        bank = START + np.cumsum(steps, axis=1)
        bank = np.concatenate([np.full((N, 1), START), bank], axis=1)
        hit0 = np.where((bank <= 0).any(1), np.argmax(bank <= 0, axis=1), MAX_BETS + 1)
        hit2 = np.where((bank >= TARGET).any(1), np.argmax(bank >= TARGET, axis=1), MAX_BETS + 1)
        self.end_bet = np.minimum(hit0, hit2)
        self.broke = hit0 < hit2
        self.doubled = hit2 < hit0
        for i in range(N):                       # freeze at the boundary after the end
            e = self.end_bet[i]
            if e <= MAX_BETS:
                bank[i, e:] = 0 if self.broke[i] else TARGET
        self.paths = np.ascontiguousarray(bank[:, ::DS].astype(np.float64))
        self.alive_until = np.minimum(np.ceil(self.end_bet / DS).astype(np.int64), self.paths.shape[1] - 1)
        self.n_broke_final = int(self.broke.sum())
        self.n_dbl_final = int(self.doubled.sum())
        self.p_slow = (1 - ((1 - P_WIN) / P_WIN) ** START) / (1 - ((1 - P_WIN) / P_WIN) ** TARGET)
        self.allin_win = rng.random(N) < P_WIN
        self.n_allin = int(self.allin_win.sum())
        self.cols_alive = np.tile(np.array(srgb_to_linear(rgb(AMBER)), np.float32) * 0.9, (N, 1)).astype(np.float32)
        self.col_broke = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.col_dbl = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        one_in = int(round((1 / self.p_slow) / 50) * 50)
        hook = [("You have $100. Goal: $200.", "black", 66, WHITE, 300, 1), ("Roulette, red or black. House edge 5.3%.", "dbold", 42, WHITE, 392, 0.9)]
        return [
            (0.0, 2.2, hook, 0, 6, "none"),
            (2.4, 4.8, [("A: $1 a bet, slow and careful.", "black", 54, AMBER, 304, 1), ("B: all in, once. Which doubles more often?", "dbold", 42, WHITE, 396, 0.9)], 8, 6, "pop"),
            (5.2, 7.2, [("Strategy A. 1,000 gamblers.", "black", 60, AMBER, 312, 1), ("$1 a bet. Watch the river.", "dbold", 46, WHITE, 396, 0.9)], 8, 6, "pop"),
            (8.0, 10.2, [("The edge drags every walk down.", "black", 58, WHITE, 350, 1)], 8, 6, "pop"),
            (11.0, 12.9, [(f"Doubled: {self.n_dbl_final} of 1,000.", "black", 72, CYAN, 312, 1), (f"Broke: {self.n_broke_final}.", "black", 72, MAGENTA, 404, 1)], 6, 6, "chaos"),
            (ALLIN_T0, 14.7, [("Strategy B. All in. Once.", "black", 66, CYAN, 350, 1)], 8, 6, "pop"),
            (15.0, 16.8, [(f"Doubled: {self.n_allin} of 1,000.", "black", 72, CYAN, 312, 1), (f"Broke: {N - self.n_allin}.", "black", 72, MAGENTA, 404, 1)], 6, 6, "chaos"),
            (17.0, 19.2, [(f"Slow and careful: 1 in {one_in:,}.", "black", 54, AMBER, 312, 1), ("All in: 47%.", "black", 72, CYAN, 404, 1)], 8, 6, "pop"),
            (19.4, 21.3, [("Every bet pays the house its edge.", "black", 52, WHITE, 312, 1), ("Gambler's ruin.", "black", 80, MAGENTA, 404, 1)], 8, 6, "pop"),
            (21.5, 22.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1000) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#0E1226")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ schedule

    def bets_shown(self, t):
        if t < SIM_T0:
            return 0
        if t >= SIM_T1:
            return MAX_BETS
        u = (t - SIM_T0) / (SIM_T1 - SIM_T0)
        return int(MAX_BETS * u ** 1.4)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        fade = 1 - smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 1.0
        x0, y0, x1, y1 = CHART
        acc = np.zeros((H, W, 3), np.float32)
        bets = self.bets_shown(t)
        mode = "allin" if t >= ALLIN_T0 and fade > 0.5 else "slow"
        if mode == "slow" and bets > 0 and fade > 0.5:
            n_show = min(self.paths.shape[1] - 1, bets // DS)
            splat_walks(acc, self.paths, self.alive_until, n_show, self.cols_alive, self.col_broke, self.col_dbl,
                        float(x0), float(y0), float(x1 - x0), float(y1 - y0), 0.06)
        elif mode == "allin":
            self._allin(acc, t)
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.3 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, bets, mode, fade)
        return dither(img, f)

    def _allin(self, acc, t):
        """1,000 bankrolls at $100; at ALLIN_HIT each jumps to $200 (cyan) or $0 (magenta) in one bet."""
        x0, y0, x1, y1 = CHART
        u = 0.0 if t < ALLIN_HIT else min(1.0, (t - ALLIN_HIT) / 0.5)
        u = ease_out_cubic(u)
        xs = np.linspace(x0 + 40, x1 - 40, N)
        ymid = y0 + (y1 - y0) * 0.5
        for i in range(N):
            target_y = y0 if self.allin_win[i] else y1
            y = ymid + (target_y - ymid) * u
            col = self.col_dbl if self.allin_win[i] else self.col_broke
            c = col if u > 0 else self.cols_alive[0]
            # vertical streak from mid to current
            ya, yb = sorted((ymid, y))
            for yy in range(int(ya), int(yb) + 1, 2):
                xi, yi = int(xs[i]), yy
                if 0 <= xi < W - 1 and 0 <= yi < H - 1:
                    acc[yi, xi] += c * 0.08
            xi, yi = int(xs[i]), int(y)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if 0 <= xi + dx < W and 0 <= yi + dy < H:
                        acc[yi + dy, xi + dx] += c * math.exp(-(dx * dx + dy * dy) / 3.0) * 0.5

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, bets, mode, fade):
        surf, ctx = layer()
        x0, y0, x1, y1 = CHART
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, "%": 60})
        if fade < 0.5:
            bets, mode = 0, "slow"
        if mode == "slow":
            resolved = self.end_bet <= bets
            n_broke = int((resolved & self.broke).sum())
            n_dbl = int((resolved & self.doubled).sum())
            left_l, left_v = "BETS EACH", f"{min(bets, MAX_BETS):,}"
        else:
            hit = t >= ALLIN_HIT
            n_broke = (N - self.n_allin) if hit else 0
            n_dbl = self.n_allin if hit else 0
            left_l, left_v = "BETS EACH", "1" if hit else "0"
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, left_l, 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "STILL PLAYING", 0.14 * 24)
        show_tracked(ctx, "STILL PLAYING", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, left_v, 70, 545, big)
        alive = N - n_broke - n_dbl
        s = f"{alive:,}"
        ctx.set_source_rgb(*(rgb(AMBER) if alive else rgb(MAGENTA)))
        tw = cells_width(ctx, s, big)
        show_cells(ctx, s, 910 - tw, 545, big)
        # broke / doubled pills
        font(ctx, "bold", 26)
        for (lab, val, col, xx, align) in ((f"BROKE  {n_broke:,}", n_broke, MAGENTA, 70, "l"), (f"DOUBLED  {n_dbl:,}", n_dbl, CYAN, 910, "r")):
            lw = tracked_width(ctx, lab, 0.08 * 26)
            px = xx if align == "l" else xx - lw - 36
            ctx.set_source_rgba(*rgb(col), 0.14)
            ctx.rectangle(px, 586, lw + 36, 44)
            ctx.fill()
            ctx.set_source_rgba(*rgb(col), 0.95)
            show_tracked(ctx, lab, px + 18, 617, 0.08 * 26)
        # chart axes
        ctx.set_line_width(1)
        for frac, lab in ((0.0, "$200  GOAL"), (0.5, "$100  START"), (1.0, "$0  BROKE")):
            y = y0 + (y1 - y0) * frac
            ctx.set_source_rgba(1, 1, 1, 0.22 if frac != 0.5 else 0.12)
            ctx.set_dash([4, 6] if frac == 0.5 else [])
            ctx.move_to(x0, y + 0.5)
            ctx.line_to(x1, y + 0.5)
            ctx.stroke()
            ctx.set_dash([])
            font(ctx, "bold", 20)
            ctx.set_source_rgba(*(rgb(CYAN) if frac == 0 else rgb(MAGENTA) if frac == 1 else (1, 1, 1)), 0.75)
            show_tracked(ctx, lab, x0, y - 8 if frac < 1 else y + 26, 0.08 * 20)
        if mode == "slow":
            font(ctx, "bold", 18)
            ctx.set_source_rgba(1, 1, 1, 0.45)
            for k in range(0, MAX_BETS + 1, 1000):
                xx = x0 + (x1 - x0) * k / MAX_BETS
                s = f"{k // 1000}k" if k else "0"
                ctx.move_to(xx - ctx.text_extents(s).x_advance / 2, y1 + 50)
                ctx.show_text(s)
            s = "BETS"
            ctx.move_to(x1 - ctx.text_extents(s).x_advance, y1 + 72)
            ctx.show_text(s)
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
                img[band] += g[..., None] * np.array(rgb(MAGENTA), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
