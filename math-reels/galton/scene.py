"""Galton board: 10,000 glowing balls fall through 16 rows of pegs; at each peg a fair coin flip sends a ball left
or right; they pile up in 17 bins into a bell curve, then the exact prediction, C(16, k) / 2^16, is drawn over the
piles; the edges (16 flips the same way, 1 in 65,536) stay empty; the floor opens and it loops.
Balls do not collide with each other: each one's path is its 16 coin flips (the ideal Galton board).
    frame(f) -> uint8 1080x1920 RGB"""

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
import gsim  # noqa: E402

FPS = 60
N_FRAMES = 1320                                  # 22 s
T_LOOP = N_FRAMES / FPS
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
SLATE = "#7C88B5"
CX = 470
MAX_TEXT_W = 820
N, ROWS = gsim.N, gsim.ROWS
NB = ROWS + 1
# board geometry (px)
DX = 48.0                                        # peg spacing = bin width
PEG_Y0, DY = 650.0, 27.0                         # first peg row, row spacing
PEG_R, BALL_R = 3.2, 2.6
Y_START = 600.0
Y_EXIT = PEG_Y0 + ROWS * DY                      # 1082: where a ball leaves the last row
WALL_TOP, FLOOR = 1092.0, 1532.0
PER_ROW, RH = 14, 2.9                            # balls per layer in a bin, layer height
# timing (s)
ENTRY, HOP, G = 0.14, 0.07, 6000.0               # drop to the first peg, one row, gravity in px/s^2
RAMP0, RAMP1 = -1.0, 12.0                        # releases, faster and faster; the first second wraps from the end
REVEAL = 13.8
EDGE_T = 16.1
DRAIN0, DRAIN1 = 20.3, 21.2
LOOP_FADE = (21.3, 22.0)


def bin_x(k):
    return CX + (k - ROWS / 2) * DX


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
    def __init__(self):
        fl = gsim.flips().astype(np.float64)
        self.S = np.concatenate([np.zeros((N, 1)), np.cumsum(fl - 0.5, 1)], 1)    # offset in pegs after j rows
        self.bin = (self.S[:, -1] + ROWS / 2).round().astype(int)
        self.counts = np.bincount(self.bin, minlength=NB)
        self.expected = gsim.expected()
        # release times: released by time t = N ((t - RAMP0) / (RAMP1 - RAMP0))^2; negative ones start at the loop's end
        i = np.arange(N, dtype=np.float64)
        self.rel = RAMP0 + (RAMP1 - RAMP0) * np.sqrt((i + 0.5) / N)
        # each ball's slot in its bin, in release order
        self.slot = np.zeros(N, int)
        seen = np.zeros(NB, int)
        for b_i in range(N):
            k = self.bin[b_i]
            self.slot[b_i] = seen[k]
            seen[k] += 1
        col = self.slot % PER_ROW
        lay = self.slot // PER_ROW
        self.sx = bin_x(self.bin) - DX / 2 + 2 + (col + 0.5) * (DX - 4) / PER_ROW
        self.sy = FLOOR - 2 - (lay + 0.5) * RH
        self.t_fall = np.sqrt(2 * (self.sy - Y_EXIT) / G)
        self.t_exit = self.rel + ENTRY + ROWS * HOP
        self.t_land = self.t_exit + self.t_fall
        # landed balls by bin colour: cyan in the middle to magenta at the edges
        u = np.abs(self.bin - ROWS / 2) / (ROWS / 2)
        c_mid = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        c_edge = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        w_ = np.clip(u * 1.6, 0, 1)[:, None]
        self.col_land = (c_mid * (1 - w_) + c_edge * w_).astype(np.float32)
        self.col_fly = np.array(srgb_to_linear(rgb("#FFF1D6")), np.float32)
        # peg flashes: hits per frame per peg, decayed
        n_pegs = ROWS * (ROWS + 1) // 2
        self.peg_xy = np.array([(CX + (k - j / 2) * DX, PEG_Y0 + j * DY) for j in range(ROWS) for k in range(j + 1)])
        hit_t = self.rel[:, None] + ENTRY + np.arange(ROWS)[None, :] * HOP
        peg_k = (self.S[:, :ROWS] + np.arange(ROWS)[None, :] / 2).round().astype(int)
        peg_id = (np.arange(ROWS)[None, :] * (np.arange(ROWS)[None, :] + 1) // 2) + peg_k
        hits = np.zeros((N_FRAMES, n_pegs))
        np.add.at(hits, (np.floor(hit_t * FPS).astype(int).ravel() % N_FRAMES, peg_id.ravel()), 1.0)
        self.hit_frames = np.floor(hit_t * FPS).astype(int) % N_FRAMES          # for the sound
        self.hit_x = self.peg_xy[peg_id, 0]
        flash = np.zeros_like(hits)
        acc = np.zeros(n_pegs)
        for f in list(range(N_FRAMES)) * 2:                                     # twice: the decay wraps
            acc = acc * math.exp(-1 / (0.06 * FPS)) + hits[f]
            flash[f] = acc
        self.flash = flash
        self.hits_per_frame = hits.sum(1)
        self.c_peg = np.array(srgb_to_linear(rgb(SLATE)), np.float32)
        self.c_amber = np.array(srgb_to_linear(rgb(AMBER)), np.float32)
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        mid, ex_mid = int(self.counts[ROWS // 2]), self.expected[ROWS // 2]
        hook = [("10,000 balls.", "black", 96, WHITE, 300, 1), ("Every peg: a coin flip. Left or right.", "dbold", 46, WHITE, 392, 0.85)]
        return [
            (0.0, 2.3, hook, 0, 6, "none"),
            (2.5, 4.4, [("Where will each one land?", "black", 66, WHITE, 312, 1), ("Nobody can say.", "dbold", 52, AMBER, 400, 1)], 8, 6, "pop"),
            (5.0, 7.0, [("16 flips per ball.", "black", 70, WHITE, 312, 1),
                        (f"Pure chance, {N * ROWS:,} times.", "dbold", 50, WHITE, 400, 0.9)], 8, 6, "pop"),
            (REVEAL, 15.9, [("A bell curve.", "black", 90, CYAN, 316, 1),
                            ("Predicted before the first ball fell.", "dbold", 44, AMBER, 404, 1)], 6, 6, "chaos"),
            (EDGE_T, 18.1, [("To land on an edge:", "black", 66, WHITE, 312, 1),
                            (f"16 flips the same way. 1 in {2 ** ROWS:,}.", "dbold", 44, MAGENTA, 400, 1)], 8, 6, "pop"),
            (18.3, 20.2, [("One ball: pure chance.", "black", 62, WHITE, 312, 1),
                          ("10,000: almost exactly predictable.", "black", 52, CYAN, 396, 1)], 8, 6, "pop"),
            (LOOP_FADE[0], LOOP_FADE[1] + 0.2, hook, 10, 0, "fade"),
        ], (mid, ex_mid)

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1000) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ ball positions

    def positions(self, t):
        """(xy of balls in flight, xy of landed balls, landed mask, released count). Balls released before 0 are
        also drawn at the end of the loop, so the stream is already falling on the first frame."""
        e = t - self.rel
        fly = (e >= 0) & (t < self.t_land)
        landed = t >= self.t_land
        idx = np.nonzero(fly)[0]
        xy = self._pos(idx, e[idx])
        released = int(np.count_nonzero(e >= 0))
        if t >= T_LOOP + RAMP0:
            ew = t - T_LOOP - self.rel
            iw = np.nonzero(ew >= 0)[0]
            xy = np.concatenate([xy, self._pos(iw, ew[iw])])
            released = len(iw)
        return xy, np.stack([self.sx[landed], self.sy[landed]], 1), landed, released

    def _pos(self, idx, ee):
        xy = np.zeros((len(idx), 2))
        idx_all = idx
        idx = np.arange(len(idx))
        # entry: fall from the top to the first peg
        a = ee < ENTRY
        if a.any():
            u = ee[a] / ENTRY
            xy[idx[a], 0] = CX
            xy[idx[a], 1] = Y_START + (PEG_Y0 - PEG_R - BALL_R - Y_START) * u * u
        # hops between rows (row ROWS is the exit)
        b = (ee >= ENTRY) & (ee < ENTRY + ROWS * HOP)
        if b.any():
            ib = idx[b]
            gb = idx_all[b]
            s = (ee[b] - ENTRY) / HOP
            j = np.floor(s).astype(int)
            u = s - j
            x0 = CX + self.S[gb, j] * DX
            x1 = CX + self.S[gb, j + 1] * DX
            y0 = PEG_Y0 + j * DY - PEG_R - BALL_R
            y1 = np.where(j + 1 < ROWS, PEG_Y0 + (j + 1) * DY - PEG_R - BALL_R, Y_EXIT)
            xy[ib, 0] = x0 + (x1 - x0) * u
            xy[ib, 1] = y0 + (y1 - y0) * u * u - 9 * 4 * u * (1 - u)
        # fall into the bin
        c = ee >= ENTRY + ROWS * HOP
        if c.any():
            ic = idx[c]
            gc = idx_all[c]
            tf = ee[c] - ENTRY - ROWS * HOP
            u = np.clip(tf / self.t_fall[gc], 0, 1)
            x_exit = CX + self.S[gc, ROWS] * DX
            xy[ic, 0] = x_exit + (self.sx[gc] - x_exit) * smooth_arr(u)
            xy[ic, 1] = np.minimum(Y_EXIT + 0.5 * G * tf * tf, self.sy[gc])
        return xy

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        fly_xy, land_xy, landed, released = self.positions(t)
        if t >= DRAIN0:                                   # the floor opens: everything drops out
            td = t - DRAIN0
            land_xy = land_xy.copy()
            land_xy[:, 1] += 0.5 * 3500 * td * td
            a_land = 1 - smooth(td / (DRAIN1 - DRAIN0))
        else:
            a_land = 1.0
        acc = np.zeros((H, W, 3), np.float32)
        if len(land_xy) and a_land > 0:
            keep = land_xy[:, 1] < H + 10
            splat_points(acc, land_xy[keep].astype(np.float64), self.col_land[landed][keep] * a_land, 0.55, 1.05)
        if len(fly_xy):
            cols = np.repeat(self.col_fly[None, :], len(fly_xy), 0)
            splat_points(acc, fly_xy.astype(np.float64), cols, 0.9, 1.5)
        # pegs: a dim dot each, flashing when hit
        fl = self.flash[f]
        lvl = 0.18 + 0.55 * (1 - np.exp(-0.6 * fl))
        cols = (self.c_peg[None, :] * 0.6 + self.c_amber[None, :] * 0.4 * (1 - np.exp(-0.6 * fl))[:, None]) * lvl[:, None]
        splat_points(acc, self.peg_xy.astype(np.float64), (cols * 6).astype(np.float32), 0.5, 1.6)
        np.clip(acc, 0, 3, out=acc)
        small = cv2.resize(acc, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.5 * cv2.GaussianBlur(small, (0, 0), 2) + 0.35 * cv2.GaussianBlur(small, (0, 0), 8),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (acc + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        surf, ctx = layer()
        self._board(ctx, t)
        self._hud(ctx, t, landed, released, mix)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)
        return dither(img, f)

    # ------------------------------------------------------------ board, prediction, HUD

    def _board(self, ctx, t):
        wall_drop = 0.0
        if t >= DRAIN0:
            wall_drop = smooth((t - DRAIN0) / 0.25)
        ctx.set_line_width(2)
        for k in range(NB + 1):
            x = bin_x(k) - DX / 2
            ctx.set_source_rgba(*rgb(SLATE), 0.35)
            ctx.move_to(x, WALL_TOP)
            ctx.line_to(x, FLOOR)
            ctx.stroke()
        # the floor (swings open at the end)
        ctx.set_source_rgba(*rgb(SLATE), 0.5 * (1 - wall_drop))
        ctx.set_line_width(3)
        ctx.move_to(bin_x(0) - DX / 2, FLOOR)
        ctx.line_to(bin_x(ROWS) + DX / 2, FLOOR)
        ctx.stroke()
        # the prediction: expected count per bin, drawn left to right after the last ball lands
        if REVEAL <= t < DRAIN0 + 0.4:
            u = smooth((t - REVEAL) / 0.9)
            a = 1 - smooth((t - DRAIN0) / 0.4) if t >= DRAIN0 else 1.0
            pts = [(bin_x(k), FLOOR - 2 - self.expected[k] / PER_ROW * RH) for k in range(NB)]
            xs = np.linspace(pts[0][0], pts[-1][0], 400)
            ys = np.interp(xs, [p[0] for p in pts], [p[1] for p in pts])
            # smooth through the points: Catmull-Rom between bin centres
            ys = catmull(pts, xs)
            n = int(len(xs) * u)
            if n > 1:
                ctx.set_source_rgba(*rgb(AMBER), 0.95 * a)
                ctx.set_line_width(4)
                ctx.set_dash([10, 7])
                ctx.move_to(xs[0], ys[0])
                for x, y in zip(xs[1:n], ys[1:n]):
                    ctx.line_to(x, y)
                ctx.stroke()
                ctx.set_dash([])
            for k, (x, y) in enumerate(pts):
                if x <= xs[max(n - 1, 0)]:
                    ctx.set_source_rgba(*rgb(AMBER), a)
                    ctx.arc(x, y, 5, 0, 2 * math.pi)
                    ctx.fill()
            # the middle bin: predicted vs landed
            if t >= REVEAL + 0.9:
                k = ROWS // 2
                aa = smooth((t - REVEAL - 0.9) / 0.4) * a
                y = pts[k][1] - 26
                font(ctx, "black", 30)
                s1 = f"PREDICTED {self.expected[k]:,.0f}"
                s2 = f"LANDED {int(self.counts[k]):,}"
                ctx.set_source_rgba(*rgb(AMBER), aa)
                ctx.move_to(CX - ctx.text_extents(s1).x_advance / 2, y - 38)
                ctx.show_text(s1)
                ctx.set_source_rgba(*rgb(CYAN), aa)
                ctx.move_to(CX - ctx.text_extents(s2).x_advance / 2, y)
                ctx.show_text(s2)
        # the edges: 1 in 65,536 each
        if EDGE_T <= t < DRAIN0 + 0.3:
            a = smooth((t - EDGE_T) / 0.4) * (1 - smooth((t - DRAIN0) / 0.3) if t >= DRAIN0 else 1.0)
            for k in (0, ROWS):
                x = bin_x(k)
                ctx.set_source_rgba(*rgb(MAGENTA), 0.9 * a)
                ctx.set_line_width(3)
                ctx.rectangle(x - DX / 2 + 2, WALL_TOP, DX - 4, FLOOR - WALL_TOP)
                ctx.stroke()
                font(ctx, "black", 40)
                s = f"{int(self.counts[k])}"
                ctx.move_to(x - ctx.text_extents(s).x_advance / 2, FLOOR + 50)
                ctx.show_text(s)
                font(ctx, "bold", 18)
                s = "BALLS"
                ctx.move_to(x - ctx.text_extents(s).x_advance / 2, FLOOR + 76)
                ctx.show_text(s)

    def _hud(self, ctx, t, landed, released, mix):
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        n_land = int(landed.sum())
        if t >= DRAIN0:
            n_land_show = N if mix <= 0.5 else 0
        else:
            n_land_show = n_land
        drop = released if (mix > 0.5 or t < DRAIN0) else N
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "BALLS DROPPED", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "IN THE MIDDLE BIN", 0.14 * 24)
        show_tracked(ctx, "IN THE MIDDLE BIN", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{drop:,}", 70, 545, big)
        if n_land_show:
            if t >= DRAIN0:
                mid = int(self.counts[ROWS // 2])
            else:
                mid = int(np.count_nonzero(landed & (self.bin == ROWS // 2)))
            s = f"{100 * mid / n_land_show:.1f}%"
            ctx.set_source_rgb(*rgb(CYAN))
            show_cells(ctx, s, 910 - cells_width(ctx, s, big), 545, big)
        else:
            font(ctx, "dbold", 56)
            ctx.set_source_rgb(*rgb(AMBER))
            ctx.move_to(910 - ctx.text_extents("?").x_advance, 540)
            ctx.show_text("?")

    def _headlines(self, img, f, t):
        cues, _ = self.cues
        for (t_in, t_out, lines, n_in, n_out, style) in cues:
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


def smooth_arr(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def catmull(pts, xs):
    """Catmull-Rom spline through points with increasing x, evaluated at xs (by segment parameter)."""
    P = np.array(pts, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = np.empty_like(xs)
    for n, x in enumerate(xs):
        i = min(max(int((x - P[1, 0]) / (P[2, 0] - P[1, 0])), 0), len(pts) - 2)
        u = (x - P[i + 1, 0]) / (P[i + 2, 0] - P[i + 1, 0])
        p0, p1, p2, p3 = P[i, 1], P[i + 1, 1], P[i + 2, 1], P[i + 3, 1]
        out[n] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
    return out
