"""Frame renderer for the birthday-paradox reel.  frame(f) -> uint8 1080x1920 RGB.

Act A  0.0-10.2  one random room fills to 23 people on a 365-day ring; the exact P(match) climbs
Act B 10.2-15.4  10,000 simulated rooms of 23 light up a 100x100 grid; the real count is shown
Act C 15.4-17.8  the 253 pairs among 23 people are drawn as chords
Act D 17.8-19.4  the room fills to 70 people (99.9%)
Act E 19.4-20.0  dissolve back to the empty ring so the reel loops
"""

import math
import os
import sys
from math import prod

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)

FPS = 60
N_FRAMES = 1200
SEED = 141
CX = 470
RING_C = (470, 1110)
RING_R = 330
GRID_PITCH, GRID_CELL = 7, 6
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
MONTHS = "JFMAMJJASOND"
MONTH_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MAX_TEXT_W = 820

T_FIRST, DT_PERSON, FLIGHT = 0.6, 0.29, 0.30          # people 1..23
T_D0, DT_FAST, FLIGHT_FAST = 17.85, 0.025, 0.18       # people 24..70
GRID_T0, GRID_T1 = 10.6, 14.6
CHORD_T0, CHORD_T1 = 15.8, 16.8
LOOP_FADE = (19.4, 20.0)


def p_match(n):
    return 1 - prod((365 - k) / 365 for k in range(n))


def arrival(k):
    """Departure time of person k (1-based)."""
    if k <= 23:
        return T_FIRST + DT_PERSON * (k - 1), FLIGHT
    return T_D0 + DT_FAST * (k - 24), FLIGHT_FAST


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.room = rng.integers(0, 365, 70)
        rooms = rng.integers(0, 365, (10000, 23))
        self.grid_match = np.array([len(set(r)) < 23 for r in rooms])
        self.n_match = int(self.grid_match.sum())
        # stacking index per person (how many earlier people share the day) and match flags
        seen = {}
        self.stack, self.matched_with = [], []
        for i, d in enumerate(self.room):
            self.stack.append(seen.get(d, 0))
            self.matched_with.append(seen.get(d, 0) and i)  # truthy if a duplicate
            seen[d] = seen.get(d, 0) + 1
        self.first_match = next(i + 1 for i in range(70) if self.stack[i] > 0)
        assert self.first_match <= 23
        self.P = [p_match(n) for n in range(71)]
        self.pairs = [(i, j) for i in range(23) for j in range(i + 1, 23)]
        self.cues = self._cues()
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.bg = self._background()
        # grid cell template: 100x100 blocks of 7 px with a 1 px gap
        self.mask = np.zeros((GRID_PITCH, GRID_PITCH), np.float32)
        self.mask[:GRID_CELL, :GRID_CELL] = 1

    # ------------------------------------------------------------ setup

    def _cues(self):
        pct = f"{self.n_match / 100:.1f}%"
        hook = [("How many people", "dbold", 54, WHITE, 292, 0.8),
                ("until two share a birthday?", "black", 72, WHITE, 384, 1)]
        return [
            (0.0, 2.2, hook, 0, 6, "none"),
            (2.4, 4.6, [("Guess before you scroll.", "black", 70, AMBER, 350, 1)], 8, 6, "pop"),
            (7.45, 10.0, [("23 people.", "black", 92, WHITE, 312, 1), ("50% chance.", "black", 92, MAGENTA, 404, 1)], 8, 6, "pop"),
            (10.4, 12.4, [("Don't believe it?", "dbold", 58, WHITE, 300, 0.85), ("10,000 rooms of 23.", "black", 80, WHITE, 392, 1)], 8, 6, "pop"),
            (13.9, 15.3, [(f"{self.n_match:,} of 10,000.", "black", 80, WHITE, 312, 1), (f"{pct}.", "black", 96, MAGENTA, 408, 1)], 8, 6, "pop"),
            (15.6, 17.6, [("23 people = 253 pairs.", "black", 70, WHITE, 312, 1), ("Each pair: a 1-in-365 shot.", "dbold", 50, AMBER, 396, 1)], 8, 6, "pop"),
            (17.9, 19.4, [("70 people:", "dbold", 60, WHITE, 296, 0.85), ("99.9%.", "black", 120, CYAN, 404, 1)], 8, 6, "chaos"),
            (19.5, 20.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - RING_C[0]) ** 2 + (yy - RING_C[1]) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#0D1230")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1000, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    def tick_xy(self, day, stack=0):
        a = 2 * math.pi * day / 365 - math.pi / 2
        rr = RING_R + 18 * stack
        return RING_C[0] + rr * math.cos(a), RING_C[1] + rr * math.sin(a)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        ring_alpha = 1.0
        if 10.2 <= t < 15.6:
            ring_alpha = 1 - smooth((t - 10.2) / 0.4) if t < 10.6 else smooth((t - 15.2) / 0.4)
        grid_alpha = 0.0
        if 10.4 <= t < 15.6:
            grid_alpha = smooth((t - 10.4) / 0.4) if t < 10.8 else 1 - smooth((t - 15.2) / 0.4)

        # people present / in flight
        people = []          # (x, y, progress 0..1, stack, k)
        n_landed = 0
        for k in range(1, 71):
            t0, fl = arrival(k)
            if t < t0:
                break
            u = min(1.0, (t - t0) / fl)
            if u >= 1:
                n_landed = k
            day, st = self.room[k - 1], self.stack[k - 1]
            tx, ty = self.tick_xy(day, st)
            e = ease_out_cubic(u)
            people.append((RING_C[0] + (tx - RING_C[0]) * e, RING_C[1] + (ty - RING_C[1]) * e, u, st, k))
        if t >= 15.4 and t < 17.85:
            people = people[:23]
            n_landed = 23

        emis = self._emissive(t, people, n_landed, ring_alpha, grid_alpha, 1 - mix)
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.55 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.35 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.5 * (emis + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)

        self._hud(img, f, t, n_landed, ring_alpha, grid_alpha, mix)
        return dither(img, f)

    # ------------------------------------------------------------ emissive

    def _emissive(self, t, people, n_landed, ring_alpha, grid_alpha, people_alpha=1.0):
        surf, ctx = layer()
        if ring_alpha > 0:
            self._ring(ctx, t, ring_alpha)
            ring_alpha = ring_alpha * people_alpha   # people, chords and flares dissolve at the loop
            # chords (act C)
            if CHORD_T0 <= t < 17.95:
                n = len(self.pairs) if t >= CHORD_T1 else int(len(self.pairs) * smooth((t - CHORD_T0) / (CHORD_T1 - CHORD_T0)))
                a = ring_alpha * (1 - smooth((t - 17.85) / 0.1) if t >= 17.85 else 1)
                ctx.set_line_width(1.1)
                pos = {p[4]: (p[0], p[1]) for p in people}
                for (i, j) in self.pairs[:n]:
                    if i + 1 in pos and j + 1 in pos:
                        same = self.room[i] == self.room[j]
                        ctx.set_source_rgba(*(rgb(MAGENTA) if same else (1, 1, 1)), (0.9 if same else 0.16) * a)
                        ctx.move_to(*pos[i + 1])
                        ctx.line_to(*pos[j + 1])
                        ctx.stroke()
            # people
            for (x, y, u, st, k) in people:
                t0, fl = arrival(k)
                age = t - t0 - fl
                dup = st > 0
                col = rgb(MAGENTA) if dup and u >= 1 else rgb(CYAN)
                r = 9.0 if k <= 23 else 7.0
                ctx.set_source_rgba(*col, ring_alpha * (0.6 + 0.4 * u))
                ctx.arc(x, y, r, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 0.9 * ring_alpha)
                ctx.arc(x, y, 2.5, 0, 2 * math.pi)
                ctx.fill()
                if dup and 0 <= age < 0.6:               # match flare + the original person lights up
                    w = age / 0.6
                    ctx.set_line_width(3 * (1 - w) + 0.5)
                    ctx.set_source_rgba(*rgb(MAGENTA), (1 - w) * ring_alpha)
                    ctx.arc(x, y, 12 + 70 * ease_out_cubic(w), 0, 2 * math.pi)
                    ctx.stroke()
                    ox, oy = self.tick_xy(self.room[k - 1], 0)
                    ctx.set_source_rgba(*rgb(MAGENTA), (1 - w) * ring_alpha)
                    ctx.arc(ox, oy, 10 + 6 * (1 - w), 0, 2 * math.pi)
                    ctx.fill()
                if dup and u >= 1:                        # highlight the shared tick
                    ox, oy = self.tick_xy(self.room[k - 1], 0)
                    ctx.set_source_rgba(*rgb(MAGENTA), 0.9 * ring_alpha)
                    ctx.arc(ox, oy, 7.5, 0, 2 * math.pi)
                    ctx.fill()
        prgb, _ = surf_rgba(surf)
        emis = (prgb ** 2.2 * 0.9).astype(np.float32)
        if grid_alpha > 0:
            emis += self._grid(t) * grid_alpha
        return emis

    def _ring(self, ctx, t, a):
        cx, cy = RING_C
        # sweeping highlight so frame 0 already moves
        sweep = (t / 6.0) % 1.0
        for i in range(365):
            ang = 2 * math.pi * i / 365 - math.pi / 2
            ln = 18 if i in MONTH_START else 9
            d = (i / 365 - sweep) % 1.0
            hi = max(0.0, 1 - d / 0.08) if d < 0.08 else 0.0
            ctx.set_line_width(1.6 if ln == 9 else 2.2)
            ctx.set_source_rgba(1, 1, 1, (0.34 + 0.5 * hi) * a)
            ctx.move_to(cx + (RING_R - ln) * math.cos(ang), cy + (RING_R - ln) * math.sin(ang))
            ctx.line_to(cx + RING_R * math.cos(ang), cy + RING_R * math.sin(ang))
            ctx.stroke()
        font(ctx, "bold", 22)
        ctx.set_source_rgba(1, 1, 1, 0.5 * a)
        for m, d0 in enumerate(MONTH_START):
            ang = 2 * math.pi * (d0 + 15) / 365 - math.pi / 2
            s = MONTHS[m]
            w = ctx.text_extents(s).x_advance
            ctx.move_to(cx + (RING_R + 36) * math.cos(ang) - w / 2, cy + (RING_R + 36) * math.sin(ang) + 8)
            ctx.show_text(s)

    def _grid(self, t):
        n = 10000 if t >= GRID_T1 else int(10000 * smooth((t - GRID_T0) / (GRID_T1 - GRID_T0)))
        cells = np.zeros((10000, 3), np.float32)
        cells[:n] = np.where(self.grid_match[:n, None], self.c_mag * 0.55, self.c_cyan * 0.30)
        if n < 10000:
            cells[n] = 1.0
        img = np.kron(cells.reshape(100, 100, 3), self.mask[..., None])
        out = np.zeros((H, W, 3), np.float32)
        x0, y0 = RING_C[0] - 350, RING_C[1] - 350
        out[y0:y0 + 700, x0:x0 + 700] = img
        return out

    # ------------------------------------------------------------ HUD

    def _hud(self, img, f, t, n_landed, ring_alpha, grid_alpha, mix):
        surf, ctx = layer()
        big = {str(k): 50 for k in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        font(ctx, "bold", 24)
        if grid_alpha > 0.5:
            n = 10000 if t >= GRID_T1 else int(10000 * smooth((t - GRID_T0) / (GRID_T1 - GRID_T0)))
            m = int(self.grid_match[:n].sum())
            left_l, left_v, left_c = "ROOMS", f"{n:,}", rgb(WHITE)
            right_l, right_v = "ROOMS WITH A MATCH", f"{m:,}"
            right_c = rgb(MAGENTA)
            sub = f"{100 * m / n:.1f}%" if n else ""
        elif 15.4 <= t < 17.85:
            left_l, left_v, left_c = "PEOPLE", "23", rgb(WHITE)
            right_l, right_v, right_c, sub = "PAIRS", f"{len(self.pairs):,}", rgb(AMBER), ""
        else:
            n = 0 if mix > 0.5 else n_landed
            p = self.P[n]
            left_l, left_v, left_c = "PEOPLE", f"{n}", rgb(WHITE)
            right_l, right_v = "CHANCE OF A MATCH", f"{100 * p:.1f}%"
            right_c = rgb(CYAN) if p >= 0.99 else rgb(MAGENTA) if p >= 0.5 else rgb(AMBER)
            sub = ""
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, left_l, 70, 470, 0.14 * 24)
        w = tracked_width(ctx, right_l, 0.14 * 24)
        show_tracked(ctx, right_l, 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*left_c)
        show_cells(ctx, left_v, 70, 545, big)
        ctx.set_source_rgb(*right_c)
        tw = cells_width(ctx, right_v, big)
        show_cells(ctx, right_v, 910 - tw, 545, big)
        if sub:
            font(ctx, "dbold", 34)
            ctx.set_source_rgba(1, 1, 1, 0.8)
            ctx.move_to(910 - ctx.text_extents(sub).x_advance, 590)
            ctx.show_text(sub)
        # pivot hint: "1 in 365" on the ring centre during act A
        if ring_alpha > 0 and t < 10.2:
            font(ctx, "bold", 26)
            ctx.set_source_rgba(1, 1, 1, 0.35 * ring_alpha)
            s = "365 days"
            ctx.move_to(RING_C[0] - ctx.text_extents(s).x_advance / 2, RING_C[1] + 9)
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
                img[band] += g[..., None] * np.array(rgb(CYAN), np.float32)
            sh = np.roll(cv2.GaussianBlur(a_band, (0, 0), 10) * 0.65 * alpha, 3, axis=0)
            img[band] *= (1 - sh)[..., None]
            img[band] *= (1 - a_band * alpha)[..., None]
            img[band] += trgb[band] * alpha
        np.clip(img, 0, 1, out=img)
