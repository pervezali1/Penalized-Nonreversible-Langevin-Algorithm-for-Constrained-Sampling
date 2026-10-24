"""Monty Hall: 10,000 simulated games with STAY and SWITCH side by side (every game fills one
cell of a 100x100 grid: cyan if switching won, magenta if staying won), then the 100-door
version: you pick one, the host opens 98 goats.  frame(f) -> uint8 1080x1920 RGB."""

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textkit import (H, W, cells_width, dither, ease_out_back, ease_out_cubic, font, layer,  # noqa: E402
                     linear_to_srgb, over, rgb, show_cells, show_tracked, smooth, srgb_to_linear,
                     surf_rgba, tracked_width)

FPS = 60
N_FRAMES = 1200
SEED = 8
GAMES = 10_000
CYAN, MAGENTA, AMBER, WHITE = "#2EE6FF", "#FF3D7F", "#FFC94D", "#F5F7FF"
DOOR = "#D9CDB4"
CX = 470
SLOW_T0, SLOW_DT, SLOW_N = 4.8, 0.6, 5
FAST_T0, FAST_T1 = 7.8, 11.4
B_T0, B_OPEN0, B_OPEN1, B_REVEAL = 13.6, 14.2, 16.0, 16.2
LOOP_FADE = (19.4, 20.0)
MAX_TEXT_W = 820
COL_X = (240, 700)                  # STAY / SWITCH column centres
DOORS_Y, DOOR_W, DOOR_H, DOOR_GAP = 712, 110, 160, 14
PCT_Y, BAR_Y = 1000, 1018
GRID_C, GRID_PITCH, GRID_CELL = (470, 1290), 4.2, 3.4     # 100x100 cells -> 420 px
BIG_C, BIG_PITCH, BIG_CELL = (470, 1290), 42, 35           # 10x10 doors -> 420 px


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.car = rng.integers(0, 3, GAMES)
        self.pick = rng.integers(0, 3, GAMES)
        self.host = np.empty(GAMES, np.int64)
        for g in range(GAMES):
            options = [d for d in range(3) if d != self.pick[g] and d != self.car[g]]
            self.host[g] = options[rng.integers(0, len(options))]
        self.switch_to = np.array([[d for d in range(3) if d != self.pick[g] and d != self.host[g]][0] for g in range(GAMES)])
        self.stay_win = self.pick == self.car
        self.switch_win = self.switch_to == self.car
        assert np.all(self.stay_win != self.switch_win)
        self.cum_stay = np.concatenate([[0], np.cumsum(self.stay_win)])
        self.cum_switch = np.concatenate([[0], np.cumsum(self.switch_win)])
        self.final = (float(self.stay_win.mean()), float(self.switch_win.mean()))
        self.big_pick = int(rng.integers(0, 100))
        self.big_car = int(rng.integers(0, 100))
        while self.big_car == self.big_pick:
            self.big_car = int(rng.integers(0, 100))
        order = [d for d in range(100) if d not in (self.big_pick, self.big_car)]
        rng.shuffle(order)
        self.big_open_order = np.array(order)          # the 98 goats the host opens
        self.mask = np.zeros((5, 5), np.float32)
        self.mask[:4, :4] = 1
        self.c_cyan = np.array(srgb_to_linear(rgb(CYAN)), np.float32)
        self.c_mag = np.array(srgb_to_linear(rgb(MAGENTA)), np.float32)
        self.bg = self._background()
        self.cues = self._cues()

    def _cues(self):
        ps, pw = self.final
        hook = [("3 doors. 1 car. 2 goats.", "black", 74, WHITE, 300, 1), ("You pick one. The host opens a goat.", "dbold", 44, WHITE, 392, 0.9)]
        return [
            (0.0, 2.2, hook, 0, 6, "none"),
            (2.4, 4.6, [("Switch or stay?", "black", 96, AMBER, 320, 1), ("Most people say it doesn't matter.", "dbold", 44, WHITE, 400, 0.9)], 8, 6, "pop"),
            (7.9, 9.8, [("10,000 games.", "black", 80, WHITE, 350, 1)], 8, 6, "pop"),
            (11.6, 13.4, [(f"Switch wins {100 * pw:.1f}%.", "black", 80, CYAN, 312, 1), ("2 out of 3. Not a trick.", "dbold", 48, WHITE, 400, 0.9)], 6, 6, "chaos"),
            (B_T0, 16.0, [("Still 50/50? Try 100 doors.", "black", 60, WHITE, 312, 1), ("You pick one. The host opens 98 goats.", "dbold", 42, AMBER, 396, 1)], 8, 6, "pop"),
            (B_REVEAL, 17.6, [("Your door: 1%.", "black", 72, AMBER, 312, 1), ("The other one: 99%.", "black", 72, CYAN, 404, 1)], 8, 6, "pop"),
            (17.8, 19.3, [("Switching wins exactly when", "dbold", 46, WHITE, 306, 0.9), ("your first pick was wrong: 2 in 3.", "black", 54, AMBER, 396, 1)], 8, 6, "pop"),
            (19.5, 20.2, hook, 10, 0, "fade"),
        ]

    def _background(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - CX) ** 2 + (yy - 1000) ** 2)
        c0 = np.array(srgb_to_linear(rgb("#101528")), np.float32)
        c1 = np.array(srgb_to_linear(rgb("#05060C")), np.float32)
        u = np.clip(r / 1100, 0, 1)[..., None]
        return (c0 * (1 - u) + c1 * u).astype(np.float32)

    # ------------------------------------------------------------ schedule

    def games_done(self, t):
        if t < SLOW_T0:
            return 0
        if t < FAST_T0:
            return min(SLOW_N, int((t - SLOW_T0) // SLOW_DT) + (1 if (t - SLOW_T0) % SLOW_DT >= 0.45 else 0))
        if t >= FAST_T1:
            return GAMES
        u = (t - FAST_T0) / (FAST_T1 - FAST_T0)
        return min(GAMES, SLOW_N + int((GAMES - SLOW_N) * u ** 2.2))

    def slow_stage(self, t):
        """(game index, stage 0..3) during the slow games: 0 pick, 1 host opens, 2 switch moves, 3 reveal."""
        if not (SLOW_T0 <= t < FAST_T0):
            return None
        g = int((t - SLOW_T0) // SLOW_DT)
        if g >= SLOW_N:
            return None
        age = (t - SLOW_T0) % SLOW_DT
        return g, (0 if age < 0.12 else 1 if age < 0.27 else 2 if age < 0.45 else 3)

    # ------------------------------------------------------------ frame

    def frame(self, f):
        f %= N_FRAMES
        t = f / FPS
        mix = smooth((t - LOOP_FADE[0]) / (LOOP_FADE[1] - LOOP_FADE[0])) if t >= LOOP_FADE[0] else 0.0
        done = self.games_done(t)
        if mix > 0.5:
            done = 0
        emis = np.zeros((H, W, 3), np.float32)
        if t < B_T0 or mix > 0.5:
            self._grid(emis, done)
        else:
            self._big_doors(emis, t)
        small = cv2.resize(emis, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(0.4 * cv2.GaussianBlur(small, (0, 0), 2.5) + 0.3 * cv2.GaussianBlur(small, (0, 0), 9),
                          (W, H), interpolation=cv2.INTER_LINEAR)
        e = 1 - np.exp(-1.4 * (emis + glow))
        lin = 1 - (1 - self.bg) * (1 - e)
        img = linear_to_srgb(lin).astype(np.float32)
        self._hud(img, f, t, done, mix)
        return dither(img, f)

    def _grid(self, emis, done):
        n = done
        cells = np.zeros((GAMES, 3), np.float32)
        if n:
            cells[:n] = np.where(self.switch_win[:n, None], self.c_cyan * 0.55, self.c_mag * 0.55)
        img = cv2.resize(cells.reshape(100, 100, 3), (420, 420), interpolation=cv2.INTER_NEAREST)
        x0, y0 = GRID_C[0] - 210, GRID_C[1] - 210
        emis[y0:y0 + 420, x0:x0 + 420] += img

    def _big_doors(self, emis, t):
        """100 doors as emissive tiles: opened goats magenta (dim), your pick amber, the last one cyan."""
        n_open = 0 if t < B_OPEN0 else 98 if t >= B_OPEN1 else int(98 * smooth((t - B_OPEN0) / (B_OPEN1 - B_OPEN0)))
        opened = set(self.big_open_order[:n_open].tolist())
        revealed = t >= B_REVEAL
        for d in range(100):
            r, c = divmod(d, 10)
            x = BIG_C[0] - 210 + c * BIG_PITCH + 3
            y = BIG_C[1] - 210 + r * BIG_PITCH + 3
            if d in opened:
                col = self.c_mag * 0.18
            elif d == self.big_pick:
                col = np.array(srgb_to_linear(rgb(AMBER)), np.float32) * 0.8
            elif d == self.big_car and revealed:
                col = self.c_cyan * 0.9
            else:
                col = np.array(srgb_to_linear(rgb(DOOR)), np.float32) * 0.35
            emis[int(y):int(y + BIG_CELL), int(x):int(x + BIG_CELL)] += col

    # ------------------------------------------------------------ HUD

    def _door(self, ctx, x, y, state, alpha=1.0):
        """state: 'closed' | 'goat' | 'car' | 'closed_pick'."""
        r = 10
        ctx.new_sub_path()
        ctx.arc(x + DOOR_W - r, y + r, r, -math.pi / 2, 0)
        ctx.line_to(x + DOOR_W, y + DOOR_H)
        ctx.line_to(x, y + DOOR_H)
        ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        if state in ("closed", "closed_pick"):
            ctx.set_source_rgba(*rgb(DOOR), 0.9 * alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(*rgb(AMBER), (1.0 if state == "closed_pick" else 0.0) * alpha)
            ctx.set_line_width(5)
            ctx.stroke()
            ctx.set_source_rgba(0.1, 0.1, 0.15, 0.8 * alpha)
            ctx.arc(x + DOOR_W - 18, y + DOOR_H / 2, 4, 0, 2 * math.pi)
            ctx.fill()
        else:
            ctx.set_source_rgba(0.06, 0.07, 0.12, 0.95 * alpha)
            ctx.fill()
            cx, cy = x + DOOR_W / 2, y + DOOR_H / 2
            if state == "car":
                ctx.set_source_rgba(*rgb(CYAN), alpha)
                ctx.rectangle(cx - 26, cy - 6, 52, 20)
                ctx.fill()
                ctx.rectangle(cx - 16, cy - 20, 32, 16)
                ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
                ctx.arc(cx - 15, cy + 16, 6, 0, 2 * math.pi)
                ctx.fill()
                ctx.arc(cx + 15, cy + 16, 6, 0, 2 * math.pi)
                ctx.fill()
            else:
                ctx.set_source_rgba(*rgb(MAGENTA), alpha)
                ctx.arc(cx, cy + 4, 18, 0, 2 * math.pi)
                ctx.fill()
                for sx in (-1, 1):
                    ctx.move_to(cx + sx * 10, cy - 10)
                    ctx.line_to(cx + sx * 22, cy - 30)
                    ctx.line_to(cx + sx * 4, cy - 14)
                    ctx.close_path()
                    ctx.fill()
                ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
                ctx.arc(cx - 6, cy, 2.5, 0, 2 * math.pi)
                ctx.fill()
                ctx.arc(cx + 6, cy, 2.5, 0, 2 * math.pi)
                ctx.fill()

    def _hud(self, img, f, t, done, mix):
        surf, ctx = layer()
        big = {str(c): 50 for c in range(10)}
        big.update({",": 22, ".": 22, "%": 60})
        font(ctx, "bold", 24)
        ctx.set_source_rgba(1, 1, 1, 0.55)
        show_tracked(ctx, "GAMES", 70, 470, 0.14 * 24)
        w = tracked_width(ctx, "AHEAD", 0.14 * 24)
        show_tracked(ctx, "AHEAD", 910 - w, 470, 0.14 * 24)
        font(ctx, "dbold", 76)
        ctx.set_source_rgb(*rgb(WHITE))
        show_cells(ctx, f"{done:,}", 70, 545, big)
        fs = self.cum_stay[done] / done if done else 0
        fw = self.cum_switch[done] / done if done else 0
        if done == 0:
            s, col = "?", rgb(AMBER)
        else:
            s, col = ("SWITCH", rgb(CYAN)) if fw >= fs else ("STAY", rgb(MAGENTA))
        font(ctx, "dbold", 56)
        ctx.set_source_rgb(*col)
        ctx.move_to(910 - ctx.text_extents(s).x_advance, 540)
        ctx.show_text(s)
        in_big = t >= B_T0 and mix <= 0.5
        if not in_big:
            self._columns(ctx, t, done)
        else:
            self._big_labels(ctx, t)
        trgb, ta = surf_rgba(surf)
        over(img, trgb, ta)
        self._headlines(img, f, t)

    def _columns(self, ctx, t, done):
        slow = self.slow_stage(t)
        g = slow[0] if slow else max(0, done - 1)
        stage = slow[1] if slow else 3
        for ci, (name, colx, col) in enumerate((("STAY", COL_X[0], MAGENTA), ("SWITCH", COL_X[1], CYAN))):
            font(ctx, "black", 30)
            ctx.set_source_rgba(*rgb(col), 0.95)
            ctx.move_to(colx - ctx.text_extents(name).x_advance / 2, DOORS_Y - 22)
            ctx.show_text(name)
            x0 = colx - (3 * DOOR_W + 2 * DOOR_GAP) / 2
            for d in range(3):
                x = x0 + d * (DOOR_W + DOOR_GAP)
                if done == 0 and not slow:
                    state = "closed"
                else:
                    chosen = self.pick[g] if (ci == 0 or stage < 2) else self.switch_to[g]
                    if stage >= 1 and d == self.host[g]:
                        state = "goat"
                    elif stage >= 3 and d == chosen:
                        state = "car" if d == self.car[g] else "goat"
                    elif stage >= 3 and d != chosen:
                        state = "car" if d == self.car[g] else "goat"
                    elif d == chosen:
                        state = "closed_pick"
                    else:
                        state = "closed"
                    if stage >= 3 and d != chosen and d != self.host[g]:
                        state = "closed"            # the untaken door stays shut
                self._door(ctx, x, DOORS_Y, state)
            # result flag during the slow games
            if slow and stage >= 3:
                won = bool(self.stay_win[g] if ci == 0 else self.switch_win[g])
                font(ctx, "black", 34)
                s = "WIN" if won else "LOSE"
                ctx.set_source_rgba(*(rgb(CYAN) if won else rgb(MAGENTA)), 1)
                ctx.move_to(colx - ctx.text_extents(s).x_advance / 2, DOORS_Y + DOOR_H + 50)
                ctx.show_text(s)
            # percentage + bar
            if done:
                fr = (self.cum_stay if ci == 0 else self.cum_switch)[done] / done
                font(ctx, "dbold", 64)
                s = f"{100 * fr:.1f}%"
                ctx.set_source_rgb(*rgb(col))
                ctx.move_to(colx - ctx.text_extents(s).x_advance / 2, PCT_Y)
                ctx.show_text(s)
                bw = 3 * DOOR_W + 2 * DOOR_GAP
                ctx.set_source_rgba(1, 1, 1, 0.10)
                ctx.rectangle(colx - bw / 2, BAR_Y, bw, 10)
                ctx.fill()
                ctx.set_source_rgba(*rgb(col), 0.9)
                ctx.rectangle(colx - bw / 2, BAR_Y, bw * fr, 10)
                ctx.fill()
        # grid legend
        font(ctx, "bold", 20)
        ctx.set_source_rgba(*rgb(CYAN), 0.8)
        show_tracked(ctx, "■ SWITCH WON", GRID_C[0] - 210, GRID_C[1] - 228, 0.06 * 20)
        ctx.set_source_rgba(*rgb(MAGENTA), 0.8)
        s = "■ STAY WON"
        show_tracked(ctx, s, GRID_C[0] + 210 - tracked_width(ctx, s, 0.06 * 20), GRID_C[1] - 228, 0.06 * 20)

    def _big_labels(self, ctx, t):
        n_open = 0 if t < B_OPEN0 else 98 if t >= B_OPEN1 else int(98 * smooth((t - B_OPEN0) / (B_OPEN1 - B_OPEN0)))
        font(ctx, "bold", 20)
        ctx.set_source_rgba(*rgb(AMBER), 0.9)
        show_tracked(ctx, "■ YOUR PICK", GRID_C[0] - 210, GRID_C[1] - 228, 0.06 * 20)
        ctx.set_source_rgba(1, 1, 1, 0.7)
        s = f"GOATS OPENED: {n_open} / 98"
        show_tracked(ctx, s, GRID_C[0] + 210 - tracked_width(ctx, s, 0.06 * 20), GRID_C[1] - 228, 0.06 * 20)
        if t >= B_REVEAL:
            r, c = divmod(self.big_car, 10)
            x = BIG_C[0] - 210 + c * BIG_PITCH + 3 + BIG_CELL / 2
            y = BIG_C[1] - 210 + r * BIG_PITCH + 3
            font(ctx, "black", 26)
            ctx.set_source_rgba(*rgb(CYAN), 1)
            s = "99%"
            ctx.move_to(x - ctx.text_extents(s).x_advance / 2, y - 10)
            ctx.show_text(s)
            r, c = divmod(self.big_pick, 10)
            x = BIG_C[0] - 210 + c * BIG_PITCH + 3 + BIG_CELL / 2
            y = BIG_C[1] - 210 + r * BIG_PITCH + 3
            ctx.set_source_rgba(*rgb(AMBER), 1)
            s = "1%"
            ctx.move_to(x - ctx.text_extents(s).x_advance / 2, y - 10)
            ctx.show_text(s)

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
